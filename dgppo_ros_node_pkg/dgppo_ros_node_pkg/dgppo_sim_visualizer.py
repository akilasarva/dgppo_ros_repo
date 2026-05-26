#!/usr/bin/env python3
"""
DGPPO Sim I/O Visualizer

Shows the 2D sim coordinate frame ([0,1.5]²) with agent position, lidar hits,
bearing and action arrows, plus a sim-frame velocity lag plot (commanded vs
observed) to verify coordinate transforms live.

Usage:
  ros2 run dgppo_ros_node_pkg dgppo_sim_visualizer -- [plan.json] [--port 8766]
  SSH tunnel: ssh -L 8766:localhost:8766 swarm@10.29.167.251
  Then open: http://localhost:8766
"""

import sys, os, json, math, threading, argparse, time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int32, Float32MultiArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from flask import Flask, jsonify, Response

# ── Constants ──────────────────────────────────────────────────────────────────

SCALE_2D_3D  = 11.0
SIM_MAX_VEL  = 1.5 / SCALE_2D_3D   # ≈ 0.1364 sim units/s
AREA_SIZE    = 1.5
N_RAYS_PHYS  = 72
N_RAYS_TRAIN = 32
VEL_HIST_LEN = 300                  # ~10 s at 30 Hz
WEB_PORT     = 8766

TERRAIN_NAMES = {0: "Road", 1: "Grass", 2: "Sidewalk"}
CLUSTER_NAMES = {0: "open_space", 1: "approach", 2: "on_bridge", 3: "exit"}

# ── Shared state ───────────────────────────────────────────────────────────────

class DebugState:
    def __init__(self, plan_sequence, bearing_map, centroids, log_file=None):
        self._lock          = threading.Lock()
        self.state_debug    = None      # 18-float array from /dgppo_state_debug
        self.spot_yaw       = 0.0
        self.world_alpha    = 0.0
        self.processed_ranges = None    # numpy array (72,)
        self.raw_cluster    = None
        self.terrain_id     = 1
        self.plan_step      = 0
        self.plan_sequence  = plan_sequence
        self.bearing_map    = bearing_map
        self.centroids      = centroids  # {"1": [fwd_m, lat_m, ...], ...}
        self._log_file      = log_file   # open file handle for JSONL output, or None
        # velocity history for lag plot
        self._vel_t0        = None
        self.vel_times      = deque(maxlen=VEL_HIST_LEN)
        self.cmd_vx_hist    = deque(maxlen=VEL_HIST_LEN)
        self.cmd_vy_hist    = deque(maxlen=VEL_HIST_LEN)
        self.obs_vx_hist    = deque(maxlen=VEL_HIST_LEN)
        self.obs_vy_hist    = deque(maxlen=VEL_HIST_LEN)

    def set_state_debug(self, data):
        with self._lock:
            self.state_debug = list(data)
            now = time.time()
            if self._vel_t0 is None:
                self._vel_t0 = now
            self.vel_times.append(now - self._vel_t0)
            # cmd vel: action_post * SIM_MAX_VEL  (indices 15, 16)
            if len(data) >= 17:
                self.cmd_vx_hist.append(data[15] * SIM_MAX_VEL)
                self.cmd_vy_hist.append(data[16] * SIM_MAX_VEL)
            # obs vel: sim-frame from state_debug (indices 8, 9)
            if len(data) >= 10:
                self.obs_vx_hist.append(data[8])
                self.obs_vy_hist.append(data[9])
            if self._log_file is not None:
                self._log_tick(data, now)

    def _log_tick(self, sd, now):
        """Write one JSONL record. Called inside the lock from set_state_debug."""
        ps    = self.plan_step
        plan  = self.plan_sequence
        start_c = plan[ps]['start'] if ps < len(plan) else None
        next_c  = plan[ps]['next']  if ps < len(plan) else None
        if start_c is not None and next_c is not None:
            key     = f"{start_c}-{next_c}"
            bearing = self.bearing_map.get(key, 0.0) + math.pi / 2
        else:
            bearing = 0.0
        record = {
            't':              now,
            'plan_step':      ps,
            'plan_start':     start_c,
            'plan_next':      next_c,
            'bearing_rad':    bearing,
            'bearing_deg':    math.degrees(bearing),
            'raw_cluster':    self.raw_cluster,
            'terrain_id':     self.terrain_id,
            'yaw_deg':        math.degrees(self.spot_yaw),
            'world_alpha_rad': self.world_alpha,
            # full state_debug vector (18 floats)
            'state_debug':    list(sd),
            # named extracts for convenience
            'vision_pos':     [sd[0], sd[1]],
            'vision_vel_ms':  [sd[2], sd[3]],
            'sim_pos':        [sd[6], sd[7]]    if len(sd) > 7  else None,
            'sim_vel':        [sd[8], sd[9]]    if len(sd) > 9  else None,
            'cmd_body_ms':    [sd[10], sd[11]]  if len(sd) > 11 else None,
            'action_raw':     [sd[12], sd[13]]  if len(sd) > 13 else None,
            'inference_ms':   sd[14]            if len(sd) > 14 else None,
            'action_post':    [sd[15], sd[16]]  if len(sd) > 16 else None,
            'cmd_vx_sim':     sd[15] * SIM_MAX_VEL if len(sd) > 16 else None,
            'cmd_vy_sim':     sd[16] * SIM_MAX_VEL if len(sd) > 16 else None,
            # ranges summary (avoid logging 72 floats every tick unless needed)
            'ranges_min_m':   float(self.processed_ranges.min()) if self.processed_ranges is not None else None,
            'ranges_max_m':   float(self.processed_ranges.max()) if self.processed_ranges is not None else None,
            'ranges_mean_m':  float(self.processed_ranges.mean()) if self.processed_ranges is not None else None,
            'ranges_raw':     self.processed_ranges.tolist() if self.processed_ranges is not None else None,
        }
        try:
            self._log_file.write(json.dumps(record) + '\n')
            self._log_file.flush()
        except Exception:
            pass

    def set_spot_yaw(self, yaw):
        with self._lock: self.spot_yaw = yaw

    def set_world_alpha(self, alpha):
        with self._lock: self.world_alpha = alpha

    def set_ranges(self, r):
        with self._lock: self.processed_ranges = r

    def set_cluster(self, raw):
        with self._lock: self.raw_cluster = raw

    def set_terrain(self, tid):
        with self._lock: self.terrain_id = tid

    def set_plan_step(self, step):
        with self._lock: self.plan_step = step

    def snapshot(self):
        with self._lock:
            return dict(
                state_debug    = self.state_debug,
                spot_yaw       = self.spot_yaw,
                world_alpha    = self.world_alpha,
                ranges         = (self.processed_ranges.tolist()
                                  if self.processed_ranges is not None else None),
                raw_cluster    = self.raw_cluster,
                terrain_id     = self.terrain_id,
                plan_step      = self.plan_step,
                plan_sequence  = self.plan_sequence,
                bearing_map    = self.bearing_map,
                centroids      = self.centroids,
                vel_times      = list(self.vel_times),
                cmd_vx_hist    = list(self.cmd_vx_hist),
                cmd_vy_hist    = list(self.cmd_vy_hist),
                obs_vx_hist    = list(self.obs_vx_hist),
                obs_vy_hist    = list(self.obs_vy_hist),
            )


# ── ROS subscriber node ────────────────────────────────────────────────────────

class SimVisSubscriber(Node):
    def __init__(self, state: DebugState):
        super().__init__('dgppo_sim_visualizer')
        self.state = state
        sub = self.create_subscription
        sub(Float32MultiArray, '/dgppo_state_debug', self._cb_sd,      10)
        sub(Float32MultiArray, '/dgppo_spot_yaw',    self._cb_yaw,     10)
        sub(Float32MultiArray, '/processed_ranges',  self._cb_ranges,  10)
        sub(Int16,             '/predicted_cluster', self._cb_cluster, 10)
        sub(Int32,             '/current_terrain',   self._cb_terrain, 10)
        sub(Int32,             '/dgppo_plan_step',   self._cb_plan,    10)
        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        sub(Float32MultiArray, '/dgppo_world_alpha', self._cb_alpha, _latched)

    def _cb_sd(self, msg):      self.state.set_state_debug(msg.data)
    def _cb_yaw(self, msg):     self.state.set_spot_yaw(msg.data[0] if msg.data else 0.0)
    def _cb_alpha(self, msg):   self.state.set_world_alpha(msg.data[0] if msg.data else 0.0)
    def _cb_ranges(self, msg):  self.state.set_ranges(np.array(msg.data, dtype=np.float32))
    def _cb_cluster(self, msg): self.state.set_cluster(msg.data)
    def _cb_terrain(self, msg): self.state.set_terrain(msg.data)
    def _cb_plan(self, msg):    self.state.set_plan_step(msg.data)


# ── Lidar hit reconstruction ───────────────────────────────────────────────────

def compute_obs_hits(ranges_72, agent_x, agent_y, yaw, world_alpha):
    """Reconstruct 32 lidar hit positions in sim world frame.

    Replicates the transform in spot_dgppo_ros_node.py lines 721-742.
    """
    angles_phys = np.linspace(0, 2 * np.pi, N_RAYS_PHYS, endpoint=False)
    angles_beam = np.linspace(-np.pi, np.pi - 2 * np.pi / N_RAYS_TRAIN, N_RAYS_TRAIN)
    ranges_sim  = np.asarray(ranges_72, dtype=np.float64) / SCALE_2D_3D
    lookup      = np.mod(angles_beam - np.pi / 2 - world_alpha - yaw, 2 * np.pi)
    ranges_res  = np.interp(lookup, angles_phys, ranges_sim)
    hits = np.stack([
        agent_x + ranges_res * np.cos(angles_beam),
        agent_y + ranges_res * np.sin(angles_beam),
    ], axis=1)
    top_k_idx = np.argsort(ranges_res)[:8].tolist()
    return hits.tolist(), top_k_idx


def _map_cluster(raw):
    m = {
        **{k: 0 for k in [0, 1]},
        **{k: 1 for k in [2, 3, 10, 11]},
        **{k: 2 for k in [5, 6, 7, 8, 9, 12]},
        **{k: 3 for k in [-1, 4]},
    }
    return m.get(raw, raw)


# ── Flask app ──────────────────────────────────────────────────────────────────

app    = Flask(__name__)
_state: DebugState = None   # injected in main()


@app.route('/api/state')
def api_state():
    s  = _state.snapshot()
    sd = s['state_debug']

    result = dict(
        area_size    = AREA_SIZE,
        sim_max_vel  = SIM_MAX_VEL,
        terrain      = s['terrain_id'],
        terrain_name = TERRAIN_NAMES.get(s['terrain_id'], '?'),
        plan_step    = s['plan_step'],
        vel_times    = s['vel_times'],
        cmd_vx_hist  = s['cmd_vx_hist'],
        cmd_vy_hist  = s['cmd_vy_hist'],
        obs_vx_hist  = s['obs_vx_hist'],
        obs_vy_hist  = s['obs_vy_hist'],
        obs_hits     = [],
        top_k_idx    = [],
        cluster_centroids = {},
        ready        = False,
    )

    # Cluster / plan step
    raw_c   = s['raw_cluster']
    mapped_c = _map_cluster(raw_c) if raw_c is not None else None
    ps      = s['plan_step']
    plan    = s['plan_sequence']
    start_c = plan[ps]['start'] if ps < len(plan) else None
    next_c  = plan[ps]['next']  if ps < len(plan) else None
    result['cluster'] = dict(
        current      = mapped_c,
        start        = start_c,
        next         = next_c,
        current_name = CLUSTER_NAMES.get(mapped_c, '?') if mapped_c is not None else '?',
        start_name   = CLUSTER_NAMES.get(start_c,  '?') if start_c  is not None else '?',
        next_name    = CLUSTER_NAMES.get(next_c,   '?') if next_c   is not None else '?',
    )

    # Bearing
    if start_c is not None and next_c is not None:
        key     = f"{start_c}-{next_c}"
        bearing = s['bearing_map'].get(key, 0.0) + math.pi / 2
    else:
        bearing = 0.0
    result['bearing']     = bearing
    result['bearing_deg'] = math.degrees(bearing)

    # Cluster centroids → sim frame
    # centroid[0] = forward_m → sim Y;  centroid[1] = lateral_m → sim X
    for cid, centroid in s['centroids'].items():
        if len(centroid) >= 2:
            result['cluster_centroids'][cid] = [
                float(centroid[1]) / SCALE_2D_3D,   # sim X
                float(centroid[0]) / SCALE_2D_3D,   # sim Y
            ]

    if sd is None:
        return jsonify(result)

    result['ready']        = True
    result['vision_pos']   = [sd[0], sd[1]]
    result['vision_vel']   = [sd[2], sd[3]]
    result['sim_pos']      = [sd[6], sd[7]]
    result['sim_vel']      = [sd[8], sd[9]]
    result['cmd_body']     = [sd[10], sd[11]]
    result['action_raw']   = [sd[12], sd[13]]
    result['inference_ms'] = sd[14] if len(sd) > 14 else 0.0
    result['action_post']  = [sd[15], sd[16]] if len(sd) > 16 else [0.0, 0.0]
    result['yaw_deg']      = math.degrees(s['spot_yaw'])

    # Lidar hits in sim frame
    if s['ranges'] is not None and len(s['ranges']) == N_RAYS_PHYS:
        hits, top_k = compute_obs_hits(
            s['ranges'], sd[6], sd[7], s['spot_yaw'], s['world_alpha']
        )
        result['obs_hits']  = hits
        result['top_k_idx'] = top_k

    return jsonify(result)


@app.route('/')
def index():
    return Response(HTML_PAGE, mimetype='text/html')


# ── HTML page ──────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DGPPO Sim Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: 'Courier New', monospace;
         font-size: 12px; padding: 12px; }
  h1 { color: #58a6ff; font-size: 14px; margin-bottom: 10px; letter-spacing: 0.5px; }
  h2 { color: #58a6ff; font-size: 11px; font-weight: bold; letter-spacing: 0.8px;
       margin-bottom: 8px; text-transform: uppercase; }
  .main-row { display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; }
  canvas { display: block; }

  /* I/O table */
  .io-panel { width: 270px; flex-shrink: 0; }
  .section { color: #58a6ff; font-size: 10px; font-weight: bold; letter-spacing: 1.2px;
             margin: 10px 0 5px 0; padding-bottom: 3px; border-bottom: 1px solid #30363d; }
  .section:first-child { margin-top: 0; }
  .row { display: flex; justify-content: space-between; align-items: center;
         margin: 2px 0; min-height: 17px; }
  .lbl { color: #8b949e; }
  .val { color: #e6edf3; font-weight: bold; text-align: right; max-width: 160px; word-break: break-all; }
  .val.good { color: #3fb950; }
  .val.warn { color: #d29922; }
  .badge { display: inline-block; padding: 1px 5px; border-radius: 3px;
           font-size: 10px; font-weight: bold; }
  .b0 { background:#1c2d3d; color:#58a6ff; }
  .b1 { background:#1a2e1a; color:#3fb950; }
  .b2 { background:#2d210a; color:#d29922; }
  .b3 { background:#2d0f1a; color:#ff6eb4; }

  /* Legend */
  .legend { display: flex; gap: 12px; margin-top: 6px; flex-wrap: wrap; }
  .li { display: flex; align-items: center; gap: 4px; font-size: 10px; color: #8b949e; }
  .ls  { display: inline-block; width: 18px; height: 2px; }
  .lsd { display: inline-block; width: 18px; height: 0;
         border-top: 2px dashed currentColor; }

  /* Waiting */
  .waiting { color: #8b949e; font-style: italic; padding: 8px 0; }
</style>
</head>
<body>
<h1>&#11044; DGPPO Sim I/O Visualizer</h1>

<div class="main-row">
  <!-- 2D sim canvas -->
  <div class="panel">
    <h2>2D Sim World Frame &nbsp;[0, 1.5]&sup2;</h2>
    <canvas id="simCanvas" width="460" height="460"></canvas>
    <div class="legend" style="margin-top:6px;">
      <div class="li"><span class="ls" style="background:#00cc44"></span>hits (32)</div>
      <div class="li"><span class="ls" style="background:#ff6600"></span>top-8</div>
      <div class="li"><span class="ls" style="background:#ffd700"></span>bearing</div>
      <div class="li"><span class="ls" style="background:#00cfff"></span>action</div>
      <div class="li"><span class="ls" style="background:#cc44ff"></span>obs vel</div>
      <div class="li"><span class="ls" style="background:#58a6ff"></span>centroids</div>
    </div>
  </div>

  <!-- I/O table -->
  <div class="panel io-panel">
    <h2>Policy I/O</h2>
    <div id="io-content"><div class="waiting">waiting for /dgppo_state_debug&hellip;</div></div>
  </div>
</div>

<!-- Lag plot -->
<div class="panel">
  <h2>Sim-Frame Velocity &mdash; Commanded vs Observed &nbsp;(rolling 10 s)</h2>
  <canvas id="lagCanvas" width="940" height="190"></canvas>
  <div class="legend" style="margin-top:6px;">
    <div class="li"><span class="ls" style="background:#00cfff"></span>cmd vx</div>
    <div class="li"><span class="lsd" style="color:#00cfff"></span>obs vx</div>
    <div class="li"><span class="ls" style="background:#ffd700"></span>cmd vy</div>
    <div class="li"><span class="lsd" style="color:#ffd700"></span>obs vy</div>
    <div class="li" style="margin-left:14px; color:#555;">(gap between solid/dashed = lag or scale error)</div>
  </div>
</div>

<script>
// ── Constants (match Python) ───────────────────────────────────────────────────
const AREA        = 1.5;
const SIM_MAX_VEL = 1.5 / 11.0;
const PAD         = 22;          // canvas padding (pixels)
const SZ          = 460;         // simCanvas size
const INNER       = SZ - 2*PAD;

const CLUSTER_NAMES = {0:'open_space', 1:'approach', 2:'on_bridge', 3:'exit'};
const TERRAIN_NAMES = {0:'Road', 1:'Grass', 2:'Sidewalk'};
const BADGE_CLS     = {0:'b0', 1:'b1', 2:'b2', 3:'b3'};

// ── Coordinate helpers ─────────────────────────────────────────────────────────
// Sim (sx,sy) → canvas (cx,cy).  Y is flipped: sim +Y up, canvas +Y down.
function s2c(sx, sy) {
  return {
    x: PAD + (sx / AREA) * INNER,
    y: PAD + (1 - sy / AREA) * INNER
  };
}

function drawArrow(ctx, x1, y1, x2, y2, headLen=9) {
  const dx = x2-x1, dy = y2-y1, len = Math.hypot(dx,dy);
  if (len < 2) return;
  const a = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
  ctx.lineTo(x2 - headLen*Math.cos(a-Math.PI/6), y2 - headLen*Math.sin(a-Math.PI/6));
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen*Math.cos(a+Math.PI/6), y2 - headLen*Math.sin(a+Math.PI/6));
  ctx.stroke();
}

// ── Sim canvas ─────────────────────────────────────────────────────────────────
function drawSim(d) {
  const cv  = document.getElementById('simCanvas');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, SZ, SZ);

  // Background
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, SZ, SZ);

  // Grid lines at 0.5 intervals
  ctx.strokeStyle = '#1a1f27';
  ctx.lineWidth = 1;
  for (let v = 0; v <= AREA + 0.01; v += 0.5) {
    const p0 = s2c(v, 0), p1 = s2c(0, v);
    ctx.beginPath(); ctx.moveTo(p0.x, PAD); ctx.lineTo(p0.x, SZ-PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, p1.y); ctx.lineTo(SZ-PAD, p1.y); ctx.stroke();
  }

  // Grid labels
  ctx.fillStyle = '#3a4050';
  ctx.font = '9px monospace';
  for (let v = 0; v <= AREA + 0.01; v += 0.5) {
    const px = s2c(v, 0).x;
    const py = s2c(0, v).y;
    ctx.fillText(v.toFixed(1), px-8, SZ-5);
    ctx.fillText(v.toFixed(1), 2, py+3);
  }

  // Domain border
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 2;
  ctx.strokeRect(PAD, PAD, INNER, INNER);

  // Axis labels
  ctx.fillStyle = '#30363d';
  ctx.font = '10px monospace';
  ctx.fillText('+X →', SZ - PAD - 30, SZ - 5);
  ctx.save(); ctx.translate(8, PAD+35); ctx.rotate(-Math.PI/2);
  ctx.fillText('+Y', 0, 0); ctx.restore();

  if (!d || !d.ready) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '13px monospace';
    ctx.fillText('waiting for data...', PAD+10, PAD+30);
    return;
  }

  // Cluster centroids
  for (const [cid, pos] of Object.entries(d.cluster_centroids || {})) {
    const p = s2c(pos[0], pos[1]);
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(p.x-7,p.y); ctx.lineTo(p.x+7,p.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.x,p.y-7); ctx.lineTo(p.x,p.y+7); ctx.stroke();
    ctx.fillStyle = '#58a6ff';
    ctx.font = '9px monospace';
    ctx.fillText(CLUSTER_NAMES[parseInt(cid)] || cid, p.x+8, p.y+4);
  }

  // Lidar hits
  const topk = new Set(d.top_k_idx || []);
  for (let i = 0; i < (d.obs_hits || []).length; i++) {
    const [hx, hy] = d.obs_hits[i];
    const p = s2c(hx, hy);
    const isTopK = topk.has(i);
    ctx.fillStyle = isTopK ? '#ff6600' : '#00cc44';
    ctx.beginPath();
    ctx.arc(p.x, p.y, isTopK ? 4 : 2.5, 0, 2*Math.PI);
    ctx.fill();
  }

  const ap = s2c(d.sim_pos[0], d.sim_pos[1]);

  // Observed velocity arrow (purple dashed) — scale so SIM_MAX_VEL = 80px
  const velPx = (INNER / AREA) * 0.6;
  const vxC =  d.sim_vel[0] * velPx;
  const vyC = -d.sim_vel[1] * velPx;   // Y-flip
  if (Math.hypot(vxC, vyC) > 2) {
    ctx.strokeStyle = '#cc44ff';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    drawArrow(ctx, ap.x, ap.y, ap.x+vxC, ap.y+vyC, 7);
    ctx.setLineDash([]);
  }

  // Bearing arrow (gold) — fixed 80 px
  const BL = 80;
  const bxC =  Math.cos(d.bearing) * BL;
  const byC = -Math.sin(d.bearing) * BL;  // Y-flip
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth = 2;
  drawArrow(ctx, ap.x, ap.y, ap.x+bxC, ap.y+byC, 10);

  // Action arrow (cyan) — magnitude-proportional, max 70 px
  const AL = 70;
  const [ax, ay] = d.action_post;
  const aMag = Math.hypot(ax, ay);
  if (aMag > 0.04) {
    const axC =  (ax / aMag) * AL * Math.min(aMag, 1.0);
    const ayC = -(ay / aMag) * AL * Math.min(aMag, 1.0);  // Y-flip
    ctx.strokeStyle = '#00cfff';
    ctx.lineWidth = 2;
    drawArrow(ctx, ap.x, ap.y, ap.x+axC, ap.y+ayC, 10);
  }

  // Agent circle
  ctx.fillStyle = '#ffffff';
  ctx.beginPath(); ctx.arc(ap.x, ap.y, 7, 0, 2*Math.PI); ctx.fill();
  ctx.strokeStyle = '#0a0e14';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Position label
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px monospace';
  ctx.fillText(`(${d.sim_pos[0].toFixed(3)}, ${d.sim_pos[1].toFixed(3)})`, ap.x+10, ap.y-10);
}

// ── Lag plot ───────────────────────────────────────────────────────────────────
function drawLag(d) {
  const cv  = document.getElementById('lagCanvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const PL=54, PR=16, PT=12, PB=28;
  const PW = W-PL-PR, PH = H-PT-PB;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, W, H);

  // Y-axis grid & labels
  const yMax = SIM_MAX_VEL * 1.15;
  const yTicks = [-SIM_MAX_VEL, -SIM_MAX_VEL/2, 0, SIM_MAX_VEL/2, SIM_MAX_VEL];
  function ty(v) { return PT + PH/2 - (v / yMax) * (PH/2); }
  for (const yt of yTicks) {
    const py = ty(yt);
    ctx.strokeStyle = yt === 0 ? '#30363d' : '#1a1f27';
    ctx.lineWidth = yt === 0 ? 1.5 : 1;
    ctx.beginPath(); ctx.moveTo(PL, py); ctx.lineTo(PL+PW, py); ctx.stroke();
    ctx.fillStyle = '#3a4050';
    ctx.font = '9px monospace';
    ctx.fillText(yt.toFixed(3), 2, py+3);
  }

  // Border
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.strokeRect(PL, PT, PW, PH);

  // SIM_MAX_VEL label
  ctx.fillStyle = '#3a4050';
  ctx.font = '9px monospace';
  ctx.fillText('sim/s', 2, PT+10);

  if (!d || !d.vel_times || d.vel_times.length < 2) {
    ctx.fillStyle = '#8b949e';
    ctx.font = '12px monospace';
    ctx.fillText('waiting for velocity data...', PL+10, PT+PH/2);
    return;
  }

  const times = d.vel_times;
  const tMax  = times[times.length-1];
  const tMin  = Math.max(0, tMax - 10.0);

  function tx(t) { return PL + ((t - tMin) / Math.max(tMax - tMin, 0.001)) * PW; }

  function drawLine(arr, color, dashed) {
    if (!arr || arr.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < arr.length; i++) {
      if (times[i] < tMin) continue;
      const px = tx(times[i]);
      const py = ty(arr[i]);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else { ctx.lineTo(px, py); }
    }
    if (started) ctx.stroke();
    ctx.setLineDash([]);
  }

  drawLine(d.cmd_vx_hist, '#00cfff', false);
  drawLine(d.obs_vx_hist, '#00cfff', true);
  drawLine(d.cmd_vy_hist, '#ffd700', false);
  drawLine(d.obs_vy_hist, '#ffd700', true);

  // X-axis labels
  ctx.fillStyle = '#3a4050';
  ctx.font = '9px monospace';
  ctx.fillText(`${tMin.toFixed(0)}s`, PL, H-6);
  ctx.fillText(`${((tMin+tMax)/2).toFixed(1)}s`, PL+PW/2-12, H-6);
  ctx.fillText(`${tMax.toFixed(1)}s`, PL+PW-22, H-6);
}

// ── I/O table ──────────────────────────────────────────────────────────────────
function fv(v, dec) {
  return (typeof v === 'number') ? v.toFixed(dec ?? 4) : '—';
}
function fv2(arr, dec) {
  return arr ? `[${fv(arr[0],dec??4)},&nbsp;${fv(arr[1],dec??4)}]` : '—';
}
function badge(id) {
  if (id === null || id === undefined) return '<span class="badge b0">?</span>';
  const cls  = `badge ${BADGE_CLS[id] || 'b0'}`;
  const name = CLUSTER_NAMES[id] || id;
  return `<span class="${cls}">${name}</span>`;
}

function updateIO(d) {
  const el = document.getElementById('io-content');
  if (!d || !d.ready) {
    el.innerHTML = '<div class="waiting">waiting for /dgppo_state_debug&hellip;</div>';
    return;
  }
  const c   = d.cluster || {};
  const inf = d.inference_ms;
  const infClass = inf < 50 ? 'good' : 'warn';

  el.innerHTML = `
<div class="section">INPUTS</div>
<div class="row"><span class="lbl">pos sim</span><span class="val">${fv2(d.sim_pos)}</span></div>
<div class="row"><span class="lbl">vel sim</span><span class="val">${fv2(d.sim_vel)}</span></div>
<div class="row"><span class="lbl">bearing</span><span class="val">${fv(d.bearing)} rad</span></div>
<div class="row"><span class="lbl"></span><span class="val">${fv(d.bearing_deg,1)}&deg;</span></div>
<div class="row"><span class="lbl">cluster cur</span><span class="val">${badge(c.current)}</span></div>
<div class="row"><span class="lbl">cluster start</span><span class="val">${badge(c.start)}</span></div>
<div class="row"><span class="lbl">cluster next</span><span class="val">${badge(c.next)}</span></div>
<div class="row"><span class="lbl">terrain</span><span class="val">${TERRAIN_NAMES[d.terrain] ?? d.terrain}</span></div>

<div class="section">OUTPUTS</div>
<div class="row"><span class="lbl">action raw</span><span class="val">${fv2(d.action_raw)}</span></div>
<div class="row"><span class="lbl">action post</span><span class="val">${fv2(d.action_post)}</span></div>
<div class="row"><span class="lbl">cmd body m/s</span><span class="val">${fv2(d.cmd_body,3)}</span></div>
<div class="row"><span class="lbl">inference</span><span class="val ${infClass}">${fv(inf,1)} ms</span></div>

<div class="section">CONTEXT</div>
<div class="row"><span class="lbl">vision pos</span><span class="val">${fv2(d.vision_pos,3)} m</span></div>
<div class="row"><span class="lbl">vision vel</span><span class="val">${fv2(d.vision_vel,3)} m/s</span></div>
<div class="row"><span class="lbl">yaw</span><span class="val">${fv(d.yaw_deg,1)}&deg;</span></div>
<div class="row"><span class="lbl">plan step</span><span class="val">${d.plan_step}</span></div>
<div class="row"><span class="lbl">SIM_MAX_VEL</span><span class="val">${SIM_MAX_VEL.toFixed(4)} sim/s</span></div>
<div class="row"><span class="lbl">scale</span><span class="val">11 m/sim</span></div>`;
}

// ── Poll loop ──────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    drawSim(d);
    drawLag(d);
    updateIO(d);
  } catch(e) {
    console.error('[sim viz] poll error:', e);
  }
}

// Initial draw (shows "waiting..." on canvases)
drawSim(null);
drawLag(null);

setInterval(poll, 100);   // 10 Hz
poll();
</script>
</body>
</html>"""


# ── Plan loader ────────────────────────────────────────────────────────────────

def _load_plan(path):
    try:
        with open(path) as f:
            data = json.load(f)
        seq     = data.get('plan_sequence', [])
        bmap    = data.get('bearing_map', {})
        cents   = data.get('centroids', {})
        print(f"[SIM VIZ] Loaded plan: {len(seq)} steps, {len(bmap)} bearings, {len(cents)} centroids")
        return seq, bmap, cents
    except Exception as e:
        print(f"[SIM VIZ] Warning: could not load plan from {path}: {e}")
        return [], {}, {}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DGPPO Sim I/O Visualizer')
    parser.add_argument('plan_json', nargs='?', default='plans/bridge.json',
                        help='Path to plan JSON file')
    parser.add_argument('--port', type=int, default=WEB_PORT,
                        help=f'Flask port (default {WEB_PORT})')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable JSONL logging to disk')
    args, _ = parser.parse_known_args()

    seq, bmap, cents = _load_plan(args.plan_json)

    # Open log file
    log_file = None
    if not args.no_log:
        import datetime
        log_dir = os.path.join(os.path.dirname(__file__), 'debug_logs')
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'simviz_{ts}.jsonl')
        log_file = open(log_path, 'w')
        print(f"[SIM VIZ] Logging to {log_path}")

    global _state
    _state = DebugState(seq, bmap, cents, log_file=log_file)

    rclpy.init()
    node = SimVisSubscriber(_state)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    print(f"[SIM VIZ] Listening on http://0.0.0.0:{args.port}")
    print(f"[SIM VIZ] SSH tunnel: ssh -L {args.port}:localhost:{args.port} swarm@<robot-ip>")
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == '__main__':
    main()
