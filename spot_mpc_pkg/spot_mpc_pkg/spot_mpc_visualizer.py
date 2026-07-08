#!/usr/bin/env python3
"""
Spot MPC Debug Visualizer

Web dashboard for spot_mpc_node: live LiDAR/point-cloud view with the MPC's
chosen candidate rollout overlaid, plus guidance/score telemetry and camera
streams.

Usage:
  ros2 run spot_mpc_pkg spot_mpc_visualizer
  ros2 run spot_mpc_pkg spot_mpc_visualizer -- --port 8080

Coordinate conventions:
  Display: TOP = robot FORWARD, RIGHT = robot RIGHT, LEFT = robot LEFT
  - Raw cloud: driver already in Spot body frame (+x fwd, +y left); 90° CW to display → (−y, x)
  - Processed-range beams / MPC rollout: same body frame (forward, left) → (−r·sin θ, r·cos θ)

Z-height mode  → yellow slice; z-band sent to clustering node → affects actual clustering
Intensity mode → magenta slice; z-band AND intensity band sent to clustering node → affects actual clustering
"""

import os, json, threading, argparse, time
from collections import deque

import numpy as np

try:
    import cv2
    from cv_bridge import CvBridge as _CvBridge
    _bridge = _CvBridge()
    _HAS_CV = True
except ImportError:
    _HAS_CV = False

_CAM_W = 480   # max JPEG width for streaming
_CAM_Q = 55    # JPEG quality

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int32, Float32MultiArray
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py.point_cloud2 import read_points
from rclpy.qos import qos_profile_sensor_data

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K       = 8
WEB_PORT    = 8765

TERRAIN_NAMES = {0: "Road", 1: "Grass", 2: "Sidewalk"}
CLUSTER_NAMES = {0: "open_space", 1: "approach", 2: "on_corridor", 3: "exit"}
RAW_TO_MAPPED = {
    **{k: 0 for k in [0, 1]},
    **{k: 1 for k in [2, 3, 10, 11]},
    **{k: 2 for k in [5, 6, 7, 8, 9, 12]},
    **{k: 3 for k in [-1, 4]},
}


# ── Filter config ─────────────────────────────────────────────────────────────

_CFG_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'filter_config.json')

class FilterConfig:
    def __init__(self):
        self._lock     = threading.Lock()
        self.z_upper   = 0.15
        self.z_lower   = -0.5
        self.z2_upper  = 0.0
        self.z2_lower  = 0.0
        self.max_range = 8.0
        self.min_range = 0.5
        self.use_intensity  = False
        self.int_lower      = 0.0
        self.int_upper      = 255.0
        self.density_radius = 0.30
        self.min_neighbors  = 4
        self._load()

    def _load(self):
        try:
            with open(_CFG_SAVE_PATH) as f:
                d = json.load(f)
            for k, v in d.items():
                if hasattr(self, k) and not k.startswith('_'):
                    setattr(self, k, v)
            print(f"[CFG] Loaded filter config from {_CFG_SAVE_PATH}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[CFG] Could not load filter config: {e}")

    def _save(self):
        try:
            with open(_CFG_SAVE_PATH, 'w') as f:
                json.dump(self.get(), f, indent=2)
        except Exception as e:
            print(f"[CFG] Could not save filter config: {e}")

    def get(self):
        with self._lock:
            return dict(
                z_upper=self.z_upper, z_lower=self.z_lower,
                z2_upper=self.z2_upper, z2_lower=self.z2_lower,
                max_range=self.max_range, min_range=self.min_range,
                use_intensity=self.use_intensity,
                int_lower=self.int_lower, int_upper=self.int_upper,
                density_radius=self.density_radius, min_neighbors=self.min_neighbors,
            )

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        self._save()


# ── Shared state ──────────────────────────────────────────────────────────────

class DebugState:
    def __init__(self, filter_cfg: FilterConfig):
        self._lock            = threading.Lock()
        self.raw_cluster      = None
        self.terrain_id       = 1
        self.plan_step        = 0
        self.spot_yaw         = None
        self.processed_ranges = None
        self.raw_cloud        = None   # (N,4): x,y,z,intensity from lidar
        self.filter_cfg       = filter_cfg
        self.raw_frame_jpg    = None   # bytes: JPEG of raw ZED image
        self.hsv_frame_jpg    = None   # bytes: JPEG of HSV-segmented image
        self.mpc_state        = None   # [pos_x,pos_y,vel_x,vel_y,v_cmd,om_cmd,yaw,start_cluster,target_cluster]
        self.best_rollout      = None   # (N,3) np.ndarray, body frame (fwd, left, theta)
        self.guidance_debug    = None   # [guidance_ratio, cluster_score, bearing_score, edt_blocked_frac]
        self.mpc_scores        = None   # (<=10,) top rollout scores, descending

    def set_cluster(self, raw):
        with self._lock: self.raw_cluster = raw

    def set_terrain(self, tid):
        with self._lock: self.terrain_id = tid

    def set_plan_step(self, step):
        with self._lock: self.plan_step = step

    def set_processed_ranges(self, r):
        with self._lock: self.processed_ranges = r

    def set_raw_cloud(self, pts):
        with self._lock: self.raw_cloud = pts

    def set_raw_frame(self, jpg):
        with self._lock: self.raw_frame_jpg = jpg

    def set_hsv_frame(self, jpg):
        with self._lock: self.hsv_frame_jpg = jpg

    def set_mpc_state(self, data):
        with self._lock:
            self.mpc_state = data
            if len(data) >= 7:
                self.spot_yaw = data[6]

    def set_best_rollout(self, arr):
        with self._lock: self.best_rollout = arr

    def set_guidance_debug(self, arr):
        with self._lock: self.guidance_debug = arr

    def set_mpc_scores(self, arr):
        with self._lock: self.mpc_scores = arr

    def get_raw_frame(self):
        with self._lock: return self.raw_frame_jpg

    def get_hsv_frame(self):
        with self._lock: return self.hsv_frame_jpg

    def snapshot(self):
        with self._lock:
            return dict(
                raw_cluster      = self.raw_cluster,
                terrain_id       = self.terrain_id,
                plan_step        = self.plan_step,
                spot_yaw         = self.spot_yaw,
                processed_ranges = self.processed_ranges.copy()
                                   if self.processed_ranges is not None else None,
                raw_cloud        = self.raw_cloud.copy()
                                   if self.raw_cloud is not None else None,
                filter_cfg       = self.filter_cfg.get(),
                mpc_state        = self.mpc_state,
                best_rollout     = self.best_rollout.copy()
                                   if self.best_rollout is not None else None,
                guidance_debug   = self.guidance_debug,
                mpc_scores       = self.mpc_scores,
            )


# ── ROS node ──────────────────────────────────────────────────────────────────

class DebugSubscriber(Node):
    def __init__(self, state: DebugState):
        super().__init__('spot_mpc_visualizer')
        self.state = state
        sub = self.create_subscription
        sub(Int16,             '/predicted_cluster', self._cb_cluster,  10)
        sub(Int16,             '/current_terrain',   self._cb_terrain,  10)
        sub(Int32,             '/mpc_plan_step',     self._cb_planstep, 10)
        sub(Float32MultiArray, '/processed_ranges',       self._cb_ranges,        10)
        sub(Float32MultiArray, '/mpc_state',              self._cb_mpc_state,     10)
        sub(Float32MultiArray, '/mpc_best_rollout',       self._cb_best_rollout,  10)
        sub(Float32MultiArray, '/mpc_guidance_debug',     self._cb_guidance,      10)
        sub(Float32MultiArray, '/mpc_scores',             self._cb_mpc_scores,    10)
        sub(PointCloud2,       '/livox/lidar',       self._cb_cloud,    qos_profile_sensor_data)
        sub(Image, '/hamilton_zed2i/zed_node/rgb/image_rect_color', self._cb_raw_img, qos_profile_sensor_data)
        sub(Image, '/segmentor_image',                   self._cb_hsv_img, 10)
        self._cfg_pub = self.create_publisher(Float32MultiArray, '/lidar_filter_config', 10)
        self.create_timer(0.2, self._pub_cfg)
        self._reset_origin_client = self.create_client(Trigger, '/mpc_reset_origin')

    def call_reset_origin(self, timeout_sec=2.0):
        """Blocking call to /mpc_reset_origin. Safe to call from a Flask
        request thread — the response is received by this node's own spin
        thread (see _ros_thread) while this call waits on it."""
        if not self._reset_origin_client.wait_for_service(timeout_sec=timeout_sec):
            return False, '/mpc_reset_origin service not available — is spot_mpc_node running?'
        try:
            result = self._reset_origin_client.call(Trigger.Request())
            return result.success, result.message
        except Exception as e:
            return False, str(e)

    def _cb_cluster(self, msg):  self.state.set_cluster(msg.data)
    def _cb_terrain(self, msg):  self.state.set_terrain(msg.data)
    def _cb_planstep(self, msg): self.state.set_plan_step(msg.data)

    def _cb_ranges(self, msg):
        if msg.data:
            self.state.set_processed_ranges(np.array(msg.data, dtype=np.float32))

    def _cb_mpc_state(self, msg):
        if len(msg.data) >= 9:
            self.state.set_mpc_state(list(msg.data))

    def _cb_best_rollout(self, msg):
        if msg.data:
            arr = np.array(msg.data, dtype=np.float32)
            self.state.set_best_rollout(arr.reshape(-1, 3))

    def _cb_guidance(self, msg):
        if msg.data:
            self.state.set_guidance_debug(list(msg.data))

    def _cb_mpc_scores(self, msg):
        self.state.set_mpc_scores(list(msg.data))

    def _cb_cloud(self, msg: PointCloud2):
        try:
            fields = {f.name for f in msg.fields}
            has_i  = 'intensity' in fields
            want   = ['x', 'y', 'z'] + (['intensity'] if has_i else [])
            raw    = read_points(msg, field_names=want, skip_nans=True)
            if raw is None or len(raw) == 0:
                return
            x = raw['x'].astype(np.float32)
            y = raw['y'].astype(np.float32)
            z = raw['z'].astype(np.float32)
            i = raw['intensity'].astype(np.float32) if has_i else np.zeros(len(x), np.float32)
            pts = np.column_stack([x, y, z, i])
            stride = max(1, len(pts) // 5000)
            self.state.set_raw_cloud(pts[::stride])
        except Exception as e:
            self.get_logger().warn(f'cloud cb error: {e}', throttle_duration_sec=5.0)

    def _cb_raw_img(self, msg):
        _encode_frame(msg, self.state.set_raw_frame)

    def _cb_hsv_img(self, msg):
        _encode_frame(msg, self.state.set_hsv_frame)

    def _pub_cfg(self):
        cfg = self.state.filter_cfg.get()
        m = Float32MultiArray()
        # Layout: [z_upper, z_lower, z2_upper, z2_lower, max_range, min_range,
        #          use_intensity, int_lower, int_upper, density_radius, min_neighbors]
        m.data = [
            cfg['z_upper'], cfg['z_lower'],
            cfg['z2_upper'], cfg['z2_lower'],
            cfg['max_range'], cfg['min_range'],
            float(cfg['use_intensity']),
            cfg['int_lower'], cfg['int_upper'],
            cfg['density_radius'], float(cfg['min_neighbors']),
        ]
        self._cfg_pub.publish(m)


def _encode_frame(msg: Image, setter):
    if not _HAS_CV:
        return
    try:
        img = _bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = img.shape[:2]
        if w > _CAM_W:
            img = cv2.resize(img, (_CAM_W, int(h * _CAM_W / w)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, _CAM_Q])
        if ok:
            setter(buf.tobytes())
    except Exception:
        pass


def _ros_thread(node: 'DebugSubscriber'):
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_density_filter(xy, neighbor_radius=0.30, min_neighbors=4):
    """Return boolean mask (len N) — True where point has >= min_neighbors within radius.

    Isolated returns (noise/rain/multipath) have 0 neighbours; solid surfaces cluster densely.
    Requires scipy; if unavailable all points are accepted (no filtering).
    """
    if len(xy) <= min_neighbors:
        return np.zeros(len(xy), dtype=bool)
    try:
        from scipy.spatial import cKDTree
        counts = cKDTree(xy).query_ball_point(xy, r=neighbor_radius, return_length=True)
        return (counts - 1) >= min_neighbors  # -1 excludes self
    except ImportError:
        return np.ones(len(xy), dtype=bool)


def _apply_slice_filter(raw_cloud, cfg):
    """Split raw_cloud (N,4) into (all_xy, slice_xy) in display frame (x=right, y=fwd).

    Z mode  → signed z band filter; same parameters sent to clustering node.
    Int mode → band filter on intensity; clustering node uses intensity too.
    Driver outputs Spot body frame (+x fwd, +y left); display maps to (−body_y, body_x).
    """
    if raw_cloud is None or len(raw_cloud) == 0:
        return None, None

    x, y, z, intensity = (raw_cloud[:, i] for i in range(4))
    dist       = np.hypot(x, y)
    range_mask = (dist >= cfg['min_range']) & (dist <= cfg['max_range'])

    band1    = (z >= cfg['z_lower'])  & (z <= cfg['z_upper'])
    band2_on = cfg['z2_upper'] > cfg['z2_lower']
    band2    = ((z >= cfg['z2_lower']) & (z <= cfg['z2_upper'])
                if band2_on else np.zeros(len(z), dtype=bool))
    band_mask = band1 | band2
    if cfg['use_intensity']:
        band_mask &= (intensity >= cfg['int_lower']) & (intensity <= cfg['int_upper'])

    # Driver in Spot body frame (+x fwd, +y left); display: x=right=−body_y, y=fwd=body_x
    xy_flipped = np.column_stack([-y, x])
    return xy_flipped, xy_flipped[band_mask & range_mask]


# ── Web server ────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spot MPC Debugger</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--grid:#30363d;--txt:#fff;--dim:#8b949e;
      --lidar:#00cc44;--topk:#ff6600;--rollout:#00cfff;--cmdvel:#ff33aa;
      --warn:#ff4444;--circ:#58a6ff;
      --sliceZ:#ffcc00;--sliceI:#ff44cc;}
body.light{--bg:#f0f4f8;--panel:#e8ecf2;--grid:#b0bac6;--txt:#0d1117;--dim:#4a5568;
           --lidar:#006e1f;--topk:#cc3d00;--rollout:#0050bb;--cmdvel:#aa0066;
           --warn:#bb0000;--circ:#1a4db5;
           --sliceZ:#996600;--sliceI:#990066;}
body.light #badge{background:#dde3ec;color:#4a5568}
body.light #badge.live{background:#c6efce;color:#006e1f}
body.light .mbtn.on{background:#fff0e0;border-color:var(--topk);color:var(--topk)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:monospace;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{padding:6px 14px;background:var(--panel);border-bottom:1px solid var(--grid);
       font-size:13px;font-weight:bold;display:flex;align-items:center;gap:12px}
#badge{font-size:10px;padding:2px 8px;border-radius:10px;background:#222;color:var(--dim)}
#badge.live{background:#0a2a0a;color:var(--lidar)}
main{display:flex;flex:1;min-height:0}
#cw{flex:1;display:flex;align-items:center;justify-content:center;padding:6px}
canvas{background:var(--panel);border:1px solid var(--grid)}
#info{width:260px;background:var(--panel);border-left:1px solid var(--grid);
      overflow-y:auto;padding:10px;font-size:12px}
.row{display:flex;justify-content:space-between;margin:2px 0}
.row .k{color:var(--dim)}.row .v{font-weight:bold}
hr{border:none;border-top:1px solid var(--grid);margin:5px 0}
#sliders{background:var(--panel);border-top:1px solid var(--grid);padding:8px 14px}
#sliders h4{font-size:10px;color:var(--dim);margin-bottom:5px;letter-spacing:.05em}
.sr{display:flex;align-items:center;gap:6px;margin:2px 0;font-size:11px}
.sr label{width:56px;color:var(--dim)}.sr span{width:40px;text-align:right}
input[type=range]{flex:1}
input.z{accent-color:var(--sliceZ)}
input.i{accent-color:var(--sliceI)}
input.r{accent-color:var(--lidar)}
.grp{font-size:9px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px;margin-top:4px}
.mbtn{padding:3px 10px;font-size:10px;border:1px solid var(--grid);
      background:var(--bg);color:var(--dim);cursor:pointer;border-radius:3px;font-family:monospace}
.mbtn.on{border-color:var(--topk);color:var(--topk);background:#1a0800}
.num-in{width:52px;background:var(--panel);color:var(--txt);border:1px solid var(--grid);
        font-family:monospace;font-size:11px;padding:1px 3px;border-radius:2px;text-align:right}
.num-in:focus{outline:none;border-color:var(--circ)}
#elev-wrap{width:260px;background:var(--panel);border-left:1px solid var(--grid);
           display:flex;flex-direction:column;padding:4px}
#three-wrap{flex:1;width:100%;min-height:0;overflow:hidden}
#cameras{display:flex;gap:8px;background:var(--panel);border-top:1px solid var(--grid);
         padding:4px 10px;height:180px;flex-shrink:0;overflow:hidden}
.cam-wrap{display:flex;flex-direction:column;gap:2px;flex:1;min-width:0}
.cam-lbl{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.cam-wrap img{width:100%;height:100%;object-fit:contain;border:1px solid var(--grid);background:#000}
.rsz-h{width:5px;cursor:col-resize;background:var(--grid);flex-shrink:0;transition:background .15s}
.rsz-h:hover,.rsz-h.rsz-act{background:var(--rollout)}
.rsz-v{height:5px;cursor:row-resize;background:var(--grid);flex-shrink:0;transition:background .15s}
.rsz-v:hover,.rsz-v.rsz-act{background:var(--rollout)}
</style>
<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
<header>Spot MPC Debugger
  <span id="badge">connecting…</span>
  <span style="font-size:10px;color:var(--dim)">TOP=FWD · RIGHT=RIGHT · Driver in Spot body frame</span>
  <button id="reset-origin-btn" class="mbtn" onclick="resetOrigin()" style="margin-left:auto;border-color:var(--rollout);color:var(--rollout)">⟲ Reset Origin</button>
  <button id="theme-btn" class="mbtn" onclick="toggleTheme()">☀ Light</button>
</header>
<main>
  <div id="cw"><canvas id="cv"></canvas></div>
  <div class="rsz-h" id="rsz1"></div>
  <div id="elev-wrap">
    <div id="three-wrap"></div>
  </div>
  <div class="rsz-h" id="rsz2"></div>
  <div id="info">
    <div style="font-size:9px;color:var(--dim);letter-spacing:.06em;text-transform:uppercase;margin:3px 0">── Spot state ──</div>
    <div class="row"><span class="k">pos_vis x/y m</span><span class="v" id="i-pv">—</span></div>
    <div class="row"><span class="k">vel_vis x/y m/s</span><span class="v" id="i-vv">—</span></div>
    <div class="row"><span class="k">yaw</span><span class="v" id="i-yw">—</span></div>
    <hr>
    <div class="row"><span class="k">TERRAIN</span><span class="v" id="i-ter">—</span></div>
    <hr>
    <div class="row"><span class="k">CLUSTER raw</span><span class="v" id="i-cr">—</span></div>
    <div class="row"><span class="k">CLUSTER mapped</span><span class="v" id="i-cm">—</span></div>
    <hr>
    <div class="row"><span class="k">PLAN STEP</span><span class="v" id="i-ps">—</span></div>
    <div class="row"><span class="k">FROM</span><span class="v" id="i-pf">—</span></div>
    <div class="row"><span class="k">TO</span><span class="v" id="i-pt">—</span></div>
    <hr>
    <div style="font-size:9px;color:var(--rollout);letter-spacing:.06em;text-transform:uppercase;margin:3px 0">── SamplingMPC ──</div>
    <div class="row"><span class="k">v_cmd / om_cmd</span><span class="v" id="i-mpc-vo">—</span></div>
    <div class="row"><span class="k">guidance ratio</span><span class="v" id="i-mpc-gr">—</span></div>
    <div class="row"><span class="k">cluster score</span><span class="v" id="i-mpc-cs">—</span></div>
    <div class="row"><span class="k">bearing score</span><span class="v" id="i-mpc-bs">—</span></div>
    <div class="row"><span class="k">edt blocked frac</span><span class="v" id="i-mpc-eb">—</span></div>
    <div class="row"><span class="k">top rollout scores</span></div>
    <div id="i-mpc-scores"></div>
    <hr>
    <div class="row"><span class="k" style="color:var(--topk)">TOP-K PTS</span><span class="v" style="color:var(--topk);font-size:9px">(x right, y fwd) world</span></div>
    <div id="i-topk"></div>
    <hr>
    <div class="row"><span class="k">Mode</span><span class="v" id="i-md">—</span></div>
    <div class="row">
      <span class="k" style="color:var(--sliceZ)">Z slice (→ cluster)</span>
      <span class="v" id="i-zs" style="color:var(--sliceZ)">—</span>
    </div>
    <div class="row">
      <span class="k" style="color:var(--sliceI)">Int slice (→ cluster)</span>
      <span class="v" id="i-is" style="color:var(--sliceI)">—</span>
    </div>
    <div class="row"><span class="k">Range</span><span class="v" id="i-rg">—</span></div>
  </div>
</main>
<div class="rsz-v" id="rsz3"></div>
<div id="cameras">
  <div class="cam-wrap">
    <div class="cam-lbl">RAW ZED</div>
    <img src="/stream/raw" onerror="this.style.opacity='0.3'">
  </div>
  <div class="cam-wrap">
    <div class="cam-lbl">HSV FILTER</div>
    <img src="/stream/hsv" onerror="this.style.opacity='0.3'">
  </div>
</div>
<div class="rsz-v" id="rsz4"></div>
<div id="sliders">
  <h4>LIDAR FILTER  ·  publishes → /lidar_filter_config every 200 ms</h4>
  <div style="display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start">
    <div>
      <div class="grp" style="color:var(--sliceZ)">Z height  (→ clustering node)</div>
      <div class="sr"><label>Z min</label>
        <input class="z" type="range" id="sl-zlo" min="-3" max="1" step="0.01" value="-0.50">
        <input class="num-in" type="number" id="v-zlo" value="-0.50" step="0.01" min="-3" max="1"></div>
      <div class="sr"><label>Z max</label>
        <input class="z" type="range" id="sl-zhi" min="-1" max="3" step="0.01" value="0.15">
        <input class="num-in" type="number" id="v-zhi" value="0.15" step="0.01" min="-1" max="3"></div>
    </div>
    <div>
      <div class="grp" style="color:var(--sliceI)">Intensity  (AND z-height → clustering node)</div>
      <div class="sr"><label>Int min</label>
        <input class="i" type="range" id="sl-ilo" min="0" max="255" step="1" value="0">
        <input class="num-in" type="number" id="v-ilo" value="0" step="1" min="0" max="255"></div>
      <div class="sr"><label>Int max</label>
        <input class="i" type="range" id="sl-ihi" min="0" max="255" step="1" value="255">
        <input class="num-in" type="number" id="v-ihi" value="255" step="1" min="0" max="255"></div>
    </div>
    <div>
      <div class="grp" style="color:var(--lidar)">Range</div>
      <div class="sr"><label>R min</label>
        <input class="r" type="range" id="sl-rmin" min="0" max="2" step="0.05" value="0.5">
        <input class="num-in" type="number" id="v-rmin" value="0.50" step="0.05" min="0" max="2"></div>
      <div class="sr"><label>R max</label>
        <input class="r" type="range" id="sl-rmax" min="1" max="20" step="0.1" value="8">
        <input class="num-in" type="number" id="v-rmax" value="8.00" step="0.1" min="1" max="20"></div>
    </div>
    <div>
      <div class="grp">Slice mode</div>
      <div style="display:flex;gap:6px;margin-top:2px">
        <button class="mbtn on" id="btn-z" onclick="setMode('z')">Z height</button>
        <button class="mbtn"    id="btn-i" onclick="setMode('i')">Intensity</button>
      </div>
    </div>
  </div>
</div>
<script>
const CN={0:'open_space',1:'approach',2:'on_corridor',3:'exit'};
const TN={0:'Road',1:'Grass',2:'Sidewalk'};
const TC={0:'#ffaa44',1:'#44ff88',2:'#aaaaff'};
const RM={2:1,3:1,5:2,6:2,7:2,8:2,9:2,'-1':3,4:3,11:1,0:0,1:0};
const DARK={lidar:'#00cc44',topk:'#ff6600',rollout:'#00cfff',cmdvel:'#ff33aa',warn:'#ff4444',grid:'#30363d',dim:'#8b949e',
            circ:'#58a6ff',bg:'#161b22',
            cloudAll:'rgba(42,42,58,0.7)',sliceZ:'#ffcc00',sliceI:'#ff44cc'};
const LIGHT={lidar:'#006e1f',topk:'#cc3d00',rollout:'#0050bb',cmdvel:'#aa0066',warn:'#bb0000',grid:'#b0bac6',dim:'#4a5568',
             circ:'#1a4db5',bg:'#f0f4f8',
             cloudAll:'rgba(160,170,185,0.7)',sliceZ:'#996600',sliceI:'#990066'};
let C={...DARK};
const TOP_K=8;
let useIntensity=false, lastData=null, lightMode=false;

function toggleTheme(){
  lightMode=!lightMode;
  C=lightMode?{...LIGHT}:{...DARK};
  document.body.classList.toggle('light',lightMode);
  document.getElementById('theme-btn').textContent=lightMode?'🌙 Dark':'☀ Light';
  if(_t3.scene){
    _t3.scene.background=new THREE.Color(lightMode?0xf0f4f8:0x161b22);
    _t3.scene.children.forEach(o=>{
      if(o.isGridHelper){
        if(Array.isArray(o.material)){
          o.material[0].color.set(lightMode?0xb0bac6:0x30363d);
          o.material[1].color.set(lightMode?0xb0bac6:0x222830);
        }
      }
    });
    if(_t3.ptsMesh)_t3.ptsMesh.material.color.set(lightMode?0x8899aa:0x3a3a5a);
  }
  if(lastData)draw(lastData);
}
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');

function resetOrigin(){
  if(!confirm("Reset the plan origin to Spot's current pose (position + yaw)?\nDo this only when Spot is at the reference pose you want (0,0)/yaw=0 to mean — e.g. facing straight down the hallway."))return;
  const btn=document.getElementById('reset-origin-btn');
  btn.disabled=true;btn.textContent='resetting…';
  fetch('/api/reset_origin',{method:'POST'})
    .then(r=>r.json())
    .then(d=>{alert(d.message||(d.ok?'Origin reset.':'Reset failed.'));})
    .catch(e=>alert('Request failed: '+e))
    .finally(()=>{btn.disabled=false;btn.textContent='⟲ Reset Origin';});
}

function setMode(m){
  useIntensity=(m==='i');
  document.getElementById('btn-z').className='mbtn'+(useIntensity?'':' on');
  document.getElementById('btn-i').className='mbtn'+(useIntensity?' on':'');
  post();
}
function sliderVals(){
  return{z_lower:+sl('zlo'),z_upper:+sl('zhi'),
         int_lower:+sl('ilo'),int_upper:+sl('ihi'),
         min_range:+sl('rmin'),max_range:+sl('rmax'),
         use_intensity:useIntensity};
}
function sl(id){return document.getElementById('sl-'+id).value}
function post(){
  fetch('/api/set_config',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(sliderVals())});
}
['zlo','zhi','ilo','ihi','rmin','rmax'].forEach(id=>{
  const rng=document.getElementById('sl-'+id);
  const num=document.getElementById('v-'+id);
  rng.addEventListener('input',()=>{num.value=parseFloat(rng.value).toFixed(2);post();});
  num.addEventListener('change',()=>{
    const v=parseFloat(num.value);
    if(!isNaN(v)){
      rng.value=Math.max(parseFloat(rng.min),Math.min(parseFloat(rng.max),v));
      num.value=parseFloat(rng.value).toFixed(2);
      post();
    }
  });
});

/* Lidar beams: driver in Spot body frame (+x fwd, +y left). Display: x=right=−body_y, y=fwd=body_x → x=−r·sin(a), y=r·cos(a). */
function lidarPt(r,a,cx,cy,sc){
  return[cx - r*Math.sin(a)*sc, cy - r*Math.cos(a)*sc];
}
/* MPC rollout waypoints are (fwd, left) in the same body frame as the lidar beams above. */
function bodyPt(fwd,left,cx,cy,sc){
  return[cx - left*sc, cy - fwd*sc];
}

/* Distinct overlay for the single immediate command actually being sent this
   tick, as opposed to the full N-step hypothetical rollout drawn separately.
   Length ∝ v_cmd (0.75 = SamplingMPC's fixed native max sampled speed).
   Direction is a qualitative tilt toward the turn direction (om_cmd), not a
   physically-exact dt-based projection — dt isn't sent to the frontend, and
   this avoids the overlay silently going stale if MPC_DT is changed. */
function drawCommandArrow(vCmd,omCmd,cx,cy,sc){
  const V_REF=0.75, OM_CLAMP=1.5, TILT_SCALE=0.35, MAX_LEN_PX=sc*0.9;
  const vNorm=Math.max(0,Math.min(1,vCmd/V_REF));
  const len=Math.min(MAX_LEN_PX, sc*vNorm);
  if(len<2)return;
  const tilt=Math.max(-OM_CLAMP,Math.min(OM_CLAMP,omCmd))*TILT_SCALE;
  const ex=cx+len*Math.sin(tilt), ey=cy-len*Math.cos(tilt);
  ctx.strokeStyle=C.cmdvel;ctx.lineWidth=4;ctx.globalAlpha=0.95;
  ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();
  const dx=ex-cx,dy=ey-cy,L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L,hl=9;
  ctx.fillStyle=C.cmdvel;ctx.beginPath();
  ctx.moveTo(ex,ey);
  ctx.lineTo(ex-hl*(ux+0.4*uy),ey-hl*(uy-0.4*ux));
  ctx.lineTo(ex-hl*(ux-0.4*uy),ey-hl*(uy+0.4*ux));
  ctx.closePath();ctx.fill();
  ctx.globalAlpha=1;
}

function draw(d){
  const W=cv.width,H=cv.height,cx=W/2,cy=H/2;
  const maxR=(d.filter_cfg&&d.filter_cfg.max_range)||8;
  const sc=(W/2-24)/maxR;
  ctx.clearRect(0,0,W,H);ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,H);

  // Range rings
  ctx.setLineDash([4,6]);ctx.strokeStyle=C.grid;ctx.lineWidth=0.8;
  for(let r=2;r<=maxR;r+=2){
    ctx.beginPath();ctx.arc(cx,cy,r*sc,0,2*Math.PI);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=C.dim;ctx.font='10px monospace';ctx.textAlign='center';
    ctx.fillText(r+'m',cx+r*sc*0.72,cy-r*sc*0.72);
    ctx.setLineDash([4,6]);
  }
  ctx.setLineDash([]);

  // Axes
  ctx.strokeStyle=C.grid;ctx.lineWidth=0.8;
  ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(W,cy);ctx.stroke();
  ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();

  // Unit circle
  ctx.strokeStyle=C.circ;ctx.lineWidth=1.8;
  ctx.beginPath();ctx.arc(cx,cy,sc,0,2*Math.PI);ctx.stroke();

  // Labels — physical robot directions after x-flip
  ctx.fillStyle=C.dim;ctx.font='11px monospace';ctx.textAlign='center';
  ctx.fillText('FWD',cx,14);ctx.fillText('BCK',cx,H-3);
  ctx.textAlign='right';ctx.fillText('LEFT',24,cy+4);
  ctx.textAlign='left'; ctx.fillText('RIGHT',W-24,cy+4);
  ctx.textAlign='center';

  // Layer 1: raw cloud all — body frame (x=right, y=fwd; pre-rotated 90° CW server-side)
  if(d.cloud_all&&d.cloud_all.length){
    ctx.fillStyle=C.cloudAll;
    d.cloud_all.forEach(([x,y])=>{
      ctx.beginPath();ctx.arc(cx+x*sc,cy-y*sc,1.5,0,2*Math.PI);ctx.fill();
    });
  }

  // Layer 2: slice — noise=red (density-fail), structure=slice color (density-pass)
  const slCol=useIntensity?C.sliceI:C.sliceZ;
  if(d.cloud_slice_fail&&d.cloud_slice_fail.length){
    ctx.fillStyle='#ff4444';ctx.globalAlpha=0.70;
    d.cloud_slice_fail.forEach(([x,y])=>{
      ctx.beginPath();ctx.arc(cx+x*sc,cy-y*sc,2.5,0,2*Math.PI);ctx.fill();
    });
    ctx.globalAlpha=1;
  }
  if(d.cloud_slice_pass&&d.cloud_slice_pass.length){
    ctx.fillStyle=slCol;ctx.globalAlpha=0.85;
    d.cloud_slice_pass.forEach(([x,y])=>{
      ctx.beginPath();ctx.arc(cx+x*sc,cy-y*sc,2.5,0,2*Math.PI);ctx.fill();
    });
    ctx.globalAlpha=1;
  }

  // Processed ranges — x-flip via lidarPt (client-computed)
  if(d.processed_ranges&&d.processed_ranges.length){
    const n=d.processed_ranges.length;
    const hits=[];
    for(let i=0;i<n;i++){
      const r=d.processed_ranges[i];
      if(r<maxR*0.999) hits.push({r,a:2*Math.PI*i/n});
    }
    // Beams
    ctx.strokeStyle=C.lidar;ctx.lineWidth=1.1;ctx.globalAlpha=0.6;
    hits.forEach(h=>{
      const[ex,ey]=lidarPt(h.r,h.a,cx,cy,sc);
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();
    });
    ctx.globalAlpha=1;
    // Dots
    ctx.fillStyle=C.lidar;
    hits.forEach(h=>{
      const[ex,ey]=lidarPt(h.r,h.a,cx,cy,sc);
      ctx.beginPath();ctx.arc(ex,ey,3,0,2*Math.PI);ctx.fill();
    });
    // Top-k
    const sorted=[...hits].sort((a,b)=>a.r-b.r).slice(0,TOP_K);
    sorted.forEach(h=>{
      const[ex,ey]=lidarPt(h.r,h.a,cx,cy,sc);
      ctx.strokeStyle=C.topk;ctx.lineWidth=1.8;ctx.globalAlpha=0.85;
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();
      ctx.globalAlpha=1;
      ctx.fillStyle=C.topk;
      ctx.beginPath();ctx.arc(ex,ey,6,0,2*Math.PI);ctx.fill();
      ctx.strokeStyle='#fff';ctx.lineWidth=1;
      ctx.beginPath();ctx.arc(ex,ey,6,0,2*Math.PI);ctx.stroke();
    });
  }

  // MPC best-rollout path overlay (body frame: fwd, left)
  if(d.best_rollout&&d.best_rollout.length){
    ctx.strokeStyle=C.rollout;ctx.lineWidth=2.2;ctx.globalAlpha=0.9;
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    d.best_rollout.forEach(([fwd,left])=>{
      const[px,py]=bodyPt(fwd,left,cx,cy,sc);
      ctx.lineTo(px,py);
    });
    ctx.stroke();ctx.globalAlpha=1;
    ctx.fillStyle=C.rollout;
    d.best_rollout.forEach(([fwd,left])=>{
      const[px,py]=bodyPt(fwd,left,cx,cy,sc);
      ctx.beginPath();ctx.arc(px,py,3,0,2*Math.PI);ctx.fill();
    });
  }else{
    ctx.fillStyle=C.dim;ctx.font='13px monospace';ctx.textAlign='center';
    ctx.fillText('waiting for MPC rollout',cx,cy-16);
  }

  // Commanded velocity — the single immediate (v_cmd, om_cmd) actually being
  // sent this tick, distinct from the full N-step rollout drawn above.
  if(d.mpc_state&&d.mpc_state.length>=9){
    drawCommandArrow(d.mpc_state[4],d.mpc_state[5],cx,cy,sc);
  }

  // Robot origin dot
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(cx,cy,5,0,2*Math.PI);ctx.fill();

  // Legend — top-left corner
  const slCol2=useIntensity?C.sliceI:C.sliceZ;
  const slLbl=useIntensity?'Intensity slice':'Z slice';
  const items=[
    [C.cloudAll,       'Raw cloud (all)'],
    [slCol2,           slLbl],
    [C.lidar,          'Lidar beams'],
    [C.topk,           'Top-K inputs'],
    [C.rollout,        'MPC best rollout (N=8 lookahead)'],
    [C.cmdvel,         'Commanded velocity (this tick)'],
  ];
  const lx=8,ly=22,lh=16,dotR=5;
  ctx.save();
  ctx.globalAlpha=0.82;
  ctx.fillStyle=C.bg;
  ctx.fillRect(lx-4,ly-14,160,items.length*lh+6);
  ctx.globalAlpha=1;
  ctx.font='10px monospace';ctx.textAlign='left';
  items.forEach(([col,lbl],i)=>{
    const y=ly+i*lh;
    ctx.fillStyle=col;
    ctx.beginPath();ctx.arc(lx+dotR,y-3,dotR,0,2*Math.PI);ctx.fill();
    ctx.fillStyle=C.dim;
    ctx.fillText(lbl,lx+dotR*2+5,y);
  });
  ctx.restore();
}

/* Three.js 3D point cloud — robot (x,y,z) → Three.js (x, z, -y) so Z height = Three Y */
let _t3={inited:false,renderer:null,scene:null,camera:null,controls:null,
         ptsMesh:null,sliceMesh:null,planeLo:null,planeHi:null};

function initThree(){
  const wrap=document.getElementById('three-wrap');
  if(!wrap||_t3.inited)return;
  _t3.inited=true;
  const W=wrap.clientWidth||250, H=wrap.clientHeight||400;
  _t3.scene=new THREE.Scene();
  _t3.scene.background=new THREE.Color(0x161b22);
  _t3.camera=new THREE.PerspectiveCamera(50,W/H,0.05,200);
  _t3.camera.position.set(5,4,5);
  _t3.renderer=new THREE.WebGLRenderer({antialias:true});
  _t3.renderer.setSize(W,H);
  _t3.renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  wrap.appendChild(_t3.renderer.domElement);
  _t3.controls=new THREE.OrbitControls(_t3.camera,_t3.renderer.domElement);
  _t3.controls.enableDamping=true; _t3.controls.dampingFactor=0.08;
  _t3.scene.add(new THREE.GridHelper(16,8,0x30363d,0x222830));
  // All-points cloud (dim)
  _t3.ptsMesh=new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({size:0.06,color:0x3a3a5a,transparent:true,opacity:0.7})
  );
  _t3.scene.add(_t3.ptsMesh);
  // In-slice cloud — structure (density-pass, bright)
  _t3.sliceMesh=new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({size:0.10,color:0xffcc00})
  );
  _t3.scene.add(_t3.sliceMesh);
  // In-slice cloud — noise (density-fail, red)
  _t3.noiseMesh=new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({size:0.10,color:0xff4444})
  );
  _t3.scene.add(_t3.noiseMesh);
  // Slice planes
  const plGeo=new THREE.PlaneGeometry(16,16);
  const plMat=()=>new THREE.MeshBasicMaterial({color:0xffcc00,transparent:true,opacity:0.07,side:THREE.DoubleSide});
  _t3.planeLo=new THREE.Mesh(plGeo.clone(),plMat());
  _t3.planeHi=new THREE.Mesh(plGeo.clone(),plMat());
  _t3.planeLo.rotation.x=_t3.planeHi.rotation.x=-Math.PI/2;
  _t3.scene.add(_t3.planeLo); _t3.scene.add(_t3.planeHi);
  (function animate(){requestAnimationFrame(animate);_t3.controls.update();_t3.renderer.render(_t3.scene,_t3.camera);})();
}

function updateThree(d){
  if(!_t3.inited)initThree();
  if(!_t3.renderer)return;
  const cfg=d.filter_cfg||{};
  const zLo=cfg.z_lower??-1.26, zHi=cfg.z_upper??-0.56;
  const sliceHex=useIntensity?0xff44cc:0xffcc00;
  if(_t3.planeLo){_t3.planeLo.position.y=zLo;_t3.planeLo.material.color.setHex(sliceHex);}
  if(_t3.planeHi){_t3.planeHi.position.y=zHi;_t3.planeHi.material.color.setHex(sliceHex);}
  if(_t3.sliceMesh)_t3.sliceMesh.material.color.setHex(sliceHex);
  if(!d.cloud_3d||!d.cloud_3d.length)return;
  const pts=d.cloud_3d;
  const psi3=d.spot_yaw||0.0,cosP3=Math.cos(psi3),sinP3=Math.sin(psi3);
  // pts[i]=[body_fwd, body_left, z]; rotate to vision frame; Three.js: x=vx, y=z, z=-vy
  function toW3(bf,bl,z){const vx=bf*cosP3-bl*sinP3,vy=bf*sinP3+bl*cosP3;return[vx,z,-vy];}
  // All points
  const posA=new Float32Array(pts.length*3);
  for(let i=0;i<pts.length;i++){const[tx,ty,tz]=toW3(pts[i][0],pts[i][1],pts[i][2]);posA[i*3]=tx;posA[i*3+1]=ty;posA[i*3+2]=tz;}
  _t3.ptsMesh.geometry.setAttribute('position',new THREE.BufferAttribute(posA,3));
  _t3.ptsMesh.geometry.computeBoundingSphere();
  // Structure points (density-pass) — slice color
  function fillMesh(mesh,arr){
    if(!mesh)return;
    if(arr&&arr.length){
      const pos=new Float32Array(arr.length*3);
      for(let i=0;i<arr.length;i++){const[tx,ty,tz]=toW3(arr[i][0],arr[i][1],arr[i][2]);pos[i*3]=tx;pos[i*3+1]=ty;pos[i*3+2]=tz;}
      mesh.geometry.setAttribute('position',new THREE.BufferAttribute(pos,3));
      mesh.geometry.computeBoundingSphere();
    } else {
      mesh.geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(0),3));
    }
  }
  fillMesh(_t3.sliceMesh,d.cloud_3d_pass);
  fillMesh(_t3.noiseMesh,d.cloud_3d_fail);
}

function resizeThree(){
  const wrap=document.getElementById('three-wrap');
  if(!wrap||!_t3.renderer||!_t3.camera)return;
  const W=wrap.clientWidth, H=wrap.clientHeight;
  if(W<1||H<1)return;
  _t3.camera.aspect=W/H;
  _t3.camera.updateProjectionMatrix();
  _t3.renderer.setSize(W,H);
}

function $t(id,v,c){const e=document.getElementById(id);if(!e)return;e.textContent=v;if(c)e.style.color=c;}
function panel(d){
  const tid=d.terrain_id;
  $t('i-ter',TN[tid]||'T'+tid,TC[tid]||'#fff');
  const raw=d.raw_cluster;
  $t('i-cr',raw!=null?String(raw):'—');
  if(raw!=null){const m=RM[raw]??RM[String(raw)]??raw;$t('i-cm',m+' '+(CN[m]||'—'));}

  const ms=d.mpc_state;
  if(ms&&ms.length>=9){
    const f3=v=>(v>=0?'+':'')+v.toFixed(3);
    $t('i-pv',f3(ms[0])+' / '+f3(ms[1]));
    $t('i-vv',f3(ms[2])+' / '+f3(ms[3]));
    $t('i-yw',(ms[6]*180/Math.PI).toFixed(1)+'°');
    $t('i-mpc-vo',f3(ms[4])+' / '+f3(ms[5]));
    $t('i-pf',CN[ms[7]]||String(ms[7]),'#ff9955');
    $t('i-pt',CN[ms[8]]||String(ms[8]),'#ff9955');
  }
  $t('i-ps',String(d.plan_step??'—'),'#ffdd88');

  const gd=d.guidance_debug;
  if(gd&&gd.length>=4){
    $t('i-mpc-gr',gd[0].toFixed(3));
    $t('i-mpc-cs',gd[1].toFixed(2));
    $t('i-mpc-bs',gd[2].toFixed(2));
    const eb=gd[3];
    $t('i-mpc-eb',(eb*100).toFixed(0)+'%', eb<0.3?'#00cc44':eb<0.6?'#ffaa00':'#ff4444');
  }
  const scEl=document.getElementById('i-mpc-scores');
  if(scEl){
    if(d.mpc_scores&&d.mpc_scores.length){
      const mx=Math.max(...d.mpc_scores.map(Math.abs),1e-6);
      scEl.innerHTML=d.mpc_scores.map(s=>{
        const w=Math.min(100,Math.abs(s)/mx*100);
        const col=s>=0?'#00cc44':'#ff4444';
        return `<div style="display:flex;align-items:center;gap:4px;margin:1px 0">
          <div style="width:${w}px;height:6px;background:${col}"></div>
          <span style="font-size:9px;color:var(--dim)">${s.toFixed(1)}</span></div>`;
      }).join('');
    } else {
      scEl.innerHTML='<div class="row"><span class="v" style="color:var(--dim)">waiting...</span></div>';
    }
  }

  const topkEl=document.getElementById('i-topk');
  if(topkEl){
    if(d.processed_ranges&&d.processed_ranges.length){
      const n=d.processed_ranges.length;
      const maxR=(d.filter_cfg&&d.filter_cfg.max_range)||8.0;
      const psi=d.spot_yaw||0.0;
      const cosP=Math.cos(psi), sinP=Math.sin(psi);
      const hits=[];
      for(let i=0;i<n;i++){
        const r=d.processed_ranges[i];
        if(r<maxR*0.999){const a=2*Math.PI*i/n;hits.push({r,a});}
      }
      const sorted=[...hits].sort((a,b)=>a.r-b.r).slice(0,TOP_K);
      topkEl.innerHTML=sorted.map((h,i)=>{
        // display frame: xb=right=−r*sin(a), yb=fwd=r*cos(a)
        // world frame: rotate by yaw psi to undo robot rotation
        const xb=-h.r*Math.sin(h.a), yb=h.r*Math.cos(h.a);
        const px=(xb*cosP - yb*sinP).toFixed(2);
        const py=(xb*sinP + yb*cosP).toFixed(2);
        return `<div class="row"><span class="k" style="color:var(--topk)">  pt ${i}</span>`+
               `<span class="v" style="color:var(--topk);font-family:monospace">(${px}, ${py})</span></div>`;
      }).join('')||'<div class="row"><span class="v" style="color:var(--dim)">none</span></div>';
    }else{
      topkEl.innerHTML='<div class="row"><span class="v" style="color:var(--dim)">waiting...</span></div>';
    }
  }
  const cfg=d.filter_cfg||{};
  const mStr=cfg.use_intensity?'Intensity + Z → clustering':'Z height → clustering';
  $t('i-md',mStr,cfg.use_intensity?'#ff44cc':'#ffcc00');
  $t('i-zs','['+Number(cfg.z_lower||0).toFixed(2)+', '+Number(cfg.z_upper||0).toFixed(2)+']');
  $t('i-is','['+Number(cfg.int_lower||0).toFixed(0)+', '+Number(cfg.int_upper||500).toFixed(0)+']');
  $t('i-rg','['+Number(cfg.min_range||0).toFixed(1)+', '+Number(cfg.max_range||8).toFixed(1)+'] m');
}

async function loop(){
  const badge=document.getElementById('badge');
  while(true){
    try{
      const r=await fetch('/api/state');
      if(r.ok){const d=await r.json();lastData=d;draw(d);panel(d);updateThree(d);
               badge.textContent='live';badge.className='live';}
    }catch(e){badge.textContent='disconnected';badge.className='';}
    await new Promise(r=>setTimeout(r,80));
  }
}
function resize(){
  const w=document.getElementById('cw');
  const s=Math.min(w.clientWidth-12,w.clientHeight-12,700);
  cv.width=s;cv.height=s;if(lastData)draw(lastData);
  resizeThree();
}
function initSliders(cfg){
  if(!cfg)return;
  const map={zlo:'z_lower',zhi:'z_upper',ilo:'int_lower',ihi:'int_upper',rmin:'min_range',rmax:'max_range'};
  for(const[id,key] of Object.entries(map)){
    const el=document.getElementById('sl-'+id);
    const sp=document.getElementById('v-'+id);
    if(el&&cfg[key]!=null){el.value=cfg[key];sp.value=parseFloat(cfg[key]).toFixed(2);}
  }
}
function makeSplitter(el,a,b,axis){
  el.addEventListener('mousedown',function(e){
    e.preventDefault();
    el.classList.add('rsz-act');
    var start=axis==='h'?e.clientX:e.clientY;
    var aSize=axis==='h'?a.offsetWidth:a.offsetHeight;
    var bSize=axis==='h'?b.offsetWidth:b.offsetHeight;
    function onMove(ev){
      var d=(axis==='h'?ev.clientX:ev.clientY)-start;
      a.style.flex='none';
      if(axis==='h'){
        a.style.width=Math.max(80,aSize+d)+'px';
        b.style.width=Math.max(80,bSize-d)+'px';
      } else {
        a.style.height=Math.max(80,aSize+d)+'px';
        b.style.height=Math.max(40,bSize-d)+'px';
      }
      resize();
    }
    function onUp(){
      el.classList.remove('rsz-act');
      document.removeEventListener('mousemove',onMove);
      document.removeEventListener('mouseup',onUp);
    }
    document.addEventListener('mousemove',onMove);
    document.addEventListener('mouseup',onUp);
  });
}
makeSplitter(document.getElementById('rsz1'),document.getElementById('cw'),document.getElementById('elev-wrap'),'h');
makeSplitter(document.getElementById('rsz2'),document.getElementById('elev-wrap'),document.getElementById('info'),'h');
makeSplitter(document.getElementById('rsz3'),document.querySelector('main'),document.getElementById('cameras'),'v');
makeSplitter(document.getElementById('rsz4'),document.getElementById('cameras'),document.getElementById('sliders'),'v');
window.addEventListener('resize',resize);
fetch('/api/state').then(r=>r.json()).then(d=>{initSliders(d.filter_cfg);resize();loop();}).catch(()=>{resize();loop();});
</script>
</body>
</html>"""


def run_web(state: DebugState, ros_node: 'DebugSubscriber', port=WEB_PORT):
    try:
        from flask import Flask, jsonify, request as freq, Response
    except ImportError:
        print("[ERROR] Flask not installed.  pip install flask")
        return
    app = Flask(__name__)

    @app.route('/')
    def index():
        return Response(_HTML, mimetype='text/html')

    @app.route('/api/state')
    def api_state():
        snap = state.snapshot()

        cloud_all = []
        cloud_slice_pass, cloud_slice_fail = [], []
        cloud_3d, cloud_3d_pass, cloud_3d_fail = [], [], []
        rc  = snap.get('raw_cloud')
        cfg_s = snap['filter_cfg']
        if rc is not None and len(rc):
            all_xy, slice_xy = _apply_slice_filter(rc, cfg_s)
            if all_xy is not None and len(all_xy):
                stride = max(1, len(all_xy) // 600)
                cloud_all = all_xy[::stride].tolist()
            if slice_xy is not None and len(slice_xy):
                pm = _apply_density_filter(slice_xy,
                                           cfg_s['density_radius'],
                                           cfg_s['min_neighbors'])
                for pts_sub, dest in [(slice_xy[pm], cloud_slice_pass),
                                      (slice_xy[~pm], cloud_slice_fail)]:
                    if len(pts_sub):
                        s = max(1, len(pts_sub) // 300)
                        dest.extend(pts_sub[::s].tolist())

            z_all_w = rc[:, 2]
            ib_all  = (z_all_w >= cfg_s['z_lower']) & (z_all_w <= cfg_s['z_upper'])
            dp3     = np.zeros(len(rc), dtype=bool)
            if np.any(ib_all):
                pm3 = _apply_density_filter(rc[ib_all, :2],
                                            cfg_s['density_radius'],
                                            cfg_s['min_neighbors'])
                dp3[np.where(ib_all)[0][pm3]] = True
            df3 = ib_all & ~dp3

            stride_3d = max(1, len(rc) // 400)
            cloud_3d  = rc[::stride_3d, :3].tolist()
            for pts_sub, dest, cap in [(rc[dp3], cloud_3d_pass, 200),
                                       (rc[df3], cloud_3d_fail, 100)]:
                if len(pts_sub):
                    s = max(1, len(pts_sub) // cap)
                    dest.extend(pts_sub[::s, :3].tolist())

        return jsonify(dict(
            terrain_id       = snap['terrain_id'],
            raw_cluster      = snap['raw_cluster'],
            plan_step        = snap['plan_step'],
            spot_yaw         = snap['spot_yaw'],
            processed_ranges = snap['processed_ranges'].tolist()
                               if snap['processed_ranges'] is not None else None,
            cloud_all        = cloud_all,
            cloud_slice_pass = cloud_slice_pass,
            cloud_slice_fail = cloud_slice_fail,
            cloud_3d         = cloud_3d,
            cloud_3d_pass    = cloud_3d_pass,
            cloud_3d_fail    = cloud_3d_fail,
            filter_cfg       = snap['filter_cfg'],
            mpc_state        = snap.get('mpc_state'),
            best_rollout     = snap['best_rollout'].tolist()
                               if snap.get('best_rollout') is not None else None,
            guidance_debug   = snap.get('guidance_debug'),
            mpc_scores       = snap.get('mpc_scores'),
        ))

    # Build a "no signal" placeholder JPEG once at startup
    if _HAS_CV:
        _ns = np.zeros((30, 160, 3), dtype=np.uint8)
        cv2.putText(_ns, 'no signal', (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
        _, _buf = cv2.imencode('.jpg', _ns, [cv2.IMWRITE_JPEG_QUALITY, 50])
        _NO_SIGNAL = _buf.tobytes()
    else:
        _NO_SIGNAL = b''

    def _mjpeg(get_frame):
        while True:
            jpg = get_frame() or _NO_SIGNAL
            if jpg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            time.sleep(0.1)

    @app.route('/stream/raw')
    def stream_raw():
        return Response(_mjpeg(state.get_raw_frame),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/stream/hsv')
    def stream_hsv():
        return Response(_mjpeg(state.get_hsv_frame),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/set_config', methods=['POST'])
    def api_set_config():
        data = freq.get_json(silent=True) or {}
        allowed = {'z_upper', 'z_lower', 'z2_upper', 'z2_lower',
                   'max_range', 'min_range', 'use_intensity',
                   'int_lower', 'int_upper'}
        state.filter_cfg.set(**{k: v for k, v in data.items() if k in allowed})
        return jsonify({'ok': True})

    @app.route('/api/reset_origin', methods=['POST'])
    def api_reset_origin():
        ok, msg = ros_node.call_reset_origin()
        return jsonify({'ok': ok, 'message': msg})

    print(f"[WEB] http://0.0.0.0:{port}  (remote: http://<robot-ip>:{port})")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=WEB_PORT)
    args, _ = ap.parse_known_args()

    filter_cfg = FilterConfig()
    state      = DebugState(filter_cfg)
    rclpy.init()
    node = DebugSubscriber(state)
    ros_t = threading.Thread(target=_ros_thread, args=(node,), daemon=True)
    ros_t.start()

    run_web(state, node, port=args.port)


if __name__ == '__main__':
    main()
