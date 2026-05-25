#!/usr/bin/env python3
"""
DGPPO Debug Visualizer v2

Usage:
  ros2 run dgppo_ros_node_pkg dgppo_debug_visualizer_v2 -- [plan.json]
  ros2 run dgppo_ros_node_pkg dgppo_debug_visualizer_v2 -- [plan.json] --port 8080

Coordinate conventions (after corrections):
  Display: TOP = robot FORWARD, RIGHT = robot RIGHT, LEFT = robot LEFT
  - Raw cloud: driver already in Spot body frame (+x fwd, +y left); 90° CW to display → (−y, x)
  - Processed-range beams: same net rotation → (−r·sin θ, r·cos θ)
  - Bearing/heading arrows: angle 0 = forward = UP  (stored as raw radians, +π/2 applied at draw)
  - Action arrow: atan2(a1_fwd, a0_right); forward → UP, right → RIGHT — already correct

Z-height mode  → yellow slice; z-band sent to clustering node → affects actual clustering
Intensity mode → magenta slice; z-band AND intensity band sent to clustering node → affects actual clustering
"""

import sys, os, json, math, threading, argparse, time
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
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py.point_cloud2 import read_points
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_RANGES   = 72
TOP_K        = 8
HISTORY_LEN  = 20
VEL_HIST_LEN = 300   # ~24 s at 80 ms update rate (fits 3+ cycles of 4 s half-period step test)
WEB_PORT     = 8765

TERRAIN_NAMES = {0: "Road", 1: "Grass", 2: "Sidewalk"}
CLUSTER_NAMES = {0: "open_space", 1: "approach_bridge", 2: "on_bridge", 3: "exit_bridge"}
RAW_TO_MAPPED = {
    **{k: 0 for k in [0, 1]},
    **{k: 1 for k in [2, 3, 10, 11]},
    **{k: 2 for k in [5, 6, 7, 8, 9, 12]},
    **{k: 3 for k in [-1, 4]},
}

# Outdoor palette
C_BG            = '#0d1117'
C_PANEL         = '#161b22'
C_GRID          = '#30363d'
C_CIRCLE        = '#58a6ff'
C_TEXT          = '#ffffff'
C_DIM           = '#8b949e'
C_LIDAR         = '#00cc44'    # green: processed-range beams
C_TOPK          = '#ff6600'    # orange: top-k policy inputs
C_ACTION        = '#00cfff'    # cyan: action direction
C_BEARING       = '#ffd700'    # gold: plan bearing
C_HEADING       = '#cc44ff'    # purple: spot heading
C_TRAIL         = '#3060cc'    # blue: action trail
C_WARN          = '#ff4444'    # red: stop/no-action
C_REP_VEL       = '#44aaff'    # blue: reported velocity arrow (matches vel-plot rep color)
C_RAW_ACTION    = '#ff9900'    # orange: raw (pre-rotation) policy action unit vector
C_ROT_ACTION    = '#ff4400'    # red-orange: post-rotation action (what is actually sent to Spot)
C_CLOUD_ALL     = '#2a2a3a'    # dim: all raw cloud XY
C_CLOUD_SLICE_Z = '#ffcc00'    # yellow: Z-height slice (controls clustering)
C_CLOUD_SLICE_I = '#ff44cc'    # magenta: intensity slice (visual only)


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
    def __init__(self, plan_sequence, bearing_map, filter_cfg: FilterConfig):
        self._lock           = threading.Lock()
        self.action          = np.zeros(2)
        self.has_action      = False
        self.raw_cluster     = None
        self.terrain_id      = 1
        self.plan_step       = 0
        self.plan_sequence   = plan_sequence
        self.bearing_map     = bearing_map
        self.spot_yaw        = None
        self.processed_ranges = None
        self.raw_cloud       = None   # (N,4): x,y,z,intensity from lidar
        self.filter_cfg      = filter_cfg
        self.raw_frame_jpg   = None   # bytes: JPEG of raw ZED image
        self.hsv_frame_jpg   = None   # bytes: JPEG of HSV-segmented image
        self.state_debug     = None   # 12-float transform debug from /dgppo_state_debug
        self.world_alpha_rad = 0.0    # CW angle from Spot boot-up fwd to sim +Y, in radians
        # velocity time-series (cmd vs reported, vision frame)
        self._vel_t0    = None
        self.vel_times   = deque(maxlen=VEL_HIST_LEN)
        self.cmd_vx_hist = deque(maxlen=VEL_HIST_LEN)
        self.cmd_vy_hist = deque(maxlen=VEL_HIST_LEN)
        self.rep_vx_hist = deque(maxlen=VEL_HIST_LEN)
        self.rep_vy_hist = deque(maxlen=VEL_HIST_LEN)
        # step-test metrics: persist once detected, reset when cmd returns to ~0
        self.step_delay_ms = None
        self.step_rise_ms  = None
        # per-cycle metrics: one entry per rising edge seen
        self.cycle_metrics = deque(maxlen=50)
        # DGPPO inference time history (ms, from sd[14])
        self.inf_ms_hist = deque(maxlen=VEL_HIST_LEN)

    def set_action(self, a0, a1):
        with self._lock:
            self.action[:] = [a0, a1]
            self.has_action = True

    def set_cluster(self, raw):
        with self._lock: self.raw_cluster = raw

    def set_terrain(self, tid):
        with self._lock: self.terrain_id = tid

    def set_plan_step(self, step):
        with self._lock: self.plan_step = step

    def set_spot_yaw(self, yaw):
        with self._lock: self.spot_yaw = yaw

    def set_processed_ranges(self, r):
        with self._lock: self.processed_ranges = r

    def set_raw_cloud(self, pts):
        with self._lock: self.raw_cloud = pts

    def set_raw_frame(self, jpg):
        with self._lock: self.raw_frame_jpg = jpg

    def set_hsv_frame(self, jpg):
        with self._lock: self.hsv_frame_jpg = jpg

    def set_state_debug(self, data):
        with self._lock:
            self.state_debug = data
            now = time.time()
            if self._vel_t0 is None:
                self._vel_t0 = now
            self.vel_times.append(now - self._vel_t0)
            self.cmd_vx_hist.append(data[10])
            self.cmd_vy_hist.append(data[11])
            self.rep_vx_hist.append(data[2])
            self.rep_vy_hist.append(data[3])
            if len(data) > 14:
                self.inf_ms_hist.append(data[14])
            # Persist step-test metrics; reset only when cmd returns near zero
            t_arr  = np.array(self.vel_times)
            cmd_vx = np.array(self.cmd_vx_hist)
            rep_vx = np.array(self.rep_vx_hist)
            if len(t_arr) >= 5 and float(np.mean(cmd_vx[-5:])) < 0.1:
                self.step_delay_ms = None
                self.step_rise_ms  = None
            else:
                d, r = _detect_step_metrics(t_arr, cmd_vx, rep_vx)
                if d is not None:
                    self.step_delay_ms = d
                if r is not None:
                    self.step_rise_ms = r
            # Per-cycle metrics: append any new edges not yet recorded
            cycles = _detect_cycle_metrics(t_arr, cmd_vx, rep_vx)
            if cycles:
                last_t = self.cycle_metrics[-1][0] if self.cycle_metrics else -1.0
                for entry in cycles:
                    if entry[0] > last_t + 0.1:
                        self.cycle_metrics.append(entry)
                        last_t = entry[0]

    def get_raw_frame(self):
        with self._lock: return self.raw_frame_jpg

    def get_hsv_frame(self):
        with self._lock: return self.hsv_frame_jpg

    def snapshot(self):
        with self._lock:
            return dict(
                action           = self.action.copy(),
                has_action       = self.has_action,
                raw_cluster      = self.raw_cluster,
                terrain_id       = self.terrain_id,
                plan_step        = self.plan_step,
                plan_sequence    = self.plan_sequence,
                bearing_map      = self.bearing_map,
                spot_yaw         = self.spot_yaw,
                processed_ranges = self.processed_ranges.copy()
                                   if self.processed_ranges is not None else None,
                raw_cloud        = self.raw_cloud.copy()
                                   if self.raw_cloud is not None else None,
                filter_cfg       = self.filter_cfg.get(),
                state_debug      = self.state_debug,
                vel_times        = list(self.vel_times),
                cmd_vx_hist      = list(self.cmd_vx_hist),
                cmd_vy_hist      = list(self.cmd_vy_hist),
                rep_vx_hist      = list(self.rep_vx_hist),
                rep_vy_hist      = list(self.rep_vy_hist),
                step_delay_ms    = self.step_delay_ms,
                step_rise_ms     = self.step_rise_ms,
                cycle_metrics    = list(self.cycle_metrics),
                inf_ms_hist      = list(self.inf_ms_hist),
                world_alpha_rad  = self.world_alpha_rad,
            )


# ── ROS node ──────────────────────────────────────────────────────────────────

class DebugSubscriber(Node):
    def __init__(self, state: DebugState):
        super().__init__('dgppo_debug_visualizer_v2')
        self.state = state
        sub = self.create_subscription
        sub(Float32MultiArray, '/dgppo_action',      self._cb_action,   10)
        sub(Int16,             '/predicted_cluster', self._cb_cluster,  10)
        sub(Int32,             '/current_terrain',   self._cb_terrain,  10)
        sub(Int32,             '/dgppo_plan_step',   self._cb_planstep, 10)
        sub(Float32MultiArray, '/processed_ranges',  self._cb_ranges,      10)
        sub(Float32MultiArray, '/dgppo_spot_yaw',    self._cb_spot_yaw,    10)
        sub(Float32MultiArray, '/dgppo_state_debug', self._cb_state_debug, 10)
        sub(PointCloud2,       '/livox/lidar',       self._cb_cloud,    qos_profile_sensor_data)
        sub(Image, '/hamilton_zed2i/zed_node/rgb/image_rect_color', self._cb_raw_img, qos_profile_sensor_data)
        sub(Image, '/segmentor_image',                   self._cb_hsv_img, 10)
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                  reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST)
        sub(Float32MultiArray, '/dgppo_world_alpha', self._cb_world_alpha, _latched_qos)
        self._cfg_pub = self.create_publisher(Float32MultiArray, '/lidar_filter_config', 10)
        self.create_timer(0.2, self._pub_cfg)

    def _cb_world_alpha(self, msg):
        if msg.data:
            self.state.world_alpha_rad = float(msg.data[0])
            self.get_logger().info(f"world_alpha_rad={self.state.world_alpha_rad:.4f} ({math.degrees(self.state.world_alpha_rad):.1f}°)")

    def _cb_action(self, msg):
        if len(msg.data) >= 2:
            self.state.set_action(msg.data[0], msg.data[1])

    def _cb_cluster(self, msg):  self.state.set_cluster(msg.data)
    def _cb_terrain(self, msg):  self.state.set_terrain(msg.data)
    def _cb_planstep(self, msg): self.state.set_plan_step(msg.data)

    def _cb_ranges(self, msg):
        if msg.data:
            self.state.set_processed_ranges(np.array(msg.data, dtype=np.float32))

    def _cb_spot_yaw(self, msg):
        if msg.data:
            self.state.set_spot_yaw(float(msg.data[0]))

    def _cb_state_debug(self, msg):
        if len(msg.data) >= 12:
            self.state.set_state_debug(list(msg.data))

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


def _ros_thread(state: DebugState):
    rclpy.init()
    node = DebugSubscriber(state)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── Helpers ───────────────────────────────────────────────────────────────────

def topk_from_ranges(ranges, k=TOP_K, max_range=8.0):
    """Return (k,2) display-XY of closest k bins. Display: x=right=−body_y, y=fwd=body_x."""
    n      = len(ranges)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    valid  = np.where(ranges < max_range * 0.999)[0]
    if len(valid) == 0:
        return np.empty((0, 2))
    idx = valid[np.argsort(ranges[valid])[:k]]
    return np.column_stack([-ranges[idx] * np.sin(angles[idx]),
                              ranges[idx] * np.cos(angles[idx])])


def _bearing_for_step(snap):
    ps, seq, bm = snap['plan_step'], snap['plan_sequence'], snap['bearing_map']
    if ps < len(seq):
        step = seq[ps]
        return step, bm.get(f"{step['start']}-{step['next']}")
    return None, None


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


def _estimate_lag_ms(t_arr, cmd, rep):
    """Cross-correlation lag estimate (cmd → reported) in milliseconds.
    Returns None when signal variance is too low for a reliable estimate."""
    if len(t_arr) < 20 or cmd.std() < 0.02 or rep.std() < 0.02:
        return None
    dt = float(np.mean(np.diff(t_arr)))
    cc = np.correlate(rep - rep.mean(), cmd - cmd.mean(), mode='full')
    lags = np.arange(-(len(cmd) - 1), len(cmd))
    lag_s = float(lags[int(np.argmax(cc))]) * dt
    return lag_s * 1000.0 if 0.0 <= lag_s <= 3.0 else None


def _detect_step_metrics(t_arr, cmd_vx, rep_vx):
    """Detect pure transport delay and 0→90% rise time from the most recent
    rising edge in cmd_vx.  Returns (delay_ms, rise_ms); either may be None."""
    if len(t_arr) < 5:
        return None, None
    STEP_ON = 0.25   # cmd must cross this threshold upward to count as a step
    REP_THR = 0.02   # first detectable motion in reported vel

    # Find most recent rising edge in cmd
    step_idx = None
    for i in range(len(cmd_vx) - 1, 0, -1):
        if cmd_vx[i] >= STEP_ON and cmd_vx[i - 1] < STEP_ON:
            step_idx = i
            break
    if step_idx is None:
        return None, None

    t_step = t_arr[step_idx]

    # Pure delay: first rep sample above REP_THR after the step edge
    delay_ms = None
    for i in range(step_idx, len(rep_vx)):
        if rep_vx[i] > REP_THR:
            delay_ms = (t_arr[i] - t_step) * 1000.0
            break

    # Rise time: step edge → rep reaches 90 % of its own actual peak
    # (use reported peak, not commanded, because Spot has steady-state error)
    rise_ms = None
    rep_after = rep_vx[step_idx:]
    if len(rep_after) > 3:
        actual_peak = float(np.max(rep_after))
        if actual_peak > 0.05:
            t90 = actual_peak * 0.9
            for i in range(step_idx, len(rep_vx)):
                if rep_vx[i] >= t90:
                    rise_ms = (t_arr[i] - t_step) * 1000.0
                    break

    return delay_ms, rise_ms


def _detect_cycle_metrics(t_arr, cmd_vx, rep_vx):
    """For each rising edge in cmd_vx compute per-cycle pure delay and 0→90% rise time.
    Returns list of (t_edge, delay_ms, rise_ms) — rise_ms is None if not yet reached."""
    STEP_ON = 0.25
    REP_THR = 0.02
    edges = [i for i in range(1, len(cmd_vx))
             if cmd_vx[i] >= STEP_ON and cmd_vx[i - 1] < STEP_ON]
    results = []
    for step_idx in edges:
        t_step = t_arr[step_idx]
        # Window: this edge → next falling edge (or end of buffer)
        end_idx = len(cmd_vx)
        for j in range(step_idx + 1, len(cmd_vx)):
            if cmd_vx[j] < STEP_ON and cmd_vx[j - 1] >= STEP_ON:
                end_idx = j
                break
        if end_idx - step_idx < 5:
            continue
        delay_ms = None
        for i in range(step_idx, end_idx):
            if rep_vx[i] > REP_THR:
                delay_ms = (t_arr[i] - t_step) * 1000.0
                break
        rise_ms = None
        rep_window = rep_vx[step_idx:end_idx]
        if len(rep_window) > 3:
            actual_peak = float(np.max(rep_window))
            if actual_peak > 0.05:
                t90 = actual_peak * 0.9
                for i in range(step_idx, end_idx):
                    if rep_vx[i] >= t90:
                        rise_ms = (t_arr[i] - t_step) * 1000.0
                        break
        if delay_ms is not None:
            results.append((float(t_step), float(delay_ms),
                            float(rise_ms) if rise_ms is not None else None))
    return results


# ── Web server ────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DGPPO Debugger v2</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--grid:#30363d;--txt:#fff;--dim:#8b949e;
      --lidar:#00cc44;--topk:#ff6600;--act:#00cfff;--bear:#ffd700;
      --head:#cc44ff;--warn:#ff4444;--circ:#58a6ff;
      --sliceZ:#ffcc00;--sliceI:#ff44cc;}
body.light{--bg:#f0f4f8;--panel:#e8ecf2;--grid:#b0bac6;--txt:#0d1117;--dim:#4a5568;
           --lidar:#006e1f;--topk:#cc3d00;--act:#0050bb;--bear:#8a6000;
           --head:#6e0099;--warn:#bb0000;--circ:#1a4db5;
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
.rsz-h:hover,.rsz-h.rsz-act{background:var(--act)}
.rsz-v{height:5px;cursor:row-resize;background:var(--grid);flex-shrink:0;transition:background .15s}
.rsz-v:hover,.rsz-v.rsz-act{background:var(--act)}
#vel-area{background:var(--panel);border-top:1px solid var(--grid);padding:4px 10px;height:130px;flex-shrink:0;overflow:hidden;display:flex;align-items:stretch}
#vcv{flex:1;display:block}
</style>
<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
<header>DGPPO Policy Debugger v2
  <span id="badge">connecting…</span>
  <span style="font-size:10px;color:var(--dim)">TOP=FWD · Driver in Spot body frame · Arrows 0=FWD=UP</span>
  <button id="theme-btn" class="mbtn" onclick="toggleTheme()" style="margin-left:auto">☀ Light</button>
</header>
<main>
  <div id="cw"><canvas id="cv"></canvas></div>
  <div class="rsz-h" id="rsz1"></div>
  <div id="elev-wrap">
    <div id="three-wrap"></div>
  </div>
  <div class="rsz-h" id="rsz2"></div>
  <div id="info">
    <div style="font-size:9px;color:#aaaaff;text-transform:uppercase;letter-spacing:.06em;margin:3px 0">── Sim Delay Params ──</div>
    <div class="row"><span class="k">in  (DGPPO inference)</span><span class="v" id="i-sim-in" style="color:#fff">—</span></div>
    <div class="row"><span class="k">out (Spot response)</span><span class="v" id="i-sim-out" style="color:#fff">—</span></div>
    <div class="row"><span class="k">total (in + out)</span><span class="v" id="i-sim-tot" style="color:#fff">—</span></div>
    <hr>
    <div style="font-size:9px;color:#555566;text-transform:uppercase;letter-spacing:.06em;margin:3px 0">── Latency ──</div>
    <div class="row"><span class="k" style="color:#ff4466">cmd vx/vy m/s</span><span class="v" id="i-cvx" style="color:#ff4466">—</span></div>
    <div class="row"><span class="k" style="color:#44aaff">rep vx/vy m/s</span><span class="v" id="i-rvx" style="color:#44aaff">—</span></div>
    <div class="row"><span class="k" style="color:#ffaa44">err vx/vy m/s</span><span class="v" id="i-evx" style="color:#ffaa44">—</span></div>
    <div class="row"><span class="k">xcorr lag vx</span><span class="v" id="i-lgvx">—</span></div>
    <div class="row"><span class="k">xcorr lag vy</span><span class="v" id="i-lgvy">—</span></div>
    <div class="row"><span class="k" style="color:#44aaff">σ_rep vx/vy</span><span class="v" id="i-srv" style="color:#44aaff">—</span></div>
    <div class="row"><span class="k" style="color:#ffaa44">σ_err vx/vy</span><span class="v" id="i-sev" style="color:#ffaa44">—</span></div>
    <div class="row"><span class="k">pure delay</span><span class="v" id="i-dly">—</span></div>
    <div class="row"><span class="k">rise 0→90%</span><span class="v" id="i-rse">—</span></div>
    <div class="row"><span class="k" style="color:#ff9955">DGPPO inference</span><span class="v" id="i-inf" style="color:#ff9955">—</span></div>
    <div class="row"><span class="k" style="color:#8b949e">  mean / std</span><span class="v" id="i-inf-stat" style="color:#8b949e">—</span></div>
    <hr>
    <div class="row"><span class="k">TERRAIN</span><span class="v" id="i-ter">—</span></div>
    <hr>
    <div class="row"><span class="k">CLUSTER raw</span><span class="v" id="i-cr">—</span></div>
    <div class="row"><span class="k">CLUSTER mapped</span><span class="v" id="i-cm">—</span></div>
    <hr>
    <div class="row"><span class="k">PLAN STEP</span><span class="v" id="i-ps">—</span></div>
    <div class="row"><span class="k">FROM</span><span class="v" id="i-pf">—</span></div>
    <div class="row"><span class="k">TO</span><span class="v" id="i-pt">—</span></div>
    <div class="row"><span class="k">BEARING</span><span class="v" id="i-br">—</span></div>
    <div class="row"><span class="k" style="color:#ffd700">bear - α</span><span class="v" id="i-bw" style="color:#ffd700">—</span></div>
    <div class="row"><span class="k" style="color:#ffd700">bear - α - ψ</span><span class="v" id="i-bb" style="color:#ffd700">—</span></div>
    <div class="row"><span class="k">ACT↔BEAR</span><span class="v" id="i-df">—</span></div>
    <hr>
    <div class="row"><span class="k">a[0] right</span><span class="v" id="i-a0">—</span></div>
    <div class="row"><span class="k">a[1] fwd</span><span class="v" id="i-a1">—</span></div>
    <div class="row"><span class="k">|a| mag</span><span class="v" id="i-mg">—</span></div>
    <div style="font-size:9px;color:#555566;letter-spacing:.06em;margin:3px 0">── Raw Policy ──</div>
    <div class="row"><span class="k" style="color:#ff9900">raw a[0] right</span><span class="v" id="i-ra0" style="color:#ff9900">—</span></div>
    <div class="row"><span class="k" style="color:#ff9900">raw a[1] fwd</span><span class="v" id="i-ra1" style="color:#ff9900">—</span></div>
    <div class="row"><span class="k">|raw a| mag</span><span class="v" id="i-rmg">—</span></div>
    <hr>
    <div class="row"><span class="k">SPOT YAW</span><span class="v" id="i-yw">—</span></div>
    <div class="row"><span class="k">WORLD α</span><span class="v" id="i-al">—</span></div>
    <hr>
    <div style="font-size:9px;color:var(--dim);letter-spacing:.06em;text-transform:uppercase;margin:3px 0">Frame Debug</div>
    <div class="row"><span class="k" style="color:#aaddff">pos_vis x/y m</span><span class="v" id="i-dbpv" style="color:#aaddff">—</span></div>
    <div class="row"><span class="k" style="color:#aaddff">vel_vis x/y m/s</span><span class="v" id="i-dbvv" style="color:#aaddff">—</span></div>
    <div class="row"><span class="k" style="color:#aaffaa">vel_body fwd/lat</span><span class="v" id="i-dbvb" style="color:#aaffaa">—</span></div>
    <div class="row"><span class="k" style="color:#ffddaa">sim_pos x/y</span><span class="v" id="i-dbsp" style="color:#ffddaa">—</span></div>
    <div class="row"><span class="k" style="color:#ffddaa">sim_vel x/y</span><span class="v" id="i-dbsv" style="color:#ffddaa">—</span></div>
    <div class="row"><span class="k" style="color:#ffaaff">cmd vx/vy m/s</span><span class="v" id="i-dbcv" style="color:#ffaaff">—</span></div>
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
<div id="vel-area"><canvas id="vcv"></canvas></div>
<div class="rsz-v" id="rsz5"></div>
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
const CN={0:'open_space',1:'approach_bridge',2:'on_bridge',3:'exit_bridge'};
const TN={0:'Road',1:'Grass',2:'Sidewalk'};
const TC={0:'#ffaa44',1:'#44ff88',2:'#aaaaff'};
const RM={2:1,3:1,5:2,6:2,7:2,8:2,9:2,'-1':3,4:3,11:1,0:0,1:0};
const DARK={lidar:'#00cc44',topk:'#ff6600',act:'#00cfff',bear:'#ffd700',
            head:'#cc44ff',warn:'#ff4444',grid:'#30363d',dim:'#8b949e',
            circ:'#58a6ff',bg:'#161b22',trail:'#2860cc',
            cloudAll:'rgba(42,42,58,0.7)',sliceZ:'#ffcc00',sliceI:'#ff44cc'};
const LIGHT={lidar:'#006e1f',topk:'#cc3d00',act:'#0050bb',bear:'#8a6000',
             head:'#6e0099',warn:'#bb0000',grid:'#b0bac6',dim:'#4a5568',
             circ:'#1a4db5',bg:'#f0f4f8',trail:'#2244aa',
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
const vcv=document.getElementById('vcv'), vctx=vcv.getContext('2d');

let _xcorrEma={vx:null,vy:null};
const _XCORR_ALPHA=0.2;
function _arrMean(a){return a.reduce((s,v)=>s+v,0)/a.length;}
function _arrStd(a){const m=_arrMean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length);}
function _xcorrLagMs(times,cmd,rep){
  const n=times.length;
  if(n<20)return null;
  const cs=_arrStd(cmd),rs=_arrStd(rep);
  if(cs<0.02||rs<0.02)return null;
  const dt=(times[n-1]-times[0])/(n-1);
  const cm=_arrMean(cmd),rm=_arrMean(rep);
  let best=-Infinity,bestLag=0;
  for(let lag=0;lag<=Math.min(n-1,Math.round(3.0/dt));lag++){
    let s=0;
    for(let i=lag;i<n;i++) s+=(rep[i]-rm)*(cmd[i-lag]-cm);
    if(s>best){best=s;bestLag=lag;}
  }
  return bestLag*dt*1000;
}

function _stdArr(arr){
  if(arr.length<2)return 0;
  const m=arr.reduce((a,b)=>a+b,0)/arr.length;
  return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length);
}
function drawMiniChart(ctx2,x0,y0,w,h,times,cmdArr,repArr,title,lagOverride=undefined){
  ctx2.fillStyle='#161b22';ctx2.fillRect(x0,y0,w,h);
  ctx2.strokeStyle='#30363d';ctx2.lineWidth=0.7;ctx2.strokeRect(x0,y0,w,h);
  const pad={l:34,r:6,t:16,b:26};
  const pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  if(pw<10||ph<10)return;
  ctx2.fillStyle='#8b949e';ctx2.font='9px monospace';ctx2.textAlign='center';
  ctx2.fillText(title,x0+w/2,y0+11);
  const n=times.length;
  if(n<2){ctx2.fillText('waiting...',x0+w/2,y0+h/2);return;}
  const t0=times[0],tSpan=Math.max(times[n-1]-t0,1.0);
  const allV=[...cmdArr,...repArr];
  let vMin=Math.min(...allV),vMax=Math.max(...allV);
  const mg=Math.max((vMax-vMin)*0.15,0.05);vMin-=mg;vMax+=mg;
  const vSpan=vMax-vMin||1;
  const tx=t=>x0+pad.l+(t-t0)/tSpan*pw;
  const ty=v=>y0+pad.t+(1-(v-vMin)/vSpan)*ph;
  // zero line
  const zy=ty(0);
  if(zy>y0+pad.t&&zy<y0+pad.t+ph){
    ctx2.strokeStyle='#30363d';ctx2.lineWidth=0.6;ctx2.setLineDash([3,4]);
    ctx2.beginPath();ctx2.moveTo(x0+pad.l,zy);ctx2.lineTo(x0+pad.l+pw,zy);ctx2.stroke();
    ctx2.setLineDash([]);
  }
  // y labels
  ctx2.fillStyle='#8b949e';ctx2.font='8px monospace';ctx2.textAlign='right';
  ctx2.fillText(vMax.toFixed(2),x0+pad.l-2,y0+pad.t+4);
  ctx2.fillText(vMin.toFixed(2),x0+pad.l-2,y0+pad.t+ph);
  if(zy>y0+pad.t+8&&zy<y0+pad.t+ph-4)ctx2.fillText('0',x0+pad.l-2,zy+3);
  // lines
  function line(arr,col){
    if(!arr.length)return;
    ctx2.strokeStyle=col;ctx2.lineWidth=1.5;ctx2.setLineDash([]);
    ctx2.beginPath();
    arr.forEach((v,i)=>{const px=tx(times[i]),py=ty(v);i===0?ctx2.moveTo(px,py):ctx2.lineTo(px,py);});
    ctx2.stroke();
  }
  line(repArr,'#44aaff');line(cmdArr,'#ff4466');
  // legend
  ctx2.font='8px monospace';ctx2.textAlign='left';
  ctx2.fillStyle='#ff4466';ctx2.fillText('cmd', x0+pad.l+2,y0+pad.t+10);
  ctx2.fillStyle='#44aaff';ctx2.fillText('rep', x0+pad.l+28,y0+pad.t+10);
  // lag estimate — use smoothed value if provided
  const lag=lagOverride!==undefined?lagOverride:_xcorrLagMs(times,cmdArr,repArr);
  ctx2.textAlign='center';ctx2.font='9px monospace';
  if(lag!==null){
    const c=lag<150?'#00cc44':lag<400?'#ffaa00':'#ff4444';
    ctx2.fillStyle=c;
    ctx2.fillText('lag≈'+lag.toFixed(0)+'ms',x0+w/2,y0+h-14);
  }else{
    ctx2.fillStyle='#8b949e';
    ctx2.fillText('(need more signal)',x0+w/2,y0+h-14);
  }
  // variance row
  if(cmdArr.length>=2){
    const sRep=_stdArr(repArr);
    const errArr=repArr.map((v,i)=>v-cmdArr[i]);
    const sErr=_stdArr(errArr);
    ctx2.font='8px monospace';ctx2.textAlign='left';
    ctx2.fillStyle='#44aaff';ctx2.fillText('σ_rep='+sRep.toFixed(3),x0+pad.l+2,y0+h-3);
    ctx2.fillStyle='#ffaa44';ctx2.fillText('σ_err='+sErr.toFixed(3),x0+pad.l+76,y0+h-3);
  }
}
function _drawCycleHalf(ctx2,x0,y0,w,h,txFn,vals,color,title,countStr){
  ctx2.fillStyle='#161b22';ctx2.fillRect(x0,y0,w,h);
  ctx2.strokeStyle='#30363d';ctx2.lineWidth=0.7;ctx2.strokeRect(x0,y0,w,h);
  const padL=36,padT=14,padB=8,ph=h-padT-padB;
  ctx2.font='8px monospace';ctx2.fillStyle=color;ctx2.textAlign='left';
  ctx2.fillText(title,x0+padL+2,y0+11);
  ctx2.fillStyle='#8b949e';ctx2.textAlign='right';
  ctx2.fillText(countStr,x0+w-4,y0+11);
  if(!vals.length){ctx2.textAlign='center';ctx2.fillText('waiting…',x0+w/2,y0+h/2);return;}
  let vMin=Math.min(...vals),vMax=Math.max(...vals);
  const mg=Math.max((vMax-vMin)*0.15,50);vMin=Math.max(0,vMin-mg);vMax+=mg;
  const vSpan=vMax-vMin||1;
  const ty=v=>y0+padT+(1-(v-vMin)/vSpan)*ph;
  ctx2.font='7px monospace';ctx2.fillStyle='#8b949e';ctx2.textAlign='right';
  ctx2.fillText(vMax.toFixed(0),x0+padL-2,y0+padT+4);
  ctx2.fillText(vMin.toFixed(0),x0+padL-2,y0+padT+ph);
  ctx2.strokeStyle=color;ctx2.lineWidth=1.5;ctx2.setLineDash([]);
  ctx2.beginPath();
  vals.forEach((v,i)=>{i===0?ctx2.moveTo(txFn(i),ty(v)):ctx2.lineTo(txFn(i),ty(v));});
  ctx2.stroke();
  vals.forEach((v,i)=>{ctx2.fillStyle=color;ctx2.beginPath();ctx2.arc(txFn(i),ty(v),3,0,2*Math.PI);ctx2.fill();});
}
function drawCycleChart(ctx2,x0,y0,w,h,cycles){
  if(!cycles||!cycles.length){
    ctx2.fillStyle='#161b22';ctx2.fillRect(x0,y0,w,h);
    ctx2.fillStyle='#8b949e';ctx2.font='9px monospace';ctx2.textAlign='center';
    ctx2.fillText('waiting for cycles…',x0+w/2,y0+h/2);return;
  }
  const gap=3,hTop=Math.floor((h-gap)/2),hBot=h-hTop-gap;
  const n=cycles.length,pw=w-42;
  const delays=cycles.map(c=>c[1]);
  const risePairs=cycles.filter(c=>c[2]!=null);
  const rises=risePairs.map(c=>c[2]);
  const txD=i=>x0+36+(n>1?i/(n-1):0.5)*pw;
  const txR=i=>{const ri=cycles.indexOf(risePairs[i]);return x0+36+(n>1?ri/(n-1):0.5)*pw;};
  _drawCycleHalf(ctx2,x0,y0,w,hTop,txD,delays,'#44aaff','pure delay',n+' cyc');
  _drawCycleHalf(ctx2,x0,y0+hTop+gap,w,hBot,txR,rises,'#00cc44','rise 0←90%',rises.length+'/'+n);
}
function drawInfChart(ctx2,x0,y0,w,h,times,infArr){
  ctx2.fillStyle='#161b22';ctx2.fillRect(x0,y0,w,h);
  ctx2.strokeStyle='#30363d';ctx2.lineWidth=0.7;ctx2.strokeRect(x0,y0,w,h);
  const pad={l:34,r:6,t:16,b:26};
  const pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
  if(pw<10||ph<10)return;
  const n=times.length;
  ctx2.fillStyle='#ff9955';ctx2.font='9px monospace';ctx2.textAlign='center';
  if(n<2||!infArr.length){ctx2.fillStyle='#8b949e';ctx2.fillText('DGPPO inference',x0+w/2,y0+11);ctx2.fillText('waiting...',x0+w/2,y0+h/2);return;}
  const t0=times[times.length-infArr.length],tSpan=Math.max(times[n-1]-t0,1.0);
  let vMin=Math.min(...infArr),vMax=Math.max(...infArr);
  const mg=Math.max((vMax-vMin)*0.20,2);vMin=Math.max(0,vMin-mg);vMax+=mg;
  const vSpan=vMax-vMin||1;
  const tx=t=>x0+pad.l+(t-t0)/tSpan*pw;
  const ty=v=>y0+pad.t+(1-(v-vMin)/vSpan)*ph;
  const infTimes=times.slice(times.length-infArr.length);
  const mean=infArr.reduce((a,b)=>a+b,0)/infArr.length;
  const std=Math.sqrt(infArr.reduce((a,b)=>a+(b-mean)**2,0)/infArr.length);
  const c=mean<20?'#00cc44':mean<50?'#ffaa00':'#ff4444';
  ctx2.font='9px monospace';ctx2.textAlign='center';ctx2.fillStyle=c;
  ctx2.fillText('DGPPO inference  (input lag)',x0+w/2,y0+11);
  ctx2.font='8px monospace';ctx2.fillStyle='#8b949e';ctx2.textAlign='right';
  ctx2.fillText(vMax.toFixed(0),x0+pad.l-2,y0+pad.t+4);
  ctx2.fillText(vMin.toFixed(0),x0+pad.l-2,y0+pad.t+ph);
  ctx2.strokeStyle='#ff9955';ctx2.lineWidth=1.5;ctx2.setLineDash([]);
  ctx2.beginPath();
  infArr.forEach((v,i)=>{const px=tx(infTimes[i]),py=ty(v);i===0?ctx2.moveTo(px,py):ctx2.lineTo(px,py);});
  ctx2.stroke();
  ctx2.fillStyle=c;ctx2.font='9px monospace';ctx2.textAlign='center';
  ctx2.fillText('mean='+mean.toFixed(1)+'ms  σ='+std.toFixed(1)+'ms',x0+w/2,y0+h-14);
}
function drawVelChart(d){
  const W=vcv.width,H=vcv.height;
  if(W<20||H<20)return;
  vctx.fillStyle=C.bg;vctx.fillRect(0,0,W,H);
  const gap=4,quarter=Math.floor((W-gap*3)/4);
  const times=d.vel_times||[];
  drawMiniChart(vctx,0,0,quarter,H,times,d.cmd_vx_hist||[],d.rep_vx_hist||[],'vx (vision frame)',_xcorrEma.vx);
  drawMiniChart(vctx,quarter+gap,0,quarter,H,times,d.cmd_vy_hist||[],d.rep_vy_hist||[],'vy (vision frame)',_xcorrEma.vy);
  drawInfChart(vctx,(quarter+gap)*2,0,quarter,H,times,d.inf_ms_hist||[]);
  drawCycleChart(vctx,(quarter+gap)*3,0,W-(quarter+gap)*3,H,d.cycle_metrics||[]);
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
  // display x=−r·sin(a) (right=−body_y), display y=r·cos(a) (fwd=body_x)
  return[cx - r*Math.sin(a)*sc, cy - r*Math.cos(a)*sc];
}
function polarPt(r,a,cx,cy,sc){
  // Standard polar: x positive = RIGHT, y positive = UP (canvas inverted)
  return[cx + r*Math.cos(a)*sc, cy - r*Math.sin(a)*sc];
}
function drawArrow(ang,sc,cx,cy,col,lw,alpha=1){
  const[ex,ey]=polarPt(1,ang,cx,cy,sc);
  ctx.save();ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.globalAlpha=alpha;
  ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();
  const hl=Math.max(10,12*lw/4),dx=ex-cx,dy=ey-cy,L=Math.hypot(dx,dy)||1;
  const ux=dx/L,uy=dy/L;
  ctx.fillStyle=col;ctx.beginPath();
  ctx.moveTo(ex,ey);
  ctx.lineTo(ex-hl*(ux+0.4*uy),ey-hl*(uy-0.4*ux));
  ctx.lineTo(ex-hl*(ux-0.4*uy),ey-hl*(uy+0.4*ux));
  ctx.closePath();ctx.fill();
  ctx.globalAlpha=1;ctx.restore();
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

  // Arrows: body-frame display.
  // Heading: robot forward is always UP — constant, never rotates.
  drawArrow(Math.PI/2, sc, cx, cy, C.head, 3.5);
  if(d.bearing_rad!=null){
    const _α=d.world_alpha_rad||0.0, _ψ=d.spot_yaw||0.0;
    const b_world=d.bearing_rad - _α;       // apply world offset
    const b_body=b_world - _ψ;              // apply spot yaw
    const a_disp=b_body + Math.PI/2;        // rotate so 0→UP in drawArrow convention
    drawArrow(b_world + Math.PI/2, sc, cx, cy, C.bear, 1.5, 0.45); // bear - α (dashed dim)
    drawArrow(b_body  + Math.PI/2, sc, cx, cy, C.bear, 1.5, 0.70); // bear - α - ψ (dotted)
    drawArrow(a_disp,              sc, cx, cy, C.bear, 3.5);        // final
  }

  // Reported velocity: body frame fwd=sd[4], lat=sd[5] positive-left → right=-lat
  if(d.state_debug&&d.state_debug.length>=6){
    const rvFwd=d.state_debug[4], rvRight=-d.state_debug[5];
    const rvMag=Math.hypot(rvFwd,rvRight);
    if(rvMag>0.02) drawArrow(Math.atan2(rvFwd,rvRight),sc,cx,cy,'#44aaff',3.0);
  }

  // Pre-rotation policy action (dashed via alpha trick): sd[12]=right, sd[13]=fwd
  if(d.state_debug&&d.state_debug.length>=14){
    const ra0=d.state_debug[12],ra1=d.state_debug[13];
    const raMag=Math.hypot(ra0,ra1);
    if(raMag>0.02) drawArrow(Math.atan2(ra1,ra0),sc,cx,cy,'#ff9900',3.0,0.45);
  }
  // Pre-rotation direction in body frame: rotate C_ACTION back by rot_deg so the gap
  // between this and cyan equals exactly action_rotation_deg, both in body frame.
  if(d.has_action&&d.state_debug&&d.state_debug.length>=18&&d.state_debug[17]!==0){
    const rotDeg=d.state_debug[17];
    const thetaBack=-rotDeg*Math.PI/180;
    const cb=Math.cos(thetaBack),sb=Math.sin(thetaBack);
    const[a0,a1]=d.action;
    const preRight=cb*a0+sb*a1, preFwd=-sb*a0+cb*a1;
    const preMag=Math.hypot(preRight,preFwd);
    if(preMag>0.02) drawArrow(Math.atan2(preFwd,preRight),sc,cx,cy,'#ff4400',4.0);
  }
  // Action (clipped): atan2(fwd, right) already gives FWD=UP — no offset needed
  if(d.has_action){
    const[a0,a1]=d.action,mag=Math.hypot(a0,a1);
    if(mag>0.02) drawArrow(Math.atan2(a1,a0),sc,cx,cy,C.act,4.5);
    else{ctx.fillStyle=C.warn;ctx.beginPath();ctx.arc(cx,cy,12,0,2*Math.PI);ctx.fill();}
  }else{
    ctx.fillStyle=C.dim;ctx.font='13px monospace';ctx.textAlign='center';
    ctx.fillText('waiting /dgppo_action',cx,cy);
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
    [C.act,            'Action cmd (body frame, post-rotation)'],
    ['#ff4400',        'Pre-rotation dir (body frame, action rotated back)'],
    ['#ff9900',        'Raw policy sim-frame output (dim, no-rotation ref)'],
    ['#44aaff',        'Reported vel (unit vec)'],
    [C.bear,           'Plan bearing'],
    [C.head,           'Spot heading'],
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
  // ── Sim delay summary ──
  const ih=d.inf_ms_hist||[];
  const avgIn=ih.length>=2?ih.reduce((a,b)=>a+b,0)/ih.length:null;
  const avgOut=_xcorrEma.vx;
  function fmsSim(ms){return ms!==null?ms.toFixed(0)+' ms':'low signal';}
  $t('i-sim-in', fmsSim(avgIn),  avgIn!==null?'#ffffff':'#8b949e');
  $t('i-sim-out',fmsSim(avgOut), avgOut!==null?'#ffffff':'#8b949e');
  const tot=(avgIn!==null&&avgOut!==null)?avgIn+avgOut:null;
  $t('i-sim-tot',fmsSim(tot), tot!==null?'#ffffff':'#8b949e');

  // ── Latency section ──
  if(d.state_debug&&d.state_debug.length>=12){
    const sd=d.state_debug;
    const f3=v=>(v>=0?'+':'')+v.toFixed(3);
    const cvx=sd[10],cvy=sd[11],rvx=sd[2],rvy=sd[3];
    $t('i-cvx',f3(cvx)+' / '+f3(cvy));
    $t('i-rvx',f3(rvx)+' / '+f3(rvy));
    const evx=cvx-rvx,evy=cvy-rvy;
    $t('i-evx',f3(evx)+' / '+f3(evy));
  }
  const vt=d.vel_times||[],cvxH=d.cmd_vx_hist||[],rvxH=d.rep_vx_hist||[];
  const cvyH=d.cmd_vy_hist||[],rvyH=d.rep_vy_hist||[];
  function showLag(elId,cmd,rep,eKey){
    const raw=_xcorrLagMs(vt,cmd,rep);
    if(raw!==null) _xcorrEma[eKey]=_xcorrEma[eKey]===null?raw:(1-_XCORR_ALPHA)*_xcorrEma[eKey]+_XCORR_ALPHA*raw;
    const lag=_xcorrEma[eKey];
    const el=document.getElementById(elId);
    if(!el)return;
    if(lag!==null){
      const c=lag<150?'#00cc44':lag<400?'#ffaa00':'#ff4444';
      el.textContent=lag.toFixed(0)+' ms';el.style.color=c;
    }else{el.textContent='low signal';el.style.color='#8b949e';}
  }
  showLag('i-lgvx',cvxH,rvxH,'vx');
  showLag('i-lgvy',cvyH,rvyH,'vy');
  if(cvxH.length>=2){
    const sRepVx=_stdArr(rvxH),sRepVy=_stdArr(rvyH);
    const sErrVx=_stdArr(rvxH.map((v,i)=>v-cvxH[i]));
    const sErrVy=_stdArr(rvyH.map((v,i)=>v-cvyH[i]));
    $t('i-srv',sRepVx.toFixed(3)+' / '+sRepVy.toFixed(3));
    $t('i-sev',sErrVx.toFixed(3)+' / '+sErrVy.toFixed(3));
  }
  function showStep(elId,ms,lo,hi){
    const el=document.getElementById(elId);if(!el)return;
    if(ms!=null){
      const c=ms<lo?'#00cc44':ms<hi?'#ffaa00':'#ff4444';
      el.textContent=ms.toFixed(0)+' ms';el.style.color=c;
    }else{el.textContent='—';el.style.color='#8b949e';}
  }
  showStep('i-dly',d.step_delay_ms??null,300,600);
  showStep('i-rse',d.step_rise_ms??null,500,900);
  if(d.state_debug&&d.state_debug.length>=15){
    const inf=d.state_debug[14];
    const ci=inf<20?'#00cc44':inf<50?'#ffaa00':'#ff4444';
    $t('i-inf',inf.toFixed(1)+' ms',ci);
    const ih=d.inf_ms_hist||[];
    if(ih.length>=2){
      const m=ih.reduce((a,b)=>a+b,0)/ih.length;
      const s=Math.sqrt(ih.reduce((a,b)=>a+(b-m)**2,0)/ih.length);
      $t('i-inf-stat',m.toFixed(1)+' / '+s.toFixed(1)+' ms');
    }
  }
  const tid=d.terrain_id;
  $t('i-ter',TN[tid]||'T'+tid,TC[tid]||'#fff');
  const raw=d.raw_cluster;
  $t('i-cr',raw!=null?String(raw):'—');
  if(raw!=null){const m=RM[raw]??RM[String(raw)]??raw;$t('i-cm',m+' '+(CN[m]||'—'));}
  const ps=d.plan_step,seq=d.plan_sequence||[];
  if(ps<seq.length){
    const s=seq[ps];
    $t('i-ps',(ps+1)+' / '+seq.length,'#ffdd88');
    $t('i-pf',CN[s.start]||String(s.start),'#ff9955');
    $t('i-pt',CN[s.next] ||String(s.next), '#ff9955');
  }else if(seq.length>0){$t('i-ps','COMPLETE','#00cc44');}
  if(d.bearing_rad!=null){
    const _α2=d.world_alpha_rad||0.0, _ψ2=d.spot_yaw||0.0;
    $t('i-br',(d.bearing_rad*180/Math.PI).toFixed(1)+'°');
    $t('i-bw',((d.bearing_rad-_α2)*180/Math.PI).toFixed(1)+'°');
    $t('i-bb',((d.bearing_rad-_α2-_ψ2)*180/Math.PI).toFixed(1)+'°');
  }
  if(d.has_action){
    const[a0,a1]=d.action,mag=Math.hypot(a0,a1);
    $t('i-a0',a0.toFixed(4));$t('i-a1',a1.toFixed(4));$t('i-mg',mag.toFixed(4));
    if(d.bearing_rad!=null&&mag>0.02){
      const diff=((Math.atan2(a1,a0)*180/Math.PI-d.bearing_rad*180/Math.PI+180)%360)-180;
      $t('i-df',(diff>=0?'+':'')+diff.toFixed(1)+'°',
         Math.abs(diff)<30?'#00cc44':Math.abs(diff)<60?'#ffaa00':'#ff4444');
    }
  }
  if(d.spot_yaw!=null)$t('i-yw',(d.spot_yaw*180/Math.PI).toFixed(1)+'°');
  $t('i-al',((d.world_alpha_rad||0)*180/Math.PI).toFixed(1)+'°');
  if(d.state_debug&&d.state_debug.length>=12){
    const sd=d.state_debug;
    const f3=v=>(v>=0?'+':'')+v.toFixed(3), f4=v=>(v>=0?'+':'')+v.toFixed(4);
    $t('i-dbpv',f3(sd[0])+' / '+f3(sd[1]));
    $t('i-dbvv',f3(sd[2])+' / '+f3(sd[3]));
    $t('i-dbvb',f3(sd[4])+' / '+f3(sd[5]));
    $t('i-dbsp',f3(sd[6])+' / '+f3(sd[7]));
    $t('i-dbsv',f4(sd[8])+' / '+f4(sd[9]));
    $t('i-dbcv',f3(sd[10])+' / '+f3(sd[11]));
    if(sd.length>=14){
      const ra0=sd[12],ra1=sd[13];
      $t('i-ra0',(ra0>=0?'+':'')+ra0.toFixed(4));
      $t('i-ra1',(ra1>=0?'+':'')+ra1.toFixed(4));
      $t('i-rmg',Math.hypot(ra0,ra1).toFixed(4));
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
      if(r.ok){const d=await r.json();lastData=d;draw(d);panel(d);updateThree(d);drawVelChart(d);
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
  const va=document.getElementById('vel-area');
  if(va){vcv.width=va.clientWidth-20;vcv.height=va.clientHeight-8;if(lastData)drawVelChart(lastData);}
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
makeSplitter(document.getElementById('rsz4'),document.getElementById('cameras'),document.getElementById('vel-area'),'v');
makeSplitter(document.getElementById('rsz5'),document.getElementById('vel-area'),document.getElementById('sliders'),'v');
window.addEventListener('resize',resize);
fetch('/api/state').then(r=>r.json()).then(d=>{initSliders(d.filter_cfg);resize();loop();}).catch(()=>{resize();loop();});
</script>
</body>
</html>"""


def run_web(state: DebugState, port=WEB_PORT):
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
        a0, a1 = float(snap['action'][0]), float(snap['action'][1])
        current_step, bearing_rad = _bearing_for_step(snap)

        cloud_all = []
        cloud_slice_pass, cloud_slice_fail = [], []
        cloud_3d, cloud_3d_pass, cloud_3d_fail = [], [], []
        rc  = snap.get('raw_cloud')
        cfg_s = snap['filter_cfg']
        if rc is not None and len(rc):
            # _apply_slice_filter returns x already negated for upside-down correction;
            # web JS uses cx+x*sc (no additional negation)
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

            # 3D: density-classify on pre-stride cloud, then stride for transfer
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
            action           = [a0, a1],
            has_action       = snap['has_action'],
            terrain_id       = snap['terrain_id'],
            raw_cluster      = snap['raw_cluster'],
            plan_step        = snap['plan_step'],
            plan_sequence    = snap['plan_sequence'],
            bearing_rad      = bearing_rad,
            world_alpha_rad  = snap.get('world_alpha_rad', 0.0),
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
            state_debug      = snap.get('state_debug'),
            vel_times        = snap.get('vel_times', []),
            cmd_vx_hist      = snap.get('cmd_vx_hist', []),
            cmd_vy_hist      = snap.get('cmd_vy_hist', []),
            rep_vx_hist      = snap.get('rep_vx_hist', []),
            rep_vy_hist      = snap.get('rep_vy_hist', []),
            step_delay_ms    = snap.get('step_delay_ms'),
            step_rise_ms     = snap.get('step_rise_ms'),
            cycle_metrics    = snap.get('cycle_metrics', []),
            inf_ms_hist      = snap.get('inf_ms_hist', []),
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

    print(f"[WEB] http://0.0.0.0:{port}  (remote: http://<robot-ip>:{port})")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan', nargs='?', default='')
    ap.add_argument('--port', type=int, default=WEB_PORT)
    args, _ = ap.parse_known_args()

    plan_path = args.plan or os.path.join(os.path.dirname(__file__), 'plans', 'bridge.json')
    plan_sequence, bearing_map = [], {}
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            d = json.load(f)
        plan_sequence = d.get('plan_sequence', [])
        bearing_map   = d.get('bearing_map', {})
        print(f"Loaded plan: {len(plan_sequence)} steps  [{plan_path}]")
    else:
        print(f"[WARN] Plan not found: {plan_path}")

    filter_cfg = FilterConfig()
    state      = DebugState(plan_sequence, bearing_map, filter_cfg)
    ros_t = threading.Thread(target=_ros_thread, args=(state,), daemon=True)
    ros_t.start()

    run_web(state, port=args.port)


if __name__ == '__main__':
    main()
