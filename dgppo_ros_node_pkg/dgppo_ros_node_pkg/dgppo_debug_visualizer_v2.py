#!/usr/bin/env python3
"""
DGPPO Debug Visualizer v2

Usage:
  python3 dgppo_debug_visualizer_v2.py [plan.json]           # desktop (matplotlib)
  python3 dgppo_debug_visualizer_v2.py [plan.json] --web     # web server only
  python3 dgppo_debug_visualizer_v2.py [plan.json] --both    # desktop + web

Web server publishes filter config to /lidar_filter_config (Float32MultiArray:
  [z_upper, z_lower, z2_upper, z2_lower, max_range, min_range, use_intensity])
so the clustering node can subscribe and apply changes dynamically.

Subscribes to:
  /dgppo_action       Float32MultiArray  [a0, a1]
  /predicted_cluster  Int16
  /current_terrain    Int32
  /dgppo_plan_step    Int32
  /processed_ranges   Float32MultiArray  (72-bin range array from clustering node)
  /spot/odometry      nav_msgs/Odometry  (Spot robot heading)
"""

import sys, os, json, math, threading, argparse
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.collections import LineCollection

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int32, Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from rclpy.qos import qos_profile_sensor_data

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_RANGES = 72
TOP_K      = 8
HISTORY_LEN = 20
WEB_PORT   = 8765

TERRAIN_NAMES  = {0: "Road", 1: "Grass", 2: "Sidewalk"}
CLUSTER_NAMES  = {0: "open_space", 1: "approach_bridge", 2: "on_bridge", 3: "exit_bridge"}
RAW_TO_MAPPED  = {
    **{k: 1 for k in [2, 3]},
    **{k: 2 for k in [5, 6, 7, 8, 9]},
    **{k: 3 for k in [-1, 4]},
    **{k: 0 for k in [0, 1]},
}

# Outdoor-optimized palette: bright saturated hues on near-black background
C_BG        = '#0d1117'
C_PANEL     = '#161b22'
C_GRID      = '#30363d'
C_CIRCLE    = '#58a6ff'   # unit circle
C_TEXT      = '#ffffff'
C_DIM       = '#8b949e'
C_LIDAR     = '#00cc44'   # bright green: lidar beams
C_TOPK      = '#ff6600'   # bright orange: top-k hits
C_ACTION    = '#00cfff'   # bright cyan: action direction
C_BEARING   = '#ffd700'   # gold: plan bearing
C_HEADING   = '#cc44ff'   # purple: spot odom heading
C_TRAIL     = '#3060cc'   # blue trail
C_WARN      = '#ff4444'   # red: no-action / stop
C_CLOUD_ALL   = '#2a2a3a'   # very dim: all raw cloud points
C_CLOUD_SLICE = '#ffcc00'   # yellow: points inside z/intensity slice


# ── Filter config (shared between sliders and ROS publisher) ──────────────────

class FilterConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self.z_upper       = 1.26
        self.z_lower       = 0.56
        self.z2_upper      = 0.0
        self.z2_lower      = 0.0
        self.max_range     = 8.0
        self.min_range     = 0.5
        self.use_intensity = False

    def get(self):
        with self._lock:
            return dict(
                z_upper=self.z_upper, z_lower=self.z_lower,
                z2_upper=self.z2_upper, z2_lower=self.z2_lower,
                max_range=self.max_range, min_range=self.min_range,
                use_intensity=self.use_intensity,
            )

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                if hasattr(self, k):
                    setattr(self, k, v)


# ── Shared state ─────────────────────────────────────────────────────────────

class DebugState:
    def __init__(self, plan_sequence, bearing_map, filter_cfg: FilterConfig):
        self._lock          = threading.Lock()
        self.action         = np.zeros(2)
        self.has_action     = False
        self.raw_cluster    = None
        self.terrain_id     = 1
        self.plan_step      = 0
        self.plan_sequence  = plan_sequence
        self.bearing_map    = bearing_map
        self.spot_yaw       = None
        self.processed_ranges = None
        self.raw_cloud      = None   # (N, 4): x, y, z, intensity — downsampled
        self.filter_cfg     = filter_cfg

    def set_action(self, a0, a1):
        with self._lock:
            self.action[:] = [a0, a1]
            self.has_action = True

    def set_cluster(self, raw):
        with self._lock:
            self.raw_cluster = raw

    def set_terrain(self, tid):
        with self._lock:
            self.terrain_id = tid

    def set_plan_step(self, step):
        with self._lock:
            self.plan_step = step

    def set_spot_yaw(self, yaw):
        with self._lock:
            self.spot_yaw = yaw

    def set_processed_ranges(self, r):
        with self._lock:
            self.processed_ranges = r

    def set_raw_cloud(self, pts):
        with self._lock:
            self.raw_cloud = pts

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
            )


# ── ROS node ─────────────────────────────────────────────────────────────────

class DebugSubscriber(Node):
    def __init__(self, state: DebugState):
        super().__init__('dgppo_debug_visualizer_v2')
        self.state = state
        sub = self.create_subscription
        sub(Float32MultiArray, '/dgppo_action',      self._cb_action,   10)
        sub(Int16,             '/predicted_cluster', self._cb_cluster,  10)
        sub(Int32,             '/current_terrain',   self._cb_terrain,  10)
        sub(Int32,             '/dgppo_plan_step',   self._cb_planstep, 10)
        sub(Float32MultiArray, '/processed_ranges',  self._cb_ranges,    10)
        sub(Float32MultiArray, '/dgppo_spot_yaw',    self._cb_spot_yaw,  10)
        sub(PointCloud2,       '/livox/lidar',       self._cb_cloud,     qos_profile_sensor_data)
        self._cfg_pub = self.create_publisher(Float32MultiArray, '/lidar_filter_config', 10)
        self.create_timer(0.2, self._pub_cfg)

    def _cb_action(self, msg):
        if len(msg.data) >= 2:
            self.state.set_action(msg.data[0], msg.data[1])

    def _cb_cluster(self, msg):
        self.state.set_cluster(msg.data)

    def _cb_terrain(self, msg):
        self.state.set_terrain(msg.data)

    def _cb_planstep(self, msg):
        self.state.set_plan_step(msg.data)

    def _cb_ranges(self, msg):
        if msg.data:
            self.state.set_processed_ranges(np.array(msg.data, dtype=np.float32))

    def _cb_spot_yaw(self, msg):
        if msg.data:
            self.state.set_spot_yaw(float(msg.data[0]))

    def _cb_cloud(self, msg: PointCloud2):
        try:
            fields = {f.name for f in msg.fields}
            want   = ['x', 'y', 'z'] + (['intensity'] if 'intensity' in fields else [])
            pts    = np.array(list(read_points(msg, field_names=want, skip_nans=True)),
                              dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] == 0:
                return
            if pts.shape[1] == 3:                               # no intensity field
                pts = np.hstack([pts, np.zeros((len(pts), 1), dtype=np.float32)])
            # Stride-subsample to max 5000 points (deterministic → no flicker)
            stride = max(1, len(pts) // 5000)
            self.state.set_raw_cloud(pts[::stride])
        except Exception:
            pass

    def _pub_cfg(self):
        cfg = self.state.filter_cfg.get()
        m = Float32MultiArray()
        m.data = [
            cfg['z_upper'], cfg['z_lower'],
            cfg['z2_upper'], cfg['z2_lower'],
            cfg['max_range'], cfg['min_range'],
            float(cfg['use_intensity']),
        ]
        self._cfg_pub.publish(m)


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
    n = len(ranges)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    valid  = np.where(ranges < max_range * 0.999)[0]
    if len(valid) == 0:
        return np.empty((0, 2))
    idx = valid[np.argsort(ranges[valid])[:k]]
    return np.column_stack([ranges[idx] * np.cos(angles[idx]),
                            ranges[idx] * np.sin(angles[idx])])


def _bearing_for_step(snap):
    ps  = snap['plan_step']
    seq = snap['plan_sequence']
    bm  = snap['bearing_map']
    if ps < len(seq):
        step = seq[ps]
        return step, bm.get(f"{step['start']}-{step['next']}")
    return None, None


def _apply_slice_filter(raw_cloud, cfg):
    """Split raw cloud (N,4) into (all_xy, slice_xy) using current filter config.

    all_xy   — every point's XY (stride-limited to ≤1500 for display)
    slice_xy — only points passing z/intensity band + range bounds
    """
    if raw_cloud is None or len(raw_cloud) == 0:
        return None, None

    x, y, z, intensity = raw_cloud[:, 0], raw_cloud[:, 1], raw_cloud[:, 2], raw_cloud[:, 3]
    dist = np.hypot(x, y)

    range_mask = (dist >= cfg['min_range']) & (dist <= cfg['max_range'])

    if cfg['use_intensity']:
        lo, hi     = cfg['z_lower'], cfg['z_upper']
        band_mask  = (intensity >= lo) & (intensity <= hi)
    else:
        abs_z      = np.abs(z)
        band1      = (abs_z >= cfg['z_lower'])  & (abs_z <= cfg['z_upper'])
        band2_on   = cfg['z2_upper'] > cfg['z2_lower']
        band2      = (abs_z >= cfg['z2_lower']) & (abs_z <= cfg['z2_upper']) if band2_on \
                     else np.zeros(len(z), dtype=bool)
        band_mask  = band1 | band2

    all_xy   = raw_cloud[:, :2]
    slice_xy = raw_cloud[band_mask & range_mask, :2]
    return all_xy, slice_xy


# ── Desktop visualizer ────────────────────────────────────────────────────────

def _build_figure():
    fig = plt.figure(figsize=(15, 9), facecolor=C_BG)
    fig.suptitle('DGPPO Policy Debugger  v2', color=C_TEXT, fontsize=14,
                 y=0.985, fontweight='bold')

    # Main lidar/arrow frame
    ax = fig.add_axes([0.03, 0.19, 0.57, 0.77])
    ax.set_facecolor(C_PANEL)
    lim = 9.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_color(C_GRID)
    ax.tick_params(colors=C_DIM, labelsize=7)
    ax.axhline(0, color=C_GRID, lw=0.8)
    ax.axvline(0, color=C_GRID, lw=0.8)

    th = np.linspace(0, 2 * np.pi, 300)
    for r in [2, 4, 6, 8]:
        ax.plot(r * np.cos(th), r * np.sin(th), color=C_GRID, lw=0.6, ls='--')
        ax.text(r * 0.72, r * 0.72, f'{r}m', color=C_DIM, fontsize=6, ha='center')
    # Unit circle highlights action arrow tip
    ax.plot(np.cos(th), np.sin(th), color=C_CIRCLE, lw=1.8, ls='-', alpha=0.75)

    for xy, txt in [((0, lim*0.96), 'FWD +y'), ((0, -lim*0.96), 'BCK -y'),
                    ((lim*0.96, 0), 'RT +x'), ((-lim*0.96, 0), 'LT -x')]:
        ax.text(xy[0], xy[1], txt, color=C_DIM, fontsize=8, ha='center', va='center')

    ax.set_title(
        'Processed ranges  |  Orange = closest 8 (policy input)  |  Cyan arrow = action direction (unit)',
        color=C_TEXT, fontsize=9, pad=5)

    leg = [
        mpatches.Patch(color='#555566',   label='raw cloud (all XY)'),
        mpatches.Patch(color=C_CLOUD_SLICE, label='z/intensity slice'),
        mpatches.Patch(color=C_LIDAR,   label=f'processed ranges ({NUM_RANGES} bins)'),
        mpatches.Patch(color=C_TOPK,    label=f'top-{TOP_K} closest → policy'),
        mpatches.Patch(color=C_ACTION,  label='action direction (unit vector)'),
        mpatches.Patch(color=C_BEARING, label='plan bearing'),
        mpatches.Patch(color=C_HEADING, label='spot odom heading'),
    ]
    ax.legend(handles=leg, loc='lower right', facecolor=C_BG,
              edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=8)

    # Info panel
    ax_info = fig.add_axes([0.63, 0.19, 0.35, 0.77])
    ax_info.set_facecolor(C_PANEL)
    ax_info.axis('off')

    # ── Slider row (bottom) ───────────────────────────────────────────────────
    s_h, s_y = 0.028, 0.025
    g = 0.115
    ax_zlo  = fig.add_axes([0.03,        s_y, 0.09, s_h], facecolor=C_PANEL)
    ax_zhi  = fig.add_axes([0.03 + g,    s_y, 0.09, s_h], facecolor=C_PANEL)
    ax_rmin = fig.add_axes([0.03 + g*2,  s_y, 0.09, s_h], facecolor=C_PANEL)
    ax_rmax = fig.add_axes([0.03 + g*3,  s_y, 0.09, s_h], facecolor=C_PANEL)
    ax_mode = fig.add_axes([0.03 + g*4,  s_y - 0.01, 0.10, s_h + 0.04], facecolor=C_PANEL)

    def _sl(axes, lbl, lo, hi, init, color):
        sl = Slider(axes, lbl, lo, hi, valinit=init, color=color)
        sl.label.set_color(C_TEXT); sl.label.set_fontsize(8)
        sl.valtext.set_color(C_TEXT); sl.valtext.set_fontsize(8)
        return sl

    sl_zlo  = _sl(ax_zlo,  'Z min', 0.0,  3.0, 0.56, C_TOPK)
    sl_zhi  = _sl(ax_zhi,  'Z max', 0.0,  3.0, 1.26, C_TOPK)
    sl_rmin = _sl(ax_rmin, 'R min', 0.0,  2.0, 0.50, C_LIDAR)
    sl_rmax = _sl(ax_rmax, 'R max', 1.0, 20.0, 8.00, C_LIDAR)

    rb_mode = RadioButtons(ax_mode, ('Z height', 'Intensity'),
                           activecolor=C_TOPK)
    for lbl in rb_mode.labels:
        lbl.set_color(C_TEXT); lbl.set_fontsize(8)

    sliders = dict(zlo=sl_zlo, zhi=sl_zhi, rmin=sl_rmin, rmax=sl_rmax, mode=rb_mode)
    return fig, ax, ax_info, sliders


def run_desktop(state: DebugState):
    fig, ax, ax_info, sliders = _build_figure()
    history = deque(maxlen=HISTORY_LEN)
    H = {'lidar': [], 'arrow': None, 'bearing': None, 'heading': None,
         'trail': [], 'texts': []}

    def _apply_sliders(_=None):
        state.filter_cfg.set(
            z_lower    = sliders['zlo'].val,
            z_upper    = sliders['zhi'].val,
            min_range  = sliders['rmin'].val,
            max_range  = sliders['rmax'].val,
            use_intensity = (sliders['mode'].value_selected == 'Intensity'),
        )

    for key in ('zlo', 'zhi', 'rmin', 'rmax'):
        sliders[key].on_changed(_apply_sliders)
    sliders['mode'].on_clicked(_apply_sliders)

    def _rm(lst):
        for h in lst:
            try: h.remove()
            except Exception: pass
        lst.clear()

    def _clear():
        _rm(H['lidar'])
        _rm(H['trail'])
        _rm(H['texts'])
        for key in ('arrow', 'bearing', 'heading'):
            if H[key] is not None:
                try: H[key].remove()
                except Exception: pass
            H[key] = None

    def _arrow(xy_tip, color, lw, alpha=1.0):
        return ax.annotate('', xy=xy_tip, xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->',
                                           color=color, lw=lw,
                                           mutation_scale=28, alpha=alpha))

    def update(_frame):
        snap = state.snapshot()
        _clear()

        a0, a1       = float(snap['action'][0]), float(snap['action'][1])
        mag          = math.hypot(a0, a1)
        cfg          = snap['filter_cfg']
        max_range    = cfg['max_range']
        ranges       = snap['processed_ranges']
        spot_yaw     = snap['spot_yaw']
        terrain_id   = snap['terrain_id']
        raw_cluster  = snap['raw_cluster']
        current_step, bearing_rad = _bearing_for_step(snap)
        mapped       = RAW_TO_MAPPED.get(raw_cluster, raw_cluster) \
                       if raw_cluster is not None else None

        # ── Raw cloud layers ──────────────────────────────────────────────
        raw_cloud = snap.get('raw_cloud')
        if raw_cloud is not None:
            all_xy, slice_xy = _apply_slice_filter(raw_cloud, cfg)

            # Layer 1 — all XY (dim gray, stride-limited)
            if all_xy is not None and len(all_xy):
                stride = max(1, len(all_xy) // 1500)
                disp   = all_xy[::stride]
                h = ax.scatter(disp[:, 0], disp[:, 1], s=1.5,
                               color=C_CLOUD_ALL, zorder=1, linewidths=0, alpha=0.7)
                H['lidar'].append(h)

            # Layer 2 — z/intensity slice (yellow)
            if slice_xy is not None and len(slice_xy):
                stride = max(1, len(slice_xy) // 800)
                disp   = slice_xy[::stride]
                h = ax.scatter(disp[:, 0], disp[:, 1], s=4,
                               color=C_CLOUD_SLICE, zorder=2, linewidths=0, alpha=0.85)
                H['lidar'].append(h)

        # ── Processed-range LiDAR ────────────────────────────────────────
        if ranges is not None and len(ranges) > 0:
            n      = len(ranges)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            valid  = ranges < max_range * 0.999
            ex     = np.where(valid, ranges * np.cos(angles), np.nan)
            ey     = np.where(valid, ranges * np.sin(angles), np.nan)

            # Beam lines
            segs = [[[0.0, 0.0], [float(ex[i]), float(ey[i])]]
                    for i in range(n) if valid[i]]
            if segs:
                lc = LineCollection(segs, colors=C_LIDAR, linewidths=1.0,
                                    alpha=0.6, zorder=2)
                ax.add_collection(lc)
                H['lidar'].append(lc)

            # Hit dots
            vx, vy = ex[valid], ey[valid]
            if len(vx):
                h = ax.scatter(vx, vy, s=8, color=C_LIDAR,
                               zorder=3, linewidths=0)
                H['lidar'].append(h)

            # Top-k in orange
            topk = topk_from_ranges(ranges, k=TOP_K, max_range=max_range)
            if len(topk):
                segs_k = [[[0.0, 0.0], [p[0], p[1]]] for p in topk]
                lc_k = LineCollection(segs_k, colors=C_TOPK, linewidths=1.8,
                                      alpha=0.85, zorder=4)
                ax.add_collection(lc_k)
                H['lidar'].append(lc_k)
                h = ax.scatter(topk[:, 0], topk[:, 1], s=90, color=C_TOPK,
                               zorder=5, linewidths=1.0, edgecolors='white')
                H['lidar'].append(h)

        # ── Plan bearing (unit arrow) ─────────────────────────────────────
        if bearing_rad is not None:
            H['bearing'] = _arrow((math.cos(bearing_rad), math.sin(bearing_rad)),
                                  C_BEARING, 3.0)

        # ── Spot odom heading (unit arrow) ────────────────────────────────
        if spot_yaw is not None:
            H['heading'] = _arrow((math.cos(spot_yaw), math.sin(spot_yaw)),
                                  C_HEADING, 3.0)

        # ── Action direction (unit vector) ────────────────────────────────
        if snap['has_action']:
            if mag > 0.02:
                ux, uy = a0 / mag, a1 / mag
                history.append((ux, uy))
                trail = list(history)[:-1]
                for i, (hx, hy) in enumerate(trail):
                    t = i / max(len(trail), 1)
                    h = _arrow((hx, hy), C_TRAIL, 0.5 + 1.5 * t,
                               alpha=0.04 + 0.2 * t)
                    H['trail'].append(h)
                H['arrow'] = _arrow((ux, uy), C_ACTION, 4.5)
            else:
                h, = ax.plot([0], [0], 'o', color=C_WARN, ms=16, zorder=6)
                H['arrow'] = h
                history.append((0.0, 0.0))
        else:
            t = ax.text(0, 0, 'waiting\n/dgppo_action', color=C_DIM,
                        ha='center', va='center', fontsize=11)
            H['texts'].append(t)

        # ── Info panel ────────────────────────────────────────────────────
        tc     = {0: '#ffaa44', 1: '#44ff88', 2: '#aaaaff'}.get(terrain_id, C_TEXT)
        tname  = TERRAIN_NAMES.get(terrain_id, f'T{terrain_id}')
        cname  = CLUSTER_NAMES.get(mapped, f'cls_{mapped}') \
                 if mapped is not None else '—'

        rows = [('TERRAIN', tname, tc), ('', '', '')]
        if raw_cluster is not None:
            rows += [('CLUSTER raw', str(raw_cluster), '#ddddff'),
                     ('CLUSTER mapped', f'{mapped}  {cname}', '#aaaaff')]
        else:
            rows.append(('CLUSTER', 'waiting...', C_DIM))
        rows.append(('', '', ''))

        ps  = snap['plan_step']
        seq = snap['plan_sequence']
        if ps < len(seq) and current_step is not None:
            sc = CLUSTER_NAMES.get(current_step['start'], str(current_step['start']))
            nc = CLUSTER_NAMES.get(current_step['next'],  str(current_step['next']))
            rows += [('PLAN STEP', f"{ps+1} / {len(seq)}", '#ffdd88'),
                     ('FROM', sc, '#ff9955'),
                     ('TO',   nc, '#ff9955')]
            if bearing_rad is not None:
                bd = math.degrees(bearing_rad)
                rows.append(('BEARING', f'{bd:+.1f}°', C_BEARING))
                if snap['has_action'] and mag > 0.02:
                    ad   = math.degrees(math.atan2(a1, a0))
                    diff = (ad - bd + 180) % 360 - 180
                    dc   = C_LIDAR if abs(diff) < 30 else \
                           '#ffaa00' if abs(diff) < 60 else C_WARN
                    rows.append(('ACTION ↔ BEARING', f'{diff:+.1f}°', dc))
        elif ps >= len(seq) and seq:
            rows.append(('PLAN', 'COMPLETE', C_LIDAR))
        else:
            rows.append(('PLAN', 'loading...', C_DIM))
        rows.append(('', '', ''))

        rows += [('a[0] right/+x', f'{a0:+.4f}', C_TEXT),
                 ('a[1]  fwd/+y',  f'{a1:+.4f}', C_TEXT),
                 ('|a| magnitude', f'{mag:.4f}',  C_DIM)]
        if spot_yaw is not None:
            rows += [('', '', ''),
                     ('SPOT YAW', f'{math.degrees(spot_yaw):+.1f}°', C_HEADING)]
        rows += [('', '', ''),
                 ('Z slice', f"[{cfg['z_lower']:.2f}, {cfg['z_upper']:.2f}]", '#bbbbbb'),
                 ('Range',   f"[{cfg['min_range']:.1f}, {cfg['max_range']:.1f}] m", '#bbbbbb'),
                 ('Mode', 'Intensity' if cfg['use_intensity'] else 'Z height', C_TOPK)]

        y, dy = 0.97, 0.057
        for lbl, val, clr in rows:
            if not lbl and not val:
                y -= dy * 0.35
                continue
            t1 = ax_info.text(0.04, y, lbl, transform=ax_info.transAxes,
                              color=C_DIM, fontsize=8.5, va='top', fontweight='bold')
            t2 = ax_info.text(0.96, y, val, transform=ax_info.transAxes,
                              color=clr, fontsize=9.0, va='top', ha='right',
                              fontfamily='monospace')
            H['texts'].extend([t1, t2])
            y -= dy

        fig.canvas.draw_idle()

    ani = FuncAnimation(fig, update, interval=80, cache_frame_data=False)
    plt.show()
    return ani


# ── Web server ────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DGPPO Debugger v2</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--grid:#30363d;--txt:#fff;--dim:#8b949e;
      --lidar:#00cc44;--topk:#ff6600;--act:#00cfff;--bear:#ffd700;
      --head:#cc44ff;--warn:#ff4444;--circ:#58a6ff;}
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
#info{width:250px;background:var(--panel);border-left:1px solid var(--grid);
      overflow-y:auto;padding:10px;font-size:12px}
.row{display:flex;justify-content:space-between;margin:2px 0}
.row .k{color:var(--dim)}.row .v{font-weight:bold}
hr{border:none;border-top:1px solid var(--grid);margin:5px 0}
#sliders{background:var(--panel);border-top:1px solid var(--grid);padding:8px 14px}
#sliders h4{font-size:10px;color:var(--dim);margin-bottom:5px;letter-spacing:.05em}
.sr{display:flex;align-items:center;gap:6px;margin:2px 0;font-size:11px}
.sr label{width:52px;color:var(--dim)}.sr span{width:36px;text-align:right}
input[type=range]{flex:1;accent-color:var(--topk)}
.mbtn{padding:3px 10px;font-size:10px;border:1px solid var(--grid);
      background:var(--bg);color:var(--dim);cursor:pointer;border-radius:3px;font-family:monospace}
.mbtn.on{border-color:var(--topk);color:var(--topk);background:#200d00}
</style>
</head>
<body>
<header>DGPPO Policy Debugger v2 <span id="badge">connecting…</span></header>
<main>
  <div id="cw"><canvas id="cv"></canvas></div>
  <div id="info">
    <div class="row"><span class="k">TERRAIN</span><span class="v" id="i-ter">—</span></div>
    <hr>
    <div class="row"><span class="k">CLUSTER raw</span><span class="v" id="i-cr">—</span></div>
    <div class="row"><span class="k">CLUSTER mapped</span><span class="v" id="i-cm">—</span></div>
    <hr>
    <div class="row"><span class="k">PLAN STEP</span><span class="v" id="i-ps">—</span></div>
    <div class="row"><span class="k">FROM</span><span class="v" id="i-pf">—</span></div>
    <div class="row"><span class="k">TO</span><span class="v" id="i-pt">—</span></div>
    <div class="row"><span class="k">BEARING</span><span class="v" id="i-br">—</span></div>
    <div class="row"><span class="k">ACT↔BEAR</span><span class="v" id="i-df">—</span></div>
    <hr>
    <div class="row"><span class="k">a[0] right</span><span class="v" id="i-a0">—</span></div>
    <div class="row"><span class="k">a[1] fwd</span><span class="v" id="i-a1">—</span></div>
    <div class="row"><span class="k">|a| mag</span><span class="v" id="i-mg">—</span></div>
    <hr>
    <div class="row"><span class="k">SPOT YAW</span><span class="v" id="i-yw">—</span></div>
    <hr>
    <div class="row"><span class="k">Z slice</span><span class="v" id="i-zs">—</span></div>
    <div class="row"><span class="k">Range</span><span class="v" id="i-rg">—</span></div>
    <div class="row"><span class="k">Mode</span><span class="v" id="i-md">—</span></div>
  </div>
</main>
<div id="sliders">
  <h4>LIDAR FILTER  ·  publishes → /lidar_filter_config</h4>
  <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">
    <div>
      <div class="sr"><label>Z min</label>
        <input type="range" id="sl-zlo" min="0" max="3" step="0.01" value="0.56">
        <span id="v-zlo">0.56</span></div>
      <div class="sr"><label>Z max</label>
        <input type="range" id="sl-zhi" min="0" max="3" step="0.01" value="1.26">
        <span id="v-zhi">1.26</span></div>
    </div>
    <div>
      <div class="sr"><label>R min</label>
        <input type="range" id="sl-rmin" min="0" max="2" step="0.05" value="0.5">
        <span id="v-rmin">0.50</span></div>
      <div class="sr"><label>R max</label>
        <input type="range" id="sl-rmax" min="1" max="20" step="0.1" value="8">
        <span id="v-rmax">8.0</span></div>
    </div>
    <div>
      <div style="font-size:10px;color:var(--dim);margin-bottom:4px">Cluster mode</div>
      <button class="mbtn on" id="btn-z" onclick="setMode('z')">Z height</button>
      <button class="mbtn"    id="btn-i" onclick="setMode('intensity')">Intensity</button>
    </div>
  </div>
</div>
<script>
const CN={0:'open_space',1:'approach_bridge',2:'on_bridge',3:'exit_bridge'};
const TN={0:'Road',1:'Grass',2:'Sidewalk'};
const TC={0:'#ffaa44',1:'#44ff88',2:'#aaaaff'};
const RM={2:1,3:1,5:2,6:2,7:2,8:2,9:2,'-1':3,4:3,0:0,1:0};
const C={lidar:'#00cc44',topk:'#ff6600',act:'#00cfff',bear:'#ffd700',
         head:'#cc44ff',warn:'#ff4444',grid:'#30363d',dim:'#8b949e',
         circ:'#58a6ff',bg:'#161b22',trail:'#2860cc',txt:'#fff',
         cloudAll:'#2a2a3a',cloudSlice:'#ffcc00'};
const TOP_K=8;
let useIntensity=false, lastData=null;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');

function setMode(m){
  useIntensity=(m==='intensity');
  document.getElementById('btn-z').className='mbtn'+(useIntensity?'':' on');
  document.getElementById('btn-i').className='mbtn'+(useIntensity?' on':'');
  post();
}
function sliderVals(){
  return{z_lower:+sl('zlo'),z_upper:+sl('zhi'),
         min_range:+sl('rmin'),max_range:+sl('rmax'),
         use_intensity:useIntensity};
}
function sl(id){return document.getElementById('sl-'+id).value}
function post(){
  fetch('/api/set_config',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(sliderVals())});
}
['zlo','zhi','rmin','rmax'].forEach(id=>{
  const el=document.getElementById('sl-'+id);
  const sp=document.getElementById('v-'+id);
  el.addEventListener('input',()=>{sp.textContent=parseFloat(el.value).toFixed(2);post();});
});

function tc(r,a,cx,cy,sc){return[cx+r*Math.cos(a)*sc, cy-r*Math.sin(a)*sc]}

function arrow(ctx,cx,cy,ang,sc,col,lw,alpha){
  const [ex,ey]=tc(1,ang,cx,cy,sc);
  ctx.save();
  ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.globalAlpha=alpha;
  ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();
  const hl=12*lw/4,dx=ex-cx,dy=ey-cy,L=Math.sqrt(dx*dx+dy*dy);
  const ux=dx/L,uy=dy/L;
  ctx.beginPath();
  ctx.moveTo(ex,ey);
  ctx.lineTo(ex-hl*(ux+0.4*uy),ey-hl*(uy-0.4*ux));
  ctx.lineTo(ex-hl*(ux-0.4*uy),ey-hl*(uy+0.4*ux));
  ctx.closePath();ctx.fillStyle=col;ctx.fill();
  ctx.globalAlpha=1;ctx.restore();
}

function draw(d){
  const W=cv.width,H=cv.height,cx=W/2,cy=H/2;
  const maxR=d.filter_cfg.max_range||8;
  const sc=(W/2-22)/maxR;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,H);

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

  // Labels
  ctx.fillStyle=C.dim;ctx.font='11px monospace';ctx.textAlign='center';
  ctx.fillText('FWD',cx,13);ctx.fillText('BCK',cx,H-3);
  ctx.textAlign='left';ctx.fillText('RT',W-22,cy+4);
  ctx.textAlign='right';ctx.fillText('LT',22,cy+4);
  ctx.textAlign='center';

  // Layer 1 — raw cloud all XY (dim gray)
  if(d.cloud_all&&d.cloud_all.length){
    ctx.fillStyle=C.cloudAll;ctx.globalAlpha=0.7;
    d.cloud_all.forEach(([x,y])=>{
      ctx.beginPath();ctx.arc(cx+x*sc,cy-y*sc,1.5,0,2*Math.PI);ctx.fill();
    });
    ctx.globalAlpha=1;
  }
  // Layer 2 — z/intensity slice (yellow)
  if(d.cloud_slice&&d.cloud_slice.length){
    ctx.fillStyle=C.cloudSlice;ctx.globalAlpha=0.8;
    d.cloud_slice.forEach(([x,y])=>{
      ctx.beginPath();ctx.arc(cx+x*sc,cy-y*sc,2.5,0,2*Math.PI);ctx.fill();
    });
    ctx.globalAlpha=1;
  }

  // LiDAR beams from processed_ranges
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
      const[ex,ey]=tc(h.r,h.a,cx,cy,sc);
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();
    });
    ctx.globalAlpha=1;
    // Dots
    ctx.fillStyle=C.lidar;
    hits.forEach(h=>{
      const[ex,ey]=tc(h.r,h.a,cx,cy,sc);
      ctx.beginPath();ctx.arc(ex,ey,3,0,2*Math.PI);ctx.fill();
    });
    // Top-k
    const sorted=[...hits].sort((a,b)=>a.r-b.r).slice(0,TOP_K);
    sorted.forEach(h=>{
      const[ex,ey]=tc(h.r,h.a,cx,cy,sc);
      ctx.strokeStyle=C.topk;ctx.lineWidth=1.8;ctx.globalAlpha=0.85;
      ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ex,ey);ctx.stroke();
      ctx.globalAlpha=1;
      ctx.fillStyle=C.topk;
      ctx.beginPath();ctx.arc(ex,ey,6,0,2*Math.PI);ctx.fill();
      ctx.strokeStyle='#fff';ctx.lineWidth=1;
      ctx.beginPath();ctx.arc(ex,ey,6,0,2*Math.PI);ctx.stroke();
    });
  }

  // Plan bearing
  if(d.bearing_rad!=null) arrow(ctx,cx,cy,d.bearing_rad,sc,C.bear,3.5,0.9);
  // Spot heading
  if(d.spot_yaw!=null)    arrow(ctx,cx,cy,d.spot_yaw,sc,C.head,3.5,0.9);
  // Action (unit vector)
  if(d.has_action){
    const[a0,a1]=d.action,mag=Math.sqrt(a0*a0+a1*a1);
    if(mag>0.02) arrow(ctx,cx,cy,Math.atan2(a1,a0),sc,C.act,4.5,1.0);
    else{ctx.fillStyle=C.warn;ctx.beginPath();ctx.arc(cx,cy,12,0,2*Math.PI);ctx.fill();}
  }else{
    ctx.fillStyle=C.dim;ctx.font='13px monospace';ctx.textAlign='center';
    ctx.fillText('waiting /dgppo_action',cx,cy);
  }
  // Robot dot
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(cx,cy,5,0,2*Math.PI);ctx.fill();
}

function $t(id,v,c){const e=document.getElementById(id);e.textContent=v;if(c)e.style.color=c;}
function panel(d){
  const tid=d.terrain_id;
  $t('i-ter',TN[tid]||'T'+tid,TC[tid]||'#fff');
  const raw=d.raw_cluster;
  $t('i-cr',raw!=null?raw:'—');
  if(raw!=null){const m=RM[raw]??RM[String(raw)]??raw;$t('i-cm',m+' '+(CN[m]||'—'));}
  const ps=d.plan_step,seq=d.plan_sequence||[];
  if(ps<seq.length){
    const s=seq[ps];
    $t('i-ps',(ps+1)+' / '+seq.length);
    $t('i-pf',CN[s.start]||s.start);
    $t('i-pt',CN[s.next]||s.next);
  }else if(seq.length>0){$t('i-ps','COMPLETE','#00cc44');}
  if(d.bearing_rad!=null)$t('i-br',(d.bearing_rad*180/Math.PI).toFixed(1)+'°');
  if(d.has_action){
    const[a0,a1]=d.action,mag=Math.sqrt(a0*a0+a1*a1);
    $t('i-a0',a0.toFixed(4));$t('i-a1',a1.toFixed(4));$t('i-mg',mag.toFixed(4));
    if(d.bearing_rad!=null&&mag>0.02){
      const diff=((Math.atan2(a1,a0)*180/Math.PI-(d.bearing_rad*180/Math.PI)+180)%360)-180;
      $t('i-df',(diff>=0?'+':'')+diff.toFixed(1)+'°',
         Math.abs(diff)<30?'#00cc44':Math.abs(diff)<60?'#ffaa00':'#ff4444');
    }
  }
  if(d.spot_yaw!=null)$t('i-yw',(d.spot_yaw*180/Math.PI).toFixed(1)+'°');
  const cfg=d.filter_cfg||{};
  $t('i-zs','['+((cfg.z_lower||0).toFixed(2))+', '+((cfg.z_upper||0).toFixed(2))+']');
  $t('i-rg','['+((cfg.min_range||0).toFixed(1))+', '+((cfg.max_range||8).toFixed(1))+'] m');
  $t('i-md',cfg.use_intensity?'Intensity':'Z height');
}

async function loop(){
  const badge=document.getElementById('badge');
  while(true){
    try{
      const r=await fetch('/api/state');
      if(r.ok){const d=await r.json();lastData=d;draw(d);panel(d);
               badge.textContent='live';badge.className='live';}
    }catch(e){badge.textContent='disconnected';badge.className='';}
    await new Promise(r=>setTimeout(r,80));
  }
}
function resize(){
  const w=document.getElementById('cw');
  const s=Math.min(w.clientWidth-12,w.clientHeight-12,680);
  cv.width=s;cv.height=s;if(lastData)draw(lastData);
}
window.addEventListener('resize',resize);resize();loop();
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

        # Pre-filter cloud on server; send compact arrays so the browser doesn't parse huge JSON
        cloud_all, cloud_slice = [], []
        rc = snap.get('raw_cloud')
        if rc is not None and len(rc):
            all_xy, slice_xy = _apply_slice_filter(rc, snap['filter_cfg'])
            if all_xy is not None and len(all_xy):
                stride = max(1, len(all_xy) // 600)
                cloud_all = all_xy[::stride].tolist()
            if slice_xy is not None and len(slice_xy):
                stride = max(1, len(slice_xy) // 300)
                cloud_slice = slice_xy[::stride].tolist()

        return jsonify(dict(
            action           = [a0, a1],
            has_action       = snap['has_action'],
            terrain_id       = snap['terrain_id'],
            raw_cluster      = snap['raw_cluster'],
            plan_step        = snap['plan_step'],
            plan_sequence    = snap['plan_sequence'],
            bearing_rad      = bearing_rad,
            spot_yaw         = snap['spot_yaw'],
            processed_ranges = snap['processed_ranges'].tolist()
                               if snap['processed_ranges'] is not None else None,
            cloud_all        = cloud_all,
            cloud_slice      = cloud_slice,
            filter_cfg       = snap['filter_cfg'],
        ))

    @app.route('/api/set_config', methods=['POST'])
    def api_set_config():
        data = freq.get_json(silent=True) or {}
        allowed = {'z_upper', 'z_lower', 'z2_upper', 'z2_lower',
                   'max_range', 'min_range', 'use_intensity'}
        state.filter_cfg.set(**{k: v for k, v in data.items() if k in allowed})
        return jsonify({'ok': True})

    print(f"[WEB] http://0.0.0.0:{port}  (remote: http://<robot-ip>:{port})")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan', nargs='?', default='')
    ap.add_argument('--web',  action='store_true')
    ap.add_argument('--both', action='store_true')
    ap.add_argument('--port', type=int, default=WEB_PORT)
    args = ap.parse_args()

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

    if args.web:
        run_web(state, port=args.port)
    elif args.both:
        web_t = threading.Thread(target=run_web, args=(state, args.port), daemon=True)
        web_t.start()
        _ani = run_desktop(state)
    else:
        _ani = run_desktop(state)


if __name__ == '__main__':
    main()
