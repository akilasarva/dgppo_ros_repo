#!/usr/bin/env python3
"""
DGPPO Debug Visualizer v2

Usage:
  ros2 run dgppo_ros_node_pkg dgppo_debug_visualizer_v2 -- [plan.json]
  ros2 run dgppo_ros_node_pkg dgppo_debug_visualizer_v2 -- [plan.json] --web
  ros2 run dgppo_ros_node_pkg dgppo_debug_visualizer_v2 -- [plan.json] --both

Coordinate conventions (after corrections):
  Display: TOP = robot FORWARD, RIGHT = robot RIGHT, LEFT = robot LEFT
  - Raw cloud: x-flip (upside-down mount) then 90° CW rotation → (x,y) → (y_raw, x_raw)
  - Processed-range beams: same net rotation → (r·sin θ, r·cos θ)
  - Bearing/heading arrows: angle 0 = forward = UP  (stored as raw radians, +π/2 applied at draw)
  - Action arrow: atan2(a1_fwd, a0_right); forward → UP, right → RIGHT — already correct

Z-height mode  → yellow slice; same z-band sent to clustering node → affects actual clustering
Intensity mode → magenta slice; visual-only; clustering node still uses Z-height
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int32, Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from rclpy.qos import qos_profile_sensor_data

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_RANGES  = 72
TOP_K       = 8
HISTORY_LEN = 20
WEB_PORT    = 8765

TERRAIN_NAMES = {0: "Road", 1: "Grass", 2: "Sidewalk"}
CLUSTER_NAMES = {0: "open_space", 1: "approach_bridge", 2: "on_bridge", 3: "exit_bridge"}
RAW_TO_MAPPED = {
    **{k: 1 for k in [2, 3]},
    **{k: 2 for k in [5, 6, 7, 8, 9]},
    **{k: 3 for k in [-1, 4]},
    **{k: 0 for k in [0, 1]},
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
C_CLOUD_ALL     = '#2a2a3a'    # dim: all raw cloud XY
C_CLOUD_SLICE_Z = '#ffcc00'    # yellow: Z-height slice (controls clustering)
C_CLOUD_SLICE_I = '#ff44cc'    # magenta: intensity slice (visual only)


# ── Filter config ─────────────────────────────────────────────────────────────

class FilterConfig:
    def __init__(self):
        self._lock     = threading.Lock()
        self.z_upper   = -0.56
        self.z_lower   = -1.26
        self.z2_upper  = 0.0
        self.z2_lower  = 0.0
        self.max_range = 8.0
        self.min_range = 0.5
        self.use_intensity = False
        # Intensity bounds — visualizer slice only; clustering always uses Z
        self.int_lower = 0.0
        self.int_upper = 500.0

    def get(self):
        with self._lock:
            return dict(
                z_upper=self.z_upper, z_lower=self.z_lower,
                z2_upper=self.z2_upper, z2_lower=self.z2_lower,
                max_range=self.max_range, min_range=self.min_range,
                use_intensity=self.use_intensity,
                int_lower=self.int_lower, int_upper=self.int_upper,
            )

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                if hasattr(self, k):
                    setattr(self, k, v)


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
        sub(Float32MultiArray, '/processed_ranges',  self._cb_ranges,   10)
        sub(Float32MultiArray, '/dgppo_spot_yaw',    self._cb_spot_yaw, 10)
        sub(PointCloud2,       '/livox/lidar',       self._cb_cloud,    qos_profile_sensor_data)
        self._cfg_pub = self.create_publisher(Float32MultiArray, '/lidar_filter_config', 10)
        self.create_timer(0.2, self._pub_cfg)

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

    def _cb_cloud(self, msg: PointCloud2):
        try:
            fields = {f.name for f in msg.fields}
            want   = ['x', 'y', 'z'] + (['intensity'] if 'intensity' in fields else [])
            pts    = np.array(list(read_points(msg, field_names=want, skip_nans=True)),
                              dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] == 0:
                return
            if pts.shape[1] == 3:
                pts = np.hstack([pts, np.zeros((len(pts), 1), dtype=np.float32)])
            stride = max(1, len(pts) // 5000)
            self.state.set_raw_cloud(pts[::stride])
        except Exception:
            pass

    def _pub_cfg(self):
        cfg = self.state.filter_cfg.get()
        m = Float32MultiArray()
        # Matches clustering node's _cb_filter_cfg layout:
        # [z_upper, z_lower, z2_upper, z2_lower, max_range, min_range, use_intensity]
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
    """Return (k,2) XY of closest k bins, rotated 90° CW to align with robot frame."""
    n      = len(ranges)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    valid  = np.where(ranges < max_range * 0.999)[0]
    if len(valid) == 0:
        return np.empty((0, 2))
    idx = valid[np.argsort(ranges[valid])[:k]]
    return np.column_stack([ranges[idx] * np.sin(angles[idx]),
                             ranges[idx] * np.cos(angles[idx])])


def _bearing_for_step(snap):
    ps, seq, bm = snap['plan_step'], snap['plan_sequence'], snap['bearing_map']
    if ps < len(seq):
        step = seq[ps]
        return step, bm.get(f"{step['start']}-{step['next']}")
    return None, None


def _apply_slice_filter(raw_cloud, cfg):
    """Split raw_cloud (N,4) into (all_xy, slice_xy) with x already negated.

    Z mode  → band filter on abs(z); same parameters sent to clustering node.
    Int mode → band filter on intensity; clustering node unaffected.
    The x-negation corrects for upside-down lidar mounting throughout.
    """
    if raw_cloud is None or len(raw_cloud) == 0:
        return None, None

    x, y, z, intensity = (raw_cloud[:, i] for i in range(4))
    dist       = np.hypot(x, y)
    range_mask = (dist >= cfg['min_range']) & (dist <= cfg['max_range'])

    if cfg['use_intensity']:
        band_mask = (intensity >= cfg['int_lower']) & (intensity <= cfg['int_upper'])
    else:
        band1    = (z >= cfg['z_lower'])  & (z <= cfg['z_upper'])
        band2_on = cfg['z2_upper'] > cfg['z2_lower']
        band2    = ((z >= cfg['z2_lower']) & (z <= cfg['z2_upper'])
                    if band2_on else np.zeros(len(z), dtype=bool))
        band_mask = band1 | band2

    # x-flip (upside-down mount) then 90° CW rotation: net result is (y_raw, x_raw)
    xy_flipped = np.column_stack([y, x])
    return xy_flipped, xy_flipped[band_mask & range_mask]


# ── Desktop visualizer ────────────────────────────────────────────────────────

def _style_3d(ax3d):
    ax3d.set_facecolor(C_PANEL)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(C_GRID)
    ax3d.tick_params(colors=C_DIM, labelsize=6)
    ax3d.xaxis.label.set_color(C_DIM); ax3d.xaxis.label.set_fontsize(7)
    ax3d.yaxis.label.set_color(C_DIM); ax3d.yaxis.label.set_fontsize(7)
    ax3d.zaxis.label.set_color(C_DIM); ax3d.zaxis.label.set_fontsize(7)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title('Point Cloud 3D · yellow = z slice', color=C_TEXT, fontsize=8)
    ax3d.view_init(elev=20, azim=-60)

def _build_figure():
    fig = plt.figure(figsize=(22, 9), facecolor=C_BG)
    fig.suptitle('DGPPO Policy Debugger  v2', color=C_TEXT, fontsize=14,
                 y=0.985, fontweight='bold')

    ax = fig.add_axes([0.02, 0.17, 0.37, 0.79])
    ax.set_facecolor(C_PANEL)
    lim = 9.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color(C_GRID)
    ax.tick_params(colors=C_DIM, labelsize=7)
    ax.axhline(0, color=C_GRID, lw=0.8); ax.axvline(0, color=C_GRID, lw=0.8)

    th = np.linspace(0, 2 * np.pi, 300)
    for r in [2, 4, 6, 8]:
        ax.plot(r * np.cos(th), r * np.sin(th), color=C_GRID, lw=0.6, ls='--')
        ax.text(r * 0.72, r * 0.72, f'{r}m', color=C_DIM, fontsize=6, ha='center')
    ax.plot(np.cos(th), np.sin(th), color=C_CIRCLE, lw=1.8, alpha=0.75)

    for (px, py), txt in [((0,  lim*0.96), 'FWD'),  ((0, -lim*0.96), 'BCK'),
                           ((lim*0.96, 0),  'RIGHT'), ((-lim*0.96, 0), 'LEFT')]:
        ax.text(px, py, txt, color=C_DIM, fontsize=8, ha='center', va='center')

    ax.set_title(
        'TOP=FWD · Lidar x-flipped (upside-down mount) · Orange=top-8 policy inputs · Cyan=action',
        color=C_TEXT, fontsize=8.5, pad=5)

    leg = [
        mpatches.Patch(color='#555566',       label='raw cloud (all XY)'),
        mpatches.Patch(color=C_CLOUD_SLICE_Z, label='Z-height slice  → clustering node'),
        mpatches.Patch(color=C_CLOUD_SLICE_I, label='Intensity slice (visual only)'),
        mpatches.Patch(color=C_LIDAR,         label=f'processed ranges ({NUM_RANGES} bins)'),
        mpatches.Patch(color=C_TOPK,          label=f'top-{TOP_K} closest → policy input'),
        mpatches.Patch(color=C_ACTION,        label='action direction (unit vec)'),
        mpatches.Patch(color=C_BEARING,       label='plan bearing  (0=FWD=UP)'),
        mpatches.Patch(color=C_HEADING,       label='spot heading  (0=FWD=UP)'),
    ]
    ax.legend(handles=leg, loc='lower right', facecolor=C_BG,
              edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=7.5)

    ax3d = fig.add_axes([0.42, 0.17, 0.24, 0.79], projection='3d')
    _style_3d(ax3d)

    ax_info = fig.add_axes([0.69, 0.17, 0.29, 0.79])
    ax_info.set_facecolor(C_PANEL); ax_info.axis('off')

    # Slider row — 6 sliders + mode toggle
    s_h, s_y, g = 0.028, 0.020, 0.087
    sl_axes = [fig.add_axes([0.03 + i * g, s_y, 0.075, s_h], facecolor=C_PANEL)
               for i in range(6)]
    ax_mode = fig.add_axes([0.03 + 6 * g, s_y - 0.012, 0.10, s_h + 0.04], facecolor=C_PANEL)

    def _sl(axes, lbl, lo, hi, init, color):
        sl = Slider(axes, lbl, lo, hi, valinit=init, color=color)
        sl.label.set_color(C_TEXT);   sl.label.set_fontsize(7)
        sl.valtext.set_color(C_TEXT); sl.valtext.set_fontsize(7)
        return sl

    sl_zlo  = _sl(sl_axes[0], 'Z min',   -3.0,    1.0,  -1.26, C_CLOUD_SLICE_Z)
    sl_zhi  = _sl(sl_axes[1], 'Z max',   -3.0,    3.0,  -0.56, C_CLOUD_SLICE_Z)
    sl_ilo  = _sl(sl_axes[2], 'Int min', 0.0, 1000.0,    0.0, C_CLOUD_SLICE_I)
    sl_ihi  = _sl(sl_axes[3], 'Int max', 0.0, 1000.0,  500.0, C_CLOUD_SLICE_I)
    sl_rmin = _sl(sl_axes[4], 'R min',   0.0,    2.0,   0.50, C_LIDAR)
    sl_rmax = _sl(sl_axes[5], 'R max',   1.0,   20.0,   8.00, C_LIDAR)

    rb_mode = RadioButtons(ax_mode, ('Z height', 'Intensity'), activecolor=C_TOPK)
    for lbl in rb_mode.labels:
        lbl.set_color(C_TEXT); lbl.set_fontsize(7.5)

    sliders = dict(zlo=sl_zlo, zhi=sl_zhi, ilo=sl_ilo, ihi=sl_ihi,
                   rmin=sl_rmin, rmax=sl_rmax, mode=rb_mode)
    return fig, ax, ax3d, ax_info, sliders


def run_desktop(state: DebugState):
    fig, ax, ax3d, ax_info, sliders = _build_figure()
    history = deque(maxlen=HISTORY_LEN)
    H = {'lidar': [], 'arrow': None, 'bearing': None, 'heading': None,
         'trail': [], 'texts': []}

    def _apply_sliders(_=None):
        state.filter_cfg.set(
            z_lower       = sliders['zlo'].val,
            z_upper       = sliders['zhi'].val,
            int_lower     = sliders['ilo'].val,
            int_upper     = sliders['ihi'].val,
            min_range     = sliders['rmin'].val,
            max_range     = sliders['rmax'].val,
            use_intensity = (sliders['mode'].value_selected == 'Intensity'),
        )

    for key in ('zlo', 'zhi', 'ilo', 'ihi', 'rmin', 'rmax'):
        sliders[key].on_changed(_apply_sliders)
    sliders['mode'].on_clicked(_apply_sliders)

    def _rm(lst):
        for h in lst:
            try: h.remove()
            except Exception: pass
        lst.clear()

    def _clear():
        _rm(H['lidar']); _rm(H['trail']); _rm(H['texts'])
        for k in ('arrow', 'bearing', 'heading'):
            if H[k] is not None:
                try: H[k].remove()
                except Exception: pass
            H[k] = None
        ax3d.cla()
        _style_3d(ax3d)

    def _arrow(xy_tip, color, lw, alpha=1.0):
        return ax.annotate('', xy=xy_tip, xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=color,
                                           lw=lw, mutation_scale=28, alpha=alpha))

    def update(_frame):
        snap = state.snapshot()
        _clear()

        a0, a1      = float(snap['action'][0]), float(snap['action'][1])
        mag         = math.hypot(a0, a1)
        cfg         = snap['filter_cfg']
        max_range   = cfg['max_range']
        ranges      = snap['processed_ranges']
        spot_yaw    = snap['spot_yaw']
        raw_cluster = snap['raw_cluster']
        current_step, bearing_rad = _bearing_for_step(snap)
        mapped = RAW_TO_MAPPED.get(raw_cluster, raw_cluster) \
                 if raw_cluster is not None else None
        slice_color = C_CLOUD_SLICE_I if cfg['use_intensity'] else C_CLOUD_SLICE_Z

        # ── Raw cloud layers (x already negated by _apply_slice_filter) ──
        raw_cloud = snap.get('raw_cloud')
        if raw_cloud is not None:
            all_xy, slice_xy = _apply_slice_filter(raw_cloud, cfg)
            if all_xy is not None and len(all_xy):
                stride = max(1, len(all_xy) // 1500)
                d = all_xy[::stride]
                H['lidar'].append(ax.scatter(d[:, 0], d[:, 1], s=1.5,
                                             color=C_CLOUD_ALL, zorder=1,
                                             linewidths=0, alpha=0.7))
            if slice_xy is not None and len(slice_xy):
                stride = max(1, len(slice_xy) // 800)
                d = slice_xy[::stride]
                H['lidar'].append(ax.scatter(d[:, 0], d[:, 1], s=4,
                                             color=slice_color, zorder=2,
                                             linewidths=0, alpha=0.85))

        # ── 3D point cloud with z-slice planes ───────────────────────────
        if raw_cloud is not None and len(raw_cloud) > 0:
            stride3 = max(1, len(raw_cloud) // 800)
            pts3    = raw_cloud[::stride3]
            x3, y3, z3 = pts3[:, 0], pts3[:, 1], pts3[:, 2]

            in_band  = (z3 >= cfg['z_lower']) & (z3 <= cfg['z_upper'])
            out_band = ~in_band

            if np.any(out_band):
                ax3d.scatter(x3[out_band], y3[out_band], z3[out_band],
                             s=1, c='#2a2a3a', alpha=0.35, linewidths=0, depthshade=False)
            if np.any(in_band):
                sc = C_CLOUD_SLICE_I if cfg['use_intensity'] else C_CLOUD_SLICE_Z
                ax3d.scatter(x3[in_band], y3[in_band], z3[in_band],
                             s=6, c=sc, alpha=0.9, linewidths=0, depthshade=False)

            # Semi-transparent planes marking z_lower and z_upper
            lim3 = cfg['max_range']
            xx3, yy3 = np.meshgrid([-lim3, lim3], [-lim3, lim3])
            plane_col = C_CLOUD_SLICE_I if cfg['use_intensity'] else C_CLOUD_SLICE_Z
            for z_val in (cfg['z_lower'], cfg['z_upper']):
                ax3d.plot_surface(xx3, yy3, np.full_like(xx3, z_val),
                                  color=plane_col, alpha=0.12, linewidth=0)

            # Auto z limits with a small margin
            z_margin = 0.3
            ax3d.set_zlim(float(z3.min()) - z_margin, float(z3.max()) + z_margin)
            ax3d.set_xlim(-lim3, lim3)
            ax3d.set_ylim(-lim3, lim3)
        else:
            ax3d.text2D(0.5, 0.5, 'waiting\nfor cloud', transform=ax3d.transAxes,
                        color=C_DIM, ha='center', va='center', fontsize=10)

        # ── Processed-range beams (90° CW rotation applied) ──────────────
        if ranges is not None and len(ranges) > 0:
            n      = len(ranges)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            valid  = ranges < max_range * 0.999
            ex = np.where(valid, ranges * np.sin(angles), np.nan)
            ey = np.where(valid, ranges * np.cos(angles), np.nan)

            segs = [[[0., 0.], [float(ex[i]), float(ey[i])]] for i in range(n) if valid[i]]
            if segs:
                lc = LineCollection(segs, colors=C_LIDAR, linewidths=1.0, alpha=0.6, zorder=2)
                ax.add_collection(lc); H['lidar'].append(lc)

            vx, vy = ex[valid], ey[valid]
            if len(vx):
                H['lidar'].append(ax.scatter(vx, vy, s=8, color=C_LIDAR,
                                             zorder=3, linewidths=0))

            topk = topk_from_ranges(ranges, k=TOP_K, max_range=max_range)
            if len(topk):
                lc_k = LineCollection([[[0., 0.], [p[0], p[1]]] for p in topk],
                                      colors=C_TOPK, linewidths=1.8, alpha=0.85, zorder=4)
                ax.add_collection(lc_k); H['lidar'].append(lc_k)
                H['lidar'].append(ax.scatter(topk[:, 0], topk[:, 1], s=90,
                                             color=C_TOPK, zorder=5,
                                             linewidths=1.0, edgecolors='white'))

        # ── Arrows: bearing/heading offset by +π/2 so 0 rad = FWD = UP ──
        if bearing_rad is not None:
            a = bearing_rad + math.pi / 2
            H['bearing'] = _arrow((math.cos(a), math.sin(a)), C_BEARING, 3.0)

        if spot_yaw is not None:
            a = spot_yaw + math.pi / 2
            H['heading'] = _arrow((math.cos(a), math.sin(a)), C_HEADING, 3.0)

        # ── Action: atan2(a1_fwd, a0_right) already gives FWD=UP ─────────
        if snap['has_action']:
            if mag > 0.02:
                ux, uy = a0 / mag, a1 / mag
                history.append((ux, uy))
                trail = list(history)[:-1]
                for i, (hx, hy) in enumerate(trail):
                    t = i / max(len(trail), 1)
                    H['trail'].append(_arrow((hx, hy), C_TRAIL,
                                            0.5 + 1.5 * t, alpha=0.04 + 0.2 * t))
                H['arrow'] = _arrow((ux, uy), C_ACTION, 4.5)
            else:
                h, = ax.plot([0], [0], 'o', color=C_WARN, ms=16, zorder=6)
                H['arrow'] = h
                history.append((0., 0.))
        else:
            H['texts'].append(ax.text(0, 0, 'waiting\n/dgppo_action',
                                      color=C_DIM, ha='center', va='center', fontsize=11))

        # ── Info panel ────────────────────────────────────────────────────
        tid   = snap['terrain_id']
        tc    = {0: '#ffaa44', 1: '#44ff88', 2: '#aaaaff'}.get(tid, C_TEXT)
        cname = CLUSTER_NAMES.get(mapped, f'cls_{mapped}') if mapped is not None else '—'

        rows = [('TERRAIN', TERRAIN_NAMES.get(tid, f'T{tid}'), tc), ('', '', '')]
        if raw_cluster is not None:
            rows += [('CLUSTER raw',    str(raw_cluster),          '#ddddff'),
                     ('CLUSTER mapped', f'{mapped}  {cname}',      '#aaaaff')]
        else:
            rows.append(('CLUSTER', 'waiting...', C_DIM))
        rows.append(('', '', ''))

        ps, seq = snap['plan_step'], snap['plan_sequence']
        if ps < len(seq) and current_step is not None:
            sc = CLUSTER_NAMES.get(current_step['start'], str(current_step['start']))
            nc = CLUSTER_NAMES.get(current_step['next'],  str(current_step['next']))
            rows += [('PLAN STEP', f"{ps+1} / {len(seq)}", '#ffdd88'),
                     ('FROM', sc, '#ff9955'), ('TO', nc, '#ff9955')]
            if bearing_rad is not None:
                bd = math.degrees(bearing_rad)
                rows.append(('BEARING (raw)', f'{bd:+.1f}°', C_BEARING))
                if snap['has_action'] and mag > 0.02:
                    ad   = math.degrees(math.atan2(a1, a0))
                    diff = (ad - bd + 180) % 360 - 180
                    dc   = C_LIDAR if abs(diff) < 30 else \
                           '#ffaa00' if abs(diff) < 60 else C_WARN
                    rows.append(('ACT ↔ BEAR', f'{diff:+.1f}°', dc))
        elif ps >= len(seq) and seq:
            rows.append(('PLAN', 'COMPLETE', C_LIDAR))
        else:
            rows.append(('PLAN', 'loading...', C_DIM))
        rows.append(('', '', ''))

        rows += [('a[0] right', f'{a0:+.4f}', C_TEXT),
                 ('a[1] fwd',   f'{a1:+.4f}', C_TEXT),
                 ('|a| mag',    f'{mag:.4f}',  C_DIM)]
        if spot_yaw is not None:
            rows += [('', '', ''), ('SPOT YAW', f'{math.degrees(spot_yaw):+.1f}°', C_HEADING)]
        rows.append(('', '', ''))

        mode_lbl = 'Intensity (visual only)' if cfg['use_intensity'] else 'Z height → clustering'
        rows += [('Mode',     mode_lbl,                                               C_TOPK),
                 ('Z slice',  f"[{cfg['z_lower']:.2f}, {cfg['z_upper']:.2f}]",       C_CLOUD_SLICE_Z),
                 ('Int slice',f"[{cfg['int_lower']:.0f}, {cfg['int_upper']:.0f}]",   C_CLOUD_SLICE_I),
                 ('Range',    f"[{cfg['min_range']:.1f}, {cfg['max_range']:.1f}] m", '#bbbbbb')]

        y, dy = 0.97, 0.057
        for lbl, val, clr in rows:
            if not lbl and not val:
                y -= dy * 0.35; continue
            t1 = ax_info.text(0.04, y, lbl, transform=ax_info.transAxes,
                              color=C_DIM, fontsize=8.5, va='top', fontweight='bold')
            t2 = ax_info.text(0.96, y, val, transform=ax_info.transAxes,
                              color=clr, fontsize=9.0, va='top', ha='right',
                              fontfamily='monospace')
            H['texts'].extend([t1, t2]); y -= dy

        fig.canvas.draw_idle()

    ani = FuncAnimation(fig, update, interval=80, cache_frame_data=False)
    plt.show()
    return ani


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
#elev-wrap{width:190px;background:var(--panel);border-left:1px solid var(--grid);
           display:flex;flex-direction:column;align-items:center;padding:6px 4px}
#elev-title{font-size:9px;color:var(--dim);text-align:center;margin-top:4px;
            letter-spacing:.04em;text-transform:uppercase}
</style>
</head>
<body>
<header>DGPPO Policy Debugger v2
  <span id="badge">connecting…</span>
  <span style="font-size:10px;color:var(--dim)">TOP=FWD · Lidar x-flipped · Arrows 0=FWD=UP</span>
</header>
<main>
  <div id="cw"><canvas id="cv"></canvas></div>
  <div id="elev-wrap">
    <canvas id="elev-cv"></canvas>
    <div id="elev-title">Point Cloud Elevation<br>— range vs Z height —</div>
  </div>
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
    <div class="row"><span class="k">Mode</span><span class="v" id="i-md">—</span></div>
    <div class="row">
      <span class="k" style="color:var(--sliceZ)">Z slice (→ cluster)</span>
      <span class="v" id="i-zs" style="color:var(--sliceZ)">—</span>
    </div>
    <div class="row">
      <span class="k" style="color:var(--sliceI)">Int slice (visual)</span>
      <span class="v" id="i-is" style="color:var(--sliceI)">—</span>
    </div>
    <div class="row"><span class="k">Range</span><span class="v" id="i-rg">—</span></div>
  </div>
</main>
<div id="sliders">
  <h4>LIDAR FILTER  ·  publishes → /lidar_filter_config every 200 ms</h4>
  <div style="display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start">
    <div>
      <div class="grp" style="color:var(--sliceZ)">Z height  (→ clustering node)</div>
      <div class="sr"><label>Z min</label>
        <input class="z" type="range" id="sl-zlo" min="-3" max="1" step="0.01" value="-1.26">
        <span id="v-zlo">-1.26</span></div>
      <div class="sr"><label>Z max</label>
        <input class="z" type="range" id="sl-zhi" min="-3" max="3" step="0.01" value="-0.56">
        <span id="v-zhi">-0.56</span></div>
    </div>
    <div>
      <div class="grp" style="color:var(--sliceI)">Intensity  (visual only — cluster uses Z)</div>
      <div class="sr"><label>Int min</label>
        <input class="i" type="range" id="sl-ilo" min="0" max="1000" step="1" value="0">
        <span id="v-ilo">0</span></div>
      <div class="sr"><label>Int max</label>
        <input class="i" type="range" id="sl-ihi" min="0" max="1000" step="1" value="500">
        <span id="v-ihi">500</span></div>
    </div>
    <div>
      <div class="grp" style="color:var(--lidar)">Range</div>
      <div class="sr"><label>R min</label>
        <input class="r" type="range" id="sl-rmin" min="0" max="2" step="0.05" value="0.5">
        <span id="v-rmin">0.50</span></div>
      <div class="sr"><label>R max</label>
        <input class="r" type="range" id="sl-rmax" min="1" max="20" step="0.1" value="8">
        <span id="v-rmax">8.0</span></div>
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
const RM={2:1,3:1,5:2,6:2,7:2,8:2,9:2,'-1':3,4:3,0:0,1:0};
const C={lidar:'#00cc44',topk:'#ff6600',act:'#00cfff',bear:'#ffd700',
         head:'#cc44ff',warn:'#ff4444',grid:'#30363d',dim:'#8b949e',
         circ:'#58a6ff',bg:'#161b22',trail:'#2860cc',
         cloudAll:'rgba(42,42,58,0.7)',sliceZ:'#ffcc00',sliceI:'#ff44cc'};
const TOP_K=8;
let useIntensity=false, lastData=null;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');

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
  const el=document.getElementById('sl-'+id);
  const sp=document.getElementById('v-'+id);
  el.addEventListener('input',()=>{sp.textContent=parseFloat(el.value).toFixed(2);post();});
});

/* Lidar beams: 90° CW rotation applied (sin/cos swapped). Bearing/heading arrows: +π/2 so 0=FWD=UP. */
function lidarPt(r,a,cx,cy,sc){
  // 90° CW rotation (x-flip + rotate): display x=r·sin(a), display y=r·cos(a)
  return[cx + r*Math.sin(a)*sc, cy - r*Math.cos(a)*sc];
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

  // Layer 1: raw cloud all (pre-rotated 90° CW server-side, use cx+x*sc)
  if(d.cloud_all&&d.cloud_all.length){
    ctx.fillStyle=C.cloudAll;
    d.cloud_all.forEach(([x,y])=>{
      ctx.beginPath();ctx.arc(cx+x*sc,cy-y*sc,1.5,0,2*Math.PI);ctx.fill();
    });
  }

  // Layer 2: slice — Z=yellow, Intensity=magenta (x pre-negated server-side)
  const slCol=useIntensity?C.sliceI:C.sliceZ;
  if(d.cloud_slice&&d.cloud_slice.length){
    ctx.fillStyle=slCol;ctx.globalAlpha=0.85;
    d.cloud_slice.forEach(([x,y])=>{
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

  // Arrows: bearing and heading add π/2 so raw 0 = FWD = UP
  if(d.bearing_rad!=null) drawArrow(d.bearing_rad+Math.PI/2,sc,cx,cy,C.bear,3.5);
  if(d.spot_yaw!=null)    drawArrow(d.spot_yaw+Math.PI/2,  sc,cx,cy,C.head,3.5);

  // Action: atan2(fwd, right) already gives FWD=UP — no offset needed
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
}

function drawElev(d){
  const ec=document.getElementById('elev-cv');
  if(!ec)return;
  const W=ec.width,H=ec.height,pad=22;
  const e2=ec.getContext('2d');
  const cfg=d.filter_cfg||{};
  const maxR=cfg.max_range||8;
  const zLo=cfg.z_lower??-1.26, zHi=cfg.z_upper??-0.56;

  // Z axis range: span slice band plus margin
  const zSpan=Math.max(Math.abs(zHi-zLo),0.5);
  const zMin=zLo-zSpan*0.8, zMax=zHi+zSpan*0.8;

  function toEc(range,z){
    const ex=pad+(range/maxR)*(W-2*pad);
    const ey=H-pad-((z-zMin)/(zMax-zMin))*(H-2*pad);
    return[ex,ey];
  }

  // Background
  e2.fillStyle='#161b22';e2.fillRect(0,0,W,H);

  // Z-slice band fill
  const[,y1]=toEc(0,zHi);const[,y2]=toEc(0,zLo);
  const slCol=useIntensity?C.sliceI:C.sliceZ;
  e2.fillStyle=useIntensity?'rgba(255,68,204,0.12)':'rgba(255,204,0,0.12)';
  e2.fillRect(pad,y1,W-2*pad,y2-y1);

  // Z-slice boundary lines
  e2.strokeStyle=slCol;e2.lineWidth=1.5;e2.setLineDash([4,3]);
  [zLo,zHi].forEach(z=>{
    const[,ey]=toEc(0,z);
    e2.beginPath();e2.moveTo(pad,ey);e2.lineTo(W-pad,ey);e2.stroke();
  });
  e2.setLineDash([]);

  // Zero Z line
  if(zMin<0&&zMax>0){
    const[,ey0]=toEc(0,0);
    e2.strokeStyle=C.grid;e2.lineWidth=0.7;
    e2.beginPath();e2.moveTo(pad,ey0);e2.lineTo(W-pad,ey0);e2.stroke();
  }

  // Cloud points
  if(d.cloud_elev&&d.cloud_elev.length){
    d.cloud_elev.forEach(([range,z])=>{
      const inBand=z>=zLo&&z<=zHi;
      const[ex,ey]=toEc(range,z);
      e2.fillStyle=inBand?slCol:'rgba(42,42,58,0.8)';
      e2.beginPath();e2.arc(ex,ey,inBand?2.5:1.2,0,2*Math.PI);e2.fill();
    });
  }

  // Axes labels
  e2.fillStyle=C.dim;e2.font='9px monospace';
  e2.textAlign='center';
  e2.fillText('0',pad,H-4);e2.fillText(maxR.toFixed(0)+'m',W-pad,H-4);
  e2.textAlign='right';
  e2.fillText(zMax.toFixed(1),pad-2,(pad+6));
  e2.fillText(zMin.toFixed(1),pad-2,H-pad+4);
  // Slice labels
  e2.fillStyle=slCol;e2.textAlign='left';
  const[,ly1]=toEc(0,zHi);const[,ly2]=toEc(0,zLo);
  e2.fillText('z='+zHi.toFixed(2),W-pad-2,ly1-3);
  e2.fillText('z='+zLo.toFixed(2),W-pad-2,ly2+10);
}

function $t(id,v,c){const e=document.getElementById(id);if(!e)return;e.textContent=v;if(c)e.style.color=c;}
function panel(d){
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
  if(d.bearing_rad!=null)$t('i-br',(d.bearing_rad*180/Math.PI).toFixed(1)+'°');
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
  const cfg=d.filter_cfg||{};
  const mStr=cfg.use_intensity?'Intensity (visual)':'Z height → clustering';
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
      if(r.ok){const d=await r.json();lastData=d;draw(d);panel(d);drawElev(d);
               badge.textContent='live';badge.className='live';}
    }catch(e){badge.textContent='disconnected';badge.className='';}
    await new Promise(r=>setTimeout(r,80));
  }
}
function resize(){
  const w=document.getElementById('cw');
  const s=Math.min(w.clientWidth-12,w.clientHeight-12,700);
  cv.width=s;cv.height=s;if(lastData)draw(lastData);
  const ew=document.getElementById('elev-wrap');
  if(ew){
    const ec=document.getElementById('elev-cv');
    ec.width=ew.clientWidth-8;
    ec.height=Math.min(ew.clientHeight-32,500);
    if(lastData)drawElev(lastData);
  }
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

        cloud_all, cloud_slice, cloud_elev = [], [], []
        rc = snap.get('raw_cloud')
        if rc is not None and len(rc):
            # _apply_slice_filter returns x already negated for upside-down correction;
            # web JS uses cx+x*sc (no additional negation)
            all_xy, slice_xy = _apply_slice_filter(rc, snap['filter_cfg'])
            if all_xy is not None and len(all_xy):
                stride = max(1, len(all_xy) // 600)
                cloud_all = all_xy[::stride].tolist()
            if slice_xy is not None and len(slice_xy):
                stride = max(1, len(slice_xy) // 300)
                cloud_slice = slice_xy[::stride].tolist()
            stride_e   = max(1, len(rc) // 400)
            pts_e      = rc[::stride_e]
            ranges_e   = np.hypot(pts_e[:, 0], pts_e[:, 1])
            cloud_elev = np.column_stack([ranges_e, pts_e[:, 2]]).tolist()

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
            cloud_elev       = cloud_elev,
            filter_cfg       = snap['filter_cfg'],
        ))

    @app.route('/api/set_config', methods=['POST'])
    def api_set_config():
        data = freq.get_json(silent=True) or {}
        allowed = {'z_upper', 'z_lower', 'z2_upper', 'z2_lower',
                   'max_range', 'min_range', 'use_intensity',
                   'int_lower', 'int_upper'}
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
