#!/usr/bin/env python3
"""
DGPPO Debug Visualizer

Run alongside dgppo_ros_node to see live policy output.

Subscribes to:
  /dgppo_action      (Float32MultiArray [a0, a1]) — policy velocity command
  /dgppo_plan_step   (Int32)                       — current index into plan_sequence
  /predicted_cluster (Int16)                        — raw cluster classifier output
  /current_terrain   (Int32)                        — terrain type: 0=Road 1=Grass 2=Sidewalk

Usage:
  python3 dgppo_debug_visualizer.py [path/to/highlevel_plan.json]

The arrow plot shows the commanded velocity vector in CARLA world coordinates
(+X = East, +Y = North) so the direction the arrow points is where the robot
will move on the ground plane.

  action[0,0] → model-x velocity → CARLA -Y   (arrow Y = -action[0,0])
  action[0,1] → model-y velocity → CARLA +X   (arrow X =  action[0,1])

The yellow dashed arrow shows the plan bearing for comparison.
"""

import sys
import os
import json
import math
import threading
import time as _time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # change to Qt5Agg if TkAgg is not available
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int32, Float32MultiArray

# ── Constants ──────────────────────────────────────────────────────────────────

TERRAIN_NAMES  = {0: "Road", 1: "Grass", 2: "Sidewalk"}
TERRAIN_COLORS = {0: "#888888", 1: "#3cb371", 2: "#aaaaaa"}

CLUSTER_NAMES = {
    0: "open_space",
    1: "approach_bridge",
    2: "on_bridge",
    3: "exit_bridge",
}

# Must mirror _map_cluster_id in dgppo_ros_node.py
RAW_TO_MAPPED = {
    **{k: 1 for k in [2, 3]},
    **{k: 2 for k in [5, 6, 7, 8, 9]},
    **{k: 3 for k in [-1, 4]},
    **{k: 0 for k in [0, 1]},
}

HISTORY_LEN  = 25
VEL_HIST_LEN = 300   # ~30 s at 100 ms update rate (fits 3+ cycles of 4 s half-period step test)


# ── Shared state ───────────────────────────────────────────────────────────────

class DebugState:
    def __init__(self, plan_sequence, bearing_map):
        self._lock = threading.Lock()
        self.action       = np.zeros(2)
        self.raw_cluster  = None
        self.terrain_id   = 1
        self.plan_step    = 0
        self.plan_sequence = plan_sequence
        self.bearing_map   = bearing_map
        self.has_action    = False
        self.imu_yaw       = None  # radians, None = not received yet
        self.lidar_all     = None  # (n_rays, 2) hit positions relative to agent
        self.lidar_topk    = None  # (top_k, 2) closest hits sent to policy
        self.state_debug   = None  # 12-float transform debug from /dgppo_state_debug
        # velocity time-series (cmd vs reported, vision frame)
        self._vel_t0    = None
        self.vel_times   = deque(maxlen=VEL_HIST_LEN)
        self.cmd_vx_hist = deque(maxlen=VEL_HIST_LEN)
        self.cmd_vy_hist = deque(maxlen=VEL_HIST_LEN)
        self.rep_vx_hist = deque(maxlen=VEL_HIST_LEN)
        self.rep_vy_hist = deque(maxlen=VEL_HIST_LEN)

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

    def set_imu_yaw(self, yaw_rad):
        with self._lock:
            self.imu_yaw = yaw_rad

    def set_state_debug(self, data):
        with self._lock:
            self.state_debug = data
            now = _time.time()
            if self._vel_t0 is None:
                self._vel_t0 = now
            self.vel_times.append(now - self._vel_t0)
            self.cmd_vx_hist.append(data[10])
            self.cmd_vy_hist.append(data[11])
            self.rep_vx_hist.append(data[2])
            self.rep_vy_hist.append(data[3])

    def set_lidar(self, all_hits, topk_hits):
        with self._lock:
            self.lidar_all  = all_hits
            self.lidar_topk = topk_hits

    def snapshot(self):
        with self._lock:
            return dict(
                action       = self.action.copy(),
                raw_cluster  = self.raw_cluster,
                terrain_id   = self.terrain_id,
                plan_step    = self.plan_step,
                plan_sequence= self.plan_sequence,
                bearing_map  = self.bearing_map,
                has_action   = self.has_action,
                imu_yaw      = self.imu_yaw,
                lidar_all    = self.lidar_all,
                lidar_topk   = self.lidar_topk,
                state_debug  = self.state_debug,
                vel_times    = list(self.vel_times),
                cmd_vx_hist  = list(self.cmd_vx_hist),
                cmd_vy_hist  = list(self.cmd_vy_hist),
                rep_vx_hist  = list(self.rep_vx_hist),
                rep_vy_hist  = list(self.rep_vy_hist),
            )


# ── ROS subscriber node ────────────────────────────────────────────────────────

class DebugSubscriber(Node):
    def __init__(self, state: DebugState):
        super().__init__('dgppo_debug_visualizer')
        self.state = state
        self.create_subscription(Float32MultiArray, '/dgppo_action',      self._cb_action,   10)
        self.create_subscription(Int16,             '/predicted_cluster', self._cb_cluster,  10)
        self.create_subscription(Int32,             '/current_terrain',   self._cb_terrain,  10)
        self.create_subscription(Int32,             '/dgppo_plan_step',   self._cb_planstep, 10)
        self.create_subscription(Float32MultiArray, '/dgppo_imu_yaw',     self._cb_imu_yaw,  10)
        self.create_subscription(Float32MultiArray, '/dgppo_lidar_all',   self._cb_lidar_all,   10)
        self.create_subscription(Float32MultiArray, '/dgppo_lidar_topk',  self._cb_lidar_topk,  10)
        self.create_subscription(Float32MultiArray, '/dgppo_state_debug', self._cb_state_debug, 10)
        self._lidar_all_buf  = None

    def _cb_action(self, msg):
        if len(msg.data) >= 2:
            self.state.set_action(msg.data[0], msg.data[1])

    def _cb_cluster(self, msg):
        self.state.set_cluster(msg.data)

    def _cb_terrain(self, msg):
        self.state.set_terrain(msg.data)

    def _cb_planstep(self, msg):
        self.state.set_plan_step(msg.data)

    def _cb_imu_yaw(self, msg):
        if len(msg.data) >= 1:
            self.state.set_imu_yaw(msg.data[0])

    def _cb_lidar_all(self, msg):
        if len(msg.data) >= 2:
            self._lidar_all_buf = np.array(msg.data, dtype=np.float32).reshape(-1, 2)

    def _cb_lidar_topk(self, msg):
        if len(msg.data) >= 2 and self._lidar_all_buf is not None:
            topk = np.array(msg.data, dtype=np.float32).reshape(-1, 2)
            self.state.set_lidar(self._lidar_all_buf, topk)

    def _cb_state_debug(self, msg):
        if len(msg.data) >= 12:
            self.state.set_state_debug(list(msg.data))


def ros_thread(state: DebugState):
    rclpy.init()
    node = DebugSubscriber(state)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _estimate_lag_ms(t_arr, cmd, rep):
    """Cross-correlation lag estimate (cmd → reported) in milliseconds.
    Returns None when signal variance is too low for a reliable estimate.
    Positive result means reported lags behind cmd (expected for any physical system)."""
    if len(t_arr) < 20 or cmd.std() < 0.02 or rep.std() < 0.02:
        return None
    dt = float(np.mean(np.diff(t_arr)))
    cc = np.correlate(rep - rep.mean(), cmd - cmd.mean(), mode='full')
    lags = np.arange(-(len(cmd) - 1), len(cmd))
    lag_s = float(lags[int(np.argmax(cc))]) * dt
    return lag_s * 1000.0 if 0.0 <= lag_s <= 3.0 else None


# ── Matplotlib visualizer ──────────────────────────────────────────────────────

BG_DARK  = '#1a1a2e'
BG_MID   = '#16213e'
BG_PANEL = '#0f3460'
GRAY     = '#666666'
WHITE    = '#e0e0e0'


def build_figure():
    fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)
    fig.suptitle('DGPPO Policy Debugger', color=WHITE, fontsize=13, y=0.98)

    # Top-left: arrow plot (shrunk vertically to make room for vel plots below)
    ax = fig.add_axes([0.04, 0.40, 0.50, 0.55])
    ax.set_facecolor(BG_MID)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.7)
    for s in ax.spines.values():
        s.set_color('#333333')
    ax.tick_params(colors=GRAY, labelsize=7)

    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='#333333', lw=1, ls='--')

    ax.text( 1.15,  0.0,  'right\n(+x)',   color='#555', fontsize=7, ha='center', va='center')
    ax.text(-1.15,  0.0,  'left\n(-x)',    color='#555', fontsize=7, ha='center', va='center')
    ax.text( 0.0,   1.15, 'forward\n(+y)', color='#555', fontsize=7, ha='center', va='center')
    ax.text( 0.0,  -1.15, 'back\n(-y)',    color='#555', fontsize=7, ha='center', va='center')
    ax.set_title('Policy velocity command  (ground plane)', color=WHITE, fontsize=10, pad=6)

    leg = [
        mpatches.Patch(color='#00e676', label='action (policy)'),
        mpatches.Patch(color='#ffcc00', label='plan bearing'),
        mpatches.Patch(color='#e040fb', label='cart heading (IMU)'),
        mpatches.Patch(color='#444488', label='lidar hits (all)'),
        mpatches.Patch(color='#ff9900', label='lidar hits (top-k)'),
    ]
    ax.legend(handles=leg, loc='lower right', facecolor=BG_MID, edgecolor=GRAY,
              labelcolor=WHITE, fontsize=8)

    # Bottom-left: vx over time (cmd vs reported, vision frame)
    ax_vx = fig.add_axes([0.04, 0.06, 0.23, 0.30])
    ax_vx.set_facecolor(BG_MID)
    ax_vx.set_title('vx  (vision frame)', color=WHITE, fontsize=9, pad=4)
    ax_vx.set_xlabel('time  s', color=GRAY, fontsize=7)
    ax_vx.set_ylabel('m/s', color=GRAY, fontsize=7)
    ax_vx.tick_params(colors=GRAY, labelsize=7)
    ax_vx.axhline(0, color=GRAY, lw=0.6, ls='--')
    for s in ax_vx.spines.values():
        s.set_color('#333333')
    vx_leg = [
        mpatches.Patch(color='#ff4466', label='cmd'),
        mpatches.Patch(color='#44aaff', label='reported'),
    ]
    ax_vx.legend(handles=vx_leg, loc='upper left', facecolor=BG_MID, edgecolor=GRAY,
                 labelcolor=WHITE, fontsize=7)

    # Bottom-center-left: vy over time
    ax_vy = fig.add_axes([0.29, 0.06, 0.23, 0.30])
    ax_vy.set_facecolor(BG_MID)
    ax_vy.set_title('vy  (vision frame)', color=WHITE, fontsize=9, pad=4)
    ax_vy.set_xlabel('time  s', color=GRAY, fontsize=7)
    ax_vy.set_ylabel('m/s', color=GRAY, fontsize=7)
    ax_vy.tick_params(colors=GRAY, labelsize=7)
    ax_vy.axhline(0, color=GRAY, lw=0.6, ls='--')
    for s in ax_vy.spines.values():
        s.set_color('#333333')
    vy_leg = [
        mpatches.Patch(color='#ff4466', label='cmd'),
        mpatches.Patch(color='#44aaff', label='reported'),
    ]
    ax_vy.legend(handles=vy_leg, loc='upper left', facecolor=BG_MID, edgecolor=GRAY,
                 labelcolor=WHITE, fontsize=7)

    # Right: info panel
    ax_info = fig.add_axes([0.56, 0.06, 0.41, 0.88])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.axis('off')

    return fig, ax, ax_vx, ax_vy, ax_info


def run_visualizer(state: DebugState):
    fig, ax, ax_vx, ax_vy, ax_info = build_figure()
    history = deque(maxlen=HISTORY_LEN)

    # velocity plot line handles — created once, updated each frame
    _empty: list = []
    ln_cmd_vx,  = ax_vx.plot(_empty, _empty, color='#ff4466', lw=1.5, label='cmd')
    ln_rep_vx,  = ax_vx.plot(_empty, _empty, color='#44aaff', lw=1.5, label='reported')
    ln_cmd_vy,  = ax_vy.plot(_empty, _empty, color='#ff4466', lw=1.5, label='cmd')
    ln_rep_vy,  = ax_vy.plot(_empty, _empty, color='#44aaff', lw=1.5, label='reported')

    # mutable handles so we can remove/redraw each frame
    handles = {'arrow': None, 'bearing': None, 'imu': None, 'trail': [], 'texts': [], 'speed_ring': None, 'lidar': []}

    def _clear():
        for h in handles['lidar']:
            try:
                h.remove()
            except Exception:
                pass
        handles['lidar'].clear()
        for key in ('arrow', 'bearing', 'imu', 'speed_ring'):
            h = handles[key]
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
            handles[key] = None
        for h in handles['trail']:
            try:
                h.remove()
            except Exception:
                pass
        handles['trail'].clear()
        for t in handles['texts']:
            try:
                t.remove()
            except Exception:
                pass
        handles['texts'].clear()

    def update(_frame):
        snap = state.snapshot()
        _clear()

        a0, a1 = float(snap['action'][0]), float(snap['action'][1])
        # Model space: forward = +y = action[1], right = +x = action[0]
        # Plot: up = forward, right = right → direct mapping
        arrow_x = a0
        arrow_y = a1
        mag = math.hypot(arrow_x, arrow_y)

        terrain_id    = snap['terrain_id']
        raw_cluster   = snap['raw_cluster']
        plan_step     = snap['plan_step']
        plan_sequence = snap['plan_sequence']
        bearing_map   = snap['bearing_map']
        imu_yaw       = snap['imu_yaw']
        lidar_all     = snap.get('lidar_all')
        lidar_topk    = snap.get('lidar_topk')

        mapped_cluster = RAW_TO_MAPPED.get(raw_cluster, raw_cluster) if raw_cluster is not None else None

        # ── History trail ─────────────────────────────────────────────────────
        history.append((arrow_x, arrow_y))
        n = len(history)
        for i, (hx, hy) in enumerate(list(history)[:-1]):
            if math.hypot(hx, hy) < 0.02:
                continue
            alpha = 0.05 + 0.25 * i / max(n, 1)
            width = 0.5 + 1.0 * i / max(n, 1)
            h = ax.annotate(
                '', xy=(hx, hy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#4488ff', alpha=alpha, lw=width)
            )
            handles['trail'].append(h)

        # ── Plan bearing arrow ────────────────────────────────────────────────
        bearing_rad = None
        current_step = None
        if plan_step < len(plan_sequence):
            current_step = plan_sequence[plan_step]
            key = f"{current_step['start']}-{current_step['next']}"
            bearing_rad = bearing_map.get(key)

        if bearing_rad is not None:
            # bearing_rad = atan2(dy_carla, dx_carla) for CARLA world
            bx = math.cos(bearing_rad)
            by = math.sin(bearing_rad)
            handles['bearing'] = ax.annotate(
                '', xy=(bx * 1.05, by * 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=2.5,
                                alpha=0.75, mutation_scale=20)
            )

        # ── LiDAR rays (lines from origin to hit) ────────────────────────────
        if lidar_all is not None:
            # one Line2D per ray is slow; use LineCollection for all at once
            from matplotlib.collections import LineCollection
            segs = [[[0, 0], [x, y]] for x, y in lidar_all]
            lc = LineCollection(segs, colors='#444444', linewidths=0.6,
                                alpha=0.5, zorder=2)
            ax.add_collection(lc)
            handles['lidar'].append(lc)
            # small dot at each hit
            h = ax.scatter(lidar_all[:, 0], lidar_all[:, 1],
                           s=6, color='#666666', zorder=2, linewidths=0)
            handles['lidar'].append(h)
        if lidar_topk is not None:
            h = ax.scatter(lidar_topk[:, 0], lidar_topk[:, 1],
                           s=40, color='#ff9900', zorder=3, linewidths=0)
            handles['lidar'].append(h)

        # ── IMU cart heading arrow ────────────────────────────────────────────
        # yaw=0 = IMU reference direction at startup; shown as a unit-length arrow.
        # Direction is relative to the IMU's own frame — compare it visually to the
        # policy arrow to judge if the cart is pointed the right way.
        if imu_yaw is not None:
            display_yaw = imu_yaw + math.pi / 2  # mounting offset: IMU 0 is East, display 0 is North
            ix = math.cos(display_yaw)
            iy = math.sin(display_yaw)
            handles['imu'] = ax.annotate(
                '', xy=(ix, iy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#e040fb', lw=2.5,
                                alpha=0.8, mutation_scale=20)
            )

        # ── Main action arrow ─────────────────────────────────────────────────
        if snap['has_action']:
            color = '#00e676' if mag > 0.05 else '#ff5252'
            if mag > 0.01:
                handles['arrow'] = ax.annotate(
                    '', xy=(arrow_x, arrow_y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=3.5, mutation_scale=30)
                )
            else:
                h, = ax.plot([0], [0], 'o', color='#ff5252', ms=14, zorder=5)
                handles['arrow'] = h

            # Speed ring: circle scaled to action magnitude
            from matplotlib.patches import Circle
            ring = Circle((0, 0), mag, fill=False, color=color, lw=1.2,
                           alpha=0.35, linestyle=':')
            ax.add_patch(ring)
            handles['speed_ring'] = ring
        else:
            t = ax.text(0, 0, 'waiting\nfor\n/dgppo_action', color='#888',
                        ha='center', va='center', fontsize=11)
            handles['texts'].append(t)

        # ── Info panel ────────────────────────────────────────────────────────
        terrain_name  = TERRAIN_NAMES.get(terrain_id, f'Unknown({terrain_id})')
        terrain_color = TERRAIN_COLORS.get(terrain_id, WHITE)
        cluster_name  = CLUSTER_NAMES.get(mapped_cluster, f'cluster_{mapped_cluster}') \
                        if mapped_cluster is not None else '—'

        rows = []  # (label, value, color)

        rows.append(('TERRAIN', terrain_name, terrain_color))
        rows.append(('', '', ''))

        if raw_cluster is not None:
            rows.append(('CLUSTER raw', str(raw_cluster), '#ddddff'))
            rows.append(('CLUSTER mapped', f'{mapped_cluster}  {cluster_name}', '#aaaaff'))
        else:
            rows.append(('CLUSTER', 'waiting...', GRAY))
        rows.append(('', '', ''))

        if plan_step < len(plan_sequence) and current_step is not None:
            sc = CLUSTER_NAMES.get(current_step['start'], str(current_step['start']))
            nc = CLUSTER_NAMES.get(current_step['next'],  str(current_step['next']))
            rows.append(('PLAN STEP', f"{plan_step + 1} / {len(plan_sequence)}", '#ffdd88'))
            rows.append(('FROM', sc, '#ff9955'))
            rows.append(('TO',   nc, '#ff9955'))
            rows.append(('', '', ''))
            if bearing_rad is not None:
                bearing_deg = math.degrees(bearing_rad)
                rows.append(('BEARING', f'{bearing_deg:+.1f}°', '#ffcc00'))
                if snap['has_action']:
                    action_angle = math.degrees(math.atan2(arrow_y, arrow_x))
                    diff = (action_angle - bearing_deg + 180) % 360 - 180
                    diff_color = '#00e676' if abs(diff) < 30 else \
                                 '#ffaa00' if abs(diff) < 60 else '#ff5252'
                    rows.append(('ACTION vs BEARING', f'{diff:+.1f}°', diff_color))
        elif plan_step >= len(plan_sequence):
            rows.append(('PLAN', 'COMPLETE', '#00e676'))
        else:
            rows.append(('PLAN', 'loading...', GRAY))
        rows.append(('', '', ''))

        rows.append(('a[0] (right/+x)', f'{a0:+.4f}', '#cccccc'))
        rows.append(('a[1] (fwd/+y)',   f'{a1:+.4f}', '#cccccc'))
        rows.append(('SPEED |a|', f'{mag:.4f}', '#aaaaaa'))

        if imu_yaw is not None:
            rows.append(('', '', ''))
            rows.append(('IMU YAW (rel)', f'{math.degrees(imu_yaw + math.pi / 2):+.1f}°', '#e040fb'))

        sd = snap.get('state_debug')
        if sd and len(sd) >= 12:
            rows.append(('', '', ''))
            rows.append(('── FRAME DEBUG ──', '', '#555566'))
            rows.append(('pos_vis x/y  m',     f'{sd[0]:+.3f} / {sd[1]:+.3f}',  '#aaddff'))
            rows.append(('vel_vis x/y  m/s',   f'{sd[2]:+.3f} / {sd[3]:+.3f}',  '#aaddff'))
            rows.append(('vel_body fwd/lat',    f'{sd[4]:+.3f} / {sd[5]:+.3f}',  '#aaffaa'))
            rows.append(('sim_pos x/y',         f'{sd[6]:+.3f} / {sd[7]:+.3f}',  '#ffddaa'))
            rows.append(('sim_vel x/y',         f'{sd[8]:+.4f} / {sd[9]:+.4f}',  '#ffddaa'))
            rows.append(('cmd vx/vy  m/s',      f'{sd[10]:+.3f} / {sd[11]:+.3f}', '#ffaaff'))

        # Draw rows
        y = 0.97
        dy = 0.065
        texts = handles['texts']
        for label, value, color in rows:
            if not label and not value:
                y -= dy * 0.4
                continue
            t1 = ax_info.text(0.04, y, f'{label}', transform=ax_info.transAxes,
                               color='#777777', fontsize=8.5, va='top', fontweight='bold')
            t2 = ax_info.text(0.96, y, value, transform=ax_info.transAxes,
                               color=color, fontsize=8.5, va='top', ha='right',
                               fontfamily='monospace')
            texts.extend([t1, t2])
            y -= dy

        # ── Velocity time-series plots ────────────────────────────────────────
        t_arr = np.array(snap.get('vel_times', []))
        if len(t_arr) >= 2:
            cmd_vx = np.array(snap['cmd_vx_hist'])
            rep_vx = np.array(snap['rep_vx_hist'])
            cmd_vy = np.array(snap['cmd_vy_hist'])
            rep_vy = np.array(snap['rep_vy_hist'])

            ln_cmd_vx.set_data(t_arr, cmd_vx)
            ln_rep_vx.set_data(t_arr, rep_vx)
            ln_cmd_vy.set_data(t_arr, cmd_vy)
            ln_rep_vy.set_data(t_arr, rep_vy)

            t_min, t_max = t_arr[0], t_arr[-1]
            t_span = max(t_max - t_min, 1.0)
            for _ax, _cv, _rv, _base in (
                    (ax_vx, cmd_vx, rep_vx, 'vx  (vision frame)'),
                    (ax_vy, cmd_vy, rep_vy, 'vy  (vision frame)')):
                _ax.set_xlim(t_min, t_min + t_span)
                all_vals = np.concatenate([_cv, _rv])
                v_lo, v_hi = all_vals.min(), all_vals.max()
                margin = max((v_hi - v_lo) * 0.15, 0.05)
                _ax.set_ylim(v_lo - margin, v_hi + margin)
                lag = _estimate_lag_ms(t_arr, _cv, _rv)
                if lag is not None:
                    c = '#00cc44' if lag < 150 else '#ffaa00' if lag < 400 else '#ff4444'
                    _ax.set_title(f'{_base}   lag ≈ {lag:.0f} ms', color=c, fontsize=9, pad=4)
                else:
                    _ax.set_title(f'{_base}   (need more signal)', color=GRAY, fontsize=9, pad=4)

        fig.canvas.draw_idle()

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.show()
    return ani  # keep reference so GC doesn't collect it


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    plan_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), 'plans', 'bridge.json')

    plan_sequence, bearing_map = [], {}
    if os.path.exists(plan_path):
        
        with open(plan_path) as f:
            data = json.load(f)
        plan_sequence = data.get('plan_sequence', [])
        bearing_map   = data.get('bearing_map', {})
        print(f"Loaded plan: {len(plan_sequence)} steps from {plan_path}")
    else:
        print(f"[WARN] Plan file not found: {plan_path}  — cluster/bearing info will be blank")

    state = DebugState(plan_sequence, bearing_map)

    t = threading.Thread(target=ros_thread, args=(state,), daemon=True)
    t.start()

    _ani = run_visualizer(state)


if __name__ == '__main__':
    main()
