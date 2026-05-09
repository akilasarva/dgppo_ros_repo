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

HISTORY_LEN = 25


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


def ros_thread(state: DebugState):
    rclpy.init()
    node = DebugSubscriber(state)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── Matplotlib visualizer ──────────────────────────────────────────────────────

BG_DARK  = '#1a1a2e'
BG_MID   = '#16213e'
BG_PANEL = '#0f3460'
GRAY     = '#666666'
WHITE    = '#e0e0e0'


def build_figure():
    fig = plt.figure(figsize=(12, 7), facecolor=BG_DARK)
    fig.suptitle('DGPPO Policy Debugger', color=WHITE, fontsize=13, y=0.97)

    # Left: arrow plot
    ax = fig.add_axes([0.05, 0.08, 0.55, 0.84])
    ax.set_facecolor(BG_MID)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.7)
    for s in ax.spines.values():
        s.set_color('#333333')
    ax.tick_params(colors=GRAY, labelsize=7)

    # unit circle reference
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='#333333', lw=1, ls='--')

    # axis labels (world-frame)
    ax.text( 1.15,  0.0,  'CARLA +X\n(East)',   color='#555', fontsize=7, ha='center', va='center')
    ax.text(-1.15,  0.0,  'CARLA -X\n(West)',   color='#555', fontsize=7, ha='center', va='center')
    ax.text( 0.0,   1.15, 'CARLA +Y\n(North)',  color='#555', fontsize=7, ha='center', va='center')
    ax.text( 0.0,  -1.15, 'CARLA -Y\n(South)',  color='#555', fontsize=7, ha='center', va='center')
    ax.set_title('Policy velocity command  (ground plane)', color=WHITE, fontsize=10, pad=6)

    # legend patches
    leg = [
        mpatches.Patch(color='#00e676', label='action (policy)'),
        mpatches.Patch(color='#ffcc00', label='plan bearing'),
        mpatches.Patch(color='#e040fb', label='cart heading (IMU)'),
    ]
    ax.legend(handles=leg, loc='lower right', facecolor=BG_MID, edgecolor=GRAY,
              labelcolor=WHITE, fontsize=8)

    # Right: info panel
    ax_info = fig.add_axes([0.63, 0.08, 0.34, 0.84])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.axis('off')

    return fig, ax, ax_info


def run_visualizer(state: DebugState):
    fig, ax, ax_info = build_figure()
    history = deque(maxlen=HISTORY_LEN)

    # mutable handles so we can remove/redraw each frame
    handles = {'arrow': None, 'bearing': None, 'imu': None, 'trail': [], 'texts': [], 'speed_ring': None}

    def _clear():
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
        # Convert model-space action to CARLA world arrow components:
        #   action[0] (model-x) → CARLA -Y  → arrow_y = -a0
        #   action[1] (model-y) → CARLA +X  → arrow_x =  a1
        arrow_x =  a1
        arrow_y = -a0
        mag = math.hypot(arrow_x, arrow_y)

        terrain_id    = snap['terrain_id']
        raw_cluster   = snap['raw_cluster']
        plan_step     = snap['plan_step']
        plan_sequence = snap['plan_sequence']
        bearing_map   = snap['bearing_map']
        imu_yaw       = snap['imu_yaw']

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

        # ── IMU cart heading arrow ────────────────────────────────────────────
        # yaw=0 = IMU reference direction at startup; shown as a unit-length arrow.
        # Direction is relative to the IMU's own frame — compare it visually to the
        # policy arrow to judge if the cart is pointed the right way.
        if imu_yaw is not None:
            ix = math.cos(imu_yaw)
            iy = math.sin(imu_yaw)
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

        rows.append(('a[0] (→ CARLA -Y)', f'{a0:+.4f}', '#cccccc'))
        rows.append(('a[1] (→ CARLA +X)', f'{a1:+.4f}', '#cccccc'))
        rows.append(('SPEED |a|', f'{mag:.4f}', '#aaaaaa'))

        if imu_yaw is not None:
            rows.append(('', '', ''))
            rows.append(('IMU YAW (rel)', f'{math.degrees(imu_yaw):+.1f}°', '#e040fb'))

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

        fig.canvas.draw_idle()

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.show()
    return ani  # keep reference so GC doesn't collect it


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    plan_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), 'plans', 'highlevel_plan.json')

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
