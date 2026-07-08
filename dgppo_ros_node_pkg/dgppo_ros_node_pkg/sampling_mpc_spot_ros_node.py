"""
sampling_mpc_spot_ros_node.py
==============================
ROS2 Spot testing node that runs SamplingMPC (topological) instead of
the DGPPO neural network policy.

Minimal diff from spot_dgppo_ros_node.py:
  - _init_model  → loads SamplingMPC + BehaviorAssociator from plan JSON
  - _run_inference → replaced by _run_sampling_mpc_step (no JAX graph)
  - _apply_action → SamplingMPC outputs body-frame (v_cmd, om_cmd) directly;
                    no world_alpha / DGPPO_TO_VISION_R transforms needed
  - send_velocity_with_yaw → new helper that sends v_rot=om_cmd to Spot

Everything else (Spot SDK, plan advancement, background threads,
/processed_ranges subscriber, debug logging) is kept identical.

Debug topics added:
  /sampling_mpc_scores         — Float32MultiArray  top-10 rollout scores
  /sampling_mpc_best_rollout   — Float32MultiArray  N×3 best waypoints (body frame)
  /sampling_mpc_guidance_debug — Float32MultiArray
      [guidance_ratio, cluster_score, bearing_score, edt_blocked_frac, v_cmd, om_cmd]
"""

import datetime
import json
import math
import os
import sys
import threading
import time
from typing import NamedTuple, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile,
    ReliabilityPolicy, qos_profile_sensor_data,
)

from std_msgs.msg import Int16, Int32, Float32MultiArray
from std_srvs.srv import Trigger

import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.exceptions import LeaseUseError
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient

from .plan import load_plan, map_cluster_id
from .utils import SCALE_SPOT_TO_SIM, rot2d, apply_rot2d
from .spot_utils import get_spot_state, send_velocity

# ── SamplingMPC + BehaviorAssociator (loaded from new_dgppo repo) ─────────────
import importlib.util as _ilu

def _load_mod(name, rel):
    _root = os.path.join(os.path.dirname(__file__),
                          "..", "..", "..", "..", "new_dgppo")
    path = os.path.join(_root, rel)
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ba_mod  = _load_mod("behavior_associator", "dgppo/env/behavior_associator.py")
_mpc_mod = _load_mod("sampling_mpc",        "dgppo/env/sampling_mpc.py")

BehaviorAssociator = _ba_mod.BehaviorAssociator
SamplingMPC        = _mpc_mod.SamplingMPC
unicycle_rollout   = _mpc_mod.unicycle_rollout


# ══════════════════════════════════════════════════════════════════════════════

class SpotSimState(NamedTuple):
    x: float; y: float; vx: float; vy: float; yaw: float
    raw_ranges: np.ndarray    # (72,) LiDAR ranges in metres


class MPCResult(NamedTuple):
    v_cmd:          float
    om_cmd:         float
    guidance_ratio: float     # cluster_score / (|cluster| + |bearing| + eps)
    cluster_score:  float
    bearing_score:  float
    edt_blocked:    float     # fraction of rollouts blocked by EDT
    best_rollout:   np.ndarray  # (N, 3) best rollout in body frame


class SamplingMPCSpotNode(Node):
    def __init__(self):
        super().__init__("sampling_mpc_spot_node")
        self.get_logger().info("Initializing SamplingMPC Spot Node...")
        self._init_params()
        self._init_model()
        self._init_ros()
        self._init_spot()
        self.get_logger().info("SamplingMPC Spot Node fully initialized.")

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_params(self):
        self.declare_parameter("debug_mode",   False)
        self.declare_parameter("current_cluster_id", 1)
        self.declare_parameter("dry_run",      False)
        self.declare_parameter("sampling_mpc_K",            500)
        self.declare_parameter("sampling_mpc_N",            8)
        self.declare_parameter("sampling_mpc_safety_radius", 0.3)
        self.declare_parameter("sampling_mpc_dt",           0.2)
        # world_alpha_rad: kept as a stub for future calibration; not used for
        # SamplingMPC since it outputs body-frame commands directly.
        self.declare_parameter("world_y_offset_deg", 0.0)
        self._world_alpha_rad = math.radians(
            self.get_parameter("world_y_offset_deg").get_parameter_value().double_value
        )
        self.n_rays_phys = 72

    def _init_model(self):
        """Load plan, build BehaviorAssociator and SamplingMPC from plan centroids."""
        _spot_plan = os.path.join(os.path.dirname(__file__), "plans", "bridge_spot.json")
        self.plan_sequence, self.bearing_map, self.cluster_centroids = load_plan(_spot_plan)
        self.current_plan_step_index = 0

        K = self.get_parameter("sampling_mpc_K").get_parameter_value().integer_value
        N = self.get_parameter("sampling_mpc_N").get_parameter_value().integer_value
        dt = self.get_parameter("sampling_mpc_dt").get_parameter_value().double_value
        sr = self.get_parameter("sampling_mpc_safety_radius").get_parameter_value().double_value

        # Build bridge geometry from plan centroids
        # cluster_centroids is {"0": [x,y], "1": [x,y], ...}
        cent_array = np.array([
            self.cluster_centroids[str(k)]
            for k in sorted(int(k) for k in self.cluster_centroids)
        ], dtype=np.float32)
        self._true_cents = cent_array[:, :2]   # (n_c, 2) — drop z column

        # BehaviorAssociator uses bridge wall OBBs; derive them from centroid spread.
        # The plan JSON must include "bridges" key with OBB list; fall back to
        # a flat horizontal bridge if absent.
        plan_bridges = self._load_plan_bridges()
        self._assoc = BehaviorAssociator(bridges=plan_bridges, buildings=[], obstacles=[])
        self._rid   = self._assoc.region_name_to_id

        self._mpc_K  = K
        self._mpc_N  = N
        self._mpc_dt = dt
        self._mpc_sr = sr
        self._mpc = SamplingMPC(K=K, N=N, dt=dt, safety_radius=sr, lidar_grid_size=1.5)
        self._mpc.reset(self._assoc)
        self._mpc._centroids_global = cent_array.copy()

        # Current state variables
        self.latest_ranges_msg         = None
        self.latest_predicted_cluster_id = None
        self.is_first_run              = True
        self._current_pair_idx         = 0
        self._start_id  = None
        self._target_id = None
        self._forbidden = []
        self._n_c = len(cent_array)
        self.get_logger().info(
            f"SamplingMPC ready: K={K} N={N} dt={dt} sr={sr}  "
            f"n_clusters={self._n_c}"
        )

    def _load_plan_bridges(self) -> list:
        """Return bridge OBB list from plan JSON, or a default horizontal bridge."""
        plan_path = os.path.join(os.path.dirname(__file__), "plans", "highlevel_plan.json")
        try:
            with open(plan_path) as f:
                plan_data = json.load(f)
            if "bridges" in plan_data:
                return [tuple(b) for b in plan_data["bridges"]]
        except Exception as e:
            self.get_logger().warning(f"Could not load bridges from plan: {e}")
        # Default: horizontal bridge at centroid mean ± 0.18 in y
        cx = float(np.mean(self._true_cents[:, 0]))
        cy = float(np.mean(self._true_cents[:, 1]))
        return [(cx, cy + 0.18, 1.0, 0.06, 0.0),
                (cx, cy - 0.18, 1.0, 0.06, 0.0)]

    def _init_ros(self):
        self.ranges_sub = self.create_subscription(
            Float32MultiArray, "/processed_ranges",
            self.ranges_callback, qos_profile=qos_profile_sensor_data)
        self.predicted_cluster_sub = self.create_subscription(
            Int16, "/predicted_cluster", self.predicted_cluster_callback, 10)
        self.terrain_sub = self.create_subscription(
            Int16, "/current_terrain", self.terrain_callback, 10)

        self.spot_yaw_pub   = self.create_publisher(Float32MultiArray, "/dgppo_spot_yaw",  10)
        self.spot_act_pub   = self.create_publisher(Float32MultiArray, "/dgppo_action",    100)
        self.state_debug_pub = self.create_publisher(Float32MultiArray, "/dgppo_state_debug", 10)
        self.plan_step_pub  = self.create_publisher(Int32, "/dgppo_plan_step", 10)

        # New SamplingMPC debug topics
        self.scores_pub   = self.create_publisher(Float32MultiArray, "/sampling_mpc_scores",         10)
        self.rollout_pub  = self.create_publisher(Float32MultiArray, "/sampling_mpc_best_rollout",    10)
        self.guidance_pub = self.create_publisher(Float32MultiArray, "/sampling_mpc_guidance_debug",  10)

        _latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.world_alpha_pub = self.create_publisher(
            Float32MultiArray, "/dgppo_world_alpha", _latched_qos)
        _alpha_msg = Float32MultiArray()
        _alpha_msg.data = [self._world_alpha_rad]
        self.world_alpha_pub.publish(_alpha_msg)

        self._take_lease_srv = self.create_service(
            Trigger, "/dgppo_take_lease", self._take_lease_callback)

        dt_sec = self.get_parameter("sampling_mpc_dt").get_parameter_value().double_value
        self.timer = self.create_timer(dt_sec, self.control_loop)

        _log_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(_log_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_log_path = os.path.join(_log_dir, f"smpc_run_{_ts}.jsonl")
        self._debug_log_file = open(self._debug_log_path, "w")
        self.get_logger().info(f"Debug log: {self._debug_log_path}")
        self.latest_terrain_id = 1

    def _init_spot(self):
        self.get_logger().info("Initializing Spot robot.")
        self.sdk = bosdyn.client.create_standard_sdk("smpc-spot")
        self.robot = self.sdk.create_robot("10.0.0.3")
        self.robot.authenticate(username="user", password="pass")
        self.robot.time_sync.wait_for_sync()
        self.state_client   = self.robot.ensure_client("robot-state")
        self.lease_client   = self.robot.ensure_client("lease")
        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name)
        self.lease_client.take()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)

        self._cached_robot_state = None
        self._state_lock = threading.Lock()
        threading.Thread(target=self._poll_spot_state, daemon=True).start()

        self._cmd_vel  = (0.0, 0.0, 0.0)   # (vx, vy, v_rot)
        self._cmd_lock = threading.Lock()
        threading.Thread(target=self._send_commands, daemon=True).start()
        self._tablet_has_lease = False

    # ── Background threads ────────────────────────────────────────────────────

    def _poll_spot_state(self):
        while True:
            try:
                rs = self.state_client.get_robot_state()
                with self._state_lock:
                    self._cached_robot_state = rs
            except Exception:
                pass
            time.sleep(0.02)

    def _send_commands(self):
        while True:
            with self._cmd_lock:
                vx, vy, v_rot = self._cmd_vel
            try:
                self._send_velocity_with_yaw(vx, vy, v_rot)
                self._tablet_has_lease = False
            except (LeaseUseError, bosdyn.client.lease.NotActiveLeaseError):
                self._tablet_has_lease = True
            except Exception:
                pass
            time.sleep(0.04)

    def _send_velocity_with_yaw(self, vx: float, vy: float, v_rot: float):
        """Send body-frame velocity to Spot including yaw rate.

        vx     = forward m/s  (body +x)
        vy     = leftward m/s (body +y) — set to 0 for unicycle motion
        v_rot  = yaw rate rad/s (CCW positive)
        """
        from bosdyn.client.robot_command import RobotCommandBuilder
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=vx, v_y=vy, v_rot=v_rot,
            body_height=0.0, locomotion_hint=1)
        self.command_client.robot_command(
            command=cmd, end_time_secs=time.time() + 0.5)

    def _get_spot_state(self) -> Tuple:
        with self._state_lock:
            rs = self._cached_robot_state
        if rs is None:
            rs = self.state_client.get_robot_state()
        return get_spot_state(rs)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def ranges_callback(self, msg: Float32MultiArray):
        self.latest_ranges_msg = msg

    def predicted_cluster_callback(self, msg: Int16):
        self.latest_predicted_cluster_id = msg.data

    def terrain_callback(self, msg: Int16):
        self.latest_terrain_id = msg.data

    def _take_lease_callback(self, request, response):
        try:
            self.lease_keep_alive.shutdown()
            self.lease_client.take()
            self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)
            self._tablet_has_lease = False
            response.success = True; response.message = "Lease reclaimed."
        except Exception as e:
            response.success = False; response.message = str(e)
        return response

    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.current_plan_step_index >= len(self.plan_sequence):
            self._handle_plan_complete()
            return

        current_plan_step    = self.plan_sequence[self.current_plan_step_index]
        expected_start       = current_plan_step["start"]
        expected_next        = current_plan_step["next"]
        debug_mode = self.get_parameter("debug_mode").get_parameter_value().bool_value

        if not self._check_topics(debug_mode):
            return

        if self._tablet_has_lease:
            self.get_logger().info(
                "Tablet holds lease — paused. "
                "Reclaim: ros2 service call /dgppo_take_lease std_srvs/srv/Trigger '{}'",
                throttle_duration_sec=3.0)
            return

        current_cluster_raw = (
            self.get_parameter("current_cluster_id").get_parameter_value().integer_value
            if debug_mode else self.latest_predicted_cluster_id
        )
        mapped_cluster = map_cluster_id(current_cluster_raw)

        if mapped_cluster == expected_next:
            self._advance_plan(expected_start, expected_next)
            if self.current_plan_step_index >= len(self.plan_sequence):
                return
            current_plan_step = self.plan_sequence[self.current_plan_step_index]
            expected_start    = current_plan_step["start"]
            expected_next     = current_plan_step["next"]

        # Update pair variables
        self._start_id  = expected_start
        self._target_id = expected_next
        self._forbidden = [c for c in range(self._n_c)
                           if c not in (expected_start, expected_next)]

        state  = self._update_spot_state()
        result = self._run_sampling_mpc_step(state, expected_start, expected_next)
        self._apply_action(state, result)
        self._publish_debug(state, result, expected_start, expected_next)

    def _check_topics(self, debug_mode: bool) -> bool:
        missing = []
        if self.latest_ranges_msg is None:
            missing.append("/processed_ranges")
        if not debug_mode and self.latest_predicted_cluster_id is None:
            missing.append("/predicted_cluster")
        if missing:
            self.get_logger().warning(
                "Waiting for: " + ", ".join(missing), throttle_duration_sec=3.0)
            return False
        return True

    def _advance_plan(self, start: int, nxt: int):
        self.current_plan_step_index += 1
        self.get_logger().info(
            f"Plan advanced {start}→{nxt}  "
            f"(step {self.current_plan_step_index}/{len(self.plan_sequence)})")

    def _handle_plan_complete(self):
        self.get_logger().info("Plan complete. Stopping.")
        try:
            self.command_client.robot_command(command=RobotCommandBuilder.stop_command())
        except Exception:
            pass
        self.timer.cancel()

    def _update_spot_state(self) -> SpotSimState:
        raw_ranges = np.array(self.latest_ranges_msg.data, dtype=np.float32)
        x, y, vx, vy, yaw = self._get_spot_state()
        yaw_msg = Float32MultiArray(); yaw_msg.data = [yaw]
        self.spot_yaw_pub.publish(yaw_msg)
        return SpotSimState(x=x, y=y, vx=vx, vy=vy, yaw=yaw, raw_ranges=raw_ranges)

    def _run_sampling_mpc_step(
        self,
        state: SpotSimState,
        start_id: int,
        target_id: int,
    ) -> MPCResult:
        """
        Core SamplingMPC inference step.

        1. Convert /processed_ranges → local 2D hit points (x=forward, y=left)
        2. Filter degenerate d<0.005m hits
        3. Generate K rollouts, EDT mask, Voronoi scoring (TRUE centroids)
        4. Bearing from plan centroid direction (same centroid — deployable map belief)
        5. CBF post-filter
        """
        # ── Step 1: LiDAR hits in body frame ──────────────────────────────────
        ranges = state.raw_ranges
        n_bins = len(ranges)
        angles = np.linspace(0.0, 2 * math.pi, n_bins, endpoint=False)
        # Convention: angle 0 = robot forward (+x body), angles CCW
        # x_body = r*cos(a),  y_body = r*sin(a) ... but CARLA/Spot uses y=forward.
        # Use Spot/CARLA convention: x=r*sin(a), y=r*cos(a) → forward=y, right=x.
        # SamplingMPC unicycle uses forward=x body.  Map: smpc_x=y_spot, smpc_y=-x_spot.
        x_spot = ranges * np.sin(angles)
        y_spot = ranges * np.cos(angles)
        hits_local = np.stack([y_spot, -x_spot], axis=1).astype(np.float32)   # (n_bins,2)
        max_r = float(ranges.max())
        mask = ranges < max_r * 0.99   # only real returns (not max-range misses)
        hits_local = hits_local[mask]

        # Filter degenerate d=0 hits (wall-boundary artefact)
        if len(hits_local) > 0:
            dists = np.linalg.norm(hits_local, axis=1)
            hits_local = hits_local[dists > 0.005]
        hits_in = hits_local if len(hits_local) > 0 else None

        # ── Step 2: BehaviorAssociator pair advancement (Tier A) ───────────────
        # Use nearest-centroid Voronoi as a proxy for current cluster identity.
        sim_pos = np.array([state.x, state.y])
        dists_to_cents = np.linalg.norm(self._true_cents - sim_pos, axis=1)
        curr_id = int(np.argmin(dists_to_cents))

        # ── Step 3: Generate rollouts ──────────────────────────────────────────
        K, N = self._mpc_K, self._mpc_N
        dt   = self._mpc_dt
        v_seqs  = np.random.uniform(0.0, 0.75, (K, N))
        om_seqs = np.random.uniform(-1.5, 1.5,  (K, N))
        ctrl    = np.stack([v_seqs, om_seqs], axis=2)          # (K, N, 2)
        rollouts    = unicycle_rollout(np.zeros(3), ctrl, dt)   # (K, N+1, 3)
        traj_local  = rollouts[:, 1:, :]                        # (K, N, 3)

        # ── Step 4: EDT collision check ───────────────────────────────────────
        self._mpc._occ_grid.update(hits_in)
        dist_values = self._mpc._occ_grid.check_collisions(traj_local)

        # ── Step 5: Global endpoints ──────────────────────────────────────────
        yaw = state.yaw
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R_l2g = np.array([[cy_, -sy_], [sy_, cy_]])
        end_local  = traj_local[:, -1, :2]                      # (K,2)
        end_global = end_local @ R_l2g.T + sim_pos              # (K,2)

        # ── Step 6: TRUE Voronoi cluster membership ────────────────────────────
        dt_c = np.linalg.norm(end_global[:, None, :] - self._true_cents[None, :, :], axis=2)
        nearest = np.argmin(dt_c, axis=1)                       # (K,)
        in_target    = (nearest == target_id).astype(float)
        in_start     = (nearest == start_id).astype(float)
        in_forbidden = np.isin(nearest, self._forbidden).astype(float)
        cluster_parts = 10.0 * in_target + 1.0 * in_start - 15.0 * in_forbidden
        scores = cluster_parts.copy()

        # ── Step 7: Bearing toward plan centroid (distorted direction hint) ────
        tgt_global = self._true_cents[target_id]   # in deployment: distorted centroid
        tgt_local  = (tgt_global - sim_pos) @ np.array([[cy_, sy_], [-sy_, cy_]])
        bearing_local = math.atan2(tgt_local[1], tgt_local[0])
        bearing_cos   = 3.0 * np.cos(traj_local[:, -1, 2] - bearing_local)
        cdir   = tgt_local / (np.linalg.norm(tgt_local) + 1e-6)
        bearing_dot   = 1.0 * (
            np.cos(traj_local[:, -1, 2]) * cdir[0] +
            np.sin(traj_local[:, -1, 2]) * cdir[1])
        bearing_parts = bearing_cos + bearing_dot
        scores += bearing_parts

        # Progress along approach→target direction
        axis_dir = (tgt_global - self._true_cents[start_id])
        ax_norm  = np.linalg.norm(axis_dir) + 1e-6
        axis_dir = axis_dir / ax_norm
        end_along = (end_global - sim_pos) @ axis_dir
        scores += 0.8 * np.clip(end_along, 0, None)

        # ── Step 8: EDT mask ──────────────────────────────────────────────────
        collision = (dist_values < self._mpc.safety_radius).any(axis=1)
        scores[collision] = -np.inf
        edt_blocked = float(np.sum(collision)) / K

        if np.all(~np.isfinite(scores)):
            v_cmd = 0.15; om_cmd = 0.0; best_k = 0
            best_cp = 0.0; best_bp = 0.0
        else:
            best_k  = int(np.argmax(scores))
            v_cmd   = float(ctrl[best_k, 0, 0])
            om_cmd  = float(ctrl[best_k, 0, 1])
            best_cp = float(cluster_parts[best_k])
            best_bp = float(bearing_parts[best_k])

        # ── Step 9: CBF filter ────────────────────────────────────────────────
        if hits_in is not None and len(hits_in) > 0:
            dists   = np.linalg.norm(hits_in, axis=1)
            d_min   = float(np.min(dists))
            nearest_hit = hits_in[int(np.argmin(dists))]
            cos_a   = math.cos(math.atan2(nearest_hit[1], nearest_hit[0]))
            d_safe, alpha = 0.3, 2.0
            h = d_min - d_safe
            if cos_a > 1e-3 and v_cmd * cos_a > alpha * h:
                v_cmd = max(0.0, alpha * h / cos_a)
            if d_min < 2.0 * d_safe and cos_a > 0.3:
                sin_a = math.sin(math.atan2(nearest_hit[1], nearest_hit[0]))
                om_cmd = float(np.clip(
                    om_cmd - 2.0 * alpha * sin_a * (1.0 - h / d_safe), -1.5, 1.5))

        eps = 1e-6
        guidance_ratio = abs(best_cp) / (abs(best_cp) + abs(best_bp) + eps)

        self.get_logger().info(
            f"cluster {curr_id}→{target_id}  "
            f"v={v_cmd:.3f}  ω={om_cmd:.3f}  "
            f"guidance_ratio={guidance_ratio:.2f}  "
            f"edt_blocked={edt_blocked:.2f}",
            throttle_duration_sec=0.5,
        )

        return MPCResult(
            v_cmd=v_cmd,
            om_cmd=om_cmd,
            guidance_ratio=guidance_ratio,
            cluster_score=best_cp,
            bearing_score=best_bp,
            edt_blocked=edt_blocked,
            best_rollout=rollouts[best_k, :, :].copy(),
        )

    def _apply_action(self, state: SpotSimState, result: MPCResult):
        """Send SamplingMPC commands to Spot.

        SamplingMPC outputs body-frame unicycle commands:
          v_cmd  = forward speed (m/s) in SamplingMPC sim scale
          om_cmd = yaw rate (rad/s)

        Scale v_cmd to Spot metric m/s, map om_cmd → v_rot.
        No DGPPO_TO_VISION_R or world_alpha transforms needed.
        """
        SPOT_MAX_VEL = 0.5  # m/s
        # Sim-to-real velocity scale: SamplingMPC sim velocities were calibrated
        # against 0.75 m/s max; Spot real max is 0.5 m/s.
        scale = SPOT_MAX_VEL / 0.75
        vx_body = float(np.clip(result.v_cmd * scale, 0.0, SPOT_MAX_VEL))
        v_rot   = float(np.clip(result.om_cmd, -1.5, 1.5))

        dry = self.get_parameter("dry_run").get_parameter_value().bool_value
        if dry:
            self.get_logger().info(
                f"[DRY RUN] vx={vx_body:.3f}  v_rot={v_rot:.3f}  (no command sent)",
                throttle_duration_sec=0.5)
        else:
            with self._cmd_lock:
                self._cmd_vel = (vx_body, 0.0, v_rot)

        self.get_logger().info(
            f"vx={vx_body:.3f}  v_rot={v_rot:.3f}",
            throttle_duration_sec=0.5)

    def _publish_debug(
        self,
        state: SpotSimState,
        result: MPCResult,
        start_id: int,
        target_id: int,
    ):
        # Guidance debug topic
        gd_msg = Float32MultiArray()
        gd_msg.data = [
            result.guidance_ratio,
            result.cluster_score,
            result.bearing_score,
            result.edt_blocked,
            result.v_cmd,
            result.om_cmd,
        ]
        self.guidance_pub.publish(gd_msg)

        # Best rollout waypoints (N×3)
        br_msg = Float32MultiArray()
        br_msg.data = result.best_rollout.flatten().tolist()
        self.rollout_pub.publish(br_msg)

        # State debug (compatible with spot_dgppo_ros_node format, blanks for DGPPO fields)
        dbg_msg = Float32MultiArray()
        dbg_msg.data = [
            state.x, state.y,   # [0,1] vision frame pos
            state.vx, state.vy, # [2,3] vision frame vel
            result.v_cmd, 0.0,  # [4,5] body fwd/lat (lat=0 for unicycle)
            0.0, 0.0,           # [6,7] sim pos (not applicable)
            0.0, 0.0,           # [8,9] sim vel
            result.v_cmd, 0.0,  # [10,11] cmd fwd/left
            float(start_id),    # [12] plan start cluster
            float(target_id),   # [13] plan target cluster
            result.guidance_ratio,  # [14] guidance ratio (replaces inf_ms)
            result.cluster_score,   # [15]
            result.bearing_score,   # [16]
        ]
        self.state_debug_pub.publish(dbg_msg)

        act_msg = Float32MultiArray()
        act_msg.data = [0.0, result.v_cmd]  # [right=0, fwd=v_cmd]
        self.spot_act_pub.publish(act_msg)

        plan_msg = Int32(); plan_msg.data = int(self.current_plan_step_index)
        self.plan_step_pub.publish(plan_msg)

        # Debug log
        record = {
            "t":              time.time(),
            "plan_start":     start_id,
            "plan_next":      target_id,
            "plan_step":      self.current_plan_step_index,
            "v_cmd":          result.v_cmd,
            "om_cmd":         result.om_cmd,
            "guidance_ratio": result.guidance_ratio,
            "cluster_score":  result.cluster_score,
            "bearing_score":  result.bearing_score,
            "edt_blocked":    result.edt_blocked,
            "spot_x":         state.x,
            "spot_y":         state.y,
            "spot_yaw_deg":   math.degrees(state.yaw),
        }
        self._debug_log_file.write(json.dumps(record) + "\n")
        self._debug_log_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = SamplingMPCSpotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.lease_keep_alive.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
