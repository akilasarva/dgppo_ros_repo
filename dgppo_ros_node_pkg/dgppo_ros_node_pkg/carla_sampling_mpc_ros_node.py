"""
carla_sampling_mpc_ros_node.py
===============================
CARLA ROS2 node running sampling MPC (topological). Pure numpy — no DGPPO, no JAX.

Coordinate frame (physical metres):
  CARLA world → proc 2D:
    proc_x = -carla_y - ORIG_Y
    proc_y =  carla_x - ORIG_X
  Yaw: sim_yaw_rad = -radians(carla_yaw_deg)

Centroid JSON convention: centroid[0]=carla_x, centroid[1]=-carla_y
  → proc_x = centroid[1] - ORIG_Y
  → proc_y = centroid[0] - ORIG_X

Debug topics:
  /sampling_mpc_scores         Float32MultiArray  top-10 scores
  /sampling_mpc_best_rollout   Float32MultiArray  N×3 waypoints (body frame)
  /sampling_mpc_guidance_debug Float32MultiArray
      [guidance_ratio, cluster_score, bearing_score, edt_blocked_frac, v_cmd, om_cmd]
"""

import datetime
import json
import math
import os
import time
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int16, Float32MultiArray
import carla


# ── Inline unicycle rollout ────────────────────────────────────────────────────

def unicycle_rollout(state0: np.ndarray, ctrl: np.ndarray, dt: float) -> np.ndarray:
    """Batch unicycle rollout.

    state0 : (3,)       initial [x, y, theta] (same for all rollouts)
    ctrl   : (K, N, 2)  control sequences [v, omega]
    dt     : float      time step
    returns: (K, N+1, 3)
    """
    K, N = ctrl.shape[:2]
    traj = np.zeros((K, N + 1, 3), dtype=np.float32)
    traj[:, 0, :] = state0
    for n in range(N):
        x  = traj[:, n, 0]
        y  = traj[:, n, 1]
        th = traj[:, n, 2]
        v  = ctrl[:, n, 0]
        om = ctrl[:, n, 1]
        traj[:, n + 1, 0] = x  + v * np.cos(th) * dt
        traj[:, n + 1, 1] = y  + v * np.sin(th) * dt
        traj[:, n + 1, 2] = th + om * dt
    return traj


# ── Inline EDT occupancy grid ──────────────────────────────────────────────────

class _LocalOccGrid:
    """Minimal EDT: stores LiDAR hits and returns min-distance to any hit."""

    def __init__(self):
        self._hits: np.ndarray = np.empty((0, 2), dtype=np.float32)

    def update(self, hits):
        if hits is None or len(hits) == 0:
            self._hits = np.empty((0, 2), dtype=np.float32)
        else:
            self._hits = np.asarray(hits, dtype=np.float32)

    def check_collisions(self, traj: np.ndarray) -> np.ndarray:
        """Return (K, N) array of min-distance to nearest hit for each rollout point."""
        K, N = traj.shape[:2]
        if len(self._hits) == 0:
            return np.full((K, N), np.inf, dtype=np.float32)
        pts  = traj[:, :, :2]                                    # (K, N, 2)
        diff = pts[:, :, None, :] - self._hits[None, None, :, :]  # (K, N, M, 2)
        return np.linalg.norm(diff, axis=-1).min(axis=-1).astype(np.float32)


class SamplingMPCCarlaNode(Node):
    ORIG_X = 55.0
    ORIG_Y = -210.0

    def __init__(self):
        super().__init__("sampling_mpc_carla_node")
        self.get_logger().info("Initializing SamplingMPC CARLA Node...")
        self._init_params()
        self._init_model()
        self._init_ros()
        self._init_carla()
        self.get_logger().info("SamplingMPC CARLA Node fully initialized.")

    # ── Initialization ─────────────────────────────────────────────────────────

    def _init_params(self):
        self.declare_parameter("debug_mode",                  False)
        self.declare_parameter("current_cluster_id",          1)
        self.declare_parameter("angular_offset_deg",          0.0)
        self.declare_parameter("plan_file",                   "bridge.json")
        self.declare_parameter("distortion_angle_deg",        0.0)
        self.declare_parameter("distortion_scale",            1.0)
        self.declare_parameter("distortion_tx",               0.0)
        self.declare_parameter("distortion_ty",               0.0)
        self.declare_parameter("sampling_mpc_K",              500)
        self.declare_parameter("sampling_mpc_N",              8)
        self.declare_parameter("sampling_mpc_safety_radius",  0.5)
        self.declare_parameter("sampling_mpc_dt",             0.1)
        self.declare_parameter("dry_run",                     False)
        self.num_clusters = 4

    def _init_model(self):
        self.plan_file = self.get_parameter("plan_file").get_parameter_value().string_value
        self.plan_sequence, self.bearing_map, self.cluster_centroids = \
            self._load_plan_and_cluster_data()
        self.current_plan_step_index = 0

        K  = self.get_parameter("sampling_mpc_K").get_parameter_value().integer_value
        N  = self.get_parameter("sampling_mpc_N").get_parameter_value().integer_value
        dt = self.get_parameter("sampling_mpc_dt").get_parameter_value().double_value
        sr = self.get_parameter("sampling_mpc_safety_radius").get_parameter_value().double_value

        # Convert centroids: centroid[0]=carla_x, centroid[1]=-carla_y → physical metres
        sorted_keys = sorted(self.cluster_centroids.keys(), key=lambda k: int(k))
        self._true_cents_sim = np.array([
            [self.cluster_centroids[k][1] - self.ORIG_Y,
             self.cluster_centroids[k][0] - self.ORIG_X]
            for k in sorted_keys
        ], dtype=np.float32)   # (n_c, 2) in proc metres
        self._n_c = len(self._true_cents_sim)

        # Distorted centroids (bearing hint only; identical to true when no distortion)
        self._compute_distorted_centroids()

        # Bridge geometry for recovery: axis + perp from centroid spread
        c_start = self._true_cents_sim[0]
        c_end   = self._true_cents_sim[-1]
        axis    = c_end - c_start
        ax_len  = np.linalg.norm(axis) + 1e-6
        self._bridge_axis   = (axis / ax_len).astype(np.float64)
        self._bridge_perp   = np.array([-self._bridge_axis[1], self._bridge_axis[0]])
        self._bridge_center = self._true_cents_sim[len(self._true_cents_sim) // 2].astype(np.float64)

        self._occ_grid = _LocalOccGrid()
        self._mpc_K  = K
        self._mpc_N  = N
        self._mpc_dt = dt
        self._mpc_sr = sr
        self.latest_ranges_msg           = None
        self.latest_predicted_cluster_id = None
        self.get_logger().info(
            f"SamplingMPC ready: K={K} N={N} dt={dt} sr={sr} n_c={self._n_c}")

    def _compute_distorted_centroids(self):
        """Apply rotation/scale/translation to true centroids for bearing hint."""
        angle = math.radians(
            self.get_parameter("distortion_angle_deg").get_parameter_value().double_value)
        scale = self.get_parameter("distortion_scale").get_parameter_value().double_value
        tx    = self.get_parameter("distortion_tx").get_parameter_value().double_value
        ty    = self.get_parameter("distortion_ty").get_parameter_value().double_value
        pivot = self._true_cents_sim[len(self._true_cents_sim) // 2].astype(float)
        ca, sa = math.cos(angle), math.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        self._dist_cents_sim = np.array([
            R @ ((c.astype(float) - pivot) * scale) + pivot + np.array([tx, ty])
            for c in self._true_cents_sim
        ], dtype=np.float32)

    def _load_plan_and_cluster_data(self):
        plan_path = os.path.join(os.path.dirname(__file__), "plans", self.plan_file)
        if not os.path.exists(plan_path):
            self.get_logger().error(f"Plan file not found: {plan_path}")
            return [], {}, {}
        with open(plan_path) as f:
            data = json.load(f)
        return (data.get("plan_sequence", []),
                data.get("bearing_map", {}),
                data.get("centroids", {}))

    def _init_ros(self):
        self.ranges_sub = self.create_subscription(
            Float32MultiArray, "/processed_ranges",
            self.ranges_callback, qos_profile=qos_profile_sensor_data)
        self.predicted_cluster_sub = self.create_subscription(
            Int16, "/predicted_cluster",
            self.predicted_cluster_callback, 10)

        self.scores_pub   = self.create_publisher(
            Float32MultiArray, "/sampling_mpc_scores",        10)
        self.rollout_pub  = self.create_publisher(
            Float32MultiArray, "/sampling_mpc_best_rollout",   10)
        self.guidance_pub = self.create_publisher(
            Float32MultiArray, "/sampling_mpc_guidance_debug", 10)

        dt_sec = self.get_parameter("sampling_mpc_dt").get_parameter_value().double_value
        self.timer = self.create_timer(dt_sec, self.control_loop)

        _log_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(_log_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_log_path = os.path.join(_log_dir, f"smpc_carla_{_ts}.jsonl")
        self._debug_log_file = open(self._debug_log_path, "w")

    def _init_carla(self):
        self.carla_client = carla.Client("localhost", 2000)
        self.carla_client.set_timeout(10.0)
        self.world        = self.carla_client.get_world()
        self.ego_vehicle  = None
        self.spectator    = self.world.get_spectator()
        self.is_vehicle_ready = False
        self.is_first_run     = True

    # ── ROS callbacks ──────────────────────────────────────────────────────────

    def ranges_callback(self, msg: Float32MultiArray):
        self.latest_ranges_msg = msg

    def predicted_cluster_callback(self, msg: Int16):
        self.latest_predicted_cluster_id = msg.data

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def _carla_to_sim(self, cx: float, cy: float) -> Tuple[float, float]:
        """CARLA world (x, y) → proc frame physical metres."""
        proc_x = -cy - self.ORIG_Y
        proc_y =  cx - self.ORIG_X
        return proc_x, proc_y

    def _get_vehicle_state(self) -> Tuple[float, float, float]:
        """Return (proc_x, proc_y, sim_yaw_rad) from current CARLA transform."""
        t = self.ego_vehicle.get_transform()
        proc_x, proc_y = self._carla_to_sim(t.location.x, t.location.y)
        sim_yaw = -math.radians(t.rotation.yaw)
        return proc_x, proc_y, sim_yaw

    # ── Control loop ───────────────────────────────────────────────────────────

    def control_loop(self):
        if not self.is_vehicle_ready:
            actors   = self.world.get_actors()
            vehicles = actors.filter("*vehicle*")
            if vehicles:
                self.ego_vehicle      = vehicles[0]
                self.is_vehicle_ready = True
                self.ego_vehicle.set_simulate_physics(True)
                self.get_logger().info("Ego vehicle found. Physics enabled.")
            else:
                self.get_logger().info(
                    "Waiting for ego vehicle...", throttle_duration_sec=3.0)
                return

        if self.is_first_run:
            if not self._initial_teleport():
                return
            self.is_first_run = False

        if self.current_plan_step_index >= len(self.plan_sequence):
            self._handle_plan_complete()
            return

        current_step   = self.plan_sequence[self.current_plan_step_index]
        expected_start = current_step["start"]
        expected_next  = current_step["next"]

        if self.latest_ranges_msg is None:
            self.get_logger().warning(
                "Waiting for /processed_ranges", throttle_duration_sec=3.0)
            return

        debug_mode = self.get_parameter("debug_mode").get_parameter_value().bool_value
        if debug_mode:
            raw_cluster = self.get_parameter("current_cluster_id") \
                              .get_parameter_value().integer_value
        else:
            if self.latest_predicted_cluster_id is None:
                self.get_logger().warning(
                    "Waiting for /predicted_cluster", throttle_duration_sec=3.0)
                return
            raw_cluster = self.latest_predicted_cluster_id

        mapped_cluster = self._map_cluster_id(raw_cluster)

        if mapped_cluster == expected_next:
            self.current_plan_step_index += 1
            self.get_logger().info(
                f"Plan advanced {expected_start}→{expected_next} "
                f"(step {self.current_plan_step_index}/{len(self.plan_sequence)})")
            if self.current_plan_step_index >= len(self.plan_sequence):
                self._handle_plan_complete()
                return
            current_step   = self.plan_sequence[self.current_plan_step_index]
            expected_start = current_step["start"]
            expected_next  = current_step["next"]

        sim_x, sim_y, sim_yaw = self._get_vehicle_state()
        sim_pos = np.array([sim_x, sim_y])

        forbidden = [c for c in range(self._n_c)
                     if c not in (expected_start, expected_next)]
        (v_cmd, om_cmd, guidance_ratio, cp, bp,
         edt_blocked, best_rollout, recovery, d_min, n_hits) = \
            self._run_sampling_mpc_step(
                sim_pos, sim_yaw,
                self.latest_ranges_msg,
                expected_start, expected_next, forbidden)

        dry = self.get_parameter("dry_run").get_parameter_value().bool_value
        if not dry:
            self._apply_action(v_cmd, om_cmd, sim_yaw)

        spec_tf = carla.Transform(
            self.ego_vehicle.get_transform().transform(carla.Location(x=-4, z=50)),
            carla.Rotation(yaw=-180, pitch=-90))
        self.spectator.set_transform(spec_tf)

        gd = Float32MultiArray()
        gd.data = [guidance_ratio, cp, bp, edt_blocked, v_cmd, om_cmd]
        self.guidance_pub.publish(gd)

        br = Float32MultiArray()
        br.data = best_rollout.flatten().tolist()
        self.rollout_pub.publish(br)

        self._write_debug_log(sim_x, sim_y, sim_yaw, expected_start, expected_next,
                              v_cmd, om_cmd, guidance_ratio, cp, bp, edt_blocked,
                              recovery, d_min, n_hits, raw_cluster, mapped_cluster)

    def _initial_teleport(self) -> bool:
        if not self.plan_sequence:
            self.get_logger().error("Empty plan sequence.")
            return False
        step0          = self.plan_sequence[0]
        start_key      = str(step0["start"])
        next_key       = str(step0["next"])
        angular_offset = self.get_parameter("angular_offset_deg") \
                             .get_parameter_value().double_value

        if start_key not in self.cluster_centroids:
            self.get_logger().error(f"Start centroid '{start_key}' not in plan.")
            return False

        centroid    = self.cluster_centroids[start_key]
        plan_key    = f"{start_key}-{next_key}"
        bearing_rad = self.bearing_map.get(plan_key, 0.0)
        bearing_deg = math.degrees(bearing_rad) + angular_offset

        # centroid[0]=carla_x, centroid[1]=-carla_y → real CARLA y = -centroid[1]
        init_loc = carla.Location(
            x=float(centroid[0]),
            y=float(-1.0 * centroid[1]),
            z=self.ego_vehicle.get_transform().location.z)
        init_rot = carla.Rotation(pitch=0, yaw=-1.0 * bearing_deg, roll=0)
        self.get_logger().info(
            f"Initial teleport: CARLA ({init_loc.x:.1f}, {init_loc.y:.1f}) "
            f"bearing={bearing_deg:.1f}°")
        self.ego_vehicle.set_simulate_physics(False)
        self.ego_vehicle.set_transform(carla.Transform(init_loc, init_rot))
        self.ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        time.sleep(0.1)
        self.ego_vehicle.set_simulate_physics(True)
        return True

    def _handle_plan_complete(self):
        self.get_logger().info("Plan complete. Stopping vehicle.")
        self.ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        self.timer.cancel()

    # ── SamplingMPC inference ──────────────────────────────────────────────────

    def _run_sampling_mpc_step(
        self,
        sim_pos:    np.ndarray,
        sim_yaw:    float,
        ranges_msg: Float32MultiArray,
        start_id:   int,
        target_id:  int,
        forbidden:  list,
    ):
        """Returns (v_cmd, om_cmd, guidance_ratio, cluster_score, bearing_score,
                    edt_frac, best_rollout, recovery, d_min, n_lidar_hits)."""
        # ── LiDAR hits in body frame (physical metres, no SCALE) ───────────────
        ranges = np.array(ranges_msg.data, dtype=np.float32)
        n_bins = len(ranges)
        angles = np.linspace(0.0, 2 * math.pi, n_bins, endpoint=False)
        # CARLA body frame: x=right, y=forward → SamplingMPC: x=forward, y=left
        x_carla = ranges * np.sin(angles)
        y_carla = ranges * np.cos(angles)
        hits_local = np.stack([y_carla, -x_carla], axis=1).astype(np.float32)

        max_r  = float(ranges.max())
        mask   = ranges < max_r * 0.99       # drop max-range misses
        hits_local = hits_local[mask]
        if len(hits_local) > 0:
            dists      = np.linalg.norm(hits_local, axis=1)
            hits_local = hits_local[dists > 0.1]   # drop artefacts < 10 cm
        hits_in = hits_local if len(hits_local) > 0 else None
        n_hits  = len(hits_in) if hits_in is not None else 0

        # ── Generate K rollouts ────────────────────────────────────────────────
        K, N = self._mpc_K, self._mpc_N
        dt   = self._mpc_dt
        v_seqs  = np.random.uniform(0.0, 0.75, (K, N))
        om_seqs = np.random.uniform(-1.5, 1.5,  (K, N))
        ctrl    = np.stack([v_seqs, om_seqs], axis=2)         # (K, N, 2)
        rollouts   = unicycle_rollout(np.zeros(3), ctrl, dt)  # (K, N+1, 3)
        traj_local = rollouts[:, 1:, :]                       # (K, N, 3)

        # ── EDT collision mask ─────────────────────────────────────────────────
        self._occ_grid.update(hits_in)
        dist_vals = self._occ_grid.check_collisions(traj_local)
        collision = (dist_vals < self._mpc_sr).any(axis=1)

        # ── Global rollout endpoints ───────────────────────────────────────────
        cy, sy  = math.cos(sim_yaw), math.sin(sim_yaw)
        R_l2g   = np.array([[cy, -sy], [sy, cy]])
        end_local  = traj_local[:, -1, :2]
        end_global = end_local @ R_l2g.T + sim_pos

        # ── TRUE Voronoi cluster membership ────────────────────────────────────
        dt_c    = np.linalg.norm(
            end_global[:, None, :] - self._true_cents_sim[None, :, :], axis=2)
        nearest = np.argmin(dt_c, axis=1)
        in_target    = (nearest == target_id).astype(float)
        in_start     = (nearest == start_id).astype(float)
        in_forbidden = np.isin(nearest, forbidden).astype(float)
        cluster_parts = 10.0 * in_target + 1.0 * in_start - 15.0 * in_forbidden
        scores = cluster_parts.copy()

        # ── Bearing toward DISTORTED target centroid ───────────────────────────
        tgt_sim   = self._dist_cents_sim[target_id]
        tgt_local = (tgt_sim - sim_pos) @ np.array([[cy, sy], [-sy, cy]])
        bear_local = math.atan2(tgt_local[1], tgt_local[0])
        cdir       = tgt_local / (np.linalg.norm(tgt_local) + 1e-6)
        bearing_cos  = 3.0 * np.cos(traj_local[:, -1, 2] - bear_local)
        bearing_dot  = 1.0 * (
            np.cos(traj_local[:, -1, 2]) * cdir[0] +
            np.sin(traj_local[:, -1, 2]) * cdir[1])
        bearing_parts = bearing_cos + bearing_dot
        scores += bearing_parts

        # ── Progress along start→target axis ──────────────────────────────────
        axis = self._true_cents_sim[target_id] - self._true_cents_sim[start_id]
        axis = axis / (np.linalg.norm(axis) + 1e-6)
        end_along = (end_global - sim_pos) @ axis
        scores += 0.8 * np.clip(end_along, 0, None)

        scores[collision] = -np.inf
        edt_blocked = float(np.sum(collision)) / K

        # ── Recovery: steer toward bridge centre when all rollouts blocked ─────
        recovery = False
        if np.all(~np.isfinite(scores)):
            across  = float((self._bridge_center - sim_pos.astype(np.float64))
                            @ self._bridge_perp)
            om_cmd  = 0.8 * float(np.sign(across)) if abs(across) > 0.5 else 0.0
            v_cmd   = 0.15
            best_k  = 0
            best_cp = 0.0
            best_bp = 0.0
            recovery = True
        else:
            best_k  = int(np.argmax(scores))
            v_cmd   = float(ctrl[best_k, 0, 0])
            om_cmd  = float(ctrl[best_k, 0, 1])
            best_cp = float(cluster_parts[best_k])
            best_bp = float(bearing_parts[best_k])

        # ── Publish top-10 rollout scores ──────────────────────────────────────
        finite_mask = np.isfinite(scores)
        if finite_mask.any():
            fs   = scores[finite_mask]
            top_n = min(10, len(fs))
            top10 = np.sort(np.partition(fs, -top_n)[-top_n:])[::-1]
        else:
            top10 = np.array([], dtype=np.float32)
        sc_msg = Float32MultiArray()
        sc_msg.data = top10.tolist()
        self.scores_pub.publish(sc_msg)

        # ── CBF post-filter ────────────────────────────────────────────────────
        d_min = math.inf
        if hits_in is not None and len(hits_in) > 0:
            dists  = np.linalg.norm(hits_in, axis=1)
            d_min  = float(np.min(dists))
            near_h = hits_in[int(np.argmin(dists))]
            cos_a  = math.cos(math.atan2(near_h[1], near_h[0]))
            d_safe, alpha = 0.5, 2.0   # physical metres
            h = d_min - d_safe
            if cos_a > 1e-3 and v_cmd * cos_a > alpha * h:
                v_cmd = max(0.0, alpha * h / cos_a)
            if d_min < 2.0 * d_safe and cos_a > 0.3:
                sin_a  = math.sin(math.atan2(near_h[1], near_h[0]))
                om_cmd = float(np.clip(
                    om_cmd - 2.0 * alpha * sin_a * (1.0 - h / d_safe), -1.5, 1.5))

        eps = 1e-6
        guidance_ratio = abs(best_cp) / (abs(best_cp) + abs(best_bp) + eps)

        self.get_logger().info(
            f"[smpc] pos=({sim_pos[0]:.1f},{sim_pos[1]:.1f}) "
            f"tgt={target_id} v={v_cmd:.3f} ω={om_cmd:.3f} "
            f"edt={edt_blocked:.2f} d_min={d_min:.2f} "
            f"ratio={guidance_ratio:.2f} recovery={recovery}",
            throttle_duration_sec=0.5)

        return (v_cmd, om_cmd, guidance_ratio, best_cp, best_bp,
                edt_blocked, rollouts[best_k, :, :].copy(),
                recovery, d_min, n_hits)

    def _apply_action(self, v_cmd: float, om_cmd: float, sim_yaw: float):
        """Drive CARLA vehicle with real physics.

        v_cmd ∈ [0, 0.75] (MPC velocity units) → clamped to 1.0 m/s max in CARLA.
        om_cmd in rad/s (CCW positive in sim) → CARLA CW: negate.
        """
        CARLA_MAX_VEL = 1.0
        v_real        = float(np.clip(v_cmd * (CARLA_MAX_VEL / 0.75), 0.0, CARLA_MAX_VEL))
        carla_yaw_rad = -sim_yaw
        vx_world      = v_real * math.cos(carla_yaw_rad)
        vy_world      = v_real * math.sin(carla_yaw_rad)
        self.ego_vehicle.set_target_velocity(
            carla.Vector3D(vx_world, vy_world, 0.0))
        self.ego_vehicle.set_target_angular_velocity(
            carla.Vector3D(0.0, 0.0, -math.degrees(om_cmd)))

    def _map_cluster_id(self, cluster_id: int) -> int:
        """Map raw perception cluster ID → plan cluster ID."""
        if cluster_id in [0, 1]:
            return 1
        elif cluster_id in [-1, 4]:
            return 2
        elif cluster_id in [2, 3]:
            return 3
        elif cluster_id in [5, 6, 7, 8, 9]:
            return 0
        return cluster_id

    def _write_debug_log(self, sx, sy, syaw, s_id, t_id,
                         v_cmd, om_cmd, gr, cp, bp, edt,
                         recovery, d_min, n_hits, raw_cluster, mapped_cluster):
        record = {
            "t":                time.time(),
            "sim_x":            sx,
            "sim_y":            sy,
            "sim_yaw_deg":      math.degrees(syaw),
            "plan_start":       s_id,
            "plan_next":        t_id,
            "plan_step":        self.current_plan_step_index,
            "v_cmd":            v_cmd,
            "om_cmd":           om_cmd,
            "guidance_ratio":   gr,
            "cluster_score":    cp,
            "bearing_score":    bp,
            "edt_blocked":      edt,
            "recovery":         recovery,
            "d_min":            d_min if math.isfinite(d_min) else -1.0,
            "n_lidar_hits":     n_hits,
            "raw_cluster_id":   raw_cluster,
            "mapped_cluster_id": mapped_cluster,
        }
        self._debug_log_file.write(json.dumps(record) + "\n")
        self._debug_log_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = SamplingMPCCarlaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
