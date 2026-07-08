"""
spot_mpc_node.py
=================
ROS2 node that drives a real Boston Dynamics Spot down a hallway using a
sampling-based MPC (topological/Voronoi-guided, EDT-based collision
avoidance). Connects to Spot directly via the bosdyn SDK.

All positions/distances are real metres — there is no sim-to-real scale
factor here (that only ever mattered for the DGPPO neural-net policy this
node replaces, which needed inputs matching its training distribution;
SamplingMPC's rollouts, safety radius, and LiDAR hits are already
calibrated in real units, so introducing a scale factor just for the plan
centroids was a latent unit-mismatch bug, not a real requirement).

Plan centroids are expressed in a "plan frame" that starts out identical to
Spot's vision frame, but can be re-zeroed at any time via /mpc_reset_origin
— call it with Spot at a known pose (e.g. facing straight down a hallway)
to make that pose (0, 0, yaw=0) the new reference. From there, a centroid
is just the real (forward, left) distance in metres from that pose to that
waypoint — no scale factor, no need to know Spot's boot-time frame.

Write real waypoint positions (e.g. the actual hallway endpoint), not
placeholders — the Voronoi cluster term (in_target/in_start/in_forbidden)
needs real positions to mean anything, and it's what lets the controller
tell "still approaching" from "arrived" and penalize wandering into a
forbidden region on a multi-segment plan. Positions don't need to be
survey-precise: per the upstream SamplingMPC design (~/new_dgppo/dgppo/env/
sampling_mpc.py), the centroid-direction term is explicitly "robust to
metric inaccuracies in centroid placement" — rough map locations are fine.

There is no BehaviorAssociator/"bridges" geometry in this node — that was
only ever a synthetic-corridor proxy for simulation and this node's actual
scoring doesn't read anything derived from it (confirmed by tracing
SamplingMPC.reset()/plan()). Real deployment has three real inputs per
plan.json: (1) the cluster sequence (plan_sequence), (2) each cluster's
rough location (centroids), and (3) the bearing between adjacent cluster
pairs (bearing_map, in the plan frame — see /mpc_reset_origin above) —
bearing is supplied explicitly rather than derived from centroid
positions, matching SamplingMPC's own real-robot scoring design.

Published topics:
  /mpc_state           Float32MultiArray
      [pos_x, pos_y, vel_x, vel_y, v_cmd, om_cmd, yaw, start_cluster, target_cluster]
      (position/velocity in the plan frame, i.e. relative to the last /mpc_reset_origin)
  /mpc_plan_step       Int32   — index into the plan sequence
  /mpc_best_rollout    Float32MultiArray  — flattened N×3 best candidate path (body frame)
  /mpc_guidance_debug  Float32MultiArray  — [guidance_ratio, cluster_score, bearing_score, edt_blocked_frac]
  /mpc_scores          Float32MultiArray  — top-10 rollout scores, descending

Subscribed topics:
  /processed_ranges    Float32MultiArray  — pre-processed LiDAR range bins
  /predicted_cluster   Int16              — live cluster classifier output
  /current_terrain     Int16              — terrain type

Services:
  /mpc_take_lease      std_srvs/Trigger — reclaim the Spot lease if the tablet took it
  /mpc_reset_origin    std_srvs/Trigger — re-zero the plan frame to Spot's current pose
"""

import datetime
import json
import math
import os
import threading
import time
from typing import NamedTuple, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Int16, Int32, Float32MultiArray
from std_srvs.srv import Trigger

try:
    import casadi as ca
    _CASADI_OK = True
except ImportError:
    _CASADI_OK = False

import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.exceptions import LeaseUseError
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient

from .plan import map_cluster_id
from .spot_utils import get_spot_state

# ── Inlined rollout + collision helpers (no external deps) ────────────────────

def unicycle_rollout(state0, ctrl, dt):
    K, N = ctrl.shape[:2]
    traj = np.zeros((K, N + 1, 3), dtype=np.float32)
    traj[:, 0, :] = state0
    for n in range(N):
        x, y, th = traj[:, n, 0], traj[:, n, 1], traj[:, n, 2]
        v, om = ctrl[:, n, 0], ctrl[:, n, 1]
        traj[:, n + 1, 0] = x + v * np.cos(th) * dt
        traj[:, n + 1, 1] = y + v * np.sin(th) * dt
        traj[:, n + 1, 2] = th + om * dt
    return traj


class _LocalOccGrid:
    def __init__(self):
        self._hits = np.empty((0, 2), dtype=np.float32)

    def update(self, hits):
        if hits is None or len(hits) == 0:
            self._hits = np.empty((0, 2), dtype=np.float32)
        else:
            self._hits = np.asarray(hits, dtype=np.float32)

    def check_collisions(self, traj):
        if len(self._hits) == 0:
            return np.full(traj.shape[:2], np.inf, dtype=np.float32)
        pts = traj[:, :, :2]
        diff = pts[:, :, None, :] - self._hits[None, None, :, :]
        return np.linalg.norm(diff, axis=-1).min(axis=-1).astype(np.float32)


# ── Carson NMPC (CasADi/IPOPT) ────────────────────────────────────────────────
# Adapted from Carson's waypoint_follow.py.  Only built when casadi is available.
# State: [x, y, yaw, v, omega] in plan frame.
# Control: [a_v, a_omega] (accelerations, not velocities directly).

class _CarsonMPC:
    """CasADi/IPOPT NMPC for point-to-point navigation with soft obstacle avoidance."""

    MAX_OBS = 8

    def __init__(self, N=8, dt=0.2, max_v=0.75, max_omega=1.5,
                 max_acc=1.0, max_yaw_acc=2.5, ipopt_max_iter=50):
        self.N, self.dt = N, dt
        self.max_v, self.max_omega = max_v, max_omega
        self.max_acc, self.max_yaw_acc = max_acc, max_yaw_acc
        self._prev_X = self._prev_U = None
        self._build(ipopt_max_iter)

    def _build(self, ipopt_max_iter):
        N, dt = self.N, self.dt
        n_obs = self.MAX_OBS
        opti = ca.Opti()

        X = opti.variable(5, N + 1)          # [x, y, yaw, v, omega] per step
        U = opti.variable(2, N)               # [a_v, a_omega] per step
        S = opti.variable(n_obs, N + 1)       # obstacle slack (>= 0)
        P_x0   = opti.parameter(5)
        P_goal = opti.parameter(3)            # [goal_x, goal_y, goal_yaw]
        P_obs  = opti.parameter(3, n_obs)     # [ox, oy, r] per obstacle

        opti.subject_to(X[:, 0] == P_x0)
        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            opti.subject_to(X[:, k + 1] == ca.vertcat(
                xk[0] + xk[3] * ca.cos(xk[2]) * dt,
                xk[1] + xk[3] * ca.sin(xk[2]) * dt,
                xk[2] + xk[4] * dt,
                xk[3] + uk[0] * dt,
                xk[4] + uk[1] * dt,
            ))
            opti.subject_to(opti.bounded(-self.max_acc,     U[0, k], self.max_acc))
            opti.subject_to(opti.bounded(-self.max_yaw_acc, U[1, k], self.max_yaw_acc))
            opti.subject_to(opti.bounded(0.0, X[3, k + 1], self.max_v))
            opti.subject_to(opti.bounded(-self.max_omega, X[4, k + 1], self.max_omega))

        opti.subject_to(ca.vec(S) >= 0.0)
        for k in range(N + 1):
            for j in range(n_obs):
                dx = X[0, k] - P_obs[0, j]
                dy = X[1, k] - P_obs[1, j]
                opti.subject_to(
                    ca.sqrt(dx * dx + dy * dy + 1e-9) + S[j, k] >= P_obs[2, j])

        obj = 0
        for k in range(1, N + 1):
            scale = 2.0 if k == N else 1.0
            obj += scale * 500.0 * ((X[0, k] - P_goal[0]) ** 2 + (X[1, k] - P_goal[1]) ** 2)
            obj += scale * 30.0  * (X[2, k] - P_goal[2]) ** 2
        obj += 30.0 * X[3, N] ** 2 + 30.0 * X[4, N] ** 2   # slow to stop at goal
        for k in range(N):
            obj += 0.1 * U[0, k] ** 2 + 10.0 * U[1, k] ** 2
        obj += 1e5 * ca.sumsqr(S)
        opti.minimize(obj)

        opti.solver('ipopt', {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.max_iter': ipopt_max_iter, 'ipopt.tol': 2e-3,
            'ipopt.acceptable_tol': 1e-2, 'ipopt.acceptable_iter': 1,
            'ipopt.constr_viol_tol': 1e-4, 'ipopt.compl_inf_tol': 1e-2,
        })
        self._opti = opti
        self._X, self._U, self._S = X, U, S
        self._P_x0, self._P_goal, self._P_obs = P_x0, P_goal, P_obs

    def solve(self, x0, goal_xyz, obstacles):
        """
        x0:         (5,) [x, y, yaw, v, omega] — plan frame
        goal_xyz:   (3,) [goal_x, goal_y, goal_yaw] — plan frame
        obstacles:  list of (ox, oy, r) — plan frame, already includes safety margin
        Returns (v_next, omega_next, X_pred (N+1,5)) or None on solver failure.
        """
        obs_p = np.zeros((3, self.MAX_OBS)); obs_p[0, :] = 1e4
        for j, (ox, oy, r) in enumerate(obstacles[:self.MAX_OBS]):
            obs_p[:, j] = [ox, oy, r]

        self._opti.set_value(self._P_x0,  x0)
        self._opti.set_value(self._P_goal, goal_xyz)
        self._opti.set_value(self._P_obs,  obs_p)

        N = self.N
        if self._prev_X is not None and self._prev_X.shape == (N + 1, 5):
            self._opti.set_initial(self._X, self._prev_X.T)
            self._opti.set_initial(self._U, self._prev_U)
        else:
            self._opti.set_initial(self._X, np.tile(x0, (N + 1, 1)).T)
            self._opti.set_initial(self._U, np.zeros((2, N)))
        self._opti.set_initial(self._S, np.zeros((self.MAX_OBS, N + 1)))

        try:
            sol = self._opti.solve()
        except Exception:
            return None

        X_pred = np.array(sol.value(self._X)).T   # (N+1, 5)
        U_val  = np.array(sol.value(self._U))      # (2, N)
        self._prev_X, self._prev_U = X_pred, U_val

        v_next     = float(np.clip(x0[3] + U_val[0, 0] * self.dt, 0.0,          self.max_v))
        omega_next = float(np.clip(x0[4] + U_val[1, 0] * self.dt, -self.max_omega, self.max_omega))
        return v_next, omega_next, X_pred

    def reset(self):
        self._prev_X = self._prev_U = None


# ══════════════════════════════════════════════════════════════════════════════

class SpotState(NamedTuple):
    x: float; y: float; vx: float; vy: float; yaw: float
    raw_ranges: np.ndarray    # (n_bins,) LiDAR ranges in metres


class MPCResult(NamedTuple):
    v_cmd:          float
    om_cmd:         float
    guidance_ratio: float     # cluster_score / (|cluster| + |bearing| + eps)
    cluster_score:  float
    bearing_score:  float
    edt_blocked:    float     # fraction of rollouts blocked by EDT
    best_rollout:   np.ndarray  # (N, 3) best rollout in body frame
    top_scores:     np.ndarray  # (<=10,) top finite rollout scores, descending


class SpotMPCNode(Node):
    def __init__(self):
        super().__init__("spot_mpc_node")
        self.get_logger().info("Initializing Spot MPC Node...")
        self._init_params()
        self._init_model()
        self._init_ros()
        if self._no_spot:
            self.get_logger().warn(
                "no_spot:=true — skipping bosdyn connection. Pose is stubbed "
                "at the plan-frame origin (0,0,0); commands are never sent, "
                "only logged. Use this to bench-test perception→MPC without Spot.")
            self._init_spot_stub()
        else:
            self._init_spot()
        self.get_logger().info("Spot MPC Node fully initialized.")

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_params(self):
        self.declare_parameter("debug_mode", False)
        self.declare_parameter("current_cluster_id", 0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("no_spot", False)
        self._no_spot = self.get_parameter("no_spot").get_parameter_value().bool_value
        self.declare_parameter("sampling_mpc_K", 500)
        self.declare_parameter("sampling_mpc_N", 8)
        self.declare_parameter("sampling_mpc_safety_radius", 0.06)
        self.declare_parameter("sampling_mpc_dt", 0.2)
        self.declare_parameter("bearing_only", False)
        self.declare_parameter("goto_point", False)
        self.declare_parameter("goal_x", 0.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("use_carson", False)
        self._last_omega_cmd = 0.0   # tracked for Carson NMPC initial state
        self.n_rays_phys = 72
        self._origin_file = os.path.join(
            os.path.dirname(__file__), "plans", "origin.json")
        # Plan-frame origin — persisted to origin.json across restarts.
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._origin_yaw = 0.0
        self._load_origin()

    def _init_model(self):
        """Load the plan (plan_sequence, bearing_map, centroids) and build
        SamplingMPC. No BehaviorAssociator/bridges — see module docstring."""
        plan_path = os.path.join(os.path.dirname(__file__), "plans", "hallway.json")
        with open(plan_path) as f:
            plan_data = json.load(f)
        self.plan_sequence = plan_data.get("plan_sequence", [])
        self.bearing_map = plan_data.get("bearing_map", {})
        self.cluster_centroids = plan_data.get("centroids", {})
        self.current_plan_step_index = 0

        K = self.get_parameter("sampling_mpc_K").get_parameter_value().integer_value
        N = self.get_parameter("sampling_mpc_N").get_parameter_value().integer_value
        dt = self.get_parameter("sampling_mpc_dt").get_parameter_value().double_value
        sr = self.get_parameter("sampling_mpc_safety_radius").get_parameter_value().double_value

        # centroids is {"0": [x,y,...], "1": [x,y,...], ...}; keep only x,y.
        # These are rough map locations in the plan frame (see /mpc_reset_origin) —
        # not survey-precise, only used for Voronoi cluster identity and a soft
        # directional nudge; see _run_sampling_mpc_step for how bearing (the
        # primary steering signal) is sourced from bearing_map instead.
        cent_array = np.array([
            self.cluster_centroids[str(k)][:2]
            for k in sorted(int(k) for k in self.cluster_centroids)
        ], dtype=np.float32)
        self._true_cents = cent_array   # (n_c, 2)

        self._mpc_K = K
        self._mpc_N = N
        self._mpc_dt = dt
        self._mpc_sr = sr
        self._occ_grid = _LocalOccGrid()

        # Carson NMPC — only built if casadi is available and use_carson requested
        self._carson = None
        if self.get_parameter("use_carson").get_parameter_value().bool_value:
            if not _CASADI_OK:
                self.get_logger().error(
                    "use_carson:=true but casadi is not installed. "
                    "Run: pip3 install casadi")
            else:
                self._carson = _CarsonMPC(N=N, dt=dt)
                self.get_logger().info("CarsonNMPC (CasADi/IPOPT) initialised.")

        # Current state variables
        self.latest_ranges_msg = None
        self.latest_predicted_cluster_id = None
        self.is_first_run = True
        self._start_id = None
        self._target_id = None
        self._forbidden = []
        self._n_c = len(cent_array)
        self.get_logger().info(
            f"SamplingMPC ready: K={K} N={N} dt={dt} sr={sr}  "
            f"n_clusters={self._n_c}"
        )

    def _init_ros(self):
        self.ranges_sub = self.create_subscription(
            Float32MultiArray, "/processed_ranges",
            self.ranges_callback, qos_profile=qos_profile_sensor_data)
        self.predicted_cluster_sub = self.create_subscription(
            Int16, "/predicted_cluster", self.predicted_cluster_callback, 10)
        self.terrain_sub = self.create_subscription(
            Int16, "/current_terrain", self.terrain_callback, 10)

        self.state_pub = self.create_publisher(Float32MultiArray, "/mpc_state", 10)
        self.plan_step_pub = self.create_publisher(Int32, "/mpc_plan_step", 10)
        self.rollout_pub = self.create_publisher(Float32MultiArray, "/mpc_best_rollout", 10)
        self.guidance_pub = self.create_publisher(Float32MultiArray, "/mpc_guidance_debug", 10)
        self.scores_pub = self.create_publisher(Float32MultiArray, "/mpc_scores", 10)

        self._take_lease_srv = self.create_service(
            Trigger, "/mpc_take_lease", self._take_lease_callback)
        self._reset_origin_srv = self.create_service(
            Trigger, "/mpc_reset_origin", self._reset_origin_callback)

        dt_sec = self.get_parameter("sampling_mpc_dt").get_parameter_value().double_value
        self.timer = self.create_timer(dt_sec, self.control_loop)

        _log_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(_log_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._debug_log_path = os.path.join(_log_dir, f"mpc_run_{_ts}.jsonl")
        self._debug_log_file = open(self._debug_log_path, "w")
        self.get_logger().info(f"Debug log: {self._debug_log_path}")
        self.latest_terrain_id = 1

    def _init_spot(self):
        self.get_logger().info("Initializing Spot robot.")
        self.sdk = bosdyn.client.create_standard_sdk("spot-mpc")
        self.robot = self.sdk.create_robot("10.0.0.3")
        self.robot.authenticate(username="user", password="pass")
        self.robot.time_sync.wait_for_sync()
        self.state_client = self.robot.ensure_client("robot-state")
        self.lease_client = self.robot.ensure_client("lease")
        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name)
        self.lease_client.take()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)

        self._cached_robot_state = None
        self._state_lock = threading.Lock()
        threading.Thread(target=self._poll_spot_state, daemon=True).start()

        self._cmd_vel = (0.0, 0.0, 0.0)   # (vx, vy, v_rot)
        self._cmd_lock = threading.Lock()
        threading.Thread(target=self._send_commands, daemon=True).start()
        self._tablet_has_lease = False

    def _init_spot_stub(self):
        """No bosdyn connection at all — for bench-testing perception/MPC
        computation without Spot powered on. Pose is fixed at the plan-frame
        origin (moving the robot has no effect on the stubbed pose)."""
        self.lease_keep_alive = None
        self._cached_robot_state = None
        self._state_lock = threading.Lock()
        self._cmd_vel = (0.0, 0.0, 0.0)
        self._cmd_lock = threading.Lock()
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
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=vx, v_y=vy, v_rot=v_rot,
            body_height=0.0, locomotion_hint=1)
        self.command_client.robot_command(
            command=cmd, end_time_secs=time.time() + 0.5)

    def _get_spot_state(self) -> Tuple:
        if self._no_spot:
            return 0.0, 0.0, 0.0, 0.0, 0.0
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
        if self._no_spot:
            response.success = False; response.message = "no_spot mode: no real Spot connection."
            return response
        try:
            self.lease_keep_alive.shutdown()
            self.lease_client.take()
            self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)
            self._tablet_has_lease = False
            response.success = True; response.message = "Lease reclaimed."
        except Exception as e:
            response.success = False; response.message = str(e)
        return response

    def _load_origin(self):
        try:
            with open(self._origin_file) as f:
                d = json.load(f)
            self._origin_x = float(d["x"])
            self._origin_y = float(d["y"])
            self._origin_yaw = float(d["yaw"])
            self.get_logger().info(
                f"Loaded saved origin: ({self._origin_x:.2f}, {self._origin_y:.2f}), "
                f"yaw={math.degrees(self._origin_yaw):.1f}°")
        except FileNotFoundError:
            pass  # no saved origin yet — start at (0,0,0)
        except Exception as e:
            self.get_logger().warn(f"Could not load origin.json: {e}")

    def _save_origin(self):
        try:
            with open(self._origin_file, "w") as f:
                json.dump({"x": self._origin_x,
                           "y": self._origin_y,
                           "yaw": self._origin_yaw}, f)
        except Exception as e:
            self.get_logger().warn(f"Could not save origin.json: {e}")

    def _reset_origin_callback(self, request, response):
        """Re-zero the plan frame to Spot's current pose (position + yaw)."""
        try:
            x, y, _vx, _vy, yaw = self._get_spot_state()
            self._origin_x, self._origin_y, self._origin_yaw = x, y, yaw
            self._save_origin()
            response.success = True
            response.message = (
                f"Origin reset to vision-frame ({x:.2f}, {y:.2f}), "
                f"yaw={math.degrees(yaw):.1f}°"
            )
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False; response.message = str(e)
        return response

    def _to_plan_frame(self, x: float, y: float, yaw: float):
        """Transform a vision-frame pose into the plan frame established by
        the last /mpc_reset_origin call (identity if never called)."""
        dx, dy = x - self._origin_x, y - self._origin_y
        c, s = math.cos(-self._origin_yaw), math.sin(-self._origin_yaw)
        px = c * dx - s * dy
        py = s * dx + c * dy
        return px, py, yaw - self._origin_yaw

    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.current_plan_step_index >= len(self.plan_sequence):
            self._handle_plan_complete()
            return

        current_plan_step = self.plan_sequence[self.current_plan_step_index]
        expected_start = current_plan_step["start"]
        expected_next = current_plan_step["next"]
        debug_mode = self.get_parameter("debug_mode").get_parameter_value().bool_value

        if not self._check_topics(debug_mode):
            return

        if self._tablet_has_lease:
            self.get_logger().info(
                "Tablet holds lease — paused. "
                "Reclaim: ros2 service call /mpc_take_lease std_srvs/srv/Trigger '{}'",
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
            expected_start = current_plan_step["start"]
            expected_next = current_plan_step["next"]

        self._start_id = expected_start
        self._target_id = expected_next
        self._forbidden = [c for c in range(self._n_c)
                           if c not in (expected_start, expected_next)]

        state = self._update_spot_state()
        if self._carson is not None:
            result = self._run_carson_mpc_step(state)
        else:
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

    def _update_spot_state(self) -> SpotState:
        raw_ranges = np.array(self.latest_ranges_msg.data, dtype=np.float32)
        x, y, vx, vy, yaw = self._get_spot_state()
        px, py, pyaw = self._to_plan_frame(x, y, yaw)
        c, s = math.cos(-self._origin_yaw), math.sin(-self._origin_yaw)
        pvx, pvy = c * vx - s * vy, s * vx + c * vy
        return SpotState(x=px, y=py, vx=pvx, vy=pvy, yaw=pyaw, raw_ranges=raw_ranges)

    def _run_carson_mpc_step(self, state: SpotState) -> MPCResult:
        """Run Carson's CasADi/IPOPT NMPC toward goal_x/goal_y (plan frame)."""
        N = self._mpc_N
        goal_x = self.get_parameter("goal_x").get_parameter_value().double_value
        goal_y = self.get_parameter("goal_y").get_parameter_value().double_value

        # Initial state: [x, y, yaw, v, omega] — plan frame
        v_est = math.sqrt(state.vx ** 2 + state.vy ** 2)
        x0 = np.array([state.x, state.y, state.yaw, v_est, self._last_omega_cmd])

        # Goal yaw: point toward goal; hold current heading when already there
        dx, dy = goal_x - state.x, goal_y - state.y
        dist = math.sqrt(dx * dx + dy * dy)
        goal_yaw = math.atan2(dy, dx) if dist > 0.1 else state.yaw
        goal_xyz = np.array([goal_x, goal_y, goal_yaw])

        # Obstacles: LiDAR hits converted to plan frame, closest MAX_OBS as discs
        ranges = state.raw_ranges
        n_bins = len(ranges)
        angles = np.linspace(0.0, 2 * math.pi, n_bins, endpoint=False)
        hits_local = np.stack(
            [ranges * np.cos(angles), ranges * np.sin(angles)], axis=1
        ).astype(np.float32)
        max_r = float(ranges.max())
        hits_local = hits_local[ranges < max_r * 0.99]
        if len(hits_local) > 0:
            hits_local = hits_local[np.linalg.norm(hits_local, axis=1) > 0.005]

        obstacles = []
        if len(hits_local) > 0:
            yaw = state.yaw
            cy_, sy_ = math.cos(yaw), math.sin(yaw)
            R_l2g = np.array([[cy_, -sy_], [sy_, cy_]])
            hits_global = hits_local @ R_l2g.T + np.array([state.x, state.y])
            dists = np.linalg.norm(hits_global - np.array([state.x, state.y]), axis=1)
            idx = np.argsort(dists)[:_CarsonMPC.MAX_OBS]
            safe_r = self._mpc_sr + 0.05   # obstacle disc radius = safety margin + hit width
            for i in idx:
                obstacles.append((float(hits_global[i, 0]),
                                   float(hits_global[i, 1]),
                                   safe_r))

        result = self._carson.solve(x0, goal_xyz, obstacles)
        if result is None:
            self.get_logger().warn("CarsonNMPC solve failed — stopping.",
                                   throttle_duration_sec=1.0)
            v_cmd, om_cmd = 0.0, 0.0
        else:
            v_cmd, om_cmd, _ = result

        self._last_omega_cmd = om_cmd
        dummy_rollout = np.zeros((N, 3), dtype=np.float32)
        return MPCResult(
            v_cmd=v_cmd, om_cmd=om_cmd,
            guidance_ratio=0.0, cluster_score=0.0, bearing_score=0.0,
            edt_blocked=0.0, best_rollout=dummy_rollout,
            top_scores=np.zeros(0, dtype=np.float32),
        )

    def _run_sampling_mpc_step(
        self,
        state: SpotState,
        start_id: int,
        target_id: int,
    ) -> MPCResult:
        """
        Core SamplingMPC inference step.

        1. Convert /processed_ranges → local 2D hit points (x=forward, y=left)
        2. Filter degenerate d<0.005m hits
        3. Generate K rollouts, EDT mask, Voronoi scoring (rough plan centroids)
        4. Bearing from bearing_map (explicit, not centroid-derived) + soft
           centroid-direction pull
        5. CBF post-filter
        """
        # ── Step 1: LiDAR hits in body frame ──────────────────────────────────
        ranges = state.raw_ranges
        n_bins = len(ranges)
        angles = np.linspace(0.0, 2 * math.pi, n_bins, endpoint=False)
        # Convention: angle 0 = robot forward (+x body), angles CCW (standard math/ROS).
        # Standard polar → body frame: x_body = r*cos(a), y_body = r*sin(a).
        hits_local = np.stack(
            [ranges * np.cos(angles), ranges * np.sin(angles)], axis=1
        ).astype(np.float32)   # (n_bins, 2), columns = [x_body=fwd, y_body=left]
        max_r = float(ranges.max())
        mask = ranges < max_r * 0.99   # only real returns (not max-range misses)
        hits_local = hits_local[mask]

        # Filter degenerate d=0 hits (wall-boundary artefact)
        if len(hits_local) > 0:
            dists = np.linalg.norm(hits_local, axis=1)
            hits_local = hits_local[dists > 0.005]
        hits_in = hits_local if len(hits_local) > 0 else None

        # ── Step 2: nearest-centroid Voronoi as a proxy for current cluster ────
        # state.x/y are already in the plan frame (metres) — see _to_plan_frame.
        sim_pos = np.array([state.x, state.y])
        dists_to_cents = np.linalg.norm(self._true_cents - sim_pos, axis=1)
        curr_id = int(np.argmin(dists_to_cents))

        # ── Step 3: Generate rollouts ──────────────────────────────────────────
        K, N = self._mpc_K, self._mpc_N
        dt = self._mpc_dt
        v_seqs = np.random.uniform(0.0, 0.75, (K, N))
        om_seqs = np.random.uniform(-1.0, 1.0, (K, N))
        ctrl = np.stack([v_seqs, om_seqs], axis=2)          # (K, N, 2)
        rollouts = unicycle_rollout(np.zeros(3), ctrl, dt)   # (K, N+1, 3)
        traj_local = rollouts[:, 1:, :]                       # (K, N, 3)

        # ── Step 4: EDT collision check ───────────────────────────────────────
        self._occ_grid.update(hits_in)
        dist_values = self._occ_grid.check_collisions(traj_local)

        # ── Step 5: Global endpoints ──────────────────────────────────────────
        yaw = state.yaw
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R_l2g = np.array([[cy_, -sy_], [sy_, cy_]])
        end_local = traj_local[:, -1, :2]                      # (K,2)
        end_global = end_local @ R_l2g.T + sim_pos              # (K,2)

        # ── Step 6 + 7: Scoring ───────────────────────────────────────────────
        goto_point = self.get_parameter("goto_point").get_parameter_value().bool_value

        if goto_point:
            # Go-to-point mode: steer toward a fixed (x,y) in the plan frame.
            # Set goal_x / goal_y at launch or via `ros2 param set`.
            goal_plan = np.array([
                self.get_parameter("goal_x").get_parameter_value().double_value,
                self.get_parameter("goal_y").get_parameter_value().double_value,
            ], dtype=np.float32)
            R_g2l = np.array([[cy_, sy_], [-sy_, cy_]])
            goal_local = (goal_plan - sim_pos) @ R_g2l
            dist_to_goal = float(np.linalg.norm(goal_local))
            bearing_to_goal = math.atan2(float(goal_local[1]), float(goal_local[0]))
            # heading alignment toward goal
            heading_score = 3.0 * np.cos(traj_local[:, -1, 2] - bearing_to_goal)
            # distance progress: rollouts that end closer to goal score higher
            dist_after = np.linalg.norm(end_local - goal_local[np.newaxis, :], axis=1)
            progress_score = np.clip(
                (dist_to_goal - dist_after).astype(np.float32), -5.0, 5.0)
            scores = heading_score + progress_score
            cluster_parts = np.zeros(K, dtype=np.float32)   # for debug log
            bearing_parts = scores.copy()
        else:
            # ── Step 6: TRUE Voronoi cluster membership ────────────────────
            dt_c = np.linalg.norm(end_global[:, None, :] - self._true_cents[None, :, :], axis=2)
            nearest = np.argmin(dt_c, axis=1)                       # (K,)
            in_target = (nearest == target_id).astype(float)
            in_start = (nearest == start_id).astype(float)
            in_forbidden = np.isin(nearest, self._forbidden).astype(float)
            cluster_parts = 10.0 * in_target + 1.0 * in_start - 15.0 * in_forbidden
            bearing_only = self.get_parameter("bearing_only").get_parameter_value().bool_value
            scores = np.zeros(K, dtype=np.float32) if bearing_only else cluster_parts.copy()

            # ── Step 7: Bearing alignment + soft centroid-direction pull ───
            tgt_global = self._true_cents[target_id]
            tgt_local = (tgt_global - sim_pos) @ np.array([[cy_, sy_], [-sy_, cy_]])
            plan_bearing = self.bearing_map.get(f"{start_id}-{target_id}")
            if plan_bearing is not None:
                bearing_local = float(plan_bearing) - yaw
            else:
                bearing_local = math.atan2(tgt_local[1], tgt_local[0])
            bearing_cos = 3.0 * np.cos(traj_local[:, -1, 2] - bearing_local)
            cdir = tgt_local / (np.linalg.norm(tgt_local) + 1e-6)
            bearing_dot = 1.0 * (
                np.cos(traj_local[:, -1, 2]) * cdir[0] +
                np.sin(traj_local[:, -1, 2]) * cdir[1])
            bearing_parts = bearing_cos + bearing_dot
            scores += bearing_parts

        # Penalise first-step angular velocity — breaks the tie between smooth
        # straight rollouts and zigzag rollouts that happen to end at the same
        # heading. Weight 0.5 is enough to prefer straight over wild-then-recover
        # without fighting legitimate turns (bearing score still wins at corners).
        scores -= 0.5 * np.abs(om_seqs[:, 0])

        # ── Step 8: EDT mask ──────────────────────────────────────────────────
        collision = (dist_values < self._mpc_sr).any(axis=1)
        scores[collision] = -np.inf
        edt_blocked = float(np.sum(collision)) / K

        finite_scores = scores[np.isfinite(scores)]
        top_scores = (np.sort(finite_scores)[::-1][:10].astype(np.float32)
                      if finite_scores.size else np.zeros(0, dtype=np.float32))

        if np.all(~np.isfinite(scores)):
            v_cmd = 0.15; om_cmd = 0.0; best_k = 0
            best_cp = 0.0; best_bp = 0.0
        else:
            best_k = int(np.argmax(scores))
            v_cmd = float(ctrl[best_k, 0, 0])
            om_cmd = float(ctrl[best_k, 0, 1])
            best_cp = float(cluster_parts[best_k])
            best_bp = float(bearing_parts[best_k])

        # ── Step 9: CBF filter ────────────────────────────────────────────────
        if hits_in is not None and len(hits_in) > 0:
            dists = np.linalg.norm(hits_in, axis=1)
            d_min = float(np.min(dists))
            nearest_hit = hits_in[int(np.argmin(dists))]
            cos_a = math.cos(math.atan2(nearest_hit[1], nearest_hit[0]))
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
            top_scores=top_scores,
        )

    def _apply_action(self, state: SpotState, result: MPCResult):
        """Send SamplingMPC commands to Spot.

        SamplingMPC outputs body-frame unicycle commands:
          v_cmd  = forward speed (m/s) in SamplingMPC sim scale
          om_cmd = yaw rate (rad/s)
        """
        SPOT_MAX_VEL = 0.5  # m/s
        # Sim-to-real velocity scale: SamplingMPC sim velocities were calibrated
        # against 0.75 m/s max; Spot real max is 0.5 m/s.
        scale = SPOT_MAX_VEL / 0.75
        vx_body = float(np.clip(result.v_cmd * scale, 0.0, SPOT_MAX_VEL))
        v_rot = float(np.clip(result.om_cmd, -1.5, 1.5))

        dry = self._no_spot or self.get_parameter("dry_run").get_parameter_value().bool_value
        if dry:
            tag = "[NO SPOT]" if self._no_spot else "[DRY RUN]"
            self.get_logger().info(
                f"{tag} vx={vx_body:.3f}  v_rot={v_rot:.3f}  (no command sent)",
                throttle_duration_sec=0.5)
        else:
            with self._cmd_lock:
                self._cmd_vel = (vx_body, 0.0, v_rot)

        self.get_logger().info(
            f"vx={vx_body:.3f}  v_rot={v_rot:.3f}",
            throttle_duration_sec=0.5)

    def _publish_debug(
        self,
        state: SpotState,
        result: MPCResult,
        start_id: int,
        target_id: int,
    ):
        state_msg = Float32MultiArray()
        state_msg.data = [
            state.x, state.y, state.vx, state.vy,
            result.v_cmd, result.om_cmd, state.yaw,
            float(start_id), float(target_id),
        ]
        self.state_pub.publish(state_msg)

        gd_msg = Float32MultiArray()
        gd_msg.data = [
            result.guidance_ratio,
            result.cluster_score,
            result.bearing_score,
            result.edt_blocked,
        ]
        self.guidance_pub.publish(gd_msg)

        br_msg = Float32MultiArray()
        br_msg.data = result.best_rollout.flatten().tolist()
        self.rollout_pub.publish(br_msg)

        sc_msg = Float32MultiArray()
        sc_msg.data = result.top_scores.tolist()
        self.scores_pub.publish(sc_msg)

        plan_msg = Int32(); plan_msg.data = int(self.current_plan_step_index)
        self.plan_step_pub.publish(plan_msg)

        record = {
            "t": time.time(),
            "plan_start": start_id,
            "plan_next": target_id,
            "plan_step": self.current_plan_step_index,
            "v_cmd": result.v_cmd,
            "om_cmd": result.om_cmd,
            "guidance_ratio": result.guidance_ratio,
            "cluster_score": result.cluster_score,
            "bearing_score": result.bearing_score,
            "edt_blocked": result.edt_blocked,
            "spot_x": state.x,
            "spot_y": state.y,
            "spot_yaw_deg": math.degrees(state.yaw),
        }
        self._debug_log_file.write(json.dumps(record) + "\n")
        self._debug_log_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = SpotMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.lease_keep_alive is not None:
        node.lease_keep_alive.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
