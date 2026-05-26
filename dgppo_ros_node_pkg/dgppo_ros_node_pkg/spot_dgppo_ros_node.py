import rclpy
from rclpy.node import Node
import jax.numpy as jnp
import jax.random as jr
import jax
import yaml
import os
import numpy as np
import threading
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
import json
import math
from typing import NamedTuple, Tuple

# ROS2 Messages
from std_msgs.msg import Int16, Int32, Float32MultiArray
from std_srvs.srv import Trigger

# Boston Dynamics
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.exceptions import LeaseUseError
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient)

# Local utilities
from .utils import (
    SCALE_SPOT_TO_SIM,
    VISION_TO_DGPPO_R, DGPPO_TO_VISION_R,
    rot2d, apply_rot2d,
)
from .spot_utils import get_spot_state, send_velocity
from .plan import load_plan, map_cluster_id, initial_sim_state
from .graph_builder import build_graph

import time

# DGPPO and LidarEnv components
from .dgppo.dgppo.env.lidar_env.lidar_target import (
    LidarTarget, LidarTargetV1, LidarTargetV2, LidarTargetV3, LidarTargetV4,
    LidarTargetBFLag2, LidarTargetBFLag8, LidarTargetDRTLag12
)
from .dgppo.dgppo.algo import make_algo
from .dgppo.dgppo.utils.graph import GraphsTuple


class SpotSimState(NamedTuple):
    x: float           # vision frame pos x (m)
    y: float           # vision frame pos y (m)
    vx: float          # vision frame vel x (m/s)
    vy: float          # vision frame vel y (m/s)
    yaw: float         # body yaw in vision frame (radians)
    sim_pos_x: float   # DGPPO sim frame pos x (scaled)
    sim_pos_y: float   # DGPPO sim frame pos y (scaled)
    sim_vel_x: float   # DGPPO sim frame vel x (scaled)
    sim_vel_y: float   # DGPPO sim frame vel y (scaled)
    vel_body_fwd: float   # body frame vel forward (m/s)
    vel_body_lat: float   # body frame vel lateral/left (m/s)
    scaled_ranges: np.ndarray   # (n_rays_phys,) lidar ranges / SCALE_SPOT_TO_SIM
    raw_ranges: np.ndarray      # (n_rays_phys,) lidar ranges (m)


class InferenceResult(NamedTuple):
    action: jnp.ndarray        # (1,2) clipped, possibly rotated, action in sim frame
    action_raw_flat: list      # pre-rotation policy output as floats
    action_flat: list          # post-rotation output as floats (what goes to Spot)
    rot_deg: float             # action_rotation_deg used this tick
    inf_ms: float              # DGPPO inference wall-clock time (ms)
    graph: GraphsTuple         # graph used for this inference step (needed for get_reward)


class DGPPOROSNode(Node):
    def __init__(self):
        super().__init__('dgppo_ros_node')
        self.get_logger().info("Initializing DGPPO ROS Node...")
        self._init_params()
        self._init_model()
        self._init_ros()
        self._init_spot()
        self.get_logger().info("DGPPO ROS Node fully initialized and ready.")

    # ── Initialization helpers ────────────────────────────────────────────────

    def _init_params(self):
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('current_cluster_id', 1)
        self.declare_parameter('angular_offset_deg', 0.0)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('step_test', False)
        self.declare_parameter('step_test_once', False)
        self._step_test_t0 = None
        self.declare_parameter('spoof_action', False)
        self.declare_parameter('spoof_action_x', 0.0)
        self.declare_parameter('spoof_action_y', 0.0)
        self.declare_parameter('use_projected_vel', False)
        self.declare_parameter('action_rotation_deg', 0.0)
        self.declare_parameter('world_y_offset_deg', 0.0)
        _world_y_deg = self.get_parameter('world_y_offset_deg').get_parameter_value().double_value
        self._world_alpha_rad = math.radians(_world_y_deg)
        self.declare_parameter('transform_verbosity', 0)
        self._transform_verbosity = self.get_parameter('transform_verbosity').get_parameter_value().integer_value
        self.get_logger().info(
            f"world_y_offset_deg={_world_y_deg:.1f}deg — "
            "CW angle (viewed from above) from Spot boot-up forward to desired sim +Y direction"
        )
        self.num_clusters = 4
        self.twod_area_size = 1.5
        self.scale_2d_3d = 11
        self.origin_x = 0.0
        self.origin_y = 0.0

    def _init_model(self):
        _ENV_CLASSES = {
            'LidarTarget':         LidarTarget,
            'LidarTargetV1':       LidarTargetV1,
            'LidarTargetV2':       LidarTargetV2,
            'LidarTargetV3':       LidarTargetV3,
            'LidarTargetV4':       LidarTargetV4,
            'LidarTargetBFLag2':   LidarTargetBFLag2,
            'LidarTargetBFLag8':   LidarTargetBFLag8,
            'LidarTargetDRTLag12': LidarTargetDRTLag12,
        }

        model_dir = "dgppo/logs/LidarTargetV1/dgppo/BF_weights"
        config_path = os.path.join(model_dir, "config.yaml")
        params_path = os.path.join(model_dir, "models")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        step = self._get_model_step(model_dir)
        env_kwargs = config.get("env_kwargs", {})
        self.get_logger().info(f"Loaded config: {config}")

        self.n_rays_phys = 72  # TODO: verify Spot LiDAR bin count
        env_class_name = config.get('env', 'LidarTarget')
        env_class = _ENV_CLASSES.get(env_class_name, LidarTarget)
        self.get_logger().info(f"Using env class: {env_class_name}")

        merged_params = {**env_class.PARAMS, **env_kwargs.get('params', {})}
        merged_params['top_k_rays'] = 8
        merged_params['comm_radius'] = 0.5

        self.env_instance = env_class(
            num_agents=config.get('num_agents'),
            params=merged_params,
            **{k: v for k, v in env_kwargs.items() if k != 'params'}
        )

        algo_kwargs = config.get("algo_kwargs", {})
        self.algo = make_algo(
            algo=config.get('algo'),
            env=self.env_instance,
            node_dim=self.env_instance.node_dim,
            edge_dim=self.env_instance.edge_dim,
            state_dim=self.env_instance.state_dim,
            action_dim=self.env_instance.action_dim,
            n_agents=self.env_instance.num_agents,
            **algo_kwargs
        )

        self.plan_sequence, self.bearing_map, self.cluster_centroids = load_plan()
        self.current_plan_step_index = 0

        self.algo.load(params_path, step=step)

        self.rng_key = jr.PRNGKey(config.get('seed', 0))
        self.rnn_state = self.algo.init_rnn_state

        self.current_agent_state = None
        self.latest_ranges_msg = None
        self.latest_agent_state = None
        self.latest_predicted_cluster_id = None
        self.latest_terrain_id = 1  # Grass default until /current_terrain publishes
        self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        self._tablet_has_lease = False
        self.is_first_run = True
        self._projected_sim_vel = None
        self._projected_sim_pos = None

        if self.plan_sequence:
            _st = initial_sim_state(
                self.cluster_centroids,
                self.plan_sequence[0]["start"],
                self.scale_2d_3d,
                self.origin_x,
                self.origin_y,
            )
            self.sim_origin_x = float(_st[0])
            self.sim_origin_y = float(_st[1])
        else:
            self.sim_origin_x = 0.0
            self.sim_origin_y = 0.0

    def _init_ros(self):
        self.ranges_sub = self.create_subscription(
            Float32MultiArray,
            '/processed_ranges',
            self.ranges_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.predicted_cluster_sub = self.create_subscription(
            Int16,
            '/predicted_cluster',
            self.predicted_cluster_callback,
            10
        )
        self.terrain_sub = self.create_subscription(
            Int16,
            '/current_terrain',
            self.terrain_callback,
            10
        )

        self.spot_yaw_pub = self.create_publisher(Float32MultiArray, '/dgppo_spot_yaw', 10)
        self.spot_act_pub = self.create_publisher(Float32MultiArray, '/dgppo_action', 100)
        self.state_debug_pub = self.create_publisher(Float32MultiArray, '/dgppo_state_debug', 10)
        self.plan_step_pub = self.create_publisher(Int32, '/dgppo_plan_step', 10)
        _latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST
        )
        self.world_alpha_pub = self.create_publisher(Float32MultiArray, '/dgppo_world_alpha', _latched_qos)
        _alpha_msg = Float32MultiArray()
        _alpha_msg.data = [self._world_alpha_rad]
        self.world_alpha_pub.publish(_alpha_msg)

        self._take_lease_srv = self.create_service(Trigger, '/dgppo_take_lease', self._take_lease_callback)
        self.timer = self.create_timer(self.env_instance.dt, self.control_loop)

        import datetime
        _log_dir = os.path.join(os.path.dirname(__file__), 'debug_logs')
        os.makedirs(_log_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._debug_log_path = os.path.join(_log_dir, f'dgppo_run_{_ts}.jsonl')
        self._debug_log_file = open(self._debug_log_path, 'w')
        self.get_logger().info(f"Debug log: {self._debug_log_path}")
        self.get_logger().info("Default mode: Listening for predicted cluster ID on /predicted_cluster_id.")
        self.get_logger().info("To activate debug mode: 'ros2 param set /dgppo_ros_node debug_mode true'")
        self.get_logger().info("When in debug mode: 'ros2 param set /dgppo_ros_node current_cluster_id <new_id>'")

    def _init_spot(self):
        self.get_logger().info("Initializing the Spot robot.")
        self.sdk = bosdyn.client.create_standard_sdk("understanding-spot")
        self.robot = self.sdk.create_robot("10.0.0.3")
        self.robot.authenticate(username="user", password="pass")
        self.robot.time_sync.wait_for_sync()

        self.state_client = self.robot.ensure_client("robot-state")
        self.lease_client = self.robot.ensure_client("lease")
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

        self.lease_client.take()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)

        self.get_logger().info("Current state")
        self.get_logger().info(str(self.state_client.get_robot_state()))

        self._cached_robot_state = None
        self._state_lock = threading.Lock()
        self._state_poller = threading.Thread(target=self._poll_spot_state, daemon=True)
        self._state_poller.start()

        self._cmd_vel = (0.0, 0.0)
        self._cmd_lock = threading.Lock()
        self._cmd_sender = threading.Thread(target=self._send_commands, daemon=True)
        self._cmd_sender.start()

    # ── Background threads ────────────────────────────────────────────────────

    def _tlog(self, level: int, msg: str):
        """Log transform debug info when transform_verbosity >= level."""
        if self._transform_verbosity >= level:
            self.get_logger().info(msg)

    def _poll_spot_state(self):
        """Background thread: keeps _cached_robot_state fresh at ~50 Hz."""
        while True:
            try:
                rs = self.state_client.get_robot_state()
                with self._state_lock:
                    self._cached_robot_state = rs
            except Exception:
                pass
            time.sleep(0.02)

    def _send_commands(self):
        """Background thread: forwards _cmd_vel to Spot at ~25 Hz."""
        while True:
            with self._cmd_lock:
                v_x, v_y = self._cmd_vel
            try:
                send_velocity(self.command_client, v_x, v_y, end_time_secs=time.time() + 0.5)
                self._tablet_has_lease = False
            except (LeaseUseError, bosdyn.client.lease.NotActiveLeaseError):
                self._tablet_has_lease = True
            except Exception:
                pass
            time.sleep(0.04)  # 25 Hz — well within the 500 ms command expiry window

    def _get_spot_state(self):
        """Read cached Spot state and return (x, y, vx, vy, yaw) in vision frame."""
        with self._state_lock:
            robot_state = self._cached_robot_state
        if robot_state is None:
            robot_state = self.state_client.get_robot_state()
        return get_spot_state(robot_state)

    def _get_model_step(self, model_dir):
        model_path = os.path.join(model_dir, "models")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        self.get_logger().info(f"Loading latest model from step: {step}")
        return step

    def _take_lease_callback(self, request, response):
        """Service handler: reclaim the lease from the tablet and resume the plan."""
        try:
            self.lease_keep_alive.shutdown()
            self.lease_client.take()
            self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)
            self._tablet_has_lease = False
            self.get_logger().info("Lease reclaimed from tablet. Plan resuming.")
            response.success = True
            response.message = "Lease reclaimed. Plan resuming."
        except Exception as e:
            self.get_logger().error(f"Failed to reclaim lease: {e}")
            response.success = False
            response.message = f"Failed to reclaim lease: {e}"
        return response

    def _handle_plan_complete(self):
        """Stop the robot and cancel the control timer when the plan finishes."""
        self.get_logger().info("High-level plan is complete. Stopping control loop.")
        if not self.get_parameter('dry_run').get_parameter_value().bool_value:
            try:
                self.command_client.robot_command(command=RobotCommandBuilder.stop_command())
            except (LeaseUseError, bosdyn.client.lease.NotActiveLeaseError) as e:
                self.get_logger().warning(
                    f"Could not send stop command — tablet may hold lease. ({type(e).__name__})"
                )
        self.timer.cancel()

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def ranges_callback(self, msg: Float32MultiArray):
        self.latest_ranges_msg = msg

    def predicted_cluster_callback(self, msg: Int16):
        self.latest_predicted_cluster_id = msg.data

    def terrain_callback(self, msg: Int16):
        self.latest_terrain_id = msg.data

    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.is_first_run:
            if not self._first_run_init():
                return

        if self.current_plan_step_index >= len(self.plan_sequence):
            self._handle_plan_complete()
            return

        current_plan_step = self.plan_sequence[self.current_plan_step_index]
        expected_start_cluster = current_plan_step["start"]
        expected_next_cluster  = current_plan_step["next"]
        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        if not self._check_topics(debug_mode):
            return

        if debug_mode:
            current_cluster_id = self.get_parameter('current_cluster_id').get_parameter_value().integer_value
            self.get_logger().info(
                f"DEBUG MODE: Using manual cluster ID {current_cluster_id}", throttle_duration_sec=1.0
            )
        else:
            current_cluster_id = self.latest_predicted_cluster_id
            self.get_logger().info(
                f"Default MODE: Using predicted cluster ID {current_cluster_id}", throttle_duration_sec=1.0
            )

        mapped_current_cluster = map_cluster_id(current_cluster_id)
        self.get_logger().info(
            f"cluster raw={current_cluster_id} mapped={mapped_current_cluster}", throttle_duration_sec=1.0
        )

        if self._tablet_has_lease:
            self.get_logger().info(
                "Tablet holds lease — plan paused, state updates running. "
                "To reclaim: ros2 service call /dgppo_take_lease std_srvs/srv/Trigger '{}'",
                throttle_duration_sec=3.0,
            )
            self._update_paused_state()
            return

        if mapped_current_cluster == expected_next_cluster:
            self._advance_plan(expected_start_cluster, expected_next_cluster)

        if self.current_plan_step_index >= len(self.plan_sequence):
            return

        state  = self._update_spot_state()
        result = self._run_inference(
            state, expected_start_cluster, expected_next_cluster,
            mapped_current_cluster, current_cluster_id,
        )
        v_cmd = self._apply_action(state, result)
        self._publish_and_log(state, result, v_cmd)

    def _first_run_init(self) -> bool:
        """Set initial agent state from start-cluster centroid. Returns False if centroid missing."""
        if self.current_plan_step_index < len(self.plan_sequence):
            start_cluster_id = str(self.plan_sequence[self.current_plan_step_index]["start"])
            next_cluster_id  = str(self.plan_sequence[self.current_plan_step_index]["next"])
            if start_cluster_id in self.cluster_centroids:
                centroid = self.cluster_centroids[start_cluster_id]
                self.get_logger().info(
                    f"Setting initial agent state to centroid of cluster {start_cluster_id}: {centroid}"
                )
                _st = initial_sim_state(
                    self.cluster_centroids, start_cluster_id, self.scale_2d_3d,
                    self.origin_x, self.origin_y,
                )
                self.latest_agent_state = jnp.expand_dims(jnp.array(_st), axis=0)
                plan_key = f"{start_cluster_id}-{next_cluster_id}"
                bearing_rad = self.bearing_map.get(plan_key, 0.0)
                self.get_logger().info(
                    f"Initial Bearing from plan: {bearing_rad:.2f} rad ({math.degrees(bearing_rad):.2f} deg)"
                )
            else:
                self.get_logger().error(
                    f"Centroid for cluster {start_cluster_id} not found in plan data!"
                )
                return False
        self.is_first_run = False
        return True

    def _check_topics(self, debug_mode: bool) -> bool:
        """Return False (and warn) if required topics have not yet arrived."""
        missing = []
        if self.latest_ranges_msg is None:
            missing.append('/processed_ranges  ← clustering node must be running and receiving lidar')
        if not debug_mode and self.latest_predicted_cluster_id is None:
            missing.append('/predicted_cluster ← clustering node must be running and receiving lidar')
        if missing:
            self.get_logger().warning(
                'Waiting for topics:\n  ' + '\n  '.join(missing),
                throttle_duration_sec=3.0,
            )
            return False
        return True

    def _advance_plan(self, expected_start_cluster: int, expected_next_cluster: int):
        """Increment the plan step index and log the transition."""
        self.current_plan_step_index += 1
        if self.current_plan_step_index >= len(self.plan_sequence):
            self.get_logger().info(
                f"Plan step complete. Transitioning to cluster {expected_next_cluster}. "
                "Plan is now finished."
            )
        else:
            self.get_logger().info(
                f"Plan step complete. Transitioning from cluster {expected_start_cluster} "
                f"to {expected_next_cluster}. "
                f"Next step is from cluster {self.plan_sequence[self.current_plan_step_index]['start']}."
            )

    def _update_paused_state(self):
        """While the tablet holds the lease: keep the sim state current from Spot odometry."""
        try:
            x, y, vx, vy, yaw = self._get_spot_state()
            yaw_msg = Float32MultiArray()
            yaw_msg.data = [yaw]
            self.spot_yaw_pub.publish(yaw_msg)
            self._tlog(1, f"[PAUSE] vision_pos=({x:.3f},{y:.3f})  vision_vel=({vx:.3f},{vy:.3f})")

            # Position: vision → sim (two separate transforms)
            pos_after_fixed = apply_rot2d(VISION_TO_DGPPO_R, np.array([x, y]))
            self._tlog(2, f"[PAUSE][pos] after VISION_TO_DGPPO_R: {pos_after_fixed}")
            R_alpha = rot2d(self._world_alpha_rad)
            pos_sim_unscaled = apply_rot2d(R_alpha, pos_after_fixed)
            self._tlog(
                2, f"[PAUSE][pos] after alpha R (alpha={math.degrees(self._world_alpha_rad):.1f}°): "
                   f"{pos_sim_unscaled}"
            )
            sim_pos_x = pos_sim_unscaled[0] / SCALE_SPOT_TO_SIM + self.sim_origin_x
            sim_pos_y = pos_sim_unscaled[1] / SCALE_SPOT_TO_SIM + self.sim_origin_y
            self._tlog(1, f"[PAUSE] sim_pos=({sim_pos_x:.3f},{sim_pos_y:.3f})")

            # Velocity: same two transforms, no origin offset
            vel_after_fixed = apply_rot2d(VISION_TO_DGPPO_R, np.array([vx, vy]))
            self._tlog(2, f"[PAUSE][vel] after VISION_TO_DGPPO_R: {vel_after_fixed}")
            vel_sim_unscaled = apply_rot2d(R_alpha, vel_after_fixed)
            sim_vel_x = vel_sim_unscaled[0] / SCALE_SPOT_TO_SIM
            sim_vel_y = vel_sim_unscaled[1] / SCALE_SPOT_TO_SIM
            self._tlog(2, f"[PAUSE] sim_vel=({sim_vel_x:.3f},{sim_vel_y:.3f})")

            self.latest_agent_state = jnp.expand_dims(
                jnp.array([sim_pos_x, sim_pos_y, sim_vel_x, sim_vel_y], dtype=jnp.float32), axis=0
            )
        except Exception as e:
            self.get_logger().warning(
                f"State read failed while paused: {e}", throttle_duration_sec=2.0
            )
        self._projected_sim_vel = None
        self._projected_sim_pos = None

    def _update_spot_state(self) -> SpotSimState:
        """Read Spot odometry + lidar, transform to sim frame. Updates latest_agent_state."""
        raw_ranges_np    = np.array(self.latest_ranges_msg.data, dtype=np.float32)
        scaled_ranges_np = raw_ranges_np / self.scale_2d_3d

        x, y, vx, vy, yaw = self._get_spot_state()
        self._tlog(
            1, f"[STATE] vision_pos=({x:.3f},{y:.3f})  vision_vel=({vx:.3f},{vy:.3f})  "
               f"yaw={math.degrees(yaw):.1f}°"
        )
        yaw_msg = Float32MultiArray()
        yaw_msg.data = [yaw]
        self.spot_yaw_pub.publish(yaw_msg)

        # Position: vision → sim (two separate transforms)
        pos_after_fixed = apply_rot2d(VISION_TO_DGPPO_R, np.array([x, y]))
        self._tlog(2, f"[STATE][pos] after VISION_TO_DGPPO_R: {pos_after_fixed}")
        R_alpha = rot2d(self._world_alpha_rad)
        pos_sim_unscaled = apply_rot2d(R_alpha, pos_after_fixed)
        self._tlog(
            2, f"[STATE][pos] after alpha R (alpha={math.degrees(self._world_alpha_rad):.1f}°): "
               f"{pos_sim_unscaled}"
        )
        spot_sim_pos_x = pos_sim_unscaled[0] / SCALE_SPOT_TO_SIM + self.sim_origin_x
        spot_sim_pos_y = pos_sim_unscaled[1] / SCALE_SPOT_TO_SIM + self.sim_origin_y
        self._tlog(1, f"[STATE] sim_pos=({spot_sim_pos_x:.3f},{spot_sim_pos_y:.3f})")

        # Velocity: same two transforms, no origin offset
        vel_after_fixed = apply_rot2d(VISION_TO_DGPPO_R, np.array([vx, vy]))
        self._tlog(2, f"[STATE][vel] after VISION_TO_DGPPO_R: {vel_after_fixed}")
        vel_sim_unscaled = apply_rot2d(R_alpha, vel_after_fixed)  # reuse R_alpha from above
        spot_sim_vel_x = vel_sim_unscaled[0] / SCALE_SPOT_TO_SIM
        spot_sim_vel_y = vel_sim_unscaled[1] / SCALE_SPOT_TO_SIM
        self._tlog(2, f"[STATE] sim_vel=({spot_sim_vel_x:.3f},{spot_sim_vel_y:.3f})")

        if (self.get_parameter('use_projected_vel').get_parameter_value().bool_value
                and self._projected_sim_vel is not None
                and self._projected_sim_pos is not None):
            sim_pos_x, sim_pos_y = self._projected_sim_pos
            sim_vel_x, sim_vel_y = self._projected_sim_vel
            self.get_logger().info(
                f'[STATE] spot_odom_pos=({spot_sim_pos_x:.3f}, {spot_sim_pos_y:.3f})  '
                f'proj_pos=({sim_pos_x:.3f}, {sim_pos_y:.3f})  '
                f'spot_odom_vel=({spot_sim_vel_x:.3f}, {spot_sim_vel_y:.3f})  '
                f'proj_vel=({sim_vel_x:.3f}, {sim_vel_y:.3f})  [using projected]',
                throttle_duration_sec=0.5,
            )
        else:
            sim_pos_x = spot_sim_pos_x
            sim_pos_y = spot_sim_pos_y
            sim_vel_x = spot_sim_vel_x
            sim_vel_y = spot_sim_vel_y
            self.get_logger().info(
                f'[STATE] odom_pos=({spot_sim_pos_x:.3f}, {spot_sim_pos_y:.3f})  '
                f'odom_vel=({spot_sim_vel_x:.3f}, {spot_sim_vel_y:.3f})  [using odometry]',
                throttle_duration_sec=0.5,
            )

        # Body-frame velocity: vision → body (rot2d(-yaw))
        R_vis_to_body = rot2d(-yaw)
        self._tlog(2, f"[STATE] R_vis_to_body (yaw={math.degrees(yaw):.1f}°):\n{R_vis_to_body}")
        v_body = apply_rot2d(R_vis_to_body, np.array([vx, vy]))
        vel_body_fwd, vel_body_lat = float(v_body[0]), float(v_body[1])
        self._tlog(2, f"[STATE] body_vel fwd={vel_body_fwd:.3f} lat={vel_body_lat:.3f} m/s")

        scaled_latest_state_np = np.array([sim_pos_x, sim_pos_y, sim_vel_x, sim_vel_y], dtype=np.float32)
        self.latest_agent_state = jnp.expand_dims(jnp.array(scaled_latest_state_np), axis=0)
        self.get_logger().info(
            f'[GRAPH INPUT] pos=({sim_pos_x:.3f}, {sim_pos_y:.3f})  '
            f'vel=({sim_vel_x:.3f}, {sim_vel_y:.3f})',
            throttle_duration_sec=0.5,
        )

        return SpotSimState(
            x=float(x), y=float(y), vx=float(vx), vy=float(vy), yaw=float(yaw),
            sim_pos_x=float(sim_pos_x), sim_pos_y=float(sim_pos_y),
            sim_vel_x=float(sim_vel_x), sim_vel_y=float(sim_vel_y),
            vel_body_fwd=vel_body_fwd, vel_body_lat=vel_body_lat,
            scaled_ranges=scaled_ranges_np,
            raw_ranges=raw_ranges_np,
        )

    def _run_inference(
        self,
        state: SpotSimState,
        expected_start_cluster: int,
        expected_next_cluster: int,
        mapped_current_cluster: int,
        current_cluster_id: int,
    ) -> InferenceResult:
        """Build graph, run DGPPO policy, apply rotation/spoof. Populates self._tick_record."""
        max_range_val = float(state.raw_ranges.max())
        n_maxed = int((state.raw_ranges >= max_range_val * 0.99).sum())
        angular_offset = self.get_parameter('angular_offset_deg').get_parameter_value().double_value
        bearing_key = f"{expected_start_cluster}-{expected_next_cluster}"
        bearing_val = self.bearing_map.get(bearing_key, 0.0) + math.pi / 2 + math.radians(angular_offset)

        self._tick_record = {
            't':               time.time(),
            'cluster_raw':     int(current_cluster_id),
            'cluster_mapped':  int(mapped_current_cluster),
            'plan_start':      int(expected_start_cluster),
            'plan_next':       int(expected_next_cluster),
            'plan_step':       int(self.current_plan_step_index),
            'ranges_min_m':    float(state.raw_ranges.min()),
            'ranges_max_m':    float(max_range_val),
            'ranges_mean_m':   float(state.raw_ranges.mean()),
            'n_beams_at_max':  n_maxed,
            'n_beams_total':   len(state.raw_ranges),
            'ranges_raw':      state.raw_ranges.tolist(),
            'scale_2d_3d':     self.scale_2d_3d,
            'world_alpha_deg': float(math.degrees(self._world_alpha_rad)),
            'pos_spot_x':      state.x,
            'pos_spot_y':      state.y,
            'vel_spot_x':      state.vx,
            'vel_spot_y':      state.vy,
            'yaw_deg':         float(math.degrees(state.yaw)),
            'sim_pos_x':       state.sim_pos_x,
            'sim_pos_y':       state.sim_pos_y,
            'sim_vel_x':       state.sim_vel_x,
            'sim_vel_y':       state.sim_vel_y,
            'bearing_plan_deg':  float(math.degrees(self.bearing_map.get(bearing_key, 0.0))),
            'bearing_dgppo_deg': float(math.degrees(bearing_val)),
            'angular_offset_deg': float(angular_offset),
            'terrain_id':      int(self.latest_terrain_id),
        }
        self.get_logger().info(
            f"CLUSTER raw={current_cluster_id} mapped={mapped_current_cluster} "
            f"plan={expected_start_cluster}→{expected_next_cluster} | "
            f"RANGES min={state.raw_ranges.min():.2f}m mean={state.raw_ranges.mean():.2f}m "
            f"n_at_max={n_maxed}/{len(state.raw_ranges)} | "
            f"BEARING={math.degrees(bearing_val):.1f}° YAW={math.degrees(state.yaw):.1f}°",
            throttle_duration_sec=0.5,
        )

        graph = build_graph(
            env_instance=self.env_instance,
            agent_state_np=self.latest_agent_state,
            scaled_ranges=state.scaled_ranges,
            current_cluster_id=mapped_current_cluster,
            start_cluster_id=expected_start_cluster,
            next_cluster_id=expected_next_cluster,
            bonus_awarded=self.next_cluster_bonus_awarded,
            terrain_id=self.latest_terrain_id,
            bearing_map=self.bearing_map,
            num_clusters=self.num_clusters,
            n_rays_phys=self.n_rays_phys,
            world_alpha_rad=self._world_alpha_rad,
            yaw=state.yaw,
            angular_offset_rad=math.radians(angular_offset),
            logger=self.get_logger(),
        )

        self.rng_key, _ = jr.split(self.rng_key)
        _t_inf0 = time.time()
        action, new_rnn_state = self.algo.act(
            graph=graph,
            rnn_state=self.rnn_state,
            params={'policy': self.algo.policy_train_state.params}
        )
        _inf_ms = (time.time() - _t_inf0) * 1000.0
        self.get_logger().info(f"inference {_inf_ms:.1f} ms", throttle_duration_sec=1.0)
        self.rnn_state = new_rnn_state

        if self.get_parameter('spoof_action').get_parameter_value().bool_value:
            sx = self.get_parameter('spoof_action_x').get_parameter_value().double_value
            sy = self.get_parameter('spoof_action_y').get_parameter_value().double_value
            action = jnp.array([[sx, sy]], dtype=jnp.float32)
            self.get_logger().info(f'[SPOOF] action x={sx:.3f}  y={sy:.3f}', throttle_duration_sec=0.5)

        action = jnp.clip(action, -1.0, 1.0)
        action_raw_flat = [float(a) for a in np.array(action).flatten()]
        rot_deg = self.get_parameter('action_rotation_deg').get_parameter_value().double_value
        if rot_deg != 0.0:
            theta = math.radians(rot_deg)  # positive = CW
            c, s = math.cos(theta), math.sin(theta)
            ax, ay = action_raw_flat[0], action_raw_flat[1]
            action = jnp.array([[c * ax + s * ay, -s * ax + c * ay]], dtype=jnp.float32)
            self.get_logger().info(
                f'[ACTION ROT] raw=({ax:.3f}, {ay:.3f})  '
                f'rotated=({float(action[0, 0]):.3f}, {float(action[0, 1]):.3f})  '
                f'rot={rot_deg:.1f}°CW',
                throttle_duration_sec=0.5,
            )

        action_flat = [float(a) for a in np.array(action).flatten()]
        self._tick_record['action_raw_sim_x'] = action_raw_flat[0] if len(action_raw_flat) > 0 else 0.0
        self._tick_record['action_raw_sim_y'] = action_raw_flat[1] if len(action_raw_flat) > 1 else 0.0
        self._tick_record['action_sim_x']     = action_flat[0]     if len(action_flat)     > 0 else 0.0
        self._tick_record['action_sim_y']     = action_flat[1]     if len(action_flat)     > 1 else 0.0
        self._tick_record['inference_ms']     = round(_inf_ms, 2)

        return InferenceResult(
            action=action,
            action_raw_flat=action_raw_flat,
            action_flat=action_flat,
            rot_deg=rot_deg,
            inf_ms=_inf_ms,
            graph=graph,
        )

    def _apply_action(self, state: SpotSimState, result: InferenceResult) -> Tuple[float, float]:
        """Project next state, update reward/bonus, transform sim vel → body cmd. Returns (vx, vy)."""
        new_movement_targets = jnp.squeeze(
            self.env_instance.agent_step_euler(self.latest_agent_state, result.action), axis=0
        )
        self._projected_sim_pos = (float(new_movement_targets[0]), float(new_movement_targets[1]))
        self._projected_sim_vel = (float(new_movement_targets[2]), float(new_movement_targets[3]))

        reward, bonus_awarded_updated = self.env_instance.get_reward(result.graph, result.action)
        if bonus_awarded_updated.size == 0:
            self.get_logger().warning("Received an empty bonus array from get_reward. Resetting.")
            self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        else:
            self.next_cluster_bonus_awarded = bonus_awarded_updated

        # Transform DGPPO sim-world velocity → Spot body-frame velocity command
        SPOT_MAX_VEL = 0.5  # m/s — conservative safe limit
        _sim_vx = float(new_movement_targets[2])
        _sim_vy = float(new_movement_targets[3])
        self._tlog(1, f"[ACTION] dgppo_vel_sim=({_sim_vx:.3f},{_sim_vy:.3f})")

        # Step 1: undo world_alpha (inverse of transform 2 in vision→sim)
        R_alpha_inv = rot2d(-self._world_alpha_rad)
        vel_after_alpha_inv = apply_rot2d(R_alpha_inv, np.array([_sim_vx, _sim_vy]))
        self._tlog(
            2, f"[ACTION] after alpha_inv (alpha={math.degrees(self._world_alpha_rad):.1f}°): "
               f"{vel_after_alpha_inv}"
        )

        # Step 2: fixed 90° CCW from above (DGPPO → vision frame), then scale to m/s
        vel_vision = apply_rot2d(DGPPO_TO_VISION_R, vel_after_alpha_inv) * SCALE_SPOT_TO_SIM
        self._tlog(1, f"[ACTION] vel_vision=({vel_vision[0]:.3f},{vel_vision[1]:.3f}) m/s")

        # Step 3: vision → body frame (rot2d(-yaw))
        R_vis_to_body_cmd = rot2d(-state.yaw)
        self._tlog(2, f"[ACTION] R_vis_to_body (yaw={math.degrees(state.yaw):.1f}°):\n{R_vis_to_body_cmd}")
        v_body_cmd = apply_rot2d(R_vis_to_body_cmd, vel_vision)
        v_x_raw, v_y_raw = float(v_body_cmd[0]), float(v_body_cmd[1])
        self._tlog(1, f"[ACTION] vel_body_raw=({v_x_raw:.3f},{v_y_raw:.3f}) m/s")

        # Step 4: proportional clamp to SPOT_MAX_VEL
        max_component = max(abs(v_x_raw), abs(v_y_raw))
        clamp = min(1.0, SPOT_MAX_VEL / max_component) if max_component > 0.0 else 1.0
        v_x_target = v_x_raw * clamp
        v_y_target = v_y_raw * clamp
        self._tlog(
            1, f"[ACTION] vel_body_clamped=({v_x_target:.3f},{v_y_target:.3f}) m/s  clamp={clamp:.3f}"
        )

        # Step-test overrides
        STEP_VX = 0.4  # m/s — safe walking speed
        if self.get_parameter('step_test_once').get_parameter_value().bool_value:
            if self._step_test_t0 is None:
                self._step_test_t0 = time.time()
            v_x_target = STEP_VX
            v_y_target = 0.0
            self.get_logger().info(
                f'[STEP ONCE] t={time.time() - self._step_test_t0:.2f}s  vx={v_x_target:.2f} m/s',
                throttle_duration_sec=0.5,
            )
        elif self.get_parameter('step_test').get_parameter_value().bool_value:
            STEP_HALF_PERIOD = 4.0
            if self._step_test_t0 is None:
                self._step_test_t0 = time.time()
            phase = (time.time() - self._step_test_t0) % (2.0 * STEP_HALF_PERIOD)
            v_x_target = STEP_VX if phase < STEP_HALF_PERIOD else 0.0
            v_y_target = 0.0
            self.get_logger().info(
                f'[STEP TEST] phase={phase:.2f}s  vx={v_x_target:.2f} m/s',
                throttle_duration_sec=0.5,
            )
        else:
            self._step_test_t0 = None

        self._tick_record['cmd_body_fwd_ms']  = float(v_x_target)
        self._tick_record['cmd_body_left_ms'] = float(v_y_target)
        self._debug_log_file.write(json.dumps(self._tick_record) + '\n')
        self._debug_log_file.flush()

        dry_run = self.get_parameter('dry_run').get_parameter_value().bool_value
        if dry_run:
            self.get_logger().info(
                f"[DRY RUN] vx={v_x_target:.3f}  vy={v_y_target:.3f}  (no command sent)",
                throttle_duration_sec=0.5,
            )
        else:
            with self._cmd_lock:
                self._cmd_vel = (v_x_target, v_y_target)
            self.get_logger().info(
                f"vx={v_x_target:.3f}  vy={v_y_target:.3f}",
                throttle_duration_sec=0.5,
            )

        return (v_x_target, v_y_target)

    def _publish_and_log(
        self,
        state: SpotSimState,
        result: InferenceResult,
        v_cmd: Tuple[float, float],
    ):
        """Publish all ROS debug/action/plan topics."""
        v_x_target, v_y_target = v_cmd

        _dbg = Float32MultiArray()
        _dbg.data = [
            state.x,             state.y,             # [0,1]  vision frame pos (m)
            state.vx,            state.vy,            # [2,3]  vision frame vel (m/s)
            state.vel_body_fwd,  state.vel_body_lat,  # [4,5]  body vel: fwd, left (m/s)
            state.sim_pos_x,     state.sim_pos_y,     # [6,7]  DGPPO sim pos (scaled)
            state.sim_vel_x,     state.sim_vel_y,     # [8,9]  DGPPO sim vel (scaled)
            float(v_x_target),   float(v_y_target),   # [10,11] cmd fwd/left (m/s)
            float(result.action_raw_flat[0]) if len(result.action_raw_flat) > 0 else 0.0,  # [12]
            float(result.action_raw_flat[1]) if len(result.action_raw_flat) > 1 else 0.0,  # [13]
            float(result.inf_ms),                                                            # [14]
            float(result.action_flat[0]) if len(result.action_flat) > 0 else 0.0,          # [15]
            float(result.action_flat[1]) if len(result.action_flat) > 1 else 0.0,          # [16]
            float(result.rot_deg),                                                           # [17]
        ]
        self.state_debug_pub.publish(_dbg)

        act_msg = Float32MultiArray()
        act_msg.data = [-v_y_target, v_x_target]  # [right, fwd] matches visualizer canvas convention
        self.spot_act_pub.publish(act_msg)

        plan_step_msg = Int32()
        plan_step_msg.data = int(self.current_plan_step_index)
        self.plan_step_pub.publish(plan_step_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DGPPOROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.lease_keep_alive.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
