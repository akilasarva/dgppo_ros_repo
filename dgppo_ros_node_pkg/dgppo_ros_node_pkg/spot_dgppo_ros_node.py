import rclpy
from rclpy.node import Node
import jax.numpy as jnp
import jax.random as jr
import jax
import yaml
import os
import numpy as np
import threading
from rclpy.qos import qos_profile_sensor_data
import json
import math
from typing import NamedTuple, Tuple, Optional, List, Dict

# ROS2 Messages
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16, Int32, Float32MultiArray
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger

# Boston Dynamics
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.exceptions import LeaseUseError
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient)
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_se2_a_tform_b,
)

import time

# DGPPO and LidarEnv components
from .dgppo.dgppo.env.lidar_env.lidar_target import (
    LidarTarget, LidarTargetV1, LidarTargetV2, LidarTargetV3, LidarTargetV4,
)
from .dgppo.dgppo.env.lidar_env.base import LidarEnvState
from .dgppo.dgppo.env.lidar_env.base import get_terrain_id as _compute_terrain_id
from .dgppo.dgppo.algo.dgppo import DGPPO
from .dgppo.dgppo.algo import make_algo
from .dgppo.dgppo.utils.graph import GraphsTuple
from .dgppo.dgppo.utils.typing import Array, Action, AgentState, State
from .dgppo.dgppo.utils.utils import jax_vmap, tree_index

class DGPPOROSNode(Node):
    def __init__(self):
        super().__init__('dgppo_ros_node')

        self.get_logger().info("Initializing DGPPO ROS Node...")

        self.declare_parameter('debug_mode', False)
        self.declare_parameter('current_cluster_id', 1)
        self.declare_parameter('angular_offset_deg', 0.0)
        self.declare_parameter('dry_run', False)  # if True: full pipeline runs but NO motor commands sent
        self.declare_parameter('step_test', False)       # repeating square wave: 0→0.4→0 m/s, 4 s half-period
        self.declare_parameter('step_test_once', False)  # single step: 0→0.4 m/s, holds until disabled
        self._step_test_t0 = None  # set on first step-test tick
        self.declare_parameter('spoof_action', False)    # bypass policy; use fixed action below
        self.declare_parameter('spoof_action_x', 0.0)   # sim-X component [-1,1]: +X = Spot LEFT
        self.declare_parameter('spoof_action_y', 0.0)   # sim-Y component [-1,1]: +Y = Spot FORWARD
        self.declare_parameter('use_projected_vel', False)  # if True: feed projected action vel (not Spot odometry vel) to next step
        self.num_clusters = 4
        self.twod_area_size = 1.5

        _ENV_CLASSES = {
            'LidarTarget':   LidarTarget,
            'LidarTargetV1': LidarTargetV1,
            'LidarTargetV2': LidarTargetV2,
            'LidarTargetV3': LidarTargetV3,
            'LidarTargetV4': LidarTargetV4,
        }

        model_dir = "dgppo/logs/LidarTargetV1/dgppo/terrain_bent_bridge"
        config_path = os.path.join(model_dir, "config.yaml")
        params_path = os.path.join(model_dir, "models")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        step = self._get_model_step(model_dir)

        env_kwargs = config.get("env_kwargs", {})

        self.get_logger().info(f"Loaded config: {config}")

        # Physical LiDAR bins from /processed_ranges — independent of training n_rays
        self.n_rays_phys = 72  # TODO: verify Spot LiDAR bin count
        # Merge class defaults so all keys (including n_rays=32) are present,
        # then apply specific overrides.
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

        self.plan_sequence, self.bearing_map, self.cluster_centroids = self._load_plan_and_cluster_data(model_dir)
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
        self._tablet_has_lease = False  # True while tablet holds lease; plan pauses

        self.is_first_run = True
        self._projected_sim_vel = None  # last action's projected sim-frame velocity (for use_projected_vel mode)

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

        self.scale_2d_3d = 11
        # sim_origin is computed on the first control-loop tick from Spot's live
        # position so the current physical location maps to centroid/scale in sim.
        # Initialise to centroid/scale as a safe fallback (assumes robot at vision origin).
        _start_id = str(self.plan_sequence[0]["start"]) if self.plan_sequence else None
        _c = self.cluster_centroids.get(_start_id, [0.0, 0.0, 0.0]) if _start_id else [0.0, 0.0, 0.0]
        self.sim_origin_x = _c[1] / self.scale_2d_3d
        self.sim_origin_y = _c[0] / self.scale_2d_3d

        self.spot_yaw_pub = self.create_publisher(Float32MultiArray, '/dgppo_spot_yaw', 10)
        self.spot_act_pub = self.create_publisher(Float32MultiArray, '/dgppo_action', 100)
        self.state_debug_pub = self.create_publisher(Float32MultiArray, '/dgppo_state_debug', 10)
        self.plan_step_pub = self.create_publisher(Int32, '/dgppo_plan_step', 10)

        self._take_lease_srv = self.create_service(Trigger, '/dgppo_take_lease', self._take_lease_callback)

        self.timer = self.create_timer(self.env_instance.dt, self.control_loop)

        import datetime
        _log_dir = os.path.join(os.path.dirname(__file__), 'debug_logs')
        os.makedirs(_log_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._debug_log_path = os.path.join(_log_dir, f'dgppo_run_{_ts}.jsonl')
        self._debug_log_file = open(self._debug_log_path, 'w')
        self.get_logger().info(f"Debug log: {self._debug_log_path}")

        self.get_logger().info("DGPPO ROS Node fully initialized and ready.")
        self.get_logger().info("Default mode: Listening for predicted cluster ID on /predicted_cluster_id.")
        self.get_logger().info("To activate debug mode: 'ros2 param set /dgppo_ros_node debug_mode true'")
        self.get_logger().info("When in debug mode: 'ros2 param set /dgppo_ros_node current_cluster_id <new_id>'")

        # yveys: Here we can initialize the Spot robot.
        self.get_logger().info("Initializing the Spot robot.")
        self.sdk = bosdyn.client.create_standard_sdk("understanding-spot")
        self.robot = self.sdk.create_robot("10.0.0.3")
        self.robot.authenticate(username="dcist", password="bbbdddaaaiii")
        self.robot.time_sync.wait_for_sync()

        self.state_client = self.robot.ensure_client("robot-state")
        self.lease_client = self.robot.ensure_client("lease")
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

        # yveys: Take the lease from the tablet.
        self.lease_client.take()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)

        self.get_logger().info("Current state")
        self.get_logger().info(str(self.state_client.get_robot_state()))

        # Cache Spot state in a background thread so the control loop never blocks
        # on a gRPC call.  The poller runs at ~50 Hz (20 ms); the control loop reads
        # self._cached_robot_state which is always fresh enough.
        self._cached_robot_state = None
        self._state_lock = threading.Lock()
        self._state_poller = threading.Thread(target=self._poll_spot_state, daemon=True)
        self._state_poller.start()

        # Command sender: inference writes (v_x, v_y) here; a background thread
        # forwards it to Spot so robot_command gRPC never blocks the control loop.
        self._cmd_vel = (0.0, 0.0)
        self._cmd_lock = threading.Lock()
        self._cmd_sender = threading.Thread(target=self._send_commands, daemon=True)
        self._cmd_sender.start()

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
        """Background thread: forwards _cmd_vel to Spot at ~25 Hz so robot_command
        gRPC never blocks the inference loop."""
        while True:
            with self._cmd_lock:
                v_x, v_y = self._cmd_vel
            try:
                cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=0.0)
                self.command_client.robot_command(command=cmd, end_time_secs=time.time() + 0.5)
                self._tablet_has_lease = False
            except (LeaseUseError, bosdyn.client.lease.NotActiveLeaseError):
                self._tablet_has_lease = True
            except Exception:
                pass
            time.sleep(0.04)  # 25 Hz — well within the 500 ms command expiry window

    # yveys: Spot get_state function for easier access.
    def _get_spot_state(self):
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        with self._state_lock:
            robot_state = self._cached_robot_state
        if robot_state is None:
            # Fallback: blocking call on first tick before cache is warm
            robot_state = self.state_client.get_robot_state()

        kinematic_state = robot_state.kinematic_state
        pos_transforms = kinematic_state.transforms_snapshot

        assert str(pos_transforms) != ""

        tform_body_in_vision = get_se2_a_tform_b(
            pos_transforms, VISION_FRAME_NAME, BODY_FRAME_NAME
        )

        pos = Point(tform_body_in_vision.x, tform_body_in_vision.y)
        vel = Point(kinematic_state.velocity_of_body_in_vision.linear.x, kinematic_state.velocity_of_body_in_vision.linear.y)
        # .angle = body yaw (radians) in vision frame — valid even when stationary
        yaw = tform_body_in_vision.angle

        return pos, vel, yaw

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

    def _get_model_step(self, model_dir):
        model_path = os.path.join(model_dir, "models")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        self.get_logger().info(f"Loading latest model from step: {step}")
        return step

    def _load_plan_and_cluster_data(self, model_dir):
        plan_file_path = "plans/bridge.json"
        if not os.path.exists(plan_file_path):
            self.get_logger().error(f"High level plan file not found at {plan_file_path}")
            return [], {}, {}
        with open(plan_file_path, "r") as f:
            data = json.load(f)
            return data.get("plan_sequence", []), data.get("bearing_map", {}), data.get("centroids", {})

    def _map_cluster_id(self, cluster_id: int) -> int:  ### USE GROUND LIDAR (bridge)
        # Maps raw classifier output → canonical bridge cluster IDs:
        #   0 = open_space, 1 = approach_bridge_0, 2 = on_bridge_0, 3 = exit_bridge_0
        self.get_logger().info(f"cluster id: {cluster_id}")
        if cluster_id in [0, 1]:
            return 0  # open_space
        elif cluster_id in [2, 3, 10, 11]:
            return 1  # approach_bridge_0
        elif cluster_id in [5, 6, 7, 8, 9, 12]:
            return 2  # on_bridge_0
        elif cluster_id in [-1, 4]:
            return 3  # exit_bridge_0
        else:
            return cluster_id

    def ranges_callback(self, msg: Float32MultiArray):
        self.latest_ranges_msg = msg

    def predicted_cluster_callback(self, msg: Int16):
        self.latest_predicted_cluster_id = msg.data

    def terrain_callback(self, msg: Int16):
        # Terrain ID: Road=0, Grass=1, Sidewalk=2
        self.latest_terrain_id = msg.data

    def control_loop(self):
        # angular_offset = self.get_parameter('angular_offset_deg').get_parameter_value().double_value  # used in _build_state_and_graph

        if self.is_first_run:
            # Auto-compute sim_origin from Spot's current position so the physical
            # start location maps to centroid/scale in sim regardless of how far Spot
            # has travelled since boot.
            _pos_init, _, _ = self._get_spot_state()
            _start_id_fr = str(self.plan_sequence[0]["start"]) if self.plan_sequence else None
            _c_fr = self.cluster_centroids.get(_start_id_fr, [0.0, 0.0, 0.0]) if _start_id_fr else [0.0, 0.0, 0.0]
            self.sim_origin_x = (_c_fr[1] + _pos_init.y) / self.scale_2d_3d
            self.sim_origin_y = (_c_fr[0] - _pos_init.x) / self.scale_2d_3d
            self.get_logger().info(
                f"[INIT] physical_start=({_pos_init.x:.3f} m, {_pos_init.y:.3f} m)  "
                f"sim_origin=({self.sim_origin_x:.3f}, {self.sim_origin_y:.3f})  "
                f"sim_start=({_c_fr[1]/self.scale_2d_3d:.3f}, {_c_fr[0]/self.scale_2d_3d:.3f})"
            )

            if self.current_plan_step_index < len(self.plan_sequence):
                plan_key = f"{self.plan_sequence[0]['start']}-{self.plan_sequence[0]['next']}"
                bearing_rad = self.bearing_map.get(plan_key, 0.0)
                self.get_logger().info(f"Initial Bearing from plan: {bearing_rad:.2f} rad ({math.degrees(bearing_rad):.2f} deg)")
            self.is_first_run = False

        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        if self.current_plan_step_index >= len(self.plan_sequence):
            self.get_logger().info("High-level plan is complete. Stopping control loop.")
            with self._cmd_lock:
                self._cmd_vel = (0.0, 0.0)
            if not self.get_parameter('dry_run').get_parameter_value().bool_value:
                try:
                    self.command_client.robot_command(command=RobotCommandBuilder.stop_command())
                except (LeaseUseError, bosdyn.client.lease.NotActiveLeaseError) as e:
                    self.get_logger().warning(f"Could not send stop command — tablet may hold lease. ({type(e).__name__})")
            self.timer.cancel()
            return

        current_plan_step = self.plan_sequence[self.current_plan_step_index]
        expected_start_cluster = current_plan_step["start"]
        expected_next_cluster = current_plan_step["next"]

        missing = []
        if self.latest_ranges_msg is None:
            missing.append('/processed_ranges  ← clustering node must be running and receiving lidar')
        if not debug_mode and self.latest_predicted_cluster_id is None:
            missing.append('/predicted_cluster ← clustering node must be running and receiving lidar')
        if missing:
            self.get_logger().warning(
                'Waiting for topics:\n  ' + '\n  '.join(missing),
                throttle_duration_sec=3.0)
            return

        if debug_mode:
            current_cluster_id = self.get_parameter('current_cluster_id').get_parameter_value().integer_value
            self.get_logger().info(f"DEBUG MODE: Using manual cluster ID {current_cluster_id}", throttle_duration_sec=1.0)
        else:
            current_cluster_id = self.latest_predicted_cluster_id
            self.get_logger().info(f"Default MODE: Using predicted cluster ID {current_cluster_id}", throttle_duration_sec=1.0)

        mapped_current_cluster = self._map_cluster_id(current_cluster_id)
        self.get_logger().info(f"cluster raw={current_cluster_id} mapped={mapped_current_cluster}", throttle_duration_sec=1.0)
        if self._tablet_has_lease:
            self.get_logger().info(
                "Tablet holds lease — plan paused, state updates running. "
                "To reclaim: ros2 service call /dgppo_take_lease std_srvs/srv/Trigger '{}'",
                throttle_duration_sec=3.0
            )
            # Still read and publish state so we stay current.
            try:
                pos, vel, yaw = self._get_spot_state()
                yaw_msg = Float32MultiArray()
                yaw_msg.data = [yaw]
                self.spot_yaw_pub.publish(yaw_msg)
                sim_pos_x = -pos.y / self.scale_2d_3d + self.sim_origin_x
                sim_pos_y =  pos.x / self.scale_2d_3d + self.sim_origin_y
                sim_vel_x = -vel.y / self.scale_2d_3d
                sim_vel_y =  vel.x / self.scale_2d_3d
                self.latest_agent_state = jnp.expand_dims(
                    jnp.array([sim_pos_x, sim_pos_y, sim_vel_x, sim_vel_y], dtype=jnp.float32), axis=0
                )
            except Exception as e:
                self.get_logger().warning(f"State read failed while paused: {e}", throttle_duration_sec=2.0)
            self._projected_sim_vel = None  # re-anchor to real odometry when plan resumes
            return
        if mapped_current_cluster == expected_next_cluster:
            self.current_plan_step_index += 1
            if self.current_plan_step_index >= len(self.plan_sequence):
                self.get_logger().info(f"Plan step complete. Transitioning to cluster {expected_next_cluster}. Plan is now finished.")
            else:
                self.get_logger().info(f"Plan step complete. Transitioning from cluster {expected_start_cluster} to {expected_next_cluster}. Next step is from cluster {self.plan_sequence[self.current_plan_step_index]['start']}.")

        if self.current_plan_step_index >= len(self.plan_sequence):
            return

        raw_ranges_np = np.array(self.latest_ranges_msg.data, dtype=np.float32)
        old_scaled_ranges_np = raw_ranges_np / self.scale_2d_3d
        scaled_ranges_np = old_scaled_ranges_np  # no reversal: clustering node bins by atan2 (CCW), matches visualizer
        # Update agent state from real Spot odometry
        pos, vel, yaw = self._get_spot_state()
        yaw_msg = Float32MultiArray()
        yaw_msg.data = [yaw]
        self.spot_yaw_pub.publish(yaw_msg)
        sim_pos_x = -pos.y / self.scale_2d_3d + self.sim_origin_x   # Spot Y (left)  → Sim X
        sim_pos_y =  pos.x / self.scale_2d_3d + self.sim_origin_y   # Spot X (front) → Sim Y
        if (self.get_parameter('use_projected_vel').get_parameter_value().bool_value
                and self._projected_sim_vel is not None):
            # Single-integrator assumption: robot reaches commanded velocity instantaneously,
            # so feed last action's projected vel rather than Spot's lagged odometry vel.
            sim_vel_x, sim_vel_y = self._projected_sim_vel
        else:
            sim_vel_x = -vel.y / self.scale_2d_3d                    # Spot Y-vel → Sim X-vel
            sim_vel_y =  vel.x / self.scale_2d_3d                    # Spot X-vel → Sim Y-vel
        vel_body_fwd =  vel.x * math.cos(yaw) + vel.y * math.sin(yaw)   # body +x (forward)
        vel_body_lat = -vel.x * math.sin(yaw) + vel.y * math.cos(yaw)   # body +y (left)
        scaled_latest_state_np = np.array([sim_pos_x, sim_pos_y, sim_vel_x, sim_vel_y], dtype=np.float32)
        self.latest_agent_state = jnp.expand_dims(jnp.array(scaled_latest_state_np), axis=0)

        import time as _time
        max_range_val = float(raw_ranges_np.max())
        n_maxed = int((raw_ranges_np >= max_range_val * 0.99).sum())
        angular_offset = self.get_parameter('angular_offset_deg').get_parameter_value().double_value
        bearing_key = f"{expected_start_cluster}-{expected_next_cluster}"
        bearing_val  = self.bearing_map.get(bearing_key, 0.0) + math.radians(angular_offset)
        self._tick_record = {
            't': _time.time(),
            'cluster_raw': int(current_cluster_id),
            'cluster_mapped': int(mapped_current_cluster),
            'plan_start': int(expected_start_cluster),
            'plan_next': int(expected_next_cluster),
            'plan_step': int(self.current_plan_step_index),
            'ranges_min_m': float(raw_ranges_np.min()),
            'ranges_max_m': float(max_range_val),
            'ranges_mean_m': float(raw_ranges_np.mean()),
            'n_beams_at_max': n_maxed,
            'n_beams_total': len(raw_ranges_np),
            'ranges_raw': raw_ranges_np.tolist(),
            'scale_2d_3d': self.scale_2d_3d,
            'pos_spot_x': float(pos.x),
            'pos_spot_y': float(pos.y),
            'vel_spot_x': float(vel.x),
            'vel_spot_y': float(vel.y),
            'yaw_deg': float(math.degrees(yaw)),
            'bearing_deg': float(math.degrees(bearing_val)),
            'angular_offset_deg': float(angular_offset),
            'terrain_id': int(self.latest_terrain_id),
        }
        self.get_logger().info(
            f"CLUSTER raw={current_cluster_id} mapped={mapped_current_cluster} "
            f"plan={expected_start_cluster}→{expected_next_cluster} | "
            f"RANGES min={raw_ranges_np.min():.2f}m mean={raw_ranges_np.mean():.2f}m "
            f"n_at_max={n_maxed}/{len(raw_ranges_np)} | "
            f"BEARING={math.degrees(bearing_val):.1f}° YAW={math.degrees(yaw):.1f}°",
            throttle_duration_sec=0.5,
        )

        graph = self._build_state_and_graph(
            self.latest_agent_state,
            scaled_ranges_np,
            mapped_current_cluster,
            expected_start_cluster,
            expected_next_cluster,
            self.next_cluster_bonus_awarded,
            yaw=yaw,
        )

        self.rng_key, action_key = jr.split(self.rng_key)
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
        action = self.clip_action(action)
        action_flat = [float(a) for a in np.array(action).flatten()]
        self._tick_record['action'] = action_flat
        self._tick_record['inference_ms'] = round(_inf_ms, 2)
        self._tick_record['action_vx_ms'] = action_flat[0] * self.scale_2d_3d if len(action_flat) > 0 else 0.0
        self._tick_record['action_vy_ms'] = action_flat[1] * self.scale_2d_3d if len(action_flat) > 1 else 0.0
        self._debug_log_file.write(json.dumps(self._tick_record) + '\n')
        self._debug_log_file.flush()

        new_movement_targets = jnp.squeeze(self.agent_step_euler(self.latest_agent_state, action), axis=0)
        # Store projected sim-frame velocity for next tick (indices [2,3] of Euler step output).
        self._projected_sim_vel = (float(new_movement_targets[2]), float(new_movement_targets[3]))

        reward, bonus_awarded_updated = self.env_instance.get_reward(graph, action)
        if bonus_awarded_updated.size == 0:
            self.get_logger().warning("Received an empty bonus array from get_reward. Resetting.")
            self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        else:
            self.next_cluster_bonus_awarded = bonus_awarded_updated

        # new_movement_targets[2:4] = velocity in sim space (vel = action * 0.5)
        # Reverse sim→Spot axis mapping: v_spot_x = sim_vel_y, v_spot_y = -sim_vel_x
        # Clamp to Spot's safe walking speed (SDK hard limit is 2.0 m/s)
        SPOT_MAX_VEL = 1  # m/s — conservative safe limit
        v_x_raw = float(new_movement_targets[3]) * self.scale_2d_3d
        v_y_raw = -float(new_movement_targets[2]) * self.scale_2d_3d
        max_component = max(abs(v_x_raw), abs(v_y_raw))
        scale = min(1.0, SPOT_MAX_VEL / max_component) if max_component > 0 else 1.0
        v_x_target = v_x_raw * scale
        v_y_target = v_y_raw * scale

        # ── Step-test overrides ───────────────────────────────────────────────
        STEP_VX = 0.4  # m/s forward — safe walking speed for both modes

        if self.get_parameter('step_test_once').get_parameter_value().bool_value:
            # Single step: command STEP_VX and hold.
            # Tells you:
            #   pure delay  → time from step edge to first detectable motion in reported vel
            #   rise time   → time for reported vel to climb from 0 to ~90% of STEP_VX
            # Disable with: ros2 param set /dgppo_ros_node step_test_once false
            if self._step_test_t0 is None:
                self._step_test_t0 = time.time()
            v_x_target = STEP_VX
            v_y_target = 0.0
            self.get_logger().info(
                f'[STEP ONCE] t={time.time() - self._step_test_t0:.2f}s  vx={v_x_target:.2f} m/s',
                throttle_duration_sec=0.5,
            )
        elif self.get_parameter('step_test').get_parameter_value().bool_value:
            # Repeating square wave: 0 → STEP_VX → 0, 4 s per half-cycle.
            # Tells you:
            #   phase lag   → cross-correlation peak (automated number on the plot)
            # 4 s half-period >> expected rise time (~300-500 ms) so Spot fully
            # settles before each transition — clean edges for xcorr.
            # Run for ≥30 s (3+ full cycles) for a stable estimate.
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
            self._step_test_t0 = None  # reset timer when both modes are off

        _dbg = Float32MultiArray()
        _dbg.data = [
            float(pos.x),        float(pos.y),        # [0,1]  vision frame pos (m)
            float(vel.x),        float(vel.y),        # [2,3]  vision frame vel (m/s)
            float(vel_body_fwd), float(vel_body_lat), # [4,5]  body frame vel: fwd, left (m/s)
            float(sim_pos_x),    float(sim_pos_y),    # [6,7]  DGPPO sim pos (scaled)
            float(sim_vel_x),    float(sim_vel_y),    # [8,9]  DGPPO sim vel (scaled)
            float(v_x_target),   float(v_y_target),   # [10,11] cmd to Spot, vision frame (m/s)
            float(action_flat[0]) if len(action_flat) > 0 else 0.0,  # [12] raw policy a[0] (sim-X → right)
            float(action_flat[1]) if len(action_flat) > 1 else 0.0,  # [13] raw policy a[1] (sim-Y → fwd)
        ]
        self.state_debug_pub.publish(_dbg)

        act_msg = Float32MultiArray()
        act_msg.data = [-v_y_target, v_x_target]  # [right, fwd] matches visualizer canvas convention
        self.spot_act_pub.publish(act_msg)

        plan_step_msg = Int32()
        plan_step_msg.data = int(self.current_plan_step_index)
        self.plan_step_pub.publish(plan_step_msg)

        dry_run = self.get_parameter('dry_run').get_parameter_value().bool_value
        if dry_run:
            self.get_logger().info(
                f"[DRY RUN] vx={v_x_target:.3f}  vy={v_y_target:.3f}  (no command sent)",
                throttle_duration_sec=0.5)
        else:
            with self._cmd_lock:
                self._cmd_vel = (v_x_target, v_y_target)
            self.get_logger().info(
                f"vx={v_x_target:.3f}  vy={v_y_target:.3f}",
                throttle_duration_sec=0.5)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        return self.env_instance.agent_step_euler(agent_states, action)

    def state_lim(self) -> Tuple[State, State]:
        return self.env_instance.state_lim()

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def clip_state(self, state: State) -> State:
        lower_limit, upper_limit = self.state_lim()
        return jnp.clip(state, lower_limit, upper_limit)

    def clip_action(self, action: Action) -> Action:
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    def _build_state_and_graph(self, agent_state_np: np.ndarray, scaled_ranges: np.ndarray,
                               mapped_current_cluster_id: int, mapped_start_cluster_id: int,
                               mapped_next_cluster_id: int, bonus_awarded_updated: jnp.ndarray,
                               yaw: float = 0.0) -> GraphsTuple:
        self.get_logger().info(f"Agent state (scaled): {agent_state_np}", throttle_duration_sec=0.5)

        n_rays = self.env_instance.params['n_rays']  # 32
        agent_pos_2d = np.array(agent_state_np[0, :2])

        # ── 1. Obstacle hits: resample n_rays_phys bins → n_rays (32) training beams ──
        # LiDAR ranges are in body frame. Rotate into world (sim) frame by adding yaw.
        # Spot yaw=0 = facing +X; sim forward = +Y, so add π/2 to align conventions.
        angles_phys = np.linspace(0, 2 * np.pi, self.n_rays_phys, endpoint=False)
        angles_beam = np.linspace(-np.pi, np.pi - 2 * np.pi / n_rays, n_rays)
        # Sensor bin to look up for training beam at θ_beam (sim world angle):
        #   φ_sensor = π/2 + yaw − θ_beam
        # Upside-down mount: +Y_sensor = Spot right → φ_body = −φ_sensor.
        # Spot +X = Sim +Y: body→world heading offset is π/2, plus robot yaw.
        # Training lidar is world-frame (no agent yaw in training dirs).
        ranges_res = np.interp(np.mod(np.pi / 2 + yaw - angles_beam, 2 * np.pi), angles_phys, scaled_ranges)
        # angles_beam are sim world angles; hits are world-frame positions.
        obs_hits = np.stack([
            agent_pos_2d[0] + ranges_res * np.cos(angles_beam),
            agent_pos_2d[1] + ranges_res * np.sin(angles_beam),
        ], axis=1).astype(np.float32)  # (n_rays, 2)

        # ── 2. Terrain boundary hits: max-range endpoints (sense_range along each beam) ──
        # Placing hits at agent_pos + sense_range*dir matches the training convention for
        # "no boundary found" (alpha=1.0 in _get_semantic_lidar_single_real / base.py:ends).
        # This is position-independent and produces proximity=0 in the reward functions,
        # unlike world-origin zeros which create a position-dependent directional artifact.
        _sense_range = self.env_instance.params['comm_radius']
        bnd_hits = np.stack([
            agent_pos_2d[0] + _sense_range * np.cos(angles_beam),
            agent_pos_2d[1] + _sense_range * np.sin(angles_beam),
        ], axis=1).astype(np.float32)  # (n_rays, 2) — sense_range from agent along each beam

        # ── 3. Flat semantic lidar arrays ─────────────────────────────────────────
        all_hit_positions = np.concatenate([obs_hits, bnd_hits], axis=0)  # (2*n_rays, 2)
        all_terrain_ids   = np.ones(2 * n_rays, dtype=np.int32)           # (2*n_rays,) Grass default

        # ── 4. Agent terrain OH from /current_terrain topic (Road=0, Grass=1, Sidewalk=2) ──
        current_terrain_oh = jax.nn.one_hot(self.latest_terrain_id, 3)  # (3,)

        # ── 5. Cluster one-hots & bearing ─────────────────────────────────────────
        current_cluster_oh = jax.nn.one_hot(mapped_current_cluster_id, self.num_clusters)
        start_cluster_oh   = jax.nn.one_hot(mapped_start_cluster_id,   self.num_clusters)
        next_cluster_oh    = jax.nn.one_hot(mapped_next_cluster_id,    self.num_clusters)

        angular_offset = self.get_parameter('angular_offset_deg').get_parameter_value().double_value
        key = f"{mapped_start_cluster_id}-{mapped_next_cluster_id}"
        bearing_value = self.bearing_map.get(key, 0.0) + math.radians(angular_offset)
        self.get_logger().info(
            f"Start:{mapped_start_cluster_id} Cur:{mapped_current_cluster_id} "
            f"Next:{mapped_next_cluster_id} Bearing:{bearing_value:.3f}",
            throttle_duration_sec=0.5,
        )

        goal_state_np = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        env_state = LidarEnvState(
            agent=agent_state_np,
            goal=jnp.array([goal_state_np]),
            obstacle=None,
            bearing=jnp.array([bearing_value]),
            current_cluster_oh=jnp.array([current_cluster_oh]),
            start_cluster_oh=jnp.array([start_cluster_oh]),
            next_cluster_oh=jnp.array([next_cluster_oh]),
            next_cluster_bonus_awarded=bonus_awarded_updated,
            # New terrain fields (terrain_bent_bridge)
            current_terrain_oh=jnp.array([current_terrain_oh]),        # (1, 3)
            lidar_hit_terrain_ids=jnp.array(all_terrain_ids),          # (2*n_rays,)
            lidar_hit_positions=jnp.array(all_hit_positions),          # (2*n_rays, 2)
            # Bridge geometry scalars (zeros = no geometry during inference)
            bridge_center=jnp.zeros(2),
            bridge_length=jnp.array(0.0),
            bridge_gap_width=jnp.array(0.0),
            bridge_wall_thickness=jnp.array(0.0),
            bridge_theta=jnp.array(0.0),
            bridge_bend_angle=jnp.array(0.0),
            terrain_config=jnp.array(1, dtype=jnp.int32),
        )

        # get_graph takes (n_agents, 2*n_rays, 2) and selects top_k per type internally
        lidar_data_batched = jnp.array(all_hit_positions[np.newaxis, :, :])  # (1, 2*n_rays, 2)
        graph = self.env_instance.get_graph(env_state, lidar_data_batched)
        return graph

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

