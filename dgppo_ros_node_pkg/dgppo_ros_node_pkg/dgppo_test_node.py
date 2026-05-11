#!/usr/bin/env python3
"""
DGPPO Test Node — cart/bench testing without CARLA or Spot SDK.

Runs the full DGPPO policy using real sensor data and publishes its output
to the debug topics consumed by dgppo_debug_visualizer.py.  No robot
commands are issued; the node is purely a policy runner + data source for
the visualiser.

Required topics (must be running externally):
  /processed_ranges   Float32MultiArray  — pre-processed LiDAR range bins
  /predicted_cluster  Int16              — cluster classifier output

Optional topics (defaults used if absent):
  /current_terrain    Int32              — terrain type (0=Road,1=Grass,2=Sidewalk)
                                           defaults to Grass (1)
  /livox/imu          sensor_msgs/Imu   — used only to extract yaw heading for viz

Debug overrides (ROS params):
  debug_mode          bool  (default false)  — if true, use manual cluster
  current_cluster_id  int   (default 1)      — cluster to use in debug_mode
  angular_offset_deg  float (default 0.0)

Published topics (consumed by dgppo_debug_visualizer.py):
  /dgppo_action       Float32MultiArray  [a0, a1]
  /dgppo_plan_step    Int32
  /dgppo_imu_yaw      Float32            yaw in radians (0 = IMU reference dir)

Usage:
  # from the dgppo_ros_node_pkg/dgppo_ros_node_pkg directory:
  python3 dgppo_test_node.py

  # if you don't have a cluster classifier running, force a cluster:
  ros2 param set /dgppo_test_node debug_mode true
  ros2 param set /dgppo_test_node current_cluster_id 1

  # if you don't have a terrain node running, publish manually:
  ros2 topic pub /current_terrain std_msgs/msg/Int32 "{data: 1}" -r 1
"""

import rclpy
from rclpy.node import Node
import jax.numpy as jnp
import jax.random as jr
import jax
import yaml
import os
import numpy as np
from rclpy.qos import qos_profile_sensor_data
import json
import math
from typing import Tuple

# ROS2 messages
from std_msgs.msg import Int16, Int32, Float32MultiArray
from sensor_msgs.msg import Imu

# DGPPO components (relative imports work when running as part of the package)
from .dgppo.dgppo.env.lidar_env.lidar_target import LidarTarget, LidarEnvState
from .dgppo.dgppo.algo import make_algo
from .dgppo.dgppo.utils.graph import GraphsTuple
from .dgppo.dgppo.utils.typing import Array, Action, AgentState, State



class DGPPOTestNode(Node):
    def __init__(self):
        super().__init__('dgppo_test_node')
        self.get_logger().info("Initializing DGPPO Test Node (no robot backend)...")

        self.declare_parameter('debug_mode', False)
        self.declare_parameter('current_cluster_id', 1)
        self.declare_parameter('angular_offset_deg', 0.0)

        self.num_clusters = 4
        self.dt = 1.0 / 30
        self.twod_area_size = 1.5

        # ── Load model ──────────────────────────────────────────────────────────
        model_dir = "dgppo/logs/LidarTarget/dgppo/terrain_bent_bridge"  # TODO: set before running
        config_path = os.path.join(model_dir, "config.yaml")
        params_path = os.path.join(model_dir, "models")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        step = self._get_model_step(model_dir)
        env_kwargs = config.get("env_kwargs", {})
        self.get_logger().info(f"Loaded config: {config}")

        self.n_rays_phys = 72  # bins in /processed_ranges
        merged_params = {**LidarTarget.PARAMS, **env_kwargs.get('params', {})}
        merged_params['top_k_rays'] = 8
        merged_params['comm_radius'] = 0.5

        self.env_instance = LidarTarget(
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

        self.plan_sequence, self.bearing_map, self.cluster_centroids = \
            self._load_plan_and_cluster_data(model_dir)
        self.current_plan_step_index = 0

        self.algo.load(params_path, step=step)
        self.rng_key = jr.PRNGKey(config.get('seed', 0))
        self.rnn_state = self.algo.init_rnn_state

        # ── State ────────────────────────────────────────────────────────────────
        self.current_agent_state = None  # initialised on first control loop tick
        self.latest_ranges_msg = None
        self.latest_predicted_cluster_id = None
        self.latest_terrain_id = 1       # Grass default
        self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents,
                                                     dtype=jnp.bool_)
        self.is_first_run = True

        # Scaling constants — only needed to seed initial position from centroid.
        # The cart doesn't need these for movement; dead-reckoning is in model space.
        self.scale_2d_3d = 11.0   # matches training environment
        self.origin_x = 0.0
        self.origin_y = 0.0

        # IMU yaw integration (Livox does not populate orientation quaternion)
        self._imu_yaw = 0.0
        self._imu_last_stamp = None

        # ── Subscriptions ────────────────────────────────────────────────────────
        self.ranges_sub = self.create_subscription(
            Float32MultiArray, '/processed_ranges',
            self._cb_ranges, qos_profile=qos_profile_sensor_data
        )
        self.cluster_sub = self.create_subscription(
            Int16, '/predicted_cluster',
            self._cb_cluster, 10
        )
        self.terrain_sub = self.create_subscription(
            Int32, '/current_terrain',
            self._cb_terrain, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/livox/imu',
            self._cb_imu, 10  # RELIABLE depth-10, matches Livox driver QoS
        )

        # ── Publishers (visualiser listens to these) ──────────────────────────
        self.action_pub      = self.create_publisher(Float32MultiArray, '/dgppo_action',      10)
        self.planstep_pub    = self.create_publisher(Int32,             '/dgppo_plan_step',   10)
        self.imu_yaw_pub     = self.create_publisher(Float32MultiArray, '/dgppo_imu_yaw',     10)
        self.lidar_all_pub   = self.create_publisher(Float32MultiArray, '/dgppo_lidar_all',   10)
        self.lidar_topk_pub  = self.create_publisher(Float32MultiArray, '/dgppo_lidar_topk',  10)

        self.timer = self.create_timer(0.1, self._control_loop)
        self.get_logger().info("DGPPO Test Node ready.  Waiting for sensor data...")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_ranges(self, msg):
        self.latest_ranges_msg = msg

    def _cb_cluster(self, msg):
        self.latest_predicted_cluster_id = msg.data

    def _cb_terrain(self, msg):
        self.latest_terrain_id = msg.data

    def _cb_imu(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._imu_last_stamp is not None:
            dt = stamp - self._imu_last_stamp
            if 0.0 < dt < 1.0:
                self._imu_yaw -= msg.angular_velocity.z * dt  # negated: Livox mounted upside-down
        self._imu_last_stamp = stamp
        out = Float32MultiArray()
        out.data = [float(self._imu_yaw)]
        self.imu_yaw_pub.publish(out)

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        # ── Seed initial agent state from plan centroid ───────────────────────
        if self.is_first_run:
            if self.current_plan_step_index < len(self.plan_sequence):
                start_id = str(self.plan_sequence[self.current_plan_step_index]["start"])
                if start_id in self.cluster_centroids:
                    c = self.cluster_centroids[start_id]
                    sx = (c[1] - self.origin_y) / self.scale_2d_3d
                    sy = (c[0] - self.origin_x) / self.scale_2d_3d
                    self.current_agent_state = jnp.expand_dims(
                        jnp.array([sx, sy, 0.0, 0.0], dtype=np.float32), axis=0
                    )
                    self.get_logger().info(
                        f"Initial state set from cluster {start_id} centroid: ({sx:.3f}, {sy:.3f})"
                    )
                else:
                    # Fallback: start at origin of model space
                    self.current_agent_state = jnp.zeros((1, 4), dtype=np.float32)
                    self.get_logger().warning(
                        f"Centroid for cluster {start_id} not found; starting at model origin."
                    )
            else:
                self.current_agent_state = jnp.zeros((1, 4), dtype=np.float32)
            self.is_first_run = False

        # ── Guard: need ranges ────────────────────────────────────────────────
        if self.latest_ranges_msg is None:
            self.get_logger().warning("Waiting for /processed_ranges...")
            return

        # ── Plan complete ─────────────────────────────────────────────────────
        if self.current_plan_step_index >= len(self.plan_sequence):
            self.get_logger().info("Plan complete. Publishing zero action.")
            self._publish_zero()
            self.timer.cancel()
            return

        current_plan_step    = self.plan_sequence[self.current_plan_step_index]
        expected_start       = current_plan_step["start"]
        expected_next        = current_plan_step["next"]

        # ── Cluster ID ───────────────────────────────────────────────────────
        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        if debug_mode:
            current_cluster_id = self.get_parameter('current_cluster_id') \
                                     .get_parameter_value().integer_value
            self.get_logger().info(f"DEBUG MODE: cluster={current_cluster_id}")
        else:
            if self.latest_predicted_cluster_id is None:
                self.get_logger().warning(
                    "Waiting for /predicted_cluster.  "
                    "Use 'ros2 param set /dgppo_test_node debug_mode true' to override."
                )
                return
            current_cluster_id = self.latest_predicted_cluster_id

        mapped = self._map_cluster_id(current_cluster_id)

        # ── Plan transition check ────────────────────────────────────────────
        if mapped == expected_next:
            self.current_plan_step_index += 1
            if self.current_plan_step_index < len(self.plan_sequence):
                self.get_logger().info(
                    f"Transition: {expected_start}→{expected_next}. "
                    f"Next: {self.plan_sequence[self.current_plan_step_index]}"
                )
            else:
                self.get_logger().info("Plan complete.")
            return

        if self.current_plan_step_index >= len(self.plan_sequence):
            return

        # ── Build graph & run policy ─────────────────────────────────────────
        scaled_ranges = np.array(self.latest_ranges_msg.data, dtype=np.float32) \
                        / self.scale_2d_3d

        graph = self._build_graph(
            self.current_agent_state,
            scaled_ranges,
            mapped,
            expected_start,
            expected_next,
            self.next_cluster_bonus_awarded,
        )

        self.rng_key, _ = jr.split(self.rng_key)
        action, new_rnn_state = self.algo.act(
            graph=graph,
            rnn_state=self.rnn_state,
            params={'policy': self.algo.policy_train_state.params}
        )
        self.rnn_state = new_rnn_state
        action = self._clip_action(action)

        # Dead-reckon agent state forward (no real movement command)
        self.current_agent_state = self._euler_step(self.current_agent_state, action)

        # Reset position + RNN when agent reaches the far edge of model space.
        # In training this boundary is never reached without a cluster transition;
        # without one the RNN diverges and produces OOD actions.
        if float(self.current_agent_state[0, 1]) >= self.twod_area_size * 0.9:
            start_id = str(self.plan_sequence[self.current_plan_step_index]["start"])
            if start_id in self.cluster_centroids:
                c = self.cluster_centroids[start_id]
                sx = (c[1] - self.origin_y) / self.scale_2d_3d
                sy = (c[0] - self.origin_x) / self.scale_2d_3d
                self.current_agent_state = jnp.expand_dims(
                    jnp.array([sx, sy, 0.0, 0.0], dtype=np.float32), axis=0
                )
            else:
                self.current_agent_state = jnp.zeros((1, 4), dtype=np.float32)
            self.rnn_state = self.algo.init_rnn_state
            self.get_logger().info("Dead-reckoning boundary reached — resetting position and RNN.")

        reward, bonus = self.env_instance.get_reward(graph, action)
        self.next_cluster_bonus_awarded = (
            jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
            if bonus.size == 0 else bonus
        )

        # ── Publish debug topics ─────────────────────────────────────────────
        act_msg = Float32MultiArray()
        act_msg.data = [float(action[0, 0]), float(action[0, 1])]
        self.action_pub.publish(act_msg)

        step_msg = Int32()
        step_msg.data = self.current_plan_step_index
        self.planstep_pub.publish(step_msg)

        # ── Publish LiDAR hit positions for visualiser ───────────────────────
        n_rays  = self.env_instance.params['n_rays']
        top_k   = self.env_instance.params['top_k_rays']
        agent_pos_2d = np.array(self.current_agent_state[0, :2])
        angles_phys  = np.linspace(0, 2 * np.pi, self.n_rays_phys, endpoint=False)
        angles_beam  = np.linspace(-np.pi, np.pi - 2 * np.pi / n_rays, n_rays)
        ranges_sc    = np.array(self.latest_ranges_msg.data, dtype=np.float32) / self.scale_2d_3d
        ranges_res   = np.interp(np.mod(angles_beam, 2 * np.pi), angles_phys, ranges_sc)
        obs_hits = np.stack([
            agent_pos_2d[0] + ranges_res * np.sin(angles_beam),
            agent_pos_2d[1] + ranges_res * np.cos(angles_beam),
        ], axis=1)
        rel_hits = obs_hits - agent_pos_2d  # centre on agent for display

        all_msg = Float32MultiArray()
        all_msg.data = rel_hits.flatten().tolist()
        self.lidar_all_pub.publish(all_msg)

        dists = np.linalg.norm(rel_hits, axis=-1)
        topk_idx = np.argsort(dists)[:top_k]
        topk_msg = Float32MultiArray()
        topk_msg.data = rel_hits[topk_idx].flatten().tolist()
        self.lidar_topk_pub.publish(topk_msg)

        self.get_logger().info(
            f"Action: [{action[0,0]:.3f}, {action[0,1]:.3f}]  "
            f"Terrain: {self.latest_terrain_id}  Cluster: {current_cluster_id}→{mapped}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_zero(self):
        act_msg = Float32MultiArray()
        act_msg.data = [0.0, 0.0]
        self.action_pub.publish(act_msg)

    def _get_model_step(self, model_dir):
        model_path = os.path.join(model_dir, "models")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        steps = [int(m) for m in os.listdir(model_path) if m.isdigit()]
        step = max(steps)
        self.get_logger().info(f"Loading model step: {step}")
        return step

    def _load_plan_and_cluster_data(self, model_dir):
        plan_path = "plans/bridge.json"
        if not os.path.exists(plan_path):
            self.get_logger().error(f"Plan file not found: {plan_path}")
            return [], {}, {}
        with open(plan_path) as f:
            data = json.load(f)
        return data.get("plan_sequence", []), data.get("bearing_map", {}), data.get("centroids", {})

    def _map_cluster_id(self, cluster_id: int) -> int:
        if cluster_id in [2, 3]:      return 1
        if cluster_id in [5,6,7,8,9]: return 2
        if cluster_id in [-1, 4]:     return 3
        if cluster_id in [0, 1]:      return 0
        return cluster_id

    def _euler_step(self, agent_states, action):
        vel = action * 0.5
        next_pos = agent_states[:, :2] + vel * self.dt
        new_state = jnp.concatenate([next_pos, vel], axis=1)
        lo = jnp.array([0., 0., -0.5, -0.5])
        hi = jnp.array([self.twod_area_size, self.twod_area_size, 0.5, 0.5])
        return jnp.clip(new_state, lo, hi)

    def _clip_action(self, action):
        return jnp.clip(action, -1.0, 1.0)

    def _build_graph(self, agent_state_np, scaled_ranges, mapped_current,
                     mapped_start, mapped_next, bonus):
        n_rays = self.env_instance.params['n_rays']
        agent_pos_2d = np.array(agent_state_np[0, :2])

        angles_phys = np.linspace(0, 2 * np.pi, self.n_rays_phys, endpoint=False)
        angles_beam = np.linspace(-np.pi, np.pi - 2 * np.pi / n_rays, n_rays)
        ranges_res  = np.interp(np.mod(angles_beam, 2 * np.pi), angles_phys, scaled_ranges)
        obs_hits = np.stack([
            agent_pos_2d[0] + ranges_res * np.sin(angles_beam),
            agent_pos_2d[1] + ranges_res * np.cos(angles_beam),
        ], axis=1).astype(np.float32)

        bnd_hits          = np.zeros((n_rays, 2), dtype=np.float32)
        all_hit_positions = np.concatenate([obs_hits, bnd_hits], axis=0)
        all_terrain_ids   = np.ones(2 * n_rays, dtype=np.int32)

        current_terrain_oh = jax.nn.one_hot(self.latest_terrain_id, 3)
        current_cluster_oh = jax.nn.one_hot(mapped_current, self.num_clusters)
        start_cluster_oh   = jax.nn.one_hot(mapped_start,   self.num_clusters)
        next_cluster_oh    = jax.nn.one_hot(mapped_next,    self.num_clusters)

        angular_offset = self.get_parameter('angular_offset_deg').get_parameter_value().double_value
        key = f"{mapped_start}-{mapped_next}"
        bearing_value = self.bearing_map.get(key, 0.0) + math.radians(angular_offset)
        self.get_logger().info(
            f"Start:{mapped_start} Cur:{mapped_current} Next:{mapped_next} "
            f"Bearing:{bearing_value:.3f} rad"
        )

        env_state = LidarEnvState(
            agent=agent_state_np,
            goal=jnp.array([[0., 0., 0., 0.]]),
            obstacle=None,
            bearing=jnp.array([bearing_value]),
            current_cluster_oh=jnp.array([current_cluster_oh]),
            start_cluster_oh=jnp.array([start_cluster_oh]),
            next_cluster_oh=jnp.array([next_cluster_oh]),
            next_cluster_bonus_awarded=bonus,
            current_terrain_oh=jnp.array([current_terrain_oh]),
            lidar_hit_terrain_ids=jnp.array(all_terrain_ids),
            lidar_hit_positions=jnp.array(all_hit_positions),
            bridge_center=jnp.zeros(2),
            bridge_length=jnp.array(0.0),
            bridge_gap_width=jnp.array(0.0),
            bridge_wall_thickness=jnp.array(0.0),
            bridge_theta=jnp.array(0.0),
            bridge_bend_angle=jnp.array(0.0),
            terrain_config=jnp.array(1, dtype=jnp.int32),
        )

        lidar_batched = jnp.array(all_hit_positions[np.newaxis, :, :])
        return self.env_instance.get_graph(env_state, lidar_batched)


def main(args=None):
    rclpy.init(args=args)
    node = DGPPOTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
