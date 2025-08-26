# import rclpy
# from rclpy.node import Node
# import jax.numpy as jnp
# import jax.random as jr
# import jax
# import yaml
# import os
# import numpy as np
# from rclpy.qos import qos_profile_sensor_data
# import json

# # ROS2 Messages
# from geometry_msgs.msg import Twist
# from std_msgs.msg import Int16, Float32MultiArray
# from nav_msgs.msg import Odometry

# # DGPPO and LidarEnv components
# from .dgppo.dgppo.env.lidar_env.lidar_target import LidarTarget, LidarEnvState
# from .dgppo.dgppo.algo.dgppo import DGPPO
# from .dgppo.dgppo.algo import make_algo
# from .dgppo.dgppo.utils.graph import GraphsTuple
# from .dgppo.dgppo.utils.typing import Array, Action
# from .dgppo.dgppo.utils.utils import jax_vmap, tree_index

# class DGPPOROSNode(Node):
#     def __init__(self):
#         super().__init__('dgppo_ros_node')

#         self.get_logger().info("Initializing DGPPO ROS Node...")

#         model_dir = "dgppo/logs/LidarTarget/dgppo/bearing_new"
#         config_path = os.path.join(model_dir, "config.yaml")
#         params_path = os.path.join(model_dir, "models")
        
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)

#         step = self._get_model_step(model_dir)

#         env_kwargs = config.get("env_kwargs", {})
        
#         # --- VERIFICATION STEP ---
#         self.get_logger().info(f"Loaded config: {config}")
#         # -------------------------

#         num_range_bins = 32
#         if 'params' not in env_kwargs:
#             env_kwargs['params'] = {}
#         env_kwargs['params']['num_ranges'] = num_range_bins
#         env_kwargs['params']['top_k_rays'] = 8
        
#         env_kwargs['params']['comm_radius'] = 1.5
        
#         self.env_instance = LidarTarget(
#             num_agents=config.get('num_agents'),
#             params=env_kwargs.get('params', {}),
#             **{k: v for k, v in env_kwargs.items() if k != 'params'}
#         )

#         algo_kwargs = config.get("algo_kwargs", {})
#         self.algo = make_algo(
#             algo=config.get('algo'),
#             env=self.env_instance,
#             node_dim=self.env_instance.node_dim,
#             edge_dim=self.env_instance.edge_dim,
#             state_dim=self.env_instance.state_dim,
#             action_dim=self.env_instance.action_dim,
#             n_agents=self.env_instance.num_agents,
#             **algo_kwargs
#         )
        
#         self.cluster_bearing_map = self._load_cluster_bearings(model_dir)
#         self.algo.load(params_path, step=step)

#         self.rng_key = jr.PRNGKey(config.get('seed', 0))
#         self.rnn_state = self.algo.init_rnn_state

#         self.latest_ranges_msg = None
#         self.latest_agent_state_msg = None
#         self.latest_cluster_id_msg = None

#         self.publisher_ = self.create_publisher(Twist, '/carla/ego_vehicle/twist', 10)
#         self.agent_state_sub = self.create_subscription(
#             Odometry,
#             '/carla/ego_vehicle/odometry',
#             self.agent_state_callback,
#             10
#         )
        
#         self.ranges_sub = self.create_subscription(
#             Float32MultiArray,
#             '/processed_ranges',
#             self.ranges_callback,
#             qos_profile=qos_profile_sensor_data
#         )
        
#         self.cluster_id_sub = self.create_subscription(
#             Int16,
#             '/predicted_cluster',
#             self.cluster_id_callback,
#             10
#         )
        
#         self.timer = self.create_timer(0.1, self.control_loop)
        
#         self.get_logger().info("DGPPO ROS Node fully initialized and ready.")

#     def _get_model_step(self, model_dir):
#         model_path = os.path.join(model_dir, "models")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model directory not found at {model_path}")
#         models = os.listdir(model_path)
#         step = max([int(model) for model in models if model.isdigit()])
#         self.get_logger().info(f"Loading latest model from step: {step}")
#         return step

#     def _load_cluster_bearings(self, model_dir):
#         bearing_file_path = os.path.join(model_dir, "highlevel_plan.json")
#         if not os.path.exists(bearing_file_path):
#             self.get_logger().error(f"High level plan file not found at {bearing_file_path}")
#             return {}
#         with open(bearing_file_path, "r") as f:
#             return json.load(f)

#     def ranges_callback(self, msg: Float32MultiArray):
#         self.latest_ranges_msg = msg

#     def agent_state_callback(self, msg: Odometry):
#         self.latest_agent_state_msg = msg

#     def cluster_id_callback(self, msg: Int16):
#         self.latest_cluster_id_msg = msg

#     def control_loop(self):
#         if self.latest_ranges_msg is None or self.latest_agent_state_msg is None or self.latest_cluster_id_msg is None:
#             self.get_logger().warning("Waiting for all sensor data...")
#             return

#         graph = self._build_state_and_graph(
#             self.latest_agent_state_msg,
#             self.latest_ranges_msg,
#             self.latest_cluster_id_msg
#         )

#         self.rng_key, action_key = jr.split(self.rng_key)
        
#         action, new_rnn_state = self.algo.act(
#             graph=graph,
#             rnn_state=self.rnn_state,
#             params={'policy': self.algo.policy_train_state.params}        )

#         self.rnn_state = new_rnn_state
#         # --- FIX: Squeeze the action array to remove the batch dimension ---
#         action_squeezed = jnp.squeeze(action, axis=0) 
#         twist_msg = self._action_to_twist(action_squeezed)
#         self.publisher_.publish(twist_msg)

#     def _build_state_and_graph(self, odom_msg: Odometry, ranges_msg: Float32MultiArray, cluster_id_msg: Int16) -> GraphsTuple:
#         self.get_logger().info(f"Processing lidar data...")

#         pos = odom_msg.pose.pose.position
#         vel = odom_msg.twist.twist.linear
#         agent_state_np = np.array([pos.x, pos.y, vel.x, vel.y], dtype=np.float32)

#         ranges_np = np.array(ranges_msg.data, dtype=np.float32)
        
#         self.get_logger().info(f"Received ranges array shape: {ranges_np.shape}")
        
#         num_ranges = self.env_instance.params['num_ranges']
#         top_k = self.env_instance.params['top_k_rays']
        
#         if ranges_np.shape[0] != num_ranges:
#             self.get_logger().warn(f"Received {ranges_np.shape[0]} ranges, but expected {num_ranges}. This may cause errors.")
        
#         angle_increment = (2 * np.pi) / num_ranges
#         angles = np.arange(num_ranges) * angle_increment
        
#         x_coords = ranges_np * np.cos(angles)
#         y_coords = ranges_np * np.sin(angles)
        
#         self.get_logger().info(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}")

#         lidar_data_np = np.stack([x_coords, y_coords], axis=1)
#         self.get_logger().info(f"Stacked lidar_data_np shape: {lidar_data_np.shape}")

#         distances = np.linalg.norm(lidar_data_np, axis=1)
#         sorted_indices = np.argsort(distances)
#         final_lidar_data = lidar_data_np[sorted_indices[:top_k]]
        
#         self.get_logger().info(f"Filtered final_lidar_data shape: {final_lidar_data.shape}")
        
#         if final_lidar_data.shape[0] < top_k:
#             padding_size = top_k - final_lidar_data.shape[0]
#             padding = np.zeros((padding_size, 2), dtype=np.float32)
#             final_lidar_data = np.concatenate([final_lidar_data, padding], axis=0)
#             self.get_logger().info(f"Padded final_lidar_data shape: {final_lidar_data.shape}")
        
#         final_lidar_data = final_lidar_data[np.newaxis, :, :]
#         self.get_logger().info(f"Reshaped final_lidar_data shape before JAX conversion: {final_lidar_data.shape}")
        
#         cluster_id = cluster_id_msg.data
#         num_clusters = 4
#         current_cluster_oh = jnp.array(jax.nn.one_hot(cluster_id, num_clusters))
        
#         next_cluster_id = (cluster_id + 1) % num_clusters
#         next_cluster_oh = jnp.array(jax.nn.one_hot(next_cluster_id, num_clusters))
        
#         key = f"{cluster_id}-{next_cluster_id}"
#         bearing_value = self.cluster_bearing_map.get(key, 0.0)

#         goal_state_np = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

#         env_state = LidarEnvState(
#             agent=jnp.array([agent_state_np]), 
#             goal=jnp.array([goal_state_np]),
#             obstacle=jnp.zeros((0, 2)),
#             bearing=jnp.array([bearing_value]),
#             current_cluster_oh=jnp.array([current_cluster_oh]),
#             next_cluster_oh=jnp.array([next_cluster_oh]),
#             bridge_center=jnp.zeros((0,2)),
#             bridge_length=jnp.zeros((0,)),
#             bridge_gap_width=jnp.zeros((0,)),
#             bridge_wall_thickness=jnp.zeros((0,)),
#             bridge_theta=jnp.zeros((0,))
#         )

#         graph = self.env_instance.get_graph(env_state, jnp.array(final_lidar_data))
#         return graph

#     def _action_to_twist(self, action: Action) -> Twist:
#         twist = Twist()
#         twist.linear.x = float(action[0]) * 2.0
#         twist.angular.z = float(action[1]) * 4.0
#         return twist

# def main(args=None):
#     rclpy.init(args=args)
#     node = DGPPOROSNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

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

# ROS2 Messages
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16, Float32MultiArray
from nav_msgs.msg import Odometry

# DGPPO and LidarEnv components
from .dgppo.dgppo.env.lidar_env.lidar_target import LidarTarget, LidarEnvState
from .dgppo.dgppo.algo.dgppo import DGPPO
from .dgppo.dgppo.algo import make_algo
from .dgppo.dgppo.utils.graph import GraphsTuple
from .dgppo.dgppo.utils.typing import Array, Action
from .dgppo.dgppo.utils.utils import jax_vmap, tree_index

class DGPPOROSNode(Node):
    def __init__(self):
        super().__init__('dgppo_ros_node')

        self.get_logger().info("Initializing DGPPO ROS Node...")

        # --- PARAMETER SETUP FOR DYNAMIC TESTING ---
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('current_cluster_id', 1)
        # Assuming we have 4 clusters in total for one-hot encoding
        self.num_clusters = 4
        # --- END PARAMETER SETUP ---

        model_dir = "dgppo/logs/LidarTarget/dgppo/bearing_new"
        config_path = os.path.join(model_dir, "config.yaml")
        params_path = os.path.join(model_dir, "models")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        step = self._get_model_step(model_dir)

        env_kwargs = config.get("env_kwargs", {})
        
        self.get_logger().info(f"Loaded config: {config}")

        num_range_bins = 32
        if 'params' not in env_kwargs:
            env_kwargs['params'] = {}
        env_kwargs['params']['num_ranges'] = num_range_bins
        env_kwargs['params']['top_k_rays'] = 8
        env_kwargs['params']['comm_radius'] = 0.5
        
        self.env_instance = LidarTarget(
            num_agents=config.get('num_agents'),
            params=env_kwargs.get('params', {}),
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
        
        self.plan_sequence, self.bearing_map = self._load_plan_and_bearings(model_dir)
        self.current_plan_step_index = 0
        
        self.algo.load(params_path, step=step)

        self.rng_key = jr.PRNGKey(config.get('seed', 0))
        self.rnn_state = self.algo.init_rnn_state

        self.latest_ranges_msg = None
        self.latest_agent_state_msg = None
        self.latest_predicted_cluster_id = None

        # Publishers and subscribers
        self.publisher_ = self.create_publisher(Twist, '/carla/ego_vehicle/twist', 10)
        self.agent_state_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.agent_state_callback,
            10
        )
        
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

        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("DGPPO ROS Node fully initialized and ready.")
        self.get_logger().info("Default mode: Listening for predicted cluster ID on /predicted_cluster_id.")
        self.get_logger().info("To activate debug mode: 'ros2 param set /dgppo_ros_node debug_mode true'")
        self.get_logger().info("When in debug mode: 'ros2 param set /dgppo_ros_node current_cluster_id <new_id>'")


    def _get_model_step(self, model_dir):
        model_path = os.path.join(model_dir, "models")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        self.get_logger().info(f"Loading latest model from step: {step}")
        return step

    def _load_plan_and_bearings(self, model_dir):
        plan_file_path = os.path.join(model_dir, "highlevel_plan.json")
        if not os.path.exists(plan_file_path):
            self.get_logger().error(f"High level plan file not found at {plan_file_path}")
            return [], {}
        with open(plan_file_path, "r") as f:
            data = json.load(f)
            return data.get("plan_sequence", []), data.get("bearing_map", {})

    def _map_cluster_id(self, cluster_id: int) -> int:
        self.get_logger().info(f"cluster id: {cluster_id}")
        if cluster_id == 3:
            return 1
        elif cluster_id == 4:
            return 2
        elif cluster_id in [0, 5, 6]:
            return 3
        elif cluster_id in [1, 2]:
            return 0
        else:
            return cluster_id

    def ranges_callback(self, msg: Float32MultiArray):
        self.latest_ranges_msg = msg

    def agent_state_callback(self, msg: Odometry):
        self.latest_agent_state_msg = msg

    def predicted_cluster_callback(self, msg: Int16):
        self.latest_predicted_cluster_id = msg.data

    def control_loop(self):
        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        
        if self.current_plan_step_index >= len(self.plan_sequence):
            self.get_logger().info("High-level plan is complete. Stopping control loop.")
            self.publisher_.publish(Twist()) # Publish a zero twist to stop the vehicle
            self.timer.cancel()
            return

        current_plan_step = self.plan_sequence[self.current_plan_step_index]
        expected_start_cluster = current_plan_step["current"]
        expected_next_cluster = current_plan_step["next"]

        if self.latest_ranges_msg is None or self.latest_agent_state_msg is None:
            self.get_logger().warning("Waiting for sensor data...")
            return

        if debug_mode:
            current_cluster_id = self.get_parameter('current_cluster_id').get_parameter_value().integer_value
            self.get_logger().info(f"DEBUG MODE: Using manual cluster ID {current_cluster_id}")
        else:
            if self.latest_predicted_cluster_id is None:
                self.get_logger().warning("Waiting for predicted cluster ID...")
                return
            current_cluster_id = self.latest_predicted_cluster_id
            self.get_logger().info(f"Default MODE: Using predicted cluster ID {current_cluster_id}")

        mapped_current_cluster = self._map_cluster_id(current_cluster_id)
        self.get_logger().info(f"current before:{current_cluster_id}, mapped before check: {mapped_current_cluster}")
        if mapped_current_cluster == expected_next_cluster:
            self.current_plan_step_index += 1
            if self.current_plan_step_index >= len(self.plan_sequence):
                self.get_logger().info(f"Plan step complete. Transitioning to cluster {expected_next_cluster}. Plan is now finished.")
            else:
                self.get_logger().info(f"Plan step complete. Transitioning from cluster {expected_start_cluster} to {expected_next_cluster}. Next step is from cluster {self.plan_sequence[self.current_plan_step_index]['current']}.")

        if self.current_plan_step_index >= len(self.plan_sequence):
            return

        current_plan_step = self.plan_sequence[self.current_plan_step_index]
        current_cluster_id_for_policy = mapped_current_cluster #current_plan_step["current"]
        mapped_start_cluster_id_for_policy = current_plan_step["current"]
        mapped_next_cluster_id_for_policy = current_plan_step["next"]

        graph = self._build_state_and_graph(
            self.latest_agent_state_msg,
            self.latest_ranges_msg,
            current_cluster_id_for_policy,
            mapped_start_cluster_id_for_policy,
            mapped_next_cluster_id_for_policy
        )

        self.rng_key, action_key = jr.split(self.rng_key)
        
        action, new_rnn_state = self.algo.act(
            graph=graph,
            rnn_state=self.rnn_state,
            params={'policy': self.algo.policy_train_state.params}
        )

        self.rnn_state = new_rnn_state
        action_squeezed = jnp.squeeze(action, axis=0) 
        twist_msg = self._action_to_twist(action_squeezed)
        self.publisher_.publish(twist_msg)
        self.get_logger().info(f"Action: {action_squeezed}") #, Next: {mapped_next_cluster_id}, Bearing: {bearing_value}")


    def _build_state_and_graph(self, odom_msg: Odometry, ranges_msg: Float32MultiArray, mapped_current_cluster_id: int, mapped_start_cluster_id: int, mapped_next_cluster_id: int) -> GraphsTuple:
        pos = odom_msg.pose.pose.position
        vel = odom_msg.twist.twist.linear
        agent_state_np = np.array([pos.x, pos.y, vel.x, vel.y], dtype=np.float32)
        self.get_logger().info(f"Agent state: {agent_state_np}")

        ranges_np = np.array(ranges_msg.data, dtype=np.float32)
        self.get_logger().info(f"Max range dist: {np.max(ranges_np)}, Min dist: {np.min(ranges_np)}")
        
        num_ranges = self.env_instance.params['num_ranges']
        top_k = self.env_instance.params['top_k_rays']
        
        if ranges_np.shape[0] != num_ranges:
            self.get_logger().warn(f"Received {ranges_np.shape[0]} ranges, but expected {num_ranges}. This may cause errors.")
        
        angle_increment = (2 * np.pi) / num_ranges
        angles = np.arange(num_ranges) * angle_increment
        
        x_coords = ranges_np * np.cos(angles)
        y_coords = ranges_np * np.sin(angles)
        
        lidar_data_np = np.stack([x_coords, y_coords], axis=1)

        distances = np.linalg.norm(lidar_data_np, axis=1)
        sorted_indices = np.argsort(distances)
        final_lidar_data = lidar_data_np[sorted_indices[:top_k]]
        
        if final_lidar_data.shape[0] < top_k:
            padding_size = top_k - final_lidar_data.shape[0]
            padding = np.zeros((padding_size, 2), dtype=np.float32)
            final_lidar_data = np.concatenate([final_lidar_data, padding], axis=0)
        
        final_lidar_data = final_lidar_data[np.newaxis, :, :]
        
        # mapped_current_cluster_id = self._map_cluster_id(current_cluster_id)
        # self.get_logger().info(f"current before:{current_cluster_id}, mapped before check: {mapped_current_cluster}")

        # mapped_next_cluster_id = self._map_cluster_id(next_cluster_id)

        current_cluster_oh = jnp.array(jax.nn.one_hot(mapped_current_cluster_id, self.num_clusters))
        start_cluster_oh = jnp.array(jax.nn.one_hot(mapped_start_cluster_id, self.num_clusters))
        next_cluster_oh = jnp.array(jax.nn.one_hot(mapped_next_cluster_id, self.num_clusters))

        key = f"{mapped_start_cluster_id}-{mapped_next_cluster_id}"
        bearing_value = self.bearing_map.get(key, 0.0)
        self.get_logger().info(f"Start: {mapped_start_cluster_id}, Current: {mapped_current_cluster_id}, Next: {mapped_next_cluster_id}, Bearing: {bearing_value}")

        goal_state_np = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        env_state = LidarEnvState(
            agent=jnp.array([agent_state_np]), 
            goal=jnp.array([goal_state_np]),
            obstacle=jnp.zeros((0, 2)),
            bearing=jnp.array([bearing_value]),
            current_cluster_oh=jnp.array([current_cluster_oh]),
            start_cluster_oh = jnp.array([start_cluster_oh]),
            next_cluster_oh=jnp.array([next_cluster_oh]),
            bridge_center=jnp.zeros((0,2)),
            bridge_length=jnp.zeros((0,)),
            bridge_gap_width=jnp.zeros((0,)),
            bridge_wall_thickness=jnp.zeros((0,)),
            bridge_theta=jnp.zeros((0,))
        )

        graph = self.env_instance.get_graph(env_state, jnp.array(final_lidar_data))
        return graph

    def _action_to_twist(self, action: Action) -> Twist:
        """Converts the model's action output to a Twist message."""
        twist = Twist()
        twist.linear.x = float(action[0]) * 2.0
        twist.angular.z = float(action[1]) * 4.0
        return twist

def main(args=None):
    rclpy.init(args=args)
    node = DGPPOROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
