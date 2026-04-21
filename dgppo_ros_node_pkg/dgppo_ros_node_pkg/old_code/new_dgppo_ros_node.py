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
# import math
# from typing import NamedTuple, Tuple, Optional, List, Dict

# # ROS2 Messages
# from geometry_msgs.msg import Twist
# from std_msgs.msg import Int16, Float32MultiArray
# from nav_msgs.msg import Odometry

# import carla
# import time

# # DGPPO and LidarEnv components
# from .dgppo.dgppo.env.lidar_env.lidar_target import LidarTarget, LidarEnvState
# from .dgppo.dgppo.algo.dgppo import DGPPO
# from .dgppo.dgppo.algo import make_algo
# from .dgppo.dgppo.utils.graph import GraphsTuple
# from .dgppo.dgppo.utils.typing import Array, Action, AgentState, State
# from .dgppo.dgppo.utils.utils import jax_vmap, tree_index

# class DGPPOROSNode(Node):
#     def __init__(self):
#         super().__init__('dgppo_ros_node')

#         self.get_logger().info("Initializing DGPPO ROS Node...")

#         self.declare_parameter('debug_mode', False)
#         self.declare_parameter('current_cluster_id', 1)
#         self.num_clusters = 4
#         self.dt = 1.0/30
#         self.twod_area_size = 1.5

#         model_dir = "dgppo/logs/LidarTarget/dgppo/91bridge"
#         config_path = os.path.join(model_dir, "config.yaml")
#         params_path = os.path.join(model_dir, "models")
        
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)

#         step = self._get_model_step(model_dir)

#         env_kwargs = config.get("env_kwargs", {})
        
#         self.get_logger().info(f"Loaded config: {config}")

#         num_range_bins = 32
#         if 'params' not in env_kwargs:
#             env_kwargs['params'] = {}
#         env_kwargs['params']['num_ranges'] = num_range_bins
#         env_kwargs['params']['top_k_rays'] = 8
#         env_kwargs['params']['comm_radius'] = 0.5
        
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
        
#         self.plan_sequence, self.bearing_map, self.cluster_centroids = self._load_plan_and_cluster_data(model_dir)
#         self.current_plan_step_index = 0
        
#         self.algo.load(params_path, step=step)

#         self.rng_key = jr.PRNGKey(config.get('seed', 0))
#         self.rnn_state = self.algo.init_rnn_state

#         # State variable for the main fix
#         self.current_agent_state = None

#         self.latest_ranges_msg = None
#         self.latest_agent_state_msg = None
#         self.latest_predicted_cluster_id = None
#         self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        
#         self.is_vehicle_ready = False
#         self.is_first_run = True

#         self.carla_client = carla.Client('localhost', 2000)
#         self.carla_client.set_timeout(10.0)
#         self.world = self.carla_client.get_world()
#         self.ego_vehicle = None
        
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

#         self.predicted_cluster_sub = self.create_subscription(
#             Int16,
#             '/predicted_cluster',
#             self.predicted_cluster_callback,
#             10
#         )
        
#         self.scale_2d_3d = 47.0
#         self.origin_x = 55.0
#         self.origin_y = -210.0

#         self.timer = self.create_timer(0.1, self.control_loop)
        
#         self.get_logger().info("DGPPO ROS Node fully initialized and ready.")
#         self.get_logger().info("Default mode: Listening for predicted cluster ID on /predicted_cluster_id.")
#         self.get_logger().info("To activate debug mode: 'ros2 param set /dgppo_ros_node debug_mode true'")
#         self.get_logger().info("When in debug mode: 'ros2 param set /dgppo_ros_node current_cluster_id <new_id>'")

#     def _get_model_step(self, model_dir):
#         model_path = os.path.join(model_dir, "models")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model directory not found at {model_path}")
#         models = os.listdir(model_path)
#         step = max([int(model) for model in models if model.isdigit()])
#         self.get_logger().info(f"Loading latest model from step: {step}")
#         return step

#     def _load_plan_and_cluster_data(self, model_dir):
#         plan_file_path = "plans/highlevel_plan.json"
#         if not os.path.exists(plan_file_path):
#             self.get_logger().error(f"High level plan file not found at {plan_file_path}")
#             return [], {}, {}
#         with open(plan_file_path, "r") as f:
#             data = json.load(f)
#             # Assuming the JSON file now contains a 'centroids' key
#             return data.get("plan_sequence", []), data.get("bearing_map", {}), data.get("centroids", {})

#     # def _map_cluster_id(self, cluster_id: int) -> int:
#     #     self.get_logger().info(f"cluster id: {cluster_id}")
#     #     if cluster_id in [0, 2]:
#     #         return 1
#     #     elif cluster_id in [3, 4, 5]:
#     #         return 2
#     #     elif cluster_id in [0, 6]:
#     #         return 3
#     #     elif cluster_id in [1, 2]:
#     #         return 0
#     #     else:
#     #         return cluster_id
    
#     def _map_cluster_id(self, cluster_id: int) -> int:
#         self.get_logger().info(f"cluster id: {cluster_id}")
#         if cluster_id in [4, 5, 6]:
#             return 1
#         elif cluster_id in [0, 1, 2]:
#             return 2
#         # elif cluster_id in [0, 6]:
#         #     return 3
#         # elif cluster_id in [1, 2]:
#         #     return 0
#         else:
#             return cluster_id

#     def ranges_callback(self, msg: Float32MultiArray):
#         self.latest_ranges_msg = msg

#     def agent_state_callback(self, msg: Odometry):
#         self.latest_agent_state_msg = msg

#     def predicted_cluster_callback(self, msg: Int16):
#         self.latest_predicted_cluster_id = msg.data

#     def control_loop(self):
#         if not self.is_vehicle_ready:
#             self.get_logger().info("Waiting for ego vehicle to be spawned...")
#             actors = self.world.get_actors()
#             vehicles = actors.filter("*vehicle*")
#             if vehicles:
#                 self.ego_vehicle = vehicles[0]
#                 self.is_vehicle_ready = True
#                 self.get_logger().info("Ego vehicle found. Starting control loop.")
#             else:
#                 return # Keep waiting

#         # New logic to handle the initial state based on cluster centroids
#         if self.is_first_run:
#             if self.current_plan_step_index < len(self.plan_sequence):
#                 start_cluster_id = str(self.plan_sequence[self.current_plan_step_index]["start"])
#                 if start_cluster_id in self.cluster_centroids:
#                     # Use the cluster centroid as the initial position
#                     centroid = self.cluster_centroids[start_cluster_id]
#                     self.get_logger().info(f"Setting initial agent state to centroid of cluster {start_cluster_id}: {centroid}")
#                     scaled_pos_x_model = (centroid[1] - self.origin_y) / self.scale_2d_3d
#                     scaled_pos_y_model = (centroid[0] - self.origin_x) / self.scale_2d_3d
#                     scaled_agent_state_np = np.array([scaled_pos_x_model, scaled_pos_y_model, 0.0, 0.0], dtype=np.float32)
#                     self.current_agent_state = jnp.expand_dims(jnp.array(scaled_agent_state_np), axis=0)
#                 else:
#                     self.get_logger().error(f"Centroid for cluster {start_cluster_id} not found in plan data!")
#                     return
#             self.is_first_run = False
            
#         debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        
#         if self.current_plan_step_index >= len(self.plan_sequence):
#             self.get_logger().info("High-level plan is complete. Stopping control loop.")
#             current_transform = self.ego_vehicle.get_transform()
#             self.ego_vehicle.set_transform(current_transform)
#             self.ego_vehicle.set_target_velocity(carla.Vector3D(0,0,0))
#             self.ego_vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))
#             self.timer.cancel()
#             return

#         current_plan_step = self.plan_sequence[self.current_plan_step_index]
#         expected_start_cluster = current_plan_step["start"]
#         expected_next_cluster = current_plan_step["next"]

#         if self.latest_ranges_msg is None or self.latest_agent_state_msg is None:
#             self.get_logger().warning("Waiting for sensor data...")
#             return

#         if debug_mode:
#             current_cluster_id = self.get_parameter('current_cluster_id').get_parameter_value().integer_value
#             self.get_logger().info(f"DEBUG MODE: Using manual cluster ID {current_cluster_id}")
#         else:
#             if self.latest_predicted_cluster_id is None:
#                 self.get_logger().warning("Waiting for predicted cluster ID...")
#                 return
#             current_cluster_id = self.latest_predicted_cluster_id
#             self.get_logger().info(f"Default MODE: Using predicted cluster ID {current_cluster_id}")

#         mapped_current_cluster = self._map_cluster_id(current_cluster_id)
#         self.get_logger().info(f"current before:{current_cluster_id}, mapped before check: {mapped_current_cluster}")
#         if mapped_current_cluster == expected_next_cluster:
#             self.current_plan_step_index += 1
#             if self.current_plan_step_index >= len(self.plan_sequence):
#                 self.get_logger().info(f"Plan step complete. Transitioning to cluster {expected_next_cluster}. Plan is now finished.")
#             else:
#                 self.get_logger().info(f"Plan step complete. Transitioning from cluster {expected_start_cluster} to {expected_next_cluster}. Next step is from cluster {self.plan_sequence[self.current_plan_step_index]['start']}.")

#         if self.current_plan_step_index >= len(self.plan_sequence):
#             return

#         scaled_ranges_np = np.array(self.latest_ranges_msg.data, dtype=np.float32) / self.scale_2d_3d
        
#         graph = self._build_state_and_graph(
#             self.current_agent_state,
#             scaled_ranges_np,
#             mapped_current_cluster,
#             expected_start_cluster,
#             expected_next_cluster,
#             self.next_cluster_bonus_awarded
#         )

#         self.rng_key, action_key = jr.split(self.rng_key)
        
#         action, new_rnn_state = self.algo.act(
#             graph=graph,
#             rnn_state=self.rnn_state,
#             params={'policy': self.algo.policy_train_state.params}
#         )

#         self.rnn_state = new_rnn_state
#         action = self.clip_action(action)
        
#         # Calculate the next state in the small frame and store it for the next loop
#         self.current_agent_state = self.agent_step_euler(self.current_agent_state, action)
        
#         next_state_small_frame_squeezed = jnp.squeeze(self.current_agent_state, axis=0)
        
#         reward, bonus_awarded_updated = self.env_instance.get_reward(graph, action)
        
#         # FIX: Check for empty array and reset it
#         if bonus_awarded_updated.size == 0:
#             self.get_logger().warning("Received an empty bonus array from get_reward. Resetting.")
#             self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
#         else:
#             self.next_cluster_bonus_awarded = bonus_awarded_updated

#         # Scale the next position back up and add the origin
#         next_pos_x_carla = next_state_small_frame_squeezed[1] * self.scale_2d_3d + self.origin_x
#         next_pos_y_carla = next_state_small_frame_squeezed[0] * self.scale_2d_3d + self.origin_y
        
#         new_location = carla.Location(x=float(next_pos_x_carla), y=float(-1*next_pos_y_carla), z=self.ego_vehicle.get_transform().location.z)
#         new_transform = carla.Transform(new_location, self.ego_vehicle.get_transform().rotation)
        
#         self.teleport_and_wait(new_transform)

#         self.get_logger().info(f"Action: {action}")
#         self.get_logger().info(f"Teleporting to X: {next_pos_x_carla}, Y: {next_pos_y_carla}")
#         self.get_logger().info("2d ranges: {}".format(scaled_ranges_np))
#         self.get_logger().info("3d ranges: {}".format(self.latest_ranges_msg))
        
#     def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
#         """By default, use double integrator dynamics"""
#         assert action.shape == (self.env_instance.num_agents, self.env_instance.action_dim)
#         print(agent_states.shape)
#         print((self.env_instance.num_agents, self.env_instance.state_dim))
#         assert agent_states.shape == (self.env_instance.num_agents, self.env_instance.state_dim)
#         x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
#         n_state_agent_new = x_dot * self.dt + agent_states
#         assert n_state_agent_new.shape == (self.env_instance.num_agents, self.env_instance.state_dim)
#         return self.clip_state(n_state_agent_new)
    
#     def state_lim(self) -> Tuple[State, State]:
#         lower_lim = jnp.array([0., 0., -0.5, -0.5])
#         upper_lim = jnp.array([self.twod_area_size, self.twod_area_size, 0.5, 0.5])
#         return lower_lim, upper_lim

#     def action_lim(self) -> Tuple[Action, Action]:
#         lower_lim = jnp.ones(2) * -1.0
#         upper_lim = jnp.ones(2)
#         return lower_lim, upper_lim
    
#     def clip_state(self, state: State) -> State:
#         lower_limit, upper_limit = self.state_lim()
#         return jnp.clip(state, lower_limit, upper_limit)

#     def clip_action(self, action: Action) -> Action:
#         lower_limit, upper_limit = self.action_lim()
#         return jnp.clip(action, lower_limit, upper_limit)

#     def _build_state_and_graph(self, agent_state_np: np.ndarray, scaled_ranges: np.ndarray, mapped_current_cluster_id: int, mapped_start_cluster_id: int, mapped_next_cluster_id: int, bonus_awarded_updated: jnp.ndarray)  -> GraphsTuple:
#         self.get_logger().info(f"Agent state (scaled): {agent_state_np}")
        
#         num_ranges = self.env_instance.params['num_ranges']
#         top_k = self.env_instance.params['top_k_rays']
        
#         if scaled_ranges.shape[0] != num_ranges:
#             self.get_logger().warn(f"Received {scaled_ranges.shape[0]} ranges, but expected {num_ranges}. This may cause errors.")
        
#         angle_increment = (2 * np.pi) / num_ranges
#         angles = np.arange(num_ranges) * angle_increment
        
#         # Accounting for CARLA-2D flip
#         x_coords = scaled_ranges * np.sin(angles)
#         y_coords = scaled_ranges * np.cos(angles)
        
#         lidar_data_np = np.stack([x_coords, y_coords], axis=1)

#         distances = np.linalg.norm(lidar_data_np, axis=1)
#         sorted_indices = np.argsort(distances)
#         final_lidar_data = lidar_data_np[sorted_indices[:top_k]]
        
#         if final_lidar_data.shape[0] < top_k:
#             padding_size = top_k - final_lidar_data.shape[0]
#             padding = np.zeros((padding_size, 2), dtype=np.float32)
#             final_lidar_data = np.concatenate([final_lidar_data, padding], axis=0)
        
#         final_lidar_data = final_lidar_data[np.newaxis, :, :]

#         current_cluster_oh = jnp.array(jax.nn.one_hot(mapped_current_cluster_id, self.num_clusters))
#         start_cluster_oh = jnp.array(jax.nn.one_hot(mapped_start_cluster_id, self.num_clusters))
#         next_cluster_oh = jnp.array(jax.nn.one_hot(mapped_next_cluster_id, self.num_clusters))

#         key = f"{mapped_start_cluster_id}-{mapped_next_cluster_id}"
#         bearing_value = self.bearing_map.get(key, 0.0)
#         self.get_logger().info(f"Start: {mapped_start_cluster_id}, Current: {mapped_current_cluster_id}, Next: {mapped_next_cluster_id}, Bearing: {bearing_value}")
        
#         goal_state_np = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
#         inter_center_env_state: Float[Array, "2"] = jnp.array([0.0, 0.0])
#         passage_width_env_state: Float[Array, ""] = jnp.array(0.0)
#         obs_len_env_state: Float[Array, ""] = jnp.array(0.0)
#         global_angle_env_state: Float[Array, ""] = jnp.array(0.0)
#         is_four_way_env_state: bool = False

#         env_state = LidarEnvState(
#             agent=agent_state_np, 
#             goal=jnp.array([goal_state_np]),
#             obstacle=jnp.zeros((0, 2)),
#             bearing=jnp.array([bearing_value]),
#             current_cluster_oh=jnp.array([current_cluster_oh]),
#             start_cluster_oh = jnp.array([start_cluster_oh]),
#             next_cluster_oh=jnp.array([next_cluster_oh]),
#             next_cluster_bonus_awarded=bonus_awarded_updated,
#             # bridge_center=jnp.zeros((0,2)),
#             # bridge_length=jnp.zeros((0,)),
#             # bridge_gap_width=jnp.zeros((0,)),
#             # bridge_wall_thickness=jnp.zeros((0,)),
#             # bridge_theta=jnp.zeros((0,))
#             is_four_way=is_four_way_env_state,
#             center=inter_center_env_state,
#             passage_width=passage_width_env_state,
#             obs_len=obs_len_env_state,
#             global_angle=global_angle_env_state
#         )

#         graph = self.env_instance.get_graph(env_state, jnp.array(final_lidar_data))
#         return graph
    
#     def teleport_and_wait(self, transform: carla.Transform):
#         self.ego_vehicle.set_simulate_physics(False)
#         self.ego_vehicle.set_transform(transform)
#         self.ego_vehicle.set_target_velocity(carla.Vector3D(0,0,0))
#         self.ego_vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))
#         time.sleep(self.dt)

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
import json
import math
import random
from typing import NamedTuple, Tuple, Optional, List, Dict
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

# ROS2 Messages
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int16, Float32MultiArray
from nav_msgs.msg import Odometry

# DGPPO and LidarEnv components
from .dgppo.dgppo.env.lidar_env.lidar_target import LidarTarget, LidarEnvState
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
        self.num_clusters = 4
        self.dt = 1.0/30
        self.twod_area_size = 1.5

        model_dir = "dgppo/logs/LidarTarget/dgppo/91bridge"
        config_path = os.path.join(model_dir, "config.yaml")
        params_path = os.path.join(model_dir, "models")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        step = self._get_model_step(model_dir)

        env_kwargs = config.get("env_kwargs", {})
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
        
        # --- Transformation Parameters ---
        self.transform_type = 'angular' 
        self.transform_magnitude = 15.0 
        self.roll_angle_degrees = 15.0 

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
        self.latest_agent_state_msg = None
        self.latest_predicted_cluster_id = None
        self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        
        # --- Subscriptions ---
        self.agent_state_sub = self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.agent_state_callback, 10)
        self.ranges_sub = self.create_subscription(Float32MultiArray, '/processed_ranges', self.ranges_callback, qos_profile=qos_profile_sensor_data)
        self.predicted_cluster_sub = self.create_subscription(Int16, '/predicted_cluster', self.predicted_cluster_callback, 10)
        
        # --- Publisher: CHANGED from Twist to PoseStamped ---
        self.waypoint_publisher = self.create_publisher(PoseStamped, '/dgppo_waypoint', 10)

        # --- State variables for the new goal and map scaling ---
        self.scale_2d_3d = 47.0
        self.origin_x = 55.0
        self.origin_y = -210.0
        
        # New: Use a timer for the planning loop
        self.timer = self.create_timer(0.1, self.planning_loop)
        
        self.get_logger().info("DGPPO ROS Node fully initialized and ready.")

    def _get_model_step(self, model_dir):
        model_path = os.path.join(model_dir, "models")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        self.get_logger().info(f"Loading latest model from step: {step}")
        return step

    def _load_plan_and_cluster_data(self, model_dir):
        plan_file_path = "plans/highlevel_plan.json"
        if not os.path.exists(plan_file_path):
            self.get_logger().error(f"High level plan file not found at {plan_file_path}")
            return [], {}, {}
        with open(plan_file_path, "r") as f:
            data = json.load(f)
            return data.get("plan_sequence", []), data.get("bearing_map", {}), data.get("centroids", {})

    def _map_cluster_id(self, cluster_id: int) -> int:
        if cluster_id in [4, 5, 6]:
            return 1
        elif cluster_id in [0, 1, 2]:
            return 2
        else:
            return cluster_id

    def _get_transformed_goal_coords(self, goal_centroid_raw, vehicle_x, vehicle_y, vehicle_yaw_rad):
        transformed_goal_x, transformed_goal_y = goal_centroid_raw[0], goal_centroid_raw[1]
        
        if self.transform_type == 'roll':
            roll_angle_rad = math.radians(self.roll_angle_degrees)
            cos_theta = math.cos(roll_angle_rad)
            sin_theta = math.sin(roll_angle_rad)
            
            goal_vector_x = transformed_goal_x - vehicle_x
            goal_vector_y = transformed_goal_y - vehicle_y
            
            rotated_x = goal_vector_x * cos_theta - goal_vector_y * sin_theta
            rotated_y = goal_vector_x * sin_theta + goal_vector_y * cos_theta
            
            transformed_goal_x = vehicle_x + rotated_x
            transformed_goal_y = vehicle_y + rotated_y

        elif self.transform_type == 'homography':
            magnitude = self.transform_magnitude
            
            matrix = np.array([
                [1.0, 0.0, 0.0],
                [-0.005 * magnitude, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            
            point = np.array([transformed_goal_x, transformed_goal_y, 1.0])
            transformed_point = np.dot(matrix, point)
            
            transformed_goal_x = transformed_point[0] / transformed_point[2]
            transformed_goal_y = transformed_point[1] / transformed_point[2]
            
        elif self.transform_type == 'scaling':
            scaling_factor = random.choice([0.5, 2.0])
            vector_to_destination = np.array([transformed_goal_x - vehicle_x, transformed_goal_y - vehicle_y])
            scaled_vector = scaling_factor * vector_to_destination
            transformed_goal_x = vehicle_x + scaled_vector[0]
            transformed_goal_y = vehicle_y + scaled_vector[1]

        elif self.transform_type == 'angular':
            goal_x_offset = self.transform_magnitude * math.cos(vehicle_yaw_rad)
            goal_y_offset = self.transform_magnitude * math.sin(vehicle_yaw_rad)
            transformed_goal_x += goal_x_offset
            transformed_goal_y += goal_y_offset

        elif self.transform_type == 'positional':
            translation_x = self.transform_magnitude * random.choice([-1, 1])
            translation_y = self.transform_magnitude * random.choice([-1, 1])
            transformed_goal_x += translation_x
            transformed_goal_y += translation_y
        
        else:
            self.get_logger().warn("Invalid transform_type. No transformation applied.")

        return transformed_goal_x, transformed_goal_y

    def ranges_callback(self, msg: Float32MultiArray):
        self.latest_ranges_msg = msg

    def agent_state_callback(self, msg: Odometry):
        self.latest_agent_state_msg = msg

    def predicted_cluster_callback(self, msg: Int16):
        self.latest_predicted_cluster_id = msg.data

    def planning_loop(self):
        """
        The main planning loop. It retrieves sensor data, runs the DGPPO model,
        and publishes the resulting waypoint.
        """
        if self.latest_ranges_msg is None or self.latest_agent_state_msg is None or self.latest_predicted_cluster_id is None:
            self.get_logger().warning("Waiting for all required data before planning...")
            return

        if self.current_plan_step_index >= len(self.plan_sequence):
            self.get_logger().info("High-level plan is complete. Stopping.")
            self.timer.cancel()
            return
        
        # Determine the current plan step and clusters
        current_plan_step = self.plan_sequence[self.current_plan_step_index]
        expected_start_cluster = current_plan_step["start"]
        expected_next_cluster = current_plan_step["next"]

        # Check for transition based on the predicted cluster
        mapped_current_cluster = self._map_cluster_id(self.latest_predicted_cluster_id)
        if mapped_current_cluster == expected_next_cluster:
            self.current_plan_step_index += 1
            if self.current_plan_step_index < len(self.plan_sequence):
                self.get_logger().info(f"Plan step complete. Transitioning to cluster {expected_next_cluster}. New index: {self.current_plan_step_index}")
            else:
                self.get_logger().info(f"Plan is now finished.")
            return

        if self.current_plan_step_index >= len(self.plan_sequence):
            return
            
        # Get next goal coordinates from the centroids data
        next_goal_cluster_id = str(self.plan_sequence[self.current_plan_step_index]["next"])
        if next_goal_cluster_id not in self.cluster_centroids:
            self.get_logger().error(f"Goal cluster '{next_goal_cluster_id}' not found in centroids data. Stopping.")
            self.timer.cancel()
            return
            
        next_goal_centroid_raw = self.cluster_centroids[next_goal_cluster_id]
        
        # Get vehicle's current position and yaw from the odometry message
        current_pose_carla = self.latest_agent_state_msg.pose.pose.position
        orientation_q = self.latest_agent_state_msg.pose.pose.orientation
        _, _, vehicle_yaw_rad = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        # Apply the transformation to the goal centroid
        next_goal_x_transformed, next_goal_y_transformed = self._get_transformed_goal_coords(
            next_goal_centroid_raw,
            current_pose_carla.x,
            current_pose_carla.y,
            vehicle_yaw_rad
        )
        
        # Calculate bearing to the transformed goal
        bearing_value = math.atan2(next_goal_y_transformed - current_pose_carla.y, next_goal_x_transformed - current_pose_carla.x)
        
        # Build the graph and get action from the DGPPO model
        scaled_ranges_np = np.array(self.latest_ranges_msg.data, dtype=np.float32) / self.scale_2d_3d
        graph = self._build_state_and_graph(
            jnp.array([0., 0., 0., 0.]), # Placeholder for agent state as it's not used in this model
            scaled_ranges_np,
            mapped_current_cluster,
            expected_start_cluster,
            expected_next_cluster,
            self.next_cluster_bonus_awarded,
            bearing_value
        )
        
        self.rng_key, action_key = jr.split(self.rng_key)
        action, new_rnn_state = self.algo.act(
            graph=graph,
            rnn_state=self.rnn_state,
            params={'policy': self.algo.policy_train_state.params}
        )

        # The model's action output is now interpreted as a relative position change
        action_np = np.array(action).squeeze()
        
        # Convert the model's output (relative action) to a new absolute waypoint
        new_waypoint_x = current_pose_carla.x + action_np[0] * 5.0 # Use a small scaling factor
        new_waypoint_y = current_pose_carla.y + action_np[1] * 5.0

        # Create and publish the new waypoint message
        waypoint_msg = PoseStamped()
        waypoint_msg.header.stamp = self.get_clock().now().to_msg()
        waypoint_msg.header.frame_id = 'map'
        waypoint_msg.pose.position.x = float(new_waypoint_x)
        waypoint_msg.pose.position.y = float(new_waypoint_y)
        waypoint_msg.pose.position.z = current_pose_carla.z
        self.waypoint_publisher.publish(waypoint_msg)
        
        self.get_logger().info(f"Published new waypoint: ({new_waypoint_x:.2f}, {new_waypoint_y:.2f})")
        
        # The following sections related to state updates and reward are for the RL training loop.
        # They may not be directly needed for ROS operation but are left for completeness.
        self.rnn_state = new_rnn_state
        reward, bonus_awarded_updated = self.env_instance.get_reward(graph, action)
        if bonus_awarded_updated.size == 0:
            self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        else:
            self.next_cluster_bonus_awarded = bonus_awarded_updated
            
    # The following methods are for the model's internal state management.
    # They are not directly used to control the physical vehicle.
    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.env_instance.num_agents, self.env_instance.action_dim)
        assert agent_states.shape == (self.env_instance.num_agents, self.env_instance.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.env_instance.num_agents, self.env_instance.state_dim)
        return self.clip_state(n_state_agent_new)
    
    def state_lim(self) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.twod_area_size, self.twod_area_size, 0.5, 0.5])
        return lower_lim, upper_lim

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

    def _build_state_and_graph(self, agent_state_np: np.ndarray, scaled_ranges: np.ndarray, mapped_current_cluster_id: int, mapped_start_cluster_id: int, mapped_next_cluster_id: int, bonus_awarded_updated: jnp.ndarray, bearing_value: float) -> GraphsTuple:
        self.get_logger().info(f"Agent state (scaled): {agent_state_np}")
        
        num_ranges = self.env_instance.params['num_ranges']
        top_k = self.env_instance.params['top_k_rays']
        
        if scaled_ranges.shape[0] != num_ranges:
            self.get_logger().warn(f"Received {scaled_ranges.shape[0]} ranges, but expected {num_ranges}. This may cause errors.")
        
        angle_increment = (2 * np.pi) / num_ranges
        angles = np.arange(num_ranges) * angle_increment
        
        x_coords = scaled_ranges * np.sin(angles)
        y_coords = scaled_ranges * np.cos(angles)
        
        lidar_data_np = np.stack([x_coords, y_coords], axis=1)
        distances = np.linalg.norm(lidar_data_np, axis=1)
        sorted_indices = np.argsort(distances)
        final_lidar_data = lidar_data_np[sorted_indices[:top_k]]
        
        if final_lidar_data.shape[0] < top_k:
            padding_size = top_k - final_lidar_data.shape[0]
            padding = np.zeros((padding_size, 2), dtype=np.float32)
            final_lidar_data = np.concatenate([final_lidar_data, padding], axis=0)
        
        final_lidar_data = final_lidar_data[np.newaxis, :, :]

        current_cluster_oh = jnp.array(jax.nn.one_hot(mapped_current_cluster_id, self.num_clusters))
        start_cluster_oh = jnp.array(jax.nn.one_hot(mapped_start_cluster_id, self.num_clusters))
        next_cluster_oh = jnp.array(jax.nn.one_hot(mapped_next_cluster_id, self.num_clusters))

        self.get_logger().info(f"Start: {mapped_start_cluster_id}, Current: {mapped_current_cluster_id}, Next: {mapped_next_cluster_id}, Bearing: {bearing_value}")
        
        goal_state_np = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        inter_center_env_state: Float[Array, "2"] = jnp.array([0.0, 0.0])
        passage_width_env_state: Float[Array, ""] = jnp.array(0.0)
        obs_len_env_state: Float[Array, ""] = jnp.array(0.0)
        global_angle_env_state: Float[Array, ""] = jnp.array(0.0)
        is_four_way_env_state: bool = False

        env_state = LidarEnvState(
            agent=agent_state_np, 
            goal=jnp.array([goal_state_np]),
            obstacle=jnp.zeros((0, 2)),
            bearing=jnp.array([bearing_value]),
            current_cluster_oh=jnp.array([current_cluster_oh]),
            start_cluster_oh = jnp.array([start_cluster_oh]),
            next_cluster_oh=jnp.array([next_cluster_oh]),
            next_cluster_bonus_awarded=bonus_awarded_updated,
            is_four_way=is_four_way_env_state,
            center=inter_center_env_state,
            passage_width=passage_width_env_state,
            obs_len=obs_len_env_state,
            global_angle=global_angle_env_state
        )

        graph = self.env_instance.get_graph(env_state, jnp.array(final_lidar_data))
        return graph

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