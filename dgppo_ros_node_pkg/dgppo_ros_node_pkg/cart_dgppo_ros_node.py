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
from typing import NamedTuple, Tuple, Optional, List, Dict
from tf_transformations import euler_from_quaternion

# ROS2 Messages
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16, Float32MultiArray
from nav_msgs.msg import Odometry

import time

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
        self.declare_parameter('angular_offset_deg', 0.0)
        self.declare_parameter('ready_for_next_step', False)
        
        self.num_clusters = 4
        self.dt = 1.0/30
        self.twod_area_size = 1.5

        model_dir = "dgppo/logs/LidarTarget/dgppo/91inter"
        config_path = os.path.join(model_dir, "config.yaml")
        params_path = os.path.join(model_dir, "models")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        step = self._get_model_step(model_dir)

        env_kwargs = config.get("env_kwargs", {})
        
        self.get_logger().info(f"Loaded config: {config}")

        num_range_bins = 72
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
        
        self.plan_sequence, self.bearing_map, self.cluster_centroids = self._load_plan_and_cluster_data(model_dir)
        self.current_plan_step_index = 0
        
        self.algo.load(params_path, step=step)

        self.rng_key = jr.PRNGKey(config.get('seed', 0))
        self.rnn_state = self.algo.init_rnn_state

        # State variable for the main fix
        self.current_agent_state = None

        self.latest_ranges_msg = None
        self.latest_predicted_cluster_id = None
        self.latest_agent_state_msg = None
        self.latest_odom_msg = None
        self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        
        self.is_vehicle_ready = False
        self.is_first_run = True
        
        self.ranges_sub = self.create_subscription(
            Float32MultiArray,
            '/processed_ranges',
            self.ranges_callback,
            qos_profile=qos_profile_sensor_data
        )

        self.predicted_cluster_sub = self.create_subscription(
            Int16,
            '/predicted_cluster_local',
            self.predicted_cluster_callback,
            10
        )
        
        self.agent_state_sub = self.create_subscription(
            Odometry,
            '/dlio/odom_node/odom',
            self.agent_state_callback,
            10
        )
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/dlio/odom_node/odom',
            self.odom_callback,
            10 # QoS history depth
        )
        
        self.publisher_ = self.create_publisher(
            Int16,
            '/mapped_cluster',
            10
        )
        
        self.next_cluster_pub = self.create_publisher(
            Int16,
            '/next_cluster',
            10
        )
        
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/dgppo/policy_action',
            10
        )

        self.full_state_unscaled_pub = self.create_publisher(
            Float32MultiArray,
            '/dgppo/full_agent_state_unscaled',
            10
        )

        self.full_state_scaled_pub = self.create_publisher(
            Float32MultiArray,
            '/dgppo/full_agent_state_scaled',
            10
        )
        
        # self.scale_2d_3d = 20.0/1.5
        # self.scale_lidar = 8/0.5
        # self.origin_x = -2.5
        # self.origin_y = -2.5
        
        self.scale_2d_3d = 20 #20.0/1.5
        self.scale_lidar =15/0.5
        self.origin_x = 0
        self.origin_y = 0
        
        self.timer = self.create_timer(0.2, self.control_loop)
        
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

    def _load_plan_and_cluster_data(self, model_dir):
        plan_file_path = "plans/highlevel_plan.json"
        if not os.path.exists(plan_file_path):
            self.get_logger().error(f"High level plan file not found at {plan_file_path}")
            return [], {}, {}
        with open(plan_file_path, "r") as f:
            data = json.load(f)
            # Assuming the JSON file now contains a 'centroids' key
            return data.get("plan_sequence", []), data.get("bearing_map", {}), data.get("centroids", {})
        
    # def _map_cluster_id(self, cluster_id: int) -> int:  ### USE GROUND LIDAR (intersection)
    #     #self.get_logger().info(f"cluster id: {cluster_id}")
    #     if cluster_id in [1,2,12]:
    #         return 1
    #     elif cluster_id in [5,8]:
    #         return 4
    #     # elif cluster_id in [5]:
    #     #     return 3
    #     elif cluster_id in [0,3,4,6,7,9,10,11]:
    #         return 0
    #     else:
    #         return cluster_id
    
    # def _map_cluster_id(self, cluster_id: int) -> int:  ### USE GROUND LIDAR (intersection)
    #     #self.get_logger().info(f"cluster id: {cluster_id}")
    #     if cluster_id in [11,2,10,12]:
    #         return 1
    #     elif cluster_id in [4,5,8]:
    #         return 4
    #     # elif cluster_id in [5]:
    #     #     return 3
    #     elif cluster_id in [0,1,3,6,7,9]:
    #         return 0
    #     else:
    #         return cluster_id
    
    # def _map_cluster_id(self, cluster_id: int) -> int:  ### USE GROUND LIDAR (bridge)
    #     #self.get_logger().info(f"cluster id: {cluster_id}")
    #     if cluster_id in [11,2,10,12]:
    #         return 1
    #     elif cluster_id in [4,5,8]:
    #         return 3
    #     # elif cluster_id in [5]:
    #     #     return 3
    #     elif cluster_id in [0,1,3,6,7,9]:
    #         return 2
    #     else:
    #         return cluster_id
       
    ### latest cart test 
    # def _map_cluster_id(self, cluster_id: int) -> int:  ### livox1 (bridge)
    #     #self.get_logger().info(f"cluster id: {cluster_id}")
    #     if cluster_id in [11,2,10,12]:
    #         return 1
    #     elif cluster_id in [1,2]:
    #         return 3
    #     # elif cluster_id in [5]:
    #     #     return 3
    #     elif cluster_id in [4,6,7]:
    #         return 2
    #     else:
    #         return cluster_id
    
    def _map_cluster_id(self, cluster_id: int) -> int:  ### livox1 (bridge)
        if cluster_id in [0,4]: # in inter
            return 1
        elif cluster_id in [1]: # enter inter
            return 2
        elif cluster_id in [2,5,6]: # road
            return 0
        elif cluster_id in [3,7,8]: # exit inter
            return 3
        else:
            return cluster_id

    def ranges_callback(self, msg: Float32MultiArray):
        self.latest_ranges_msg = msg
        
    def predicted_cluster_callback(self, msg: Int16):
        self.latest_predicted_cluster_id = msg.data
        
    def odom_callback(self, msg: Odometry):
        self.latest_odom_msg = msg
        
    def agent_state_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        
        robot_state = [x, y, vx, vy]
        
        #self.get_logger().info(f'Received state: {robot_state}')
        robot_state_np = np.array(robot_state)
        self.latest_agent_state_msg = jnp.expand_dims(robot_state_np, axis=0)
        
    def control_loop(self):
        angular_offset = self.get_parameter('angular_offset_deg').get_parameter_value().double_value
        
        # New logic to handle the initial state based on cluster centroids
        if self.is_first_run:
            if self.current_plan_step_index < len(self.plan_sequence):
                start_cluster_id = str(self.plan_sequence[self.current_plan_step_index]["start"])
                
                if start_cluster_id in self.cluster_centroids:
                    centroid = self.cluster_centroids[start_cluster_id]
                    self.get_logger().info(f"Setting initial agent state to centroid of cluster {start_cluster_id}: {centroid}")
                    scaled_pos_x_model = (centroid[1] - self.origin_y) / self.scale_2d_3d
                    scaled_pos_y_model = (centroid[0] - self.origin_x) / self.scale_2d_3d
                    scaled_agent_state_np = np.array([scaled_pos_x_model, scaled_pos_y_model, 0.0, 0.0], dtype=np.float32)
                    self.current_agent_state = jnp.expand_dims(jnp.array(scaled_agent_state_np), axis=0)
                    
                    bearing_rad = self.plan_sequence[self.current_plan_step_index].get("bearing", 0.0)
                    self.get_logger().info(f"Initial Bearing from plan: {bearing_rad:.2f} rad ({math.degrees(bearing_rad):.2f} deg)")
                else:
                    self.get_logger().error(f"Centroid for cluster {start_cluster_id} not found in plan data!")
                    return
            self.is_first_run = False
            
        if self.current_plan_step_index >= len(self.plan_sequence):
            self.get_logger().info("High-level plan is complete. Stopping control loop.")
            self.timer.cancel()
            return

        if self.latest_ranges_msg is None or self.latest_odom_msg is None:
            self.get_logger().warning("Waiting for sensor data...")
            return
        
        current_plan_step = self.plan_sequence[self.current_plan_step_index]
        expected_start_cluster = current_plan_step["start"]
        expected_next_cluster = current_plan_step["next"]

        # 🟢 Print current, start, and end clusters
        self.get_logger().info(f"🗺️ Current plan step: From Cluster {expected_start_cluster} to Cluster {expected_next_cluster}")

        # --- AUTOMATIC ARRIVAL DETECTION LOGIC ---
        if hasattr(self, 'current_target_pos'):
            current_x = self.latest_odom_msg.pose.pose.position.x
            current_y = self.latest_odom_msg.pose.pose.position.y
            distance_left = math.sqrt((self.current_target_pos[0] - current_x)**2 + (self.current_target_pos[1] - current_y)**2)
            
            # 🟢 Print current robot position
            self.get_logger().info(f"📍 Current Position: X: {current_x:.2f}, Y: {current_y:.2f}")
            
            # 🟢 Add a buffer to the arrival tolerance
            arrival_tolerance = 0.5  # meters
            
            if distance_left > arrival_tolerance:
                self.get_logger().info(f"➡️ Navigating to target. Distance left: {distance_left:.2f} m")
                return

            self.get_logger().info("✅ Target reached! Calculating new target.")
        
        # --- END OF AUTOMATIC ARRIVAL LOGIC ---

        debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        if debug_mode:
            current_cluster_id = self.get_parameter('current_cluster_id').get_parameter_value().integer_value
            self.get_logger().info(f"DEBUG MODE: Using manual cluster ID {current_cluster_id}")
        else:
            if self.latest_predicted_cluster_id is None:
                self.get_logger().warning("Waiting for predicted cluster ID...")
                return
            current_cluster_id = self.latest_predicted_cluster_id
            # 🟢 Print the current mapped cluster ID
            self.get_logger().info(f"Default MODE: Using predicted cluster ID {current_cluster_id}")

        mapped_current_cluster = self._map_cluster_id(current_cluster_id)
        if mapped_current_cluster == 4:
            mapped_current_cluster = 2 if (int(expected_start_cluster) == 2 or int(expected_next_cluster) == 2) else 3
        label_msg = Int16()
        label_msg.data = mapped_current_cluster
        self.publisher_.publish(label_msg)
        self.get_logger().info(f"current before:{current_cluster_id}, mapped before check: {mapped_current_cluster}")

        if mapped_current_cluster == expected_next_cluster:
            self.current_plan_step_index += 1
            
            if self.current_plan_step_index < len(self.plan_sequence):
                new_plan_step = self.plan_sequence[self.current_plan_step_index]
                new_start_id = str(new_plan_step["start"])
                
                bearing_rad = new_plan_step.get("bearing", 0.0)
                self.get_logger().info(f"Transitioning to new bearing: {bearing_rad:.2f} rad ({math.degrees(bearing_rad):.2f} deg)")
            
            if self.current_plan_step_index >= len(self.plan_sequence):
                self.get_logger().info(f"Plan step complete. Transitioning to cluster {expected_next_cluster}. Plan is now finished.")
            else:
                self.get_logger().info(f"Plan step complete. Transitioning from cluster {expected_start_cluster} to {expected_next_cluster}. Next step is from cluster {self.plan_sequence[self.current_plan_step_index]['start']}.")
                
        if self.current_plan_step_index >= len(self.plan_sequence):
            return

        scaled_ranges_np = np.array(self.latest_ranges_msg.data, dtype=np.float32) / self.scale_lidar
        unscaled_agent_state = np.array(self.latest_agent_state_msg, dtype=np.float32).flatten()

        x = (unscaled_agent_state[0] - self.origin_x)*1.0 / self.scale_2d_3d
        y = (unscaled_agent_state[1] - self.origin_y)*1.0 / self.scale_2d_3d
        vx = unscaled_agent_state[2] / self.scale_2d_3d
        vy = unscaled_agent_state[3] / self.scale_2d_3d
        full_scaled_state = np.array([y, x, vy, vx])
        full_scaled_state_jnp = jnp.expand_dims(jnp.array(full_scaled_state), axis=0)

        self.get_logger().info(f"state: {full_scaled_state}")
        self.get_logger().info(f"ranges: {scaled_ranges_np}")
        # self.get_logger().info(f"og: {np.array(self.latest_ranges_msg.data, dtype=np.float32)}")
        
        state_unscaled_msg = Float32MultiArray()
        state_unscaled_msg.data = unscaled_agent_state.tolist()
        self.full_state_unscaled_pub.publish(state_unscaled_msg)
        
        state_scaled_msg = Float32MultiArray()
        state_scaled_msg.data = full_scaled_state.tolist()
        self.full_state_scaled_pub.publish(state_scaled_msg)
        
        next_cluster_msg = Int16()
        next_cluster_msg.data = expected_next_cluster
        self.next_cluster_pub.publish(next_cluster_msg)
                
        graph = self._build_state_and_graph(
            full_scaled_state_jnp,
            scaled_ranges_np,
            mapped_current_cluster,
            expected_start_cluster,
            expected_next_cluster,
            self.next_cluster_bonus_awarded
        )

        self.rng_key, action_key = jr.split(self.rng_key)
        
        action, new_rnn_state = self.algo.act(
            graph=graph,
            rnn_state=self.rnn_state,
            params={'policy': self.algo.policy_train_state.params}
        )

        self.rnn_state = new_rnn_state
        action = self.clip_action(action)
        
        self.current_agent_state = self.agent_step_euler(self.current_agent_state, action)
        
        next_state_small_frame_squeezed = jnp.squeeze(self.current_agent_state, axis=0)
        
        reward, bonus_awarded_updated = self.env_instance.get_reward(graph, action)
        
        if bonus_awarded_updated.size == 0:
            self.get_logger().warning("Received an empty bonus array from get_reward. Resetting.")
            self.next_cluster_bonus_awarded = jnp.zeros(self.env_instance.num_agents, dtype=jnp.bool_)
        else:
            self.next_cluster_bonus_awarded = bonus_awarded_updated

        next_pos_x = next_state_small_frame_squeezed[1] * self.scale_2d_3d + self.origin_y
        next_pos_y = next_state_small_frame_squeezed[0] * self.scale_2d_3d + self.origin_x
        
        self.current_target_pos = (next_pos_x, next_pos_y)
        
        action_np = np.array(action).flatten()
        action_msg = Float32MultiArray()
        action_msg.data = action_np.tolist()
        self.action_pub.publish(action_msg)

        self.get_logger().info(f"Action: {action}")
        self.get_logger().info(f"Calculated new target 🎯: X: {next_pos_x:.2f}, Y: {next_pos_y:.2f}")
        
        if self.latest_odom_msg is not None:
            orientation_q = self.latest_odom_msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            
            (_, _, current_yaw_rad) = euler_from_quaternion(orientation_list)
            
            current_plan_step = self.plan_sequence[self.current_plan_step_index]
            desired_bearing_rad = current_plan_step.get("bearing", current_yaw_rad)

            angular_diff_rad = desired_bearing_rad - current_yaw_rad
            angular_diff_rad = (angular_diff_rad + math.pi) % (2 * math.pi) - math.pi
            
            self.get_logger().info(f"Bearing to turn: {math.degrees(angular_diff_rad):.2f} deg")
        # ----------------------------------------------------------------------
        
    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """By default, use double integrator dynamics"""
        assert action.shape == (self.env_instance.num_agents, self.env_instance.action_dim)
        print(agent_states.shape)
        print((self.env_instance.num_agents, self.env_instance.state_dim))
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

    def _build_state_and_graph(self, agent_state_np: np.ndarray, scaled_ranges: np.ndarray, mapped_current_cluster_id: int, mapped_start_cluster_id: int, mapped_next_cluster_id: int, bonus_awarded_updated: jnp.ndarray)  -> GraphsTuple:        
        ## ADDED FOR DEBUGGING
        # scaled_ranges = np.array([0.5] * 72)
        # scaled_ranges[9:27] = 0.01
        # scaled_ranges[45:63] = 0.01
        # self.get_logger().info(f"ranges new: {scaled_ranges}")
        
        num_ranges = self.env_instance.params['num_ranges']
        top_k = self.env_instance.params['top_k_rays']
        
        if scaled_ranges.shape[0] != num_ranges:
            self.get_logger().warn(f"Received {scaled_ranges.shape[0]} ranges, but expected {num_ranges}. This may cause errors.")
        
        angle_increment = (2 * np.pi) / num_ranges
        angles = np.arange(num_ranges) * angle_increment
        
        # Accounting for CARLA-2D flip
        x_coords = scaled_ranges * np.cos(angles)
        y_coords = scaled_ranges * np.sin(angles)
        
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
        
        angular_offset = self.get_parameter('angular_offset_deg').get_parameter_value().double_value

        key = f"{mapped_start_cluster_id}-{mapped_next_cluster_id}"
        bearing_value = self.bearing_map.get(key, 0.0) + math.radians(angular_offset)
        self.get_logger().info(f"Start: {mapped_start_cluster_id}, Current: {mapped_current_cluster_id}, Next: {mapped_next_cluster_id}, Bearing: {bearing_value}")
        
        goal_state_np = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # inter_center_env_state: Float[Array, "2"] = jnp.array([0.0, 0.0])
        # passage_width_env_state: Float[Array, ""] = jnp.array(0.0)
        # obs_len_env_state: Float[Array, ""] = jnp.array(0.0)
        # global_angle_env_state: Float[Array, ""] = jnp.array(0.0)
        # is_four_way_env_state: bool = False

        env_state = LidarEnvState(
            agent=agent_state_np, 
            goal=jnp.array([goal_state_np]),
            obstacle=jnp.zeros((0, 2)),
            bearing=jnp.array([bearing_value]),
            current_cluster_oh=jnp.array([current_cluster_oh]),
            start_cluster_oh = jnp.array([start_cluster_oh]),
            next_cluster_oh=jnp.array([next_cluster_oh]),
            next_cluster_bonus_awarded=bonus_awarded_updated,
            bridge_center=jnp.zeros((0,2)),
            bridge_length=jnp.zeros((0,)),
            bridge_gap_width=jnp.zeros((0,)),
            bridge_wall_thickness=jnp.zeros((0,)),
            bridge_theta=jnp.zeros((0,))
            # is_four_way=is_four_way_env_state,
            # center=inter_center_env_state,
            # passage_width=passage_width_env_state,
            # obs_len=obs_len_env_state,
            # global_angle=global_angle_env_state
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
