import rclpy
from rclpy.node import Node
import jax.numpy as jnp
import jax.random as jr
import jax
import yaml
import os
import numpy as np

# ROS2 Messages
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16

# Custom message for agent state - you'll need to define this
# For simplicity, we'll assume a message with fields for position and velocity
# from your_package_name.msg import RobotStateMessage 

# DGPPO and LidarEnv components
from new_dgppo.dgppo.env.lidar_env.lidar_target import LidarTarget, LidarEnvState
from new_dgppo.dgppo.algo.dgppo import DGPPO
from new_dgppo.dgppo.algo import make_algo
from new_dgppo.dgppo.utils.graph import GraphsTuple
from new_dgppo.dgppo.utils.typing import Array, Action
from new_dgppo.dgppo.utils.utils import jax_vmap, tree_index

# Assume a simple message structure for agent state for this example.
# In a real-world scenario, you would create a custom ROS2 message for this.
from nav_msgs.msg import Odometry

class DGPPOROSNode(Node):
    def __init__(self):
        super().__init__('dgppo_ros_node')

        self.get_logger().info("Initializing DGPPO ROS Node...")

        # 1. Load the trained model and config.
        # This assumes your model directory has a 'config.yaml' and 'models/step' folders.
        model_dir = "../../../../../new_dgppo/logs/LidarTarget/dgppo/bearing_new"
        config_path = os.path.join(model_dir, "config.yaml")
        params_path = os.path.join(model_dir, "models")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Get which step to load (e.g., latest or best)
        step = self._get_model_step(model_dir)

        # 2. Instantiate the environment object
        env_kwargs = config.get("env_kwargs", {})
        self.env_instance = LidarTarget(
            num_agents=config.get('num_agents'),
            params=env_kwargs.get('params', {}),
            **{k: v for k, v in env_kwargs.items() if k != 'params'}
        )

        # 3. Instantiate the DGPPO algorithm
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

        # Load the trained model weights
        self.algo.load(params_path, step=step)
        
        # 4. Initialize state variables for the control loop
        self.rng_key = jr.PRNGKey(config.get('seed', 0))
        # This is the crucial initial RNN state
        self.rnn_state = self.algo.init_rnn_state

        # Store the static bridge parameters from the environment config
        self.bridge_params = self._get_static_bridge_params(config['env_kwargs'])
        
        # Latest data from subscribers
        self.latest_lidar_msg = None
        self.latest_agent_state_msg = None
        self.latest_cluster_id_msg = None
        
        # 5. Set up ROS2 subscribers and publisher
        # Publisher for the robot's command velocity
        # Publisher for the robot's command velocity
        self.publisher_ = self.create_publisher(Twist, '/warthog1/cmd_vel', 10)

        # Subscriber for the robot's odometry/state
        self.agent_state_sub = self.create_subscription(
            Odometry,
            '/warthog1/platform/odom',
            self.agent_state_callback,
            10
        )
        
        # Subscriber for the raw lidar data
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/my_lidar_scan',
            self.lidar_callback,
            10
        )
        
        # Subscriber for the predicted cluster ID from your other node
        self.cluster_id_sub = self.create_subscription(
            Int16,
            '/predicted_cluster',
            self.cluster_id_callback,
            10
        )
        
        # Use a timer to trigger the main control loop at a fixed frequency
        # This ensures the policy runs even if one sensor is slightly delayed.
        self.timer = self.create_timer(0.1, self.control_loop) # 10 Hz
        
        self.get_logger().info("DGPPO ROS Node fully initialized and ready.")

    def _get_model_step(self, model_dir):
        """Finds the latest or best model step from the log directory."""
        model_path = os.path.join(model_dir, "models")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        # Simplified: just load the latest step
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        self.get_logger().info(f"Loading latest model from step: {step}")
        return step

    def _get_static_bridge_params(self, env_kwargs):
        """Extracts the static bridge parameters needed to reconstruct the state."""
        params = env_kwargs.get('params', {})
        # Note: In your training code, these were often randomized. 
        # For a live test, you need to use the specific values from the curriculum.
        return {
            "bridge_center": jnp.array(params.get("bridge_center", [0.0, 0.0])),
            "bridge_length": params.get("bridge_length", 0.0),
            "bridge_gap_width": params.get("bridge_gap_width", 0.0),
            "bridge_wall_thickness": params.get("bridge_wall_thickness", 0.0),
            "bridge_theta": params.get("bridge_theta", 0.0),
            "num_bridges": params.get("num_bridges", 0)
        }
        
    def agent_state_callback(self, msg: Odometry):
        self.latest_agent_state_msg = msg
        
    def lidar_callback(self, msg: LaserScan):
        self.latest_lidar_msg = msg

    def cluster_id_callback(self, msg: Int16):
        self.latest_cluster_id_msg = msg
        
    def control_loop(self):
        """The main control loop triggered by a timer."""
        # 1. Check if we have received all necessary data
        if self.latest_lidar_msg is None or self.latest_agent_state_msg is None or self.latest_cluster_id_msg is None:
            self.get_logger().warn("Waiting for all sensor data...")
            return

        # 2. Build the state graph from ROS messages
        graph = self._build_state_and_graph(
            self.latest_agent_state_msg,
            self.latest_lidar_msg,
            self.latest_cluster_id_msg
        )

        # 3. Get the action from the policy
        self.rng_key, action_key = jr.split(self.rng_key)
        
        # Call the core act function from DGPPO
        action, new_rnn_state = self.algo.act(
            graph=graph, 
            rnn_state=self.rnn_state,
            params={'policy': self.algo.policy_train_state.params}, 
            key=action_key
        )
        
        # 4. Update the RNN state for the next step
        self.rnn_state = new_rnn_state
        
        # 5. Convert the action to a ROS Twist message and publish
        twist_msg = self._action_to_twist(action)
        self.publisher_.publish(twist_msg)
        
    def _build_state_and_graph(self, odom_msg: Odometry, lidar_msg: LaserScan, cluster_id_msg: Int16) -> GraphsTuple:
        """
        Converts ROS message data into a JAX GraphsTuple. This is the core
        bridging function.
        """
        
        # A. Extract data from ROS messages
        # Odometry provides position (x, y) and linear velocity (vx, vy)
        pos = odom_msg.pose.pose.position
        vel = odom_msg.twist.twist.linear
        # Agent state is [x, y, vx, vy]
        agent_state_np = np.array([pos.x, pos.y, vel.x, vel.y], dtype=np.float32)

        # Lidar data conversion
        # The LidarTarget environment expects lidar hits as (x, y) coordinates
        # You'll need to convert the raw ranges into relative coordinates.
        # This is a placeholder as the LidarTarget class expects pre-processed data.
        # Assuming you've already implemented a function to do this.
        lidar_data_np = self._process_lidar_to_coordinates(lidar_msg)
        
        # Cluster ID to one-hot encoding
        cluster_id = cluster_id_msg.data
        num_clusters = 3 # This should match your training config
        current_cluster_oh = jax.nn.one_hot(cluster_id, num_classes=num_clusters)
        
        # The target cluster ID needs to be determined by your curriculum logic.
        # For a live system, this might be a static target or a curriculum that progresses.
        # For now, let's assume the target is always the next cluster.
        next_cluster_id = (cluster_id + 1) % num_clusters
        next_cluster_oh = jax.nn.one_hot(next_cluster_id, num_classes=num_clusters)
        
        # B. Reconstruct the LidarEnvState object
        # Note: You'll also need the goal position for the goal node in the graph.
        # This example assumes the goal is static.
        goal_state_np = np.array([0.0, 0.0, 0.0, 0.0]) # Example goal
        
        env_state = LidarEnvState(
            agent=jnp.array([agent_state_np]), # Needs to be batch-sized
            goal=jnp.array([goal_state_np]),
            obstacle=self.env_instance.obstacles,
            bearing=jnp.zeros((1,)), # You might need to compute this from your heading
            current_cluster_oh=jnp.array([current_cluster_oh]),
            next_cluster_oh=jnp.array([next_cluster_oh]),
            **self.bridge_params
        )

        # C. Call the environment's method to build the GraphsTuple
        graph = self.env_instance.get_graph(env_state, jnp.array(lidar_data_np))
        return graph

    def _process_lidar_to_coordinates(self, msg: LaserScan):
        """
        Converts a LaserScan message into a flat array of (x, y) hit coordinates
        relative to the robot.
        """
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        angles = np.arange(len(ranges)) * angle_increment + angle_min
        
        # Filter out invalid ranges (e.g., inf)
        valid_indices = np.isfinite(ranges)
        ranges = ranges[valid_indices]
        angles = angles[valid_indices]
        
        # Convert polar to Cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        
        # The environment expects top_k_rays, so you'll need to select the
        # closest obstacles. This is a simplified implementation.
        data = np.stack([x_coords, y_coords], axis=-1)
        top_k = self.env_instance.params['top_k_rays']
        distances = np.linalg.norm(data, axis=-1)
        sorted_indices = np.argsort(distances)
        return data[sorted_indices[:top_k]]
    
    def _action_to_twist(self, action: Action) -> Twist:
        """Converts a JAX action array to a ROS Twist message."""
        twist = Twist()
        # Scale the action from the policy to appropriate velocities
        twist.linear.x = float(action[0]) * 1.0 # Scale linear velocity
        twist.angular.z = float(action[1]) * 2.0 # Scale angular velocity
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
