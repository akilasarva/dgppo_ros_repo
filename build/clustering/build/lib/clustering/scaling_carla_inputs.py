import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import qos_profile_sensor_data

# ROS2 Messages
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

class CarlaScaling(Node):
    def __init__(self):
        super().__init__('carla_scaling')

        self.get_logger().info("Initializing Carla Scaling Node...")

        self.latest_ranges_msg = None
        self.latest_agent_state_msg = None

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

        # Timer to print out data every 5 seconds
        self.print_timer = self.create_timer(1.0, self.print_data_callback)
        self.get_logger().info("Node initialized. Listening for messages and will print data every 5 seconds.")

    def ranges_callback(self, msg: Float32MultiArray):
        """Callback for the /processed_ranges topic."""
        self.latest_ranges_msg = msg

    def agent_state_callback(self, msg: Odometry):
        """Callback for the /carla/ego_vehicle/odometry topic."""
        self.latest_agent_state_msg = msg

    def print_data_callback(self):
        """
        Callback for the timer. This function is executed once every second
        to print the latest stored sensor data.
        """
        if self.latest_agent_state_msg is None and self.latest_ranges_msg is None:
            self.get_logger().warning("Waiting for both odometry and ranges data...")
            return

        # Print Odometry Data if available
        if self.latest_agent_state_msg is not None:
            pos = self.latest_agent_state_msg.pose.pose.position
            vel = self.latest_agent_state_msg.twist.twist.linear
            self.get_logger().info(f"Odometry: Position (x={pos.x:.2f}, y={pos.y:.2f}) | Velocity (vx={vel.x:.2f}, vy={vel.y:.2f})")
        else:
            self.get_logger().warning("No odometry data received yet.")

        # Print Ranges Data if available
        if self.latest_ranges_msg is not None:
            # Print the first 5 values from the data array
            first_five_scans = np.array(self.latest_ranges_msg.data[:5])
            self.get_logger().info(f"Lidar Scans: First 5 values are {first_five_scans.tolist()}")
        else:
            self.get_logger().warning("No ranges data received yet.")

def main(args=None):
    rclpy.init(args=args)
    node = CarlaScaling()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()