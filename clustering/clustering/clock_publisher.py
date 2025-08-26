import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

class ClockPublisher(Node):
    def __init__(self):
        super().__init__('clock_publisher')
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/warthog1/platform/odom',  # Change to a topic with a high publishing rate
            self.odom_callback,
            10
        )
        self.get_logger().info('Clock publisher node started.')

    def odom_callback(self, msg):
        # Create a new Clock message with the timestamp from the Odometry message
        clock_msg = Clock()
        clock_msg.clock = msg.header.stamp
        # Publish the Clock message
        self.clock_pub.publish(clock_msg)

def main(args=None):
    rclpy.init(args=args)
    clock_publisher = ClockPublisher()
    rclpy.spin(clock_publisher)
    clock_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
