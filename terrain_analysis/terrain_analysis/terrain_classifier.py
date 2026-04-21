import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct

class TerrainClassifier(Node):
    def __init__(self):
        super().__init__('terrain_classifier')
        # Change '/carla/ego_vehicle/lidar' if your rosbag uses a different topic name
        self.subscription = self.create_subscription(
            PointCloud2, '/livox/lidar', self.lidar_callback, 10)
        self.publisher = self.create_publisher(
            PointCloud2, '/terrain_colored_cloud', 10)
        self.get_logger().info('Terrain Classifier Node Started')

    def lidar_callback(self, msg):
        # Read points: x, y, z are floats, intensity is usually float or uint8
        points = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        points_list = list(points)
        
        if not points_list:
            return

        # Prepare output data structure
        new_points = []
        
        for p in points_list:
            x, y, z, intensity = p
            
            # Map Intensity (0-255) to RGB
            # Blue: [0,0,255], LightGrey: [211,211,211], Green: [0,255,0], DarkGrey: [50,50,50]
            if intensity <= 30:
                r, g, b = 255, 0, 0      # Road
            elif 30 < intensity <= 60:
                r, g, b = 0, 0, 255  # Sidewalk
            elif 60 < intensity <= 255:
                r, g, b = 0, 255, 0      # Grass
            else:
                r, g, b = 50, 50, 50     # Other

            # Pack RGB into a single float
            rgb = struct.unpack('f', struct.pack('I', (r << 16 | g << 8 | b)))[0]
            new_points.append([x, y, z, rgb])

        # Create the new PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        header = msg.header
        out_msg = pc2.create_cloud(header, fields, new_points)
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TerrainClassifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
