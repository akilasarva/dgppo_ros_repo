#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import open3d as o3d
import threading
import time
import sensor_msgs_py.point_cloud2 as pc2  # The ROS 2 equivalent for PointCloud2 utilities

class RealTimePointCloudVisualizer(Node):
    def __init__(self, topic_name='/velodyne_points'):
        super().__init__('pointcloud_visualizer')
        self.subscriber = self.create_subscription(
            PointCloud2,
            topic_name,
            self.callback,
            10)
        self.get_logger().info(f"Subscribed to topic: {topic_name}")
        
        self.points_lock = threading.Lock()
        self.latest_points = None
        self.visualizer_thread = threading.Thread(target=self.run_visualization)
        self.visualizer_thread.daemon = True
        self.visualizer_thread.start()

    def callback(self, data: PointCloud2):
        """
        ROS 2 callback to receive PointCloud2 messages and process them.
        """
        self.get_logger().info("Received PointCloud2 message", throttle_duration_sec=1.0)
        
        # Correctly read points and convert them to a list of tuples
        points = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
        points_list = list(points)

        if points_list:
            # Correctly convert the list of tuples to a NumPy array
            np_points_structured = np.array(points_list)
        
            # Extract the 'x', 'y', and 'z' fields and combine them into a simple Nx3 float array
            np_points = np.stack([np_points_structured['x'], 
                                np_points_structured['y'], 
                                np_points_structured['z']], axis=-1).astype(np.float32)
            
            with self.points_lock:
                self.latest_points = np_points
                
    def run_visualization(self):
        """
        Threaded function to handle the Open3D visualization loop.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        is_first_scan = True

        while rclpy.ok():
            updated_points = None
            with self.points_lock:
                if self.latest_points is not None:
                    updated_points = self.latest_points
                    self.latest_points = None # Reset to avoid visualizing the same frame
            
            if updated_points is not None:
                # Apply the visualization logic: find low-altitude points
                altitude_threshold = np.percentile(updated_points[:, 2], 15)
                low_altitude_points = updated_points[updated_points[:, 2] <= altitude_threshold]
                
                if low_altitude_points.size > 0:
                    pcd.points = o3d.utility.Vector3dVector(low_altitude_points)
                    colors = np.zeros_like(low_altitude_points)
                    colors[:, 2] = 1 # Dark blue color for visualization
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                    if is_first_scan:
                        vis.add_geometry(pcd)
                        is_first_scan = False
                    else:
                        vis.update_geometry(pcd)
                    vis.update_renderer()
                    vis.poll_events()
            
            time.sleep(0.01) # Small sleep to reduce CPU usage
        
        vis.destroy_window()

def main(args=None):
    rclpy.init(args=args)
    visualizer_node = RealTimePointCloudVisualizer(topic_name='/warthog1/sensors/ouster/points')
    rclpy.spin(visualizer_node)
    visualizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
