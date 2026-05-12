#!/usr/bin/env python3
"""
Bench spoof node — publishes fake sensor data for dgppo_test_node bench testing.

Publishes:
  /processed_ranges   Float32MultiArray  — hallway-shaped LiDAR (walls left/right)
  /predicted_cluster  Int16              — fixed cluster ID
  /current_terrain    Int32              — fixed terrain type

Usage:
  ros2 run dgppo_ros_node_pkg bench_spoof_node

  # override cluster and terrain via params:
  ros2 run dgppo_ros_node_pkg bench_spoof_node \
    --ros-args -p cluster_id:=1 -p terrain_id:=1 -p wall_dist:=0.8
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int16, Int32, Float32MultiArray
import numpy as np


def _hallway_ranges(num_bins: int, wall_dist: float, max_range: float) -> list:
    """
    Compute a 1D range array for a robot centred in an infinite hallway.
    Side walls are at `wall_dist` metres; corridor is open fore/aft.
    Range at angle θ = wall_dist / |sin(θ)|, capped at max_range.
    """
    angles = np.linspace(0.0, 2 * np.pi, num_bins, endpoint=False)
    sin_vals = np.abs(np.sin(angles))
    # avoid divide-by-zero for angles near 0° / 180°
    with np.errstate(divide='ignore', invalid='ignore'):
        ranges = np.where(sin_vals < 1e-3, max_range, wall_dist / sin_vals)
    return np.clip(ranges, 0.0, max_range).tolist()


class BenchSpoofNode(Node):
    def __init__(self):
        super().__init__('bench_spoof_node')

        self.declare_parameter('num_bins',   72)
        self.declare_parameter('wall_dist',  0.8)   # metres to side wall
        self.declare_parameter('max_range',  7.5)   # open-end cap (< 8.0 hard max)
        self.declare_parameter('cluster_id', 1)
        self.declare_parameter('terrain_id', 1)     # 0=Road 1=Grass 2=Sidewalk
        self.declare_parameter('rate_hz',    10.0)

        num_bins  = self.get_parameter('num_bins').value
        wall_dist = self.get_parameter('wall_dist').value
        max_range = self.get_parameter('max_range').value
        rate_hz   = self.get_parameter('rate_hz').value

        self._ranges = _hallway_ranges(num_bins, wall_dist, max_range)
        self.get_logger().info(
            f"Hallway scan: {num_bins} bins, wall={wall_dist} m, "
            f"max={max_range} m  (forward={self._ranges[0]:.2f} m, "
            f"side={self._ranges[num_bins // 4]:.2f} m)"
        )

        self._ranges_pub  = self.create_publisher(
            Float32MultiArray, '/processed_ranges', qos_profile_sensor_data
        )
        self._cluster_pub = self.create_publisher(Int16,  '/predicted_cluster', 10)
        self._terrain_pub = self.create_publisher(Int32,  '/current_terrain',   10)

        self.create_timer(1.0 / rate_hz, self._publish)

    def _publish(self):
        cluster_id = self.get_parameter('cluster_id').value
        terrain_id = self.get_parameter('terrain_id').value

        ranges_msg = Float32MultiArray()
        ranges_msg.data = self._ranges
        self._ranges_pub.publish(ranges_msg)

        cluster_msg = Int16()
        cluster_msg.data = cluster_id
        self._cluster_pub.publish(cluster_msg)

        terrain_msg = Int32()
        terrain_msg.data = terrain_id
        self._terrain_pub.publish(terrain_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BenchSpoofNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
