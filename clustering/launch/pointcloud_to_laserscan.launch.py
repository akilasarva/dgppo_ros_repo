from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

def generate_launch_description():
    # Define the QoS profile for the laser scan topic
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        depth=10
    )

    # Get the path to the ekf.yaml configuration file
    pkg_share_path = get_package_share_directory('clustering')
    ekf_config_path = os.path.join(pkg_share_path, 'config', 'ekf.yaml')

    return LaunchDescription([
        # 1. The pointcloud_to_laserscan node
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            remappings=[
                ('/cloud_in', '/warthog1/sensors/ouster/points'),
                ('/scan', '/my_lidar_scan')
            ],
            parameters=[{
                'target_frame': 'warthog1/lidar_link',
                'transform_tolerance': 0.01,
                'min_height': 0.65,
                'max_height': 0.75,
                'angle_increment': 0.19635,
                'scan_time': 0.1,
                'range_min': 0.0,
                'range_max': 10.0,
                'use_inf': False,
                'inf_epsilon': 1.0,
                'qos_history': 2,
                'qos_depth': 50
            }]
            # # Corrected way to set QoS overrides
            # qos_overrides={'/my_lidar_scan': qos_profile}
        ),

        # 2. The robot_localization ekf_node
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config_path, {'use_sim_time': True}],
            remappings=[('/odom', '/warthog1/platform/odom')]
        )
    ])