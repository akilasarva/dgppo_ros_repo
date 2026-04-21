import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import carla
import os
import sys
import time
import math
import random
import numpy as np
import queue
import open3d as o3d
from matplotlib import cm
import json
import threading
import rclpy.executors

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Int16

# Auxilliary Code for Visualization
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:, :3]

def add_open3d_axis(vis):
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

class CarlaBridgeNode(Node):
    def __init__(self):
        super().__init__('carla_bridge_node')
        self.get_logger().info("Initializing CARLA Bridge Node...")

        # --- CARLA and O3D Setup ---
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town01')
        self.bp_lib = self.world.get_blueprint_library()
        self.vehicle = None
        self.lidar = None
        self.collision_sensor = None
        self.is_vehicle_ready = False
        self.collision_detected = False
        self.initial_goal_published = False
        
        # --- Obstacle Avoidance Parameters ---
        self.is_obstacle_detected = False
        self.closest_obstacle_dist = float('inf')
        self.closest_obstacle_y = 0.0
        self.AVOIDANCE_DISTANCE = 10.0
        self.LATERAL_AVOIDANCE_THRESHOLD = 1.5
        
        self.destination = None
        self.transformed_start_point = None

        # --- ROS2 Publishers and Subscriber ---
        self.odom_publisher = self.create_publisher(Odometry, '/carla/ego_vehicle/odometry', 10)
        self.ranges_publisher = self.create_publisher(Float32MultiArray, '/processed_ranges', qos_profile=qos_profile_sensor_data)
        self.cluster_id_publisher = self.create_publisher(Int16, '/predicted_cluster', 10)
        self.goal_publisher = self.create_publisher(PoseStamped, '/transformed_goal', 10)
        
        self.waypoint_subscriber = self.create_subscription(PoseStamped, '/dgppo_waypoint', self.waypoint_callback, 10)
        self.current_waypoint = None

        # --- O3D Visualization Setup ---
        self.vis = None
        self.point_list = None
        
        self.get_logger().info("CARLA Bridge Node initialized, waiting for CARLA setup.")
        
        self.spectator = self.world.get_spectator()

    def setup_carla(self):
        
        if not self.is_vehicle_ready:
            self.get_logger().info("Attempting to spawn vehicle...")
            vehicle_bp = self.bp_lib.find('vehicle.lincoln.mkz_2020')

            # --- SPAWNING AND TELEPORTATION LOGIC ---
            safe_spawn_point = random.choice(self.world.get_map().get_spawn_points())
            vehicle = self.world.try_spawn_actor(vehicle_bp, safe_spawn_point)

            if vehicle is not None:
                self.vehicle = vehicle
                self.transformed_start_point = carla.Transform(carla.Location(x=90.0, y=140.0, z=1.0))
                self.vehicle.set_transform(self.transformed_start_point)
                self.get_logger().info(f"Vehicle successfully spawned at a safe point, then teleported to {self.transformed_start_point.location}")

            if self.vehicle is not None:
                self.is_vehicle_ready = True
                self.get_logger().info("Vehicle spawned. Setting up sensors...")
                
                # CORRECT ORDER: Setup visualization BEFORE spawning sensors
                self._setup_visualization()
                self._spawn_sensors()
                self.spectator = self.world.get_spectator()
                
                transformed_destination_transform = carla.Transform(carla.Location(x=-150, y=10, z=1))
                self.destination = transformed_destination_transform.location
                
                self.world.debug.draw_string(self.transformed_start_point.location, 'Start', life_time=60, color=carla.Color(0, 255, 0))
                self.world.debug.draw_string(self.destination, 'Transformed Stop', life_time=60, color=carla.Color(255, 0, 0))
            else:
                self.get_logger().error("Failed to spawn vehicle. Exiting script.")
                
    def _spawn_sensors(self):
        bp_lib = self.world.get_blueprint_library()
        
        # Lidar Sensor
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('points_per_second', '500000')
        lidar_init_trans = carla.Transform(carla.Location(z=2))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=self.vehicle)
        self.lidar.listen(lambda data: self.lidar_callback(data))
        
        if self.lidar is None:
            self.get_logger().error("Failed to spawn lidar sensor!")
            return
        
        # Collision Sensor
        collision_bp = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_callback(event))

    def _setup_visualization(self):
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(window_name='Carla Lidar', width=960, height=540, left=480, top=270)
        # self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        # self.vis.get_render_option().point_size = 1
        # self.vis.get_render_option().show_coordinate_frame = True
        # add_open3d_axis(self.vis)
        self.point_list = o3d.geometry.PointCloud()
        # self.vis.add_geometry(self.point_list)

    def waypoint_callback(self, msg: PoseStamped):
        self.current_waypoint = carla.Location(
            x=msg.pose.position.x,
            y=msg.pose.position.y,
            z=msg.pose.position.z
        )
        self.get_logger().info(f"Received new waypoint: ({self.current_waypoint.x:.2f}, {self.current_waypoint.y:.2f})")
        
    def lidar_callback(self, point_cloud):
        self.get_logger().info("Lidar data received!")

        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        
        self.is_obstacle_detected = False
        self.closest_obstacle_dist = float('inf')
        self.closest_obstacle_y = 0.0
        
        for point in data:
            x, y, z = point[0], point[1], point[2]
            distance = math.sqrt(x**2 + y**2 + z**2)
            if x > 0.5 and distance < self.AVOIDANCE_DISTANCE and abs(y) < self.LATERAL_AVOIDANCE_THRESHOLD:
                self.is_obstacle_detected = True
                if distance < self.closest_obstacle_dist:
                    self.closest_obstacle_dist = distance
                    self.closest_obstacle_y = y
        
        ranges = np.linalg.norm(data[:, :2], axis=1)
        ranges_msg = Float32MultiArray()
        ranges_msg.data = ranges.tolist()
        self.ranges_publisher.publish(ranges_msg)
        
        points = data[:, :-1]
        points[:, :1] = -points[:, :1]
        self.point_list.points = o3d.utility.Vector3dVector(points)
        # self.vis.update_geometry(self.point_list)
        
    def collision_callback(self, event):
        self.get_logger().info(f"Collision detected! Other actor: {event.other_actor.type_id}")
        self.collision_detected = True
        
    def run_main_loop(self):
        self.get_logger().info("Starting main control loop...")
        
        while not self.collision_detected and self.vehicle is not None:
            control = carla.VehicleControl()
            if self.is_obstacle_detected:
                self.get_logger().info(f"Obstacle detected! Distance: {self.closest_obstacle_dist:.2f}m")
                throttle = max(0.0, (self.closest_obstacle_dist - 1.0) / self.AVOIDANCE_DISTANCE)
                steer_factor = min(1.0, 1.0 - (self.closest_obstacle_dist / self.AVOIDANCE_DISTANCE))
                steer = np.sign(self.closest_obstacle_y) * steer_factor * 1.0
                control.throttle = throttle
                control.steer = steer
            elif self.current_waypoint is not None:
                vehicle_transform = self.vehicle.get_transform()
                direction_vector = self.current_waypoint - vehicle_transform.location
                direction_vector.z = 0
                
                forward_vector = vehicle_transform.get_forward_vector()
                dot_product = forward_vector.x * direction_vector.x + forward_vector.y * direction_vector.y
                cross_product = forward_vector.x * direction_vector.y - forward_vector.y * direction_vector.x
                angle_to_steer = math.atan2(cross_product, dot_product)
                
                control.throttle = 0.5
                control.steer = angle_to_steer
                self.get_logger().info(f"Navigating to waypoint. Steer: {angle_to_steer:.2f}")

            self.vehicle.apply_control(control)
            
            spectator_transform = carla.Transform(self.vehicle.get_transform().transform(
                carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
            self.spectator.set_transform(spectator_transform)

            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = 'map'
            odom_msg.pose.pose.position.x = self.vehicle.get_location().x
            odom_msg.pose.pose.position.y = self.vehicle.get_location().y
            odom_msg.twist.twist.linear.x = self.vehicle.get_velocity().x
            self.odom_publisher.publish(odom_msg)

            # self.vis.update_geometry(self.point_list)
            # self.vis.poll_events()
            # self.vis.update_renderer()
            
            # This sleep is important to prevent the loop from being CPU-bound
            time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    carla_node = CarlaBridgeNode()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(carla_node)
    
    # Start the ROS 2 spin in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        # The main thread now handles the CARLA setup and O3D loop
        carla_node.setup_carla()
        if carla_node.is_vehicle_ready:
            carla_node.run_main_loop()

    except KeyboardInterrupt:
        pass
    finally:
        # Proper cleanup of actors
        if carla_node.lidar is not None and carla_node.lidar.is_alive:
            carla_node.lidar.destroy()
        if carla_node.collision_sensor is not None and carla_node.collision_sensor.is_alive:
            carla_node.collision_sensor.destroy()
        if carla_node.vehicle is not None and carla_node.vehicle.is_alive:
            carla_node.vehicle.destroy()
            
        carla_node.destroy_node()
        executor.shutdown()
        executor_thread.join()
        rclpy.shutdown()

if __name__ == '__main__':
    main()