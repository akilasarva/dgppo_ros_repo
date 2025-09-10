import open3d as o3d
import numpy as np
import os
import imageio
from PIL import Image, ImageDraw, ImageFont

def visualize_low_altitude_points_gif(folder_path, 
                                     output_gif_path="low_altitude_points.gif",
                                     z_threshold_lower=0.1, z_threshold_upper=0.5,
                                     z_threshold_lower_2=1.5, z_threshold_upper_2=2.0,
                                     max_range=5.0,
                                     angle_increment_deg=11.25):
    """
    Visualizes points within specified altitude ranges from .pcd files and saves as a GIF.
    The script now combines two altitude slices and visualizes the Lidar footprint.
    
    Args:
        folder_path (str): Path to the directory containing .pcd files.
        output_gif_path (str): Filename for the output GIF.
        z_threshold_lower (float): Lower z-axis threshold for the first slice.
        z_threshold_upper (float): Upper z-axis threshold for the first slice.
        z_threshold_lower_2 (float): Lower z-axis threshold for the second slice.
        z_threshold_upper_2 (float): Upper z-axis threshold for the second slice.
        max_range (float): Maximum range for Lidar data.
        angle_increment_deg (float): Angular resolution of the Lidar sensor in degrees.
    """
    pcd_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pcd")])

    if not pcd_files:
        print(f"No .pcd files found in folder: {folder_path}")
        return

    # Use a single visualizer for performance
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Footprint Visualization")
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])
    render_option.point_size = 3.0 # Set point size for better visibility

    images = []
    num_angles = int(360 / angle_increment_deg)

    for file_name in pcd_files[1500::3]:
        file_path = os.path.join(folder_path, file_name)

        try:
            pcd_full = o3d.io.read_point_cloud(file_path)
            if pcd_full.is_empty():
                print(f"Warning: Point cloud {file_name} is empty.")
                continue

            points_full = np.asarray(pcd_full.points)
            finite_mask_full = np.all(np.isfinite(points_full), axis=1)
            points_full_finite = points_full[finite_mask_full]

            # Combine the two slice masks based on z-axis thresholds
            slice_mask_1 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower) & \
                           (np.abs(points_full_finite[:, 2]) <= z_threshold_upper)
            
            slice_mask_2 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower_2) & \
                           (np.abs(points_full_finite[:, 2]) <= z_threshold_upper_2)

            combined_slice_mask = (slice_mask_1 | slice_mask_2) & \
                                  (np.abs(points_full_finite[:, 0]) <= max_range) & \
                                  (np.abs(points_full_finite[:, 1]) <= max_range)
            points_slice =  points_full_finite[combined_slice_mask]
            
            if points_slice.size == 0:
                print(f"Warning: No points found in specified slices for {file_name}.")
                continue

            pcd_slice = o3d.geometry.PointCloud()
            pcd_slice.points = o3d.utility.Vector3dVector(points_slice)
            pcd_slice.paint_uniform_color([0, 0, 1]) # Dark blue for the points slice

            # Generate Lidar footprint lines
            footprint_points = []
            for i in range(num_angles):
                angle_deg = i * angle_increment_deg
                angle_rad = np.deg2rad(angle_deg)
                
                angular_tolerance = np.deg2rad(angle_increment_deg / 2)
                point_angles = np.arctan2(points_slice[:, 1], points_slice[:, 0])
                angular_diff = np.arctan2(np.sin(point_angles - angle_rad), np.cos(point_angles - angle_rad))
                sector_mask = np.abs(angular_diff) <= angular_tolerance
                sector_points = points_slice[sector_mask]

                closest_point = np.array([max_range * np.cos(angle_rad), max_range * np.sin(angle_rad), 0])
                if sector_points.shape[0] > 0:
                    distances = np.linalg.norm(sector_points[:, :2], axis=1)
                    min_dist = np.min(distances)
                    if min_dist < max_range:
                        closest_index = np.argmin(distances)
                        closest_point = sector_points[closest_index]
                    else:
                        closest_point = np.array([max_range * np.cos(angle_rad), max_range * np.sin(angle_rad), 0])

                footprint_points.append([0, 0, 0])
                footprint_points.append(closest_point)

            footprint_lines = [[i, i + 1] for i in range(0, len(footprint_points), 2)]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(footprint_points),
                lines=o3d.utility.Vector2iVector(footprint_lines),
            )
            line_set.paint_uniform_color([1, 0, 0]) # Red for the footprint lines

            # Add geometries to the visualizer
            vis.clear_geometries()
            vis.add_geometry(pcd_slice)
            vis.add_geometry(line_set)
            vis.poll_events()
            vis.update_renderer()

            # Capture and annotate the image
            image = vis.capture_screen_float_buffer(do_render=True)
            pil_image = Image.fromarray(np.uint8(np.asarray(image) * 255))
            draw = ImageDraw.Draw(pil_image)

            file_name_no_ext = os.path.splitext(file_name)[0]
            if len(file_name_no_ext) >= 3:
                formatted_name = file_name_no_ext[:3] + "." + file_name_no_ext[3:]
            else:
                formatted_name = file_name_no_ext

            # Font Handling
            try:
                font = ImageFont.truetype("arial.ttf", 50)
            except IOError:
                font = ImageFont.load_default()

            draw.text((10, 10), formatted_name, font=font, fill=(255, 255, 255))
            images.append(np.array(pil_image))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    vis.destroy_window()
    if images:
        imageio.mimsave(output_gif_path, images, fps=5)
        print(f"GIF saved to: {output_gif_path}")
    else:
        print("No images were generated. GIF not created.")


# # Example Usage
# folder_path = "../../../../carla_data/bridge2_carla/bridge2_carla_pcds"
# visualize_low_altitude_points_gif(folder_path, 
#                                   output_gif_path="low_altitude_points_with_footprint.gif",
#                                   z_threshold_lower=2, z_threshold_upper=2.25, # First slice (e.g., ground)
#                                   z_threshold_lower_2=0, z_threshold_upper_2=0, # Second slice (e.g., table top)
#                                   max_range=20.0,
#                                   angle_increment_deg=5)

# # Example Usage
# folder_path = "../../../../carla_data/bridge_rural/bridge_rural_pcds"
# visualize_low_altitude_points_gif(folder_path, 
#                                   output_gif_path="low_altitude_points_with_footprint.gif",
#                                   z_threshold_lower=1.8, z_threshold_upper=2.1, # First slice (e.g., ground)
#                                   z_threshold_lower_2=0, z_threshold_upper_2=0, # Second slice (e.g., table top)
#                                   max_range=20.0,
#                                   angle_increment_deg=5)

# # Example Usage
# folder_path = "../../../../carla_data/intersection_y/intersection_y_pcds"
# visualize_low_altitude_points_gif(folder_path, 
#                                   output_gif_path="low_altitude_points_with_footprint.gif",
#                                   z_threshold_lower=0, z_threshold_upper=2, # First slice (e.g., ground)
#                                   z_threshold_lower_2=0, z_threshold_upper_2=0, # Second slice (e.g., table top)
#                                   max_range=20.0,
#                                   angle_increment_deg=5)

# # Example Usage
# folder_path = "../../../../carla_data/intersection_straight/intersection_straight_pcds"
# visualize_low_altitude_points_gif(folder_path, 
#                                   output_gif_path="low_altitude_points_with_footprint.gif",
#                                   z_threshold_lower=1.5, z_threshold_upper=2.2, # First slice (e.g., ground)
#                                   z_threshold_lower_2=0, z_threshold_upper_2=0, # Second slice (e.g., table top)
#                                   max_range=20.0,
#                                   angle_increment_deg=5)

# # Example Usage
# folder_path = "../../../../carla_data/building1_carla/building1_carla_pcds"
# visualize_low_altitude_points_gif(folder_path, 
#                                   output_gif_path="low_altitude_points_with_footprint.gif",
#                                   z_threshold_lower=1.8, z_threshold_upper=2.1, # First slice (e.g., ground)
#                                   z_threshold_lower_2=0, z_threshold_upper_2=0, # Second slice (e.g., table top)
#                                   max_range=15.0,
#                                   angle_increment_deg=5)

# Example Usage
folder_path = "/home/akilasar/ros2_ws/full_bag/full_bag_pcds"
visualize_low_altitude_points_gif(folder_path, 
                                  output_gif_path="low_altitude_points_with_footprint.gif",
                                  z_threshold_lower=0, z_threshold_upper=5, # First slice (e.g., ground)
                                  z_threshold_lower_2=0, z_threshold_upper_2=0, # Second slice (e.g., table top)
                                  max_range=25.0,
                                  angle_increment_deg=5)







