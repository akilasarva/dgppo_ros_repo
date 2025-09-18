# Create a new file, e.g., 'lidar_processor.py'
import numpy as np

# def get_ranges_from_points(points, config, max_range = 24):
#     # This is the unified function for both training and inference
#     finite_mask_full = np.isfinite(points).all(axis=1)
#     points_full_finite = points[finite_mask_full]

#     slice_mask_1 = (np.abs(points_full_finite[:, 2]) >= config['z_threshold_lower']) & \
#                    (np.abs(points_full_finite[:, 2]) <= config['z_threshold_upper'])
    
#     slice_mask_2 = (np.abs(points_full_finite[:, 2]) >= config['z_threshold_lower_2']) & \
#                    (np.abs(points_full_finite[:, 2]) <= config['z_threshold_upper_2'])

#     combined_slice_mask = (slice_mask_1 | slice_mask_2) & \
#                           (np.abs(points_full_finite[:, 0]) <= max_range) & \
#                           (np.abs(points_full_finite[:, 1]) <= max_range)
#     points_slice = points_full_finite[combined_slice_mask]

#     if points_slice.shape[0] == 0:
#         return np.full(config['num_ranges'], max_range, dtype=np.float32)

#     ranges = np.full(config['num_ranges'], max_range, dtype=np.float32)
#     point_angles = np.arctan2(points_slice[:, 1], points_slice[:, 0])
#     distances = np.linalg.norm(points_slice[:, :2], axis=1)

#     for i in range(config['num_ranges']):
#         angle_rad = np.deg2rad(i * config['angle_increment_deg'])
#         angular_tolerance = np.deg2rad(config['angle_increment_deg'] / 2)
#         angular_diff = np.arctan2(np.sin(point_angles - angle_rad), np.cos(point_angles - angle_rad))
#         sector_mask = np.abs(angular_diff) <= angular_tolerance
#         sector_distances = distances[sector_mask]
        
#         if len(sector_distances) > 0:
#             min_dist = np.min(sector_distances)
#             ranges[i] = min(min_dist, max_range)
    
#     return ranges

def get_ranges_from_points(points, config):
    """
    Processes a point cloud to generate a 1D Lidar range scan.
    Points are filtered by z-axis and radial distance (min/max range).
    """
    if points.size == 0:
        return np.full(config['num_ranges'], config['max_lidar_range'])

    # Filter points by altitude
    slice_mask_1 = (np.abs(points[:, 2]) >= config['z_threshold_lower']) & \
                   (np.abs(points[:, 2]) <= config['z_threshold_upper'])
    slice_mask_2 = (np.abs(points[:, 2]) >= config['z_threshold_lower_2']) & \
                   (np.abs(points[:, 2]) <= config['z_threshold_upper_2'])

    combined_z_mask = slice_mask_1 | slice_mask_2
    points_z_filtered = points[combined_z_mask]

    if points_z_filtered.size == 0:
        return np.full(config['num_ranges'], config['max_lidar_range'])

    # Calculate radial distances and apply min/max range filter
    distances = np.linalg.norm(points_z_filtered[:, :2], axis=1)
    
    # Filter points by radial range
    radial_mask = (distances >= config['min_lidar_range']) & (distances <= config['max_lidar_range'])
    points_filtered = points_z_filtered[radial_mask]

    if points_filtered.size == 0:
        return np.full(config['num_ranges'], config['max_lidar_range'])

    # Generate 1D ranges
    ranges = []
    num_angles = config['num_ranges']
    angle_increment_deg = float(360.0 / num_angles)

    for i in range(num_angles):
        angle_deg = i * angle_increment_deg
        angle_rad = np.deg2rad(angle_deg)
        
        # Determine angular tolerance for each sector
        angular_tolerance = np.deg2rad(angle_increment_deg / 2)
        point_angles = np.arctan2(points_filtered[:, 1], points_filtered[:, 0])
        angular_diff = np.arctan2(np.sin(point_angles - angle_rad), np.cos(point_angles - angle_rad))
        sector_mask = np.abs(angular_diff) <= angular_tolerance
        sector_points = points_filtered[sector_mask]
        
        # Find the minimum distance in the sector
        if sector_points.shape[0] > 0:
            distances_in_sector = np.linalg.norm(sector_points[:, :2], axis=1)
            min_dist = np.min(distances_in_sector)
            ranges.append(min_dist)
        else:
            ranges.append(config['max_lidar_range'])
            
    return np.array(ranges)
