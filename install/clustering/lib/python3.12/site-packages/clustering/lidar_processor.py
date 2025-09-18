# Create a new file, e.g., 'lidar_processor.py'
import numpy as np

def get_ranges_from_points(points, config, max_range = 24):
    # This is the unified function for both training and inference
    finite_mask_full = np.isfinite(points).all(axis=1)
    points_full_finite = points[finite_mask_full]

    slice_mask_1 = (np.abs(points_full_finite[:, 2]) >= config['z_threshold_lower']) & \
                   (np.abs(points_full_finite[:, 2]) <= config['z_threshold_upper'])
    
    slice_mask_2 = (np.abs(points_full_finite[:, 2]) >= config['z_threshold_lower_2']) & \
                   (np.abs(points_full_finite[:, 2]) <= config['z_threshold_upper_2'])

    combined_slice_mask = (slice_mask_1 | slice_mask_2) & \
                          (np.abs(points_full_finite[:, 0]) <= max_range) & \
                          (np.abs(points_full_finite[:, 1]) <= max_range)
    points_slice = points_full_finite[combined_slice_mask]

    if points_slice.shape[0] == 0:
        return np.full(config['num_ranges'], max_range, dtype=np.float32)

    ranges = np.full(config['num_ranges'], max_range, dtype=np.float32)
    point_angles = np.arctan2(points_slice[:, 1], points_slice[:, 0])
    distances = np.linalg.norm(points_slice[:, :2], axis=1)

    for i in range(config['num_ranges']):
        angle_rad = np.deg2rad(i * config['angle_increment_deg'])
        angular_tolerance = np.deg2rad(config['angle_increment_deg'] / 2)
        angular_diff = np.arctan2(np.sin(point_angles - angle_rad), np.cos(point_angles - angle_rad))
        sector_mask = np.abs(angular_diff) <= angular_tolerance
        sector_distances = distances[sector_mask]
        
        if len(sector_distances) > 0:
            min_dist = np.min(sector_distances)
            ranges[i] = min(min_dist, max_range)
    
    return ranges
