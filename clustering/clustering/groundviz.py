import open3d as o3d
import numpy as np

pcd_path = "/media/akilasar/dcist_rrg/akila_data_hockfield/recorded_data/recorded_data_pcds/1776804786-556439104.pcd"

# 1776804740-856474635.pcd"


config = {
    'z_ground_lower':             0.83,  # ground band (intensity_edge / combined row 1)
    'z_ground_upper':             1.10,
    'structure_z_lower':          0.85,   # structure band (combined row 0 — geometry)
    'structure_z_upper':          1.2,
    'min_lidar_range':            1.0,
    'max_lidar_range':            25.0,
    'num_ranges':                 72,
    'max_intensity':              255.0,
    'intensity_change_threshold': 0.10,
}

# --- load XYZI ---
pcd_t = o3d.t.io.read_point_cloud(pcd_path)
xyz = pcd_t.point['positions'].numpy()
if 'intensity' in pcd_t.point:
    intensity = pcd_t.point['intensity'].numpy().reshape(-1, 1)
    pts = np.hstack([xyz, intensity]).astype(np.float32)
else:
    pts = xyz.astype(np.float32)
    print("WARNING: no intensity channel in this PCD — transition visualization unavailable")

has_intensity = pts.shape[1] >= 4

# --- coloring: start dark grey, paint bands, then paint intensity classification last ---
all_colors = np.ones((len(pts), 3)) * 0.15  # dark grey

z = pts[:, 2]

# Structure z-band: cyan (geometry band for combined row 0)
struct_mask = (np.abs(z) >= config['structure_z_lower']) & (np.abs(z) <= config['structure_z_upper'])
all_colors[struct_mask] = [0.2, 0.8, 0.9]

# Ground z-band + radial filter, tracking original indices (painted over structure if overlap)
gnd_mask = (np.abs(z) >= config['z_ground_lower']) & (np.abs(z) <= config['z_ground_upper'])
gnd_pts_full = pts[gnd_mask]

dists_full = np.linalg.norm(gnd_pts_full[:, :2], axis=1)
rad_mask = (dists_full >= config['min_lidar_range']) & (dists_full <= config['max_lidar_range'])
gnd_pts = gnd_pts_full[rad_mask]
dists = dists_full[rad_mask]
original_idx = np.where(gnd_mask)[0][rad_mask]

# --- intensity classification on ground band ---
n = config['num_ranges']
inc = 360.0 / n
max_i = config['max_intensity']
threshold = config['intensity_change_threshold']
angles = np.arctan2(gnd_pts[:, 1], gnd_pts[:, 0])
tol = np.deg2rad(inc / 2)

transition_hits = []

if has_intensity and gnd_pts.size > 0:
    # Pass 1: reference intensity via histogram mode
    closest_intensities = []
    for i in range(n):
        angle_rad = np.deg2rad(i * inc)
        diff = np.arctan2(np.sin(angles - angle_rad), np.cos(angles - angle_rad))
        mask = np.abs(diff) <= tol
        if mask.any():
            ci = np.argmin(dists[mask])
            closest_intensities.append(gnd_pts[mask][ci, 3] / max_i)

    if closest_intensities:
        arr = np.array(closest_intensities)
        counts, edges = np.histogram(arr, bins=10, range=(0.0, 1.0))
        mode_bin = np.argmax(counts)
        reference = float((edges[mode_bin] + edges[mode_bin + 1]) / 2.0)
        print(f"Reference intensity: {reference:.3f}  (threshold ±{threshold})")

        # Green = reference terrain, red = intensity transition
        i_norm_all = gnd_pts[:, 3] / max_i
        is_transition = np.abs(i_norm_all - reference) > threshold
        all_colors[original_idx[~is_transition]] = [0.2, 0.8, 0.2]  # green
        all_colors[original_idx[is_transition]]  = [0.9, 0.2, 0.1]  # red

        # Pass 2: per-sector walk outward to first transition hit
        for i in range(n):
            angle_rad = np.deg2rad(i * inc)
            diff = np.arctan2(np.sin(angles - angle_rad), np.cos(angles - angle_rad))
            mask = np.abs(diff) <= tol
            if mask.any():
                sector_dists = dists[mask]
                sector_pts  = gnd_pts[mask]
                order = np.argsort(sector_dists)
                for idx in order:
                    if abs(sector_pts[idx, 3] / max_i - reference) > threshold:
                        transition_hits.append(sector_pts[idx, :3])
                        break
else:
    all_colors[original_idx] = [0.2, 0.8, 0.2]

print(f"Transition hits found: {len(transition_hits)} / {n} sectors")
print(f"Structure band points: {struct_mask.sum()}")

# --- row 0 (combined): closest structure-band hit per sector ---
s_pts_full = pts[struct_mask]
s_dists_full = np.linalg.norm(s_pts_full[:, :2], axis=1)
s_rad = (s_dists_full >= config['min_lidar_range']) & (s_dists_full <= config['max_lidar_range'])
s_pts = s_pts_full[s_rad]
s_dists = s_dists_full[s_rad]
s_angles = np.arctan2(s_pts[:, 1], s_pts[:, 0]) if s_pts.size > 0 else np.array([])

structure_range_hits = []
for i in range(n):
    angle_rad = np.deg2rad(i * inc)
    diff = np.arctan2(np.sin(s_angles - angle_rad), np.cos(s_angles - angle_rad))
    mask = np.abs(diff) <= tol
    if mask.any():
        ci = np.argmin(s_dists[mask])
        structure_range_hits.append(s_pts[mask][ci, :3])

print(f"Structure range hits (row 0): {len(structure_range_hits)} / {n} sectors")

# --- build geometries ---
pcd_vis = o3d.geometry.PointCloud()
pcd_vis.points = o3d.utility.Vector3dVector(pts[:, :3])
pcd_vis.colors = o3d.utility.Vector3dVector(all_colors)

geometries = [pcd_vis]

origin = np.zeros(3)

# Blue rays to closest structure-band hit per sector (combined row 0 — geometry)
# Color gradient by hit z: deep blue (low z) -> cyan (high z)
if structure_range_hits:
    hits_arr = np.array(structure_range_hits)
    z_vals = hits_arr[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    t = (z_vals - z_min) / (z_max - z_min) if z_max > z_min else np.zeros(len(z_vals))
    low_c  = np.array([0.0, 0.2, 0.9])  # deep blue
    high_c = np.array([0.0, 0.9, 1.0])  # cyan
    ray_colors = np.outer(1 - t, low_c) + np.outer(t, high_c)

    s_ray_pts, s_ray_idx = [], []
    for hit in structure_range_hits:
        p = len(s_ray_pts)
        s_ray_pts.append(origin)
        s_ray_pts.append(hit)
        s_ray_idx.append([p, p + 1])
    s_rays = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(s_ray_pts),
        lines=o3d.utility.Vector2iVector(s_ray_idx),
    )
    s_rays.colors = o3d.utility.Vector3dVector(ray_colors)
    geometries.append(s_rays)

# Yellow rays + orange markers to intensity transition hits (combined row 1 — terrain boundary)
if transition_hits:
    ray_pts, ray_idx = [], []
    for hit in transition_hits:
        p = len(ray_pts)
        ray_pts.append(origin)
        ray_pts.append(hit)
        ray_idx.append([p, p + 1])
    rays = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ray_pts),
        lines=o3d.utility.Vector2iVector(ray_idx),
    )
    rays.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
    geometries.append(rays)

    hits_pcd = o3d.geometry.PointCloud()
    hits_pcd.points = o3d.utility.Vector3dVector(np.array(transition_hits))
    hits_pcd.paint_uniform_color([1.0, 0.5, 0.0])  # orange
    geometries.append(hits_pcd)

vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="cyan=structure band  |  blue rays=row0 geometry  |  green=reference terrain  |  red=transition  |  yellow rays + orange=row1 intensity_edge"
)
for g in geometries:
    vis.add_geometry(g)
vis.get_render_option().line_width = 4.0
vis.run()
vis.destroy_window()
