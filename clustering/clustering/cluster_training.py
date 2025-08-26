# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os
# import glob
# import open3d as o3d
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA # Import PCA for 2D visualization
# from sklearn.manifold import TSNE
# import hdbscan
# import pickle # For saving/loading HDBSCAN model
# from PIL import Image, ImageDraw, ImageFont # For GIF text overlay
# import time # For pausing between visualizations
# import json # For saving/loading cluster labels
# import imageio # Make sure imageio is imported for GIF saving
# from sklearn import metrics
# from lidar_processor import get_ranges_from_points

# class LidarEncoder(nn.Module):
#     def __init__(self, embedding_size):
#         super(LidarEncoder, self).__init__()
#         self.embedding_size = embedding_size
#         self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
#         self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         self.conv2d_3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
#         self.pool = nn.AdaptiveMaxPool2d((1,1))
#         self.fc = nn.Linear(64, embedding_size)

#     def forward(self, x):
#         x = x.unsqueeze(1).unsqueeze(1)
#         x = F.relu(self.conv2d_1(x))
#         x = F.relu(self.conv2d_2(x))
#         x = F.relu(self.conv2d_3(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         embedding = self.fc(x)
#         return embedding

# class LidarDecoder(nn.Module):
#     def __init__(self, embedding_size, num_ranges):
#         super(LidarDecoder, self).__init__()
#         self.num_ranges = num_ranges
#         self.decoder_fc1 = nn.Linear(embedding_size, 64)
#         self.decoder_fc2 = nn.Linear(64, 128)
#         self.decoder_fc3 = nn.Linear(128, self.num_ranges)

#     def forward(self, x):
#         x = F.relu(self.decoder_fc1(x))
#         x = F.relu(self.decoder_fc2(x))
#         x = self.decoder_fc3(x) 
#         return x

# class Autoencoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Autoencoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# class LidarDataset(Dataset):
#     def __init__(self, pcd_folder, config):
#         self.pcd_files = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))
#         self.config = config

#     def __len__(self):
#         return len(self.pcd_files)

#     def __getitem__(self, idx):
#         pcd_filepath = self.pcd_files[idx]
#         try:
#             pcd_full = o3d.io.read_point_cloud(pcd_filepath)
#             points = np.asarray(pcd_full.points)
#         except Exception as e:
#             print(f"Error reading PCD file {pcd_filepath}: {e}. Returning zeros.")
#             return torch.zeros(self.config['num_ranges'], dtype=torch.float32)

#         # Use the shared preprocessing function
#         ranges = get_ranges_from_points(points, self.config)
        
#         # Normalize the ranges
#         normalized_ranges = ranges / self.config['max_lidar_range']
        
#         return torch.tensor(normalized_ranges, dtype=torch.float32)

# def visualize_footprint_at_timestep(pcd_file, z_threshold_upper, z_threshold_lower, z_threshold_upper_2, z_threshold_lower_2, max_range, angle_increment_deg, show_footprint_lines=True):
#     try:
#         pcd_full = o3d.io.read_point_cloud(pcd_file)
#     except Exception as e:
#         print(f"Error reading PCD file: {e}")
#         return None, None, None, None

#     points_full = np.asarray(pcd_full.points)
#     finite_mask_full = np.isfinite(points_full).all(axis=1)
#     points_full_finite = points_full[finite_mask_full]

#     slice_mask_1 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower) & \
#                    (np.abs(points_full_finite[:, 2]) <= z_threshold_upper)
    
#     slice_mask_2 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower_2) & \
#                    (np.abs(points_full_finite[:, 2]) <= z_threshold_upper_2)

#     combined_slice_mask = (slice_mask_1 | slice_mask_2) & \
#                           (np.abs(points_full_finite[:, 0]) <= max_range) & \
#                           (np.abs(points_full_finite[:, 1]) <= max_range)
#     points_slice = points_full_finite[combined_slice_mask]

#     pcd_all_o3d = o3d.geometry.PointCloud()
#     pcd_all_o3d.points = o3d.utility.Vector3dVector(points_full_finite)
#     pcd_all_o3d.paint_uniform_color([0.5, 0.5, 0.5]) 
    
#     pcd_footprint_slice = o3d.geometry.PointCloud()
#     pcd_footprint_slice.points = o3d.utility.Vector3dVector(points_slice)

#     line_set = None
#     ranges = [] # Initialize ranges list

#     num_angles = int(360 / angle_increment_deg)
#     footprint_points = [] # Only used if show_footprint_lines is True
    
#     for i in range(num_angles):
#         angle_deg = i * angle_increment_deg
#         angle_rad = np.deg2rad(angle_deg)
#         min_dist = max_range

#         angular_tolerance = np.deg2rad(angle_increment_deg / 2)
#         point_angles = np.arctan2(points_slice[:, 1], points_slice[:, 0])
#         angular_diff = np.arctan2(np.sin(point_angles - angle_rad), np.cos(point_angles - angle_rad))
#         sector_mask = np.abs(angular_diff) <= angular_tolerance
#         sector_points = points_slice[sector_mask]

#         closest_point = None
#         if sector_points.shape[0] > 0:
#             distances = np.linalg.norm(sector_points[:, :2], axis=1)
#             closest_index = np.argmin(distances)
#             min_dist = np.min(distances)
#             closest_point = sector_points[closest_index, :2]
#             if min_dist > max_range: 
#                 closest_point = np.array([max_range * np.cos(angle_rad), max_range * np.sin(angle_rad)])
#                 min_dist = max_range
#         else:
#             closest_point = np.array([max_range * np.cos(angle_rad), max_range * np.sin(angle_rad)])

#         ranges.append(min_dist) # Populate ranges list
        
#         if show_footprint_lines:
#             footprint_points.append([0, 0, 0])
#             footprint_points.append([closest_point[0], closest_point[1], 0])

#     if show_footprint_lines:
#         footprint_lines = [[i, i + 1] for i in range(0, len(footprint_points), 2)]
#         line_set = o3d.geometry.LineSet(
#             points=o3d.utility.Vector3dVector(footprint_points),
#             lines=o3d.utility.Vector2iVector(footprint_lines),
#         )
#         line_set.paint_uniform_color([1, 0, 0])  # Red for the footprint lines

#     return pcd_all_o3d, line_set, pcd_footprint_slice, ranges

# def display_cluster_samples(pcd_folder, filenames, cluster_id, num_samples, config):
#     if not filenames:
#         print(f"  No samples to display for Cluster {cluster_id}.")
#         return

#     sample_indices = np.random.choice(len(filenames), min(num_samples, len(filenames)), replace=False)
    
#     print(f"\n--- Displaying {len(sample_indices)} samples for Cluster {cluster_id} ---")
#     print("Close the Open3D window to view the next sample or cluster.")
#     print("Press Ctrl+C in terminal to stop viewing samples and proceed.")

#     for i, idx in enumerate(sample_indices):
#         filepath = os.path.join(pcd_folder, filenames[idx])
#         pcd_all_o3d, line_set, _, _ = visualize_footprint_at_timestep(
#             filepath,
#             config['z_threshold_upper'],  # Use new config keys
#             config['z_threshold_lower'],  # Use new config keys
#             config['z_threshold_upper_2'], # Use new config keys
#             config['z_threshold_lower_2'], # Use new config keys
#             config['max_lidar_range'],
#             config['angle_increment_deg'],
#             show_footprint_lines=True # Always show footprint lines for manual inspection
#         )
#         if pcd_all_o3d:
#             # Color the whole point cloud (not just footprint) for clarity
#             pcd_all_o3d.paint_uniform_color([0.1, 0.7, 0.1]) # Greenish for clarity
            
#             vis = o3d.visualization.Visualizer()
#             vis.create_window(window_name=f"Cluster {cluster_id} Sample {i+1}/{len(sample_indices)}: {os.path.basename(filepath)}")
#             render_option = vis.get_render_option()
#             render_option.background_color = np.asarray([0, 0, 0])
#             vis.add_geometry(pcd_all_o3d)
#             if line_set:
#                 vis.add_geometry(line_set)
#             vis.run() # This blocks until the window is closed
#             vis.destroy_window()
#             time.sleep(0.5) # Small pause between windows
#         else:
#             print(f"Could not load/process sample {filepath} for Cluster {cluster_id}.")

# def generate_cluster_gif(pcd_folder, pcd_filenames, cluster_labels, cluster_id_to_label, 
#                          output_path, gif_fps, config, color_map):
#     print(f"\n--- Rendering visualizations to GIF: {output_path} ---")
#     images = []
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     render_option = vis.get_render_option()
#     render_option.background_color = np.asarray([0, 0, 0])  # Black background
    
#     for i, filename_full_path in enumerate(pcd_filenames):
#         current_cluster_id = cluster_labels[i]
#         cluster_color = color_map.get(current_cluster_id, np.array([0.5, 0.5, 0.5, 1.0])) # Default grey
        
#         cluster_description = cluster_id_to_label.get(current_cluster_id, "Unknown/Unlabeled")
        
#         print(f"  Processing frame {i+1}/{len(pcd_filenames)}: {os.path.basename(filename_full_path)} (Cluster: {current_cluster_id}, Label: {cluster_description})")

#         vis.clear_geometries()
#         pcd_all_o3d, line_set, _, _ = visualize_footprint_at_timestep(
#             filename_full_path, # Pass the full path here
#             config['z_threshold_upper'],  # Use new config keys
#             config['z_threshold_lower'],  # Use new config keys
#             config['z_threshold_upper_2'], # Use new config keys
#             config['z_threshold_lower_2'], # Use new config keys
#             config['max_lidar_range'],
#             config['angle_increment_deg'],
#             show_footprint_lines=True
#         )
        
#         if pcd_all_o3d is not None and line_set is not None:
#             pcd_all_o3d.paint_uniform_color(cluster_color[:3])
            
#             vis.add_geometry(pcd_all_o3d)
#             vis.add_geometry(line_set)
            
#             vis.reset_view_point(True)
            
#             vis.poll_events()
#             vis.update_renderer()
#             image = vis.capture_screen_float_buffer(do_render=True)
#             if image is not None:
#                 image = Image.fromarray(np.uint8(np.asarray(image) * 255))
#                 draw = ImageDraw.Draw(image)
#                 try:
#                     font = ImageFont.truetype("arial.ttf", 40)
#                 except IOError:
#                     font = ImageFont.load_default()
                
#                 text_to_overlay = f"Timestep: {i}\nCluster: {current_cluster_id}\nLabel: {cluster_description}"
#                 draw.text((10, 10), text_to_overlay, (255, 255, 255), font=font)
#                 images.append(np.array(image))
#         else:
#             print(f"Skipping frame for {os.path.basename(filename_full_path)} due to visualization error.")
#             continue

#     if vis is not None:
#         vis.destroy_window()

#     if images:
#         try:
#             imageio.mimsave(output_path, images, fps=gif_fps)
#             print(f"GIF saved to: {output_path}")
#         except Exception as e:
#             print(f"Error saving GIF: {e}")
#     else:
#         print(f"No frames were generated for {output_path}.")

# # --- Main Script Execution ---
# if __name__ == "__main__":
#     # 1. Configuration Parameters
#     training_data_name = "bridge2_carla"
#     training_pcd_folder = f"../../../../carla_data/{training_data_name}/{training_data_name}_pcds"
    
#     # Define and centralize all hyperparameters and paths
#     global_config = {
#         "training_data_name": training_data_name,
#         "training_pcd_folder": training_pcd_folder,
#         "embedding_size": 16,
#         "num_ranges": 72,
#         "angle_increment_deg": float(360.0 / 32),
#         "max_lidar_range": 20.0,
#         "z_threshold_upper": 2.25,
#         "z_threshold_lower": 2,
#         "z_threshold_upper_2": 0,
#         "z_threshold_lower_2": 0,
#         "num_epochs": 100,
#         "batch_size": 32,
#         "learning_rate": 0.001,
#         "hdbscan_min_cluster_size": 8,
#         "hdbscan_cluster_selection_epsilon": 0.0,
#         "num_samples_to_show_per_cluster": 3
#     }
    
#     # Define all model and artifact paths based on the training data name
#     global_config["model_save_path"] = f"encoder_weights/{training_data_name}/lidar_encoder_autoencoder_{training_data_name}.pth"
#     global_config["hdbscan_model_path"] = f"encoder_weights/{training_data_name}/hdbscan_model_{training_data_name}.pkl"
#     global_config["scaler_path"] = f"encoder_weights/{training_data_name}/scaler_{training_data_name}.pkl"
#     global_config["cluster_centroids_path"] = f"encoder_weights/{training_data_name}/cluster_centroids_{training_data_name}.pkl"
#     global_config["train_cluster_map_path"] = f"encoder_weights/{training_data_name}/train_cluster_to_filepaths_{training_data_name}.txt"
#     global_config["cluster_labels_mapping_path"] = f"encoder_weights/{training_data_name}/cluster_id_to_label_{training_data_name}.json"
#     global_config["output_training_plot_path"] = f"results/{training_data_name}/training_data_clusters_plot.png"
#     global_config["output_training_gif_path"] = f"results/{training_data_name}/training_data_clustered_hdbscan.gif"
#     global_config["noise_distance_plot_path"] = f"encoder_weights/{training_data_name}/noise_distance_distribution.png"
#     global_config["gif_frames_per_second"] = 15
#     global_config["training_distance_plot_path"] = f"results/{training_data_name}/distance_training_plot.png"

#     # --- 2. Device Setup and Model Initialization ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     encoder = LidarEncoder(embedding_size=global_config['embedding_size'])
#     decoder = LidarDecoder(embedding_size=global_config['embedding_size'], num_ranges=global_config['num_ranges'])
#     autoencoder = Autoencoder(encoder, decoder).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(autoencoder.parameters(), lr=global_config['learning_rate'])
    
#     # --- 3. Training the Autoencoder ---
#     if os.path.exists(global_config['model_save_path']):
#         print(f"Existing Autoencoder model found at {global_config['model_save_path']}. Skipping training.")
#         autoencoder.load_state_dict(torch.load(global_config['model_save_path'], map_location=device))
#     else:
#         print("No existing Autoencoder model found. Starting training from scratch.")
#         train_dataset = LidarDataset(global_config['training_pcd_folder'], global_config)
#         train_dataloader = DataLoader(train_dataset, batch_size=global_config['batch_size'], shuffle=True, num_workers=4)
        
#         print("Starting Autoencoder Training...")
#         for epoch in range(global_config['num_epochs']):
#             autoencoder.train()
#             total_loss = 0
#             for data in train_dataloader:
#                 data = data.to(device)
#                 reconstructions = autoencoder(data)
#                 loss = criterion(reconstructions, data)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             avg_loss = total_loss / len(train_dataloader)
#             print(f"Epoch {epoch+1}/{global_config['num_epochs']}, Average Loss: {avg_loss:.4f}")
        
#         os.makedirs(os.path.dirname(global_config['model_save_path']), exist_ok=True)
#         torch.save(autoencoder.state_dict(), global_config['model_save_path'])
#         print(f"Trained Autoencoder saved to {global_config['model_save_path']}")
        
#     autoencoder.eval()

#     # --- 4. Generate Embeddings for Clustering ---
#     print("\n--- Generating embeddings for training data and performing clustering ---")
#     dataset_for_clustering = LidarDataset(global_config['training_pcd_folder'], global_config)
#     dataloader_for_clustering = DataLoader(dataset_for_clustering, batch_size=global_config['batch_size'], shuffle=False, num_workers=4)

#     all_train_embeddings = []
#     train_pcd_filenames_full = dataset_for_clustering.pcd_files
    
#     with torch.no_grad():
#         for data in dataloader_for_clustering:
#             data = data.to(device)
#             embeddings = autoencoder.encoder(data)
#             all_train_embeddings.append(embeddings.cpu().numpy())
    
#     all_train_embeddings_np = np.vstack(all_train_embeddings)
#     print(f"Generated {all_train_embeddings_np.shape[0]} embeddings from training data.")
    
#     # --- 5. Scale Embeddings and Perform HDBSCAN ---
#     scaler = StandardScaler()
#     scaled_train_embeddings = scaler.fit_transform(all_train_embeddings_np)
#     print("Scaled training embeddings.")
    
#     os.makedirs(os.path.dirname(global_config['scaler_path']), exist_ok=True)
#     with open(global_config['scaler_path'], 'wb') as f:
#         pickle.dump(scaler, f)
#     print(f"StandardScaler model saved to {global_config['scaler_path']}")

#     cluster_model = hdbscan.HDBSCAN(
#         min_cluster_size=global_config['hdbscan_min_cluster_size'], 
#         cluster_selection_epsilon=global_config['hdbscan_cluster_selection_epsilon'],
#         prediction_data=True,
#         core_dist_n_jobs=-1
#     )
#     train_cluster_labels = cluster_model.fit_predict(scaled_train_embeddings)
    
#     os.makedirs(os.path.dirname(global_config['hdbscan_model_path']), exist_ok=True)
#     with open(global_config['hdbscan_model_path'], 'wb') as f:
#         pickle.dump(cluster_model, f)
#     print(f"HDBSCAN model saved to {global_config['hdbscan_model_path']}")
    
#     unique_train_clusters = np.unique(train_cluster_labels)
#     print(f"Found {len(unique_train_clusters)} clusters in training data: {unique_train_clusters}")
    
#     # --- 6. Calculate and Save Cluster Centroids ---
#     print("\n--- Calculating and saving cluster centroids ---")
#     cluster_centroids = {}
#     for cluster_id in unique_train_clusters:
#         if cluster_id != -1:
#             cluster_points = scaled_train_embeddings[train_cluster_labels == cluster_id]
#             if len(cluster_points) > 0:
#                 cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)
    
#     os.makedirs(os.path.dirname(global_config['cluster_centroids_path']), exist_ok=True)
#     with open(global_config['cluster_centroids_path'], 'wb') as f:
#         pickle.dump(cluster_centroids, f)
#     print(f"Cluster centroids saved to {global_config['cluster_centroids_path']}")
    
#     # --- 7. Analyze Noise Points and Visualize Distances ---
#     print("\n--- Analyzing distances of noise points to known clusters ---")
#     noise_points_embeddings = scaled_train_embeddings[train_cluster_labels == -1]
    
#     if len(noise_points_embeddings) > 0 and cluster_centroids:
#         min_distances = [min([np.linalg.norm(noise_point - centroid) for centroid in cluster_centroids.values()]) for noise_point in noise_points_embeddings]
#         min_distances_np = np.array(min_distances)
        
#         plt.figure(figsize=(10, 6))
#         plt.hist(min_distances_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
#         plt.title('Distribution of Distances from Noise Points to Nearest Cluster Centroid')
#         plt.xlabel('Distance')
#         plt.ylabel('Frequency')
#         plt.grid(True, linestyle='--', alpha=0.6)
        
#         p95 = np.percentile(min_distances_np, 95)
#         plt.axvline(p95, color='green', linestyle='--', linewidth=1, label=f'95th Percentile: {p95:.2f}')
#         plt.legend()
        
#         os.makedirs(os.path.dirname(global_config['noise_distance_plot_path']), exist_ok=True)
#         plt.savefig(global_config['noise_distance_plot_path'])
#         print(f"Noise distance distribution plot saved to {global_config['noise_distance_plot_path']}")
#         print(f"Suggested reassignment threshold (95th percentile): {p95:.2f}")

#     # --- 8. Create Cluster-to-File Mapping and GIF ---
#     print("\n--- Generating cluster-to-filepaths mapping for training data ---")
#     cluster_to_train_filepaths = {cluster_id: [] for cluster_id in unique_train_clusters}
#     for i, label in enumerate(train_cluster_labels):
#         cluster_to_train_filepaths[label].append(os.path.basename(train_pcd_filenames_full[i]))

#     os.makedirs(os.path.dirname(global_config['train_cluster_map_path']), exist_ok=True)
#     with open(global_config['train_cluster_map_path'], 'w') as f:
#         for cluster_id in sorted(cluster_to_train_filepaths.keys()):
#             filenames_in_cluster = cluster_to_train_filepaths[cluster_id]
#             f.write(f"Cluster {cluster_id}: {', '.join(filenames_in_cluster)}\n")
#     print(f"Training cluster mapping saved to: {global_config['train_cluster_map_path']}")
    
#     labels_file_exists = os.path.exists(global_config['cluster_labels_mapping_path'])
#     cluster_id_to_label = {}
#     if labels_file_exists and os.stat(global_config['cluster_labels_mapping_path']).st_size > 2:
#         with open(global_config['cluster_labels_mapping_path'], 'r') as f:
#             cluster_id_to_label = {int(k): v for k, v in json.load(f).items()}
#         print(f"Loaded existing cluster labels from {global_config['cluster_labels_mapping_path']}")
#     else:
#         print(f"\nNo existing cluster labels found. Please review the output plot and create the file '{global_config['cluster_labels_mapping_path']}' with your labels.")
    
#         for cluster_id in sorted(unique_train_clusters):
#             if cluster_id != -1: # Don't display noise clusters for manual labeling
#                 print(f"\nViewing samples for Cluster: {cluster_id}")
#                 cluster_filenames = cluster_to_train_filepaths[cluster_id]
#                 display_cluster_samples(training_pcd_folder, cluster_filenames, cluster_id, 
#                                         3, global_config)
#             else:
#                 print(f"\nSkipping display for Noise Cluster (-1).")
        
#         print("\n" + "="*80)
#         print("END OF MANUAL LABELING STEP.")
#         print(f"Please create/edit the file '{global_config['cluster_labels_mapping_path']}' with your labels.")
#         print("Then, run the script again.")
#         print("="*80)
#         exit() 
#     # else: # If JSON exists, check for missing labels
#     #     missing_labels = [c for c in unique_train_clusters if c >= 0 and c not in cluster_id_to_label]
#     #     if missing_labels:
#     #         print(f"\nWARNING: Missing labels for clusters {missing_labels} in {global_config['cluster_labels_mapping_path']}.")
#     #         print("Please update your JSON file with labels for these clusters.")
#     #         print("You may proceed, but these clusters will not have descriptive labels in the GIF.")
#     #         proceed = input("Continue anyway? (y/n): ")
#     #         if proceed.lower() != 'y':
#     #             print("Exiting to allow label update.")
#     #             exit()
#     #     else:
#     #         print("\nAll training clusters have labels. Proceeding with inference and visualization.")

#     color_map_global = {}
#     color_map_global[-1] = np.array([0.2, 0.2, 0.2, 1.0]) # Dark Grey for HDBSCAN Noise
#     color_map_global[-2] = np.array([1.0, 0.5, 0.0, 1.0]) # Orange for New Cluster / Outlier

#     actual_clusters = sorted([c for c in unique_train_clusters if c >= 0])
#     if len(actual_clusters) > 0:
#         cluster_colors_cmap = cm.get_cmap('rainbow', len(actual_clusters))
#         for i, cluster_id in enumerate(actual_clusters):
#             color_map_global[cluster_id] = cluster_colors_cmap(i / (len(actual_clusters) - 1)) if len(actual_clusters) > 1 else cluster_colors_cmap(0.5)
    
#     print("\n--- Generating 2D t-SNE plot for training data clusters ---")
#     fig_train_plot, ax_train_plot = plt.subplots(figsize=(12,8))

#     #if embedding_size > 2:
#     if scaled_train_embeddings.shape[0] > 2:
#         tsne = TSNE(n_components=2, perplexity=min(30, scaled_train_embeddings.shape[0]-1), learning_rate='auto', init='random', random_state=42)
#         train_embeddings_2d = tsne.fit_transform(scaled_train_embeddings)
#         print("Reduced training embeddings to 2D using t-SNE.")
#     else:
#         print("Not enough samples for t-SNE. Using original embeddings if 2D or less, else will error for plotting.")
#         train_embeddings_2d = scaled_train_embeddings[:, :2] if scaled_train_embeddings.shape[1] >=2 else scaled_train_embeddings
#     # else:
#     #     train_embeddings_2d = scaled_train_embeddings
#     #     print("Training embeddings already 2D or less, no t-SNE applied.")

#     # Plot each cluster
#     for cluster_id in sorted(unique_train_clusters):
#         mask = (train_cluster_labels == cluster_id)
#         if np.any(mask): # Ensure there are points for this cluster
#             label_text = f'Cluster {cluster_id}'
#             if cluster_id in cluster_id_to_label:
#                 label_text += f' ({cluster_id_to_label[cluster_id]})'
#             elif cluster_id == -1:
#                 label_text = 'Noise/Uncertain (HDBSCAN)'

#             ax_train_plot.scatter(train_embeddings_2d[mask, 0], train_embeddings_2d[mask, 1],
#                                   color=color_map_global.get(cluster_id, np.array([0.5, 0.5, 0.5, 1.0])),
#                                   label=label_text,
#                                   s=20, alpha=0.7)

#     ax_train_plot.set_xlabel("Component 1")
#     ax_train_plot.set_ylabel("Component 2")
#     ax_train_plot.set_title("Training Data Clusters (2D t-SNE of Embeddings)")
#     ax_train_plot.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax_train_plot.grid(True)
#     plt.tight_layout()

#     os.makedirs(os.path.dirname(global_config["output_training_plot_path"]), exist_ok=True)
#     plt.savefig(global_config["output_training_plot_path"])
#     print(f"Training data clusters plot saved to: {global_config['output_training_plot_path']}")
#     plt.close(fig_train_plot) # Close the plot to free memory
    
#      # --- Plot Distances to Centroids for Training Data ---
#     print("\n--- Generating distance plot for training data ---")
#     distance_threshold_for_new_clusters = 5

#     # Create a figure with two subplots, one above the other
#     fig, (ax_dist_plot_train, ax_time_series_train) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

#     # Get unique cluster labels from the training data
#     unique_final_training_clusters = np.unique(train_cluster_labels)

#     # --- First Subplot: Distance Plot for Training Data ---
#     # Plot each cluster
#     for cluster_id in sorted(unique_final_training_clusters):
#         mask = (train_cluster_labels == cluster_id)
#         if np.any(mask): # Ensure there are points for this cluster
#             label_text = f'Cluster {cluster_id}'
#             if cluster_id in cluster_id_to_label:
#                 label_text += f' ({cluster_id_to_label[cluster_id]})'
#             elif cluster_id == -1:
#                 label_text = 'HDBSCAN Noise' 
#             elif cluster_id == -2:
#                 label_text = 'New Cluster / Outlier' 
#             else:
#                 label_text = f'Unknown Cluster {cluster_id}'

#             # Only plot valid distances
#             min_distances_to_known_centroids_training = np.full(len(train_cluster_labels), np.nan)

#             valid_timesteps = np.arange(len(train_pcd_filenames_full))[mask & ~np.isnan(min_distances_to_known_centroids_training)]
#             valid_distances = min_distances_to_known_centroids_training[mask & ~np.isnan(min_distances_to_known_centroids_training)]

#             if len(valid_timesteps) > 0:
#                 ax_dist_plot_train.scatter(valid_timesteps, valid_distances,
#                                     color=color_map_global.get(cluster_id, np.array([0.5, 0.5, 0.5, 1.0])),
#                                     label=label_text,
#                                     s=30, alpha=0.7)

#     ax_dist_plot_train.axhline(y=distance_threshold_for_new_clusters, color='r', linestyle='--', label=f'New Cluster Threshold ({distance_threshold_for_new_clusters:.2f})')
#     ax_dist_plot_train.set_xlabel("Training Data Sample Index")
#     ax_dist_plot_train.set_ylabel("Minimum Distance to Closest Known Cluster Centroid")
#     ax_dist_plot_train.set_title("Training Data: Minimum Distance to Known Cluster Centroids")
#     ax_dist_plot_train.legend(title="Assigned Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax_dist_plot_train.grid(True)


#     # --- Second Subplot: All Points in Time by Cluster ID for Training Data ---
#     ax_time_series_train.set_title("All Training Data Points by Cluster ID Over Time")
#     ax_time_series_train.set_xlabel("Training Data Sample Index")
#     # Iterate through each point and plot it individually to color it by cluster_id
#     for i, cluster_id in enumerate(train_cluster_labels):
#         color = color_map_global.get(cluster_id, np.array([0.5, 0.5, 0.5, 0.3])) # Default to grey if cluster_id not in map
#         ax_time_series_train.plot(i, 0, marker='o', markersize=5, color=color, linestyle='')

#     ax_time_series_train.set_yticks([]) # Remove Y-axis ticks and labels as requested
#     ax_time_series_train.set_ylim([-0.5, 0.5]) # Set a small y-limit to make points visible along a line
#     ax_time_series_train.grid(True, axis='x') # Only show x-axis grid

#     plt.tight_layout() # Adjust layout to prevent overlapping
#     # Ensure the directory exists before saving
#     os.makedirs(os.path.dirname(global_config["training_distance_plot_path"]), exist_ok=True)
#     plt.savefig(global_config["training_distance_plot_path"])
#     print(f"Training data clusters plot saved to: {global_config['training_distance_plot_path']}")
#     plt.close(fig)


#     print("\n--- Generating distance plot for training data ---")
#     fig, (ax_dist_plot_train, ax_time_series_train) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
#     unique_final_training_clusters = np.unique(train_cluster_labels)

#     color_map = {cluster_id: cm.jet(i / len(unique_train_clusters)) for i, cluster_id in enumerate(sorted(unique_train_clusters))}
#     color_map[-1] = np.array([0.2, 0.2, 0.2, 1.0]) # Noise color

#     os.makedirs(os.path.dirname(global_config['output_training_gif_path']), exist_ok=True)
#     generate_cluster_gif(
#         global_config['training_pcd_folder'],
#         train_pcd_filenames_full,
#         train_cluster_labels,
#         cluster_id_to_label,
#         global_config['output_training_gif_path'],
#         global_config['gif_frames_per_second'],
#         global_config,
#         color_map
#     )
#     print("Training process complete.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import open3d as o3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Import PCA for 2D visualization
from sklearn.manifold import TSNE
import hdbscan
import pickle # For saving/loading HDBSCAN model
from PIL import Image, ImageDraw, ImageFont # For GIF text overlay
import time # For pausing between visualizations
import json # For saving/loading cluster labels
import imageio # Make sure imageio is imported for GIF saving
from sklearn import metrics
from lidar_processor import get_ranges_from_points

class LidarEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(LidarEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2d_3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(64, embedding_size)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class LidarDecoder(nn.Module):
    def __init__(self, embedding_size, num_ranges):
        super(LidarDecoder, self).__init__()
        self.num_ranges = num_ranges
        self.decoder_fc1 = nn.Linear(embedding_size, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, self.num_ranges)

    def forward(self, x):
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        x = self.decoder_fc3(x) 
        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LidarDataset(Dataset):
    def __init__(self, pcd_folder, config):
        self.pcd_files = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))
        self.config = config

    def __len__(self):
        return len(self.pcd_files)

    def __getitem__(self, idx):
        pcd_filepath = self.pcd_files[idx]
        try:
            pcd_full = o3d.io.read_point_cloud(pcd_filepath)
            points = np.asarray(pcd_full.points)
        except Exception as e:
            print(f"Error reading PCD file {pcd_filepath}: {e}. Returning zeros.")
            return torch.zeros(self.config['num_ranges'], dtype=torch.float32)

        # Use the shared preprocessing function
        ranges = get_ranges_from_points(points, self.config)
        
        # Normalize the ranges
        normalized_ranges = ranges / self.config['max_lidar_range']
        
        return torch.tensor(normalized_ranges, dtype=torch.float32)

def visualize_footprint_at_timestep(pcd_file, z_threshold_upper, z_threshold_lower, z_threshold_upper_2, z_threshold_lower_2, max_range, angle_increment_deg, show_footprint_lines=True):
    try:
        pcd_full = o3d.io.read_point_cloud(pcd_file)
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return None, None, None, None

    points_full = np.asarray(pcd_full.points)
    finite_mask_full = np.isfinite(points_full).all(axis=1)
    points_full_finite = points_full[finite_mask_full]

    slice_mask_1 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower) & \
                   (np.abs(points_full_finite[:, 2]) <= z_threshold_upper)
    
    slice_mask_2 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower_2) & \
                   (np.abs(points_full_finite[:, 2]) <= z_threshold_upper_2)

    combined_slice_mask = (slice_mask_1 | slice_mask_2) & \
                          (np.abs(points_full_finite[:, 0]) <= max_range) & \
                          (np.abs(points_full_finite[:, 1]) <= max_range)
    points_slice = points_full_finite[combined_slice_mask]

    pcd_all_o3d = o3d.geometry.PointCloud()
    pcd_all_o3d.points = o3d.utility.Vector3dVector(points_full_finite)
    pcd_all_o3d.paint_uniform_color([0.5, 0.5, 0.5]) 
    
    pcd_footprint_slice = o3d.geometry.PointCloud()
    pcd_footprint_slice.points = o3d.utility.Vector3dVector(points_slice)

    line_set = None
    ranges = [] # Initialize ranges list

    num_angles = int(360 / angle_increment_deg)
    footprint_points = [] # Only used if show_footprint_lines is True
    
    for i in range(num_angles):
        angle_deg = i * angle_increment_deg
        angle_rad = np.deg2rad(angle_deg)
        min_dist = max_range

        angular_tolerance = np.deg2rad(angle_increment_deg / 2)
        point_angles = np.arctan2(points_slice[:, 1], points_slice[:, 0])
        angular_diff = np.arctan2(np.sin(point_angles - angle_rad), np.cos(point_angles - angle_rad))
        sector_mask = np.abs(angular_diff) <= angular_tolerance
        sector_points = points_slice[sector_mask]

        closest_point = None
        if sector_points.shape[0] > 0:
            distances = np.linalg.norm(sector_points[:, :2], axis=1)
            closest_index = np.argmin(distances)
            min_dist = np.min(distances)
            closest_point = sector_points[closest_index, :2]
            if min_dist > max_range: 
                closest_point = np.array([max_range * np.cos(angle_rad), max_range * np.sin(angle_rad)])
                min_dist = max_range
        else:
            closest_point = np.array([max_range * np.cos(angle_rad), max_range * np.sin(angle_rad)])

        ranges.append(min_dist) # Populate ranges list
        
        if show_footprint_lines:
            footprint_points.append([0, 0, 0])
            footprint_points.append([closest_point[0], closest_point[1], 0])

    if show_footprint_lines:
        footprint_lines = [[i, i + 1] for i in range(0, len(footprint_points), 2)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(footprint_points),
            lines=o3d.utility.Vector2iVector(footprint_lines),
        )
        line_set.paint_uniform_color([1, 0, 0])  # Red for the footprint lines

    return pcd_all_o3d, line_set, pcd_footprint_slice, ranges

def display_cluster_samples(pcd_folder, filenames, cluster_id, num_samples, config):
    if not filenames:
        print(f"  No samples to display for Cluster {cluster_id}.")
        return

    sample_indices = np.random.choice(len(filenames), min(num_samples, len(filenames)), replace=False)
    
    print(f"\n--- Displaying {len(sample_indices)} samples for Cluster {cluster_id} ---")
    print("Close the Open3D window to view the next sample or cluster.")
    print("Press Ctrl+C in terminal to stop viewing samples and proceed.")

    for i, idx in enumerate(sample_indices):
        filepath = os.path.join(pcd_folder, filenames[idx])
        pcd_all_o3d, line_set, _, _ = visualize_footprint_at_timestep(
            filepath,
            config['z_threshold_upper'],  # Use new config keys
            config['z_threshold_lower'],  # Use new config keys
            config['z_threshold_upper_2'], # Use new config keys
            config['z_threshold_lower_2'], # Use new config keys
            config['max_lidar_range'],
            config['angle_increment_deg'],
            show_footprint_lines=True # Always show footprint lines for manual inspection
        )
        if pcd_all_o3d:
            # Color the whole point cloud (not just footprint) for clarity
            pcd_all_o3d.paint_uniform_color([0.1, 0.7, 0.1]) # Greenish for clarity
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Cluster {cluster_id} Sample {i+1}/{len(sample_indices)}: {os.path.basename(filepath)}")
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0, 0, 0])
            vis.add_geometry(pcd_all_o3d)
            if line_set:
                vis.add_geometry(line_set)
            vis.run() # This blocks until the window is closed
            vis.destroy_window()
            time.sleep(0.5) # Small pause between windows
        else:
            print(f"Could not load/process sample {filepath} for Cluster {cluster_id}.")

def generate_cluster_gif(pcd_folder, pcd_filenames, cluster_labels, cluster_id_to_label, 
                         output_path, gif_fps, config, color_map, z_thresholds_gif=None):
    print(f"\n--- Rendering visualizations to GIF: {output_path} ---")
    images = []
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])  # Black background

    # Use a separate set of z_thresholds if provided for the GIF
    if z_thresholds_gif:
        z_upper_1, z_lower_1, z_upper_2, z_lower_2 = z_thresholds_gif
    else:
        z_upper_1 = config['z_threshold_upper']
        z_lower_1 = config['z_threshold_lower']
        z_upper_2 = config['z_threshold_upper_2']
        z_lower_2 = config['z_threshold_lower_2']
    
    for i, filename_full_path in enumerate(pcd_filenames):
        current_cluster_id = cluster_labels[i]
        cluster_color = color_map.get(current_cluster_id, np.array([0.5, 0.5, 0.5, 1.0])) # Default grey
        
        cluster_description = cluster_id_to_label.get(current_cluster_id, "Unknown/Unlabeled")
        
        print(f"  Processing frame {i+1}/{len(pcd_filenames)}: {os.path.basename(filename_full_path)} (Cluster: {current_cluster_id}, Label: {cluster_description})")

        vis.clear_geometries()
        pcd_all_o3d, line_set, _, _ = visualize_footprint_at_timestep(
            filename_full_path, # Pass the full path here
            z_upper_1,  
            z_lower_1,  
            z_upper_2, 
            z_lower_2, 
            config['max_lidar_range'],
            config['angle_increment_deg'],
            show_footprint_lines=True
        )
        
        if pcd_all_o3d is not None and line_set is not None:
            pcd_all_o3d.paint_uniform_color(cluster_color[:3])
            
            vis.add_geometry(pcd_all_o3d)
            vis.add_geometry(line_set)
            
            vis.reset_view_point(True)
            
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            if image is not None:
                image = Image.fromarray(np.uint8(np.asarray(image) * 255))
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.truetype("arial.ttf", 40)
                except IOError:
                    font = ImageFont.load_default()
                
                text_to_overlay = f"Timestep: {i}\nCluster: {current_cluster_id}\nLabel: {cluster_description}"
                draw.text((10, 10), text_to_overlay, (255, 255, 255), font=font)
                images.append(np.array(image))
        else:
            print(f"Skipping frame for {os.path.basename(filename_full_path)} due to visualization error.")
            continue

    if vis is not None:
        vis.destroy_window()

    if images:
        try:
            imageio.mimsave(output_path, images, fps=gif_fps)
            print(f"GIF saved to: {output_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print(f"No frames were generated for {output_path}.")

def run_autoencoder_and_cluster(data_folder_name, data_pcd_folder, global_config):
    """
    Main function to run the autoencoder training and clustering process.

    Args:
        data_folder_name (str): The name of the dataset (e.g., 'bridge2_carla').
        data_pcd_folder (str): The path to the folder containing PCD files.
        global_config (dict): The dictionary of all configuration parameters.
    """

    print(f"\n--- Starting processing for dataset: {data_folder_name} ---")

    # Define all model and artifact paths based on the training data name
    config = global_config.copy()
    config["model_save_path"] = f"encoder_weights/{data_folder_name}/lidar_encoder_autoencoder_{data_folder_name}.pth"
    config["hdbscan_model_path"] = f"encoder_weights/{data_folder_name}/hdbscan_model_{data_folder_name}.pkl"
    config["scaler_path"] = f"encoder_weights/{data_folder_name}/scaler_{data_folder_name}.pkl"
    config["cluster_centroids_path"] = f"encoder_weights/{data_folder_name}/cluster_centroids_{data_folder_name}.pkl"
    config["train_cluster_map_path"] = f"encoder_weights/{data_folder_name}/train_cluster_to_filepaths_{data_folder_name}.txt"
    config["cluster_labels_mapping_path"] = f"encoder_weights/{data_folder_name}/cluster_id_to_label_{data_folder_name}.json"
    config["output_training_plot_path"] = f"results/{data_folder_name}/training_data_clusters_plot.png"
    config["output_training_gif_path"] = f"results/{data_folder_name}/training_data_clustered_hdbscan.gif"
    config["noise_distance_plot_path"] = f"encoder_weights/{data_folder_name}/noise_distance_distribution.png"
    config["training_distance_plot_path"] = f"results/{data_folder_name}/distance_training_plot.png"

    # --- 2. Device Setup and Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = LidarEncoder(embedding_size=config['embedding_size'])
    decoder = LidarDecoder(embedding_size=config['embedding_size'], num_ranges=config['num_ranges'])
    autoencoder = Autoencoder(encoder, decoder).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=config['learning_rate'])
    
    # --- 3. Training the Autoencoder ---
    if os.path.exists(config['model_save_path']):
        print(f"Existing Autoencoder model found at {config['model_save_path']}. Skipping training.")
        autoencoder.load_state_dict(torch.load(config['model_save_path'], map_location=device))
    else:
        print("No existing Autoencoder model found. Starting training from scratch.")
        train_dataset = LidarDataset(data_pcd_folder, config)
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        
        print("Starting Autoencoder Training...")
        for epoch in range(config['num_epochs']):
            autoencoder.train()
            total_loss = 0
            for data in train_dataloader:
                data = data.to(device)
                reconstructions = autoencoder(data)
                loss = criterion(reconstructions, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{config['num_epochs']}, Average Loss: {avg_loss:.4f}")
        
        os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
        torch.save(autoencoder.state_dict(), config['model_save_path'])
        print(f"Trained Autoencoder saved to {config['model_save_path']}")
        
    autoencoder.eval()

    # --- 4. Generate Embeddings for Clustering ---
    print("\n--- Generating embeddings for training data and performing clustering ---")
    dataset_for_clustering = LidarDataset(data_pcd_folder, config)
    dataloader_for_clustering = DataLoader(dataset_for_clustering, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    all_train_embeddings = []
    train_pcd_filenames_full = dataset_for_clustering.pcd_files
    
    with torch.no_grad():
        for data in dataloader_for_clustering:
            data = data.to(device)
            embeddings = autoencoder.encoder(data)
            all_train_embeddings.append(embeddings.cpu().numpy())
    
    all_train_embeddings_np = np.vstack(all_train_embeddings)
    print(f"Generated {all_train_embeddings_np.shape[0]} embeddings from training data.")
    
    # --- 5. Scale Embeddings and Perform HDBSCAN ---
    scaler = StandardScaler()
    scaled_train_embeddings = scaler.fit_transform(all_train_embeddings_np)
    print("Scaled training embeddings.")
    
    os.makedirs(os.path.dirname(config['scaler_path']), exist_ok=True)
    with open(config['scaler_path'], 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler model saved to {config['scaler_path']}")

    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=config['hdbscan_min_cluster_size'], 
        cluster_selection_epsilon=config['hdbscan_cluster_selection_epsilon'],
        prediction_data=True,
        core_dist_n_jobs=-1
    )
    train_cluster_labels = cluster_model.fit_predict(scaled_train_embeddings)
    
    os.makedirs(os.path.dirname(config['hdbscan_model_path']), exist_ok=True)
    with open(config['hdbscan_model_path'], 'wb') as f:
        pickle.dump(cluster_model, f)
    print(f"HDBSCAN model saved to {config['hdbscan_model_path']}")
    
    unique_train_clusters = np.unique(train_cluster_labels)
    print(f"Found {len(unique_train_clusters)} clusters in training data: {unique_train_clusters}")
    
    # --- 6. Calculate and Save Cluster Centroids ---
    print("\n--- Calculating and saving cluster centroids ---")
    cluster_centroids = {}
    for cluster_id in unique_train_clusters:
        if cluster_id != -1:
            cluster_points = scaled_train_embeddings[train_cluster_labels == cluster_id]
            if len(cluster_points) > 0:
                cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)
    
    os.makedirs(os.path.dirname(config['cluster_centroids_path']), exist_ok=True)
    with open(config['cluster_centroids_path'], 'wb') as f:
        pickle.dump(cluster_centroids, f)
    print(f"Cluster centroids saved to {config['cluster_centroids_path']}")
    
    # --- 7. Analyze Noise Points and Visualize Distances ---
    print("\n--- Analyzing distances of noise points to known clusters ---")
    noise_points_embeddings = scaled_train_embeddings[train_cluster_labels == -1]
    
    if len(noise_points_embeddings) > 0 and cluster_centroids:
        min_distances = [min([np.linalg.norm(noise_point - centroid) for centroid in cluster_centroids.values()]) for noise_point in noise_points_embeddings]
        min_distances_np = np.array(min_distances)
        
        plt.figure(figsize=(10, 6))
        plt.hist(min_distances_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribution of Distances from Noise Points to Nearest Cluster Centroid')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        p95 = np.percentile(min_distances_np, 95)
        plt.axvline(p95, color='green', linestyle='--', linewidth=1, label=f'95th Percentile: {p95:.2f}')
        plt.legend()
        
        os.makedirs(os.path.dirname(config['noise_distance_plot_path']), exist_ok=True)
        plt.savefig(config['noise_distance_plot_path'])
        print(f"Noise distance distribution plot saved to {config['noise_distance_plot_path']}")
        print(f"Suggested reassignment threshold (95th percentile): {p95:.2f}")

    # --- 8. Create Cluster-to-File Mapping and GIF ---
    print("\n--- Generating cluster-to-filepaths mapping for training data ---")
    cluster_to_train_filepaths = {cluster_id: [] for cluster_id in unique_train_clusters}
    for i, label in enumerate(train_cluster_labels):
        cluster_to_train_filepaths[label].append(os.path.basename(train_pcd_filenames_full[i]))

    os.makedirs(os.path.dirname(config['train_cluster_map_path']), exist_ok=True)
    with open(config['train_cluster_map_path'], 'w') as f:
        for cluster_id in sorted(cluster_to_train_filepaths.keys()):
            filenames_in_cluster = cluster_to_train_filepaths[cluster_id]
            f.write(f"Cluster {cluster_id}: {', '.join(filenames_in_cluster)}\n")
    print(f"Training cluster mapping saved to: {config['train_cluster_map_path']}")
    
    labels_file_exists = os.path.exists(config['cluster_labels_mapping_path'])
    cluster_id_to_label = {}
    if labels_file_exists and os.stat(config['cluster_labels_mapping_path']).st_size > 2:
        with open(config['cluster_labels_mapping_path'], 'r') as f:
            cluster_id_to_label = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded existing cluster labels from {config['cluster_labels_mapping_path']}")
    else:
        print(f"\nNo existing cluster labels found. Please review the output plot and create the file '{config['cluster_labels_mapping_path']}' with your labels.")
    
        for cluster_id in sorted(unique_train_clusters):
            if cluster_id != -1: # Don't display noise clusters for manual labeling
                print(f"\nViewing samples for Cluster: {cluster_id}")
                cluster_filenames = cluster_to_train_filepaths[cluster_id]
                display_cluster_samples(data_pcd_folder, cluster_filenames, cluster_id, 
                                        3, config)
            else:
                print(f"\nSkipping display for Noise Cluster (-1).")
        
        print("\n" + "="*80)
        print("END OF MANUAL LABELING STEP.")
        print(f"Please create/edit the file '{config['cluster_labels_mapping_path']}' with your labels.")
        print("Then, run the script again.")
        print("="*80)
        exit() 

    color_map_global = {}
    color_map_global[-1] = np.array([0.2, 0.2, 0.2, 1.0]) # Dark Grey for HDBSCAN Noise
    color_map_global[-2] = np.array([1.0, 0.5, 0.0, 1.0]) # Orange for New Cluster / Outlier

    actual_clusters = sorted([c for c in unique_train_clusters if c >= 0])
    if len(actual_clusters) > 0:
        cluster_colors_cmap = cm.get_cmap('rainbow', len(actual_clusters))
        for i, cluster_id in enumerate(actual_clusters):
            color_map_global[cluster_id] = cluster_colors_cmap(i / (len(actual_clusters) - 1)) if len(actual_clusters) > 1 else cluster_colors_cmap(0.5)
    
    print("\n--- Generating 2D t-SNE plot for training data clusters ---")
    fig_train_plot, ax_train_plot = plt.subplots(figsize=(12,8))

    if scaled_train_embeddings.shape[0] > 2 and scaled_train_embeddings.shape[1] > 2:
        tsne = TSNE(n_components=2, perplexity=min(30, scaled_train_embeddings.shape[0]-1), learning_rate='auto', init='random', random_state=42)
        train_embeddings_2d = tsne.fit_transform(scaled_train_embeddings)
        print("Reduced training embeddings to 2D using t-SNE.")
    else:
        print("Not enough samples for t-SNE. Using original embeddings if 2D or less.")
        train_embeddings_2d = scaled_train_embeddings[:, :2] if scaled_train_embeddings.shape[1] >=2 else scaled_train_embeddings

    # Plot each cluster
    for cluster_id in sorted(unique_train_clusters):
        mask = (train_cluster_labels == cluster_id)
        if np.any(mask): # Ensure there are points for this cluster
            label_text = f'Cluster {cluster_id}'
            if cluster_id in cluster_id_to_label:
                label_text += f' ({cluster_id_to_label[cluster_id]})'
            elif cluster_id == -1:
                label_text = 'Noise/Uncertain (HDBSCAN)'

            ax_train_plot.scatter(train_embeddings_2d[mask, 0], train_embeddings_2d[mask, 1],
                                  color=color_map_global.get(cluster_id, np.array([0.5, 0.5, 0.5, 1.0])),
                                  label=label_text,
                                  s=20, alpha=0.7)

    ax_train_plot.set_xlabel("Component 1")
    ax_train_plot.set_ylabel("Component 2")
    ax_train_plot.set_title("Training Data Clusters (2D t-SNE of Embeddings)")
    ax_train_plot.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_train_plot.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(config["output_training_plot_path"]), exist_ok=True)
    plt.savefig(config["output_training_plot_path"])
    print(f"Training data clusters plot saved to: {config['output_training_plot_path']}")
    plt.close(fig_train_plot) # Close the plot to free memory
    
    # --- Plot Distances to Centroids for Training Data ---
    print("\n--- Generating distance plot for training data ---")
    distance_threshold_for_new_clusters = 5

    # Create a figure with two subplots, one above the other
    fig, (ax_dist_plot_train, ax_time_series_train) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Get unique cluster labels from the training data
    unique_final_training_clusters = np.unique(train_cluster_labels)

    # --- First Subplot: Distance Plot for Training Data ---
    for cluster_id in sorted(unique_final_training_clusters):
        mask = (train_cluster_labels == cluster_id)
        if np.any(mask): 
            label_text = f'Cluster {cluster_id}'
            if cluster_id in cluster_id_to_label:
                label_text += f' ({cluster_id_to_label[cluster_id]})'
            elif cluster_id == -1:
                label_text = 'HDBSCAN Noise' 
            elif cluster_id == -2:
                label_text = 'New Cluster / Outlier' 
            else:
                label_text = f'Unknown Cluster {cluster_id}'

            # Only plot valid distances
            min_distances_to_known_centroids_training = np.full(len(train_cluster_labels), np.nan)
            if cluster_centroids:
                for i, embedding in enumerate(scaled_train_embeddings):
                    distances = [np.linalg.norm(embedding - centroid) for centroid in cluster_centroids.values()]
                    if distances:
                        min_distances_to_known_centroids_training[i] = min(distances)
            
            valid_timesteps = np.arange(len(train_pcd_filenames_full))[mask & ~np.isnan(min_distances_to_known_centroids_training)]
            valid_distances = min_distances_to_known_centroids_training[mask & ~np.isnan(min_distances_to_known_centroids_training)]

            if len(valid_timesteps) > 0:
                ax_dist_plot_train.scatter(valid_timesteps, valid_distances,
                                    color=color_map_global.get(cluster_id, np.array([0.5, 0.5, 0.5, 1.0])),
                                    label=label_text,
                                    s=30, alpha=0.7)

    ax_dist_plot_train.axhline(y=distance_threshold_for_new_clusters, color='r', linestyle='--', label=f'New Cluster Threshold ({distance_threshold_for_new_clusters:.2f})')
    ax_dist_plot_train.set_xlabel("Training Data Sample Index")
    ax_dist_plot_train.set_ylabel("Minimum Distance to Closest Known Cluster Centroid")
    ax_dist_plot_train.set_title("Training Data: Minimum Distance to Known Cluster Centroids")
    ax_dist_plot_train.legend(title="Assigned Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_dist_plot_train.grid(True)


    # --- Second Subplot: All Points in Time by Cluster ID for Training Data ---
    ax_time_series_train.set_title("All Training Data Points by Cluster ID Over Time")
    ax_time_series_train.set_xlabel("Training Data Sample Index")
    # Iterate through each point and plot it individually to color it by cluster_id
    for i, cluster_id in enumerate(train_cluster_labels):
        color = color_map_global.get(cluster_id, np.array([0.5, 0.5, 0.5, 0.3])) 
        ax_time_series_train.plot(i, 0, marker='o', markersize=5, color=color, linestyle='')

    ax_time_series_train.set_yticks([]) 
    ax_time_series_train.set_ylim([-0.5, 0.5]) 
    ax_time_series_train.grid(True, axis='x') 

    plt.tight_layout() 
    os.makedirs(os.path.dirname(config["training_distance_plot_path"]), exist_ok=True)
    plt.savefig(config["training_distance_plot_path"])
    print(f"Training data clusters plot saved to: {config['training_distance_plot_path']}")
    plt.close(fig)

    print("\n--- Generating GIF for training data ---")
    color_map = {cluster_id: cm.jet(i / len(unique_train_clusters)) for i, cluster_id in enumerate(sorted(unique_train_clusters))}
    color_map[-1] = np.array([0.2, 0.2, 0.2, 1.0]) # Noise color

    os.makedirs(os.path.dirname(config['output_training_gif_path']), exist_ok=True)
    generate_cluster_gif(
        data_pcd_folder,
        train_pcd_filenames_full,
        train_cluster_labels,
        cluster_id_to_label,
        config['output_training_gif_path'],
        config['gif_frames_per_second'],
        config,
        color_map
    )
    print("Process complete.")

# --- Main Script Execution ---
if __name__ == "__main__":
    # 1. Configuration Parameters
    training_data_name = "bridge2_carla"
    training_pcd_folder = f"../../../../carla_data/{training_data_name}/{training_data_name}_pcds"
    
    # Define and centralize all hyperparameters and paths
    global_config = {
        "embedding_size": 16,
        "num_ranges": 32,
        "angle_increment_deg": float(360.0 / 32),
        "max_lidar_range": 20.0,
        "z_threshold_upper": 2.25,
        "z_threshold_lower": 2,
        "z_threshold_upper_2": 0,
        "z_threshold_lower_2": 0,
        "num_epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hdbscan_min_cluster_size": 8,
        "hdbscan_cluster_selection_epsilon": 0.0,
        "num_samples_to_show_per_cluster": 3,
        "gif_frames_per_second": 15,
    }

    # Run the main function for the training data
    run_autoencoder_and_cluster(training_data_name, training_pcd_folder, global_config)

    # -----------------------------------------------------------
    # Test Directory Integration
    # -----------------------------------------------------------

    def run_inference_on_test_data(test_data_name, test_pcd_folder, training_data_name, global_config, z_thresholds_for_test=None):
        """
        Performs inference on a new test dataset using pre-trained models.

        Args:
            test_data_name (str): The name of the test dataset.
            test_pcd_folder (str): The path to the folder with test PCD files.
            training_data_name (str): The name of the original training dataset to load models from.
            global_config (dict): The dictionary of all configuration parameters.
            z_thresholds_for_test (tuple, optional): A tuple of (z_upper_1, z_lower_1, z_upper_2, z_lower_2)
                                                     to be used specifically for visualization of the test data.
                                                     If None, the global_config values are used.
        """
        print(f"\n--- Starting inference for test dataset: {test_data_name} ---")

        # Define artifact paths based on the training data name
        hdbscan_model_path = f"encoder_weights/{training_data_name}/hdbscan_model_{training_data_name}.pkl"
        scaler_path = f"encoder_weights/{training_data_name}/scaler_{training_data_name}.pkl"
        model_save_path = f"encoder_weights/{training_data_name}/lidar_encoder_autoencoder_{training_data_name}.pth"
        cluster_centroids_path = f"encoder_weights/{training_data_name}/cluster_centroids_{training_data_name}.pkl"
        cluster_labels_mapping_path = f"encoder_weights/{training_data_name}/cluster_id_to_label_{training_data_name}.json"

        # Check if the necessary models exist
        if not all(os.path.exists(p) for p in [hdbscan_model_path, scaler_path, model_save_path, cluster_centroids_path, cluster_labels_mapping_path]):
            print("ERROR: Required pre-trained models or artifacts from the training phase do not exist.")
            print("Please ensure you have successfully run the training script first.")
            return

        # Load pre-trained models and other artifacts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = LidarEncoder(embedding_size=global_config['embedding_size'])
        autoencoder = Autoencoder(encoder, LidarDecoder(global_config['embedding_size'], global_config['num_ranges'])).to(device)
        autoencoder.load_state_dict(torch.load(model_save_path, map_location=device))
        autoencoder.eval()
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(hdbscan_model_path, 'rb') as f:
            hdbscan_model = pickle.load(f)
        with open(cluster_centroids_path, 'rb') as f:
            cluster_centroids = pickle.load(f)
        with open(cluster_labels_mapping_path, 'r') as f:
            cluster_id_to_label = {int(k): v for k, v in json.load(f).items()}

        # Generate embeddings for test data
        test_dataset = LidarDataset(test_pcd_folder, global_config)
        test_dataloader = DataLoader(test_dataset, batch_size=global_config['batch_size'], shuffle=False, num_workers=4)
        all_test_embeddings = []
        test_pcd_filenames_full = test_dataset.pcd_files

        if not test_pcd_filenames_full:
            print("No PCD files found in the test directory. Exiting.")
            return

        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(device)
                embeddings = encoder(data)
                all_test_embeddings.append(embeddings.cpu().numpy())
        all_test_embeddings_np = np.vstack(all_test_embeddings)
        print(f"Generated {all_test_embeddings_np.shape[0]} embeddings from test data.")
        
        # Scale embeddings using the pre-trained scaler
        scaled_test_embeddings = scaler.transform(all_test_embeddings_np)
        print("Scaled test embeddings using the pre-trained scaler.")

        # Predict clusters using HDBSCAN's prediction function
        test_cluster_labels, _ = hdbscan.approximate_predict(hdbscan_model, scaled_test_embeddings)
        
        # Reassign noise points
        new_test_labels = np.copy(test_cluster_labels)
        noise_indices = np.where(test_cluster_labels == -1)[0]
        reassigned_count = 0
        distance_threshold_for_new_clusters = 5 # Use a predefined or calculated threshold

        for i in noise_indices:
            noise_point = scaled_test_embeddings[i]
            min_dist = float('inf')
            closest_cluster_id = -1
            if not cluster_centroids:
                # If there are no training clusters, all noise points are new clusters.
                new_test_labels[i] = -2
                continue
                
            for cluster_id, centroid in cluster_centroids.items():
                dist = np.linalg.norm(noise_point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster_id = cluster_id
            
            if min_dist < distance_threshold_for_new_clusters:
                new_test_labels[i] = closest_cluster_id
                reassigned_count += 1
            else:
                new_test_labels[i] = -2 # Assign a new cluster ID for true outliers
        
        print(f"Reassigned {reassigned_count} noise points to existing clusters.")
        print(f"Identified {np.sum(new_test_labels == -2)} new outliers.")

        unique_test_clusters = np.unique(new_test_labels)
        print(f"Found {len(unique_test_clusters)} clusters in test data: {unique_test_clusters}")

        # --- Generate plots and GIF for test data ---
        
        # Create a color map for the test data clusters
        color_map_test = {}
        color_map_test[-1] = np.array([0.2, 0.2, 0.2, 1.0]) 
        color_map_test[-2] = np.array([1.0, 0.5, 0.0, 1.0]) 

        actual_clusters = sorted([c for c in unique_test_clusters if c >= 0])
        if len(actual_clusters) > 0:
            cluster_colors_cmap = cm.get_cmap('rainbow', len(actual_clusters))
            for i, cluster_id in enumerate(actual_clusters):
                color_map_test[cluster_id] = cluster_colors_cmap(i / (len(actual_clusters) - 1)) if len(actual_clusters) > 1 else cluster_colors_cmap(0.5)

        # Generate 2D plot for test data
        fig_test_plot, ax_test_plot = plt.subplots(figsize=(12,8))

        if scaled_test_embeddings.shape[0] > 2 and scaled_test_embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, perplexity=min(30, scaled_test_embeddings.shape[0]-1), learning_rate='auto', init='random', random_state=42)
            test_embeddings_2d = tsne.fit_transform(scaled_test_embeddings)
        else:
            test_embeddings_2d = scaled_test_embeddings[:, :2] if scaled_test_embeddings.shape[1] >= 2 else scaled_test_embeddings

        for cluster_id in sorted(unique_test_clusters):
            mask = (new_test_labels == cluster_id)
            if np.any(mask):
                label_text = f'Cluster {cluster_id}'
                if cluster_id in cluster_id_to_label:
                    label_text += f' ({cluster_id_to_label[cluster_id]})'
                elif cluster_id == -1:
                    label_text = 'HDBSCAN Noise'
                elif cluster_id == -2:
                    label_text = 'New Cluster / Outlier'

                ax_test_plot.scatter(test_embeddings_2d[mask, 0], test_embeddings_2d[mask, 1],
                                    color=color_map_test.get(cluster_id, np.array([0.5, 0.5, 0.5, 1.0])),
                                    label=label_text,
                                    s=20, alpha=0.7)

        ax_test_plot.set_xlabel("Component 1")
        ax_test_plot.set_ylabel("Component 2")
        ax_test_plot.set_title("Test Data Clusters (2D t-SNE of Embeddings)")
        ax_test_plot.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_test_plot.grid(True)
        plt.tight_layout()

        output_test_plot_path = f"results/{test_data_name}/test_data_clusters_plot.png"
        os.makedirs(os.path.dirname(output_test_plot_path), exist_ok=True)
        plt.savefig(output_test_plot_path)
        print(f"Test data clusters plot saved to: {output_test_plot_path}")
        plt.close(fig_test_plot)
        
        
        # --- Plot Distances to Centroids for Training Data ---
        print("\n--- Generating distance plot for training data ---")
        distance_threshold_for_new_clusters = 5

        # Create a figure with two subplots, one above the other
        fig, (ax_dist_plot_train, ax_time_series_train) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Get unique cluster labels from the training data
        unique_final_training_clusters = np.unique(actual_clusters)

        # --- First Subplot: Distance Plot for Training Data ---
        # for cluster_id in sorted(unique_final_training_clusters):
        #     mask = (actual_clusters == cluster_id)
        #     if np.any(mask): 
        #         label_text = f'Cluster {cluster_id}'
        #         if cluster_id in cluster_id_to_label:
        #             label_text += f' ({cluster_id_to_label[cluster_id]})'
        #         elif cluster_id == -1:
        #             label_text = 'HDBSCAN Noise' 
        #         elif cluster_id == -2:
        #             label_text = 'New Cluster / Outlier' 
        #         else:
        #             label_text = f'Unknown Cluster {cluster_id}'

        #         # Only plot valid distances
        #         min_distances_to_known_centroids_training = np.full(len(actual_clusters), np.nan)
        #         if cluster_centroids:
        #             for i, embedding in enumerate(scaled_test_embeddings):
        #                 distances = [np.linalg.norm(embedding - centroid) for centroid in cluster_centroids.values()]
        #                 if distances:
        #                     min_distances_to_known_centroids_training[i] = min(distances)
                
        #         valid_timesteps = np.arange(len(test_pcd_filenames_full))[mask & ~np.isnan(min_distances_to_known_centroids_training)]
        #         valid_distances = min_distances_to_known_centroids_training[mask & ~np.isnan(min_distances_to_known_centroids_training)]

        #         if len(valid_timesteps) > 0:
        #             ax_dist_plot_train.scatter(valid_timesteps, valid_distances,
        #                                 color=color_map_test.get(cluster_id, np.array([0.5, 0.5, 0.5, 1.0])),
        #                                 label=label_text,
        #                                 s=30, alpha=0.7)

        # ax_dist_plot_train.axhline(y=distance_threshold_for_new_clusters, color='r', linestyle='--', label=f'New Cluster Threshold ({distance_threshold_for_new_clusters:.2f})')
        # ax_dist_plot_train.set_xlabel("Training Data Sample Index")
        # ax_dist_plot_train.set_ylabel("Minimum Distance to Closest Known Cluster Centroid")
        # ax_dist_plot_train.set_title("Training Data: Minimum Distance to Known Cluster Centroids")
        # ax_dist_plot_train.legend(title="Assigned Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        # ax_dist_plot_train.grid(True)


        # --- Second Subplot: All Points in Time by Cluster ID for Training Data ---
        ax_time_series_train.set_title("All Training Data Points by Cluster ID Over Time")
        ax_time_series_train.set_xlabel("Training Data Sample Index")
        # Iterate through each point and plot it individually to color it by cluster_id
        for i, cluster_id in enumerate(new_test_labels):
            color = color_map_test.get(cluster_id, np.array([0.5, 0.5, 0.5, 0.3])) 
            ax_time_series_train.plot(i, 0, marker='o', markersize=5, color=color, linestyle='')

        ax_time_series_train.set_yticks([]) 
        ax_time_series_train.set_ylim([-0.5, 0.5]) 
        ax_time_series_train.grid(True, axis='x') 
        
        output_test_dist_path = f"results/{test_data_name}/test_data_dist.png"

        plt.tight_layout() 
        os.makedirs(os.path.dirname(output_test_dist_path), exist_ok=True)
        plt.savefig(output_test_dist_path)
        print(f"Training data clusters plot saved to: {output_test_dist_path}")
        plt.close(fig)


        # Generate GIF for test data
        output_test_gif_path = f"results/{test_data_name}/test_data_clustered.gif"
        os.makedirs(os.path.dirname(output_test_gif_path), exist_ok=True)
        generate_cluster_gif(
            test_pcd_folder,
            test_pcd_filenames_full,
            new_test_labels,
            cluster_id_to_label,
            output_test_gif_path,
            global_config['gif_frames_per_second'],
            global_config,
            color_map_test,
            z_thresholds_gif=z_thresholds_for_test
        )
        print("Test inference process complete.")


    # --- Example of how to use the test function ---
    # Uncomment the lines below to run inference on a new test dataset
    # You must have a 'carla_data' directory with a 'test_set_1' folder containing PCDs
    # and you must run the training script first to generate the models.
    #
    # test_data_name = "intersection_straight"
    # test_pcd_folder = f"../../../../carla_data/{test_data_name}/{test_data_name}_pcds"
    
    # # # Example of a new set of z_thresholds for the test visualization
    # # # This will only affec the GIF visualization, not the clustering
    # test_z_thresholds = (2, 0, 0, 0.0)
    
    # # Run the test inference process
    # run_inference_on_test_data(test_data_name, test_pcd_folder, training_data_name, global_config, test_z_thresholds)