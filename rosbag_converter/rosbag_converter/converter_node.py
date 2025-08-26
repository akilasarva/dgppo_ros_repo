import rclpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageFilter
from ros2_numpy import point_cloud2 as ros2_np
from sensor_msgs.msg import PointCloud2
import os
import glob
import open3d as o3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import json
import imageio
from PIL import Image, ImageDraw, ImageFont
import time
from sklearn import metrics
import torch.optim as optim

# --- Configuration ---
global_config = {
    'training_data_name': 'ros2_bigbag1',
    'bag_file_path': '/ros2_data/ros2_bigbag/bigbag1.mcap', # CHANGE THIS PATH
    'point_cloud_topic': '/warthog1/sensors/ouster/points', # CHANGE THIS TOPIC NAME
    'model_save_path': 'encoder_weights/ros2_bigbag1/lidar_encoder_autoencoder.pth',
    'hdbscan_model_path': 'encoder_weights/ros2_bigbag1/hdbscan_model.pkl',
    'cluster_labels_mapping_path': 'encoder_weights/ros2_bigbag1/cluster_id_to_label.json',
    'embedding_size': 16,
    'num_ranges': 32, #1440, # Assuming 0.25 degree increment for 360 degrees
    'max_lidar_range': 5.0,
    'angle_increment_deg': 0.25,
    'z_threshold_upper': 1.5,
    'z_threshold_lower': 0.55,
    'z_threshold_upper_2': 0,
    'z_threshold_lower_2': 0,
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'hdbscan_min_cluster_size': 8,
    'output_training_gif_path': 'results/ros2_bigbag1/training_data_clustered_hdbscan.gif',
    'output_training_plot_path': 'results/ros2_bigbag1/training_data_clusters_plot.png',
    'gif_frames_per_second': 15,
    'num_samples_to_show_per_cluster': 3
}

# --- Autoencoder Model Definition ---
class LidarEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(LidarEncoder, self).__init__()
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

# --- ROS Bag Processing Function (New) ---
def get_ranges_from_bag(bag_filepath, max_lidar_range, angle_increment_deg,
                        z_threshold_upper, z_threshold_lower,
                        z_threshold_upper_2, z_threshold_lower_2,
                        point_cloud_topic):
    
    reader = SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_filepath, storage_id='mcap'),
            rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
        )
    except Exception as e:
        print(f"Error opening bag file at {bag_filepath}: {e}")
        return

    topic_filter = StorageFilter(topics=[point_cloud_topic])
    reader.set_filter(topic_filter)

    num_angles = int(360 / angle_increment_deg)
    
    while reader.has_next():
        topic, data, t = reader.read_next()
        
        if topic == point_cloud_topic:
            try:
                msg = deserialize_message(data, PointCloud2)
            except Exception as e:
                print(f"Error deserializing message: {e}")
                continue

            points_full = ros2_np.pointcloud2_to_xyz_array(msg)
            finite_mask = np.isfinite(points_full).all(axis=1)
            points_full_finite = points_full[finite_mask]

            slice_mask_1 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower) & \
                           (np.abs(points_full_finite[:, 2]) <= z_threshold_upper)
            slice_mask_2 = (np.abs(points_full_finite[:, 2]) >= z_threshold_lower_2) & \
                           (np.abs(points_full_finite[:, 2]) <= z_threshold_upper_2)
            combined_slice_mask = (slice_mask_1 | slice_mask_2) & \
                                  (np.abs(points_full_finite[:, 0]) <= max_lidar_range) & \
                                  (np.abs(points_full_finite[:, 1]) <= max_lidar_range)
            points_slice = points_full_finite[combined_slice_mask]

            ranges = np.full(num_angles, max_lidar_range, dtype=np.float32)
            if points_slice.shape[0] > 0:
                point_angles = np.arctan2(points_slice[:, 1], points_slice[:, 0])
                point_dists = np.linalg.norm(points_slice[:, :2], axis=1)

                for i in range(num_angles):
                    angle_rad = np.deg2rad(i * angle_increment_deg)
                    angular_tolerance = np.deg2rad(angle_increment_deg / 2)
                    angular_diff = np.arctan2(np.sin(point_angles - angle_rad), np.cos(point_angles - angle_rad))
                    sector_mask = np.abs(angular_diff) <= angular_tolerance

                    if np.any(sector_mask):
                        min_dist_in_sector = np.min(point_dists[sector_mask])
                        ranges[i] = min_dist_in_sector

            normalized_ranges = ranges / max_lidar_range
            yield torch.tensor(normalized_ranges, dtype=torch.float32)

    reader.close()

# --- Placeholder for visualization functions (they need to be adapted) ---
# NOTE: Visualizing directly from a bag file frame-by-frame is more complex
# than with separate PCD files. The functions below are placeholders and would
# require significant modification to work directly with a bag file.
# For now, we will focus on the training and clustering logic.

# def visualize_footprint_at_timestep(*args, **kwargs):
#     print("Visualization is not supported in this direct bag-reading mode.")
#     return None, None, None, None

# def display_cluster_samples(*args, **kwargs):
#     print("Manual sample display is not supported in this direct bag-reading mode.")

# def generate_cluster_gif(*args, **kwargs):
#     print("GIF generation is not supported in this direct bag-reading mode.")

# --- Main Script Execution ---
if __name__ == "__main__":
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Models ---
    encoder = LidarEncoder(embedding_size=global_config['embedding_size'])
    decoder = LidarDecoder(embedding_size=global_config['embedding_size'], num_ranges=global_config['num_ranges'])
    autoencoder = Autoencoder(encoder, decoder)
    print(autoencoder)

    autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=global_config['learning_rate'])
    
    # --- Load Data from Bag File ---
    all_training_ranges = []
    print(f"Starting to process ROS 2 bag file '{global_config['bag_file_path']}'...")

    try:
        for ranges_tensor in get_ranges_from_bag(
            bag_filepath=global_config['bag_file_path'],
            max_lidar_range=global_config['max_lidar_range'],
            angle_increment_deg=global_config['angle_increment_deg'],
            z_threshold_upper=global_config['z_threshold_upper'],
            z_threshold_lower=global_config['z_threshold_lower'],
            z_threshold_upper_2=global_config['z_threshold_upper_2'],
            z_threshold_lower_2=global_config['z_threshold_lower_2'],
            point_cloud_topic=global_config['point_cloud_topic']
        ):
            all_training_ranges.append(ranges_tensor)
    except Exception as e:
        print(f"An error occurred during bag processing: {e}")
        exit()

    if not all_training_ranges:
        print("No valid LiDAR data found in the bag file. Exiting.")
        exit()

    train_data = torch.stack(all_training_ranges)
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=global_config['batch_size'], shuffle=True, num_workers=0)
    
    # --- Initialize and Train Autoencoder ---
    if not os.path.exists(global_config['model_save_path']):
        print("No existing Autoencoder model found. Starting training from scratch.")
        print("Starting Autoencoder Training...")
        for epoch in range(global_config['num_epochs']):
            autoencoder.train()
            total_loss = 0
            for batch_idx, data in enumerate(train_dataloader):
                data = data[0].to(device)
                reconstructions = autoencoder(data)
                loss = criterion(reconstructions, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{global_config['num_epochs']}, Average Loss: {avg_loss:.4f}")
        
        os.makedirs(os.path.dirname(global_config['model_save_path']), exist_ok=True)
        torch.save(autoencoder.state_dict(), global_config['model_save_path'])
        print(f"Trained Autoencoder saved to {global_config['model_save_path']}")
    else:
        print(f"Loading existing Autoencoder model from {global_config['model_save_path']}")
        autoencoder.load_state_dict(torch.load(global_config['model_save_path']))
        
    autoencoder.eval() 

    # --- Generate Embeddings for Training Data and Cluster ---
    print("\n--- Generating embeddings for training data and performing clustering ---")
    all_train_embeddings = []
    
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            data = data[0].to(device)
            embeddings = autoencoder.encoder(data)
            all_train_embeddings.append(embeddings.cpu().numpy())
    
    all_train_embeddings_np = np.vstack(all_train_embeddings)
    print(f"Generated {all_train_embeddings_np.shape[0]} embeddings from training data.")

    scaler = StandardScaler()
    scaled_train_embeddings = scaler.fit_transform(all_train_embeddings_np)
    print("Scaled training embeddings.")
    
    scaler_path = os.path.join(os.path.dirname(global_config['hdbscan_model_path']), 'scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler model saved to {scaler_path}")

    cluster_model = None
    if os.path.exists(global_config['hdbscan_model_path']):
        print(f"Loading existing HDBSCAN model from {global_config['hdbscan_model_path']}")
        with open(global_config['hdbscan_model_path'], 'rb') as f:
            cluster_model = pickle.load(f)
        train_cluster_labels = cluster_model.labels_ 
    else:
        print(f"Performing HDBSCAN clustering with min_cluster_size={global_config['hdbscan_min_cluster_size']}...")
        cluster_model = hdbscan.HDBSCAN(min_cluster_size=global_config['hdbscan_min_cluster_size'], 
                                        cluster_selection_epsilon=0.0,
                                        prediction_data=True,
                                        core_dist_n_jobs=-1)
        train_cluster_labels = cluster_model.fit_predict(scaled_train_embeddings)
        os.makedirs(os.path.dirname(global_config['hdbscan_model_path']), exist_ok=True)
        with open(global_config['hdbscan_model_path'], 'wb') as f:
            pickle.dump(cluster_model, f)
        print(f"HDBSCAN model saved to {global_config['hdbscan_model_path']}")

    unique_train_clusters = np.unique(train_cluster_labels)
    print(f"Found {len(unique_train_clusters)} clusters in training data: {unique_train_clusters}")

    # --- Generate 2D t-SNE plot for training data clusters ---
    print("\n--- Generating 2D t-SNE plot for training data clusters ---")
    fig_train_plot, ax_train_plot = plt.subplots(figsize=(12,8))

    if global_config['embedding_size'] > 2:
        if scaled_train_embeddings.shape[0] > 2:
            tsne = TSNE(n_components=2, perplexity=min(30, scaled_train_embeddings.shape[0]-1), learning_rate='auto', init='random', random_state=42)
            train_embeddings_2d = tsne.fit_transform(scaled_train_embeddings)
            print("Reduced training embeddings to 2D using t-SNE.")
        else:
            print("Not enough samples for t-SNE. Using original embeddings if 2D or less.")
            train_embeddings_2d = scaled_train_embeddings[:, :2] if scaled_train_embeddings.shape[1] >=2 else scaled_train_embeddings
    else:
        train_embeddings_2d = scaled_train_embeddings
        print("Training embeddings already 2D or less, no t-SNE applied.")
        
    color_map_global = {}
    actual_clusters = sorted([c for c in unique_train_clusters if c >= 0])
    if len(actual_clusters) > 0:
        cluster_colors_cmap = cm.get_cmap('rainbow', len(actual_clusters))
        for i, cluster_id in enumerate(actual_clusters):
            color_map_global[cluster_id] = cluster_colors_cmap(i / (len(actual_clusters) - 1)) if len(actual_clusters) > 1 else cluster_colors_cmap(0.5)

    for cluster_id in sorted(unique_train_clusters):
        mask = (train_cluster_labels == cluster_id)
        if np.any(mask):
            label_text = f'Cluster {cluster_id}'
            if cluster_id == -1:
                label_text = 'Noise/Uncertain (HDBSCAN)'
                color_map_global[cluster_id] = np.array([0.2, 0.2, 0.2, 1.0])
            
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

    os.makedirs(os.path.dirname(global_config['output_training_plot_path']), exist_ok=True)
    plt.savefig(global_config['output_training_plot_path'])
    print(f"Training data clusters plot saved to: {global_config['output_training_plot_path']}")
    plt.close(fig_train_plot)