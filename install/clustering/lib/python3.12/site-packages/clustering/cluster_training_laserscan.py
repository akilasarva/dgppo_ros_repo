import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hdbscan
import pickle
import json
import imageio
from sklearn import metrics
import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan

# ======================================================================
# Classes for Autoencoder
# ======================================================================

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

# ======================================================================
# Finalized Dataset Class for reading from ROS 2 bag files
# ======================================================================
class LaserScanBagDataset(Dataset):
    def __init__(self, bag_path, topic_name, max_lidar_range):
        self.bag_path = bag_path
        self.topic_name = topic_name
        self.max_lidar_range = max_lidar_range
        self.ranges_list = self._load_data_from_bag()

    def __len__(self):
        return len(self.ranges_list)

    def __getitem__(self, idx):
        return self.ranges_list[idx]

    def _load_data_from_bag(self):
        print(f"Loading LaserScan data from bag: {self.bag_path}")
        reader = SequentialReader()

        storage_options = StorageOptions(uri=self.bag_path, storage_id='mcap')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader.open(storage_options, converter_options)
        
        # Set the filter
        storage_filter = StorageFilter(topics=[self.topic_name])
        reader.set_filter(storage_filter)
        
        ranges_list = []
        
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            
            # Use the hardcoded LaserScan message class
            msg = deserialize_message(data, LaserScan)

            ranges_np = np.array(msg.ranges, dtype=np.float32)
            ranges_np[ranges_np > self.max_lidar_range] = self.max_lidar_range
            normalized_ranges = ranges_np / self.max_lidar_range
            
            ranges_list.append(torch.tensor(normalized_ranges, dtype=torch.float32))

        print(f"Loaded {len(ranges_list)} LaserScan messages.")
        reader.close()
        return ranges_list
    
def polar_to_cartesian(ranges, angle_increment):
    """Converts a polar LaserScan to cartesian (x, y) coordinates."""
    angles = np.arange(len(ranges)) * angle_increment
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return x, y

def display_cluster_samples(dataset, cluster_labels, cluster_id, num_samples, max_lidar_range, angle_increment_deg):
    """Displays a few random LaserScan samples from a given cluster."""
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        print(f"No samples found for Cluster {cluster_id}")
        return

    print(f"\nDisplaying {min(num_samples, len(cluster_indices))} samples from Cluster: {cluster_id}")
    
    fig, axes = plt.subplots(1, min(num_samples, len(cluster_indices)), figsize=(15, 5))
    if min(num_samples, len(cluster_indices)) == 1:
        axes = [axes]  # Ensure axes is an iterable for a single plot
    
    random_indices = np.random.choice(cluster_indices, min(num_samples, len(cluster_indices)), replace=False)
    
    angle_increment = np.deg2rad(angle_increment_deg)
    
    for i, data_index in enumerate(random_indices):
        ranges_tensor = dataset[data_index]
        ranges_np = ranges_tensor.numpy()
        
        # Convert to cartesian coordinates for plotting
        x, y = polar_to_cartesian(ranges_np, angle_increment)
        
        # Plot the data
        axes[i].scatter(x, y, s=5)
        axes[i].set_title(f"Cluster {cluster_id}\nSample {data_index}")
        axes[i].set_aspect('equal', 'box')
        axes[i].set_xlim(-max_lidar_range, max_lidar_range)
        axes[i].set_ylim(-max_lidar_range, max_lidar_range)
        axes[i].set_xlabel("X (m)")
        axes[i].set_ylabel("Y (m)")
        
    plt.tight_layout()
    plt.show()
    
# ======================================================================
# Main Script Execution
# This section now includes the visualization and labeling.
# ======================================================================
if __name__ == "__main__":
    rclpy.init() # Initialize rclpy for bag reading

    # Define your project name and bag file paths
    training_data_name = "ros2_bigbag_laserscan"
    
    # Path to the bag containing the converted LaserScan messages
    laserscan_bag_path = "../../../../ros2_data/laserscan_bag" 
    laserscan_topic = "/my_lidar_scan"
    
    # --- Model and Clustering Parameters ---
    model_save_path = f"encoder_weights/{training_data_name}/lidar_encoder_autoencoder_{training_data_name}.pth"
    hdbscan_model_path = f"encoder_weights/{training_data_name}/hdbscan_model_{training_data_name}.pkl"
    scaler_path = f"encoder_weights/{training_data_name}/scaler_{training_data_name}.pkl"
    
    embedding_size = 16
    num_ranges = 32
    max_lidar_range = 10.0
    
    # NOTE: You must know the LaserScan's angular increment to plot it correctly.
    # A full 360-degree scan with 32 ranges has an angle_increment of 360/32 = 11.25 deg.
    # Adjust this value based on your specific sensor.
    angle_increment_deg = 360.0 / num_ranges
    
    num_epochs = 200
    batch_size = 32
    learning_rate = 0.001
    
    hdbscan_min_cluster_size = 8
    hdbscan_cluster_selection_epsilon = 0.0

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Models ---
    encoder = LidarEncoder(embedding_size=embedding_size)
    decoder = LidarDecoder(embedding_size=embedding_size, num_ranges=num_ranges)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    
    # --- NEW: Use the LaserScanBagDataset ---
    train_dataset = LaserScanBagDataset(
        bag_path=laserscan_bag_path,
        topic_name=laserscan_topic,
        max_lidar_range=max_lidar_range
    )
    
    if len(train_dataset) == 0:
        print("Error: No LaserScan messages found in the bag file. Exiting.")
        rclpy.shutdown()
        exit()
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # --- 1. Train the Autoencoder ---
    print("Starting Autoencoder Training...")
    for epoch in range(num_epochs):
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
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(autoencoder.state_dict(), model_save_path)
    print(f"Trained Autoencoder saved to {model_save_path}")

    # --- 2. Generate Embeddings and Cluster ---
    autoencoder.eval()
    all_train_embeddings = []
    with torch.no_grad():
        for data in train_dataloader:
            data = data.to(device)
            embeddings = autoencoder.encoder(data)
            all_train_embeddings.append(embeddings.cpu().numpy())
    
    all_train_embeddings_np = np.vstack(all_train_embeddings)
    print(f"Generated {all_train_embeddings_np.shape[0]} embeddings.")

    scaler = StandardScaler()
    scaled_train_embeddings = scaler.fit_transform(all_train_embeddings_np)
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler model saved to {scaler_path}")

    print(f"Performing HDBSCAN clustering with min_cluster_size={hdbscan_min_cluster_size}...")
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                                    cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                                    prediction_data=True,
                                    core_dist_n_jobs=-1)
    train_cluster_labels = cluster_model.fit_predict(scaled_train_embeddings)

    os.makedirs(os.path.dirname(hdbscan_model_path), exist_ok=True)
    with open(hdbscan_model_path, 'wb') as f:
        pickle.dump(cluster_model, f)
    print(f"HDBSCAN model saved to {hdbscan_model_path}")
    
    unique_train_clusters = np.unique(train_cluster_labels)
    num_train_clusters = len(unique_train_clusters)
    print(f"Found {num_train_clusters} clusters: {unique_train_clusters}")

    # --- 4. Manual Labeling and Visualization ---
    print("\nStarting manual labeling process...")
    print("We will display samples from each cluster (excluding noise) for you to inspect.")
    
    num_samples_to_show = 5 # Adjust this number as needed
    
    for cluster_id in sorted(unique_train_clusters):
        if cluster_id != -1:  # -1 is the noise cluster
            display_cluster_samples(
                dataset=train_dataset,
                cluster_labels=train_cluster_labels,
                cluster_id=cluster_id,
                num_samples=num_samples_to_show,
                max_lidar_range=max_lidar_range,
                angle_increment_deg=angle_increment_deg
            )
            input(f"Press Enter to continue to the next cluster...")

    print("End of manual labeling samples. Now you can create your cluster_id_to_label JSON file.")

    # --- 5. T-SNE Visualization of Embeddings (from your original code) ---
    print("Visualizing embeddings with t-SNE...")
    
    # Check if there's enough data for t-SNE (requires at least 2 samples per dimension)
    if scaled_train_embeddings.shape[0] < 2:
        print("Not enough samples for visualization.")
    else:
        # Use PCA for initial dimensionality reduction if embedding size is large
        if embedding_size > 50:
            pca = PCA(n_components=50)
            pca_result = pca.fit_transform(scaled_train_embeddings)
        else:
            pca_result = scaled_train_embeddings

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(pca_result) - 1), learning_rate=200, n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(pca_result)

        # Plot the clusters
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(train_cluster_labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points are colored gray
                plt.scatter(tsne_results[train_cluster_labels == label, 0],
                            tsne_results[train_cluster_labels == label, 1],
                            c='gray', s=10, alpha=0.5, label=f'Noise (Cluster -1)')
            else:
                # Other clusters are colored
                plt.scatter(tsne_results[train_cluster_labels == label, 0],
                            tsne_results[train_cluster_labels == label, 1],
                            c=[colors[i]], s=10, label=f'Cluster {label}')

        plt.title('HDBSCAN Clustering of Autoencoder Embeddings (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = f"encoder_weights/{training_data_name}/hdbscan_clusters_{training_data_name}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"Cluster plot saved to {plot_path}")

    # Clean up rclpy
    rclpy.shutdown()