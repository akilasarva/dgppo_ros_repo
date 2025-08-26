import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import StandardScaler
import os
import hdbscan
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int16, Float32MultiArray # <--- Import new message type

# Assuming lidar_processor.py is in the same directory
# This function must be identical to the one used in the training script.
from .lidar_processor import get_ranges_from_points

# --- Neural Network Models (Unchanged) ---
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

# ----------------------------------------------------------------------

class LiveClusterInferenceNode(Node):
    def __init__(self):
        super().__init__('live_cluster_inference_node')

        # --- Configuration Parameters (must match training script) ---
        training_data_name = "bridge2_carla"
        self.embedding_size = 16
        self.num_ranges = 32
        self.max_lidar_range = 20.0
        self.angle_increment_deg = float(360.0 / self.num_ranges)
        
        # Multiple altitude slices to match training data preprocessing
        self.z_threshold_upper = 2.25
        self.z_threshold_lower = 2
        self.z_threshold_upper_2 = 0
        self.z_threshold_lower_2 = 0

        # Anomaly detection threshold (tune this!)
        # Set this value to the 95th percentile from your training script's noise_distance_distribution.png
        self.distance_threshold_for_reassignment = 4.5 # <<< REPLACE THIS WITH YOUR VALUE

        # --- NEW: Define the config dictionary for the shared preprocessing function ---
        self.config = {
            "num_ranges": self.num_ranges,
            "max_lidar_range": self.max_lidar_range,
            "angle_increment_deg": self.angle_increment_deg,
            "z_threshold_upper": self.z_threshold_upper,
            "z_threshold_lower": self.z_threshold_lower,
            "z_threshold_upper_2": self.z_threshold_upper_2,
            "z_threshold_lower_2": self.z_threshold_lower_2
        }
        
        path_prefix = ""

        self.model_save_path = f"encoder_weights/{training_data_name}/lidar_encoder_autoencoder_{training_data_name}.pth"
        self.hdbscan_model_path = f"encoder_weights/{training_data_name}/hdbscan_model_{training_data_name}.pkl"
        self.scaler_path = f"encoder_weights/{training_data_name}/scaler_{training_data_name}.pkl"
        self.cluster_centroids_path = f"encoder_weights/{training_data_name}/cluster_centroids_{training_data_name}.pkl"

        # --- Device Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- Load Pre-Trained Models and Centroids ---
        self.load_models()

        # --- Subscription and Publishing ---
        self.subscription = self.create_subscription(
            PointCloud2,
            '/carla/ego_vehicle/lidar',
            self.pointcloud_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.publisher_ = self.create_publisher(Int16, '/predicted_cluster', 10)
        self.publisher_ranges = self.create_publisher(Float32MultiArray, '/processed_ranges', 10)

        self.get_logger().info("Ready for live clustering. Subscribed to '/warthog1/sensors/ouster/points' and publishing to '/predicted_cluster'")

    def load_models(self):
        """Loads the pre-trained Autoencoder, HDBSCAN model, StandardScaler, and Cluster Centroids."""
        encoder = LidarEncoder(embedding_size=self.embedding_size)
        decoder = LidarDecoder(embedding_size=self.embedding_size, num_ranges=self.num_ranges)
        self.autoencoder = Autoencoder(encoder, decoder).to(self.device)

        if os.path.exists(self.model_save_path):
            self.autoencoder.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            self.autoencoder.eval()
            self.get_logger().info(f"Successfully loaded Autoencoder model from {self.model_save_path}")
        else:
            self.get_logger().error(f"Autoencoder model not found at {self.model_save_path}! Exiting.")
            exit()
        
        if os.path.exists(self.hdbscan_model_path):
            with open(self.hdbscan_model_path, 'rb') as f:
                self.cluster_model = pickle.load(f)
            self.get_logger().info(f"Successfully loaded HDBSCAN model from {self.hdbscan_model_path}")
        else:
            self.get_logger().error(f"HDBSCAN model not found at {self.hdbscan_model_path}! Exiting.")
            exit()
            
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.get_logger().info(f"Successfully loaded StandardScaler from {self.scaler_path}")
        else:
            self.get_logger().error(f"StandardScaler not found at {self.scaler_path}! Exiting.")
            exit()

        if os.path.exists(self.cluster_centroids_path):
            with open(self.cluster_centroids_path, 'rb') as f:
                self.cluster_centroids = pickle.load(f)
            self.get_logger().info(f"Successfully loaded cluster centroids from {self.cluster_centroids_path}")
        else:
            self.get_logger().error(f"Cluster centroids not found at {self.cluster_centroids_path}! Please re-run the training script with logic to save them. Exiting.")
            exit()
            
    def reassign_labels(self, hdbscan_label, embedding):
        """
        Reassigns a label if HDBSCAN classified it as noise.
        """
        if hdbscan_label != -1:
            return hdbscan_label # No change for non-noise points

        if not self.cluster_centroids:
            return -1 # Cannot reassign if no centroids are loaded

        distances_to_all_centroids = {
            cid: np.linalg.norm(embedding - centroid)
            for cid, centroid in self.cluster_centroids.items()
        }

        closest_known_cluster_id = min(distances_to_all_centroids, key=distances_to_all_centroids.get)
        min_dist = distances_to_all_centroids[closest_known_cluster_id]

        if min_dist < self.distance_threshold_for_reassignment:
            # self.get_logger().info(f"Reassigning noise point to cluster {closest_known_cluster_id} with distance {min_dist:.2f}")
            return closest_known_cluster_id
        else:
            # self.get_logger().info(f"Classifying point as a new cluster/outlier (ID -2) with distance {min_dist:.2f}")
            return -2

    def pointcloud_callback(self, msg: PointCloud2):
        """Callback for new PointCloud2 messages. Processes the data and performs inference."""
        # --- REMOVED: Performance-intensive logging ---
        # self.get_logger().info(f"Received PointCloud2 with {msg.width * msg.height} points.")

        try:
            points_gen = read_points(msg, field_names=("x", "y", "z"))
            points_structured = np.asarray(list(points_gen), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
        except Exception as e:
            self.get_logger().error(f"Error reading points from PointCloud2: {e}")
            return
        
        if points_structured.shape[0] == 0:
            # self.get_logger().warn("Received empty PointCloud2 message. Publishing -1 (Noise).")
            label_msg = Int16()
            label_msg.data = -1
            self.publisher_.publish(label_msg)
            return

        points_np = np.vstack([points_structured['x'], points_structured['y'], points_structured['z']]).T
        
        # Use the unified pre-processing function to generate the 1D range array
        ranges_binned = get_ranges_from_points(points_np, self.config)
        
        if np.all(ranges_binned == self.config['max_lidar_range']):
            # self.get_logger().warn("All ranges are at max_lidar_range. Publishing -1 (Noise).")
            label_msg = Int16()
            label_msg.data = -1
            self.publisher_.publish(label_msg)
            return
        
        ranges_msg = Float32MultiArray()
        ranges_msg.data = ranges_binned.tolist()
        self.publisher_ranges.publish(ranges_msg)

        normalized_ranges = ranges_binned / self.max_lidar_range
        input_ranges = torch.tensor(normalized_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.autoencoder.encoder(input_ranges).cpu().numpy()
        
        scaled_embedding = self.scaler.transform(embedding)
        
        predicted_label, _ = hdbscan.approximate_predict(self.cluster_model, scaled_embedding)
        
        final_label = self.reassign_labels(int(predicted_label[0]), scaled_embedding[0])
        
        label_msg = Int16()
        label_msg.data = int(final_label)
        self.publisher_.publish(label_msg)

        # --- REMOVED: Performance-intensive logging ---
        # self.get_logger().info(f"Published Predicted Cluster: {label_msg.data}")
        
def main(args=None):
    rclpy.init(args=args)
    inference_node = LiveClusterInferenceNode()
    try:
        rclpy.spin(inference_node)
    except KeyboardInterrupt:
        pass
    inference_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
