import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import StandardScaler
import os
import hdbscan
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int16  # Import the standard Int16 message type

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

# --- 2. The ROS 2 Node with the corrected load_models function ---
class LiveClusterInferenceNode(Node):
    def __init__(self):
        super().__init__('live_cluster_inference_node')

        # --- Configuration Parameters (must match training script) ---
        training_data_name = "ros2_bigbag_laserscan"
        self.embedding_size = 16
        self.num_ranges = 32
        self.max_lidar_range = 10.0
        
        self.model_save_path = f"encoder_weights/{training_data_name}/lidar_encoder_autoencoder_{training_data_name}.pth"
        self.hdbscan_model_path = f"encoder_weights/{training_data_name}/hdbscan_model_{training_data_name}.pkl"
        self.scaler_path = f"encoder_weights/{training_data_name}/scaler_{training_data_name}.pkl"

        # --- Device Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # --- Load Pre-Trained Models ---
        self.load_models()

        self.subscription = self.create_subscription(
            LaserScan,
            '/my_lidar_scan',
            self.laserscan_callback,
            qos_profile=qos_profile_sensor_data
        )
        # Publisher for the cluster label
        self.publisher_ = self.create_publisher(Int16, '/predicted_cluster', 10)
        self.get_logger().info("Ready for live clustering. Subscribed to '/my_lidar_scan' and publishing to '/predicted_cluster'")

    def load_models(self):
        """Loads the pre-trained Autoencoder, HDBSCAN model, and StandardScaler."""
        
        # Instantiate the encoder and decoder first
        encoder = LidarEncoder(embedding_size=self.embedding_size)
        decoder = LidarDecoder(embedding_size=self.embedding_size, num_ranges=self.num_ranges)
        
        # Pass the instantiated models to the Autoencoder class
        self.autoencoder = Autoencoder(encoder, decoder).to(self.device)
        print(self.autoencoder)

        if os.path.exists(self.model_save_path):
            self.autoencoder.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            self.autoencoder.eval() # Set the model to evaluation mode
            self.get_logger().info(f"Successfully loaded Autoencoder model from {self.model_save_path}")
        else:
            self.get_logger().error(f"Autoencoder model not found at {self.model_save_path}! Exiting.")
            exit()
        
        # Load the HDBSCAN clustering model
        if os.path.exists(self.hdbscan_model_path):
            with open(self.hdbscan_model_path, 'rb') as f:
                self.cluster_model = pickle.load(f)
            self.get_logger().info(f"Successfully loaded HDBSCAN model from {self.hdbscan_model_path}")
        else:
            self.get_logger().error(f"HDBSCAN model not found at {self.hdbscan_model_path}! Exiting.")
            exit()
            
        # Load the StandardScaler
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.get_logger().info(f"Successfully loaded StandardScaler from {self.scaler_path}")
        else:
            self.get_logger().error(f"StandardScaler not found at {self.scaler_path}! Exiting.")
            exit()

    def laserscan_callback(self, msg: LaserScan):
        """Callback for new LaserScan messages. Performs real-time inference."""
        self.get_logger().info(f"Received LaserScan with {len(msg.ranges)} ranges.")

        # Convert ranges to a normalized PyTorch tensor
        ranges_np = np.array(msg.ranges, dtype=np.float32)
        ranges_np[ranges_np > self.max_lidar_range] = self.max_lidar_range
        normalized_ranges = ranges_np / self.max_lidar_range
        input_ranges = torch.tensor(normalized_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.autoencoder.encoder(input_ranges).cpu().numpy()
        
        # Scale embedding using the pre-trained scaler
        scaled_embedding = self.scaler.transform(embedding)
        
        # Predict cluster label using the HDBSCAN model
        predicted_label, _ = hdbscan.approximate_predict(self.cluster_model, scaled_embedding)
        
        # Create and publish the message
        label_msg = Int16()
        label_msg.data = int(predicted_label[0])
        self.publisher_.publish(label_msg)

        self.get_logger().info(f"Published Predicted Cluster: {label_msg.data}")

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
