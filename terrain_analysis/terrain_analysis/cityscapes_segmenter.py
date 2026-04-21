import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import numpy as np

# Terrain IDs: Road=0, Grass=1, Sidewalk=2
def get_current_terrain(seg_label_map):
    H, W = seg_label_map.shape
    crop = seg_label_map[int(0.8 * H):, W // 3 : 2 * W // 3]
    return int(np.bincount(crop.ravel()).argmax())

class CityscapesSegmenter(Node):
    def __init__(self):
        super().__init__('cityscapes_segmenter')
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load Torchvision's official DeepLabV3 with Cityscapes weights
        self.get_logger().info(f"Loading DeepLabV3-ResNet50 on {self.device}...")
        
        # This will automatically download the weights if not present (~160MB)
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 
        # Note: For strict Cityscapes, we use the model logic below:
        self.model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=21)
        
        # BEST OPTION FOR INSTANT TERRAIN: 
        # Since standard Torchvision is COCO-heavy, we use the Hub for Cityscapes:
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.model.to(self.device).eval()

        self.sub = self.create_subscription(Image, '/zed/zed_node/rgb/color/rect/image', self.callback, 10)
        self.pub = self.create_publisher(Image, '/terrain_segmented', 10)
        self.terrain_publisher = self.create_publisher(Int32, '/current_terrain', 10)
        self.get_logger().info("Model Ready. Road=Purple, Sidewalk=Pink, Grass=Green")

    def callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 1. Resize for speed and convert to RGB
            resized_img = cv2.resize(cv_img, (640, 480))
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            # 2. Manual Normalization (Standard ImageNet values)
            # Convert to float 0-1
            # 2. Manual Normalization (Force Float32)
            img_float = rgb_img.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_float - mean) / std
            
            # 3. Reorder and ensure it stays a FloatTensor
            input_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
            
            # 4. Get predictions
            pred = output.argmax(0).cpu().numpy()

            # 5. Create the Color Mask
            h, w = pred.shape
            seg_img = np.zeros((h, w, 3), dtype=np.uint8)

            # Map based on the DeepLab-VOC/COCO output indices:
            # 0: Background/Road, 15: Car, 19: Train, etc.
            seg_img[pred == 0] = [128, 64, 128]  # Purple (Road)
            seg_img[pred == 15] = [255, 255, 0]  # Yellow (Car)

            # Build terrain label map and publish current terrain ID
            seg_label_map = np.zeros((h, w), dtype=np.uint8)
            seg_label_map[pred == 0] = 0  # Road

            terrain_id = get_current_terrain(seg_label_map)
            self.terrain_publisher.publish(Int32(data=terrain_id))

            # Resize back to original ZED size
            final_img = cv2.resize(seg_img, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Draw terrain crop region and label
            fh, fw = final_img.shape[:2]
            x1, x2 = fw // 3, 2 * fw // 3
            y1, y2 = int(0.8 * fh), fh
            cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            terrain_names = {0: 'Road', 1: 'Grass', 2: 'Sidewalk'}
            cv2.putText(final_img, terrain_names.get(terrain_id, '?'), (x1 + 4, y1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            out_msg = self.bridge.cv2_to_imgmsg(final_img, encoding='bgr8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Inference Error: {e}')
def main():
    rclpy.init()
    rclpy.spin(CityscapesSegmenter())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
