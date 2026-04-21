import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Terrain IDs: Road=0, Grass=1, Sidewalk=2
def get_current_terrain(seg_label_map):
    H, W = seg_label_map.shape
    crop = seg_label_map[int(0.8 * H):, W // 3 : 2 * W // 3]
    return int(np.bincount(crop.ravel()).argmax())

class TerrainSegmenter(Node):
    def __init__(self):
        super().__init__('terrain_segmenter')
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # 1. Load SegFormer (NVIDIA Cityscapes fine-tuned)
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        #self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
        #self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
        self.model.to(self.device).eval()

        # 2. ROS Pub/Sub
        self.sub = self.create_subscription(Image, '/zed/zed_node/rgb/color/rect/image', self.callback, 10)
        self.pub = self.create_publisher(Image, '/terrain_segmented', 10)
        self.terrain_publisher = self.create_publisher(Int32, '/current_terrain', 10)
        self.get_logger().info("SegFormer Terrain Node Ready!")

    def callback(self, msg):
        try:
            # Convert ROS to CV
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # 3. Inference
            inputs = self.processor(images=rgb_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

            # 4. Upscale logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=rgb_img.shape[:2], mode='bilinear', align_corners=False
            )
            pred = upsampled_logits.argmax(1).squeeze().cpu().numpy()

            # 5. Map Cityscapes IDs to Terrain Colors
            # 0: Road, 1: Sidewalk, 8: Vegetation (Grass), 11: Person
            h, w = pred.shape
            seg_img = np.zeros((h, w, 3), dtype=np.uint8)

            seg_img[pred == 0] = [128, 64, 128]   # Purple (Road)
            seg_img[pred == 1] = [244, 35, 232]   # Pink (Sidewalk)
            seg_img[pred == 8] = [0, 255, 0]       # Green (Grass)
            seg_img[pred == 11] = [0, 0, 255]      # Red (Person)

            # 6. Build terrain label map and publish current terrain ID
            seg_label_map = np.zeros((h, w), dtype=np.uint8)
            seg_label_map[pred == 0] = 0  # Road
            seg_label_map[pred == 8] = 1  # Grass
            seg_label_map[pred == 1] = 2  # Sidewalk

            terrain_id = get_current_terrain(seg_label_map)
            self.terrain_publisher.publish(Int32(data=terrain_id))

            # 7. Draw terrain crop region and label on the segmentation image
            x1, x2 = w // 3, 2 * w // 3
            y1, y2 = int(0.8 * h), h
            cv2.rectangle(seg_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            terrain_names = {0: 'Road', 1: 'Grass', 2: 'Sidewalk'}
            cv2.putText(seg_img, terrain_names.get(terrain_id, '?'), (x1 + 4, y1 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Publish the colored mask
            out_msg = self.bridge.cv2_to_imgmsg(seg_img, encoding='rgb8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Segmenter Error: {e}')

def main():
    rclpy.init()
    rclpy.spin(TerrainSegmenter())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
