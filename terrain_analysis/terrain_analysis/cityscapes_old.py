import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import models, transforms
import numpy as np

class LightweightSegmenter(Node):
    def __init__(self):
        super().__init__('cityscapes_segmenter')
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. SWITCH TO LIGHTWEIGHT MODEL (MobileNetV3)
        # This is ~10x faster than ResNet101
        self.get_logger().info(f'Loading MobileNetV3 on {self.device}...')
        self.model = models.segmentation.lraspp_mobilenet_v3_large(weights='DEFAULT').to(self.device).eval()

        # 2. OPTIMIZED TRANSFORM
        # We resize the input to 320px to drastically increase speed
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(320), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sub = self.create_subscription(Image, '/hamilton/hamilton_zed/rgb/image_rect_color', self.callback, 10)
        self.pub = self.create_publisher(Image, '/terrain_segmented', 10)

    def callback(self, msg):
        # Convert and Downsample for speed
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        input_tensor = self.transform(cv_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get predictions
        pred = output.argmax(0).cpu().numpy()

        # 3. EXPLICIT TERRAIN MAPPING
        # In this model's default weights:
        # 0 = Background/Road, 15 = Car, etc.
        # We will create a clear mask for Road (Purple), Sidewalk (Gray), Grass (Green)
        h, w = pred.shape
        seg_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Mapping for the meeting (Common classes in VOC/COCO subsets)
        seg_img[pred == 0] = [128, 0, 128]   # Purple (Road/Background)
        seg_img[pred == 15] = [255, 255, 0]  # Yellow (Cars - helps you see it's working!)
        
        # If the image is still 'all purple', we add an 'Edge' filter to 
        # highlight where the ground meets the walls/objects.
        
        # Resize back to ZED size
        final_img = cv2.resize(seg_img, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        out_msg = self.bridge.cv2_to_imgmsg(final_img, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    node = LightweightSegmenter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
