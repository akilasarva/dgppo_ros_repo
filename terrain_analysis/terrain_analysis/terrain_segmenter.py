import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np

# Terrain IDs: Road=0, Grass=1, Sidewalk=2
def get_current_terrain(seg_label_map):
    H, W = seg_label_map.shape
    crop = seg_label_map[int(0.8 * H):, W // 3 : 2 * W // 3]
    return int(np.bincount(crop.ravel()).argmax())

class TerrainSegmenter(Node):
    def __init__(self):
        super().__init__('terrain_segmenter')
        
        # ROS 2 Setup
        self.bridge = CvBridge()
        
        # Subscribe to your ZED camera topic
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/color/rect/image',
            #'/carla/ego_vehicle/rgb_front/image',
            #'/hamilton/hamilton_zed/rgb/image_rect_color',
            self.image_callback,
            10)
            
        # Publish the new semantic mask
        self.publisher = self.create_publisher(
            Image,
            '/segmentor_image',
            #'/carla/ego_vehicle/image_segmented',
            #'/hamilton/hamilton_zed/rgb/image_segmented',
            10)

        self.terrain_publisher = self.create_publisher(Int32, '/current_terrain', 10)

        self.get_logger().info('HSV Terrain Segmenter Started - No AI Lag')

    def image_callback(self, msg):
        # 1. Convert ROS Image to OpenCV BGR
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return

        # 2. Convert to HSV for better color separation
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # 3. Define Semantic Thresholds (Adjust these if your bag is dark/bright)
        
        # GRASS (Green Range)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # ROAD (Dark Grey / Purple-ish Asphalt)
        # We look for low saturation and low-mid value
        lower_road = np.array([0, 0, 0])
        upper_road = np.array([180, 50, 70])

        # SIDEWALK (Lighter Grey / Concrete)
        # Higher value (brightness) than the road
        lower_sidewalk = np.array([0, 0, 71])
        upper_sidewalk = np.array([180, 40, 210])

        # 4. Create Masks
        mask_grass = cv2.inRange(hsv, lower_green, upper_green)
        mask_road = cv2.inRange(hsv, lower_road, upper_road)
        mask_sidewalk = cv2.inRange(hsv, lower_sidewalk, upper_sidewalk)

        # 5. Build integer label map (Road=0, Grass=1, Sidewalk=2)
        seg_label_map = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        seg_label_map[mask_road > 0] = 0
        seg_label_map[mask_sidewalk > 0] = 2
        seg_label_map[mask_grass > 0] = 1

        # 5b. Publish current terrain ID from bottom-center crop
        terrain_id = get_current_terrain(seg_label_map)
        self.terrain_publisher.publish(Int32(data=terrain_id))

        # 5c. Create the Output Semantic Image
        # Initialize as black
        seg_img = np.zeros_like(cv_img)

        # Paint the classes
        seg_img[mask_road > 0] = [128, 0, 128]      # Purple (Road)
        seg_img[mask_sidewalk > 0] = [190, 190, 190] # Grey (Sidewalk)
        seg_img[mask_grass > 0] = [0, 255, 0]        # Green (Grass)

        # 6. Optional: Add a slight blur to clean up "salt and pepper" noise
        seg_img = cv2.medianBlur(seg_img, 5)

        # 7. Draw terrain crop region and label on the segmentation image
        H, W = seg_img.shape[:2]
        x1, x2 = W // 3, 2 * W // 3
        y1, y2 = int(0.8 * H), H
        cv2.rectangle(seg_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        terrain_names = {0: 'Road', 1: 'Grass', 2: 'Sidewalk'}
        cv2.putText(seg_img, terrain_names.get(terrain_id, '?'), (x1 + 4, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 8. Convert back to ROS and Publish
        out_msg = self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TerrainSegmenter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
