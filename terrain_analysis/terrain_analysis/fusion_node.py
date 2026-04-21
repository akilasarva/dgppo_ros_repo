import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np

class TerrainFusion(Node):
    def __init__(self):
        super().__init__('terrain_fusion')
        self.bridge = CvBridge()
        self.latest_mask = None
        
        # Subscriptions
        self.create_subscription(Image, '/terrain_segmented', self.mask_cb, 10)
        self.create_subscription(Image, '/hamilton/hamilton_zed/depth/depth_registered', self.depth_cb, 10)
        
        # Publisher
        self.pc_pub = self.create_publisher(PointCloud2, '/terrain_cloud_3d', 10)
        
        # ZED Camera Intrinsics (Standard 720p values - adjust if your bag is different)
        self.fx, self.fy = 525.0, 525.0 
        self.cx, self.cy = 320.0, 240.0

    def mask_cb(self, msg):
        self.latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_cb(self, msg):
        if self.latest_mask is None:
            return

        # 1. Convert Depth and Resize Mask
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        mask = cv2.resize(self.latest_mask, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. Vectorized 3D Projection
        h, w = depth_img.shape
        i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Filter for valid depth
        valid = (depth_img > 0.1) & (depth_img < 15.0) & (~np.isnan(depth_img))
        
        z = depth_img[valid]
        x = (j[valid] - self.cx) * z / self.fx
        y = (i[valid] - self.cy) * z / self.fy
        
        # 3. Pack Colors (BGR to RGB packed into a single float/uint32)
        colors = mask[valid]
        r = colors[:, 2].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 0].astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b

        # 4. Create Structured Array for the PointCloud
        data = np.zeros(len(z), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        data['x'] = x
        data['y'] = y
        data['z'] = z
        data['rgb'] = rgb

        # 5. Correct fields for create_cloud
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Publish
        cloud_msg = pc2.create_cloud(msg.header, fields, data)
        self.pc_pub.publish(cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TerrainFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
