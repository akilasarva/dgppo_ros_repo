import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA
import numpy as np


def _make_colored_cloud(xyz, colors_float, header):
    """Build a PointCloud2 with x,y,z,rgb from (N,3) xyz and (N,3) [0,1] RGB."""
    n_pts = len(xyz)
    r = (np.clip(colors_float[:, 0], 0, 1) * 255).astype(np.uint32)
    g = (np.clip(colors_float[:, 1], 0, 1) * 255).astype(np.uint32)
    b = (np.clip(colors_float[:, 2], 0, 1) * 255).astype(np.uint32)
    rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)

    data = np.zeros(n_pts, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    data['x'] = xyz[:, 0]
    data['y'] = xyz[:, 1]
    data['z'] = xyz[:, 2]
    data['rgb'] = rgb_packed

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n_pts
    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n_pts
    msg.data = data.tobytes()
    msg.is_dense = True
    return msg


class BandVizNode(Node):
    def __init__(self):
        super().__init__('band_viz_node')

        self.declare_parameter('input_topic',               '/livox/lidar')
        self.declare_parameter('z_ground_lower',             0.83)
        self.declare_parameter('z_ground_upper',             1.10)
        self.declare_parameter('structure_z_lower',          0.85)
        self.declare_parameter('structure_z_upper',          1.20)
        self.declare_parameter('min_lidar_range',            1.0)
        self.declare_parameter('max_lidar_range',            25.0)
        self.declare_parameter('num_ranges',                 72)
        self.declare_parameter('max_intensity',              255.0)
        self.declare_parameter('intensity_change_threshold', 0.10)

        self.cfg = {
            'z_ground_lower':             self.get_parameter('z_ground_lower').value,
            'z_ground_upper':             self.get_parameter('z_ground_upper').value,
            'structure_z_lower':          self.get_parameter('structure_z_lower').value,
            'structure_z_upper':          self.get_parameter('structure_z_upper').value,
            'min_lidar_range':            self.get_parameter('min_lidar_range').value,
            'max_lidar_range':            self.get_parameter('max_lidar_range').value,
            'num_ranges':                 self.get_parameter('num_ranges').value,
            'max_intensity':              self.get_parameter('max_intensity').value,
            'intensity_change_threshold': self.get_parameter('intensity_change_threshold').value,
        }
        topic = self.get_parameter('input_topic').value

        self.sub = self.create_subscription(
            PointCloud2, topic, self._cloud_cb, qos_profile_sensor_data)
        self.pub_cloud   = self.create_publisher(PointCloud2,  '/band_viz/points',  10)
        self.pub_markers = self.create_publisher(MarkerArray,  '/band_viz/markers', 10)
        self._first_msg = True

        self.get_logger().info(f'band_viz_node ready — subscribing to {topic}')

    def _cloud_cb(self, msg: PointCloud2):
        try:
            self._process(msg)
        except Exception as e:
            self.get_logger().error(f'callback error: {e}', throttle_duration_sec=2.0)

    def _process(self, msg: PointCloud2):
        field_names = [f.name for f in msg.fields]
        if self._first_msg:
            self.get_logger().info(f'first message received — fields: {field_names}  frame: {msg.header.frame_id}  pts: {msg.width * msg.height}')
            self._first_msg = False
        has_intensity = 'intensity' in field_names
        read_fields = ['x', 'y', 'z', 'intensity'] if has_intensity else ['x', 'y', 'z']

        # read_points returns a structured numpy array in ROS2 Humble+
        raw = read_points(msg, field_names=read_fields, skip_nans=True)
        if len(raw) == 0:
            return
        if has_intensity:
            pts = np.column_stack(
                [raw['x'], raw['y'], raw['z'], raw['intensity']]
            ).astype(np.float32)
        else:
            pts = np.column_stack(
                [raw['x'], raw['y'], raw['z']]
            ).astype(np.float32)

        cfg = self.cfg
        n          = cfg['num_ranges']
        inc        = 360.0 / n
        tol        = np.deg2rad(inc / 2)
        max_i      = cfg['max_intensity']
        threshold  = cfg['intensity_change_threshold']
        max_r      = cfg['max_lidar_range']

        colors = np.ones((len(pts), 3)) * 0.15   # dark grey default
        z = pts[:, 2]

        # --- structure band: cyan ---
        struct_mask = (np.abs(z) >= cfg['structure_z_lower']) & (np.abs(z) <= cfg['structure_z_upper'])
        colors[struct_mask] = [0.2, 0.8, 0.9]

        # --- ground band: green/red intensity classification ---
        gnd_mask     = (np.abs(z) >= cfg['z_ground_lower']) & (np.abs(z) <= cfg['z_ground_upper'])
        gnd_pts_full = pts[gnd_mask]
        transition_hits = []

        if gnd_pts_full.size > 0:
            d_full   = np.linalg.norm(gnd_pts_full[:, :2], axis=1)
            rad_mask = (d_full >= cfg['min_lidar_range']) & (d_full <= max_r)
            gnd_pts  = gnd_pts_full[rad_mask]
            dists    = d_full[rad_mask]
            orig_idx = np.where(gnd_mask)[0][rad_mask]

            if has_intensity and gnd_pts.size > 0:
                angles = np.arctan2(gnd_pts[:, 1], gnd_pts[:, 0])

                # Pass 1: reference intensity via histogram mode
                closest_intensities = []
                for i in range(n):
                    angle_rad = np.deg2rad(i * inc)
                    diff = np.arctan2(np.sin(angles - angle_rad), np.cos(angles - angle_rad))
                    mask = np.abs(diff) <= tol
                    if mask.any():
                        ci = np.argmin(dists[mask])
                        closest_intensities.append(gnd_pts[mask][ci, 3] / max_i)

                if closest_intensities:
                    arr = np.array(closest_intensities)
                    counts, edges = np.histogram(arr, bins=10, range=(0.0, 1.0))
                    mode_bin  = np.argmax(counts)
                    reference = float((edges[mode_bin] + edges[mode_bin + 1]) / 2.0)

                    i_norm       = gnd_pts[:, 3] / max_i
                    is_transition = np.abs(i_norm - reference) > threshold
                    colors[orig_idx[~is_transition]] = [0.2, 0.8, 0.2]  # green
                    colors[orig_idx[is_transition]]  = [0.9, 0.2, 0.1]  # red

                    # Pass 2: per-sector first transition hit (row 1)
                    for i in range(n):
                        angle_rad = np.deg2rad(i * inc)
                        diff = np.arctan2(np.sin(angles - angle_rad), np.cos(angles - angle_rad))
                        mask = np.abs(diff) <= tol
                        if mask.any():
                            sd = dists[mask]
                            sp = gnd_pts[mask]
                            for idx in np.argsort(sd):
                                if abs(sp[idx, 3] / max_i - reference) > threshold:
                                    transition_hits.append(sp[idx, :3])
                                    break
            else:
                colors[orig_idx] = [0.2, 0.8, 0.2]

        # --- structure range hits per sector (row 0) ---
        structure_range_hits = []
        s_pts_full = pts[struct_mask]
        if s_pts_full.size > 0:
            sd_full = np.linalg.norm(s_pts_full[:, :2], axis=1)
            s_rad   = (sd_full >= cfg['min_lidar_range']) & (sd_full <= max_r)
            s_pts   = s_pts_full[s_rad]
            s_dists = sd_full[s_rad]
            if s_pts.size > 0:
                s_angles = np.arctan2(s_pts[:, 1], s_pts[:, 0])
                for i in range(n):
                    angle_rad = np.deg2rad(i * inc)
                    diff = np.arctan2(np.sin(s_angles - angle_rad), np.cos(s_angles - angle_rad))
                    mask = np.abs(diff) <= tol
                    if mask.any():
                        ci = np.argmin(s_dists[mask])
                        structure_range_hits.append(s_pts[mask][ci, :3])

        # --- publish colored cloud ---
        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        self.pub_cloud.publish(_make_colored_cloud(pts[:, :3], colors, header))

        # --- publish markers ---
        markers = MarkerArray()
        origin  = Point(x=0.0, y=0.0, z=0.0)

        # Blue-to-cyan gradient rays: structure range (row 0)
        if structure_range_hits:
            hits_arr = np.array(structure_range_hits)
            z_vals   = hits_arr[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            t_arr = (z_vals - z_min) / (z_max - z_min) if z_max > z_min else np.zeros(len(z_vals))

            m = Marker()
            m.header    = header
            m.ns        = 'structure_rays'
            m.id        = 0
            m.type      = Marker.LINE_LIST
            m.action    = Marker.ADD
            m.scale.x   = 0.03
            m.pose.orientation.w = 1.0
            for i, hit in enumerate(structure_range_hits):
                t = float(t_arr[i])
                c = ColorRGBA(r=0.0, g=0.2 + 0.7 * t, b=0.9 + 0.1 * t, a=1.0)
                m.points += [origin, Point(x=float(hit[0]), y=float(hit[1]), z=float(hit[2]))]
                m.colors += [c, c]
            markers.markers.append(m)

        # Yellow rays: intensity_edge transition hits (row 1)
        if transition_hits:
            m = Marker()
            m.header    = header
            m.ns        = 'intensity_rays'
            m.id        = 1
            m.type      = Marker.LINE_LIST
            m.action    = Marker.ADD
            m.scale.x   = 0.03
            m.color     = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            m.pose.orientation.w = 1.0
            for hit in transition_hits:
                m.points += [origin, Point(x=float(hit[0]), y=float(hit[1]), z=float(hit[2]))]
            markers.markers.append(m)

            # Orange POINTS markers at transition hits
            m2 = Marker()
            m2.header   = header
            m2.ns       = 'transition_hits'
            m2.id       = 2
            m2.type     = Marker.POINTS
            m2.action   = Marker.ADD
            m2.scale.x  = 0.08
            m2.scale.y  = 0.08
            m2.color    = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            m2.pose.orientation.w = 1.0
            for hit in transition_hits:
                m2.points.append(Point(x=float(hit[0]), y=float(hit[1]), z=float(hit[2])))
            markers.markers.append(m2)

        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = BandVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
