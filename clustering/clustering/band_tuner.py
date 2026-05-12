#!/usr/bin/env python3
"""
Interactive Z-band parameter tuner for ROS2 LiDAR bags.

Loads frames from a bag, shows the full cloud in grey and the filtered
band in cyan, with live sliders for z_lower/z_upper and min/max range.

Usage:
    python3 band_tuner.py <bag_path> [--topic /livox/lidar] [--max-frames 200]

bag_path can be the bag directory (containing metadata.yaml) or a .db3 file
(its parent directory is used automatically).
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


# ---------------------------------------------------------------------------
# Bag loading
# ---------------------------------------------------------------------------

def _storage_id(bag_dir):
    meta = os.path.join(bag_dir, "metadata.yaml")
    if os.path.exists(meta):
        with open(meta) as f:
            if "mcap" in f.read():
                return "mcap"
    return "sqlite3"


def read_bag_frames(bag_path, topic, max_frames):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py.point_cloud2 import read_points
    except ImportError as e:
        sys.exit(f"Missing ROS2 deps: {e}\nSource your ROS2 workspace first.")

    # rosbag2_py wants the directory, not the .db3 file
    if bag_path.endswith(".db3") and os.path.isfile(bag_path):
        bag_dir = os.path.dirname(bag_path) or "."
    else:
        bag_dir = bag_path

    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=_storage_id(bag_dir))
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, rosbag2_py.ConverterOptions("", ""))

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        sys.exit(
            f"Topic '{topic}' not found.\nAvailable topics:\n  "
            + "\n  ".join(sorted(type_map))
        )

    frames = []
    while reader.has_next() and len(frames) < max_frames:
        t_name, data, _ = reader.read_next()
        if t_name != topic:
            continue
        msg = deserialize_message(data, PointCloud2)
        field_names = [f.name for f in msg.fields]
        has_intensity = "intensity" in field_names
        raw = read_points(
            msg,
            field_names=["x", "y", "z", "intensity"] if has_intensity else ["x", "y", "z"],
            skip_nans=True,
        )
        if len(raw) == 0:
            continue
        if has_intensity:
            pts = np.column_stack([raw["x"], raw["y"], raw["z"], raw["intensity"]])
        else:
            pts = np.column_stack([raw["x"], raw["y"], raw["z"]])
        frames.append(pts.astype(np.float32))

    print(f"Loaded {len(frames)} frames from '{topic}'")
    return frames


# ---------------------------------------------------------------------------
# Point cloud coloring
# ---------------------------------------------------------------------------

_CYAN = np.array([0.10, 0.85, 0.90])
_DARK_GREY = np.array([0.18, 0.18, 0.18])


def _make_pcd(pts, z_lower, z_upper, min_r, max_r, i_min=0.0, i_max=255.0):
    xyz = pts[:, :3]
    z = xyz[:, 2]
    d = np.linalg.norm(xyz[:, :2], axis=1)
    mask = (z >= z_lower) & (z <= z_upper) & (d >= min_r) & (d <= max_r)

    has_intensity = pts.shape[1] >= 4
    if has_intensity:
        mask &= (pts[:, 3] >= i_min) & (pts[:, 3] <= i_max)

    colors = np.tile(_DARK_GREY, (len(pts), 1))
    colors[mask] = _CYAN

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, int(mask.sum())


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class BandTunerApp:
    _PANEL_EM = 22

    def __init__(self, frames):
        self.frames = frames
        self.frame_idx = 0
        self.z_lower = 0.83
        self.z_upper = 1.10
        self.min_r = 1.0
        self.max_r = 25.0
        self.i_min = 0.0
        self.i_max = 255.0
        self._has_intensity = frames[0].shape[1] >= 4
        self._first_frame = True

        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("Band Tuner", 1400, 900)
        w = self.window
        em = w.theme.font_size

        # 3D scene
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.05, 0.05, 0.05, 1.0])
        self._mat = rendering.MaterialRecord()
        self._mat.shader = "defaultUnlit"
        self._mat.point_size = 3.0

        # Control panel
        panel = gui.Vert(0, gui.Margins(em, em, em, em))

        # Frame navigation
        nav = gui.Horiz(0.5 * em)
        btn_prev = gui.Button("< Prev")
        btn_next = gui.Button("Next >")
        btn_prev.set_on_clicked(self._on_prev)
        btn_next.set_on_clicked(self._on_next)
        nav.add_child(btn_prev)
        nav.add_child(btn_next)
        panel.add_child(nav)
        self._lbl_frame = gui.Label(self._frame_label())
        panel.add_child(self._lbl_frame)
        panel.add_fixed(em)

        self._lbl_stats = gui.Label("Band pts: —")
        panel.add_child(self._lbl_stats)
        panel.add_fixed(em)

        # Sliders
        slider_defs = [
            ("z_lower", "Z min",       -5.0,  5.0,   self.z_lower),
            ("z_upper", "Z max",       -5.0,  5.0,   self.z_upper),
            ("min_r",   "Range min",    0.0,  10.0,  self.min_r),
            ("max_r",   "Range max",    1.0,  50.0,  self.max_r),
        ]
        if self._has_intensity:
            slider_defs += [
                ("i_min", "Intensity min", 0.0, 255.0, self.i_min),
                ("i_max", "Intensity max", 0.0, 255.0, self.i_max),
            ]
        self._slider_labels = {}
        for key, label, lo, hi, default in slider_defs:
            lbl = gui.Label(f"{label}: {default:.2f}")
            self._slider_labels[key] = (lbl, label)
            s = gui.Slider(gui.Slider.DOUBLE)
            s.set_limits(lo, hi)
            s.double_value = default
            s.set_on_value_changed(self._make_cb(key))
            panel.add_child(lbl)
            panel.add_child(s)
            panel.add_fixed(0.5 * em)

        w.set_on_layout(self._on_layout)
        w.add_child(self.scene)
        w.add_child(panel)
        self._panel = panel

        self._update()

    def _make_cb(self, key):
        def cb(val):
            setattr(self, key, val)
            lbl, label_text = self._slider_labels[key]
            lbl.text = f"{label_text}: {val:.2f}"
            self._update()
        return cb

    def _frame_label(self):
        return f"Frame {self.frame_idx + 1} / {len(self.frames)}"

    def _on_prev(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self._lbl_frame.text = self._frame_label()
            self._update()

    def _on_next(self):
        if self.frame_idx < len(self.frames) - 1:
            self.frame_idx += 1
            self._lbl_frame.text = self._frame_label()
            self._update()

    def _update(self):
        pts = self.frames[self.frame_idx]
        pcd, n_band = _make_pcd(pts, self.z_lower, self.z_upper, self.min_r, self.max_r,
                                self.i_min, self.i_max)
        total = len(pts)
        self._lbl_stats.text = (
            f"Band pts: {n_band:,} / {total:,}  ({100 * n_band / max(total, 1):.1f}%)"
        )
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("cloud", pcd, self._mat)
        if self._first_frame:
            bounds = pcd.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60, bounds, bounds.get_center())
            self._first_frame = False

    def _on_layout(self, ctx):
        r = self.window.content_rect
        panel_w = self._PANEL_EM * ctx.theme.font_size
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self._panel.frame = gui.Rect(r.x + r.width - panel_w, r.y, panel_w, r.height)

    def run(self):
        gui.Application.instance.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive Z-band tuner for ROS2 LiDAR bags")
    parser.add_argument("bag_path", help="Bag directory or .db3 file path")
    parser.add_argument("--topic", default="/livox/lidar", help="PointCloud2 topic (default: /livox/lidar)")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to load (default: 200)")
    args = parser.parse_args()

    frames = read_bag_frames(args.bag_path, args.topic, args.max_frames)
    if not frames:
        sys.exit("No frames loaded — check topic name and bag path.")

    BandTunerApp(frames).run()


if __name__ == "__main__":
    main()
