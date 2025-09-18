import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import pathlib
import functools as ft
import jax.tree_util as jtu
from functools import partial
import jax.debug as jdebug

from colour import hsl2hex
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import List, Optional, Union, Tuple, Dict

class BehaviorAssociator:
    def __init__(self, bridges, buildings, obstacles, region_name_to_id: Optional[Dict[str, int]] = None, region_id_to_name: Optional[Dict[int, str]] = None):
        self.bridges = bridges
        self.buildings = buildings
        self.obstacles = obstacles

        self.behavior_regions = self._define_behavior_regions()

        if region_name_to_id is None:
            self.region_name_to_id = {name: i for i, name in enumerate(sorted(self.behavior_regions.keys()))}
        else:
            self.region_name_to_id = region_name_to_id

        if region_id_to_name is None:
            self.region_id_to_name = {v: k for k, v in self.region_name_to_id.items()}
        else:
            self.region_id_to_name = region_id_to_name


        self.sorted_region_names = sorted(self.region_name_to_id, key=self.region_name_to_id.get)

        temp_polygons = []
        temp_centroids = []
        temp_labels = []
        temp_colors = []

        world_boundary_coords = jnp.array([(-10, -10), (60, -10), (60, 50), (-10, 50)], dtype=jnp.float32)

        self.MAX_POLYGON_VERTICES = 4
        for name, coords in self.behavior_regions.items():
            if isinstance(coords, jnp.ndarray) and coords.ndim == 2:
                self.MAX_POLYGON_VERTICES = max(self.MAX_POLYGON_VERTICES, coords.shape[0])

        for name in self.sorted_region_names:            
            coords = self.behavior_regions.get(name)

            if coords is not None:
                if isinstance(coords, jnp.ndarray) and coords.ndim == 2: # Polygon
                    temp_polygons.append(coords)
                    temp_centroids.append(jnp.mean(coords, axis=0))
                    temp_labels.append(name)
                    if name.startswith("on_bridge"): temp_colors.append('skyblue')
                    elif name.startswith("approach_bridge"): temp_colors.append('lightgreen')
                    elif name.startswith("exit_bridge"): temp_colors.append('lightcoral')
                    elif name == "open_space": temp_colors.append('lightgray')
                    # elif name == "World Boundary": temp_colors.append('lightgray')
                    else: temp_colors.append('blue')
                elif isinstance(coords, tuple) and coords[0] == "circle": # Circle
                    center_x, center_y, radius = coords[1]
                    temp_polygons.append(jnp.array([[center_x, center_y]], dtype=jnp.float32))
                    temp_centroids.append(jnp.array([center_x, center_y], dtype=jnp.float32))
                    temp_labels.append(name)
                    temp_colors.append('red')

        padded_polygons_for_stack = []
        for p in temp_polygons:
            num_actual_vertices = p.shape[0]
            if p.ndim == 1:
                p = p[jnp.newaxis, :]
                num_actual_vertices = 1
            padded_p = jnp.pad(p, ((0, self.MAX_POLYGON_VERTICES - num_actual_vertices), (0, 0)), mode='constant', constant_values=0.0)
            padded_polygons_for_stack.append(padded_p)


        self.all_region_polygons = jnp.stack(padded_polygons_for_stack, axis=0)
        self.all_region_centroids_jax_array = jnp.stack(temp_centroids, axis=0)

        self.region_labels = temp_labels
        self.region_colors = temp_colors

    def _get_rectangle_corners_np(self, center, length, width, angle_rad):
        half_l = length / 2.0
        half_w = width / 2.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        local_corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])

        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        rotated_corners = np.dot(local_corners, rotation_matrix.T)
        final_corners = rotated_corners + center
        return final_corners

    def _define_behavior_regions(self):
        regions = {}

        regions["open_space"] = jnp.array([(-10, -10), (60, -10), (60, 50), (-10, 50)], dtype=jnp.float32)
        individual_bridge_corners = []
        
        for i, (cx, cy, l, w, a) in enumerate(self.bridges): 
            rad = jnp.radians(a) # Angle in radians for trigonometric functions
            
            half_l = l / 2.0
            half_w = w / 2.0
            
            local_corners = jnp.array([
                [-half_l, -half_w], # Bottom-left
                [ half_l, -half_w], # Bottom-right
                [ half_l,  half_w], # Top-right
                [-half_l,  half_w]  # Top-left
            ], dtype=jnp.float32)

            # 2. Create the rotation matrix
            cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)
            rotation_matrix = jnp.array([
                [cos_a, -sin_a],
                [sin_a,  cos_a]
            ])

            rotated_corners_local = jnp.einsum('ij,kj->ki', rotation_matrix, local_corners)
            
            # 4. Translate the rotated corners to the actual center (cx, cy)
            corners_obstacle = rotated_corners_local + jnp.array([cx, cy])
            
            individual_bridge_corners.append(corners_obstacle)

        if len(self.bridges) >= 2:
            corners_obs1 = individual_bridge_corners[0]
            corners_obs2 = individual_bridge_corners[1]

            on_bridge_p1 = corners_obs1[3] # Corner of obs1 closer to origin on its width side
            on_bridge_p2 = corners_obs1[2] # Corner of obs1 further from origin on its width side
            on_bridge_p3 = corners_obs2[1] # Corner of obs2 further from origin on its width side
            on_bridge_p4 = corners_obs2[0] # Corner of obs2 closer to origin on its width side
            
            regions[f"on_bridge_0"] = jnp.array([
                on_bridge_p1,
                on_bridge_p4,
                on_bridge_p3,
                on_bridge_p2
            ], dtype=jnp.float32)

            # --- APPROACH BRIDGE 0 ---
            channel_entrance_p1 = on_bridge_p1 
            channel_entrance_p2 = on_bridge_p4 

            actual_center_channel_approach_arc = (channel_entrance_p1 + channel_entrance_p2) / 2.0
            channel_width = jnp.linalg.norm(on_bridge_p1 - on_bridge_p4)
            semi_circle_radius_channel = channel_width / 2.0
            
            num_points = 30
            
            # Use a1 for rotation as it was in the original, or take from self.bridges[0]
            # Ensure this 'a' is the angle of the bridge itself, not a derived one for the arc.
            # Assuming 'a' for the first bridge wall defines the orientation of the gap.
            rad_for_arcs = jnp.radians(self.bridges[0][4]) 
            
            angles_approach_arc_channel = jnp.linspace(rad_for_arcs + jnp.pi / 2, rad_for_arcs + 3 * jnp.pi / 2, num_points)
            semi_circle_approach_points_channel = jnp.array([
                actual_center_channel_approach_arc[0] + semi_circle_radius_channel * jnp.cos(angles_approach_arc_channel),
                actual_center_channel_approach_arc[1] + semi_circle_radius_channel * jnp.sin(angles_approach_arc_channel)
            ]).T
            
            # Note: original had .tolist()[::-1] which reverses. JAX needs explicit reversal.
            regions[f"approach_bridge_0"] = jnp.concatenate([
                jnp.array([channel_entrance_p2, channel_entrance_p1]),
                semi_circle_approach_points_channel[::-1] # Reverse explicitly for JAX
            ], axis=0)

            # --- EXIT BRIDGE 0 ---
            channel_exit_p1 = on_bridge_p2 
            channel_exit_p2 = on_bridge_p3 

            actual_center_channel_exit_arc = (channel_exit_p1 + channel_exit_p2) / 2.0

            angles_exit_arc_channel = jnp.linspace(rad_for_arcs - jnp.pi / 2, rad_for_arcs + jnp.pi / 2, num_points)
            semi_circle_exit_points_channel = jnp.array([
                actual_center_channel_exit_arc[0] + semi_circle_radius_channel * jnp.cos(angles_exit_arc_channel),
                actual_center_channel_exit_arc[1] + semi_circle_radius_channel * jnp.sin(angles_exit_arc_channel)
            ]).T
            
            regions[f"exit_bridge_0"] = jnp.concatenate([
                jnp.array([channel_exit_p2, channel_exit_p1]),
                semi_circle_exit_points_channel[::-1] # Reverse explicitly for JAX
            ], axis=0)
        
        for key, value in regions.items():
            if isinstance(value, list): # Ensure all are JAX arrays
                regions[key] = jnp.asarray(value, dtype=jnp.float32)

        return regions

    @partial(jax.jit, static_argnums=(0,))
    def get_current_behavior(self, robot_position):
        robot_position_jnp = jnp.asarray(robot_position, dtype=jnp.float32)

        MAX_DATA_DIM = max(self.MAX_POLYGON_VERTICES * 2 + 1, 3 + 1)

        region_ids_list = []
        region_types_list = []
        region_data_list = []

        region_ids_list.append(-1)
        region_types_list.append(-1)
        region_data_list.append(jnp.zeros(MAX_DATA_DIM, dtype=jnp.float32))

        for name in self.sorted_region_names:            
            region_id = self.region_name_to_id.get(name)
            region_data = self.behavior_regions[name]

            if isinstance(region_data, jnp.ndarray) and region_data.ndim == 2:
                num_actual_vertices = region_data.shape[0]
                padded_polygon = jnp.pad(region_data, ((0, self.MAX_POLYGON_VERTICES - num_actual_vertices), (0, 0)), mode='constant', constant_values=0.0)
                flat_data = jnp.concatenate((padded_polygon.flatten(), jnp.array([num_actual_vertices], dtype=jnp.float32)))
                region_ids_list.append(region_id)
                region_types_list.append(0)
                region_data_list.append(flat_data)
            elif isinstance(region_data, tuple) and region_data[0] == "circle":
                center_x, center_y, radius = region_data[1]
                dummy_num_vertices = 0.0
                padded_circle_data_and_num = jnp.concatenate((circle_data_flat, jnp.array([dummy_num_vertices], dtype=jnp.float32)))
                padded_circle_data_and_num = jnp.pad(padded_circle_data_and_num, (0, MAX_DATA_DIM - padded_circle_data_and_num.shape[0]), mode='constant', constant_values=0.0)

                region_ids_list.append(region_id)
                region_types_list.append(1)
                region_data_list.append(padded_circle_data_and_num)

        all_region_ids = jnp.array(region_ids_list, dtype=jnp.int32)
        all_region_types = jnp.array(region_types_list, dtype=jnp.int32)
        all_region_padded_data = jnp.stack(region_data_list, axis=0)

        @jax.jit
        def _get_behavior_jit(position, region_ids, region_types, region_padded_data):
            initial_carry = (-1, False)

            def scan_fn(carry, region_info):
                current_behavior_id, is_found_flag = carry
                region_id, region_type, flat_padded_data_and_num_vertices = region_info

                is_polygon_type = (region_type == 0)
                is_circle_type = (region_type == 1)

                num_actual_vertices = jnp.floor(flat_padded_data_and_num_vertices[-1]).astype(jnp.int32)
                polygon_coords_flat = flat_padded_data_and_num_vertices[:-1]
                polygon_coords = jnp.reshape(polygon_coords_flat, (self.MAX_POLYGON_VERTICES, 2))

                is_in_poly = self._is_point_in_polygon(position, polygon_coords, num_actual_vertices)
                is_current_in_polygon = jnp.logical_and(is_polygon_type, is_in_poly)

                center_x, center_y, radius = flat_padded_data_and_num_vertices[0], flat_padded_data_and_num_vertices[1], flat_padded_data_and_num_vertices[2]
                circle_distance = jnp.linalg.norm(position - jnp.array([center_x, center_y]))
                is_in_circle = circle_distance <= radius
                is_current_in_circle = jnp.logical_and(is_circle_type, is_in_circle)

                new_behavior_id = jnp.where(is_current_in_polygon, region_id, current_behavior_id)
                new_behavior_id = jnp.where(is_current_in_circle, region_id, new_behavior_id)
                
                new_is_found_flag = jnp.logical_or(is_found_flag, jnp.logical_or(is_current_in_polygon, is_current_in_circle))

                return (new_behavior_id, new_is_found_flag), None

            final_carry, _ = jax.lax.scan(scan_fn, initial_carry, (all_region_ids, all_region_types, all_region_padded_data))
            final_behavior_id, _ = final_carry

            return final_behavior_id

        return _get_behavior_jit(robot_position_jnp, all_region_ids, all_region_types, all_region_padded_data)

    @partial(jax.jit, static_argnums=(0,))
    def _is_point_in_polygon(self, point, polygon_padded, num_actual_vertices):
        x, y = point

        is_degenerate_polygon = (num_actual_vertices < 3)
        if_degenerate_result = jnp.array(False)

        first_vertex_index = jnp.array([0])
        first_actual_vertex = jax.lax.dynamic_slice(polygon_padded, (0, 0), (1, 2)).squeeze()

        closed_polygon_padded = jnp.concatenate((polygon_padded, first_actual_vertex[jnp.newaxis, :]), axis=0)

        p1_coords = closed_polygon_padded[:-1]
        p2_coords = closed_polygon_padded[1:]
        edges = jnp.stack((p1_coords, p2_coords), axis=1)

        is_vertex = jnp.any(jnp.where(jnp.arange(self.MAX_POLYGON_VERTICES)[:, None] < num_actual_vertices,
                                        jnp.all(jnp.isclose(point, polygon_padded), axis=1),
                                        False))

        initial_inside = jnp.where(is_vertex, jnp.array(True), jnp.array(False))

        def scan_body(current_inside_state, i):
            is_actual_edge = (i < num_actual_vertices)

            p1x, p1y = edges[i, 0]
            p2x, p2y = edges[i, 1]

            def true_fn():
                cond1 = y > jnp.min(jnp.array([p1y, p2y]))
                cond2 = y <= jnp.max(jnp.array([p1y, p2y]))
                cond3 = x <= jnp.max(jnp.array([p1x, p2x]))

                denominator = p2y - p1y
                xinters = jnp.where(
                    denominator != 0.0,
                    (y - p1y) * (p2x - p1x) / denominator + p1x,
                    jnp.inf * jnp.sign(x)
                )

                should_toggle_base = jnp.logical_and(
                    cond1,
                    jnp.logical_and(
                        cond2,
                        jnp.logical_or(
                            x <= xinters,
                            jnp.isclose(x, xinters)
                        )
                    )
                )

                line_vec = edges[i, 1] - edges[i, 0]
                point_vec = point - edges[i, 0]
                cross_product = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
                is_collinear = jnp.isclose(cross_product, 0.0)
                dot_product = jnp.dot(point_vec, line_vec)
                squared_length = jnp.dot(line_vec, line_vec)
                is_within_segment = jnp.where(squared_length > 1e-6, jnp.logical_and(dot_product >= 0, dot_product <= squared_length), False)
                is_on_segment = jnp.logical_and(is_collinear, is_within_segment)

                is_horizontal = jnp.isclose(p1y, p2y)
                should_toggle = jnp.where(is_horizontal, jnp.array(False), should_toggle_base)

                new_inside_state = jnp.where(should_toggle, jnp.logical_not(current_inside_state), current_inside_state)
                final_inside_state = jnp.where(is_on_segment, jnp.array(True), new_inside_state)
                return final_inside_state

            def false_fn():
                return current_inside_state

            updated_inside = jax.lax.cond(is_actual_edge, true_fn, false_fn)

            return updated_inside, None

        final_inside_if_valid, _ = jax.lax.scan(scan_body, initial_inside, jnp.arange(self.MAX_POLYGON_VERTICES + 1))


        return jnp.where(is_degenerate_polygon, if_degenerate_result, final_inside_if_valid)

    def visualize_behavior_regions(self, ax: Axes):
        added_labels = set()

        for i, name in enumerate(self.sorted_region_names):
            region = self.behavior_regions[name]
            color = self.region_colors[i]
            label_for_legend = None
            alpha = 0.5

            if name.startswith("on_bridge"):
                label_for_legend = "On Bridge"
            elif name.startswith("approach_bridge"):
                label_for_legend = "Approach Bridge"
            elif name.startswith("exit_bridge"):
                label_for_legend = "Exit Bridge"
            elif name.startswith("around_bridge_side"):
                label_for_legend = "Around Bridge"
                alpha = 0.3
            elif name.startswith("in_building"):
                label_for_legend = "In Building"
            elif name.startswith("approach_building"):
                label_for_legend = "Approach Building"
                alpha = 0.3
            elif name.startswith("exit_building"):
                label_for_legend = "Exit Building"
                alpha = 0.3
            elif name.startswith("around_building_side"):
                label_for_legend = "Around Building"
                alpha = 0.3
            elif name == "open_space":
                label_for_legend = "Open Space"
                alpha = 0.1
                
            if isinstance(region, jnp.ndarray) and region.ndim == 2:
                region_np = np.asarray(region)
                
                if name == "open_space":
                    min_x = np.min(region_np[:, 0])
                    max_x = np.max(region_np[:, 0])
                    min_y = np.min(region_np[:, 1])
                    max_y = np.max(region_np[:, 1])
                    patch = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                            alpha=alpha, color=color, fill=True)
                else:
                    patch = plt.Polygon(region_np, closed=True, alpha=alpha, color=color, fill=True)
                
                if label_for_legend not in added_labels:
                    patch.set_label(label_for_legend)
                    added_labels.add(label_for_legend)
                ax.add_patch(patch)

            elif isinstance(region, tuple) and region[0] == "circle":
                center_x, center_y, radius = region[1]
                circle_patch = plt.Circle((center_x, center_y), radius,
                                            alpha=alpha, color=color, fill=False, linestyle='--')
                if label_for_legend not in added_labels:
                    circle_patch.set_label(label_for_legend)
                    added_labels.add(label_for_legend)
                ax.add_patch(circle_patch)
            
            if name != "open_space":
                 centroid = np.asarray(self.all_region_centroids_jax_array[i])
                 ax.text(centroid[0], centroid[1], 'o',
                         fontsize=8, ha='center', va='center', color='black')

        ax.legend(loc='upper right', fontsize='small')


    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        centroid = self.all_region_centroids_jax_array[region_id]
        return centroid