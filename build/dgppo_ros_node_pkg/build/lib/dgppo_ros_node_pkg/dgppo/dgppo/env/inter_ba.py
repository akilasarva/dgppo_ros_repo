import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import pathlib
import functools as ft
import jax.tree_util as jtu
from functools import partial
import jax.debug as jd
import ipdb

from colour import hsl2hex
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import List, Optional, Union, Tuple, Dict, Any

class BehaviorAssociator:

    def __init__(self, intersections: List, bridges: List, buildings: List, obstacles: List, all_region_names: List[str]):
        self.intersections = intersections
        self.bridges = bridges
        self.buildings = buildings
        self.obstacles = obstacles
        self.all_region_names = all_region_names  # Store the master list
        self.behavior_regions = self._define_behavior_regions()

        self._initialize_full_region_maps()
        
        self._prepare_jax_data()


    def _initialize_full_region_maps(self):
        self.region_name_to_id = {name: i for i, name in enumerate(self.all_region_names)}
        self.region_id_to_name = {v: k for k, v in self.region_name_to_id.items()}
        self.sorted_region_names = self.all_region_names # The list is already sorted by index
        
        self.passage_ids = [self.region_name_to_id[name] for name in self.all_region_names if name.startswith("passage_")]

        if self.passage_ids:
            # These are the attributes that were missing
            self.passage_start_id = jnp.array(min(self.passage_ids))
            self.passage_end_id = jnp.array(max(self.passage_ids) + 1)
        else:
            # Handle the case where no passage regions exist
            self.passage_start_id = jnp.array(-1)
            self.passage_end_id = jnp.array(-1)
        
        self.id_to_curriculum_prefix_map = {}
        for region_id, region_name in self.region_id_to_name.items():
            if region_name.startswith("passage_"):
                self.id_to_curriculum_prefix_map[region_id] = "passage_"
            elif region_name == "in_intersection":
                self.id_to_curriculum_prefix_map[region_id] = "in_intersection"
            elif region_name.startswith("open_space_"):
                self.id_to_curriculum_prefix_map[region_id] = "open_space"
            else:
                self.id_to_curriculum_prefix_map[region_id] = region_name
                
    def _define_behavior_regions(self):
        raise NotImplementedError("Subclasses must implement _define_behavior_regions()")
    
    def _prepare_jax_data(self):
        temp_polygons = []
        temp_centroids = []
        temp_labels = []
        temp_colors = []

        self.MAX_POLYGON_VERTICES = 4
        for name, coords in self.behavior_regions.items():
            if isinstance(coords, jnp.ndarray) and coords.ndim == 2:
                self.MAX_POLYGON_VERTICES = max(self.MAX_POLYGON_VERTICES, coords.shape[0])

        #priority_order = ["open_space", "approach_bridge", "on_bridge", "exit_bridge"]
        # priority_order = ["open_space", "approach_bridge", "on_bridge", "exit_bridge"]
    
        # reordered_names = []
        # for prio_name in priority_order:
        #     for full_name in self.all_region_names:
        #         if full_name.startswith(prio_name):
        #             reordered_names.append(full_name)
        
        self.sorted_region_names = self.all_region_names

        for name in self.sorted_region_names:            
            coords = self.behavior_regions.get(name)

            if coords is not None:
                if isinstance(coords, jnp.ndarray) and coords.ndim == 2:
                    temp_polygons.append(coords)
                    temp_centroids.append(jnp.mean(coords, axis=0))
                    temp_labels.append(name)
                    temp_colors.append('blue')

        padded_polygons_for_stack = []
        for p in temp_polygons:
            num_actual_vertices = p.shape[0]
            padded_p = jnp.pad(p, ((0, self.MAX_POLYGON_VERTICES - num_actual_vertices), (0, 0)), mode='constant', constant_values=0.0)
            padded_polygons_for_stack.append(padded_p)

        self.all_region_polygons = jnp.stack(padded_polygons_for_stack, axis=0)
        self.all_region_centroids_jax_array = jnp.stack(temp_centroids, axis=0)
        self.region_labels = temp_labels
        self.region_colors = temp_colors

    @partial(jax.jit, static_argnums=(0,))
    def get_current_behavior(self, robot_position, **kwargs: Dict[str, Any]):

        robot_position_jnp = jnp.asarray(robot_position, dtype=jnp.float32)

        MAX_DATA_DIM = self.MAX_POLYGON_VERTICES * 2 + 1

        region_ids_list = []
        region_data_list = []

        regions_to_check = [name for name in self.sorted_region_names if self.region_name_to_id.get(name) != 0]

        for name in regions_to_check:            
            region_id = self.region_name_to_id.get(name)
            region_data = self.behavior_regions[name]

            if isinstance(region_data, jnp.ndarray) and region_data.ndim == 2:
                num_actual_vertices = region_data.shape[0]
                padded_polygon = jnp.pad(region_data, ((0, self.MAX_POLYGON_VERTICES - num_actual_vertices), (0, 0)), mode='constant', constant_values=0.0)
                flat_data = jnp.concatenate((padded_polygon.flatten(), jnp.array([num_actual_vertices], dtype=jnp.float32)))
                region_ids_list.append(region_id)
                region_data_list.append(flat_data)

        all_region_ids = jnp.array(region_ids_list, dtype=jnp.int32)
        all_region_padded_data = jnp.stack(region_data_list, axis=0)

        @jax.jit
        def _get_behavior_jit(position): 
            initial_carry = (0, False)

            def scan_fn(carry, region_info):
                current_behavior_id, is_found_flag = carry
                region_id, flat_padded_data_and_num_vertices = region_info

                num_actual_vertices = jnp.floor(flat_padded_data_and_num_vertices[-1]).astype(jnp.int32)
                polygon_coords_flat = flat_padded_data_and_num_vertices[:-1]
                polygon_coords = jnp.reshape(polygon_coords_flat, (self.MAX_POLYGON_VERTICES, 2))

                is_in_poly = self._is_point_in_polygon(position, polygon_coords, num_actual_vertices)

                new_behavior_id = jnp.where(is_in_poly, region_id, current_behavior_id)
                
                new_is_found_flag = jnp.logical_or(is_found_flag, is_in_poly)

                return (new_behavior_id, new_is_found_flag), None

            final_carry, _ = jax.lax.scan(scan_fn, initial_carry, (all_region_ids, all_region_padded_data))
            final_behavior_id, _ = final_carry

            return final_behavior_id

        return _get_behavior_jit(robot_position_jnp) 

    #@partial(jax.jit, static_argnums=(0, 3))
    def _is_point_in_polygon(self, point, polygon_padded, num_actual_vertices):
        x, y = point

        is_degenerate_polygon = (num_actual_vertices < 3)
        if_degenerate_result = jnp.array(False)
                
        first_actual_vertex = polygon_padded[0, :]
        all_vertices = jnp.concatenate((polygon_padded, first_actual_vertex[jnp.newaxis, :]), axis=0)

        p1_coords = all_vertices[:-1]
        p2_coords = all_vertices[1:]
        
        p2_coords = p2_coords.at[num_actual_vertices-1].set(first_actual_vertex)
        
        edges = jnp.stack((p1_coords, p2_coords), axis=1)

        is_vertex = jnp.any(jnp.where(jnp.arange(self.MAX_POLYGON_VERTICES)[:, None] < num_actual_vertices,
                                        jnp.all(jnp.isclose(point, polygon_padded), axis=1),
                                        False))
        initial_inside = jnp.where(is_vertex, jnp.array(True), jnp.array(False))

        # Core ray-casting algorithm
        def scan_body(current_inside_state, i):
            is_actual_edge = (i < num_actual_vertices)

            # This check is now robust because the edges array is correctly formed
            # with the closed polygon data, followed by padded zeros.
            p1x, p1y = edges[i, 0]
            p2x, p2y = edges[i, 1]

            # For actual edges
            def true_fn():
                
                # Horizontal ray's y-coord should lie between the y-coords of the edge vertices 
                cond1 = y > jnp.min(jnp.array([p1y, p2y]))
                cond2 = y < jnp.max(jnp.array([p1y, p2y]))
                
                denominator = p2y - p1y
                x_intercept = jnp.where(
                    denominator != 0.0,
                    (y - p1y) * (p2x - p1x) / denominator + p1x,
                    jnp.inf * jnp.sign(x)
                )
                
                # If cond1, cond2 and x < x_intercept, ray has to cross the edge so in/out flag should toggle
                should_toggle_base = jnp.logical_and(
                    cond1,
                    jnp.logical_and(
                        cond2,
                        jnp.logical_or(
                            x <= x_intercept,
                            jnp.isclose(x, x_intercept)
                        )
                    )
                )
                
                line_vec = edges[i, 1] - edges[i, 0]
                point_vec = point - edges[i, 0]
                
                # Check if point is along edge
                cross_product = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
                is_collinear = jnp.isclose(cross_product, 0.0)
                dot_product = jnp.dot(point_vec, line_vec)
                squared_length = jnp.dot(line_vec, line_vec)
                is_within_segment = jnp.where(squared_length > 1e-6, jnp.logical_and(dot_product >= 0, dot_product <= squared_length), False)
                is_on_segment = jnp.logical_and(is_collinear, is_within_segment)

                # Horizontal-edge edge case (don't count as an intersection)
                is_horizontal = jnp.isclose(p1y, p2y)
                should_toggle = jnp.where(is_horizontal, jnp.array(False), should_toggle_base)

                # Toggles if it should be toggled
                new_inside_state = jnp.where(should_toggle, jnp.logical_not(current_inside_state), current_inside_state)
                
                # On polygon border considered in 
                final_inside_state = jnp.where(is_on_segment, jnp.array(True), new_inside_state)
                
                return final_inside_state

            # For padded edges
            def false_fn():
                return current_inside_state

            updated_inside = jax.lax.cond(is_actual_edge, true_fn, false_fn)
            return updated_inside, None

        # Calls the scan_body function above for every vertex (and therefore edge) simultaneously
        final_inside_if_valid, _ = jax.lax.scan(scan_body, initial_inside, jnp.arange(self.MAX_POLYGON_VERTICES + 1))
        return jnp.where(is_degenerate_polygon, if_degenerate_result, final_inside_if_valid)

    def visualize_behavior_regions(self, ax: Axes):
        added_labels = set()
        region_properties = self._get_region_visualization_properties()

        for i, name in enumerate(self.sorted_region_names):
            region = self.behavior_regions[name]
            properties = region_properties.get(name, {})
            color = properties.get('color', 'blue')
            label_for_legend = properties.get('label', name)
            alpha = properties.get('alpha', 0.5)

            if isinstance(region, jnp.ndarray) and region.ndim == 2:
                region_np = np.asarray(region)
                
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

            centroid = np.asarray(self.all_region_centroids_jax_array[i])
            ax.text(centroid[0], centroid[1], 'o',
                    fontsize=8, ha='center', va='center', color='black')

        ax.legend(loc='upper right', fontsize='small')

    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _get_region_visualization_properties()")

    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        centroid = self.all_region_centroids_jax_array[region_id]
        return centroid


