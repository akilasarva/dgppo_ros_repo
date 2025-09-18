import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon
from typing import List, Tuple, Dict, Any
from functools import partial
import jax.random as jr
import jax
import jax.debug as jd
import jax.lax as lax
from jax.lax import cond

from ..env.behavior_associator import BehaviorAssociator

class BehaviorIntersection(BehaviorAssociator):
    def __init__(self, intersections: List, all_region_names: List[str], buildings: List = [], bridges: List = [], obstacles: List = []):
        if not intersections:
            raise ValueError("BehaviorBuildings requires at least one building.")
        super().__init__(intersections, bridges, buildings, obstacles, all_region_names)

    def _define_behavior_regions(self) -> Dict[str, Any]:
        intersection_params = self.intersections[0]
        center = jnp.array(intersection_params[0])
        global_angle = intersection_params[3]
        passage_width = intersection_params[1]
        obs_len = intersection_params[2]
        is_four_way = intersection_params[4]
        
        half_gap = passage_width / 2.0

        def four_way_intersection(args):
            """
            Calculates regions for a four-way intersection.
            The central and passage regions are defined by the inner corners of
            obstacles placed at the grid's edges and rotated.
            """
            area_size, obs_len, global_angle = args
            area_size = jnp.array(area_size, float)
            obs_len = jnp.array(obs_len, float)
            half_obs_len = obs_len / 2.0
            
            # Get the fixed positions of the obstacle centers
            obs_pos = jnp.array([
                [0.0, 0.0],
                [area_size, 0.0],
                [area_size, area_size],
                [0.0, area_size]
            ])

            # Get the rotated obstacle angles
            obs_theta = jnp.array([0.0, 0.0, jnp.pi / 2.0, jnp.pi / 2.0]) + global_angle

            # FIX: The local inner corners must be geometrically correct to form the passages.
            # This is the correct array for the inner corners.
            local_inner_corners = jnp.array([
                [half_obs_len, half_obs_len],    # Obstacle at (0, 0)
                [-half_obs_len, half_obs_len],   # Obstacle at (area_size, 0)
                [-half_obs_len, -half_obs_len],  # Obstacle at (area_size, area_size)
                [half_obs_len, -half_obs_len]    # Obstacle at (0, area_size)
            ])
            
            # Create a single rotation matrix for the global angle
            cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Use einsum to efficiently rotate and translate all four corners
            rotated_in_intersection = (jnp.einsum('ij,kj->ki', rotation_matrix, local_inner_corners) + obs_pos)

            # NEW: Define passage regions based on inner corners and an outward extension
            passage_len = 0.3  # Length of the passage polygon, as requested
            outward_vectors_local = jnp.array([
                [0.0, -1.0],  # Bottom passage direction
                [1.0, 0.0],  # Left passage direction
                [0.0, 1.0],   # Right passage direction
                [-1.0, 0.0]    # Top passage direction
            ])
            # The local passages are trapezoids defined by two inner corners and two new points
            # created by extending outwards.
            # Bottom Passage (connecting inner corners of obstacles 0 and 1)
            rotated_outward_vectors = jnp.einsum('ij,kj->ki', rotation_matrix, outward_vectors_local)

            # Construct the passage polygons directly from the rotated_in_intersection vertices
            rotated_passages = jnp.stack([
                jnp.array([
                    rotated_in_intersection[0],
                    rotated_in_intersection[1],
                    rotated_in_intersection[1] + rotated_outward_vectors[0] * passage_len,
                    rotated_in_intersection[0] + rotated_outward_vectors[0] * passage_len
                ]),
                jnp.array([
                    rotated_in_intersection[1],
                    rotated_in_intersection[2],
                    rotated_in_intersection[2] + rotated_outward_vectors[1] * passage_len,
                    rotated_in_intersection[1] + rotated_outward_vectors[1] * passage_len
                ]),
                jnp.array([
                    rotated_in_intersection[2],
                    rotated_in_intersection[3],
                    rotated_in_intersection[3] + rotated_outward_vectors[2] * passage_len,
                    rotated_in_intersection[2] + rotated_outward_vectors[2] * passage_len
                ]),
                jnp.array([
                    rotated_in_intersection[3],
                    rotated_in_intersection[0],
                    rotated_in_intersection[0] + rotated_outward_vectors[3] * passage_len,
                    rotated_in_intersection[3] + rotated_outward_vectors[3] * passage_len
                ])
            ])
            
            open_space_len = 0.45
            open_space_polygons = jnp.stack([
                jnp.array([
                    rotated_passages[0][2],
                    rotated_passages[0][3],
                    rotated_passages[0][3] + rotated_outward_vectors[0] * open_space_len,
                    rotated_passages[0][2] + rotated_outward_vectors[0] * open_space_len
                ]),
                jnp.array([
                    rotated_passages[1][2],
                    rotated_passages[1][3],
                    rotated_passages[1][3] + rotated_outward_vectors[1] * open_space_len,
                    rotated_passages[1][2] + rotated_outward_vectors[1] * open_space_len
                ]),
                jnp.array([
                    rotated_passages[2][2],
                    rotated_passages[2][3],
                    rotated_passages[2][3] + rotated_outward_vectors[2] * open_space_len,
                    rotated_passages[2][2] + rotated_outward_vectors[2] * open_space_len
                ]),
                jnp.array([
                    rotated_passages[3][2],
                    rotated_passages[3][3],
                    rotated_passages[3][3] + rotated_outward_vectors[3] * open_space_len,
                    rotated_passages[3][2] + rotated_outward_vectors[3] * open_space_len
                ])
            ])

            return rotated_in_intersection, rotated_passages, open_space_polygons
        
        def three_way_intersection(args):
            """
            Calculates regions for a three-way intersection using the specified logic.
            """
            area_size, obs_len, global_angle = args
            half_obs_len = obs_len / 2.0

            grid_corners = jnp.array([
                [0.0, 0.0],
                [area_size, 0.0],
                [area_size, area_size],
                [0.0, area_size]
            ])

            # Determine the inner corners for the central region
            # This is a trapezoid connecting the inner corners of the three effective obstacles.

            # Inner corner of small obstacle at top-left (grid_corners[3])
            c3_inner = grid_corners[3] + jnp.array([half_obs_len, -half_obs_len])
            
            # Inner corner of small obstacle at top-right (grid_corners[2])
            c2_inner = grid_corners[2] + jnp.array([-half_obs_len, -half_obs_len])
            
            # Inner corner of the large obstacle (effectively from grid_corners[0] and grid_corners[1])
            # The bottom edge of the central region will be from (0 + half_obs_len, 0 + half_obs_len)
            # to (area_size - half_obs_len, 0 + half_obs_len)
            
            c0_inner_for_large_obs = grid_corners[0] + jnp.array([half_obs_len, half_obs_len])
            c1_inner_for_large_obs = grid_corners[1] + jnp.array([-half_obs_len, half_obs_len])
            
            in_intersection_region = jnp.array([
                c3_inner,  # Top-left vertex of trapezoid
                c2_inner,  # Top-right vertex of trapezoid
                c1_inner_for_large_obs, # Bottom-right vertex of trapezoid
                c0_inner_for_large_obs  # Bottom-left vertex of trapezoid
            ])

            # Define the three outward-facing passages using the central region's vertices
            passage_len = 0.25
            
            # Outward vectors for the three passages (Left, Right, Bottom)
            # Note: The order here corresponds to how we'll extract the sides from in_intersection_region
            outward_vectors_local = jnp.array([
                [-1.0, 0.0],   # Left passage direction
                [1.0, 0.0],    # Right passage direction
                [0.0, -1.0]    # Bottom passage direction
            ])
            
            cos_a, sin_a = jnp.cos(global_angle), jnp.sin(global_angle)
            rotation_matrix = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_outward_vectors = jnp.einsum('ij,kj->ki', rotation_matrix, outward_vectors_local)

            # Construct the passage polygons
            rotated_passages = jnp.stack([
                jnp.array([ # Left passage (connects c3_inner and c0_inner_for_large_obs)
                    in_intersection_region[0], # c3_inner
                    in_intersection_region[3], # c0_inner_for_large_obs
                    in_intersection_region[3] + rotated_outward_vectors[0] * passage_len,
                    in_intersection_region[0] + rotated_outward_vectors[0] * passage_len
                ]),
                jnp.array([ # Right passage (connects c2_inner and c1_inner_for_large_obs)
                    in_intersection_region[1], # c2_inner
                    in_intersection_region[2], # c1_inner_for_large_obs
                    in_intersection_region[2] + rotated_outward_vectors[1] * passage_len,
                    in_intersection_region[1] + rotated_outward_vectors[1] * passage_len
                ]),
                jnp.array([ # Bottom passage (connects c0_inner_for_large_obs and c1_inner_for_large_obs)
                    in_intersection_region[3], # c0_inner_for_large_obs
                    in_intersection_region[2], # c1_inner_for_large_obs
                    in_intersection_region[2] + rotated_outward_vectors[2] * passage_len,
                    in_intersection_region[3] + rotated_outward_vectors[2] * passage_len
                ]),
                jnp.zeros((4, 2)) # Dummy passage for consistency with 4-way output
            ])
            
            # Apply global rotation and translation
            rotated_in_intersection = jnp.einsum('ij,kj->ki', rotation_matrix, in_intersection_region - center) + center
            rotated_passages = jnp.einsum('ij,kj->ki', rotation_matrix, rotated_passages.reshape(-1, 2) - center) + center
            rotated_passages = rotated_passages.reshape(4, 4, 2)

            return rotated_in_intersection, rotated_passages, rotated_passages
        
        # Use jax.lax.cond as before
        in_intersection, passage_polygons, open_space_polygons = jax.lax.cond(
            is_four_way,
            four_way_intersection,
            three_way_intersection,
            (1.5, obs_len, global_angle)
        )

        regions = {}
        regions["in_intersection"] = in_intersection

        for i, passage in enumerate(passage_polygons):
            # Assign specific passage polygons to both enter and exit
            regions[f"passage_{i}_enter"] = passage_polygons[i]
            regions[f"passage_{i}_exit"] = passage_polygons[i]
            
            # Assign the corresponding open_space polygon
            regions[f"open_space_{i}"] = open_space_polygons[i]
            
        return regions

    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        properties = {
            "in_intersection": {"label": "Intersection", "color": "darkorange", "alpha": 0.8},
        }
        is_four_way = self.intersections[0][4]
        num_passages = 4 if is_four_way else 3
        for p in range(num_passages):
            properties[f"passage_{p}_enter"] = {"label": f"Passage {p} Enter", "color": "lightblue", "alpha": 0.5}
            properties[f"passage_{p}_exit"] = {"label": f"Passage {p} Exit", "color": "lightgreen", "alpha": 0.5}
            properties[f"open_space_{p}"] = {"label": f"Open Space {p}", "color": "yellow", "alpha": 0.1}

        return properties

    @partial(jax.jit, static_argnums=(0,))
    def get_current_behavior_direction(self, pos: jnp.ndarray, vel: jnp.ndarray) -> jnp.ndarray:
        """
        Determines the current region and categorizes enter/exit based on direction.
        Args:
            pos: Agent's position (2,)
            vel: Agent's velocity (2,)
        Returns:
            The integer ID of the resolved behavioral region.
        """
        current_region_id = self.get_current_behavior(pos)
        
        in_intersection_id = self.region_name_to_id["in_intersection"]
        centroid_in_intersection = self.get_region_centroid(in_intersection_id)
        
        vec_to_centroid = centroid_in_intersection - pos
        dot_product = jnp.dot(vel, vec_to_centroid)
        is_moving_towards = dot_product > 0.01 
        
        # Determine if the current region is a passage using JAX-compatible logic
        is_passage = jnp.logical_and(current_region_id >= self.passage_start_id, current_region_id < self.passage_end_id)
        
        final_id = lax.cond(
            current_region_id == in_intersection_id,
            lambda: in_intersection_id,
            lambda: lax.cond(
                is_passage,
                lambda: self._get_enter_exit_id(current_region_id, is_moving_towards),
                lambda: current_region_id
            )
        )
        return final_id

    def _get_enter_exit_id(self, passage_id, is_moving_towards):
        base_passage_idx = (passage_id - self.passage_start_id) // 2
        
        # Calculate the IDs for both enter and exit
        enter_id = self.passage_start_id + base_passage_idx * 2
        exit_id = self.passage_start_id + base_passage_idx * 2 + 1
        
        # Use jnp.where to choose the correct ID based on the condition
        return jnp.where(is_moving_towards, enter_id, exit_id)

