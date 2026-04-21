import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon
from typing import List, Dict, Any
from functools import partial
import numpy as np 
import jax

from ..env.behavior_associator import BehaviorAssociator

class BehaviorBridge(BehaviorAssociator):
    def __init__(self, bridges: List, buildings: List = [], obstacles: List = [], all_region_names: List[str] = []):
        if not bridges:
            raise ValueError("BehaviorBridge requires at least one bridge.")
        super().__init__(bridges, buildings, obstacles, all_region_names)

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
            rad = jnp.radians(a)
            
            half_l = l / 2.0
            half_w = w / 2.0
            
            local_corners = jnp.array([
                [-half_l, -half_w], # Bottom-left
                [ half_l, -half_w], # Bottom-right
                [ half_l,  half_w], # Top-rightaq   
                [-half_l,  half_w]  # Top-left
            ], dtype=jnp.float32)

            cos_a, sin_a = jnp.cos(rad), jnp.sin(rad)
            rotation_matrix = jnp.array([
                [cos_a, -sin_a],
                [sin_a,  cos_a]
            ])

            rotated_corners_local = jnp.einsum('ij,kj->ki', rotation_matrix, local_corners)
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

            channel_entrance_p1 = on_bridge_p1 
            channel_entrance_p2 = on_bridge_p4 

            actual_center_channel_approach_arc = (channel_entrance_p1 + channel_entrance_p2) / 2.0
            channel_width = jnp.linalg.norm(on_bridge_p1 - on_bridge_p4)
            semi_circle_radius_channel = channel_width / 2.0
            
            num_points = 30

            rad_for_arcs = jnp.radians(self.bridges[0][4]) 
            
            angles_approach_arc_channel = jnp.linspace(rad_for_arcs + jnp.pi / 2, rad_for_arcs + 3 * jnp.pi / 2, num_points)
            semi_circle_approach_points_channel = jnp.array([
                actual_center_channel_approach_arc[0] + semi_circle_radius_channel * jnp.cos(angles_approach_arc_channel),
                actual_center_channel_approach_arc[1] + semi_circle_radius_channel * jnp.sin(angles_approach_arc_channel)
            ]).T
            
            regions[f"approach_bridge_0"] = jnp.concatenate([
                jnp.array([channel_entrance_p2, channel_entrance_p1]),
                semi_circle_approach_points_channel[::-1] 
            ], axis=0)

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
                semi_circle_exit_points_channel[::-1] 
            ], axis=0)
        
        for key, value in regions.items():
            if isinstance(value, list): 
                regions[key] = jnp.asarray(value, dtype=jnp.float32)

        return regions


    def _get_region_visualization_properties(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary mapping bridge region names to visualization properties.
        """
        return {
            "on_bridge_0": {"label": "On Bridge", "color": "skyblue"},
            "approach_bridge_0": {"label": "Approach Bridge", "color": "lightgreen"},
            "exit_bridge_0": {"label": "Exit Bridge", "color": "lightcoral"},
            "open_space": {"label": "Open Space", "color": "lightgray", "alpha": 0.1},
        }


    @partial(jax.jit, static_argnums=(0,))
    def get_region_centroid(self, region_id: jnp.ndarray):
        centroid = self.all_region_centroids_jax_array[region_id]
        return centroid

