import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import pathlib
import functools as ft

from colour import hsl2hex
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon as MplPolygon # Renamed to avoid conflict with your Polygon class
from matplotlib.patches import Rectangle as MplRectangle
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import List, Optional, Union, Tuple

from ..trainer.data import Rollout
from ..trainer.utils import centered_norm
from ..utils.typing import EdgeIndex, Pos2d, Pos3d, Array
from ..utils.utils import merge01, tree_index, MutablePatchCollection, save_anim
from dgppo.env.obstacle import Cuboid, Sphere, Obstacle, Rectangle

# Assuming your BehaviorAssociator is in the same file or can be imported.
# If BehaviorAssociator is in a different file, adjust the import path:
# from your_module.behavior_associator import BehaviorAssociator
class BehaviorAssociator:
    def __init__(self, bridges, buildings, obstacles):
        self.bridges = bridges
        self.buildings = buildings
        self.obstacles = obstacles # Store obstacles if needed for region definition
        self.behavior_regions = self._define_behavior_regions()
        print(self.behavior_regions)
        
        self.regions = []        # Initialize as an empty list
        self.region_labels = []  # Initialize as an empty list
        self.region_colors = []  # Initialize as an empty list
        # -----------------------------------------------------------
        
        # Add world boundary (example, assuming it's unbatched and fixed)
        self.regions.append([(-10, -10), (60, -10), (60, 50), (-10, 50)])
        self.region_labels.append("World Boundary")
        self.region_colors.append('lightgray')
        
    def _get_rectangle_corners_np(self, center, length, width, angle_rad):
        half_l = length / 2.0
        half_w = width / 2.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Corners relative to an unrotated center (0,0) and unrotated axis
        local_corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])

        # Rotation matrix
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])

        # Rotate and translate
        rotated_corners = np.dot(local_corners, rotation_matrix.T)
        final_corners = rotated_corners + center
        return final_corners

    

    def _define_behavior_regions(self):
        regions = {}

        regions["open_space"] = [(-10, -10), (60, -10), (60, 50), (-10, 50)]

        individual_bridge_corners = []
        for i, (sx, sy, l, w, a) in enumerate(self.bridges):
            rad = np.radians(a)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            corners_obstacle = np.array([
                (sx, sy),
                (sx + l * cos_a, sy + l * sin_a),
                (sx + l * cos_a - w * sin_a, sy + l * sin_a + w * cos_a),
                (sx - w * sin_a, sy + w * cos_a)
            ])
            individual_bridge_corners.append(corners_obstacle)
            
            # # Define "Around Bridge" for each individual obstacle (KEEP THIS BLOCK)
            # around_border_thick = 5
            # width_unit_vec = np.array([-sin_a, cos_a]) # Points from (sx,sy) to (sx-w*sin_a, sy+w*cos_a)

            # # Side 1 (outward from corners_obstacle[0] and corners_obstacle[1])
            # around1_p1 = corners_obstacle[0]
            # around1_p2 = corners_obstacle[1]
            # around1_p3 = corners_obstacle[1] + around_border_thick * width_unit_vec
            # around1_p4 = corners_obstacle[0] + around_border_thick * width_unit_vec
            # regions[f"around_bridge1_{i}"] = [around1_p1.tolist(), around1_p2.tolist(), around1_p3.tolist(), around1_p4.tolist()]

            # # Side 2 (outward from corners_obstacle[3] and corners_obstacle[2])
            # around2_p1 = corners_obstacle[3]
            # around2_p2 = corners_obstacle[2]
            # around2_p3 = corners_obstacle[2] - around_border_thick * width_unit_vec
            # around2_p4 = corners_obstacle[3] - around_border_thick * width_unit_vec
            # regions[f"around_bridge2_{i}"] = [around2_p1.tolist(), around2_p2.tolist(), around2_p3.tolist(), around2_p4.tolist()]

        if len(self.bridges) >= 2:
            # Get data for the two obstacles forming the channel
            (sx1, sy1, l1, w1, a1) = self.bridges[0] # First obstacle
            (sx2, sy2, l2, w2, a2) = self.bridges[1] # Second obstacle

            # Calculate corners for each individual obstacle
            rad = np.radians(a1) # Assuming both obstacles have the same angle
            cos_a, sin_a = np.cos(rad), np.sin(rad)

            corners_obs1 = individual_bridge_corners[0] # Use pre-calculated corners
            corners_obs2 = individual_bridge_corners[1]
            
            on_bridge_p1 = corners_obs1[3] # Corner of obs1 closer to origin on its width side
            on_bridge_p2 = corners_obs1[2] # Corner of obs1 further from origin on its width side
            on_bridge_p3 = corners_obs2[1] # Corner of obs2 further from origin on its width side
            on_bridge_p4 = corners_obs2[0] # Corner of obs2 closer to origin on its width side
            
            # Using 'on_bridge_0' as the key for the channel
            regions[f"on_bridge_0"] = [on_bridge_p1.tolist(), on_bridge_p4.tolist(), on_bridge_p3.tolist(), on_bridge_p2.tolist()]

            channel_entrance_p1 = on_bridge_p1 
            channel_entrance_p2 = on_bridge_p4 

            actual_center_channel_approach_arc = (channel_entrance_p1 + channel_entrance_p2) / 2.0
            channel_width = np.linalg.norm(on_bridge_p1 - on_bridge_p4)
            semi_circle_radius_channel = channel_width / 2.0
            
            num_points = 30
            angles_approach_arc_channel = np.linspace(rad + np.pi / 2, rad + 3 * np.pi / 2, num_points)
            semi_circle_approach_points_channel = np.array([
                actual_center_channel_approach_arc[0] + semi_circle_radius_channel * np.cos(angles_approach_arc_channel),
                actual_center_channel_approach_arc[1] + semi_circle_radius_channel * np.sin(angles_approach_arc_channel)
            ]).T
            # Using 'approach_bridge_0' as the key for the channel
            regions[f"approach_bridge_0"] = [channel_entrance_p2.tolist(), channel_entrance_p1.tolist()] + semi_circle_approach_points_channel.tolist()[::-1]

            # Removed the duplicate "Around Bridge" definition here.
            # The definition is correctly handled in the loop above.

            channel_exit_p1 = on_bridge_p2 
            channel_exit_p2 = on_bridge_p3 

            actual_center_channel_exit_arc = (channel_exit_p1 + channel_exit_p2) / 2.0

            angles_exit_arc_channel = np.linspace(rad - np.pi / 2, rad + np.pi / 2, num_points)
            semi_circle_exit_points_channel = np.array([
                actual_center_channel_exit_arc[0] + semi_circle_radius_channel * np.cos(angles_exit_arc_channel),
                actual_center_channel_exit_arc[1] + semi_circle_radius_channel * np.sin(angles_exit_arc_channel)
            ]).T
            # Using 'exit_bridge_0' as the key for the channel
            regions[f"exit_bridge_0"] = [channel_exit_p2.tolist(), channel_exit_p1.tolist()] + semi_circle_exit_points_channel.tolist()[::-1]

            # Calculate overall channel center (midpoint between the centers of the two obstacles)
            center1 = np.array([(sx1 + l1 * cos_a - w1 * sin_a + sx1 + l1 * cos_a) / 2.0, (sy1 + l1 * sin_a + w1 * cos_a + sy1 + l1 * sin_a) / 2.0]) # This is approximate center of obs1
            center2 = np.array([(sx2 + l2 * cos_a - w2 * sin_a + sx2 + l2 * cos_a) / 2.0, (sy2 + l2 * sin_a + w2 * cos_a + sy2 + l2 * sin_a) / 2.0]) # This is approximate center of obs2

            # Better center calculation: average the four "inner" corners of the channel
            # P1 = corners_obs1[3], P2 = corners_obs1[2], P3 = corners_obs2[1], P4 = corners_obs2[0]
            overall_bridge_center_np = (on_bridge_p1 + on_bridge_p2 + on_bridge_p3 + on_bridge_p4) / 4.0

            # Use the length of one of the bridge obstacles for the around region length
            bridge_length = l1 # Assuming l1 and l2 are the same
            
            # The width of the 'around' region (the yellow area)
            around_border_thick = 10.0 # From your image, these are quite wide

            # `width_unit_vec` (perpendicular to channel length)
            # This vector points from the lower wall towards the upper wall of the channel.
            # (on_bridge_p4 - on_bridge_p1) is a vector along the width of the channel.
            width_unit_vec = (on_bridge_p4 - on_bridge_p1)
            width_unit_vec = width_unit_vec / np.linalg.norm(width_unit_vec)


            # Distance from the overall channel center to the center of an 'around' region
            # This is half the channel width + half the wall thickness + half the around region thickness
            # (channel_width / 2.0) is already `semi_circle_radius_channel`
            # `w1 / 2.0` is `half_wall_thickness` (assuming w1 is wall thickness)
            
            dist_to_around_center = semi_circle_radius_channel + (w1 / 2.0) + (around_border_thick / 2.0)

            # Region 1: along one side (e.g., "below" the lower bridge obstacle)
            center_around1 = overall_bridge_center_np - dist_to_around_center * width_unit_vec
            around_corners1 = self._get_rectangle_corners_np(
                center_around1, bridge_length, around_border_thick, rad
            )
            regions[f"around_bridge_outer_0"] = around_corners1.tolist() # Renamed for consistency

            # Region 2: along the other side (e.g., "above" the upper bridge obstacle)
            center_around2 = overall_bridge_center_np + dist_to_around_center * width_unit_vec
            around_corners2 = self._get_rectangle_corners_np(
                center_around2, bridge_length, around_border_thick, rad
            )
            regions[f"around_bridge_outer_1"] = around_corners2.tolist() # Renamed for consistency



        # Add back any building regions or other regions that might have been removed.
        # Ensure 'return regions' is at the very end of the method.
        return regions
    
    def get_current_behavior(self, robot_position):
        # Prioritize specific behaviors
        # Iterating through all defined regions
        for behavior, region in self.behavior_regions.items():
            if isinstance(region, tuple) and region[0] == "circle":
                # Handle circle obstacle regions separately for point-in-region check
                center_x, center_y, radius = region[1]
                if np.linalg.norm(np.array(robot_position) - np.array([center_x, center_y])) <= radius:
                    return behavior
            elif isinstance(region, list): # Polygon
                if self._is_point_in_polygon(robot_position, region):
                    return behavior
        
        # Fallback to open space if defined and within its bounds, otherwise unknown
        open_space_region = self.behavior_regions.get("open_space")
        if open_space_region and self._is_point_in_polygon(robot_position, open_space_region):
            return "open_space"
        else:
            return "unknown_space"

    def _is_point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def visualize_behavior_regions(self, ax):
        # A set to keep track of labels already added to avoid duplicates in the legend
        added_labels = set()

        for behavior, region in self.behavior_regions.items():
            print(behavior)
            color = None
            label = None
            alpha = 0.5

            if behavior.startswith("on_bridge"):
                color = 'skyblue'
                label = "On Bridge"
            elif behavior.startswith("approach_bridge"):
                color = 'lightgreen'
                label = "Approach Bridge"
            elif behavior.startswith("exit_bridge"):
                color = 'lightcoral'
                label = "Exit Bridge"
            elif behavior.startswith("around_bridge_side"):
                color = 'gold'
                label = "Around Bridge"
                alpha = 0.3
            elif behavior.startswith("in_building"):
                color = 'salmon'
                label = "In Building"
            elif behavior.startswith("approach_building"):
                color = 'mediumseagreen'
                label = "Approach Building"
            elif behavior.startswith("exit_building"):
                color = 'sandybrown'
                label = "Exit Building"
            elif behavior.startswith("around_building_side"):
                color = 'darkgoldenrod'
                label = "Around Building"
                alpha = 0.3
            elif behavior == "open_space":
                # For open_space, draw a rectangle rather than a polygon from points
                min_x = min(p[0] for p in region)
                max_x = max(p[0] for p in region)
                min_y = min(p[1] for p in region)
                max_y = max(p[1] for p in region)
                patch = MplRectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                          alpha=0.1, color='lightgray') # label will be set below
                label = "Open Space" 
                continue # Skip to next iteration

            if color and region:
                polygon = MplPolygon(region, closed=True, alpha=alpha, color=color)
                # Only add label to legend if it hasn't been added yet
                if label not in added_labels:
                    polygon.set_label(label)
                    added_labels.add(label)
                ax.add_patch(polygon)
        ax.legend(loc='upper right', fontsize='small')

    def get_region_centroid(self, region_name):
        """
        Returns the centroid of a specified behavior region.
        Assumes regions are defined as lists of points (polygons).
        """
        region_coords = self.behavior_regions.get(region_name)
        if region_coords is None:
            return None # Region not found
        
        if isinstance(region_coords, list): # Polygon
            coords = np.array(region_coords)
            centroid = np.mean(coords, axis=0)
            return centroid
        elif isinstance(region_coords, tuple) and region_coords[0] == "circle":
            return np.array(region_coords[1][:2]) # Center (x, y) for a circle
        return None


def plot_graph(
        ax: Axes,
        pos: Pos2d,
        radius: Union[float, List[float]],
        color: Union[str, List[str]],
        with_label: Union[bool, List[bool]] = True,
        plot_edge: bool = False,
        edge_index: Optional[EdgeIndex] = None,
        edge_color: Union[str, List[str]] = 'k',
        alpha: float = 1.0,
        obstacle_color: str = '#000000',
) -> Axes:
    if isinstance(radius, float):
        radius = np.ones(pos.shape[0]) * radius
    if isinstance(radius, list):
        radius = np.array(radius)
    if isinstance(color, str):
        color = [color for _ in range(pos.shape[0])]
    if isinstance(with_label, bool):
        with_label = [with_label for _ in range(pos.shape[0])]
    circles = []
    for i in range(pos.shape[0]):
        circles.append(plt.Circle((float(pos[i, 0]), float(pos[i, 1])),
                                  radius=radius[i], color=color[i], clip_on=False, alpha=alpha, linewidth=0.0))
        if with_label[i]:
            ax.text(float(pos[i, 0]), float(pos[i, 1]), f'{i}', size=12, color="k",
                    family="sans-serif", weight="normal", horizontalalignment="center", verticalalignment="center",
                    transform=ax.transData, clip_on=True)
    circles = PatchCollection(circles, match_original=True)
    ax.add_collection(circles)

    if plot_edge and edge_index is not None:
        if isinstance(edge_color, str):
            edge_color = [edge_color for _ in range(edge_index.shape[1])]
        start, end = pos[edge_index[0, :]], pos[edge_index[1, :]]
        direction = (end - start) / jnp.linalg.norm(end - start, axis=1, keepdims=True)
        start = start + direction * radius[edge_index[0, :]][:, None]
        end = end - direction * radius[edge_index[1, :]][:, None]
        widths = (radius[edge_index[0, :]] + radius[edge_index[1, :]]) * 20
        lines = np.stack([start, end], axis=1)
        edges = LineCollection(lines, colors=edge_color, linewidths=widths, alpha=0.5)
        ax.add_collection(edges)
    return ax


def plot_node_3d(ax, pos: Pos3d, r: float, color: str, alpha: float, grid: int = 10) -> Axes:
    u = np.linspace(0, 2 * np.pi, grid)
    v = np.linspace(0, np.pi, grid)
    x = r * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
    return ax


def plot_graph_3d(
        ax,
        pos: Pos3d,
        radius: float,
        color: Union[str, List[str]],
        with_label: bool = True,
        plot_edge: bool = False,
        edge_index: Optional[EdgeIndex] = None,
        edge_color: Union[str, List[str]] = 'k',
        alpha: float = 1.0,
        obstacle_color: str = '#000000',
):
    if isinstance(color, str):
        color = [color for _ in range(pos.shape[0])]
    for i in range(pos.shape[0]):
        plot_node_3d(ax, pos[i], radius, color[i], alpha)
        if with_label:
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2], f'{i}', size=12, color="k", family="sans-serif", weight="normal",
                    horizontalalignment="center", verticalalignment="center")
    if plot_edge:
        if isinstance(edge_color, str):
            edge_color = [edge_color for _ in range(edge_index.shape[1])]
        for i_edge in range(edge_index.shape[1]):
            i = edge_index[0, i_edge]
            j = edge_index[1, i_edge]
            vec = pos[i, :] - pos[j, :]
            x = [pos[i, 0] - 2 * radius * vec[0], pos[j, 0] + 2 * radius * vec[0]]
            y = [pos[i, 1] - 2 * radius * vec[1], pos[j, 1] + 2 * radius * vec[1]]
            z = [pos[i, 2] - 2 * radius * vec[2], pos[j, 2] + 2 * radius * vec[2]]
            ax.plot(x, y, z, linewidth=1.0, color=edge_color[i_edge])
    return ax


def get_BuRd():
    # blue = "#3182bd"
    # blue = hsl2hex([0.57, 0.59, 0.47])
    blue = hsl2hex([0.57, 0.5, 0.55])
    light_blue = hsl2hex([0.5, 1.0, 0.995])

    # Tint it to orange a bit.
    # red = "#de2d26"
    # red = hsl2hex([0.04, 0.74, 0.51])
    red = hsl2hex([0.028, 0.62, 0.59])
    light_red = hsl2hex([0.098, 1.0, 0.995])

    sdf_cm = LinearSegmentedColormap.from_list("SDF", [(0, light_blue), (0.5, blue), (0.5, red), (1, light_red)], N=256)
    return sdf_cm


def get_faces_cuboid(points: Pos3d) -> Array:
    point_id = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]])
    faces = points[point_id]
    return faces


def get_cuboid_collection(
        obstacles: Cuboid, alpha: float = 0.8, linewidth: float = 1.0, edgecolor: str = 'k', facecolor: str = 'r'
) -> Poly3DCollection:
    get_faces_vmap = jax.vmap(get_faces_cuboid)
    cuboid_col = Poly3DCollection(
        merge01(get_faces_vmap(obstacles.points)),
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    return cuboid_col


def get_sphere_collection(
        obstacles: Sphere, alpha: float = 0.8, facecolor: str = 'r'
) -> Poly3DCollection:
    def get_sphere(inp):
        center = inp[:3]
        radius = inp[3]
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        return jnp.stack([x, y, z], axis=-1)

    get_sphere_vmap = jax.vmap(get_sphere)
    sphere_col = Poly3DCollection(
        merge01(get_sphere_vmap(jnp.concatenate([obstacles.center, obstacles.radius[:, None]], axis=-1))),
        alpha=alpha,
        linewidth=0.0,
        edgecolor='k',
        facecolor=facecolor
    )

    return sphere_col


def get_obs_collection(
        obstacles: Obstacle, color: str, alpha: float
):
    if isinstance(obstacles, Rectangle):
        n_obs = len(obstacles.center)
        obs_polys = [MplPolygon(obstacles.points[ii]) for ii in range(n_obs)] # Use MplPolygon
        obs_col = PatchCollection(obs_polys, color="#8a0000", alpha=1.0, zorder=99)
    elif isinstance(obstacles, Cuboid):
        obs_col = get_cuboid_collection(obstacles, alpha=alpha, facecolor=color)
    elif isinstance(obstacles, Sphere):
        obs_col = get_sphere_collection(obstacles, alpha=alpha, facecolor=color)
    else:
        raise NotImplementedError
    return obs_col


def get_f1tenth_body(
        pos: Pos2d, theta: Array, delta: Array, radius: Array
):
    pos1 = pos + radius / 2 * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    pos2 = pos - radius / 2 * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    theta1 = theta + delta
    theta2 = theta
    pos = jnp.concatenate([pos1, pos2], axis=0)
    theta = jnp.concatenate([theta1, theta2], axis=0)
    body = jax.vmap(
        ft.partial(Rectangle.create, width=radius, height=radius / 4)
    )(pos, theta=theta)

    return body


def render_mpe(
        rollout: Rollout,
        video_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        n_obs: int,
        r: float,
        obs_r: float,
        cost_components: Tuple[str, ...],
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        n_goal: Optional[int] = None,
        **kwargs
):
    assert dim == 1 or dim == 2 or dim == 3
    if n_goal is None:
        n_goal = n_agent

    # set up visualization option
    if dim == 1 or dim == 2:
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    else:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax: Axes3D = fig.add_subplot(projection='3d')
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    if dim == 3:
        ax.set_zlim(0., side_length)
    ax.set(aspect="equal")
    if dim == 2:
        plt.axis("off")

    if viz_opts is None:
        viz_opts = {}

    # plot the first frame
    T_graph = rollout.graph
    graph0 = tree_index(T_graph, 0)

    agent_color = "#0068ff"
    goal_color = "#2fdd00"
    obs_color = "#8a0000"
    edge_goal_color = goal_color

    # plot obstacles
    if hasattr(graph0.env_states, "obs"):
        obs = graph0.env_states.obs
        if obs is not None:
            obs_circs = []
            for ii in range(len(obs)):
                obs_circs.append(plt.Circle(obs[ii, :2], obs_r, color=obs_color, linewidth=0.0))
            obs_col = MutablePatchCollection(obs_circs, match_original=True, zorder=1)
            ax.add_collection(obs_col)

    # plot agents
    n_color = [agent_color] * n_agent + [goal_color] * n_goal
    n_pos = np.array(graph0.states[:n_agent + n_goal, :dim]).astype(np.float32)
    n_radius = np.array([r] * (n_agent + n_goal))
    if dim == 1 or dim == 2:
        if dim == 1:
            n_pos = np.concatenate([n_pos, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)
        agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
                       for ii in range(n_agent + n_goal)]
        agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        ax.add_collection(agent_col)
    else:
        plot_r = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        agent_col = ax.scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent might not be correct

    # plot edges
    all_pos = graph0.states[:n_agent + n_goal + n_obs, :dim]
    if dim == 1:
        all_pos = np.concatenate([all_pos, np.ones((n_agent + n_goal + n_obs, 1)) * side_length / 2], axis=1)
    edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
    is_pad = np.any(edge_index == n_agent + n_goal + n_obs, axis=0)
    e_edge_index = edge_index[:, ~is_pad]
    e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
    e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
    e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent + n_goal)
    e_is_goal = e_is_goal[~is_pad]
    e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
    if dim == 1:
        e_lines = e_lines[~e_is_goal]
        e_colors = "0.2"
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    elif dim == 2:
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    else:
        edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    ax.add_collection(edge_col)

    # text for cost and reward
    text_font_opts = dict(
        size=16,
        color="k",
        family="cursive",
        weight="normal",
        transform=ax.transAxes,
    )
    if dim == 1 or dim == 2:
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
    else:
        cost_text = ax.text2D(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)

    # text for safety
    safe_text = []
    if Ta_is_unsafe is not None:
        if dim == 1 or dim == 2:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        else:
            safe_text = [ax.text2D(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]

    # text for time step
    if dim == 1 or dim == 2:
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
    else:
        kk_text = ax.text2D(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)

    # add agent labels
    label_font_opts = dict(
        size=20,
        color="k",
        family="cursive",
        weight="normal",
        ha="center",
        va="center",
        transform=ax.transData,
        clip_on=True,
        zorder=7,
    )
    agent_labels = []
    if dim == 1 or dim == 2:
        agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], f"{ii}", **label_font_opts) for ii in range(n_agent)]
    else:
        for ii in range(n_agent):
            pos2d = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], ax.get_proj())[:2]
            agent_labels.append(ax.text2D(pos2d[0], pos2d[1], f"{ii}", **label_font_opts))

    # plot cbf
    cnt_col = []
    if "cbf" in viz_opts:
        if dim == 1 or dim == 3:
            print('Warning: CBF visualization is not supported in 1D or 3D.')
        else:
            Tb_xs, Tb_ys, Tbb_h, cbf_num = viz_opts["cbf"]
            bb_Xs, bb_Ys = np.meshgrid(Tb_xs[0], Tb_ys[0])
            norm = centered_norm(Tbb_h.min(), Tbb_h.max())
            levels = np.linspace(norm.vmin, norm.vmax, 15)

            cmap = get_BuRd().reversed()
            contour_opts = dict(cmap=cmap, norm=norm, levels=levels, alpha=0.9)
            cnt = ax.contourf(bb_Xs, bb_Ys, Tbb_h[0], **contour_opts)

            contour_line_opts = dict(levels=[0.0], colors=["k"], linewidths=3.0)
            cnt_line = ax.contour(bb_Xs, bb_Ys, Tbb_h[0], **contour_line_opts)

            cbar = fig.colorbar(cnt, ax=ax)
            cbar.add_lines(cnt_line)
            cbar.ax.tick_params(labelsize=36, labelfontfamily="Times New Roman")

            cnt_col = [*cnt.collections, *cnt_line.collections]

            ax.text(0.5, 1.0, "CBF for {}".format(cbf_num), transform=ax.transAxes, va="bottom")
    if "Vh" in viz_opts:
        if dim == 1 or dim == 2:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", zorder=100, **text_font_opts)
        else:
            Vh_text = ax.text2D(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

    # init function for animation
    def init_fn() -> list[plt.Artist]:
        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]

    # update function for animation
    def update(kk: int) -> list[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = graph.states[:-1, :dim]
        if dim == 1:
            n_pos_t = np.concatenate([n_pos_t, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)

        # update agent positions
        if dim == 1 or dim == 2:
            for ii in range(n_agent):
                agent_circs[ii].set_center(tuple(n_pos_t[ii]))
        else:
            agent_col.set_offsets(n_pos_t[:n_agent + n_goal, :2])
            agent_col.set_3d_properties(n_pos_t[:n_agent + n_goal, 2], zdir='z')

        # update edges
        e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        is_pad_t = np.any(e_edge_index_t == n_agent + n_goal + n_obs, axis=0)
        e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent + n_goal)
        e_is_goal_t = e_is_goal_t[~is_pad_t]
        e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        if dim == 1:
            e_lines_t = e_lines_t[~e_is_goal_t]
            e_colors_t = "0.2"
        edge_col.set_segments(e_lines_t)
        edge_col.set_colors(e_colors_t)

        # update agent labels
        for ii in range(n_agent):
            if dim == 1 or dim == 2:
                agent_labels[ii].set_position(n_pos_t[ii])
            else:
                text_pos = proj3d.proj_transform(n_pos_t[ii, 0], n_pos_t[ii, 1], n_pos_t[ii, 2], ax.get_proj())[:2]
                agent_labels[ii].set_position(text_pos)

        # update cost and safe labels
        if kk < len(rollout.costs):
            all_costs = ""
            for i_cost in range(rollout.costs[kk].shape[1]):
                all_costs += f"    {cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
            all_costs = all_costs[:-2]
            cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
        else:
            cost_text.set_text("")
        if kk < len(Ta_is_unsafe):
            a_is_unsafe = Ta_is_unsafe[kk]
            unsafe_idx = np.where(a_is_unsafe)[0]
            safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
        else:
            safe_text[0].set_text("Unsafe: {}")

        # Update the contourf.
        nonlocal cnt, cnt_line
        if "cbf" in viz_opts and dim == 2:
            for c in cnt.collections:
                c.remove()
            for c in cnt_line.collections:
                c.remove()

            bb_Xs_t, bb_Ys_t = np.meshgrid(Tb_xs[kk], Tb_ys[kk])
            cnt = ax.contourf(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_opts)
            cnt_line = ax.contour(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_line_opts)

            cnt_col_t = [*cnt.collections, *cnt_line.collections]
        else:
            cnt_col_t = []

        if "Vh" in viz_opts:
            Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

        kk_text.set_text("kk={:04}".format(kk))

        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col_t, kk_text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(T_graph.n_node)
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)


def render_lidar(
        rollout: Rollout,
        video_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        n_rays: int,
        r: float,
        cost_components: Tuple[str, ...],
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        n_goal: Optional[int] = None,
        **kwargs
):
    assert dim == 1 or dim == 2 or dim == 3
    if n_goal is None:
        n_goal = n_agent

    # set up visualization option
    if dim == 1 or dim == 2:
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    else:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax: Axes3D = fig.add_subplot(projection='3d')
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    if dim == 3:
        ax.set_zlim(0., side_length)
    ax.set(aspect="equal")
    if dim == 2:
        plt.axis("off")

    if viz_opts is None:
        viz_opts = {}

    # Plot behavior regions
    # Extract bridge parameters from kwargs
    # We expect these to be passed from the environment's render_video method
    bridge_center = kwargs.get("bridge_center", np.array([0., 0.]))
    bridge_length = kwargs.get("bridge_length", 0.0)
    bridge_gap_width = kwargs.get("bridge_gap_width", 0.0)
    bridge_wall_thickness = kwargs.get("bridge_wall_thickness", 0.0)
    bridge_theta = kwargs.get("bridge_theta", 0.0)

    # Convert bridge parameters to the format expected by BehaviorAssociator
    # BehaviorAssociator expects a list of (sx, sy, l, w, a)
    # The `create_single_bridge` function in your env gives you 2 rectangles for a bridge.
    # To pass to BehaviorAssociator, you'd need the parameters for each of those.
    # A simpler approach for visualization, assuming one bridge, is to calculate its overall bounding box
    # or pass the main bridge parameters and let BehaviorAssociator recreate the walls for its regions.

    # Let's assume for now that the `kwargs` related to the bridge represent the *overall* bridge structure
    # that `create_single_bridge` would use.
    # We will pass a single "conceptual" bridge to BehaviorAssociator
    # If your `LidarEnv` generates multiple bridges, you'd need to collect all their parameters here.
    
    # Reconstruct the two bridge walls for BehaviorAssociator
    # This logic should mirror `create_single_bridge` from your environment code
    print(f"DEBUG: Type of bridge_length: {type(bridge_length)}")
    if isinstance(bridge_length, (jnp.ndarray, np.ndarray)):
        print(f"DEBUG: Shape of bridge_length: {bridge_length.shape}")
        print(f"DEBUG: Value of bridge_length: {bridge_length}")
        
    if float(bridge_length) > 0:
        half_dist_to_wall_center = (bridge_gap_width / 2) + (bridge_wall_thickness / 2)
        offset_x = -half_dist_to_wall_center * np.sin(bridge_theta)
        offset_y = half_dist_to_wall_center * np.cos(bridge_theta)

        center1 = bridge_center + np.array([offset_x, offset_y])
        center2 = bridge_center - np.array([offset_x, offset_y])

        wall_width = bridge_length
        wall_height = bridge_wall_thickness
        
        # Convert theta from radians (env) to degrees (BehaviorAssociator expects degrees)
        bridge_theta_deg = np.degrees(bridge_theta)

        # Each "bridge" entry for BehaviorAssociator will be one of the two walls
        # The `sx, sy` for BehaviorAssociator's Rectangle is the bottom-left corner when unrotated.
        # We need to adjust `center1` and `center2` to be the "sx, sy" as per BehaviorAssociator's rectangle definition.
        # BehaviorAssociator's (sx, sy) is the 'bottom-left' if angle is 0.
        # For a rectangle centered at (Cx, Cy) with length L and width W and angle A,
        # its "bottom-left" corner (sx, sy) can be found by:
        # sx = Cx - (L/2)*cos(A) + (W/2)*sin(A)
        # sy = Cy - (L/2)*sin(A) - (W/2)*cos(A)
        
        # For wall1 (centered at `center1`):
        sx1 = center1[0] - (wall_width / 2) * np.cos(bridge_theta) + (wall_height / 2) * np.sin(bridge_theta)
        sy1 = center1[1] - (wall_width / 2) * np.sin(bridge_theta) - (wall_height / 2) * np.cos(bridge_theta)
        
        # For wall2 (centered at `center2`):
        sx2 = center2[0] - (wall_width / 2) * np.cos(bridge_theta) + (wall_height / 2) * np.sin(bridge_theta)
        sy2 = center2[1] - (wall_width / 2) * np.sin(bridge_theta) - (wall_height / 2) * np.cos(bridge_theta)

        bridges_for_associator = [
            (sx1, sy1, wall_width, wall_height, bridge_theta_deg),
            (sx2, sy2, wall_width, wall_height, bridge_theta_deg)
        ]
    else:
        bridges_for_associator = []

    buildings_for_associator = [] # Placeholder if you don't have buildings yet

    obstacles_for_associator = []
    first_env_state = rollout.graph.env_states
    if first_env_state.obstacle is not None:
        if isinstance(first_env_state.obstacle, Rectangle):
            # Assuming Rectangle obstacles are defined by center, lengths, and theta
            for i in range(len(first_env_state.obstacle.center)):
                center_x, center_y = first_env_state.obstacle.center[i][:2].tolist()
                # --- ADD THESE DEBUG PRINTS ---
                # print(f"DEBUG: first_env_state.obstacle.width type: {type(first_env_state.obstacle.width)}")
                # if hasattr(first_env_state.obstacle.width, 'shape'):
                #     print(f"DEBUG: first_env_state.obstacle.width shape: {first_env_state.obstacle.width.shape}")
                # print(f"DEBUG: first_env_state.obstacle.width value: {first_env_state.obstacle.width}")

                # print(f"DEBUG: first_env_state.obstacle.width[{i}] type: {type(first_env_state.obstacle.width[i])}")
                # if hasattr(first_env_state.obstacle.width[i], 'shape'):
                #     print(f"DEBUG: first_env_state.obstacle.width[{i}] shape: {first_env_state.obstacle.width[i].shape}")
                # print(f"DEBUG: first_env_state.obstacle.width[{i}] value: {first_env_state.obstacle.width[i]}")
                # # ------------------------------

                length_x = float(first_env_state.obstacle.width[i][0])
                length_y = float(first_env_state.obstacle.height[i][0])
                theta = float(first_env_state.obstacle.theta[i][0])
                
                # Convert center to (sx, sy) for BehaviorAssociator's Rectangle
                # sx = Cx - (L/2)*cos(A) + (W/2)*sin(A)
                # sy = Cy - (L/2)*sin(A) - (W/2)*cos(A)
                obs_sx = center_x - (length_x / 2) * np.cos(theta) + (length_y / 2) * np.sin(theta)
                obs_sy = center_y - (length_x / 2) * np.sin(theta) - (length_y / 2) * np.cos(theta)

                obstacles_for_associator.append(("rect", (obs_sx, obs_sy, length_x, length_y, np.degrees(theta))))
        elif isinstance(first_env_state.obstacle, Sphere):
            # Assuming Sphere obstacles are defined by center and radius
            for i in range(len(first_env_state.obstacle.center)):
                center_x, center_y = first_env_state.obstacle.center[i][:2].tolist() # Assuming 2D projection for sphere
                radius = float(first_env_state.obstacle.radius[i])
                obstacles_for_associator.append(("circle", (center_x, center_y, radius)))
        # Add Cuboid handling if needed, projecting to 2D for BehaviorAssociator

    behavior_associator = BehaviorAssociator(
        bridges=bridges_for_associator,
        buildings=buildings_for_associator,
        obstacles=obstacles_for_associator # Pass the converted obstacles
    )
    
    # Visualize the behavior regions
    if dim == 2: # Only visualize in 2D
        behavior_associator.visualize_behavior_regions(ax)


    # plot the first frame
    T_graph = rollout.graph
    graph0 = tree_index(T_graph, 0)

    agent_color = "#0068ff"
    goal_color = "#2fdd00"
    obs_color = "#8a0000"
    edge_goal_color = goal_color

    # plot obstacles (existing logic) - these are the actual obstacle patches
    if hasattr(graph0.env_states, "obstacle"):
        obs = graph0.env_states.obstacle
        if obs is not None:
            if isinstance(obs, Rectangle):
                # The `get_obs_collection` returns a PatchCollection which works for 2D.
                obs_col = get_obs_collection(obs, obs_color, alpha=1.0)
                ax.add_collection(obs_col)
            elif dim == 3 and (isinstance(obs, Cuboid) or isinstance(obs, Sphere)):
                # Handle 3D obstacles
                obs_col = get_obs_collection(obs, obs_color, alpha=0.8)
                ax.add_collection(obs_col)

    # plot agents
    n_hits = n_agent * n_rays
    n_color = [agent_color] * n_agent + [goal_color] * n_goal
    n_pos = np.array(graph0.states[:n_agent + n_goal, :dim]).astype(np.float32)
    n_radius = np.array([r] * (n_agent + n_goal))
    if dim == 1 or dim == 2:
        if dim == 1:
            n_pos = np.concatenate([n_pos, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)
        agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
                       for ii in range(n_agent + n_goal)]
        agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        ax.add_collection(agent_col)
    else:
        plot_r = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        agent_col = ax.scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent might not be correct

    # plot edges
    all_pos = graph0.states[:n_agent + n_goal + n_hits, :dim]
    if dim == 1:
        all_pos = np.concatenate([all_pos, np.ones((n_agent + n_goal + n_hits, 1)) * side_length / 2], axis=1)
    edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
    is_pad = np.any(edge_index == n_agent + n_goal + n_hits, axis=0)
    e_edge_index = edge_index[:, ~is_pad]
    e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
    e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
    e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent + n_goal)
    e_is_goal = e_is_goal[~is_pad]
    e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
    if dim == 1:
        e_lines = e_lines[~e_is_goal]
        e_colors = "0.2"
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    elif dim == 2:
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    else:
        edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    ax.add_collection(edge_col)

    # text for cost and reward
    text_font_opts = dict(
        size=16,
        color="k",
        family="cursive",
        weight="normal",
        transform=ax.transAxes,
    )
    if dim == 1 or dim == 2:
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
    else:
        cost_text = ax.text2D(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)

    # text for safety
    safe_text = []
    if Ta_is_unsafe is not None:
        if dim == 1 or dim == 2:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        else:
            safe_text = [ax.text2D(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]

    # text for time step
    if dim == 1 or dim == 2:
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
    else:
        kk_text = ax.text2D(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)

    # add agent labels
    label_font_opts = dict(
        size=20,
        color="k",
        family="cursive",
        weight="normal",
        ha="center",
        va="center",
        transform=ax.transData,
        clip_on=True,
        zorder=7,
    )
    agent_labels = []
    if dim == 1 or dim == 2:
        agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], f"{ii}", **label_font_opts) for ii in range(n_agent)]
    else:
        for ii in range(n_agent):
            pos2d = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], ax.get_proj())[:2]
            agent_labels.append(ax.text2D(pos2d[0], pos2d[1], f"{ii}", **label_font_opts))

    # plot cbf
    cnt_col = []
    if "cbf" in viz_opts:
        if dim == 1 or dim == 3:
            print('Warning: CBF visualization is not supported in 1D or 3D.')
        else:
            Tb_xs, Tb_ys, Tbb_h, cbf_num = viz_opts["cbf"]
            bb_Xs, bb_Ys = np.meshgrid(Tb_xs[0], Tb_ys[0])
            norm = centered_norm(Tbb_h.min(), Tbb_h.max())
            levels = np.linspace(norm.vmin, norm.vmax, 15)

            cmap = get_BuRd().reversed()
            contour_opts = dict(cmap=cmap, norm=norm, levels=levels, alpha=0.9)
            cnt = ax.contourf(bb_Xs, bb_Ys, Tbb_h[0], **contour_opts)

            contour_line_opts = dict(levels=[0.0], colors=["k"], linewidths=3.0)
            cnt_line = ax.contour(bb_Xs, bb_Ys, Tbb_h[0], **contour_line_opts)

            cbar = fig.colorbar(cnt, ax=ax)
            cbar.add_lines(cnt_line)
            cbar.ax.tick_params(labelsize=36, labelfontfamily="Times New Roman")

            cnt_col = [*cnt.collections, *cnt_line.collections]

            ax.text(0.5, 1.0, "CBF for {}".format(cbf_num), transform=ax.transAxes, va="bottom")
    if "Vh" in viz_opts:
        if dim == 1 or dim == 2:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", zorder=100, **text_font_opts)
        else:
            Vh_text = ax.text2D(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

    # init function for animation
    def init_fn() -> list[plt.Artist]:
        # This list should contain all artists that will be updated in the animation.
        # When visualizing static regions, they are added to `ax` directly and don't
        # need to be part of the `init_fn` return if they are not dynamically updated.
        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]

    # update function for animation
    def update(kk: int) -> list[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = graph.states[:-1, :dim]
        if dim == 1:
            n_pos_t = np.concatenate([n_pos_t, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)

        # update agent positions
        if dim == 1 or dim == 2:
            for ii in range(n_agent):
                agent_circs[ii].set_center(tuple(n_pos_t[ii]))
        else:
            agent_col.set_offsets(n_pos_t[:n_agent + n_goal, :2])
            agent_col.set_3d_properties(n_pos_t[:n_agent + n_goal, 2], zdir='z')

        # update edges
        e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        is_pad_t = np.any(e_edge_index_t == n_agent + n_goal + n_hits, axis=0)
        e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent + n_goal)
        e_is_goal_t = e_is_goal_t[~is_pad_t]
        e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        if dim == 1:
            e_lines_t = e_lines_t[~e_is_goal_t]
            e_colors_t = "0.2"
        edge_col.set_segments(e_lines_t)
        edge_col.set_colors(e_colors_t)

        # update agent labels
        for ii in range(n_agent):
            if dim == 1 or dim == 2:
                agent_labels[ii].set_position(n_pos_t[ii])
            else:
                text_pos = proj3d.proj_transform(n_pos_t[ii, 0], n_pos_t[ii, 1], n_pos_t[ii, 2], ax.get_proj())[:2]
                agent_labels[ii].set_position(text_pos)

        # update cost and safe labels
        if kk < len(rollout.costs):
            all_costs = ""
            for i_cost in range(rollout.costs[kk].shape[1]):
                all_costs += f"    {cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
            all_costs = all_costs[:-2]
            cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
        else:
            cost_text.set_text("")
        if kk < len(Ta_is_unsafe):
            a_is_unsafe = Ta_is_unsafe[kk]
            unsafe_idx = np.where(a_is_unsafe)[0]
            safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
        else:
            safe_text[0].set_text("Unsafe: {}")

        # Update the contourf.
        nonlocal cnt, cnt_line
        if "cbf" in viz_opts and dim == 2:
            for c in cnt.collections:
                c.remove()
            for c in cnt_line.collections:
                c.remove()

            bb_Xs_t, bb_Ys_t = np.meshgrid(Tb_xs[kk], Tb_ys[kk])
            cnt = ax.contourf(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_opts)
            cnt_line = ax.contour(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_line_opts)

            cnt_col_t = [*cnt.collections, *cnt_line.collections]
        else:
            cnt_col_t = []

        if "Vh" in viz_opts:
            Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

        kk_text.set_text("kk={:04}".format(kk))

        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col_t, kk_text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(T_graph.n_node)
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)