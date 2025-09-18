import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np # Keep if needed for other non-JAX ops, but avoid for JAX arrays
import functools as ft
import jax
import jax.debug as jd

from typing import NamedTuple, Tuple, Optional, List, Dict # Added List, Dict for type hints
from abc import ABC, abstractmethod

from jaxtyping import Float

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState, PRNGKey
from ...utils.utils import merge01, jax_vmap
from ..base import MultiAgentEnv
from ..obstacle import Obstacle, Rectangle
from ..plot import render_lidar, BehaviorAssociator # Ensure BehaviorAssociator is imported from plot
from ..utils import get_lidar, get_node_goal_rng


class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle

    bearing: Float[Array, "n_agent"]
    current_cluster_oh: Float[Array, "n_agent n_cluster"]
    start_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_bonus_awarded: jnp.ndarray

    bridge_center: Float[Array, "2"]
    bridge_length: float
    bridge_gap_width: float
    bridge_wall_thickness: float
    bridge_theta: float # Stored in radians

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]

    @property
    def n_cluster(self) -> int:
        return 4


LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]

def create_single_bridge(
    bridge_center: jnp.ndarray, # (2,) for x, y
    bridge_length: float,
    bridge_gap_width: float,
    bridge_wall_thickness: float,
    bridge_theta: float # In radians
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    # Calculate half of the total bridge width
    half_dist_to_wall_center = (bridge_gap_width / 2) + (bridge_wall_thickness / 2)

    offset_x = -half_dist_to_wall_center * jnp.sin(bridge_theta)
    offset_y = half_dist_to_wall_center * jnp.cos(bridge_theta)

    # Calculate centers for the two bridge walls
    center1 = bridge_center + jnp.array([offset_x, offset_y])
    center2 = bridge_center - jnp.array([offset_x, offset_y])

    # Each wall has the same length, thickness, and orientation as the bridge
    wall_width = bridge_length # The 'width' of the rectangle is the bridge 'length'
    wall_height = bridge_wall_thickness # The 'height' of the rectangle is the wall 'thickness'

    # Combine parameters for the two rectangles
    obs_pos = jnp.stack([center1, center2]) # Shape (2, 2)
    obs_len_x = jnp.array([wall_width, wall_width]) # Shape (2,)
    obs_len_y = jnp.array([wall_height, wall_height]) # Shape (2,)
    obs_theta = jnp.array([bridge_theta, bridge_theta]) # Shape (2,)

    return obs_pos, obs_len_x, obs_len_y, obs_theta

class LidarEnv(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        "num_bridges": 1, # Default to 1 bridge - Important for this scenario
        "bridge_length_range": [0.5, 1.0],
        "bridge_gap_width_range": [0.2, 0.4],
        "bridge_wall_thickness_range": [0.05, 0.1],
        "open_space_goal_distance": 0.2, # Added parameter for open space goal placement
        "goal_offset_from_centroid": 0.0
    }

    ALL_POSSIBLE_REGION_NAMES = [
        "open_space",
        "approach_bridge_0",
        "on_bridge_0",
        "exit_bridge_0"

    ]

    # CLUSTER_MAP now defines the canonical integer IDs for region types
    CLUSTER_MAP: Dict[str, int] = {
        "open_space": 0,
        "approach_bridge_0": 1,
        "on_bridge_0": 2,
        "exit_bridge_0":3

        #"around_bridge_0": 4, # You may also have "start" or "cross_bridge_gap" - ensure consistency
    }

    # Define curriculum transitions as a Python list of string tuples (for human readability/initial setup)
    _CURRICULUM_TRANSITIONS_STR = [ # Renamed to avoid confusion with the JAX array
        #("open_space", "approach_bridge_0"),
        ("approach_bridge_0", "on_bridge_0"), # Using "cross_bridge_gap" from plot.py
        ("on_bridge_0", "exit_bridge_0"),
        ("exit_bridge_0", "open_space"),
    ]

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None,
            n_cluster: int = 4
    ):
        area_size = LidarEnv.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarEnv, self).__init__(num_agents, area_size, max_step, dt, params)
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.num_goals = self._num_agents
        self.n_cluster = n_cluster

        if self.n_cluster < len(set(self.CLUSTER_MAP.values())):
            print(f"Warning: n_cluster ({self.n_cluster}) is less than the number of unique cluster IDs in CLUSTER_MAP ({len(set(self.CLUSTER_MAP.values()))}).")

        # Map integer IDs back to *base* string names (e.g., 0 -> "open_space", not "open_space_0")
        self._id_to_curriculum_prefix_map: Dict[int, str] = {v: k for k, v in self.CLUSTER_MAP.items()}

        allowed_id_transitions_list: List[Tuple[int, int]] = []
        for start_prefix, end_prefix in self._CURRICULUM_TRANSITIONS_STR:
            # Use .get() with a default value (e.g., -1) to handle prefixes not in CLUSTER_MAP gracefully,
            # though they should ideally all be defined.
            start_id = self.CLUSTER_MAP.get(start_prefix, -1)
            end_id = self.CLUSTER_MAP.get(end_prefix, -1)
            if start_id != -1 and end_id != -1: # Only add if both IDs are valid
                allowed_id_transitions_list.append((start_id, end_id))
            else:
                print(f"Warning: Curriculum transition ({start_prefix}, {end_prefix}) contains unknown cluster prefixes.")

        # Convert to JAX array. If the list is empty, create an empty (0, 2) array.
        if allowed_id_transitions_list:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.array(allowed_id_transitions_list, dtype=jnp.int32)
        else:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.empty((0, 2), dtype=jnp.int32)

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def bearing_dim(self) -> int:
        return 1 # Bearing is a single float

    @property
    def cluster_oh_dim(self) -> int:
        return self.n_cluster # One-hot encoding dimension

    @property
    def node_dim(self) -> int:
        return self.state_dim + self.bearing_dim + 3 * self.cluster_oh_dim + 3

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_vel, vy_vel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(
        self,
        key: PRNGKey,
        current_clusters: Optional[Array] = None,
        start_clusters: Optional[Array] = None,
        next_clusters: Optional[Array] = None,
        custom_obstacles: Optional[Tuple[Array, Array, Array, Array]] = None,
        transition_index: Optional[int] = None # This is now ONLY for starting a specific episode
    ) -> GraphsTuple:

        if current_clusters is None:
            current_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
        if start_clusters is None:
            start_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
        if next_clusters is None:
            next_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)

        all_obs_pos_list: List[jnp.ndarray] = []
        all_obs_len_x_list: List[jnp.ndarray] = []
        all_obs_len_y_list: List[jnp.ndarray] = []
        all_obs_theta_list: List[jnp.ndarray] = []

        bridge_center_env_state: Float[Array, "2"] = jnp.array([0.0, 0.0])
        bridge_length_env_state: Float[Array, ""] = jnp.array(0.0)
        bridge_gap_width_env_state: Float[Array, ""] = jnp.array(0.0)
        bridge_wall_thickness_env_state: Float[Array, ""] = jnp.array(0.0)
        bridge_theta_env_state: Float[Array, ""] = jnp.array(0.0)
        initial_bonus_awarded = jnp.zeros(self.num_agents, dtype=jnp.bool_)

        num_bridges = 1

        # jd.print("transition index {}", transition_index)
        if num_bridges > 0:
            if transition_index is None:
                # Original logic: pick a random transition
                transition_idx_key, key = jr.split(key)
                num_transitions = self.CURRICULUM_TRANSITIONS_INT_IDS.shape[0]
                transition_index = jr.randint(transition_idx_key, (), 0, num_transitions)

            start_region_id_tracer, next_region_id_tracer = self.CURRICULUM_TRANSITIONS_INT_IDS[transition_index]
            # jd.print("current index {}", start_region_id_tracer)
            # jd.print("next index {}", next_region_id_tracer)

        if custom_obstacles is not None:
            obs_pos, obs_len_x, obs_len_y, obs_theta = custom_obstacles
            all_obs_pos_list.append(obs_pos)
            all_obs_len_x_list.append(obs_len_x)
            all_obs_len_y_list.append(obs_len_y)
            all_obs_theta_list.append(obs_theta)
        else:
            n_rng_obs = self._params["n_obs"]
            assert n_rng_obs >= 0

            if n_rng_obs > 0:
                obstacle_key, key = jr.split(key, 2)
                obs_pos_orig = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
                length_key, key = jr.split(key, 2)
                obs_len_orig = jr.uniform(
                    length_key,
                    (n_rng_obs, 2),
                    minval=self._params["obs_len_range"][0],
                    maxval=self._params["obs_len_range"][1],
                )
                theta_key, key = jr.split(key, 2)
                obs_theta_orig = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * jnp.pi)

                all_obs_pos_list.append(obs_pos_orig)
                all_obs_len_x_list.append(obs_len_orig[:, 0])
                all_obs_len_y_list.append(obs_len_orig[:, 1])
                all_obs_theta_list.append(obs_theta_orig)

            if num_bridges > 0:
                if num_bridges != 1:
                    print("Warning: Only 1 bridge is currently supported for direct parameter storage.")

                bridge_rand_key, key = jr.split(key)
                center_key, length_key, gap_key, thickness_key, theta_key = jr.split(bridge_rand_key, 5)

                bridge_length = jr.uniform(length_key, (),
                                            minval=self._params.get("bridge_length_range", (0.5, 1.0))[0],
                                            maxval=self._params.get("bridge_length_range", (0.5, 1.0))[1])
                bridge_gap_width = jr.uniform(gap_key, (),
                                                minval=self._params.get("bridge_gap_width_range", (0.2, 0.4))[0],
                                                maxval=self._params.get("bridge_gap_width_range", (0.2, 0.4))[1])
                bridge_wall_thickness = jr.uniform(thickness_key, (),
                                                    minval=self._params.get("bridge_wall_thickness_range", (0.03, 0.1))[0],
                                                    maxval=self._params.get("bridge_wall_thickness_range", (0.03, 0.1))[1])
                bridge_theta = jr.uniform(theta_key, (), minval=0, maxval=2 * jnp.pi)

                effective_len = bridge_length
                effective_width = bridge_gap_width + 2 * bridge_wall_thickness

                cos_abs = jnp.abs(jnp.cos(bridge_theta))
                sin_abs = jnp.abs(jnp.sin(bridge_theta))

                max_x_extent = 0.5 * (effective_len * cos_abs + effective_width * sin_abs)
                max_y_extent = 0.5 * (effective_len * sin_abs + effective_width * cos_abs)

                min_center_x = max_x_extent
                max_center_x = self.area_size - max_x_extent
                min_center_y = max_y_extent
                max_center_y = self.area_size - max_y_extent

                min_center_x = jnp.maximum(0.0, min_center_x)
                max_center_x = jnp.maximum(min_center_x, max_center_x) # Ensure max is not less than min

                min_center_y = jnp.maximum(0.0, min_center_y)
                max_center_y = jnp.maximum(min_center_y, max_center_y) # Ensure max is not less than min
                bridge_center = jr.uniform(center_key, (2,),
                                            minval=jnp.array([min_center_x, min_center_y]),
                                            maxval=jnp.array([max_center_x, max_center_y]))

                bridge_obs_pos, bridge_obs_len_x, bridge_obs_len_y, bridge_obs_theta = \
                    create_single_bridge(
                        bridge_center,
                        bridge_length,
                        bridge_gap_width,
                        bridge_wall_thickness,
                        bridge_theta
                    )
                all_obs_pos_list.append(bridge_obs_pos)
                all_obs_len_x_list.append(bridge_obs_len_x)
                all_obs_len_y_list.append(bridge_obs_len_y)
                all_obs_theta_list.append(bridge_obs_theta)

                bridge_center_env_state = bridge_center
                bridge_length_env_state = bridge_length
                bridge_gap_width_env_state = bridge_gap_width
                bridge_wall_thickness_env_state = bridge_wall_thickness
                bridge_theta_env_state = bridge_theta

        if all_obs_pos_list:
            combined_obs_pos = jnp.concatenate(all_obs_pos_list, axis=0)
            combined_obs_len_x = jnp.concatenate(all_obs_len_x_list, axis=0)
            combined_obs_len_y = jnp.concatenate(all_obs_len_y_list, axis=0)
            combined_obs_theta = jnp.concatenate(all_obs_theta_list, axis=0)
            obstacles = self.create_obstacles(
                combined_obs_pos,
                combined_obs_len_x,
                combined_obs_len_y,
                combined_obs_theta
            )
        else:
            obstacles = None

        if num_bridges > 0:

            half_dist_to_wall_center = (bridge_gap_width_env_state / 2) + (bridge_wall_thickness_env_state / 2)
            offset_x = -half_dist_to_wall_center * jnp.sin(bridge_theta_env_state)
            offset_y = half_dist_to_wall_center * jnp.cos(bridge_theta_env_state)

            center1 = bridge_center_env_state + jnp.array([offset_x, offset_y])
            center2 = bridge_center_env_state - jnp.array([offset_x, offset_y])

            wall_width = bridge_length_env_state
            wall_height = bridge_wall_thickness_env_state

            bridge_theta_deg = jnp.degrees(bridge_theta_env_state)

            bridges_for_associator = [
                (center1[0], center1[1], wall_width, wall_height, bridge_theta_deg),
                (center2[0], center2[1], wall_width, wall_height, bridge_theta_deg)
            ]

            associator = BehaviorBridge(
                bridges=bridges_for_associator,
                all_region_names=self.ALL_POSSIBLE_REGION_NAMES
            )

            agent_states_list = []
            goal_states_list = []
            bearings_list = []
            current_clusters_oh_list = []
            start_clusters_oh_list = []
            next_clusters_oh_list = []

            open_space_id = self.CLUSTER_MAP["open_space"]
            exit_bridge_id = self.CLUSTER_MAP["exit_bridge_0"]

            bridge_dir_vec = jnp.array([jnp.cos(bridge_theta_env_state), jnp.sin(bridge_theta_env_state)])

            bridge_end1 = bridge_center_env_state - (bridge_length_env_state / 2) * bridge_dir_vec
            bridge_end2 = bridge_center_env_state + (bridge_length_env_state / 2) * bridge_dir_vec

            exit_bridge_centroid = associator.get_region_centroid(exit_bridge_id)

            exit_direction_vector = jnp.array([jnp.cos(bridge_theta_env_state), jnp.sin(bridge_theta_env_state)])

            open_space_goal_distance = 0.4 #self.params["open_space_goal_distance"]

            open_space_goal_pos = exit_bridge_centroid + exit_direction_vector * open_space_goal_distance

            for i in range(self.num_agents):
                key_agent_pos, key_goal_pos, key_rot, key_loop = jr.split(key, 4)
                key = key_loop

                initial_pos_candidate = associator.get_region_centroid(start_region_id_tracer)

                initial_pos = jnp.where(jnp.isnan(initial_pos_candidate).any(),
                                        jnp.zeros(2), # Default if NaN
                                        initial_pos_candidate)

                initial_pos = initial_pos + jr.normal(key_agent_pos, (2,)) * 0.05
                initial_pos = jnp.clip(initial_pos, 0, self.area_size)

                goal_pos_candidate = associator.get_region_centroid(next_region_id_tracer)

                is_open_space_goal = (next_region_id_tracer == self.CLUSTER_MAP["open_space"])

                goal_pos = jnp.where(
                    is_open_space_goal,
                    open_space_goal_pos,
                    goal_pos_candidate
                )

                goal_pos = jnp.where(jnp.isnan(goal_pos).any(),
                                    jnp.array([self.area_size / 2, self.area_size / 2]), # Fallback to center, not origin
                                    goal_pos)

                goal_pos = goal_pos + jr.normal(key_goal_pos, (2,)) * 0.05

                rot_angle = 0 #jr.uniform(key_rot, (), minval=-jnp.pi/4, maxval=jnp.pi/4)

                cos_rot = jnp.cos(rot_angle)
                sin_rot = jnp.sin(rot_angle)

                rotation_matrix = jnp.array([
                    [cos_rot, -sin_rot],
                    [sin_rot, cos_rot]
                ])

                goal_pos_relative = goal_pos - bridge_center_env_state
                rotated_goal_pos_relative = jnp.dot(rotation_matrix, goal_pos_relative)
                rotated_goal_pos = bridge_center_env_state + rotated_goal_pos_relative

                goal_pos = rotated_goal_pos

                bearing = jnp.arctan2(goal_pos[1] - initial_pos[1], goal_pos[0] - initial_pos[0]) #- jnp.pi/4

                start_cluster_idx = start_region_id_tracer
                current_cluster_idx = jnp.squeeze(associator.get_current_behavior(initial_pos))
                next_cluster_idx = next_region_id_tracer

                current_cluster_oh = jax.nn.one_hot(current_cluster_idx, self.n_cluster)
                start_cluster_oh = jax.nn.one_hot(start_cluster_idx, self.n_cluster)
                next_cluster_oh = jax.nn.one_hot(next_cluster_idx, self.n_cluster)

                agent_states_list.append(jnp.array([initial_pos[0], initial_pos[1], 0.0, 0.0]))
                goal_states_list.append(jnp.array([goal_pos[0], goal_pos[1], 0.0, 0.0]))
                bearings_list.append(bearing)
                current_clusters_oh_list.append(current_cluster_oh)
                start_clusters_oh_list.append(start_cluster_oh)
                next_clusters_oh_list.append(next_cluster_oh)

            states = jnp.stack(agent_states_list)
            goals = jnp.stack(goal_states_list)
            bearing = jnp.stack(bearings_list)
            current_clusters = jnp.stack(current_clusters_oh_list)
            start_clusters = jnp.stack(start_clusters_oh_list)
            next_clusters = jnp.stack(next_clusters_oh_list)

        else: # Fallback if no bridges are generated or BehaviorAssociator not used
            states, goals = get_node_goal_rng(
                key, self.area_size, 2, self.num_agents, 2.2 * self._params["car_radius"], obstacles)

            states = jnp.concatenate(
                [states, jnp.zeros((self.num_agents, self.state_dim - states.shape[1]), dtype=states.dtype)], axis=1)
            goals = jnp.concatenate(
                [goals, jnp.zeros((self.num_goals, self.state_dim - goals.shape[1]), dtype=goals.dtype)], axis=1)

            bearing = jnp.zeros((self.num_agents,), dtype=jnp.float32)
            current_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
            next_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
            associator = BehaviorAssociator(bridges=[], buildings=[], obstacles=[],
                                            region_name_to_id=self.CLUSTER_MAP,
                                            region_id_to_name=self._id_to_curriculum_prefix_map)


        assert states.shape == (self.num_agents, self.state_dim)
        assert goals.shape == (self.num_goals, self.state_dim)

        env_states = LidarEnvState(
            agent=states,
            goal=goals,
            obstacle=obstacles,
            bearing=bearing,
            current_cluster_oh=current_clusters,
            start_cluster_oh=start_clusters,
            next_cluster_oh=next_clusters,
            next_cluster_bonus_awarded=initial_bonus_awarded,
            bridge_center=bridge_center_env_state,
            bridge_length=bridge_length_env_state,
            bridge_gap_width=bridge_gap_width_env_state,
            bridge_wall_thickness=bridge_wall_thickness_env_state,
            bridge_theta=bridge_theta_env_state,
        )

        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        lidar_data = None
        if self.params["n_obs"] > 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    num_beams=self._params["n_rays"],
                    sense_range=self._params["comm_radius"],
                    max_returns=self._params["top_k_rays"],
                )
            )
            lidar_data = get_lidar_vmap(states[:, :2])
            assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 2)
        return lidar_data

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        agent_base_states = graph.env_states.agent # Get the underlying agent state from env_states
        goals = graph.env_states.goal
        obstacles = graph.env_states.obstacle

        bridge_center = graph.env_states.bridge_center
        bridge_length = graph.env_states.bridge_length
        bridge_gap_width = graph.env_states.bridge_gap_width
        bridge_wall_thickness = graph.env_states.bridge_wall_thickness
        bridge_theta = graph.env_states.bridge_theta

        cos_theta = jnp.cos(bridge_theta)
        sin_theta = jnp.sin(bridge_theta)

        half_gap_plus_half_wall = (bridge_gap_width / 2.0) + (bridge_wall_thickness / 2.0)

        offset_x_perp = -sin_theta * half_gap_plus_half_wall
        offset_y_perp = cos_theta * half_gap_plus_half_wall

        offset_vector = jnp.array([offset_x_perp, offset_y_perp])

        wall1_center = bridge_center + offset_vector
        wall2_center = bridge_center - offset_vector

        wall_length = bridge_length
        wall_width = bridge_wall_thickness
        wall_angle_deg = jnp.degrees(bridge_theta)

        bridge_walls_params = [
            (wall1_center[0], wall1_center[1], wall_length, wall_width, wall_angle_deg),
            (wall2_center[0], wall2_center[1], wall_length, wall_width, wall_angle_deg)
        ]

        bearing = graph.env_states.bearing
        from dgppo.env.bridge_behavior_associator import BehaviorBridge
        associator = BehaviorBridge(
            bridges=bridge_walls_params,
            all_region_names=self.ALL_POSSIBLE_REGION_NAMES
        )
        current_id = jax_vmap(associator.get_current_behavior)(agent_base_states[:, :2])
        current_cluster_oh = jax.nn.one_hot(current_id, self.n_cluster)
        start_cluster_oh = graph.env_states.start_cluster_oh
        next_cluster_oh = graph.env_states.next_cluster_oh

        # calculate next states
        action = self.clip_action(action)
        next_agent_base_states = self.agent_step_euler(agent_base_states, action) # Only update (x,y,vx,vy)

        reward, bonus_awarded_updated = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        next_env_state = LidarEnvState(
            next_agent_base_states,
            goals,
            obstacles,
            bearing,
            current_cluster_oh,
            start_cluster_oh,
            next_cluster_oh,
            bonus_awarded_updated,
            bridge_center,
            bridge_length,
            bridge_gap_width,
            bridge_wall_thickness,
            bridge_theta
        )

        lidar_data_next = self.get_lidar_data(next_agent_base_states, obstacles)
        info = {}
        done = jnp.array(False)

        return self.get_graph(next_env_state, lidar_data_next), reward, cost, done, info

    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

        return cost

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        from dgppo.env.plot import render_lidar

        first_env_state = rollout.graph.env_states
        bridge_params_for_render = {
            "bridge_center": first_env_state.bridge_center,
            "bridge_length": first_env_state.bridge_length,
            "bridge_gap_width": first_env_state.bridge_gap_width,
            "bridge_wall_thickness": first_env_state.bridge_wall_thickness,
            "bridge_theta": first_env_state.bridge_theta,
        }

        render_lidar(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["top_k_rays"] if self.params["n_obs"] > 0 or self.params["num_bridges"] > 0 else 0,
            r=self.params["car_radius"],
            obs_r=0.0, #
            cost_components=self.cost_components,
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            n_goal=self.num_goals,
            dpi=dpi,
            **bridge_params_for_render,
            **kwargs )

    @abstractmethod
    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        pass

    def get_graph(self, state: LidarEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_hits = 8
        n_nodes = self.num_agents + self.num_goals + n_hits

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)
        elif n_hits > 0:
            lidar_data = jnp.zeros((n_hits, 2), dtype=jnp.float32)

        # Node features: (x, y, vx, vy, bearing, current_cluster_oh, start_cluster_oh, next_cluster_oh, is_obs, is_goal, is_agent)
        node_feats = jnp.zeros((n_nodes, self.node_dim), dtype=jnp.float32)

        agent_start_idx = 0
        # Base state
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, :self.state_dim].set(state.agent)
        # Bearing
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim].set(state.bearing)
        # Current cluster
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim:self.state_dim+self.bearing_dim+self.n_cluster].set(state.current_cluster_oh)
        # Start cluster
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim+self.n_cluster:self.state_dim+self.bearing_dim+2*self.n_cluster].set(state.start_cluster_oh)
        # Next cluster
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim+2*self.n_cluster:self.state_dim+self.bearing_dim+3*self.n_cluster].set(state.next_cluster_oh)
        # Is agent indicator (last of 3)
        node_feats = node_feats.at[agent_start_idx:agent_start_idx+self.num_agents, self.state_dim+self.bearing_dim+3*self.n_cluster+2].set(1.0)

        # Goal features
        goal_start_idx = self.num_agents
        node_feats = node_feats.at[goal_start_idx:goal_start_idx+self.num_goals, :self.state_dim].set(state.goal)
        # Is goal indicator (middle)
        node_feats = node_feats.at[goal_start_idx:goal_start_idx+self.num_goals, self.state_dim+self.bearing_dim+3*self.n_cluster+1].set(1.0)

        # Obstacle (lidar hits)
        if n_hits > 0 and lidar_data is not None:
            obs_start_idx = self.num_agents + self.num_goals
            node_feats = node_feats.at[obs_start_idx:obs_start_idx+n_hits, :2].set(lidar_data)
            # Is obs indicator (first)
            node_feats = node_feats.at[obs_start_idx:obs_start_idx+n_hits, self.state_dim+self.bearing_dim+3*self.n_cluster].set(1.0)

        # Node types
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(LidarEnv.AGENT)
        node_type = node_type.at[self.num_agents:self.num_agents+self.num_goals].set(LidarEnv.GOAL)
        if n_hits > 0:
            node_type = node_type.at[self.num_agents+self.num_goals:].set(LidarEnv.OBS)

        edge_blocks = self.edge_blocks(state, lidar_data)

        # Raw states
        raw_states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states_padded = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - 2), dtype=lidar_data.dtype)],
                axis=1
            )
            raw_states = jnp.concatenate([raw_states, lidar_states_padded], axis=0)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=raw_states
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim
