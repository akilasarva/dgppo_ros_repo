import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np # Keep if needed for other non-JAX ops, but avoid for JAX arrays
import functools as ft
import jax
import jax.debug as jd
from jax.random import PRNGKey, uniform, split

from typing import NamedTuple, Tuple, Optional, List, Dict 
from abc import ABC, abstractmethod

from jaxtyping import Float

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState, PRNGKey
from ...utils.utils import merge01, jax_vmap
from ..base import MultiAgentEnv
from ..obstacle import Obstacle, Rectangle
from ..plot import render_lidar # Ensure BehaviorAssociator is imported from plot
from ..utils import get_lidar, get_node_goal_rng
from ..intersection_behavior_associator import BehaviorIntersection

from jax import jit, lax

class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle
    
    bearing: Float[Array, "n_agent"]
    current_cluster_oh: Float[Array, "n_agent n_cluster"]
    start_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_bonus_awarded: jnp.ndarray
    
    is_four_way: bool
    center: jnp.ndarray
    passage_width: float
    obs_len: float
    global_angle: float
    
    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]
    
    @property
    def n_cluster(self) -> int:
        return 4

LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]

def create_fourway_intersection(
    area_size: float,
    passage_width: float,
    obs_len: float,
    global_angle: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generates four obstacles, with each one centered directly on a corner of the grid.
    The corners are defined as (0, 0), (area_size, 0), (0, area_size), and (area_size, area_size).
    """
    # The positions are now the absolute corner coordinates.
    # The passage_width and obs_len are used for obstacle dimensions, not for position offsets.
    obs_pos = jnp.array([
        [0.0, 0.0],
        [area_size, 0.0],
        [0.0, area_size],
        [area_size, area_size]
    ])

    obs_len_x = jnp.array([obs_len, obs_len, obs_len, obs_len])
    obs_len_y = jnp.array([obs_len, obs_len, obs_len, obs_len])
    
    # The angles are set to align the obstacles with the axes at each corner.
    obs_theta = jnp.array([0.0, 0.0, jnp.pi / 2.0, jnp.pi / 2.0])

    # No rotation or translation is needed since the positions are absolute.
    rotated_obs_pos = obs_pos
    rotated_obs_theta = obs_theta + global_angle

    return rotated_obs_pos, obs_len_x, obs_len_y, rotated_obs_theta

def create_threeway_intersection(
    area_size: jnp.ndarray,
    passage_width: float,
    obs_len: float,
    global_angle: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generates a three-way intersection with one large obstacle covering two adjacent corners
    and two smaller obstacles covering the other two corners.
    """
    obs_pos = jnp.array([
        [area_size/2, 0.0],  # Large obstacle (centered later)
        [0.0, area_size], # Small obstacle 1
        [area_size, area_size] # Small obstacle 2
    ])

    # Define the dimensions (lengths) for the three obstacles
    obs_len_x = jnp.array([area_size*2, obs_len, obs_len])
    obs_len_y = jnp.array([obs_len/2, obs_len, obs_len])
    
    # Define the local angles for the obstacles
    obs_theta = jnp.array([0.0, jnp.pi / 2.0, jnp.pi / 2.0])

    # No rotation or translation is needed since the positions are absolute.
    rotated_obs_pos = obs_pos
    rotated_obs_theta = obs_theta + global_angle

    return rotated_obs_pos, obs_len_x, obs_len_y, rotated_obs_theta

class LidarEnv(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "n_obs": 1,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        "num_intersections": 1,
        
        "is_four_way_p": 0.5, # Probability of generating a 4-way intersection (0.5 for a 50/50 chance)
        "intersection_size_range": [0.5, 0.75], # Overall size of the intersection region
        "passage_width_range": [0.25, 0.5], # Min/max for the width of the road passages
        "obs_wall_range": [2, 4],
    }
    
    # In LidarEnv class
    ALL_POSSIBLE_REGION_NAMES = [
        "open_space_0", "open_space_1", "open_space_2", "open_space_3",
        "in_intersection",
        "passage_0_enter", "passage_0_exit",
        "passage_1_enter", "passage_1_exit",
        "passage_2_enter", "passage_2_exit",
        "passage_3_enter", "passage_3_exit",
    ]

    CLUSTER_MAP: Dict[str, int] = {
        "open_space": 0,
        "in_intersection": 1,
        "passage_enter": 2,
        "passage_exit": 3,
    }
    
    _SPECIFIC_TO_GENERAL_MAP = {
        "in_intersection": "in_intersection",
        "passage_0_enter": "passage_enter",
        "passage_0_exit": "passage_exit",
        "passage_1_enter": "passage_enter",
        "passage_1_exit": "passage_exit",
        "passage_2_enter": "passage_enter",
        "passage_2_exit": "passage_exit",
        "passage_3_enter": "passage_enter",
        "passage_3_exit": "passage_exit",
        "open_space_0": "open_space",
        "open_space_1": "open_space",
        "open_space_2": "open_space",
        "open_space_3": "open_space",
        "open_space": "open_space",
        "passage_enter": "passage_enter",
        "passage_exit": "passage_exit",
    }
    
    # 3. Override _CURRICULUM_TRANSITIONS_STR for building tasks
    _CURRICULUM_TRANSITIONS_STR = [
        ("open_space", "passage_enter"),
        ("passage_enter", "in_intersection"),
        ("in_intersection", "passage_exit"),
        ("passage_exit", "open_space")
    ]
    
    # 3. Override _CURRICULUM_TRANSITIONS_STR for building tasks
    VALID_GOAL_MAP = {
        #"open_space": ["passage_0_enter", "passage_1_enter", "passage_2_enter", "passage_3_enter"],
        "open_space_0": ["passage_0_enter"],
        "open_space_1": ["passage_1_enter"],
        "open_space_2": ["passage_2_enter"],
        "open_space_3": ["passage_3_enter"],
        "passage_0_enter": ["in_intersection"],
        "passage_1_enter": ["in_intersection"],
        "passage_2_enter": ["in_intersection"],
        "passage_3_enter": ["in_intersection"],
        "in_intersection": [
            "passage_0_exit", "passage_1_exit", "passage_2_exit", "passage_3_exit"
        ],
        "passage_0_exit": ["open_space_0"],
        "passage_1_exit": ["open_space_1"],
        "passage_2_exit": ["open_space_2"],
        "passage_3_exit": ["open_space_3"]
    }

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

        self._id_to_curriculum_prefix_map: Dict[int, str] = {v: k for k, v in self.CLUSTER_MAP.items()}
        
        allowed_id_transitions_list: List[Tuple[int, int]] = []
        for start_specific, end_specific in self._CURRICULUM_TRANSITIONS_STR:
            start_general = self._SPECIFIC_TO_GENERAL_MAP.get(start_specific, None)
            end_general = self._SPECIFIC_TO_GENERAL_MAP.get(end_specific, None)

            if start_general is not None and end_general is not None:
                start_id = self.CLUSTER_MAP.get(start_general, -1)
                end_id = self.CLUSTER_MAP.get(end_general, -1)

                if start_id != -1 and end_id != -1:
                    allowed_id_transitions_list.append((start_id, end_id))
                else:
                    print(f"Warning: Cluster map missing IDs for {start_general} or {end_general}.")
            else:
                print(f"Warning: Specific to general map missing entry for {start_specific} or {end_specific}.")

        if allowed_id_transitions_list:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.array(allowed_id_transitions_list, dtype=jnp.int32)
        else:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.empty((0, 2), dtype=jnp.int32)

        default_inter_params = [(
            jnp.array([self.area_size / 2, self.area_size / 2]),
            0.5, 0.5, 0.0, True
        )]

        self.associator = BehaviorIntersection(
            intersections=default_inter_params,
            all_region_names=self.ALL_POSSIBLE_REGION_NAMES
        )

        # Directly store the consistent mappings from the associator
        self._name_to_id = self.associator.region_name_to_id
        self._id_to_name = self.associator.region_id_to_name
        self.all_region_centroids_jax_array = self.associator.all_region_centroids_jax_array
        
        # Use the consistent mapping to prepare the curriculum
        self._prepare_transition_centroids()

        # Create the specific-to-specific transitions array using the consistent ID mapping
        valid_transitions_list = []
        for start_name, goal_names in self.VALID_GOAL_MAP.items():
            if start_name in self._name_to_id:
                start_id = self._name_to_id[start_name]
                for goal_name in goal_names:
                    if goal_name in self._name_to_id:
                        goal_id = self._name_to_id[goal_name]
                        # Exclude transitions involving open_space
                        valid_transitions_list.append((start_id, goal_id))
        
        # Store the specific-to-specific transitions for the reset function
        self.VALID_TRANSITIONS_INT_IDS = jnp.array(valid_transitions_list, dtype=jnp.int32)
        # jd.print("valid trans: {}", self.VALID_TRANSITIONS_INT_IDS)
        # Also create a mapping from specific to general IDs
        sorted_unique_names = self.ALL_POSSIBLE_REGION_NAMES #sorted(list(self._name_to_id.keys()))
        # jd.print("associatr labels: {}", self.associator.sorted_region_names)
        # jd.print("labels: {}", sorted_unique_names)
        self._specific_id_to_general_id = jnp.array([
            self.CLUSTER_MAP.get(self._SPECIFIC_TO_GENERAL_MAP.get(name)) for name in sorted_unique_names
        ], dtype=jnp.int32)
        
    def _prepare_transition_centroids(self):
        # This method is called from __init__
        valid_transitions_list = []
        
        for start_name, goal_names in self.VALID_GOAL_MAP.items():
            if start_name in self._name_to_id:
                start_id = self._name_to_id[start_name]
                for goal_name in goal_names:
                    if goal_name in self._name_to_id:
                        goal_id = self._name_to_id[goal_name]
                        
                        # Look up the centroids using the consistent ID-to-centroid mapping
                        start_centroid = self.associator.all_region_centroids_jax_array[start_id]
                        goal_centroid = self.associator.all_region_centroids_jax_array[goal_id]

                        # Store the centroids as a pair
                        valid_transitions_list.append((start_centroid, goal_centroid))

        # Convert the list to a JAX array
        self.all_region_centroid_pairs_jax_array = jnp.array(valid_transitions_list)

    def _get_valid_transitions_as_ids(self):
        valid_transitions = []
        for start_specific, goals_specific in self.VALID_GOAL_MAP.items():
            if start_specific in self._name_to_id:
                start_id = self._name_to_id[start_specific]
                for goal_specific in goals_specific:
                    if goal_specific in self._name_to_id:
                        goal_id = self._name_to_id[goal_specific]
                        if start_id != self._name_to_id["open_space"] and goal_id != self._name_to_id["open_space"]:
                            valid_transitions.append((start_id, goal_id))
        return jnp.array(valid_transitions, dtype=jnp.int32)
        
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
        key: jr.PRNGKey,
        current_clusters: Optional[Array] = None,
        start_clusters: Optional[Array] = None,
        next_clusters: Optional[Array] = None,
        custom_obstacles: Optional[Tuple[Array, Array, Array, Array]] = None,
        transition_index: Optional[int] = None
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
        
        inter_center_env_state: Float[Array, "2"] = jnp.array([0.0, 0.0])
        passage_width_env_state: Float[Array, ""] = jnp.array(0.0)
        obs_len_env_state: Float[Array, ""] = jnp.array(0.0)
        global_angle_env_state: Float[Array, ""] = jnp.array(0.0)
        initial_bonus_awarded = jnp.zeros(self.num_agents, dtype=jnp.bool_)
        is_four_way_env_state: bool = False
        num_inter = 1 #self._params.get("num_intersections", 0)

        if custom_obstacles is not None:
            obs_pos, obs_len_x, obs_len_y, obs_theta = custom_obstacles
            all_obs_pos_list.append(obs_pos)
            all_obs_len_x_list.append(obs_len_x)
            all_obs_len_y_list.append(obs_len_y)
            all_obs_theta_list.append(obs_theta)
        else:
            n_rng_obs = self._params.get("n_obs", 0)
            assert n_rng_obs >= 0
            
            if n_rng_obs > 0:
                obstacle_key, key = jr.split(key, 2)
                obs_pos_orig = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
                length_key, theta_key, key = jr.split(key, 3)
                obs_len_orig = jr.uniform(
                    length_key,
                    (n_rng_obs, 2),
                    minval=self._params["obs_len_range"][0],
                    maxval=self._params["obs_len_range"][1],
                )
                obs_theta_orig = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 *jnp.pi)
                all_obs_pos_list.append(obs_pos_orig)
                all_obs_len_x_list.append(obs_len_orig[:, 0])
                all_obs_len_y_list.append(obs_len_orig[:, 1])
                all_obs_theta_list.append(obs_theta_orig)

            intersection_key, key = jr.split(key)
            is_four_way_key, center_key, theta_key, pass_key, obs_len_key = jr.split(intersection_key, 5)

            is_four_way_choice = 4 #jr.choice(is_four_way_key, jnp.array([3,4]))
            area_size = LidarEnv.PARAMS["default_area_size"] 
            
            if num_inter > 0:
                # inter_rand_key, key = jr.split(key)
                # center_key, pass_key, obs_len_key, theta_key = jr.split(inter_rand_key, 4)

                passage_width = jr.uniform(pass_key, (), minval=self._params["passage_width_range"][0], maxval=self._params["passage_width_range"][1])
                obs_len = jr.uniform(obs_len_key, (), minval=self._params["obs_wall_range"][0], maxval=self._params["obs_wall_range"][1])
                global_angle = jr.uniform(theta_key, (), minval=-jnp.pi/6, maxval=jnp.pi/6)
                
                inter_center = jr.uniform(center_key, (2,),
                                            minval=jnp.array([self.area_size*0.3, self.area_size*0.7]),
                                            maxval=jnp.array([self.area_size*0.3, self.area_size*0.7]))            
                
                def three_way_branch(args):
                    inter_center, passage_width, obs_len, global_angle = args
                    obs_pos, obs_len_x, obs_len_y, obs_theta = create_threeway_intersection(
                        self.area_size,
                        passage_width,
                        obs_len,
                        global_angle
                    )
                    dummy_pos = jnp.array([1000.0, 1000.0])
                    dummy_len_x = 0.0
                    dummy_len_y = 0.0
                    dummy_theta = 0.0

                    obs_pos = jnp.concatenate([obs_pos, dummy_pos[None, :]], axis=0)
                    obs_len_x = jnp.concatenate([obs_len_x, jnp.array([dummy_len_x])], axis=0)
                    obs_len_y = jnp.concatenate([obs_len_y, jnp.array([dummy_len_y])], axis=0)
                    obs_theta = jnp.concatenate([obs_theta, jnp.array([dummy_theta])], axis=0)

                    return False, obs_pos, obs_len_x, obs_len_y, obs_theta

                def four_way_branch(args):
                    inter_center, passage_width, obs_len, global_angle = args
                    obs_pos, obs_len_x, obs_len_y, obs_theta = create_fourway_intersection(
                        self.area_size,
                        passage_width,
                        obs_len,
                        global_angle
                    )
                    return True, obs_pos, obs_len_x, obs_len_y, obs_theta

                is_four_way_env_state, obs_pos, obs_len_x, obs_len_y, obs_theta = jax.lax.cond(
                    is_four_way_choice == 3, # This is the boolean condition
                    three_way_branch, # This is the function for the true branch
                    four_way_branch, # This is the function for the false branch
                    (inter_center, passage_width, obs_len, global_angle) # These are the arguments passed to the functions
                )
                
                all_obs_pos_list.append(obs_pos)
                all_obs_len_x_list.append(obs_len_x)
                all_obs_len_y_list.append(obs_len_y)
                all_obs_theta_list.append(obs_theta)

                inter_center_env_state = inter_center
                passage_width_env_state = passage_width
                obs_len_env_state = obs_len
                global_angle_env_state = global_angle

        # Combine all obstacles and create the final obstacle object
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

        if num_inter > 0:
            inter_params_for_associator = [(
                inter_center_env_state, 
                passage_width_env_state, 
                obs_len_env_state, 
                global_angle_env_state,
                is_four_way_env_state
            )]
            
            # In a real environment, you would filter the region names here
            if is_four_way_choice == 4:
                associator = BehaviorIntersection(
                    intersections=inter_params_for_associator,
                    all_region_names=self.ALL_POSSIBLE_REGION_NAMES,
                )
            else:
                # Pass a filtered list of region names for a 3-way intersection
                filtered_names = [name for name in self.ALL_POSSIBLE_REGION_NAMES if not name.startswith("passage_3_")]
                associator = BehaviorIntersection(
                    intersections=inter_params_for_associator,
                    all_region_names=filtered_names,
                )

            # Use the associator to select a valid curriculum transition
            valid_transitions_list = []
            for start_name, goal_names in self.VALID_GOAL_MAP.items():
                if start_name in associator.region_name_to_id:
                    start_id = associator.region_name_to_id[start_name]
                    for goal_name in goal_names:
                        if goal_name in associator.region_name_to_id:
                            goal_id = associator.region_name_to_id[goal_name]
                            valid_transitions_list.append((start_id, goal_id))
            
            VALID_TRANSITIONS_INT_IDS_local = jnp.array(valid_transitions_list)
        
            key_select, key_loop = jr.split(key)
            transition_idx = jr.choice(key_select, jnp.arange(VALID_TRANSITIONS_INT_IDS_local.shape[0]))
            start_specific_id, goal_specific_id = VALID_TRANSITIONS_INT_IDS_local[transition_idx]
            
            # --- Start Cluster ID Logic ---
            start_cluster_id = self._specific_id_to_general_id[start_specific_id]
            
            # Get the initial position based on the start region
            initial_pos = self._get_initial_position(key_loop, associator, start_specific_id)

            # --- Goal Cluster ID Logic ---
            next_cluster_id = self._specific_id_to_general_id[goal_specific_id]
            
            # Get the goal position based on the goal region
            key_loop, key_goal_pos = jr.split(key_loop)
            goal_pos = self._get_initial_position(key_goal_pos, associator, goal_specific_id)
            
            #jd.print("current: {}, start: {}, end: {}", start_cluster_id, start_cluster_id, next_cluster_id)
            
            agent_states_list = []
            goal_states_list = []
            bearings_list = []
            current_clusters_oh_list = []
            start_clusters_oh_list = []
            next_clusters_oh_list = []
            
            for i in range(self.num_agents):
                key_agent_pos, key_goal_pos, key_rot, key_loop = jr.split(key_loop, 4)

                initial_pos_agent = initial_pos + jr.normal(key_agent_pos, (2,)) * 0.025
                goal_pos_agent = goal_pos + jr.normal(key_goal_pos, (2,)) * 0.025
                
                initial_pos_agent = jnp.clip(initial_pos_agent, 0, self.area_size)
                goal_pos_agent = jnp.clip(goal_pos_agent, 0, self.area_size)

                rot_angle = jr.uniform(key_rot, (), minval=-jnp.pi/4, maxval=jnp.pi/4)

                cos_rot = jnp.cos(rot_angle)
                sin_rot = jnp.sin(rot_angle)
                
                rotation_matrix = jnp.array([
                    [cos_rot, -sin_rot],
                    [sin_rot, cos_rot]
                ])

                goal_pos_relative = goal_pos_agent - inter_center_env_state
                rotated_goal_pos_relative = jnp.dot(rotation_matrix, goal_pos_relative)
                rotated_goal_pos = inter_center_env_state + rotated_goal_pos_relative
                
                goal_pos_agent = rotated_goal_pos

                bearing = jnp.arctan2(goal_pos_agent[1] - initial_pos_agent[1], goal_pos_agent[0] - initial_pos_agent[0])
                
                current_cluster_oh = jax.nn.one_hot(start_cluster_id, self.n_cluster)
                start_cluster_oh = jax.nn.one_hot(start_cluster_id, self.n_cluster)
                next_cluster_oh = jax.nn.one_hot(next_cluster_id, self.n_cluster)

                agent_states_list.append(jnp.array([initial_pos_agent[0], initial_pos_agent[1], 0.0, 0.0]))
                goal_states_list.append(jnp.array([goal_pos_agent[0], goal_pos_agent[1], 0.0, 0.0]))
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

            
        else:
            print(f"Warning: No intersections found.")
            
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
            is_four_way=is_four_way_env_state,
            center=inter_center_env_state,
            passage_width=passage_width_env_state,
            obs_len=obs_len_env_state,
            global_angle=global_angle_env_state
        )

        lidar_data = self.get_lidar_data(states, obstacles)
        return self.get_graph(env_states, lidar_data)
    
    def _get_initial_position(self, key, associator, specific_id):
        # The key is no longer needed since we are not adding noise at this step
        # The `associator` now provides a valid centroid for all region types
        return associator.get_region_centroid(specific_id)
        
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
        """By default, use double integrator dynamics"""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)


    def step(
        self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        agent_base_states = graph.env_states.agent
        goals = graph.env_states.goal
        obstacles = graph.env_states.obstacle
        
        is_four_way = graph.env_states.is_four_way
        inter_center = graph.env_states.center
        passage_width = graph.env_states.passage_width
        obs_len = graph.env_states.obs_len
        global_angle = graph.env_states.global_angle
        bearing = graph.env_states.bearing
        
        inter_params = [(
            inter_center,
            passage_width,
            obs_len,
            global_angle,
            is_four_way
        )]
        
        associator = BehaviorIntersection(
            intersections=inter_params,
            all_region_names=self.ALL_POSSIBLE_REGION_NAMES
        )
        
        current_id = self._specific_id_to_general_id[jax_vmap(associator.get_current_behavior_direction)(agent_base_states[:, :2], agent_base_states[:, 2:4])]
        current_cluster_oh = jax.nn.one_hot(current_id, self.n_cluster)
        start_cluster_oh = graph.env_states.start_cluster_oh        
        next_cluster_oh = graph.env_states.next_cluster_oh
        jd.print("current:{}, start: {}, end: {}", current_id, jnp.argmax(start_cluster_oh), jnp.argmax(next_cluster_oh))
        action = self.clip_action(action)
        next_agent_base_states = self.agent_step_euler(agent_base_states, action)

        reward, bonus_awarded_updated = self.get_reward(graph, action) #, bonus_awarded_state)
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
            is_four_way=is_four_way,
            center=inter_center,
            passage_width=passage_width,
            obs_len=obs_len,
            global_angle=global_angle
        )
        
        lidar_data_next = self.get_lidar_data(next_agent_base_states, obstacles)
        info = {}
        done = jnp.array(False)

        return self.get_graph(next_env_state, lidar_data_next), reward, cost, done, info
    
    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward: #, bonus_awarded_state: jnp.ndarray
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

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
        
        first_env_state = rollout.graph.env_states
        intersection_params_for_render = {
            "is_four_way": first_env_state.is_four_way,
            "center": first_env_state.center,
            "passage_width": first_env_state.passage_width,
            "obs_len": first_env_state.obs_len,
            "global_angle": first_env_state.global_angle,
        }
        
        render_lidar(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2, # Assuming 2D for buildings environment visualization
            n_agent=self.num_agents,
            n_rays=self.params["top_k_rays"] if self.params["n_obs"] > 0 or self.params["num_buildings"] > 0 else 0,
            r=self.params["car_radius"],
            obs_r=0.0,
            cost_components=self.cost_components,
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            n_goal=self.num_goals,
            dpi=dpi,
            **intersection_params_for_render, # Pass the extracted building parameters explicitly
            **kwargs
        )

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
