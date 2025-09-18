# dgppo/env/lidar_env/lidar_target.py

import jax.numpy as jnp
import jax.debug as jdebug

from typing import Optional, Dict
import functools as ft

from dgppo.utils.graph import EdgeBlock
from dgppo.utils.typing import Action, Array, Pos2d, Reward, State
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from dgppo.utils.utils import jax_vmap

def _calculate_cluster_reward_per_agent(
    agent_pos_i: Array,
    current_cluster_oh_episode_i: Array,
    next_cluster_oh_episode_i: Array,
    # Bridge parameters needed to reconstruct BehaviorAssociator
    bridge_center: Array,
    bridge_length: float, # Overall bridge length
    bridge_gap_width: float,
    bridge_wall_thickness: float,
    bridge_theta: float, # In radians
    environment_area_size: float,
    # LidarEnv class attributes needed for logic (passed from LidarEnv instance)
    cluster_map: Dict[str, int],
    id_to_curriculum_prefix_map: Dict[int, str], # Python dict
    allowed_id_transitions_array: Array, # JAX array of (start_id, end_id) pairs
    # NEW: Next cluster centroid for reward shaping
    next_cluster_centroid: Array, # Added parameter for continuous shaping
    # Reward coefficients
    next_cluster_bonus: float,
    stay_in_cluster_bonus: float,
    incorrect_cluster_penalty: float
) -> float:

    from dgppo.env.plot import BehaviorAssociator # Keep this import here as it's used within this function

    current_cluster_id_episode_i = jnp.argmax(current_cluster_oh_episode_i)
    next_cluster_id_episode_i = jnp.argmax(next_cluster_oh_episode_i)

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
    
    associator = BehaviorAssociator(
        bridges=bridge_walls_params, 
        buildings=[],
        obstacles=[],
        region_name_to_id=cluster_map, 
        region_id_to_name=id_to_curriculum_prefix_map 
    )

    actual_cluster_id = associator.get_current_behavior(agent_pos_i)

    current_agent_reward = 0.0

    # NEW: Continuous penalty based on distance to the *next target cluster centroid*
    # This provides a dense reward signal for getting closer to the goal cluster.
    dist_to_next_cluster_centroid = jnp.linalg.norm(agent_pos_i - next_cluster_centroid)
    # Scale this penalty. Adjust coefficient (0.5 here) based on desired impact.
    current_agent_reward -= dist_to_next_cluster_centroid * 0.5


    has_moved_into_next_cluster = (actual_cluster_id == next_cluster_id_episode_i) & \
                                 (actual_cluster_id != current_cluster_id_episode_i)

    current_agent_reward += jnp.where(has_moved_into_next_cluster, next_cluster_bonus, 0.0)
    
    is_in_next_cluster = (actual_cluster_id == next_cluster_id_episode_i)
    current_agent_reward += jnp.where(is_in_next_cluster, stay_in_cluster_bonus, 0.0)

    has_moved_from_initial = (actual_cluster_id != current_cluster_id_episode_i)
    is_not_target = (actual_cluster_id != next_cluster_id_episode_i)

    potential_transition_id_pair = jnp.array([current_cluster_id_episode_i, actual_cluster_id])

    is_allowed_curriculum_transition = jnp.any(
        jnp.all(allowed_id_transitions_array == potential_transition_id_pair[None, :], axis=-1)
    )

    is_incorrect_path_taken = has_moved_from_initial & is_not_target & (~is_allowed_curriculum_transition)

    current_agent_reward += jnp.where(
        is_incorrect_path_taken,
        incorrect_cluster_penalty,
        0.0
    )

    return current_agent_reward

class LidarTarget(LidarEnv):

    COSINE_SIM_REWARD_COEFF = 0.1
    # UPDATED: Increased bonus/penalty magnitudes
    NEXT_CLUSTER_BONUS = 8.0      # Significant bonus for reaching target cluster
    INCORRECT_CLUSTER_PENALTY = -4.0 # Large penalty for moving to an unauthorized cluster
    STAY_IN_CLUSTER_BONUS = 0.25 # A small, continuous bonus for staying in the target cluster

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        
        # Ensure bridge-related params are also in LidarEnv's PARAMS for reset() to use
        "num_bridges": 1,
        "bridge_length_range": [0.5, 1.0],
        "bridge_gap_width_range": [0.2, 0.4],
        "bridge_wall_thickness_range": [0.05, 0.1]
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarTarget.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarTarget, self).__init__(num_agents, area_size, max_step, dt, params)
        
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        
        # --- Extract necessary info from env_states ---
        env_states = graph.env_states
        current_cluster_oh_episode = env_states.current_cluster_oh # Per-agent current high-level cluster OH
        next_cluster_oh_episode = env_states.next_cluster_oh     # Per-agent next high-level cluster OH

        # Bridge parameters needed for BehaviorAssociator (from env_states)
        bridge_center = env_states.bridge_center
        bridge_length = env_states.bridge_length
        bridge_gap_width = env_states.bridge_gap_width
        bridge_wall_thickness = env_states.bridge_wall_thickness
        bridge_theta = env_states.bridge_theta # In radians
        # NEW: Get the next_cluster_centroid
        next_cluster_centroid_from_env = env_states.next_cluster_centroid_for_all_agents

        # --- 2. Cluster Bonus/Penalty ---
        # Apply only if bridges were generated in the environment (signified by bridge_length > 0)
        is_bridge_env = bridge_length > 0.0
        
        vmap_cluster_reward_fn = jax_vmap(ft.partial(
            _calculate_cluster_reward_per_agent,
            # Pass bridge parameters required by BehaviorAssociator
            bridge_center=bridge_center,
            bridge_length=bridge_length,
            bridge_gap_width=bridge_gap_width,
            bridge_wall_thickness=bridge_wall_thickness,
            bridge_theta=bridge_theta,
            environment_area_size=self.area_size,
            cluster_map=self.CLUSTER_MAP,
            id_to_curriculum_prefix_map=self._id_to_curriculum_prefix_map, 
            allowed_id_transitions_array=self.CURRICULUM_TRANSITIONS_INT_IDS,
            # NEW: Pass the next cluster centroid to the vmapped function
            next_cluster_centroid=next_cluster_centroid_from_env,
            stay_in_cluster_bonus=self.STAY_IN_CLUSTER_BONUS,
            next_cluster_bonus=self.NEXT_CLUSTER_BONUS,
            incorrect_cluster_penalty=self.INCORRECT_CLUSTER_PENALTY
        ), in_axes=(0, 0, 0)) # vmap over agent_pos_i, current_cluster_oh_episode_i, next_cluster_oh_episode_i

        reward_cluster = jnp.where(is_bridge_env,
            jnp.mean(vmap_cluster_reward_fn(
                agent_states[:, :2], # agent_pos_i for each agent
                current_cluster_oh_episode, # current_cluster_oh_episode_i for each agent
                next_cluster_oh_episode # next_cluster_oh_episode_i for each agent
            )),
            0.0 # If no bridge, cluster rewards are zero
        )
        reward += reward_cluster

        # goal distance penalty - REDUCED COEFF
        reward -= (dist2goal.mean()) * 0.005 # Reduced from 0.01
        
        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001
        
        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        # REMOVED: Overall scaling factor (reward * 0.1)
        return reward 
    
    def state2feat(self, state: State) -> Array:
        return state

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.agent)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # # agent - goal connection
        agent_goal_edges = []
        # for i_agent in range(self.num_agents):
        #     agent_state_i = state.agent[i_agent]
        #     goal_state_i = state.goal[i_agent]
        #     agent_goal_feats_i = self.state2feat(agent_state_i) - self.state2feat(goal_state_i)
        #     agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
        #                                       jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])))

        # agent - obs connection
        agent_obs_edges = []
        n_hits = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges
