import jax.numpy as jnp
import jax.debug as jdebug

from typing import Optional, Dict
import functools as ft

from dgppo.utils.graph import EdgeBlock
from dgppo.utils.typing import Action, Array, Pos2d, Reward, State
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from dgppo.utils.utils import jax_vmap

import functools as ft
from typing import Optional, Dict

def _calculate_cluster_reward_per_agent(
    agent_pos_i: Array,
    current_cluster_oh_episode_i: Array,
    next_cluster_oh_episode_i: Array,
    # New building parameters needed to reconstruct BehaviorAssociator
    building_center: Array,
    building_width: float,
    building_height: float,
    building_theta: float,
    environment_area_size: float,
    # LidarEnv class attributes needed for logic (passed from LidarEnv instance)
    cluster_map: Dict[str, int],
    id_to_curriculum_prefix_map: Dict[int, str],
    allowed_id_transitions_array: Array,
    # Reward coefficients
    next_cluster_bonus: float,
    stay_in_cluster_bonus: float,
    incorrect_cluster_penalty: float
) -> float:

    from dgppo.env.plot import BehaviorBuildings

    current_cluster_id_episode_i = jnp.argmax(current_cluster_oh_episode_i)
    next_cluster_id_episode_i = jnp.argmax(next_cluster_oh_episode_i)
    
    # NEW: Building parameters for the associator
    building_params = [(
        building_center,
        building_width,
        building_height,
        building_theta
    )]

    # NEW: Instantiate the BehaviorBuildings associator
    associator = BehaviorBuildings(
        buildings=building_params,
        region_name_to_id=cluster_map,
        region_id_to_name=id_to_curriculum_prefix_map
    )

    actual_cluster_id = associator.get_current_behavior(agent_pos_i)

    current_agent_reward = 0.0

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
    NEXT_CLUSTER_BONUS = 4.0
    INCORRECT_CLUSTER_PENALTY = -2.0
    STAY_IN_CLUSTER_BONUS = 0.2

    # NEW: Update PARAMS for the building environment
    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        
        # New building-related params
        "num_buildings": 1,
        "num_bridges": 0, # Ensure this is set to 0
        "building_width_range": [0.3, 0.6],
        "building_height_range": [0.3, 0.6],
        "building_theta_range": [0, 2 * jnp.pi],
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
        current_cluster_oh_episode = env_states.current_cluster_oh
        next_cluster_oh_episode = env_states.next_cluster_oh
        
        # NEW: Extract building parameters instead of bridge parameters
        building_center = env_states.building_center
        building_width = env_states.building_width
        building_height = env_states.building_height
        building_theta = env_states.building_theta

        # --- 2. Cluster Bonus/Penalty ---
        # NEW: Apply only if buildings were generated
        is_building_env = building_width > 0.0
        
        vmap_cluster_reward_fn = jax_vmap(ft.partial(
            _calculate_cluster_reward_per_agent,
            # Pass new building parameters
            building_center=building_center,
            building_width=building_width,
            building_height=building_height,
            building_theta=building_theta,
            environment_area_size=self.area_size,
            cluster_map=self.CLUSTER_MAP,
            id_to_curriculum_prefix_map=self._id_to_curriculum_prefix_map,
            allowed_id_transitions_array=self.CURRICULUM_TRANSITIONS_INT_IDS,
            stay_in_cluster_bonus=self.STAY_IN_CLUSTER_BONUS,
            next_cluster_bonus=self.NEXT_CLUSTER_BONUS,
            incorrect_cluster_penalty=self.INCORRECT_CLUSTER_PENALTY
        ), in_axes=(0, 0, 0))

        reward_cluster = jnp.where(is_building_env,
            jnp.mean(vmap_cluster_reward_fn(
                agent_states[:, :2],
                current_cluster_oh_episode,
                next_cluster_oh_episode
            )),
            0.0 # If no building, cluster rewards are zero
        )
        reward += reward_cluster

        # goal distance penalty
        reward -= (dist2goal.mean()) * 0.01
        
        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001
        
        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward * 0.1
    
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

