import jax.numpy as jnp
import jax.debug as jdebug

from typing import Optional, Dict, Tuple
import functools as ft

from ...utils.graph import EdgeBlock
from ...utils.typing import Action, Array, Pos2d, Reward, State
from ..lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from ...utils.utils import jax_vmap


ALL_POSSIBLE_REGION_NAMES = [
        "open_space",
        "approach_bridge_0",
        "on_bridge_0",
        "exit_bridge_0"
    ]

def _calculate_bearing_reward(agent_vel: jnp.ndarray, target_bearing: float) -> float:
    target_vector = jnp.array([jnp.cos(target_bearing), jnp.sin(target_bearing)])

    norm_agent_vel = jnp.linalg.norm(agent_vel)
    safe_norm_agent_vel = jnp.where(norm_agent_vel > 1e-6, norm_agent_vel, 1e-6)

    cosine_sim = jnp.dot(agent_vel, target_vector) / safe_norm_agent_vel

    return cosine_sim

def _calculate_cluster_reward_per_agent(
    current_cluster_oh_episode_i: Array,
    start_cluster_oh_episode_i: Array,
    next_cluster_oh_episode_i: Array,
    has_been_awarded: jnp.ndarray,
    # Reward coefficients
    next_cluster_bonus: float,
    stay_in_cluster_bonus: float,
    incorrect_cluster_penalty: float

) -> float:

    current_cluster_id_episode_i = jnp.argmax(current_cluster_oh_episode_i)
    start_cluster_id_episode_i = jnp.argmax(start_cluster_oh_episode_i)
    next_cluster_id_episode_i = jnp.argmax(next_cluster_oh_episode_i)

    current_agent_reward = 0.0

    has_moved_into_next_cluster = (current_cluster_id_episode_i == next_cluster_id_episode_i) & \
                                 (current_cluster_id_episode_i != start_cluster_id_episode_i)

    # `give_bonus` is true only on the first step the agent moves into the next cluster
    give_bonus = has_moved_into_next_cluster & (~has_been_awarded)

    # Award the one-time bonus correctly
    bonus_reward = jnp.where(give_bonus, next_cluster_bonus, 0.0)

    # Update the flag to prevent future bonuses
    updated_has_been_awarded = jnp.logical_or(has_been_awarded, give_bonus)

    current_agent_reward += bonus_reward
    jdebug.print("step1: {}", current_agent_reward)

    # This is the line that was missing. It awards a continuous bonus for staying in the target cluster.
    is_in_next_cluster = (current_cluster_id_episode_i == next_cluster_id_episode_i)
    current_agent_reward += jnp.where(is_in_next_cluster, stay_in_cluster_bonus, 0.0)

    is_incorrect_path_taken = (current_cluster_id_episode_i != start_cluster_id_episode_i) & \
                             (current_cluster_id_episode_i != next_cluster_id_episode_i)
    jdebug.print("step2: {}", current_agent_reward)
    current_agent_reward += jnp.where(
        is_incorrect_path_taken,
        incorrect_cluster_penalty,
        0.0
    )
    jdebug.print("step3: {}", current_agent_reward)
    return current_agent_reward, updated_has_been_awarded

class LidarTarget(LidarEnv):

    # COSINE_SIM_REWARD_COEFF = 0.25
    # NEXT_CLUSTER_BONUS = 30.0      # Significant bonus for reaching target cluster
    # INCORRECT_CLUSTER_PENALTY = -1.0 # Large penalty for moving to an unauthorized cluster
    # STAY_IN_CLUSTER_BONUS = 0.3 # A small, continuous bonus for staying in the target cluster
    # VELOCITY_PENALTY_IN_CLUSTER = -0.5

    COSINE_SIM_REWARD_COEFF = 0.1
    NEXT_CLUSTER_BONUS = 4.0      # Significant bonus for reaching target cluster
    INCORRECT_CLUSTER_PENALTY = -2.0 # Large penalty for moving to an unauthorized cluster
    STAY_IN_CLUSTER_BONUS = 0.2 # A small, continuous bonus for staying in the target cluster
    VELOCITY_PENALTY_IN_CLUSTER = -2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 2,
        "default_area_size": 1.5,
        "dist2goal": 0.01, # This value is now largely unused for reward unless repurposed
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

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Tuple[Reward, jnp.ndarray]:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        agent_vel = agent_states[:, 2:4]

        # --- Extract necessary info from env_states ---
        env_states = graph.env_states
        current_cluster_oh_episode = env_states.current_cluster_oh # Per-agent current high-level cluster OH
        start_cluster_oh_episode = env_states.start_cluster_oh
        next_cluster_oh_episode = env_states.next_cluster_oh     # Per-agent next high-level cluster OH
        actual_cluster_id = jnp.argmax(current_cluster_oh_episode, axis=-1)
        bonus_awarded_state = env_states.next_cluster_bonus_awarded

        bridge_center = env_states.bridge_center
        bridge_length = env_states.bridge_length
        bridge_gap_width = env_states.bridge_gap_width
        bridge_wall_thickness = env_states.bridge_wall_thickness
        bridge_theta = env_states.bridge_theta # In radians

        bearing = env_states.bearing

        # Check if the agent is in the next cluster
        next_cluster_id = jnp.argmax(next_cluster_oh_episode, axis=-1)
        is_in_next_cluster = (actual_cluster_id == next_cluster_id)

        dense_reward = jnp.where(
            ~is_in_next_cluster,  # Condition: If NOT in the next cluster
            jax_vmap(_calculate_bearing_reward, in_axes=(0, 0))(agent_vel, bearing).mean(),
            0.0 # If in the next cluster, the reward is 0.0
        ).mean()
        reward = dense_reward * self.COSINE_SIM_REWARD_COEFF
        jdebug.print("dense: {}", reward)
    
        bridge_length = 10
        is_bridge_env = bridge_length > 0.0

        vmap_cluster_reward_fn = jax_vmap(ft.partial(
            _calculate_cluster_reward_per_agent,
            stay_in_cluster_bonus=self.STAY_IN_CLUSTER_BONUS,
            next_cluster_bonus=self.NEXT_CLUSTER_BONUS,
            incorrect_cluster_penalty=self.INCORRECT_CLUSTER_PENALTY,
        ), in_axes=(0, 0, 0, 0)) # vmap over agent_pos_i, current, start, next, and bonus_awarded_state

        # The function call should NOT include next_cluster_bonus again
        per_agent_reward, per_agent_bonus_awarded_updated = vmap_cluster_reward_fn(
            current_cluster_oh_episode,
            start_cluster_oh_episode,
            next_cluster_oh_episode,
            bonus_awarded_state
        )

        # Use jnp.where separately for each output
        reward_cluster = jnp.where(is_bridge_env, per_agent_reward, 0.0)
        next_cluster_bonus_awarded_updated = jnp.where(is_bridge_env, per_agent_bonus_awarded_updated, bonus_awarded_state)

        reward += jnp.mean(reward_cluster)
        jdebug.print("cluster: {}", jnp.mean(reward_cluster))

        velocity_magnitude_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        velocity_penalty = jnp.where(is_in_next_cluster, velocity_magnitude_sq, 0.0).mean()
        reward += velocity_penalty * self.VELOCITY_PENALTY_IN_CLUSTER
        jdebug.print("vel: {}", velocity_penalty * self.VELOCITY_PENALTY_IN_CLUSTER)

        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.001

        return reward*0.1, next_cluster_bonus_awarded_updated

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

