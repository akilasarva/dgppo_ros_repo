import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional, List
from abc import ABC, abstractmethod

from jaxtyping import Float

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ...utils.utils import merge01, jax_vmap
from ..base import MultiAgentEnv
from dgppo.env.obstacle import Obstacle, Rectangle
from dgppo.env.utils import get_lidar, get_node_goal_rng


# --- NEW: Bridge Creation Function ---
def create_single_bridge(
    bridge_center: jnp.ndarray, # (2,) for x, y
    bridge_length: float,
    bridge_gap_width: float,
    bridge_wall_thickness: float,
    bridge_theta: float
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


# --- MODIFIED: LidarEnvState to include bridge parameters ---
class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Optional[Obstacle] # Obstacles can be None if n_obs=0
    
    # NEW: Bridge specific parameters
    bridge_center: Float[Array, "2"]
    bridge_length: float
    bridge_gap_width: float
    bridge_wall_thickness: float
    bridge_theta: float

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]

LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]

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
        # NEW: Bridge Parameters added to PARAMS
        "num_bridges": 1, # Default to 1 bridge - Important for this scenario
        "bridge_length_range": [0.5, 1.0],
        "bridge_gap_width_range": [0.2, 0.4],
        "bridge_wall_thickness_range": [0.05, 0.1],
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarEnv.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarEnv, self).__init__(num_agents, area_size, max_step, dt, params)
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.num_goals = self._num_agents

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        # REVERTED: Original node_dim (state dim (4) + indicator: agent: 001, goal: 010, obstacle: 100)
        return 7

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

    def reset(self, key: Array) -> GraphsTuple:
        all_obs_pos_list: List[jnp.ndarray] = []
        all_obs_len_x_list: List[jnp.ndarray] = []
        all_obs_len_y_list: List[jnp.ndarray] = []
        all_obs_theta_list: List[jnp.ndarray] = []

        # Initialize bridge parameters to default/zero values in case no bridge is generated
        bridge_center_env_state: Float[Array, "2"] = jnp.zeros(2)
        bridge_length_env_state: float = 0.0
        bridge_gap_width_env_state: float = 0.0
        bridge_wall_thickness_env_state: float = 0.0
        bridge_theta_env_state: float = 0.0

        # Retrieve num_bridges from PARAMS, defaulting to 0 if not specified
        num_bridges_to_generate = 1 #self._params.get("num_bridges", 0) 

        # Generate random obstacles if n_rng_obs > 0
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0

        if n_rng_obs > 0:
            obstacle_key, key = jr.split(key, 2)
            obs_pos_orig = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            length_key, key = jr.split(key, 2)
            obs_len_orig = jr.uniform(
                length_key,
                (n_rng_obs, 2), # Corrected: should be n_rng_obs, not self._params["n_obs"]
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key, 2)
            obs_theta_orig = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * jnp.pi)

            all_obs_pos_list.append(obs_pos_orig)
            all_obs_len_x_list.append(obs_len_orig[:, 0])
            all_obs_len_y_list.append(obs_len_orig[:, 1])
            all_obs_theta_list.append(obs_theta_orig) 
            
        # NEW: Bridge generation logic
        if num_bridges_to_generate > 0:
            # Assuming only 1 bridge for simplicity as in the inspiration code
            if num_bridges_to_generate != 1:
                print("Warning: Currently, `LidarEnv` only directly supports the generation of 1 bridge via specific parameter storage. Multiple bridges will be generated but only the last one's parameters will be stored in `env_states`.")

            # Split key for bridge parameters
            bridge_rand_key, key = jr.split(key)
            center_key, length_key, gap_key, thickness_key, theta_key = jr.split(bridge_rand_key, 5)

            # Sample bridge parameters from defined ranges
            bridge_center = jr.uniform(center_key, (2,), minval=0, maxval=self.area_size)
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

            # Create bridge obstacle parameters using the new function
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

            # Store the single bridge's parameters for env_state, as there's only one slot
            bridge_center_env_state = bridge_center
            bridge_length_env_state = bridge_length
            bridge_gap_width_env_state = bridge_gap_width
            bridge_wall_thickness_env_state = bridge_wall_thickness
            bridge_theta_env_state = bridge_theta
            
        # Combine all generated obstacles into a single Obstacle object
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

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles)
        states = jnp.concatenate(
            [states, jnp.zeros((self.num_agents, self.state_dim - states.shape[1]), dtype=states.dtype)], axis=1)
        goals = jnp.concatenate(
            [goals, jnp.zeros((self.num_goals, self.state_dim - goals.shape[1]), dtype=goals.dtype)], axis=1)

        assert states.shape == (self.num_agents, self.state_dim)
        assert goals.shape == (self.num_goals, self.state_dim)
        
        # MODIFIED: Pass bridge parameters to LidarEnvState
        env_states = LidarEnvState(
            agent=states, 
            goal=goals, 
            obstacle=obstacles,
            bridge_center=bridge_center_env_state,
            bridge_length=bridge_length_env_state,
            bridge_gap_width=bridge_gap_width_env_state,
            bridge_wall_thickness=bridge_wall_thickness_env_state,
            bridge_theta=bridge_theta_env_state,
        )

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        lidar_data = None
        # Condition changed to also check for num_bridges > 0, as bridge creates obstacles
        if self.params["n_obs"] > 0 or self._params.get("num_bridges", 0) > 0:
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
        # get information from graph
        agent_base_states = graph.env_states.agent # Get the underlying agent state from env_states
        goals = graph.env_states.goal
        obstacles = graph.env_states.obstacle

        # NEW: Pass bridge parameters through
        bridge_center = graph.env_states.bridge_center
        bridge_length = graph.env_states.bridge_length
        bridge_gap_width = graph.env_states.bridge_gap_width
        bridge_wall_thickness = graph.env_states.bridge_wall_thickness
        bridge_theta = graph.env_states.bridge_theta

        # calculate next states
        action = self.clip_action(action)
        next_agent_base_states = self.agent_step_euler(agent_base_states, action) # Only update (x,y,vx,vy)

        # Reconstruct the next LidarEnvState, passing bridge parameters
        next_env_state = LidarEnvState(
            next_agent_base_states, 
            goals, 
            obstacles,
            bridge_center,
            bridge_length,
            bridge_gap_width,
            bridge_wall_thickness,
            bridge_theta,
        )
        
        lidar_data_next = self.get_lidar_data(next_agent_base_states, obstacles)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        return self.get_graph(next_env_state, lidar_data_next), reward, cost, done, info

    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params['n_obs'] == 0 and self._params.get("num_bridges", 0) == 0: # Check for both types of obstacles
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
        # Import render_lidar here if it's only used in this method and to avoid circular imports
        from dgppo.env.plot import render_lidar 

        # Extract bridge parameters from the first frame's env_states in the rollout
        # We assume bridge parameters are constant throughout a rollout.
        first_env_state = rollout.graph.env_states
        bridge_params_for_render = {
            "bridge_center": first_env_state.bridge_center[0], # Take the first center from the batch
            "bridge_length": float(first_env_state.bridge_length[0]), # Take the first length, then convert to float
            "bridge_gap_width": float(first_env_state.bridge_gap_width[0]), # Take the first, convert to float
            "bridge_wall_thickness": float(first_env_state.bridge_wall_thickness[0]), # Take the first, convert to float
            "bridge_theta": float(first_env_state.bridge_theta[0]), # Take the first, convert to float
        }
        print("before")
        print(f"DEBUG: bridges_for_associator: {bridge_params_for_render}")


        render_lidar(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2, # Assuming 2D for bridge environment visualization
            n_agent=self.num_agents,
            # NEW: Adjusted n_rays condition for rendering to include bridge obstacles
            n_obs = self.params["n_obs"],
            n_rays=self.params["top_k_rays"] if (self.params["n_obs"] > 0 or self.params.get("num_bridges", 0) > 0) else 0,
            r=self.params["car_radius"],
            obs_r=0.0, # Not directly used for Rectangle obstacles, can be 0 or removed
            cost_components=self.cost_components,
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            n_goal=self.num_goals,
            dpi=dpi,
            **bridge_params_for_render, # Pass the extracted bridge parameters explicitly
            **kwargs # Pass any other general kwargs
        )
        print("after")

    @abstractmethod
    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        pass

    def get_graph(self, state: LidarEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        # Condition changed to also check for num_bridges > 0, as bridge creates obstacles
        n_hits = self._params["top_k_rays"] * self.num_agents if (self.params["n_obs"] > 0 or self._params.get("num_bridges", 0) > 0) else 0
        n_nodes = self.num_agents + self.num_goals + n_hits

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        # node features
        # states
        node_feats = jnp.zeros((n_nodes, self.node_dim), dtype=jnp.float32) # Ensure dtype for consistency
        node_feats = node_feats.at[: self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[-n_hits:, :2].set(lidar_data)

        # indicators (remaining 3 dimensions of node_dim = 7)
        # Assuming last 3 dims are [is_obstacle, is_goal, is_agent]
        # is_obstacle indicator (index state_dim)
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits:, self.state_dim].set(1.) 
        # is_goal indicator (index state_dim + 1)
        node_feats = (
            node_feats.at[self.num_agents: self.num_agents + self.num_goals, self.state_dim + 1].set(1.))  
        # is_agent indicator (index state_dim + 2)
        node_feats = node_feats.at[: self.num_agents, self.state_dim + 2].set(1.) 

        # node type
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(LidarEnv.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(LidarEnv.GOAL)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(LidarEnv.OBS)

        # edge blocks
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        raw_states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states_padded = jnp.concatenate( # Use _padded to avoid confusion with raw_states
                [lidar_data, jnp.zeros((n_hits, self.state_dim - lidar_data.shape[1]), dtype=lidar_data.dtype)], axis=1)
            raw_states = jnp.concatenate([raw_states, lidar_states_padded], axis=0)
            
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state, # Pass the full LidarEnvState
            states=raw_states # This is the raw state array, not the extended node_feats
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim