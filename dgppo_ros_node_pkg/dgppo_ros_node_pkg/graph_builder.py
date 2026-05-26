"""
Standalone graph construction for DGPPO inference.
No ROS node, no `self` — all dependencies passed explicitly.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

from .dgppo.dgppo.env.lidar_env.base import LidarEnvState
from .dgppo.dgppo.utils.graph import GraphsTuple
from .utils import resample_lidar, lidar_angles


def build_graph(
    env_instance,
    agent_state_np: np.ndarray,
    scaled_ranges: np.ndarray,
    current_cluster_id: int,
    start_cluster_id: int,
    next_cluster_id: int,
    bonus_awarded: jnp.ndarray,
    terrain_id: int,
    bearing_map: dict,
    num_clusters: int,
    n_rays_phys: int,
    world_alpha_rad: float,
    yaw: float,
    angular_offset_rad: float = 0.0,
    logger=None,
) -> GraphsTuple:
    """
    Build a GraphsTuple for DGPPO inference from the current agent and sensor state.

    Args:
        env_instance: DGPPO environment object (provides params, get_graph)
        agent_state_np: (1,4) array [pos_x, pos_y, vel_x, vel_y] in sim frame
        scaled_ranges: (n_rays_phys,) lidar ranges already divided by SCALE_SPOT_TO_SIM
        current_cluster_id: mapped canonical cluster the robot is currently in
        start_cluster_id: mapped canonical cluster at the start of this plan step
        next_cluster_id: mapped canonical cluster at the end of this plan step
        bonus_awarded: (n_agents,) bool array tracking next-cluster bonus
        terrain_id: current terrain label (Road=0, Grass=1, Sidewalk=2)
        bearing_map: {"start-next": float} bearing angles from the plan JSON
        num_clusters: total number of canonical clusters
        n_rays_phys: number of physical lidar bins
        world_alpha_rad: CW angle (from above) from Spot boot-forward to sim +Y
        yaw: body yaw in vision frame (radians)
        angular_offset_rad: additional bearing offset (radians, from angular_offset_deg param)
        logger: optional ROS logger for debug output

    Returns:
        GraphsTuple ready for algo.act()
    """
    if logger:
        logger.info(f"Agent state (scaled): {agent_state_np}", throttle_duration_sec=0.5)

    n_rays = env_instance.params['n_rays']  # 32
    agent_pos_2d = np.array(agent_state_np[0, :2])

    # ── 1. Obstacle hits: resample n_rays_phys bins → n_rays training beams ──
    ranges_res = resample_lidar(
        scaled_ranges,
        n_rays=n_rays,
        n_rays_phys=n_rays_phys,
        world_alpha_rad=world_alpha_rad,
        yaw=yaw,
    )
    angles_beam = lidar_angles(n_rays)
    _min_idx = int(np.argmin(ranges_res))
    if logger:
        logger.info(
            f'[LIDAR] min_range={ranges_res[_min_idx]:.2f}sim  '
            f'beam_idx={_min_idx}/32  angle_deg={math.degrees(angles_beam[_min_idx]):.0f}°  '
            f'(0°=right  90°=fwd  ±180°=left)',
            throttle_duration_sec=1.0,
        )
    obs_hits = np.stack([
        agent_pos_2d[0] + ranges_res * np.cos(angles_beam),
        agent_pos_2d[1] + ranges_res * np.sin(angles_beam),
    ], axis=1).astype(np.float32)  # (n_rays, 2)

    # ── 2. Terrain boundary hits: zeros (geometry not wired yet) ─────────────
    bnd_hits = np.zeros((n_rays, 2), dtype=np.float32)

    # ── 3. Flat semantic lidar arrays ─────────────────────────────────────────
    all_hit_positions = np.concatenate([obs_hits, bnd_hits], axis=0)  # (2*n_rays, 2)
    all_terrain_ids   = np.ones(2 * n_rays, dtype=np.int32)           # Grass default

    # ── 4. One-hots ──────────────────────────────────────────────────────────
    current_terrain_oh  = jax.nn.one_hot(terrain_id, 3)              # (3,)
    current_cluster_oh  = jax.nn.one_hot(current_cluster_id, num_clusters)
    start_cluster_oh    = jax.nn.one_hot(start_cluster_id,   num_clusters)
    next_cluster_oh     = jax.nn.one_hot(next_cluster_id,    num_clusters)

    # ── 5. Bearing ────────────────────────────────────────────────────────────
    key = f"{start_cluster_id}-{next_cluster_id}"
    bearing_value = bearing_map.get(key, 0.0) + math.pi / 2 + angular_offset_rad
    if logger:
        logger.info(
            f"Start:{start_cluster_id} Cur:{current_cluster_id} "
            f"Next:{next_cluster_id} Bearing:{bearing_value:.3f}",
            throttle_duration_sec=0.5,
        )

    env_state = LidarEnvState(
        agent=agent_state_np,
        goal=jnp.array([np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)]),
        obstacle=None,
        bearing=jnp.array([bearing_value]),
        current_cluster_oh=jnp.array([current_cluster_oh]),
        start_cluster_oh=jnp.array([start_cluster_oh]),
        next_cluster_oh=jnp.array([next_cluster_oh]),
        next_cluster_bonus_awarded=bonus_awarded,
        current_terrain_oh=jnp.array([current_terrain_oh]),
        lidar_hit_terrain_ids=jnp.array(all_terrain_ids),
        lidar_hit_positions=jnp.array(all_hit_positions),
        bridge_center=jnp.zeros(2),
        bridge_length=jnp.array(0.0),
        bridge_gap_width=jnp.array(0.0),
        bridge_wall_thickness=jnp.array(0.0),
        bridge_theta=jnp.array(0.0),
        bridge_bend_angle=jnp.array(0.0),
        terrain_config=jnp.array(1, dtype=jnp.int32),
    )

    lidar_data_batched = jnp.array(all_hit_positions[np.newaxis, :, :])  # (1, 2*n_rays, 2)
    return env_instance.get_graph(env_state, lidar_data_batched)
