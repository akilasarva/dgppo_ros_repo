"""
Plan loading, cluster mapping, and initial state utilities.
No ROS, no Spot SDK, no DGPPO.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple

PLAN_FILE = "plans/bridge.json"

# Maps raw land-cover classifier IDs → canonical bridge region IDs:
#   0 = open_space, 1 = approach_bridge_0, 2 = on_bridge_0, 3 = exit_bridge_0
_CLUSTER_MAP: Dict[int, int] = {
    **{k: 0 for k in [0, 1]},
    **{k: 1 for k in [2, 3, 10, 11]},
    **{k: 2 for k in [5, 6, 7, 8, 9, 12]},
    **{k: 3 for k in [-1, 4]},
}


def load_plan(plan_file_path: str = PLAN_FILE) -> Tuple[List, Dict, Dict]:
    """
    Load the navigation plan JSON.

    Returns:
        plan_sequence: list of {"start": int, "next": int} steps
        bearing_map:   {"start-next": float} bearing angles in radians
        centroids:     {cluster_id_str: [fwd_m, lat_m, ...]}
    """
    if not os.path.exists(plan_file_path):
        return [], {}, {}
    with open(plan_file_path, "r") as f:
        data = json.load(f)
    return data.get("plan_sequence", []), data.get("bearing_map", {}), data.get("centroids", {})


def map_cluster_id(raw_id: int) -> int:
    """Map raw classifier output → canonical bridge cluster ID. Passthrough if unknown."""
    return _CLUSTER_MAP.get(raw_id, raw_id)


def initial_sim_state(
    centroids: Dict,
    start_cluster_id,
    scale: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> np.ndarray:
    """
    Compute initial agent state [pos_x, pos_y, 0, 0] in sim frame from start centroid.

    Centroid format: [forward_m, lateral_m, ...].
    """
    c = centroids.get(str(start_cluster_id), [0.0, 0.0, 0.0])
    pos_x = (c[1] - origin_y) / scale  # lateral → sim x
    pos_y = (c[0] - origin_x) / scale  # forward → sim y
    return np.array([pos_x, pos_y, 0.0, 0.0], dtype=np.float32)
