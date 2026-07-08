"""
Plan loading and cluster-id mapping.
No ROS, no Spot SDK.
"""

import json
import os
from typing import Dict, List, Tuple

PLAN_FILE = "plans/hallway.json"

# Maps raw land-cover classifier IDs → canonical region IDs:
#   0 = open_space, 1 = approach_0, 2 = on_0, 3 = exit_0
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
        centroids:     {cluster_id_str: [x, y, ...]}
    """
    if not os.path.exists(plan_file_path):
        return [], {}, {}
    with open(plan_file_path, "r") as f:
        data = json.load(f)
    return data.get("plan_sequence", []), data.get("bearing_map", {}), data.get("centroids", {})


def map_cluster_id(raw_id: int) -> int:
    """Map raw classifier output → canonical cluster ID. Passthrough if unknown."""
    return _CLUSTER_MAP.get(raw_id, raw_id)
