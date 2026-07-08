"""
Pure math utilities for coordinate transforms.
No ROS, no Spot SDK.

Frame conventions:
  Spot vision frame : +X forward, +Y left, real metres
  Plan/centroid frame: same axes and units (metres) as vision frame, but
                       re-zeroed to whatever pose was active at the last
                       /mpc_reset_origin call (see spot_mpc_node.py)
"""

import math
import numpy as np


def rot2d(theta: float) -> np.ndarray:
    """2x2 CCW rotation matrix for angle theta (radians, standard math convention)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def apply_rot2d(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply a 2x2 rotation matrix to a 2-element vector. Returns R @ v."""
    return R @ v
