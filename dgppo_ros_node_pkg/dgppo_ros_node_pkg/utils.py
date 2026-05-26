"""
Pure math utilities for coordinate transforms and lidar processing.
No ROS, no Spot SDK.

Frame conventions:
  Spot vision frame : +X forward, +Y left
  DGPPO sim frame   : +X right,   +Y forward
  Scale             : 11 Spot meters = 1 sim unit

Fixed transforms (90° rotation viewed from above, i.e. z-down):
  VISION_TO_DGPPO_R = rot2d(+pi/2)  — vision → DGPPO (90° CW from above)
  DGPPO_TO_VISION_R = rot2d(-pi/2)  — DGPPO → vision (90° CCW from above)
"""

import math
import numpy as np
from typing import Tuple

SCALE_SPOT_TO_SIM: int = 11

# Fixed 90° rotation vision-frame → DGPPO-sim-frame (CW from above = +pi/2 in standard math)
VISION_TO_DGPPO_R: np.ndarray = np.array(
    [[0.0, -1.0],
     [1.0,  0.0]], dtype=np.float64
)

# Inverse: DGPPO-sim-frame → vision-frame (CCW from above = -pi/2 in standard math)
DGPPO_TO_VISION_R: np.ndarray = np.array(
    [[ 0.0, 1.0],
     [-1.0, 0.0]], dtype=np.float64
)


def rot2d(theta: float) -> np.ndarray:
    """2x2 CCW rotation matrix for angle theta (radians, standard math convention)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def apply_rot2d(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply a 2x2 rotation matrix to a 2-element vector. Returns R @ v."""
    return R @ v


def lidar_phys_angles(n_rays_phys: int) -> np.ndarray:
    """
    Physical Spot lidar bin angles [0, 2pi), Spot body frame.
    Bin 0 is at +X (forward), bins increase CCW.
    """
    return np.linspace(0.0, 2.0 * np.pi, n_rays_phys, endpoint=False)


def lidar_angles(n_rays: int) -> np.ndarray:
    """
    Training beam angles [-pi, pi), DGPPO sim world frame.
    0 = +X (right), pi/2 = +Y (forward).
    """
    return np.linspace(-np.pi, np.pi - 2.0 * np.pi / n_rays, n_rays)


def resample_lidar(
    scaled_ranges: np.ndarray,
    n_rays: int,
    n_rays_phys: int,
    world_alpha_rad: float,
    yaw: float,
) -> np.ndarray:
    """
    Resample physical Spot lidar bins into n_rays training beams.

    For each training beam at world-frame angle theta_beam, looks up the
    physical sensor bin:
        phi_sensor = (theta_beam - pi/2 - world_alpha_rad - yaw) mod 2*pi

    Args:
        scaled_ranges: (n_rays_phys,) ranges already divided by SCALE_SPOT_TO_SIM
        n_rays: number of training beams
        n_rays_phys: number of physical sensor bins
        world_alpha_rad: CW angle (from above) from Spot boot-forward to sim +Y
        yaw: body yaw in vision frame (radians)

    Returns:
        (n_rays,) resampled ranges in sim units
    """
    angles_phys = lidar_phys_angles(n_rays_phys)
    angles_beam = lidar_angles(n_rays)
    lookup = np.mod(angles_beam - np.pi / 2.0 - world_alpha_rad - yaw, 2.0 * np.pi)
    return np.interp(lookup, angles_phys, scaled_ranges)
