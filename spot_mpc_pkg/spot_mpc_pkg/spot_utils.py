"""
Spot SDK interface: state reading, transform extraction, velocity commands.
No ROS imports.
"""

import numpy as np
from typing import Tuple

from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_se2_a_tform_b,
)
from bosdyn.client.robot_command import RobotCommandBuilder

from .utils import rot2d


def get_spot_state(robot_state) -> Tuple[float, float, float, float, float]:
    """
    Extract (x, y, vx, vy, yaw) from a Spot RobotState proto, all in vision frame.

    Position (x, y): body origin in vision frame (meters).
    Velocity (vx, vy): body velocity in vision frame (m/s).
    Yaw: body orientation in vision frame (radians) from the SE2 transform.
    Z axis is ignored throughout.

    Args:
        robot_state: bosdyn RobotState proto

    Returns:
        (x, y, vx, vy, yaw) as plain floats
    """
    kinematic_state = robot_state.kinematic_state
    transforms_snapshot = kinematic_state.transforms_snapshot
    assert str(transforms_snapshot) != "", "transforms_snapshot is empty"

    tform = get_se2_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    x   = float(tform.x)
    y   = float(tform.y)
    vx  = float(kinematic_state.velocity_of_body_in_vision.linear.x)
    vy  = float(kinematic_state.velocity_of_body_in_vision.linear.y)
    yaw = float(tform.angle)
    return x, y, vx, vy, yaw


def get_vision_to_body_rotation(transforms_snapshot) -> np.ndarray:
    """
    Return the 2x2 rotation matrix that transforms vision-frame 2D vectors to body frame.

    Extracts body yaw from the SE2 body-in-vision transform, then returns rot2d(-yaw).
    Z axis and translation are ignored.

    Args:
        transforms_snapshot: bosdyn KinematicState.transforms_snapshot proto

    Returns:
        2x2 numpy array R such that v_body = R @ v_vision
    """
    tform = get_se2_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    yaw = float(tform.angle)
    return rot2d(-yaw)


def send_velocity(
    command_client,
    vx: float,
    vy: float,
    end_time_secs: float,
) -> None:
    """
    Send a body-frame planar velocity command to Spot with no rotation.

    Args:
        command_client: bosdyn RobotCommandClient
        vx: forward body velocity (m/s)
        vy: left body velocity (m/s)
        end_time_secs: absolute expiry time (time.time() + horizon)
    """
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=vx, v_y=vy, v_rot=0.0)
    command_client.robot_command(command=cmd, end_time_secs=end_time_secs)
