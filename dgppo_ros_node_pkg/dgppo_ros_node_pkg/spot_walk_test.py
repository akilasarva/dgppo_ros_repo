#!/usr/bin/env python3
"""
Minimal Spot connectivity + command test — no ROS, no policy.

Authenticates, takes the lease, sends a small forward velocity command
for WALK_DUR seconds, then stops.  Use this to verify the SDK setup in
spot_dgppo_ros_node.py works before running the full policy node.

Prerequisites:
  - Spot is powered on and standing (not sitting).
  - Tablet / E-stop is in a safe state (driver seat held or E-stop released).

Run from the workspace root:
  python3 src/dgppo_ros_node_pkg/dgppo_ros_node_pkg/spot_walk_test.py
"""

import time
import bosdyn.client
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient

SPOT_IP  = "10.0.0.3"
USERNAME = "dcist"
PASSWORD = "bbbdddaaaiii"

WALK_VX  = 0.2   # m/s forward — small enough to be safe
WALK_DUR = 1.0   # seconds


def main():
    sdk = bosdyn.client.create_standard_sdk("spot-walk-test")
    robot = sdk.create_robot(SPOT_IP)
    robot.authenticate(USERNAME, PASSWORD)
    print("Authenticated.")

    robot.time_sync.wait_for_sync()
    print("Time sync complete.")

    state_client   = robot.ensure_client("robot-state")
    lease_client   = robot.ensure_client("lease")
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    robot_state = state_client.get_robot_state()
    print(f"Robot state acquired: power={robot_state.power_state.motor_power_state}")

    lease_client.take()
    print("Lease taken.")

    with bosdyn.client.lease.LeaseKeepAlive(lease_client):
        print(f"Sending forward velocity v_x={WALK_VX} m/s for {WALK_DUR} s ...")
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=WALK_VX, v_y=0.0, v_rot=0.0
        )
        command_client.robot_command(cmd, end_time_secs=time.time() + 0.5)

        time.sleep(WALK_DUR)

        print("Stopping.")
        command_client.robot_command(RobotCommandBuilder.stop_command())

    print("Done.")


if __name__ == "__main__":
    main()
