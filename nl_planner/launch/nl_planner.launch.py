"""Launch the nl_planner planner + executor + mission-bridge nodes together.

Usage:
    ros2 launch nl_planner nl_planner.launch.py \\
      taxonomy:=$(ros2 pkg prefix nl_planner)/share/nl_planner/config/cluster_map.livox1.yaml

Required launch args:
    taxonomy           absolute path to cluster_map.<env>.yaml

Optional launch args:
    model              pydantic-ai model id (default openai:gpt-4o)
    max_attempts       generator retry cap (default 3)

Note: ``executor_node`` is now a thin shipper that converts the NavPlan tree
to brain-format JSON and ships it to ``/brain/incoming_plan``. The VLM branch
decisions happen inside ``brain_controller`` (v2 tree-aware), so the executor
no longer needs ``vlm_model`` or ``image_topic`` params.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    taxonomy = LaunchConfiguration("taxonomy")
    model = LaunchConfiguration("model")
    max_attempts = LaunchConfiguration("max_attempts")

    return LaunchDescription([
        DeclareLaunchArgument(
            "taxonomy",
            description="Absolute path to cluster_map.<env>.yaml.",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="openai:gpt-4o",
            description="pydantic-ai provider:model id for the LLM agents.",
        ),
        DeclareLaunchArgument(
            "max_attempts",
            default_value="3",
            description="Hard cap on generator retries.",
        ),

        Node(
            package="nl_planner",
            executable="planner_node",
            name="nl_planner",
            output="screen",
            parameters=[{
                "taxonomy_path": taxonomy,
                "model": model,
                "max_attempts": max_attempts,
                "verify_syntax": True,
                "verify_tripartite": True,
            }],
        ),
        Node(
            package="nl_planner",
            executable="executor_node",
            name="nl_planner_executor",
            output="screen",
            parameters=[{
                "taxonomy_path": taxonomy,
                "brain_load_plan_timeout_s": 5.0,
            }],
        ),
        Node(
            package="nl_planner",
            executable="mission_bridge",
            name="nl_planner_mission_bridge",
            output="screen",
        ),
    ])
