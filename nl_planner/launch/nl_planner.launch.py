"""Launch the nl_planner planner + executor nodes together.

Usage:
    ros2 launch nl_planner nl_planner.launch.py \\
      taxonomy:=$(ros2 pkg prefix nl_planner)/share/nl_planner/config/cluster_map.livox1.yaml

Required launch args:
    taxonomy   absolute path to cluster_map.<env>.yaml

Optional launch args:
    model              pydantic-ai model id (default openai:gpt-4o-mini)
    vlm_model          OpenAI vision model for branch decisions (default gpt-4o-mini)
    image_topic        camera topic the executor caches for VLM calls
    max_attempts       generator retry cap (default 3)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    taxonomy = LaunchConfiguration("taxonomy")
    model = LaunchConfiguration("model")
    vlm_model = LaunchConfiguration("vlm_model")
    image_topic = LaunchConfiguration("image_topic")
    max_attempts = LaunchConfiguration("max_attempts")

    return LaunchDescription([
        DeclareLaunchArgument(
            "taxonomy",
            description="Absolute path to cluster_map.<env>.yaml.",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="openai:gpt-4o-mini",
            description="pydantic-ai provider:model id for the LLM agents.",
        ),
        DeclareLaunchArgument(
            "vlm_model",
            default_value="gpt-4o-mini",
            description="OpenAI vision model used at branch decision points.",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/hamilton/hamilton_zed/rgb/image_rect_color",
            description="Camera topic the executor caches for VLM calls.",
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
                "vlm_model": vlm_model,
                "image_topic": image_topic,
                "brain_load_plan_timeout_s": 5.0,
            }],
        ),
    ])
