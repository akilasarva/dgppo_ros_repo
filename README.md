# HOW TO RUN CARLA SIM:

## Simulation
sudo docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen

## Bridge
- docker run -it --rm --name humble_dev_with_code --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/<user>/ros2_ws:/ros_ws -v /home/<user>/modified_dgppo/:/dgppo --- user=root carla-ros-bridge-dev:latest /bin/bash
- source /carla_ws/install/setup.bash
- ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py

## Live cluster inference node
- docker exec -it humble_dev_with_code /bin/bash
- source /ros_ws/install/setup.bash
- cd /ros_ws/src/clustering/clustering
- ros2 run clustering live_cluster_inference_node

## DGPPO rosnode
- docker exec -it humble_dev_with_code /bin/bash
- source /ros_ws/install/setup.bash
- cd /ros_ws/src/dgppo_ros_node_pkg/dgppo_ros_node_pkg
- ros2 run dgppo_ros_node_pkg dgppo_ros_node

## Twist to Control
- docker exec -it humble_dev_with_code /bin/bash
- source /carla_ws/install/setup.bash
- cd /carla_ws
- ros2 run carla_twist_to_control carla_twist_to_control
