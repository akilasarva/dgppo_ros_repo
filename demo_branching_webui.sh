#!/usr/bin/env bash
# End-to-end branching demo wired to the dgppo_debug_visualizer_v2 web UI.
#
# What this brings up (all on your local ROS domain):
#   - nl_planner (planner_node + executor_node + mission_bridge)  ← LLM pipeline
#   - brain_controller (tree-aware v2)                            ← walks tree
#   - dgppo_debug_visualizer_v2 in --web mode                     ← Mission Card UI
#   - demo_sim.py                                                 ← synthetic
#       /predicted_cluster + camera frames so brain advances through whatever
#       plan the planner produces
#
# Usage:
#   set -a; source /home/racecar/racecar_ws/src/dgppo_ros_repo/.env; set +a
#   /home/racecar/racecar_ws/demo_branching_webui.sh
#
# Then open http://localhost:8765 in a browser, type a mission like
#   "Drive to the open space; if you see an obstacle, detour through the
#    intersection, otherwise go straight."
# in the Mission Card, and click "Generate plan".

set -eo pipefail
# Note: no `-u` — ROS's setup.bash trips on unbound AMENT_TRACE_SETUP_FILES.

WS=/home/racecar/racecar_ws
ENV_FILE=$WS/src/dgppo_ros_repo/.env
TAXONOMY=$WS/install/nl_planner/share/nl_planner/config/cluster_map.livox1.yaml
PORT=${PORT:-8765}

# --- Source ROS + workspace + .env ----------------------------------------- #
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[demo] sourced env from $ENV_FILE (OPENAI_API_KEY=${OPENAI_API_KEY:+set})"
else
  echo "[demo] WARN: $ENV_FILE not found. The LLM pipeline will refuse to run." >&2
fi

if [[ ! -f "$TAXONOMY" ]]; then
  echo "[demo] FATAL: taxonomy not found at $TAXONOMY" >&2
  echo "[demo]        Did you run colcon build for nl_planner?" >&2
  exit 1
fi

# Headless matplotlib for the visualizer's web-only mode so it doesn't try
# to grab a Tk display the container may not have.
export MPLBACKEND=Agg

LOG_DIR=$(mktemp -d /tmp/demo_branching.XXXXXX)
echo "[demo] logs in $LOG_DIR"

pids=()

cleanup() {
  echo
  echo "[demo] shutting down..."
  # Kill each tracked child AND any descendant process group it spawned
  # (ros2 run launches the node as a separate child that pkill -P catches).
  for pid in "${pids[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 0.7
  for pid in "${pids[@]}"; do
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  done
  # Belt-and-suspenders: catch any orphaned grandchildren.
  pkill -KILL -f 'nl_planner/lib/nl_planner'  2>/dev/null || true
  pkill -KILL -f 'brain/lib/brain'            2>/dev/null || true
  pkill -KILL -f 'demo_sim.py'                2>/dev/null || true
  pkill -KILL -f 'debug_visualizer_v2'        2>/dev/null || true
  echo "[demo] logs preserved in $LOG_DIR"
}
trap cleanup EXIT INT TERM

run_bg() {
  local label=$1; shift
  local logfile=$LOG_DIR/$label.log
  echo "[demo] starting $label -> $logfile"
  # setsid puts the child in its own process group so cleanup can pkill -P it.
  setsid "$@" > "$logfile" 2>&1 &
  pids+=($!)
}

# --- 1. nl_planner stack --------------------------------------------------- #
run_bg planner_node ros2 run nl_planner planner_node \
  --ros-args \
  -p taxonomy_path:="$TAXONOMY" \
  -p model:=openai:gpt-4o \
  -p max_attempts:=5 \
  -p verify_syntax:=true \
  -p verify_tripartite:=true

run_bg executor_node ros2 run nl_planner executor_node \
  --ros-args \
  -p taxonomy_path:="$TAXONOMY" \
  -p brain_load_plan_timeout_s:=5.0

run_bg mission_bridge ros2 run nl_planner mission_bridge

# --- 2. brain (tree-aware) ------------------------------------------------- #
# image_topic matches the simulator below and the visualizer's existing sub.
run_bg brain_controller ros2 run brain brain_controller \
  --ros-args \
  -p image_topic:=/hamilton_zed2i/zed_node/rgb/image_rect_color \
  -p vlm_check_interval:=1.5 \
  -p vlm_model:=gpt-4o \
  -p vlm_decide_max_attempts:=2 \
  -p vlm_decide_backoff_s:=0.5 \
  -p plan_snapshot_path:="$WS/src/dgppo_ros_repo/brain/plan.json"

# --- 3. cluster + camera simulator ---------------------------------------- #
SIM_PY=$WS/demo_sim.py
run_bg demo_sim python3 "$SIM_PY"

# --- 4. visualizer (web only) --------------------------------------------- #
run_bg visualizer python3 -m dgppo_ros_node_pkg.dgppo_debug_visualizer_v2 \
  --web --port "$PORT"

# Give Flask a moment to bind.
sleep 2

cat <<EOF

==============================================================================
[demo] All nodes started.
[demo] Open the visualizer in your browser:

    http://localhost:$PORT

[demo] In the "Mission Card" at the top, type something like:

    Drive from the road into the intersection. If the intersection
    looks blocked, come back to the road; otherwise pass through to
    the open space.

[demo] Click "Generate plan". You should see:
[demo]   1. mission_bridge phase: planning -> ok (LLM converged)
[demo]   2. executor logs in $LOG_DIR/executor_node.log show the tree shipped
[demo]   3. brain_controller logs in $LOG_DIR/brain_controller.log show
[demo]      NAVIGATING -> ... -> DECIDING -> branch chosen -> ... -> COMPLETE
[demo]   4. demo_sim.log shows the synthetic cluster transitions it feeds in
[demo]
[demo] Tail any log live with:  tail -F $LOG_DIR/brain_controller.log
[demo] Ctrl-C here to shut everything down.
==============================================================================
EOF

# Wait for any child to exit (Ctrl-C handled by trap).
wait
