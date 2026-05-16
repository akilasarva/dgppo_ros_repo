# `nl_planner`

ROS 2 package that turns an English mission into a verified Signal Temporal
Logic (STL) formula plus a tree-shaped cluster-plan, and feeds linear segments
of that plan to `brain_controller` at runtime.

```
  /nl_planner/plan (srv)
       |               +----------- pydantic-ai Agents ---------+
       v               |                                          |
  +-------------+   gen +----> syntax verify  +----> tripartite verify --+
  | planner_node|<-----+                                                |
  +-------------+                                                       |
       | latched NavPlan JSON                                           |
       v                                                                |
  +-------------+   /brain/incoming_plan (latched String) +--------+
  | executor_   |--------------------------------------> | brain_  |
  |   node      |   /brain/load_plan (std_srvs/Trigger)  | controller|
  |             |--------------------------------------> +----------+
  +-----+-------+          /brain/state ("COMPLETE")      ^
        |   /predicted_cluster (diag), camera image        |
        |   VLM choose_branch() at decision points         |
        +---------- DISPATCHING -> WAITING -> VLM ---------+
```

## Replaces the hand-authored plan workflow

Before this package: `brain_controller` loaded a hand-written `plan.json`
(linear, no branches) and stepped through it. Plans had to be re-authored by
hand for every mission. `brain_controller` is unchanged in that mode — the
new `/brain/incoming_plan` + `/brain/load_plan` interface is purely additive,
~50 lines, and uses only `std_msgs` / `std_srvs`.

After this package:
1. Author / edit prompts in `nl_planner/nl_planner/prompts/*.md`.
2. Hand-tune `nl_planner/config/cluster_map.<env>.yaml` so the semantic mode
   names the LLM uses (`"Road: On"`, `"Intersection: In"`, …) resolve to the
   correct HDBSCAN cluster ids.
3. At runtime, send an English mission to `/nl_planner/plan`. The pipeline
   generates -> verifies -> retries up to 3x, then publishes the NavPlan tree
   on `/nl_planner/dispatch`. The executor materializes the next linear
   segment, ships it to `brain`, and at every decision point asks an OpenAI
   vision model which branch to follow.

## Layout

```
nl_planner/
  schemas.py             Pydantic contract (NavPlan, Branch, PlanStep, ...)
  prompts/
    generator.md         Prompt 1 (English -> filtered + plan + STL)
    verify_syntax.md     Prompt 2a (STL syntax)
    verify_tripartite.md Prompt 2b (English <-> STL <-> JSON)
    __init__.py          load_prompt("generator") -> str
  agents.py              pydantic-ai Agent factories
  pipeline.py            generate -> verify -> revise loop
  taxonomy.py            ClusterTaxonomy YAML loader + plan-mode validator
  branch_materializer.py NavPlan tree -> brain-shaped linear segments
  vlm_client.py          OpenAI vision helpers (confirm_cue, choose_branch)
  cli.py                 `nl_planner` CLI (no ROS)
  bootstrap/
    seed_taxonomy.py     cluster_id_to_label_*.json -> starter YAML
  nodes/
    planner_node.py      /nl_planner/plan service + dispatch publisher
    executor_node.py     dispatch sub + /brain hot-swap + VLM branch picker

config/
  cluster_map.livox1.yaml semantic mode -> [cluster_ids]
  nl_planner.yaml         default ROS params

launch/
  nl_planner.launch.py    planner + executor together

test/
  test_schemas.py
  test_taxonomy_mapper.py
  test_branch_materializer.py
  test_prompts_smoke.py

../nl_planner_msgs/srv/GeneratePlan.srv   (sibling package, ament_cmake)
```

## Install

```bash
# Workspace deps (only the pure-python ones; ROS deps come from package.xml + rosdep)
pip install 'pydantic-ai-slim[openai]' 'pydantic>=2.5' 'pyyaml>=6.0' 'openai>=1.40'

# Build (from workspace root, with brain + clustering checked out)
colcon build --packages-select nl_planner_msgs nl_planner brain
source install/setup.bash
```

## Offline (no ROS, no robot)

```bash
# 1. Seed the per-env taxonomy from the clustering team's label file:
ros2 run nl_planner seed_taxonomy \
  --labels clustering/clustering/encoder_weights/livox1/cluster_id_to_label_livox1.json \
  --out    nl_planner/config/cluster_map.livox1.yaml
# (hand-edit nl_planner/config/cluster_map.livox1.yaml so the keys match the
#  Prompt-1 vocabulary -- e.g. rename "In Corridor" -> "Road: On".)

# 2. Generate a plan from English (writes tree NavPlan JSON to plan.json,
#    prints filtered command + STL formula + attempt count to stderr):
export OPENAI_API_KEY=sk-...
nl_planner \
  --mission "drive forward and turn right at the stop sign" \
  --taxonomy nl_planner/config/cluster_map.livox1.yaml \
  --model openai:gpt-4o-mini \
  --out plan.json
```

## Full stack (rosbag or live)

```bash
# Terminal A: brain (unchanged; placeholder plan.json is fine, it'll be hot-swapped)
ros2 launch brain brain.launch

# Terminal B: clustering
ros2 run clustering live_cluster_inference_node

# Terminal C: nl_planner planner + executor
ros2 launch nl_planner nl_planner.launch.py \
  taxonomy:=$PWD/nl_planner/config/cluster_map.livox1.yaml

# Terminal D: rosbag for camera + lidar
ros2 bag play <bag>

# Trigger a mission:
ros2 service call /nl_planner/plan nl_planner_msgs/srv/GeneratePlan \
  "{mission: 'cross the bridge, take the longer road if blocked'}"
```

Expected flow:
1. `planner_node` runs generate -> syntax verify -> tripartite verify, retries
   if either verifier rejects, and publishes the tree NavPlan on
   `/nl_planner/dispatch` (transient_local, so order doesn't matter).
2. `executor_node` materializes the **root linear segment**, latches it on
   `/brain/incoming_plan`, then calls `/brain/load_plan` Trigger.
3. `brain_controller` swaps in the new plan, sets `state=NAVIGATING`, and runs
   it. When `state=COMPLETE` is published, executor either decides a branch
   (calling OpenAI vision) or marks the plan DONE.

## Tests

```bash
pip install pytest
pytest nl_planner/test/
```

Tests use only `pydantic`, `pyyaml`, and `pytest` — no LLM / ROS / network
calls. Real LLM tests would be gated behind `RUN_LIVE_LLM_TESTS=1` (not
implemented in this package).
