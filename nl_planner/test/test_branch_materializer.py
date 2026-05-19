"""Tests for branch_materializer.materialize_segment."""

from __future__ import annotations

import textwrap

import pytest

from nl_planner.branch_materializer import (
    materialize_segment,
    to_brain_tree,
    walk_all_segments,
)
from nl_planner.schemas import Branch, NavPlan, PlanStep
from nl_planner.taxonomy import load_taxonomy


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def taxonomy(tmp_path):
    p = tmp_path / "tax.yaml"
    p.write_text(textwrap.dedent("""
        environment: testenv
        modes:
          "Road: On":      [0, 1, 2]
          "Bridge: Enter": [10]
          "Bridge: On":    [11]
          "Bridge: Exit":  [12]
    """))
    return load_taxonomy(p)


def _step(idx, desc, start, goal, cue=None, branches=None):
    return PlanStep(
        step=idx, description=desc,
        start_mode=start, goal_mode=goal,
        transition_cue=cue, branches=branches,
    )


@pytest.fixture
def branch_plan():
    """Tree with one decision step."""
    return NavPlan(
        plan_name="bridge or detour", description="d",
        steps=[
            _step(0, "approach", "Road: On", "Bridge: Enter", cue="Detect(Bridge)"),
            _step(1, "decide", "Bridge: Enter", "Bridge: Enter", cue="dp", branches=[
                Branch(vlm_cue="bridge is blocked", sub_plan=[
                    _step(0, "detour", "Bridge: Enter", "Road: On"),
                ]),
                Branch(vlm_cue="default", sub_plan=[
                    _step(0, "cross", "Bridge: Enter", "Bridge: On"),
                    _step(1, "exit", "Bridge: On", "Road: On"),
                ]),
            ]),
        ],
    )


# --------------------------------------------------------------------------- #
# Root segment                                                                 #
# --------------------------------------------------------------------------- #

def test_root_segment_stops_before_decision(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=())

    # Only the leading linear step("approach") makes it into the brain plan.
    assert [s["description"] for s in seg.brain_plan["steps"]] == ["approach"]
    assert seg.brain_plan["steps"][0]["start_cluster"] == 0   # Road: On canonical id
    assert seg.brain_plan["steps"][0]["goal_cluster"] == 10  # Bridge: Enter canonical id
    assert seg.decision_step is not None
    assert seg.decision_step.description == "decide"
    assert not seg.is_terminal
    assert seg.path == ()


def test_root_segment_metadata(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=())
    from nl_planner.branch_materializer import attach_decision_metadata
    attach_decision_metadata(seg)
    meta = seg.brain_plan["nl_planner"]
    assert meta["ends_at_decision"] is True
    assert meta["decision"]["start_mode"] == "Bridge: Enter"
    assert [b["vlm_cue"] for b in meta["decision"]["branches"]] == [
        "bridge is blocked", "default",
    ]


# --------------------------------------------------------------------------- #
# Branch segments                                                              #
# --------------------------------------------------------------------------- #

def test_default_branch_terminates(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=(1,))
    assert [s["description"] for s in seg.brain_plan["steps"]] == ["cross", "exit"]
    assert seg.decision_step is None
    assert seg.is_terminal
    assert seg.path == (1,)


def test_blocked_branch_terminates(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=(0,))
    assert [s["description"] for s in seg.brain_plan["steps"]] == ["detour"]
    assert seg.is_terminal


def test_walk_enumerates_root_plus_each_branch(taxonomy, branch_plan):
    segs = walk_all_segments(branch_plan, taxonomy)
    paths = [list(s.path) for s in segs]
    assert paths == [[], [0], [1]]


# --------------------------------------------------------------------------- #
# Cluster labels                                                               #
# --------------------------------------------------------------------------- #

def test_cluster_labels_match_taxonomy(taxonomy, branch_plan):
    seg = materialize_segment(branch_plan, taxonomy, path=())
    labels = seg.brain_plan["cluster_labels"]
    assert labels["0"] == "Road: On"
    assert labels["10"] == "Bridge: Enter"
    assert labels["11"] == "Bridge: On"


# --------------------------------------------------------------------------- #
# Error paths                                                                  #
# --------------------------------------------------------------------------- #

def test_invalid_path_index_raises(taxonomy, branch_plan):
    with pytest.raises(ValueError):
        materialize_segment(branch_plan, taxonomy, path=(7,))


def test_path_past_terminal_raises(taxonomy, branch_plan):
    # default branch has no further decisions
    with pytest.raises(ValueError):
        materialize_segment(branch_plan, taxonomy, path=(1, 0))


# --------------------------------------------------------------------------- #
# to_brain_tree                                                                #
# --------------------------------------------------------------------------- #

def test_to_brain_tree_linear(taxonomy):
    plan = NavPlan(
        plan_name="linear", description="d",
        steps=[
            _step(0, "approach", "Road: On", "Bridge: Enter"),
            _step(1, "cross",    "Bridge: Enter", "Bridge: On"),
        ],
    )
    tree = to_brain_tree(plan, taxonomy)
    assert tree["plan_name"] == "linear"
    assert tree["nl_planner"]["tree_shaped"] is True
    assert tree["nl_planner"]["version"] == 2
    assert tree["nl_planner"]["environment"] == "testenv"
    assert len(tree["steps"]) == 2
    assert tree["steps"][0]["start_cluster"] == 0  # Road: On
    assert tree["steps"][0]["goal_cluster"] == 10  # Bridge: Enter
    assert tree["steps"][0]["start_mode"] == "Road: On"
    assert tree["steps"][0]["branches"] is None
    assert tree["steps"][1]["start_cluster"] == 10
    assert tree["steps"][1]["goal_cluster"] == 11
    # cluster_labels includes every taxonomy id, keyed as strings
    assert tree["cluster_labels"]["0"] == "Road: On"
    assert tree["cluster_labels"]["11"] == "Bridge: On"


def test_to_brain_tree_preserves_branches(taxonomy, branch_plan):
    tree = to_brain_tree(branch_plan, taxonomy)
    # Root list keeps both linear lead + decision step
    assert [s["description"] for s in tree["steps"]] == ["approach", "decide"]
    decision = tree["steps"][1]
    assert decision["start_cluster"] == 10  # Bridge: Enter
    assert decision["goal_cluster"] == 10
    assert decision["branches"] is not None
    cues = [b["vlm_cue"] for b in decision["branches"]]
    assert cues == ["bridge is blocked", "default"]
    # The 'default' branch's sub_plan must have two linear steps with
    # resolved cluster ids.
    default_branch = decision["branches"][1]
    assert default_branch["vlm_cue"] == "default"
    sub = default_branch["sub_plan"]
    assert [s["description"] for s in sub] == ["cross", "exit"]
    assert sub[0]["start_cluster"] == 10  # Bridge: Enter
    assert sub[0]["goal_cluster"]  == 11  # Bridge: On
    assert sub[0]["branches"]      is None


def test_to_brain_tree_round_trips_through_navigator(taxonomy, branch_plan):
    """The output is consumable by brain.plan_navigator.PlanNavigator."""
    import sys
    # PlanNavigator lives in the brain package; load it without polluting
    # the broader test session.
    nav_path = (
        "/home/racecar/racecar_ws/src/dgppo_ros_repo/brain/brain/"
        "plan_navigator.py"
    )
    import importlib.util
    spec = importlib.util.spec_from_file_location("brain_plan_navigator", nav_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tree = to_brain_tree(branch_plan, taxonomy)
    nav = mod.PlanNavigator(tree["steps"])
    # Linear lead "approach"
    assert nav.current_step["description"] == "approach"
    nav.advance()
    # Now at the decision step
    assert nav.current_has_branches
    assert nav.default_branch_idx() == 1
    nav.descend(0)  # 'bridge is blocked'
    assert nav.current_step["description"] == "detour"
    nav.advance()
    assert nav.is_complete
