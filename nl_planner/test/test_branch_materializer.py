"""Tests for branch_materializer.materialize_segment."""

from __future__ import annotations

import textwrap

import pytest

from nl_planner.branch_materializer import (
    materialize_segment,
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
