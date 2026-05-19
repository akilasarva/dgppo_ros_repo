"""Unit tests for brain.plan_navigator.PlanNavigator.

These tests are pure-Python — no ROS, no OpenAI. Run with::

    pytest brain/test/test_plan_navigator.py
"""
from __future__ import annotations

import pytest

from brain.plan_navigator import PlanNavigator, count_leaf_paths


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _step(idx, start, goal, *, cue=None, branches=None, desc=None):
    return {
        "step":           idx,
        "description":    desc or f"step {idx}",
        "start_cluster":  start,
        "goal_cluster":   goal,
        "transition_cue": cue,
        "branches":       branches,
    }


@pytest.fixture
def linear_plan():
    return [
        _step(0, 0, 1, desc="approach"),
        _step(1, 1, 2, desc="enter"),
        _step(2, 2, 3, desc="exit"),
    ]


@pytest.fixture
def branching_plan():
    """Tree with one decision step at root level."""
    return [
        _step(0, 0, 10, desc="approach intersection",
              cue="Detect(Intersection)"),
        _step(0, 10, 10, desc="decide", cue="decision",
              branches=[
                  {"vlm_cue": "path is blocked", "sub_plan": [
                      _step(0, 10, 0, desc="turn right"),
                  ]},
                  {"vlm_cue": "default", "sub_plan": [
                      _step(0, 10, 11, desc="cross"),
                      _step(1, 11, 0, desc="continue"),
                  ]},
              ]),
    ]


@pytest.fixture
def nested_branching_plan():
    """Decision step whose default branch has another decision step inside."""
    return [
        _step(0, 0, 10, desc="approach"),
        _step(1, 10, 10, desc="outer decide",
              branches=[
                  {"vlm_cue": "side road", "sub_plan": [
                      _step(0, 10, 1, desc="left turn"),
                  ]},
                  {"vlm_cue": "default", "sub_plan": [
                      _step(0, 10, 11, desc="straight start"),
                      _step(1, 11, 11, desc="inner decide",
                            branches=[
                                {"vlm_cue": "obstacle", "sub_plan": [
                                    _step(0, 11, 0, desc="reverse out"),
                                ]},
                                {"vlm_cue": "default", "sub_plan": [
                                    _step(0, 11, 12, desc="finish"),
                                ]},
                            ]),
                  ]},
              ]),
    ]


# --------------------------------------------------------------------------- #
# Initial state                                                                #
# --------------------------------------------------------------------------- #

def test_rejects_empty_plan():
    with pytest.raises(ValueError):
        PlanNavigator([])


def test_initial_state_linear(linear_plan):
    nav = PlanNavigator(linear_plan)
    assert nav.step_idx == 0
    assert nav.branch_path == []
    assert nav.is_complete is False
    assert nav.current_step["description"] == "approach"
    assert nav.current_has_branches is False
    assert nav.n_steps_in_sub_plan == 3


# --------------------------------------------------------------------------- #
# Linear advancement                                                           #
# --------------------------------------------------------------------------- #

def test_linear_walk_to_completion(linear_plan):
    nav = PlanNavigator(linear_plan)
    nav.advance()
    assert nav.step_idx == 1
    assert nav.current_step["description"] == "enter"
    assert nav.is_complete is False
    nav.advance()
    assert nav.current_step["description"] == "exit"
    nav.advance()
    assert nav.is_complete is True
    assert nav.current_step is None


def test_advance_after_complete_raises(linear_plan):
    nav = PlanNavigator(linear_plan)
    nav.advance(); nav.advance(); nav.advance()
    assert nav.is_complete is True
    with pytest.raises(RuntimeError):
        nav.advance()


# --------------------------------------------------------------------------- #
# Branching                                                                    #
# --------------------------------------------------------------------------- #

def test_advance_at_decision_step_raises(branching_plan):
    nav = PlanNavigator(branching_plan)
    nav.advance()  # arrive at decision step (idx 1)
    assert nav.current_has_branches
    with pytest.raises(RuntimeError):
        nav.advance()


def test_descend_default_branch(branching_plan):
    nav = PlanNavigator(branching_plan)
    nav.advance()  # to decision step
    assert nav.default_branch_idx() == 1  # 'default' is the second branch
    chosen = nav.descend(1)
    assert chosen["vlm_cue"] == "default"
    assert nav.branch_path == [1]
    assert nav.step_idx == 0
    assert nav.current_step["description"] == "cross"
    assert nav.n_steps_in_sub_plan == 2
    assert nav.current_has_branches is False


def test_descend_blocked_branch_completes_in_one_step(branching_plan):
    nav = PlanNavigator(branching_plan)
    nav.advance()
    nav.descend(0)  # 'path is blocked'
    assert nav.current_step["description"] == "turn right"
    assert nav.is_complete is False
    nav.advance()
    assert nav.is_complete is True


def test_descend_default_completes_after_two_steps(branching_plan):
    nav = PlanNavigator(branching_plan)
    nav.advance()
    nav.descend(1)
    nav.advance()              # "cross" -> "continue"
    assert nav.current_step["description"] == "continue"
    nav.advance()              # past "continue"
    assert nav.is_complete is True


def test_descend_with_bad_index_raises(branching_plan):
    nav = PlanNavigator(branching_plan)
    nav.advance()
    with pytest.raises(ValueError):
        nav.descend(7)
    with pytest.raises(ValueError):
        nav.descend(-1)


def test_descend_at_non_decision_raises(linear_plan):
    nav = PlanNavigator(linear_plan)
    with pytest.raises(RuntimeError):
        nav.descend(0)


def test_default_branch_idx_at_non_decision_raises(linear_plan):
    nav = PlanNavigator(linear_plan)
    with pytest.raises(RuntimeError):
        nav.default_branch_idx()


# --------------------------------------------------------------------------- #
# Nested branching                                                             #
# --------------------------------------------------------------------------- #

def test_nested_descend(nested_branching_plan):
    nav = PlanNavigator(nested_branching_plan)
    nav.advance()                  # to outer decide
    nav.descend(1)                 # default branch
    assert nav.branch_path == [1]
    assert nav.current_step["description"] == "straight start"
    nav.advance()                  # to inner decide
    assert nav.current_has_branches
    nav.descend(1)                 # inner default
    assert nav.branch_path == [1, 1]
    assert nav.current_step["description"] == "finish"
    nav.advance()
    assert nav.is_complete is True


# --------------------------------------------------------------------------- #
# count_leaf_paths                                                             #
# --------------------------------------------------------------------------- #

def test_count_leaf_paths_linear(linear_plan):
    assert count_leaf_paths(linear_plan) == 1


def test_count_leaf_paths_branching(branching_plan):
    assert count_leaf_paths(branching_plan) == 2


def test_count_leaf_paths_nested(nested_branching_plan):
    # outer has 2 branches; branch 0 has 1 leaf; branch 1 has 1 step then an
    # inner decision with 2 branches each having 1 leaf -> 2 leaves total
    # outer leaves: 1 + 2 = 3
    assert count_leaf_paths(nested_branching_plan) == 3
