"""Pydantic contract round-trip + validator tests.

Run with: pytest nl_planner/test/
"""

from __future__ import annotations

import pytest

from nl_planner.schemas import (
    DEFAULT_BRANCH_CUE,
    MAX_BRANCH_DEPTH,
    Branch,
    FilteredCommand,
    GeneratorOutput,
    NavPlan,
    PlanStep,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _linear_step(step: int, start: str = "Road: On", goal: str = "Road: On",
                 cue: str | None = None) -> PlanStep:
    return PlanStep(
        step=step, description=f"step {step}",
        start_mode=start, goal_mode=goal, transition_cue=cue,
    )


def _decision_step(step: int, branches: list[Branch],
                   mode: str = "Bridge: Enter") -> PlanStep:
    return PlanStep(
        step=step, description=f"decision @ {mode}",
        start_mode=mode, goal_mode=mode, transition_cue="decision point",
        branches=branches,
    )


def _default_branch(sub_plan: list[PlanStep] | None = None) -> Branch:
    return Branch(vlm_cue=DEFAULT_BRANCH_CUE, sub_plan=sub_plan or [_linear_step(0)])


# --------------------------------------------------------------------------- #
# Linear plan round-trip                                                       #
# --------------------------------------------------------------------------- #

def test_linear_plan_round_trip():
    plan = NavPlan(
        plan_name="x",
        description="y",
        steps=[_linear_step(0), _linear_step(1, cue="Detect(StopSign)")],
    )
    blob = plan.model_dump_json()
    again = NavPlan.model_validate_json(blob)
    assert again == plan


# --------------------------------------------------------------------------- #
# Branch invariants                                                            #
# --------------------------------------------------------------------------- #

def test_branches_require_exactly_one_default():
    with pytest.raises(Exception):
        _decision_step(
            0,
            branches=[
                Branch(vlm_cue="blocked", sub_plan=[_linear_step(0)]),
                Branch(vlm_cue="also blocked", sub_plan=[_linear_step(0)]),
            ],
        )
    with pytest.raises(Exception):
        _decision_step(
            0,
            branches=[
                _default_branch(),
                _default_branch(),  # two defaults — illegal
            ],
        )


def test_branches_require_at_least_two_entries():
    with pytest.raises(Exception):
        _decision_step(0, branches=[_default_branch()])


def test_valid_decision_step_round_trip():
    step = _decision_step(
        0,
        branches=[
            Branch(vlm_cue="bridge is blocked", sub_plan=[_linear_step(0)]),
            _default_branch(),
        ],
    )
    plan = NavPlan(plan_name="b", description="b", steps=[step])
    again = NavPlan.model_validate_json(plan.model_dump_json())
    assert again == plan


# --------------------------------------------------------------------------- #
# Depth cap                                                                    #
# --------------------------------------------------------------------------- #

def test_branch_depth_cap_enforced():
    # Build a chain of decision steps exceeding MAX_BRANCH_DEPTH.
    def nested(depth: int) -> PlanStep:
        if depth == 0:
            return _linear_step(0)
        return _decision_step(
            0,
            branches=[
                Branch(vlm_cue="dive", sub_plan=[nested(depth - 1)]),
                _default_branch(),
            ],
        )

    # Just-OK depth.
    NavPlan(plan_name="ok", description="d", steps=[nested(MAX_BRANCH_DEPTH)])

    # Too-deep depth must fail.
    with pytest.raises(Exception):
        NavPlan(
            plan_name="bad", description="d",
            steps=[nested(MAX_BRANCH_DEPTH + 1)],
        )


# --------------------------------------------------------------------------- #
# Generator output                                                             #
# --------------------------------------------------------------------------- #

def test_generator_output_round_trip():
    out = GeneratorOutput(
        filtered_command=FilteredCommand(original="go", filtered="go"),
        json_plan=NavPlan(plan_name="x", description="x", steps=[_linear_step(0)]),
        stl_formula=r"\Phi_{Road}",
    )
    again = GeneratorOutput.model_validate_json(out.model_dump_json())
    assert again == out
