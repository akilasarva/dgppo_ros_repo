"""Pydantic contract shared by generator, verifiers, executor, and brain.

This module is the single source of truth for what the LLM is allowed to emit,
and what the executor expects to consume. All validation that does NOT depend
on a per-environment taxonomy lives here as ``model_validator``s. Taxonomy-
dependent checks (semantic mode must exist in the YAML) live in
``pipeline.py`` so the retry loop can feed the failure back into the
generator.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Soft caps                                                                   #
# --------------------------------------------------------------------------- #

#: Maximum branch nesting depth before the schema validator rejects the plan.
#: Documented to Prompt 1 so the LLM should never produce deeper trees.
MAX_BRANCH_DEPTH = 3

#: Sentinel for the always-present fallback branch at every decision point.
DEFAULT_BRANCH_CUE = "default"


# --------------------------------------------------------------------------- #
# Generator-side primitives                                                   #
# --------------------------------------------------------------------------- #

class FilteredCommand(BaseModel):
    """Original English plus the cleaned-up version the LLM actually planned for."""

    original: str = Field(..., description="The verbatim mission text the user issued.")
    filtered: str = Field(
        ...,
        description=(
            "Same intent with transient features removed (bikes, pedestrians, "
            "construction, cones, etc.). Keep stable landmarks."
        ),
    )


class PlanStep(BaseModel):
    """One step in the (possibly branch-shaped) navigation plan.

    Linear steps have ``branches=None``. Decision-point steps carry a list of
    ``Branch``es; each branch's ``sub_plan`` is itself a linear sequence of
    ``PlanStep``s that may end in another branch step (up to
    ``MAX_BRANCH_DEPTH``).
    """

    step: int = Field(..., ge=0, description="0-based index within the containing sub_plan.")
    description: str = Field(..., description="One-line human-readable summary.")
    start_mode: str = Field(
        ...,
        description=(
            "Semantic mode the step starts in, e.g. 'Road: On' or "
            "'Intersection: Approach/Enter'. Must appear in the taxonomy YAML."
        ),
    )
    goal_mode: str = Field(
        ...,
        description=(
            "Semantic mode the step ends in. For a pure decision step "
            "(branches != None and the robot stays put), set goal_mode == start_mode."
        ),
    )
    transition_cue: Optional[str] = Field(
        None,
        description=(
            "Free-text visual cue brain_controller's VLM polls for on arrival "
            "at goal_mode. Use null for cue-free arrival-only advancement."
        ),
    )
    notes: Optional[str] = Field(None, description="Optional commentary for humans.")
    branches: Optional[List["Branch"]] = Field(
        None,
        description=(
            "Present only at decision points. Exactly one branch must have "
            "vlm_cue == 'default' as the fallback."
        ),
    )

    @model_validator(mode="after")
    def _check_branches(self) -> "PlanStep":
        if self.branches is None:
            return self
        if len(self.branches) < 2:
            raise ValueError(
                "branches must have >= 2 entries (at least one real choice plus default)"
            )
        defaults = [b for b in self.branches if b.vlm_cue.strip().lower() == DEFAULT_BRANCH_CUE]
        if len(defaults) != 1:
            raise ValueError(
                f"branches must contain exactly one entry with vlm_cue == "
                f"'{DEFAULT_BRANCH_CUE}'; got {len(defaults)}"
            )
        return self


class Branch(BaseModel):
    """One choice at a decision point.

    The executor calls a VLM at the decision step, scoring each ``vlm_cue``
    against the latest camera image, and follows the winning branch's
    ``sub_plan``.
    """

    vlm_cue: str = Field(
        ...,
        description=(
            "Short visual description (or the literal string 'default'). The "
            "executor will ask the VLM 'which of these is best visible?' and "
            "pick this branch on a positive match."
        ),
    )
    sub_plan: List[PlanStep] = Field(
        ...,
        min_length=1,
        description="Linear continuation taken when this branch is chosen.",
    )


PlanStep.model_rebuild()


# --------------------------------------------------------------------------- #
# Top-level container                                                         #
# --------------------------------------------------------------------------- #

class NavPlan(BaseModel):
    """Tree-shaped navigation plan. ``steps`` is the root linear segment."""

    plan_name: str = Field(..., description="Short title for logs / UIs.")
    description: str = Field(..., description="One-paragraph summary of the mission.")
    steps: List[PlanStep] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_depth(self) -> "NavPlan":
        for step in self.steps:
            _assert_depth(step, depth=1)
        return self


def _assert_depth(step: PlanStep, *, depth: int) -> None:
    if step.branches is None:
        return
    if depth > MAX_BRANCH_DEPTH:
        raise ValueError(
            f"branch nesting depth exceeds MAX_BRANCH_DEPTH={MAX_BRANCH_DEPTH}"
        )
    for branch in step.branches:
        for sub_step in branch.sub_plan:
            _assert_depth(sub_step, depth=depth + 1)


# --------------------------------------------------------------------------- #
# Agent outputs                                                               #
# --------------------------------------------------------------------------- #

class GeneratorOutput(BaseModel):
    """What the generator Agent returns in one call."""

    filtered_command: FilteredCommand
    json_plan: NavPlan
    stl_formula: str = Field(
        ...,
        description=(
            "Single STL formula (LaTeX-ish) covering ALL branches. The "
            "tripartite verifier checks it against the plan tree and the "
            "filtered English command."
        ),
    )


class SyntaxVerdict(BaseModel):
    """Output of the STL syntax-only verifier (Prompt 2a)."""

    ok: bool = Field(..., description="True iff the STL is well-formed.")
    error: Optional[str] = Field(
        None,
        description=(
            "Concrete rule(s) violated, copied verbatim into the retry feedback. "
            "Required iff ok is False."
        ),
    )


class TripartiteVerdict(BaseModel):
    """Output of the X<->Y<->Z verifier (Prompt 2b)."""

    english_stl_aligned: bool
    stl_json_aligned: bool
    english_json_aligned: bool
    notes: str = Field(
        ...,
        description="Concrete differences keyed by which pair disagreed.",
    )

    @property
    def ok(self) -> bool:
        return (
            self.english_stl_aligned
            and self.stl_json_aligned
            and self.english_json_aligned
        )


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #

class PlanGenerationError(RuntimeError):
    """Raised by ``pipeline.generate_plan`` when retries are exhausted."""


__all__ = [
    "MAX_BRANCH_DEPTH",
    "DEFAULT_BRANCH_CUE",
    "FilteredCommand",
    "PlanStep",
    "Branch",
    "NavPlan",
    "GeneratorOutput",
    "SyntaxVerdict",
    "TripartiteVerdict",
    "PlanGenerationError",
]
