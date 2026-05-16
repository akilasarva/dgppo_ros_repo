"""Turn a tree-shaped ``NavPlan`` into a sequence of brain-compatible linear plans.

The executor walks the tree with a list-of-branch-indices ``path``:

- ``materialize_segment(plan, taxonomy, path=[])`` returns the leading linear
  chunk of the root and the next ``decision_step`` (or ``None`` for terminal).
- ``materialize_segment(plan, taxonomy, path=[2])`` returns the leading linear
  chunk of the branch chosen at the first decision (index 2), and so on.

Each returned ``brain_plan`` is a dict in exactly the shape ``brain_controller``
already expects (``plan_name`` / ``description`` / ``cluster_labels`` / ``steps``),
ready to be ``json.dumps``'d into ``/brain/incoming_plan``.

Semantic modes are resolved to scalar cluster ids via the taxonomy's
``canonical_id()`` (first id listed for the mode). The executor still
watches ``/predicted_cluster`` against the full id set for transient
diagnostic / multi-id mode matching — see ``taxonomy.resolve()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .schemas import Branch, NavPlan, PlanStep
from .taxonomy import ClusterTaxonomy


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class MaterializedSegment:
    """One linear chunk of a tree NavPlan, ready to ship to brain_controller."""

    #: Plan dict in brain_controller's JSON schema.
    brain_plan: dict[str, Any]
    #: The branch-bearing PlanStep that follows this chunk (None at terminus).
    decision_step: PlanStep | None
    #: Path of branch indices that produced this segment (for diagnostic logs).
    path: tuple[int, ...]
    #: True iff this is the final segment (no more decisions in the tree).
    is_terminal: bool


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def materialize_segment(
    plan: NavPlan,
    taxonomy: ClusterTaxonomy,
    *,
    path: Sequence[int] = (),
    plan_name_override: str | None = None,
) -> MaterializedSegment:
    """Build the brain-compatible linear plan for the segment selected by ``path``.

    ``path`` is the list of branch indices chosen at each decision step
    encountered so far. ``path=[]`` selects the root linear chunk.
    """
    sub_plan = _follow_path(plan, path)

    linear_steps: list[PlanStep] = []
    decision_step: PlanStep | None = None
    for step in sub_plan:
        if step.branches is None:
            linear_steps.append(step)
        else:
            decision_step = step
            break

    if not linear_steps and decision_step is None:
        # Empty sub_plan — should be impossible because schemas.Branch enforces
        # min_length=1 on sub_plan, but guard anyway.
        raise ValueError(f"segment at path={list(path)} is empty")

    brain_plan = _to_brain_plan(
        plan=plan,
        linear_steps=linear_steps,
        taxonomy=taxonomy,
        path=tuple(path),
        plan_name_override=plan_name_override,
    )

    return MaterializedSegment(
        brain_plan=brain_plan,
        decision_step=decision_step,
        path=tuple(path),
        is_terminal=(decision_step is None),
    )


def walk_all_segments(
    plan: NavPlan,
    taxonomy: ClusterTaxonomy,
) -> list[MaterializedSegment]:
    """Enumerate every reachable linear segment in the tree (DFS over branches).

    Useful for offline validation / unit testing — not used by the executor at
    runtime, which materializes only the path the VLM picks.
    """
    out: list[MaterializedSegment] = []
    _walk(plan, taxonomy, path=[], out=out)
    return out


def _walk(plan: NavPlan, taxonomy: ClusterTaxonomy, *, path: list[int], out: list[MaterializedSegment]) -> None:
    seg = materialize_segment(plan, taxonomy, path=tuple(path))
    out.append(seg)
    if seg.decision_step is None:
        return
    for i in range(len(seg.decision_step.branches or ())):
        _walk(plan, taxonomy, path=path + [i], out=out)


# --------------------------------------------------------------------------- #
# Tree walking                                                                 #
# --------------------------------------------------------------------------- #

def _follow_path(plan: NavPlan, path: Sequence[int]) -> list[PlanStep]:
    """Return the sub_plan reached by descending ``path`` from ``plan.steps``."""
    current: list[PlanStep] = list(plan.steps)
    for depth, branch_idx in enumerate(path):
        decision: PlanStep | None = None
        for step in current:
            if step.branches is not None:
                decision = step
                break
        if decision is None or decision.branches is None:
            raise ValueError(
                f"path index {depth} expected a decision step, found none in "
                f"sub_plan {[s.description for s in current]!r}"
            )
        if branch_idx < 0 or branch_idx >= len(decision.branches):
            raise ValueError(
                f"path index {depth} = {branch_idx} out of range; "
                f"decision step has {len(decision.branches)} branches"
            )
        current = list(decision.branches[branch_idx].sub_plan)
    return current


# --------------------------------------------------------------------------- #
# brain_controller plan-dict construction                                      #
# --------------------------------------------------------------------------- #

def _to_brain_plan(
    *,
    plan: NavPlan,
    linear_steps: list[PlanStep],
    taxonomy: ClusterTaxonomy,
    path: tuple[int, ...],
    plan_name_override: str | None,
) -> dict[str, Any]:
    """Render a brain_controller-shaped plan dict from a list of linear PlanSteps."""
    brain_steps: list[dict[str, Any]] = []
    for i, step in enumerate(linear_steps):
        brain_steps.append({
            "step":           i,
            "description":    step.description,
            "start_cluster":  taxonomy.canonical_id(step.start_mode),
            "goal_cluster":   taxonomy.canonical_id(step.goal_mode),
            "start_mode":     step.start_mode,
            "goal_mode":      step.goal_mode,
            "transition_cue": step.transition_cue,
            "notes":          step.notes,
        })

    cluster_labels: dict[str, str] = {
        str(cid): label for cid, label in taxonomy.cluster_labels().items()
    }

    suffix = f" [path={list(path)}]" if path else ""
    name = plan_name_override or f"{plan.plan_name}{suffix}"

    return {
        "plan_name":      name,
        "description":    plan.description,
        "cluster_labels": cluster_labels,
        "nl_planner": {
            # Executor uses these to know "where in the tree am I" without
            # re-walking the tree itself.
            "path":           list(path),
            "ends_at_decision": False,  # filled in below
        },
        "steps": brain_steps,
    }


def attach_decision_metadata(
    segment: MaterializedSegment,
) -> None:
    """Fill in the ``ends_at_decision`` and per-branch metadata on the brain plan.

    Called by the executor after materialization so the brain JSON can be
    pretty-printed in logs / dashboards without the executor having to keep
    parallel state.
    """
    meta = segment.brain_plan.setdefault("nl_planner", {})
    meta["path"] = list(segment.path)
    meta["ends_at_decision"] = segment.decision_step is not None
    if segment.decision_step is not None:
        meta["decision"] = {
            "description":    segment.decision_step.description,
            "start_mode":     segment.decision_step.start_mode,
            "transition_cue": segment.decision_step.transition_cue,
            "branches": [
                {"index": i, "vlm_cue": b.vlm_cue}
                for i, b in enumerate(segment.decision_step.branches or ())
            ],
        }


__all__ = [
    "MaterializedSegment",
    "materialize_segment",
    "walk_all_segments",
    "attach_decision_metadata",
]
