"""Pure-Python cursor for walking a tree-shaped NavPlan inside ``brain_controller``.

Extracting this from ``brain_controller`` keeps the tree-traversal logic
independently unit-testable (no ROS, no OpenAI) and keeps the controller
free of bookkeeping noise — the controller just calls
:meth:`advance` after linear steps complete and :meth:`descend` after a VLM
branch decision.

Plan JSON shape consumed by this navigator
------------------------------------------

The plan is the brain-flavoured dict produced by
``nl_planner.branch_materializer.to_brain_tree`` (cluster_ids resolved).
Each step is::

    {
        "step":           int,
        "description":    str,
        "start_cluster":  int,
        "goal_cluster":   int,
        "start_mode":     str | None,      # diagnostic
        "goal_mode":      str | None,      # diagnostic
        "transition_cue": str | None,
        "notes":          str | None,
        "branches":       None | [
            {"vlm_cue": str, "sub_plan": [<more steps>]},
            ...
        ],
    }

Semantics
---------
- A step with ``branches`` is a *decision step*. After its goal cluster is
  reached (and any cue confirmed), the controller calls a VLM to pick a
  branch, then calls :meth:`descend` with that branch index. Anything in
  the parent ``steps`` list AFTER the decision step is unreachable — the
  branch's ``sub_plan`` becomes the new active sequence.
- A leaf path (linear ``sub_plan`` with no decisions, or the tail of one)
  finishes by calling :meth:`advance` past its last step, which flips
  :attr:`is_complete` to ``True``.

The schema validator in ``nl_planner.schemas`` already enforces:
- every decision step has at least 2 branches and exactly one ``"default"``
- nested branching depth ≤ ``MAX_BRANCH_DEPTH`` (3)
- ``sub_plan`` is non-empty

so this navigator does not re-validate; it raises ``RuntimeError`` /
``ValueError`` on programmer error only.
"""

from __future__ import annotations

from typing import Any


class PlanNavigator:
    """Stateful position cursor in a tree NavPlan.

    Not thread-safe; the controller serialises access with its own lock.
    """

    def __init__(self, steps: list[dict[str, Any]]):
        if not steps:
            raise ValueError("plan has no steps")
        self._root_steps: list[dict[str, Any]] = list(steps)
        self._sub_plan: list[dict[str, Any]] = self._root_steps
        self._step_idx: int = 0
        self._branch_path: list[int] = []
        self._complete: bool = False

    # ------------------------------------------------------------------ #
    # Queries                                                             #
    # ------------------------------------------------------------------ #

    @property
    def current_step(self) -> dict[str, Any] | None:
        """The active step in the current sub_plan, or ``None`` if complete."""
        if self._complete or self._step_idx >= len(self._sub_plan):
            return None
        return self._sub_plan[self._step_idx]

    @property
    def step_idx(self) -> int:
        return self._step_idx

    @property
    def n_steps_in_sub_plan(self) -> int:
        return len(self._sub_plan)

    @property
    def branch_path(self) -> list[int]:
        """Copy of the branch indices descended into so far."""
        return list(self._branch_path)

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def current_has_branches(self) -> bool:
        s = self.current_step
        return s is not None and bool(s.get("branches"))

    def default_branch_idx(self) -> int:
        """Index of the 'default' branch in the current decision step.

        Raises ``RuntimeError`` if the current step is not a decision step.
        Falls back to ``0`` if no branch has ``vlm_cue == "default"`` (which
        would be a schema-validation bug upstream).
        """
        s = self.current_step
        if s is None or not s.get("branches"):
            raise RuntimeError("not at a decision step")
        for i, b in enumerate(s["branches"]):
            if (b.get("vlm_cue") or "").strip().lower() == "default":
                return i
        return 0

    # ------------------------------------------------------------------ #
    # Mutations                                                           #
    # ------------------------------------------------------------------ #

    def advance(self) -> None:
        """Move past the current linear step.

        - Raises ``RuntimeError`` if already complete or at a decision step
          (caller should :meth:`descend` instead).
        - Sets :attr:`is_complete` if this was the last step of the current
          sub_plan.
        """
        if self._complete:
            raise RuntimeError("already complete")
        s = self.current_step
        if s is None:
            raise RuntimeError("no current step")
        if s.get("branches"):
            raise RuntimeError(
                "current step has branches; call descend(branch_idx) instead "
                "of advance()"
            )
        self._step_idx += 1
        if self._step_idx >= len(self._sub_plan):
            self._complete = True

    def descend(self, branch_idx: int) -> dict[str, Any]:
        """Take ``branch_idx`` of the current decision step.

        Replaces the active sub_plan with the chosen branch's ``sub_plan``
        and resets :attr:`step_idx` to 0. Returns the chosen branch dict
        (so the caller can log ``vlm_cue``).
        """
        if self._complete:
            raise RuntimeError("already complete")
        s = self.current_step
        if s is None or not s.get("branches"):
            raise RuntimeError("current step has no branches")
        branches = s["branches"]
        if not (0 <= branch_idx < len(branches)):
            raise ValueError(
                f"branch_idx {branch_idx} out of range "
                f"(have {len(branches)} branches)"
            )
        chosen = branches[branch_idx]
        sub_plan = chosen.get("sub_plan") or []
        if not sub_plan:
            raise ValueError(
                f"branch {branch_idx} ({chosen.get('vlm_cue')!r}) has empty sub_plan"
            )
        self._branch_path.append(branch_idx)
        self._sub_plan = list(sub_plan)
        self._step_idx = 0
        return chosen


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def count_leaf_paths(steps: list[dict[str, Any]]) -> int:
    """Count distinct leaf paths through a tree NavPlan (diagnostic).

    Linear plans have one path. Each decision step multiplies path count by
    the number of branches' leaf paths.
    """
    for s in steps:
        if s.get("branches"):
            return sum(
                count_leaf_paths(b.get("sub_plan") or [])
                for b in s["branches"]
            )
    return 1


__all__ = ["PlanNavigator", "count_leaf_paths"]
