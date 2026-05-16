"""Per-environment semantic-mode <-> HDBSCAN-cluster-id taxonomy.

A taxonomy is a YAML file with the schema::

    environment: <str>
    source: <path to cluster_id_to_label_*.json that seeded it>
    modes:
      "Road: On":                      [5, 6, 8, 9, 10, 11, 12]
      "Intersection: Approach/Enter":  [7]
      "Intersection: In":              [0, 1, 2]
      "Open Space":                    [3, 4]
      ...

Multiple semantic modes may map to overlapping cluster id sets — the executor
treats *any* listed id as "in this mode" when watching ``/predicted_cluster``,
and uses the **first** id as the canonical scalar ``start_cluster`` /
``goal_cluster`` in the brain-shaped linear plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml


class TaxonomyError(ValueError):
    """Raised for malformed taxonomy files or unknown-mode lookups."""


@dataclass(frozen=True)
class ClusterTaxonomy:
    """In-memory view of a ``cluster_map.<env>.yaml`` file."""

    environment: str
    source: str | None
    modes: Mapping[str, tuple[int, ...]]

    # ------------------------------------------------------------------ #
    # Lookup                                                              #
    # ------------------------------------------------------------------ #

    def __contains__(self, mode: str) -> bool:
        return mode in self.modes

    def resolve(self, mode: str) -> tuple[int, ...]:
        """Return the cluster id tuple for ``mode``.

        Raises ``TaxonomyError`` if the mode is not in the YAML.
        """
        if mode not in self.modes:
            raise TaxonomyError(
                f"semantic mode {mode!r} not in taxonomy for environment "
                f"{self.environment!r}; legal modes: {sorted(self.modes)}"
            )
        return self.modes[mode]

    def canonical_id(self, mode: str) -> int:
        """First cluster id listed for ``mode`` — used as the scalar id brain sees."""
        ids = self.resolve(mode)
        if not ids:
            raise TaxonomyError(
                f"semantic mode {mode!r} maps to an empty cluster id list"
            )
        return ids[0]

    def modes_for_prompt(self) -> list[str]:
        """Sorted list of legal mode strings, ready to splice into the user message."""
        return sorted(self.modes.keys())

    def label_for_id(self, cluster_id: int) -> str | None:
        """Inverse lookup: return any mode that contains ``cluster_id`` (or None)."""
        for mode, ids in self.modes.items():
            if cluster_id in ids:
                return mode
        return None

    def cluster_labels(self) -> dict[int, str]:
        """``{cluster_id: mode_name}`` dict suitable for brain_controller's
        ``cluster_labels`` plan field.

        When a cluster id appears in multiple modes (the YAML allows overlap),
        the *first-encountered* mode wins, which keeps logs predictable.
        """
        out: dict[int, str] = {}
        for mode, ids in self.modes.items():
            for cid in ids:
                out.setdefault(cid, mode)
        return out


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_taxonomy(path: str | Path) -> ClusterTaxonomy:
    """Read and validate ``path`` (a YAML file). Raises ``TaxonomyError``."""
    p = Path(path)
    if not p.exists():
        raise TaxonomyError(f"taxonomy file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise TaxonomyError(f"could not parse YAML at {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TaxonomyError(f"taxonomy at {p} must be a mapping at the top level")

    env = raw.get("environment")
    if not isinstance(env, str) or not env.strip():
        raise TaxonomyError(f"taxonomy at {p} is missing a non-empty 'environment' string")

    modes_raw = raw.get("modes") or {}
    if not isinstance(modes_raw, dict) or not modes_raw:
        raise TaxonomyError(f"taxonomy at {p} must declare a non-empty 'modes' mapping")

    modes: dict[str, tuple[int, ...]] = {}
    for mode, ids in modes_raw.items():
        if not isinstance(mode, str) or not mode.strip():
            raise TaxonomyError(f"mode keys must be non-empty strings; got {mode!r}")
        if not isinstance(ids, list) or not ids:
            raise TaxonomyError(
                f"mode {mode!r} must map to a non-empty list of cluster ids"
            )
        ints: list[int] = []
        for x in ids:
            if not isinstance(x, int):
                raise TaxonomyError(
                    f"cluster ids under mode {mode!r} must be ints; got {x!r}"
                )
            ints.append(x)
        modes[mode] = tuple(ints)

    source = raw.get("source")
    return ClusterTaxonomy(
        environment=env,
        source=source if isinstance(source, str) else None,
        modes=modes,
    )


# --------------------------------------------------------------------------- #
# Validation against a NavPlan                                                #
# --------------------------------------------------------------------------- #

def validate_plan_modes(plan, taxonomy: ClusterTaxonomy) -> list[str]:
    """Return a list of every ``(start|goal)_mode`` the plan uses that is not
    in the taxonomy. Walks the full branch tree. Empty list = all good.

    ``plan`` may be a ``NavPlan`` instance or a dict matching that schema.
    """
    seen: set[str] = set()
    missing: list[str] = []

    def _walk(steps: Iterable):
        for step in steps:
            sm = _attr(step, "start_mode")
            gm = _attr(step, "goal_mode")
            for mode in (sm, gm):
                if mode in seen:
                    continue
                seen.add(mode)
                if mode not in taxonomy:
                    missing.append(mode)
            branches = _attr(step, "branches")
            if branches:
                for branch in branches:
                    _walk(_attr(branch, "sub_plan"))

    _walk(_attr(plan, "steps"))
    return missing


def _attr(obj, name: str):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


__all__ = [
    "ClusterTaxonomy",
    "TaxonomyError",
    "load_taxonomy",
    "validate_plan_modes",
]
