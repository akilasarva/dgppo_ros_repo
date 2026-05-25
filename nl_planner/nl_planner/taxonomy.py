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
    # Optional CARLA-specific spatial data for dgppo_ros_node coordinate transforms.
    centroids: Mapping[str, tuple[float, float, float]] = None  # type: ignore[assignment]
    bearing_map: Mapping[str, float] = None  # type: ignore[assignment]
    # Grid parameters for scaling CARLA world coords → model [0, 1.5] space.
    grid_origin: tuple[float, float] = None  # type: ignore[assignment]
    grid_scale: float = None  # type: ignore[assignment]
    # Default DGPPO model dir; step_models overrides per transition.
    default_model_dir: str = None  # type: ignore[assignment]
    step_models: Mapping[str, str] = None  # type: ignore[assignment]
    # Per-policy raw cluster ID → dgppo slot (0-3) mapping.
    # Key = policy type name (e.g. "intersection", "bridge").
    dgppo_id_maps: Mapping[str, Mapping[int, int]] = None  # type: ignore[assignment]
    # Which policy type each model_dir uses. Key = model_dir, value = policy type name.
    model_types: Mapping[str, str] = None  # type: ignore[assignment]

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

    def centroid_map(self) -> dict[str, list[float]]:
        """Return ``{cluster_id_str: [x, y, z]}`` or empty dict if not defined."""
        if not self.centroids:
            return {}
        return {k: list(v) for k, v in self.centroids.items()}

    def bearing_map_dict(self) -> dict[str, float]:
        """Return ``{"start_id-goal_id": degrees_relative}`` or empty dict."""
        if not self.bearing_map:
            return {}
        return dict(self.bearing_map)

    def grid_params(self) -> dict[str, object]:
        """Return grid origin, scale, model dirs, and policy mappings if defined."""
        out: dict[str, object] = {}
        if self.grid_origin is not None:
            out["grid_origin"] = list(self.grid_origin)
        if self.grid_scale is not None:
            out["grid_scale"] = self.grid_scale
        if self.default_model_dir:
            out["default_model_dir"] = self.default_model_dir
        if self.step_models:
            out["step_models"] = dict(self.step_models)
        if self.dgppo_id_maps:
            out["dgppo_id_maps"] = {
                policy: dict(mapping) for policy, mapping in self.dgppo_id_maps.items()
            }
        if self.model_types:
            out["model_types"] = dict(self.model_types)
        return out

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

    # Optional CARLA spatial data: centroids + bearing_map
    centroids_raw = raw.get("centroids") or {}
    centroids: dict[str, tuple[float, float, float]] = {}
    for k, v in centroids_raw.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            centroids[str(k)] = (float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0)

    bearing_map_raw = raw.get("bearing_map") or {}
    bearing_map: dict[str, float] = {str(k): float(v) for k, v in bearing_map_raw.items()}

    # Grid parameters
    grid_origin_raw = raw.get("grid_origin")
    grid_origin: tuple[float, float] | None = None
    if isinstance(grid_origin_raw, (list, tuple)) and len(grid_origin_raw) >= 2:
        grid_origin = (float(grid_origin_raw[0]), float(grid_origin_raw[1]))
    grid_scale_raw = raw.get("grid_scale")
    grid_scale: float | None = float(grid_scale_raw) if grid_scale_raw is not None else None
    default_model_dir: str | None = raw.get("default_model_dir") or None
    step_models_raw = raw.get("step_models") or {}
    step_models: dict[str, str] = {str(k): str(v) for k, v in step_models_raw.items()}

    # Per-policy dgppo slot mappings: {policy_name: {raw_cluster_id: dgppo_slot_0_to_3}}
    dgppo_id_maps_raw = raw.get("dgppo_id_maps") or {}
    dgppo_id_maps: dict[str, dict[int, int]] = {}
    for policy_name, mapping in dgppo_id_maps_raw.items():
        if isinstance(mapping, dict):
            dgppo_id_maps[str(policy_name)] = {int(k): int(v) for k, v in mapping.items()}

    # Which policy type (key in dgppo_id_maps) each model_dir uses
    model_types_raw = raw.get("model_types") or {}
    model_types: dict[str, str] = {str(k): str(v) for k, v in model_types_raw.items()}

    return ClusterTaxonomy(
        environment=env,
        source=source if isinstance(source, str) else None,
        modes=modes,
        centroids=centroids or None,
        bearing_map=bearing_map or None,
        grid_origin=grid_origin,
        grid_scale=grid_scale,
        default_model_dir=default_model_dir,
        step_models=step_models or None,
        dgppo_id_maps=dgppo_id_maps or None,
        model_types=model_types or None,
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
