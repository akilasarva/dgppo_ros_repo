"""Bootstrap a ``cluster_map.<env>.yaml`` from a ``cluster_id_to_label_*.json``.

Reads the clustering-team's per-environment label file (which maps
``hdbscan_cluster_id -> coarse_label``), inverts it (``coarse_label -> [ids]``),
and writes a starter YAML the user can hand-edit to match the Prompt-1
semantic vocabulary.

Example::

    ros2 run nl_planner seed_taxonomy \\
      --labels clustering/clustering/encoder_weights/livox1/cluster_id_to_label_livox1.json \\
      --env livox1 \\
      --out nl_planner/config/cluster_map.livox1.yaml

The label names produced (e.g. ``"In Corridor"``) almost certainly do not
match the Prompt-1 vocabulary (e.g. ``"Road: On"``). The script prints a
reminder to stderr after writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


def invert_label_map(label_map: dict[str, str]) -> "OrderedDict[str, list[int]]":
    """``{id_str: label}`` -> ``{label: [int_ids]}`` preserving first-seen order."""
    grouped: "OrderedDict[str, list[int]]" = OrderedDict()
    for id_str, label in label_map.items():
        cid = int(id_str)
        grouped.setdefault(label, []).append(cid)
    for label in grouped:
        grouped[label].sort()
    return grouped


def render_yaml(*, environment: str, source: str, grouped: "OrderedDict[str, list[int]]") -> str:
    """Render a tidy YAML by hand (PyYAML reorders keys / uses block style by default)."""
    lines: list[str] = []
    lines.append(f"environment: {environment}")
    lines.append(f"source: {source}")
    lines.append("modes:")
    width = max(len(label) for label in grouped) + 2  # quote padding
    for label, ids in grouped.items():
        quoted = f'"{label}"'
        ids_str = "[" + ", ".join(str(i) for i in ids) + "]"
        lines.append(f"  {quoted:<{width}} {ids_str}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="seed_taxonomy",
        description=(
            "Seed a cluster_map.<env>.yaml from a cluster_id_to_label_<env>.json. "
            "Hand-edit the result so the keys match the Prompt-1 semantic vocabulary "
            "(e.g. 'Road: On', 'Intersection: In')."
        ),
    )
    p.add_argument("--labels", required=True,
                   help="Path to cluster_id_to_label_<env>.json.")
    p.add_argument("--env", default=None,
                   help="Environment name (default: inferred from labels filename).")
    p.add_argument("--out", required=True,
                   help="Output YAML path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"error: labels file not found: {labels_path}", file=sys.stderr)
        return 2
    raw = json.loads(labels_path.read_text())
    if not isinstance(raw, dict):
        print(f"error: {labels_path} must contain a JSON object", file=sys.stderr)
        return 2

    env = args.env
    if not env:
        stem = labels_path.stem
        prefix = "cluster_id_to_label_"
        if stem.startswith(prefix):
            env = stem[len(prefix):]
        else:
            env = stem
    if not env:
        print("error: could not infer --env from labels filename", file=sys.stderr)
        return 2

    grouped = invert_label_map(raw)
    body = render_yaml(environment=env, source=str(labels_path), grouped=grouped)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)

    print(
        f"\nWrote {out_path} with {len(grouped)} starter mode(s) for env={env!r}.\n"
        f"NOTE: the labels above (e.g. {next(iter(grouped))!r}) almost certainly do not\n"
        f"match the Prompt-1 vocabulary used by nl_planner. Rename the keys (e.g.\n"
        f"  'In Corridor' -> 'Road: On'\n"
        f"  'Enter Corridor' -> 'Intersection: Approach/Enter'\n"
        f"  'In Intersection' -> 'Intersection: In'\n"
        f")\nbefore using this taxonomy. Multiple Prompt-1 modes may legally share\n"
        f"cluster ids — see nl_planner/README.md.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
