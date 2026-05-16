"""Pure-python CLI for the nl_planner pipeline.

No ROS. Useful for prompt-engineering offline or for sanity-checking a
taxonomy YAML against a single mission. Installed as the ``nl_planner``
console script::

    nl_planner \\
      --mission "drive forward and turn right at the stop sign" \\
      --taxonomy nl_planner/config/cluster_map.livox1.yaml \\
      --model openai:gpt-4o-mini \\
      --out plan.json

Writes the *tree-shaped* NavPlan JSON to ``--out`` (and stdout if no ``--out``),
and prints the filtered command + STL formula to stderr. Exits nonzero (with
the verifier trace on stderr) if the retry loop exhausts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agents import DEFAULT_MODEL_ID
from .pipeline import PipelineResult, generate_plan, result_to_dict
from .schemas import PlanGenerationError
from .taxonomy import load_taxonomy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nl_planner",
        description="English mission -> verified STL + tree NavPlan (no ROS).",
    )
    p.add_argument("--mission", required=True,
                   help="The English mission text.")
    p.add_argument("--taxonomy", required=True,
                   help="Path to cluster_map.<env>.yaml.")
    p.add_argument("--model", default=None,
                   help=(f"pydantic-ai provider:model id; defaults to "
                         f"$NL_PLANNER_MODEL or {DEFAULT_MODEL_ID!r}."))
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Generator retries (default 3).")
    p.add_argument("--out", default=None,
                   help="Output JSON path. If omitted, NavPlan JSON is printed to stdout.")
    p.add_argument("--trace-out", default=None,
                   help="Optional path to write the full per-attempt trace.")
    p.add_argument("--no-verify-syntax", action="store_true",
                   help="Skip the syntax verifier (debug).")
    p.add_argument("--no-verify-tripartite", action="store_true",
                   help="Skip the tripartite verifier (debug).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model_id = args.model or os.getenv("NL_PLANNER_MODEL") or DEFAULT_MODEL_ID

    try:
        taxonomy = load_taxonomy(args.taxonomy)
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to load taxonomy: {exc}", file=sys.stderr)
        return 2

    try:
        result: PipelineResult = generate_plan(
            args.mission,
            taxonomy=taxonomy,
            model_id=model_id,
            max_attempts=args.max_attempts,
            verify_syntax=not args.no_verify_syntax,
            verify_tripartite=not args.no_verify_tripartite,
        )
    except PlanGenerationError as exc:
        print(f"PLAN GENERATION FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - one-shot CLI, surface everything
        print(f"unexpected error: {exc!r}", file=sys.stderr)
        return 2

    assert result.output is not None
    plan_json = result.output.json_plan.model_dump_json(indent=2)

    print(f"# filtered command", file=sys.stderr)
    print(result.output.filtered_command.filtered, file=sys.stderr)
    print(f"# STL formula", file=sys.stderr)
    print(result.output.stl_formula, file=sys.stderr)
    print(f"# attempts: {len(result.attempts)} ({result.elapsed_seconds}s)",
          file=sys.stderr)

    if args.out:
        Path(args.out).write_text(plan_json)
        print(f"wrote tree NavPlan to {args.out}", file=sys.stderr)
    else:
        print(plan_json)

    if args.trace_out:
        Path(args.trace_out).write_text(json.dumps(result_to_dict(result), indent=2))
        print(f"wrote trace to {args.trace_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
