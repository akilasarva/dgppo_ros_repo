"""Generate -> verify -> revise loop, orchestrated outside of pydantic-ai.

Why a manual loop instead of pydantic-ai's ``result_validators`` /
``ModelRetry``? Because the verifiers are *separate* LLM agents (Prompts 2a
and 2b), and we want to log each attempt's ``(prompt, output, verdict)``
trace so the user can debug prompt-engineering failures offline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    GeneratorOutput,
    NavPlan,
    PlanGenerationError,
    SyntaxVerdict,
    TripartiteVerdict,
)
from .taxonomy import ClusterTaxonomy, validate_plan_modes


# --------------------------------------------------------------------------- #
# Trace dataclasses                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class Attempt:
    """One generator+verifier round-trip, captured for debugging."""

    iteration: int
    generator_input: str
    generator_output: GeneratorOutput | None = None
    generator_error: str | None = None
    syntax: SyntaxVerdict | None = None
    tripartite: TripartiteVerdict | None = None
    taxonomy_missing: list[str] = field(default_factory=list)
    accepted: bool = False
    failure_reason: str | None = None


@dataclass
class PipelineResult:
    """End-to-end output of ``generate_plan``."""

    accepted: bool
    output: GeneratorOutput | None
    attempts: list[Attempt]
    elapsed_seconds: float


# --------------------------------------------------------------------------- #
# Prompt rendering                                                             #
# --------------------------------------------------------------------------- #

def _render_generator_msg(
    english: str,
    modes: list[str],
    feedback: str | None,
) -> str:
    parts = [
        "User mission:",
        english.strip(),
        "",
        "Available semantic modes (use ONLY these for start_mode / goal_mode):",
        *(f"- {m}" for m in modes),
    ]
    if feedback:
        parts += [
            "",
            "Verifier feedback (your previous attempt failed — repair every issue below):",
            feedback.strip(),
        ]
    return "\n".join(parts)


def _render_tripartite_msg(english: str, gen: GeneratorOutput) -> str:
    return (
        "[X] Original English Command\n"
        f"{english.strip()}\n\n"
        "[X] Filtered Command\n"
        f"{gen.filtered_command.filtered.strip()}\n\n"
        "[Y] STL Formula\n"
        f"{gen.stl_formula.strip()}\n\n"
        "[Z] JSON Plan\n"
        f"{gen.json_plan.model_dump_json(indent=2)}\n"
    )


def _feedback_for_taxonomy(missing: list[str], taxonomy: ClusterTaxonomy) -> str:
    legal = taxonomy.modes_for_prompt()
    return (
        f"TAXONOMY FAIL: the following start_mode / goal_mode values are NOT in "
        f"the per-environment taxonomy and must be replaced: {missing}.\n"
        f"Legal modes: {legal}"
    )


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def generate_plan(
    english: str,
    *,
    taxonomy: ClusterTaxonomy,
    agents=None,
    model_id: str | None = None,
    max_attempts: int = 3,
    verify_syntax: bool = True,
    verify_tripartite: bool = True,
) -> PipelineResult:
    """Run the generate/verify/retry loop for one English mission.

    Args:
        english: free-text mission, e.g. ``"cross the bridge, take the long road if blocked"``.
        taxonomy: legal semantic modes for the current environment.
        agents: pre-built ``AgentBundle`` (if None, one is built from ``model_id``).
        model_id: ``"provider:model"`` string forwarded to ``build_agents``.
        max_attempts: hard cap on generator retries (3 keeps p99 latency bounded).
        verify_syntax / verify_tripartite: gate off either verifier (debug only).

    Returns:
        A ``PipelineResult`` whose ``output`` is the accepted ``GeneratorOutput``.
        Raises ``PlanGenerationError`` if no attempt is accepted within
        ``max_attempts``.
    """
    if agents is None:
        from .agents import build_agents, DEFAULT_MODEL_ID
        agents = build_agents(model_id or DEFAULT_MODEL_ID)

    started = time.monotonic()
    attempts: list[Attempt] = []
    feedback: str | None = None

    for i in range(1, max_attempts + 1):
        user_msg = _render_generator_msg(english, taxonomy.modes_for_prompt(), feedback)
        attempt = Attempt(iteration=i, generator_input=user_msg)

        try:
            gen_result = agents.generator.run_sync(user_msg)
            gen: GeneratorOutput = gen_result.output  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - one LLM call, keep going
            attempt.generator_error = repr(exc)
            attempt.failure_reason = f"generator exception: {exc!r}"
            attempts.append(attempt)
            feedback = (
                "GENERATOR EXCEPTION: your previous output failed to parse against "
                f"the schema or to satisfy the in-schema validators ({exc!s}). "
                "Re-emit, paying particular attention to: at-most-one 'default' "
                "branch per decision step, branch nesting <= 3, every step has "
                "integer step >=0."
            )
            continue
        attempt.generator_output = gen

        missing = validate_plan_modes(gen.json_plan, taxonomy)
        attempt.taxonomy_missing = missing
        if missing:
            attempt.failure_reason = f"taxonomy fail: {missing}"
            attempts.append(attempt)
            feedback = _feedback_for_taxonomy(missing, taxonomy)
            continue

        if verify_syntax:
            try:
                syn_result = agents.syntax_verifier.run_sync(
                    f"Input formula:\n{gen.stl_formula}"
                )
                syn: SyntaxVerdict = syn_result.output  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                attempt.failure_reason = f"syntax verifier exception: {exc!r}"
                attempts.append(attempt)
                feedback = f"SYNTAX VERIFIER EXCEPTION: {exc!s}"
                continue
            attempt.syntax = syn
            if not syn.ok:
                attempt.failure_reason = f"syntax fail: {syn.error or '(no detail)'}"
                attempts.append(attempt)
                feedback = f"SYNTAX FAIL: {syn.error or '(no detail provided)'}"
                continue

        if verify_tripartite:
            try:
                tri_result = agents.tripartite_verifier.run_sync(
                    _render_tripartite_msg(english, gen)
                )
                tri: TripartiteVerdict = tri_result.output  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                attempt.failure_reason = f"tripartite verifier exception: {exc!r}"
                attempts.append(attempt)
                feedback = f"TRIPARTITE VERIFIER EXCEPTION: {exc!s}"
                continue
            attempt.tripartite = tri
            if not tri.ok:
                attempt.failure_reason = f"tripartite fail: {tri.notes}"
                attempts.append(attempt)
                feedback = (
                    f"TRIPARTITE FAIL "
                    f"(english_stl={tri.english_stl_aligned}, "
                    f"stl_json={tri.stl_json_aligned}, "
                    f"english_json={tri.english_json_aligned}): {tri.notes}"
                )
                continue

        attempt.accepted = True
        attempts.append(attempt)
        return PipelineResult(
            accepted=True,
            output=gen,
            attempts=attempts,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    raise PlanGenerationError(
        f"plan generation did not converge within {max_attempts} attempts. "
        f"Last failure: {attempts[-1].failure_reason if attempts else '(no attempts logged)'}"
    )


# --------------------------------------------------------------------------- #
# Trace serialization (CLI + tests)                                            #
# --------------------------------------------------------------------------- #

def attempt_to_dict(attempt: Attempt) -> dict[str, Any]:
    return {
        "iteration": attempt.iteration,
        "generator_input": attempt.generator_input,
        "generator_output": (
            attempt.generator_output.model_dump() if attempt.generator_output else None
        ),
        "generator_error": attempt.generator_error,
        "syntax": attempt.syntax.model_dump() if attempt.syntax else None,
        "tripartite": attempt.tripartite.model_dump() if attempt.tripartite else None,
        "taxonomy_missing": attempt.taxonomy_missing,
        "accepted": attempt.accepted,
        "failure_reason": attempt.failure_reason,
    }


def result_to_dict(result: PipelineResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "output": result.output.model_dump() if result.output else None,
        "attempts": [attempt_to_dict(a) for a in result.attempts],
        "elapsed_seconds": result.elapsed_seconds,
    }


__all__ = [
    "Attempt",
    "PipelineResult",
    "generate_plan",
    "attempt_to_dict",
    "result_to_dict",
]
