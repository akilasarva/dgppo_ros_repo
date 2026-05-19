"""pydantic-ai Agent factories.

Three agents, each with a typed output schema. Built lazily so importing this
module does NOT require ``pydantic_ai`` to be installed (handy for unit
tests of pure-python pieces like schemas.py / taxonomy.py).

Provider/model is passed in as a string in pydantic-ai's
``provider:model_id`` syntax, e.g. ``"openai:gpt-4o"``. The default is
read from the ROS param / env var by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import load_prompt
from .schemas import GeneratorOutput, SyntaxVerdict, TripartiteVerdict

# Default to full gpt-4o (not -mini). gpt-4o-mini was unable to reliably
# produce schema-valid branching plans + balanced STL syntax even with 5
# retries and explicit feedback — we kept hitting pydantic-ai's
# `Exceeded maximum output retries`. gpt-4o handles the JSON+STL schema in
# 1-2 attempts and pays for itself in fewer round-trips.
DEFAULT_MODEL_ID = "openai:gpt-4o"


@dataclass
class AgentBundle:
    """The three agents the pipeline needs, all bound to the same model."""

    generator: Any                # pydantic_ai.Agent[Any, GeneratorOutput]
    syntax_verifier: Any          # pydantic_ai.Agent[Any, SyntaxVerdict]
    tripartite_verifier: Any      # pydantic_ai.Agent[Any, TripartiteVerdict]
    model_id: str


def build_agents(model_id: str = DEFAULT_MODEL_ID) -> AgentBundle:
    """Construct the three Agents wired to their .md system prompts and schemas.

    Importing ``pydantic_ai`` is deferred until this is called, so the rest of
    the package can be imported (and unit-tested) without the optional dep.
    """
    try:
        from pydantic_ai import Agent  # type: ignore
    except ImportError as exc:  # pragma: no cover - import-time guidance
        raise ImportError(
            "pydantic-ai is required for the LLM pipeline. Install with:\n"
            "    pip install 'pydantic-ai-slim[openai]'\n"
            f"(original error: {exc})"
        ) from exc

    generator = Agent(
        model_id,
        output_type=GeneratorOutput,
        system_prompt=load_prompt("generator"),
    )
    syntax_verifier = Agent(
        model_id,
        output_type=SyntaxVerdict,
        system_prompt=load_prompt("verify_syntax"),
    )
    tripartite_verifier = Agent(
        model_id,
        output_type=TripartiteVerdict,
        system_prompt=load_prompt("verify_tripartite"),
    )

    return AgentBundle(
        generator=generator,
        syntax_verifier=syntax_verifier,
        tripartite_verifier=tripartite_verifier,
        model_id=model_id,
    )


__all__ = ["AgentBundle", "build_agents", "DEFAULT_MODEL_ID"]
