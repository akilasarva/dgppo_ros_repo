"""Smoke test: all known prompts load and contain non-trivial content."""

from __future__ import annotations

import pytest

from nl_planner.prompts import KNOWN_PROMPTS, list_prompts, load_prompt


@pytest.mark.parametrize("name", KNOWN_PROMPTS)
def test_known_prompts_load(name):
    body = load_prompt(name)
    assert len(body) > 200, f"prompt {name!r} suspiciously short: {len(body)} bytes"
    assert "Role" in body or "role" in body.lower(), (
        f"prompt {name!r} doesn't look like a prompt"
    )


def test_unknown_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_list_prompts_finds_each_known_one():
    found = set(list_prompts())
    missing = set(KNOWN_PROMPTS) - found
    assert not missing, f"missing prompt files on disk: {missing}"


# --------------------------------------------------------------------------- #
# STL syntax contract — keeps generator.md and verify_syntax.md in lockstep   #
# --------------------------------------------------------------------------- #
#
# We learned the hard way (May 2026) that the generator and verifier prompts
# can drift: generator's in-context examples used `\text{Detect}(\text{X})`
# while the verifier only listed bare `Detect(X)`. That mismatch caused every
# branching plan to fail with "syntax fail: offending substring \text{Detect}".
# These tests pin the contract so it can't silently regress.

GENERATOR_REQUIRED = [
    # Cheatsheet must announce both bare and \text{}-wrapped predicate forms.
    "STL Syntax Cheatsheet",
    "\\text{Detect}(\\text{Stop Sign})",
    "Detect(StopSign)",
    # Operator allow-list must be explicit so the LLM doesn't hallucinate
    # bare `U` / `F` / `&&`.
    "\\mathbf{U}",
    "\\mathbf{F}",
    "\\land",
    "\\lor",
    "\\lnot",
    # Macro allow-list must be CLOSED with an explicit warning against
    # invented macros (e.g. \Phi_{GoForward}, \Phi_{TurnRight}). We've seen
    # the LLM invent these.
    "DO NOT invent new macros",
    # Cross-vocabulary boundary: mode names (with spaces / colons) must NEVER
    # appear inside Detect(...) / Bearing(...) arguments. This was the
    # dominant remaining failure mode in May 2026.
    "Predicate Args vs Mode Names",
    "Common Generator Mistakes",
    "Detect(Open Space)",  # cited as the canonical wrong form
]

VERIFY_SYNTAX_REQUIRED = [
    # Numbered rules — keep them findable.
    "## 1. Balanced Brackets",
    "## 2. Allowed Macros",
    "## 3. Allowed Predicates",
    "## 4. Allowed Temporal Operators",
    "## 5. Allowed Boolean Operators",
    # All four predicate-styling shapes must be explicitly accepted.
    "Detect(StopSign)",
    "\\text{Detect}(StopSign)",
    "Detect(\\text{Stop Sign})",
    "\\text{Detect}(\\text{Stop Sign})",
    # Phrases inside \text{...} (spaces) must be explicitly permitted —
    # this is the exact failure mode we saw with "Intersecting Road".
    "\\text{Intersecting Road}",
    # Each section must have at least one valid + invalid example to anchor
    # few-shot behavior.
    "# Valid Examples",
    "# Invalid Examples",
    # An anti-false-positive section so the verifier doesn't reject
    # Detect(\text{Wall}) etc.
    "Notes for the Verifier",
    "Detect(\\text{Wall})",
    # Structural argument forms must be named A and B (decision procedure).
    "Form A",
    "Form B",
]

VERIFY_TRIPARTITE_REQUIRED = [
    # The tripartite verifier must explicitly state that styling-only
    # predicate differences are NOT alignment failures.
    "Predicate Equivalence",
    "\\text{Detect}(\\text{StopSign})",
    "style-only",
]


@pytest.mark.parametrize("anchor", GENERATOR_REQUIRED)
def test_generator_prompt_contract(anchor):
    body = load_prompt("generator")
    assert anchor in body, (
        f"generator.md is missing required anchor {anchor!r}. "
        f"This likely means generator.md and verify_syntax.md have drifted "
        f"out of sync. See test_prompts_smoke.py for context."
    )


@pytest.mark.parametrize("anchor", VERIFY_SYNTAX_REQUIRED)
def test_verify_syntax_prompt_contract(anchor):
    body = load_prompt("verify_syntax")
    assert anchor in body, (
        f"verify_syntax.md is missing required anchor {anchor!r}. "
        f"The syntax verifier must accept every form the generator produces."
    )


@pytest.mark.parametrize("anchor", VERIFY_TRIPARTITE_REQUIRED)
def test_verify_tripartite_prompt_contract(anchor):
    body = load_prompt("verify_tripartite")
    assert anchor in body, (
        f"verify_tripartite.md is missing required anchor {anchor!r}. "
        f"This guard prevents the tripartite verifier from rejecting plans "
        f"on stylistic-only predicate differences."
    )


def test_generator_and_syntax_verifier_use_same_predicate_examples():
    """Both prompts cite at least one identical predicate token.

    This is a coarse cross-prompt sanity check: if both prompts agree on
    e.g. ``Detect(StopSign)`` as a representative example, the LLM agents
    see the same canonical form when reading their system prompts at
    agent-build time.
    """
    gen = load_prompt("generator")
    ver = load_prompt("verify_syntax")
    shared_examples = [
        "Detect(StopSign)",
        "\\text{Detect}(\\text{Stop Sign})",
        "\\mathbf{F}",
        "\\mathbf{U}",
        "\\land",
        "\\lor",
        "\\Phi_{Bridge}",
    ]
    for ex in shared_examples:
        assert ex in gen, f"generator.md lacks shared example {ex!r}"
        assert ex in ver, f"verify_syntax.md lacks shared example {ex!r}"
