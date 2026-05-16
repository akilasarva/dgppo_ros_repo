"""Prompt loader.

System prompts live next to this file as ``.md`` so they can be edited without
touching Python. ``load_prompt(name)`` returns the file contents verbatim
(trailing newline trimmed). Names map 1-1 to filenames:

    load_prompt("generator")          -> generator.md
    load_prompt("verify_syntax")      -> verify_syntax.md
    load_prompt("verify_tripartite")  -> verify_tripartite.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).parent

#: Canonical set of prompt names. Used by the smoke test.
KNOWN_PROMPTS: tuple[str, ...] = (
    "generator",
    "verify_syntax",
    "verify_tripartite",
)


def load_prompt(name: str) -> str:
    """Read ``<name>.md`` from this directory and return its contents.

    Raises ``FileNotFoundError`` if the prompt does not exist. Caller may
    catch and surface a clearer message.
    """
    path = _HERE / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt '{name}' not found at {path}. Known prompts: {KNOWN_PROMPTS}"
        )
    return path.read_text(encoding="utf-8").rstrip("\n")


def list_prompts() -> Iterable[str]:
    """Iterate over every ``.md`` prompt file actually present on disk."""
    for p in sorted(_HERE.glob("*.md")):
        yield p.stem


__all__ = ["load_prompt", "list_prompts", "KNOWN_PROMPTS"]
