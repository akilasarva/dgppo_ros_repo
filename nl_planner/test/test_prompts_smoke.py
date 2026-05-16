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
