"""Tests for taxonomy loading + plan-mode validation + bootstrap inverter."""

from __future__ import annotations

import textwrap

import pytest

from nl_planner.bootstrap.seed_taxonomy import invert_label_map, render_yaml
from nl_planner.schemas import Branch, NavPlan, PlanStep
from nl_planner.taxonomy import TaxonomyError, load_taxonomy, validate_plan_modes


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #

def _write(tmp_path, body: str):
    p = tmp_path / "tax.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_taxonomy_happy_path(tmp_path):
    p = _write(tmp_path, """
        environment: testenv
        source: somefile.json
        modes:
          "Road: On":     [5, 6, 7]
          "Intersection": [0, 1, 2]
    """)
    tax = load_taxonomy(p)
    assert tax.environment == "testenv"
    assert tax.resolve("Road: On") == (5, 6, 7)
    assert tax.canonical_id("Intersection") == 0
    assert "Road: On" in tax
    assert tax.label_for_id(7) == "Road: On"


def test_load_taxonomy_rejects_missing_modes(tmp_path):
    p = _write(tmp_path, """
        environment: testenv
        modes: {}
    """)
    with pytest.raises(TaxonomyError):
        load_taxonomy(p)


def test_load_taxonomy_rejects_non_int_ids(tmp_path):
    p = _write(tmp_path, """
        environment: testenv
        modes:
          "Road: On": ["a"]
    """)
    with pytest.raises(TaxonomyError):
        load_taxonomy(p)


def test_modes_for_prompt_sorted(tmp_path):
    p = _write(tmp_path, """
        environment: e
        modes:
          "Bridge: On": [1]
          "Road: On":   [2]
          "Along Wall": [3]
    """)
    assert load_taxonomy(p).modes_for_prompt() == ["Along Wall", "Bridge: On", "Road: On"]


# --------------------------------------------------------------------------- #
# Plan-mode validation                                                         #
# --------------------------------------------------------------------------- #

def test_validate_plan_modes_catches_missing(tmp_path):
    p = _write(tmp_path, """
        environment: e
        modes:
          "Road: On":     [0]
          "Bridge: Enter": [1]
    """)
    tax = load_taxonomy(p)

    plan = NavPlan(
        plan_name="x", description="x",
        steps=[
            PlanStep(step=0, description="ok",
                     start_mode="Road: On", goal_mode="Bridge: Enter",
                     transition_cue=None),
            PlanStep(step=1, description="bad",
                     start_mode="Bridge: Enter", goal_mode="Made Up Mode",
                     transition_cue=None),
        ],
    )
    missing = validate_plan_modes(plan, tax)
    assert missing == ["Made Up Mode"]


def test_validate_plan_modes_walks_branches(tmp_path):
    p = _write(tmp_path, """
        environment: e
        modes:
          "Road: On":      [0]
          "Bridge: Enter": [1]
          "Bridge: On":    [2]
    """)
    tax = load_taxonomy(p)

    branch_default = Branch(
        vlm_cue="default",
        sub_plan=[
            PlanStep(step=0, description="bad",
                     start_mode="Bridge: Enter", goal_mode="Mystery Cluster",
                     transition_cue=None),
        ],
    )
    branch_blocked = Branch(
        vlm_cue="blocked",
        sub_plan=[
            PlanStep(step=0, description="ok",
                     start_mode="Bridge: Enter", goal_mode="Road: On",
                     transition_cue=None),
        ],
    )
    plan = NavPlan(
        plan_name="x", description="x",
        steps=[
            PlanStep(step=0, description="approach",
                     start_mode="Road: On", goal_mode="Bridge: Enter"),
            PlanStep(step=1, description="decide",
                     start_mode="Bridge: Enter", goal_mode="Bridge: Enter",
                     transition_cue="dp",
                     branches=[branch_blocked, branch_default]),
        ],
    )
    assert validate_plan_modes(plan, tax) == ["Mystery Cluster"]


# --------------------------------------------------------------------------- #
# seed_taxonomy bootstrap                                                      #
# --------------------------------------------------------------------------- #

def test_invert_label_map_groups_and_sorts():
    raw = {"-1": "Along Wall", "0": "In Int", "1": "In Int", "5": "Road", "3": "Road"}
    inv = invert_label_map(raw)
    assert list(inv.keys()) == ["Along Wall", "In Int", "Road"]
    assert inv["In Int"] == [0, 1]
    assert inv["Road"] == [3, 5]


def test_render_yaml_is_parseable():
    raw = {"0": "A", "1": "A", "2": "B"}
    body = render_yaml(environment="testenv", source="f.json", grouped=invert_label_map(raw))
    import yaml
    parsed = yaml.safe_load(body)
    assert parsed["environment"] == "testenv"
    assert parsed["modes"]["A"] == [0, 1]
    assert parsed["modes"]["B"] == [2]
