"""Unit tests for nl_planner.stl_syntax.quick_syntax_check.

The check is regex-based and deterministic. Every failure mode below is one
we have actually observed in live planner_node runs against gpt-4o-mini.
"""
from __future__ import annotations

import pytest

from nl_planner.stl_syntax import quick_syntax_check


# --------------------------------------------------------------------------- #
# Valid formulas (should pass)                                                 #
# --------------------------------------------------------------------------- #

VALID_FORMULAS = [
    # Bare predicate + macro
    "Detect(StopSign) \\land \\mathbf{F}\\Phi_{Int_Turn}",
    # Same with escaped underscore
    "Detect(StopSign) \\land \\mathbf{F}\\Phi_{Int\\_Turn}",
    # \text{}-wrapped name and arg with spaces
    "\\text{Detect}(\\text{Stop Sign}) \\land \\mathbf{F}\\Phi_{Bridge}",
    # Branch (\lor) with mixed styles
    (
        "(\\text{Detect}(\\text{BlockedBridge}) \\land \\mathbf{F}\\Phi_{LongRoad}) "
        "\\lor (\\lnot \\text{Detect}(\\text{BlockedBridge}) \\land \\mathbf{F}\\Phi_{Bridge})"
    ),
    # \text{} wrapper around argument only
    "Detect(\\text{Wall}) \\lor Detect(\\text{Open Space})",
    # \text{} wrapper around argument with punctuation
    "Detect(\\text{End of Bridge})",
    # Display math wrapper
    "$$\\Phi_{Road} \\ \\mathbf{U} \\ \\text{Detect}(\\text{End of Bridge})$$",
    # Multiple temporal layers
    "\\Phi_{Road} \\ \\mathbf{U} \\ \\big( \\text{Detect}(\\text{StopSign}) \\land \\mathbf{F}\\Phi_{Int\\_Turn} \\big)",
    # Bearing predicate
    "Bearing(Right) \\land \\mathbf{F}\\Phi_{Road}",
]


@pytest.mark.parametrize("stl", VALID_FORMULAS)
def test_valid_formulas_pass(stl: str):
    ok, err = quick_syntax_check(stl)
    assert ok, f"expected pass but got error: {err}\nformula was: {stl}"
    assert err is None


# --------------------------------------------------------------------------- #
# Invalid formulas — each maps to a specific failure mode we have seen        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stl,expect_rule", [
    # Empty
    ("", "Rule 0"),
    ("   ", "Rule 0"),
    # Unbalanced
    ("Detect(Bridge", "Rule 1"),
    ("Detect(Bridge))", "Rule 1"),
    ("\\Phi_{Int_Pass", "Rule 1"),  # missing }
    # ASCII boolean
    ("Detect(Bridge) && \\Phi_{Road}", "Rule 5"),
    ("Detect(Bridge) || Detect(Wall)", "Rule 5"),
    # ASCII temporal — bare F before predicate
    ("F Detect(Bridge)", "Rule 5"),
    # Invented macro
    ("\\Phi_{GoForward} \\land Detect(Bridge)", "Rule 2"),
    ("\\Phi_{TurnRight} \\lor \\Phi_{Bridge}", "Rule 2"),
    # Mode-vocab leak (THE failure mode that prompted the programmatic check)
    ("Detect(Open Space) \\land \\Phi_{Road}", "Rule 3"),
    ("Bearing(Hard Left) \\land \\Phi_{Road}", "Rule 3"),
    # Quoted arg
    ('Detect("OpenSpace") \\land \\Phi_{Road}', "Rule 3"),
    # Punctuation in bare arg
    ("Detect(Open-Space) \\land \\Phi_{Road}", "Rule 3"),
    # Empty predicate arg
    ("Detect() \\land \\Phi_{Road}", "Rule 3"),
])
def test_invalid_formulas_caught(stl: str, expect_rule: str):
    ok, err = quick_syntax_check(stl)
    assert not ok, f"expected fail but got pass for {stl!r}"
    assert err is not None
    assert expect_rule in err, (
        f"expected error to mention {expect_rule!r} for {stl!r}; got {err!r}"
    )


# --------------------------------------------------------------------------- #
# Targeted behavior tests                                                      #
# --------------------------------------------------------------------------- #

def test_camel_case_argument_with_digits_is_form_a():
    # Argument like 'Intersection1' is a single CamelCase token with a digit.
    ok, err = quick_syntax_check("Detect(Intersection1)")
    assert ok, err


def test_form_b_with_basic_punctuation_inside_text():
    # Periods, dashes, commas, colons, slashes are allowed inside \text{...}.
    ok, err = quick_syntax_check("Detect(\\text{End of Bridge, mile 1})")
    assert ok, err


def test_form_b_forbids_latex_command_inside_text():
    ok, err = quick_syntax_check("Detect(\\text{\\Phi_{Road}})")
    assert not ok
    assert "Rule 3" in err


def test_suggestion_mentions_camel_case_alternative():
    # The error message should propose a concrete fix.
    ok, err = quick_syntax_check("Detect(Open Space)")
    assert not ok
    # Suggestion should at least include 'OpenSpace' or the \text{} alternative.
    assert "OpenSpace" in err or "\\text{Open Space}" in err


def test_letter_inside_camel_case_does_not_trip_temporal_check():
    # "F" inside "EndOfBridge" should NOT be flagged as a bare temporal.
    ok, err = quick_syntax_check(
        "Detect(EndOfBridge) \\land \\mathbf{F}\\Phi_{Bridge}"
    )
    assert ok, err


def test_letter_inside_text_wrapper_does_not_trip_temporal_check():
    # "U" inside \text{Underpass} should NOT be flagged.
    ok, err = quick_syntax_check(
        "Detect(\\text{Underpass}) \\land \\mathbf{F}\\Phi_{Road}"
    )
    assert ok, err


def test_strip_display_math_wrapper():
    # The check should not be confused by `$$...$$` wrappers.
    ok, err = quick_syntax_check("$$Detect(Bridge) \\land \\Phi_{Bridge}$$")
    assert ok, err


def test_error_message_quotes_offending_substring():
    # Errors should contain the actual offending substring for the LLM to
    # repair on retry.
    ok, err = quick_syntax_check("Detect(Open Space) \\land \\Phi_{Road}")
    assert not ok
    assert "Open Space" in err
