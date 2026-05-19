"""Fast programmatic pre-check for STL formulas.

Why this exists
---------------
The LLM-based ``verify_syntax`` agent reads our prompt and decides whether a
formula is well-formed. In practice, the LLM gets the easy cases right but
disagrees with itself across runs on edge cases, and it costs a round-trip per
attempt. This module is a deterministic regex-based check that runs **before**
the LLM verifier and catches the common failure modes we have actually seen
from the generator:

1. Unbalanced ``()``/``{}``.
2. Predicate arguments that are neither bare CamelCase nor a ``\\text{...}``
   wrapper (e.g. ``Detect(Open Space)``).
3. Invented ``\\Phi_{...}`` macros (e.g. ``\\Phi_{GoForward}``).
4. ASCII boolean / temporal operators (``&&``, ``||``, bare ``F``).

When it rejects, it returns a SPECIFIC error string that the generator's
retry feedback can use verbatim — the LLM gets a fixed-instruction-style
diff rather than another vague "syntax fail" message.

This is **not** a complete STL grammar — anything not on the bullet list
above is delegated to the LLM verifier downstream.
"""
from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Allow-lists (kept in lockstep with prompts/verify_syntax.md)                #
# --------------------------------------------------------------------------- #

# Canonical macro spellings. Either form of underscore (`_` or `\_`) is OK.
_ALLOWED_MACRO_BODIES: frozenset[str] = frozenset({
    "Int_Pass",
    "Int_Turn",
    "Bridge",
    "Build_Past",
    "Road",
    "LongRoad",
})

_FORM_A_ARG = re.compile(r"^[A-Z][A-Za-z0-9]*$")
# Form B: \text{<phrase>} — letters/digits/spaces/commas/dots/dashes/colons/slashes.
# We deliberately exclude `\` so LaTeX commands cannot appear inside.
_FORM_B_ARG = re.compile(r"^\\text\{[A-Za-z0-9 ,\.\-:/]+\}$")

# Macro detection: `\Phi_{Body}` or `\Phi_{Body\_With\_Escaped\_Underscore}`.
_MACRO_RE = re.compile(r"\\Phi_\{([A-Za-z0-9_\\]+)\}")

# Predicate detection: `\text{Detect}(...)` or `Detect(...)` (likewise Bearing).
# Captures: 1=name, 2=arg.
_PREDICATE_RE = re.compile(
    r"(?:\\text\{(?P<n1>Detect|Bearing)\}|(?P<n2>Detect|Bearing))"
    r"\((?P<arg>[^()]*)\)"
)

# Forbidden ASCII operators. We match them as standalone tokens so we don't
# false-positive on letters inside CamelCase identifiers (e.g. `F` in
# `EndOfBridge` is fine; a lone `F` between tokens is not).
_FORBIDDEN_ASCII = [
    (re.compile(r"(?<![A-Za-z\\])(&&)(?!\w)"),       "&&"),
    (re.compile(r"(?<![A-Za-z\\])(\|\|)(?!\w)"),     "||"),
    (re.compile(r"(?<![A-Za-z\\])(!)(?!=)"),         "!"),
    # Bare standalone temporal letters — must be space-delimited so we don't
    # match the F in `EndOfBridge`.
    (re.compile(r"(?<![A-Za-z\\{])([UFGX])(?![A-Za-z}])"), "<bare temporal>"),
]


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def quick_syntax_check(stl: str) -> tuple[bool, str | None]:
    """Run the deterministic pre-check.

    Returns ``(True, None)`` if no obvious violation is found (the LLM
    verifier still has a chance to reject). Returns ``(False, error)``
    with a specific, actionable error message suitable for retry feedback.
    """
    if not stl or not stl.strip():
        return False, "Rule 0: empty STL formula."

    body = _strip_display_math(stl)

    # --- Rule 1: balanced brackets ---
    if body.count("(") != body.count(")"):
        return False, (
            f"Rule 1 (balanced brackets): unmatched parenthesis — "
            f"{body.count('(')} `(` vs {body.count(')')} `)`. "
            "Recount and ensure every `(` has a matching `)`."
        )
    if body.count("{") != body.count("}"):
        return False, (
            f"Rule 1 (balanced brackets): unmatched curly brace — "
            f"{body.count('{')} `{{` vs {body.count('}')} `}}`."
        )
    if body.count("[") != body.count("]"):
        return False, (
            f"Rule 1 (balanced brackets): unmatched square bracket — "
            f"{body.count('[')} `[` vs {body.count(']')} `]`."
        )

    # --- Rule 5: no ASCII boolean ops ---
    for pat, label in _FORBIDDEN_ASCII:
        m = pat.search(body)
        if m:
            return False, (
                f"Rule 5 (boolean/temporal operators): forbidden ASCII token "
                f"`{m.group(0)}` near `...{_context(body, m.start())}...`. "
                "Use `\\land`, `\\lor`, `\\lnot`, `\\mathbf{U}`, `\\mathbf{F}` "
                "instead."
            )

    # --- Rule 2: macros are from the closed allow-list ---
    for m in _MACRO_RE.finditer(body):
        normalized = m.group(1).replace("\\_", "_")
        if normalized not in _ALLOWED_MACRO_BODIES:
            allowed = sorted(_ALLOWED_MACRO_BODIES)
            backslash = "\\"
            allowed_str = ", ".join(
                backslash + "Phi_{" + a + "}" for a in allowed
            )
            return False, (
                f"Rule 2 (allowed macros): unknown macro `{m.group(0)}`. "
                f"The macro allow-list is CLOSED: only "
                f"{allowed_str} are accepted. "
                "If your route doesn't match one of these, describe it with "
                "Detect(...) / Bearing(...) predicates plus the JSON `steps` "
                "topology — do not invent new macros."
            )

    # --- Rule 3: every Detect/Bearing argument is Form A or Form B ---
    for m in _PREDICATE_RE.finditer(body):
        name = m.group("n1") or m.group("n2")
        arg = m.group("arg").strip()
        if not arg:
            return False, (
                f"Rule 3 (predicate arguments): `{name}()` has an empty "
                "argument."
            )
        if _FORM_A_ARG.match(arg):
            continue
        if _FORM_B_ARG.match(arg):
            continue
        suggestion = _suggest_fix(arg)
        return False, (
            f"Rule 3 (predicate arguments): argument `{arg}` (in "
            f"`{m.group(0)}`) is neither Form A (CamelCase identifier) "
            f"nor Form B (\\text{{phrase}}). "
            f"Rewrite as Form A `{suggestion}` or Form B "
            f"`\\text{{{arg}}}`. Mode names with spaces / colons / slashes "
            "must NEVER appear bare inside Detect(...) or Bearing(...) — "
            "they belong in the JSON `start_mode`/`goal_mode` fields, not "
            "in STL predicate arguments."
        )

    return True, None


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _strip_display_math(stl: str) -> str:
    """Drop a single matching outer `$$...$$` or `$...$` wrapper."""
    s = stl.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        return s[2:-2]
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1]
    return s


def _suggest_fix(arg: str) -> str:
    """Best-effort CamelCase suggestion for a malformed predicate argument."""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", arg)
    parts = [p for p in cleaned.split() if p]
    if not parts:
        return "Landmark"
    return "".join(p[0].upper() + p[1:] for p in parts)


def _context(s: str, idx: int, span: int = 12) -> str:
    """Return a small substring centred on ``idx`` for error messages."""
    lo = max(0, idx - span)
    hi = min(len(s), idx + span)
    return s[lo:hi].replace("\n", " ")


__all__ = ["quick_syntax_check"]
