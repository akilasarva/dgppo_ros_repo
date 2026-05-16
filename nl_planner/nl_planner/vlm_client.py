"""Tiny OpenAI-vision wrappers for the executor's runtime decisions.

This intentionally duplicates the OpenAI plumbing in ``brain_controller.py``
(``_run_vlm_check``) rather than importing it — the brain package needs to
stay independently runnable for the old hand-authored-plan workflow, and we
do not want a hard dependency from nl_planner -> brain.

Two helpers:

- ``confirm_cue(image, cue) -> bool`` — yes/no question on a single cue.
  (Available for diagnostic use; the executor mostly leans on
  ``brain_controller``'s own cue checks.)
- ``choose_branch(image, branches) -> int`` — multi-choice over the
  ``Branch`` list. Returns the index of the winning branch, falling back to
  the ``default`` branch on any ambiguity.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Iterable

from .schemas import DEFAULT_BRANCH_CUE, Branch

logger = logging.getLogger(__name__)


class VLMClient:
    """Stateless OpenAI vision helper."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai is required for VLMClient. Install with: pip install openai"
            ) from exc
        self._openai_module = openai
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def confirm_cue(self, image_jpeg_bytes: bytes, cue: str) -> bool:
        """Yes/no: is ``cue`` visible in the image?"""
        prompt = (
            "You are assisting a mobile robot navigate. Look at this image "
            "carefully and answer ONLY with YES or NO. "
            f"Question: Is '{cue}' visible in this image?"
        )
        answer = self._ask_text(prompt, image_jpeg_bytes, max_tokens=5)
        return "YES" in answer.upper()

    def choose_branch(self, image_jpeg_bytes: bytes, branches: Iterable[Branch]) -> int:
        """Pick the branch whose ``vlm_cue`` best describes the image.

        Returns the index into ``branches``. The fallback ``"default"`` branch
        (always present per the schema validator) is chosen when no other
        cue matches or when the LLM response is unparseable.
        """
        branches = list(branches)
        if not branches:
            raise ValueError("choose_branch called with no branches")

        default_idx = _find_default(branches)

        lines = [
            "You are assisting a mobile robot decide between branching plans.",
            "Look at the image and choose the option whose description best",
            "matches what you see. If none clearly match, answer with the",
            f"number of the 'default' option ({default_idx + 1}).",
            "Answer with ONLY a single integer (the option number), no prose.",
            "",
            "Options:",
        ]
        for i, branch in enumerate(branches):
            label = "default (none of the above)" if i == default_idx else branch.vlm_cue
            lines.append(f"  {i + 1}) {label}")
        prompt = "\n".join(lines)

        answer = self._ask_text(prompt, image_jpeg_bytes, max_tokens=8)
        idx = _parse_int(answer)
        if idx is None or not (1 <= idx <= len(branches)):
            logger.warning(
                "VLM choose_branch returned unparseable %r; falling back to default (%d)",
                answer, default_idx,
            )
            return default_idx
        return idx - 1

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _ask_text(self, prompt: str, image_jpeg_bytes: bytes, *, max_tokens: int) -> str:
        b64 = base64.b64encode(image_jpeg_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url":    f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _find_default(branches: list[Branch]) -> int:
    for i, b in enumerate(branches):
        if b.vlm_cue.strip().lower() == DEFAULT_BRANCH_CUE:
            return i
    raise ValueError(
        "branches list has no 'default' entry — should have been rejected by the "
        "schema validator"
    )


def _parse_int(text: str) -> int | None:
    text = text.strip()
    # Common formats: "1", "1.", "1)", "Option 1", "1\n..."
    for token in (text.split() + [text]):
        token = token.strip(".)(,:;")
        if token.isdigit():
            return int(token)
    # As a last resort, grab the first digit anywhere
    for ch in text:
        if ch.isdigit():
            return int(ch)
    return None


__all__ = ["VLMClient"]
