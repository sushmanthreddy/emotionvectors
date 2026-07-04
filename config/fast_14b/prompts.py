"""Reuse the verbatim Appendix-C prompts for the fast 14B preset."""

from __future__ import annotations

import runpy
from pathlib import Path

_PROMPTS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "prompts.py"))

EMOTIONAL_STORIES_PROMPT = _PROMPTS["EMOTIONAL_STORIES_PROMPT"]
NEUTRAL_DIALOGUES_PROMPT = _PROMPTS["NEUTRAL_DIALOGUES_PROMPT"]
PROBE_HE = _PROMPTS["PROBE_HE"]
PROBE_I = _PROMPTS["PROBE_I"]

__all__ = [
    "EMOTIONAL_STORIES_PROMPT",
    "NEUTRAL_DIALOGUES_PROMPT",
    "PROBE_HE",
    "PROBE_I",
]
