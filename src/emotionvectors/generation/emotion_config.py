"""Validated, ordered emotion configurations for story generation."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmotionSpec:
    """One exact prompt label and its safe persistence stem."""

    emotion: str

    @property
    def slug(self) -> str:
        """Filesystem- and identifier-safe form of the canonical label."""

        return emotion_slug(self.emotion)


def load_emotion_config(path: Path) -> tuple[EmotionSpec, ...]:
    """Load the ordered flat-emotion JSON schema used by generation runs."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Emotion config must contain a JSON object: {path}")
    expected_fields = {"schema_version", "emotions"}
    if set(payload) != expected_fields:
        raise ValueError(
            f"Emotion config {path} must contain exactly "
            f"{sorted(expected_fields)!r}; found {sorted(payload)!r}"
        )
    if payload.get("schema_version") != 2:
        raise ValueError(f"Emotion config {path} must use schema_version 2")

    emotions = payload.get("emotions")
    if not isinstance(emotions, list) or not emotions:
        raise ValueError(f"Emotion config {path} must contain a non-empty emotions list")
    specs: list[EmotionSpec] = []
    seen_emotions: set[str] = set()
    seen_slugs: dict[str, str] = {}
    for emotion_index, raw_emotion in enumerate(emotions):
        emotion = _validated_label(
            raw_emotion,
            kind=f"emotion at index {emotion_index}",
            path=path,
        )
        if emotion in seen_emotions:
            raise ValueError(f"Duplicate emotion {emotion!r} in {path}")
        _register_slug(emotion, seen_slugs, path=path)
        seen_emotions.add(emotion)
        specs.append(EmotionSpec(emotion=emotion))

    return tuple(specs)


def build_emotion_specs(
    emotions: list[str] | tuple[str, ...],
) -> tuple[EmotionSpec, ...]:
    """Validate ordered ad-hoc CLI emotion labels."""

    specs: list[EmotionSpec] = []
    seen: set[str] = set()
    seen_slugs: dict[str, str] = {}
    for index, raw_emotion in enumerate(emotions):
        emotion = _validated_label(
            raw_emotion,
            kind=f"command-line emotion at index {index}",
            path=None,
        )
        if emotion in seen:
            raise ValueError(f"Duplicate emotion {emotion!r} on the command line")
        _register_slug(emotion, seen_slugs, path=None)
        seen.add(emotion)
        specs.append(EmotionSpec(emotion=emotion))
    if not specs:
        raise ValueError("At least one emotion is required")
    return tuple(specs)


def emotion_slug(emotion: str) -> str:
    """Create a stable ASCII stem without changing the canonical prompt label."""

    normalized = unicodedata.normalize("NFKD", emotion).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    if not slug:
        raise ValueError(f"Emotion {emotion!r} has no usable ASCII filename characters")
    return slug


def _register_slug(
    emotion: str,
    seen_slugs: dict[str, str],
    *,
    path: Path | None,
) -> None:
    slug = emotion_slug(emotion)
    previous = seen_slugs.get(slug)
    if previous is not None:
        location = f" in {path}" if path is not None else ""
        raise ValueError(
            f"Emotion labels {previous!r} and {emotion!r}{location} both map to "
            f"the record stem {slug!r}"
        )
    seen_slugs[slug] = emotion


def _validated_label(raw: Any, *, kind: str, path: Path | None) -> str:
    location = f" in {path}" if path is not None else ""
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Invalid {kind}{location}: expected a non-empty string")
    if raw != raw.strip():
        raise ValueError(f"Invalid {kind}{location}: leading/trailing whitespace is forbidden")
    if raw in {".", ".."} or "/" in raw or "\\" in raw or "\0" in raw:
        raise ValueError(
            f"Invalid {kind}{location}: labels must also be safe record filenames"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise ValueError(f"Invalid {kind}{location}: control characters are forbidden")
    return raw


__all__ = [
    "EmotionSpec",
    "build_emotion_specs",
    "emotion_slug",
    "load_emotion_config",
]
