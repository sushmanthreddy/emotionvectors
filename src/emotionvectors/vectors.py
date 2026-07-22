"""Small, safe helpers for loading and selecting released emotion vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from .constants import EMOTIONS, HIDDEN_SIZE, NUM_LAYERS

VectorKind = Literal["raw", "raw_unit", "clean", "clean_unit"]

VECTOR_FILENAMES: dict[VectorKind, str] = {
    "raw": "emotion_vectors_raw.pt",
    "raw_unit": "emotion_vectors_unit.pt",
    "clean": "emotion_vectors_clean.pt",
    "clean_unit": "emotion_vectors_clean_unit.pt",
}


def load_vector_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Load and validate a 12-emotion vector dictionary.

    Every value is returned on CPU as float32 with shape ``[28, 3584]``.
    ``weights_only=True`` prevents arbitrary objects from being unpickled.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != set(EMOTIONS):
        raise ValueError("Vector file must contain exactly the 12 canonical emotions")

    vectors: dict[str, torch.Tensor] = {}
    for emotion in EMOTIONS:
        value = payload[emotion]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Vector for {emotion!r} is not a tensor")
        value = value.detach().to(device="cpu", dtype=torch.float32)
        if tuple(value.shape) != (NUM_LAYERS, HIDDEN_SIZE):
            raise ValueError(
                f"Vector for {emotion!r} has shape {tuple(value.shape)}, "
                f"expected {(NUM_LAYERS, HIDDEN_SIZE)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Vector for {emotion!r} contains NaN or infinity")
        vectors[emotion] = value
    return vectors


def load_vectors(
    artifact_dir: str | Path,
    *,
    kind: VectorKind = "clean_unit",
) -> dict[str, torch.Tensor]:
    """Load one released vector family from an artifact directory."""

    return load_vector_file(Path(artifact_dir) / VECTOR_FILENAMES[kind])


def get_vector(
    vectors: dict[str, torch.Tensor],
    emotion: str,
    *,
    layer: int,
) -> torch.Tensor:
    """Return one ``[3584]`` emotion direction at a zero-based layer index."""

    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion {emotion!r}; choose from {EMOTIONS}")
    if not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"layer must be in [0, {NUM_LAYERS - 1}]")
    return vectors[emotion][layer]


def stack_vectors(vectors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack a validated vector dictionary as ``[12, 28, 3584]``."""

    # Reuse the public validator semantics without writing a duplicate artifact.
    missing = set(EMOTIONS) - set(vectors)
    extra = set(vectors) - set(EMOTIONS)
    if missing or extra:
        raise ValueError(
            f"Vector keys differ from the canonical emotions; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    stacked = torch.stack([vectors[emotion] for emotion in EMOTIONS], dim=0)
    if tuple(stacked.shape) != (len(EMOTIONS), NUM_LAYERS, HIDDEN_SIZE):
        raise ValueError(
            f"Stacked vector shape is {tuple(stacked.shape)}, expected "
            f"{(len(EMOTIONS), NUM_LAYERS, HIDDEN_SIZE)}"
        )
    return stacked


__all__ = [
    "VECTOR_FILENAMES",
    "VectorKind",
    "get_vector",
    "load_vector_file",
    "load_vectors",
    "stack_vectors",
]
