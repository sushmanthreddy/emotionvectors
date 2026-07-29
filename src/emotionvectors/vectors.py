"""Small, safe helpers for loading released layer-wise emotion vectors."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch

from .constants import EMOTIONS

VectorKind = Literal["raw", "raw_unit", "clean", "clean_unit"]

VECTOR_FILENAMES: dict[VectorKind, str] = {
    "raw": "emotion_vectors_raw.pt",
    "raw_unit": "emotion_vectors_unit.pt",
    "clean": "emotion_vectors_clean.pt",
    "clean_unit": "emotion_vectors_clean_unit.pt",
}


def load_vector_file(
    path: str | Path,
    *,
    emotion_order: Sequence[str] | None = None,
    expected_layers: int | None = None,
    expected_hidden_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Load a vector dictionary whose values have shape ``[layer, hidden]``.

    The original 7B release remains the default when its canonical labels are
    detected. New model bundles can supply their ordered labels and dimensions
    explicitly or through :func:`load_vectors`.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Vector file must contain a nonempty emotion dictionary")
    if not all(isinstance(label, str) and label for label in payload):
        raise ValueError("Every vector key must be a nonempty emotion label")

    if emotion_order is None:
        emotion_order = (
            EMOTIONS if set(payload) == set(EMOTIONS) else tuple(payload)
        )
    else:
        emotion_order = tuple(emotion_order)
    if len(set(emotion_order)) != len(emotion_order):
        raise ValueError("Emotion order contains duplicate labels")
    if set(payload) != set(emotion_order):
        missing = set(emotion_order) - set(payload)
        extra = set(payload) - set(emotion_order)
        raise ValueError(
            "Vector keys differ from the requested emotion order; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    vectors: dict[str, torch.Tensor] = {}
    inferred_shape: tuple[int, int] | None = None
    for emotion in emotion_order:
        value = payload[emotion]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Vector for {emotion!r} is not a tensor")
        value = value.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim != 2:
            raise ValueError(
                f"Vector for {emotion!r} has shape {tuple(value.shape)}; "
                "expected [layer, hidden]"
            )
        value_shape = (int(value.shape[0]), int(value.shape[1]))
        if inferred_shape is None:
            inferred_shape = value_shape
        if value_shape != inferred_shape:
            raise ValueError(
                f"Vector for {emotion!r} has shape {value_shape}, "
                f"expected the common shape {inferred_shape}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Vector for {emotion!r} contains NaN or infinity")
        vectors[emotion] = value

    assert inferred_shape is not None
    if expected_layers is not None and inferred_shape[0] != expected_layers:
        raise ValueError(
            f"Vector layer count is {inferred_shape[0]}, "
            f"expected {expected_layers}"
        )
    if (
        expected_hidden_size is not None
        and inferred_shape[1] != expected_hidden_size
    ):
        raise ValueError(
            f"Vector hidden size is {inferred_shape[1]}, "
            f"expected {expected_hidden_size}"
        )
    return vectors


def load_vectors(
    artifact_dir: str | Path,
    *,
    kind: VectorKind = "clean_unit",
) -> dict[str, torch.Tensor]:
    """Load one released vector family using its bundle manifest when present."""

    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        emotion_order = manifest.get("emotion_order")
        expected_layers = manifest.get("number_of_layers")
        expected_hidden_size = manifest.get("hidden_size")
    else:
        emotion_order = None
        expected_layers = None
        expected_hidden_size = None
    return load_vector_file(
        artifact_dir / VECTOR_FILENAMES[kind],
        emotion_order=emotion_order,
        expected_layers=expected_layers,
        expected_hidden_size=expected_hidden_size,
    )


def get_vector(
    vectors: dict[str, torch.Tensor],
    emotion: str,
    *,
    layer: int,
) -> torch.Tensor:
    """Return one ``[hidden]`` direction at a zero-based layer index."""

    if emotion not in vectors:
        raise ValueError(
            f"Unknown emotion {emotion!r}; choose from {tuple(vectors)}"
        )
    value = vectors[emotion]
    if value.ndim != 2:
        raise ValueError(
            f"Vector for {emotion!r} must have shape [layer, hidden]"
        )
    if not 0 <= layer < value.shape[0]:
        raise ValueError(f"layer must be in [0, {value.shape[0] - 1}]")
    return vectors[emotion][layer]


def stack_vectors(
    vectors: dict[str, torch.Tensor],
    *,
    emotion_order: Sequence[str] | None = None,
) -> torch.Tensor:
    """Stack a vector dictionary as ``[emotion, layer, hidden]``."""

    if not vectors:
        raise ValueError("Cannot stack an empty vector dictionary")
    if emotion_order is None:
        emotion_order = tuple(vectors)
    else:
        emotion_order = tuple(emotion_order)
    missing = set(emotion_order) - set(vectors)
    extra = set(vectors) - set(emotion_order)
    if missing or extra:
        raise ValueError(
            f"Vector keys differ from the requested order; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    stacked = torch.stack([vectors[emotion] for emotion in emotion_order], dim=0)
    if stacked.ndim != 3:
        raise ValueError(
            f"Stacked vector shape is {tuple(stacked.shape)}, "
            "expected [emotion, layer, hidden]"
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
