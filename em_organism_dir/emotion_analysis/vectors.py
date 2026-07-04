"""Per-example emotion-direction construction and common neutral-PCA projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EmotionDirectionSet:
    """Full-data and topic-split directions for one model."""

    emotions: tuple[str, ...]
    full: FloatArray
    split_a: FloatArray
    split_b: FloatArray
    counts: npt.NDArray[np.int64]
    split_a_counts: npt.NDArray[np.int64]
    split_b_counts: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class NeutralPCA:
    """A separately fitted neutral PCA basis for every layer."""

    components: tuple[FloatArray, ...]
    explained_variance_ratio: tuple[FloatArray, ...]
    retained_components: npt.NDArray[np.int64]
    neutral_means: FloatArray


def _finite_float(value: Any, *, name: str) -> Float64Array:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def difference_from_other_emotions(emotion_means: Any) -> FloatArray:
    """Return ``mu_e - mean(mu_other)`` for every emotion independently.

    This is the exact construction requested for RQ1.  It is a positive scalar
    multiple of grand-mean centering for a fixed emotion set, but retaining the
    explicit definition avoids ambiguity in stored artifacts and tests.
    """

    means = _finite_float(emotion_means, name="emotion_means")
    if means.ndim < 2 or means.shape[0] < 2:
        raise ValueError("emotion_means must be [emotion, ...] with at least two emotions")
    total = means.sum(axis=0, dtype=np.float64)
    others = (total[None, ...] - means) / float(means.shape[0] - 1)
    return (means - others).astype(np.float32)


def _validate_examples(
    vectors_by_emotion: Mapping[str, Any],
    topic_ids_by_emotion: Mapping[str, Sequence[int]],
    emotions: Sequence[str],
) -> tuple[tuple[str, ...], tuple[int, int]]:
    ordered = tuple(emotions)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("emotions must contain at least two unique labels")
    expected_shape: tuple[int, int] | None = None
    for emotion in ordered:
        if emotion not in vectors_by_emotion or emotion not in topic_ids_by_emotion:
            raise KeyError(f"missing per-example data for {emotion!r}")
        values = np.asarray(vectors_by_emotion[emotion])
        topics = tuple(topic_ids_by_emotion[emotion])
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError(f"{emotion}: vectors must be non-empty [example, layer, hidden]")
        if values.shape[0] != len(topics):
            raise ValueError(f"{emotion}: topic IDs do not match example count")
        if not np.isfinite(values).all():
            raise ValueError(f"{emotion}: vectors contain non-finite values")
        shape = (int(values.shape[1]), int(values.shape[2]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"{emotion}: layer/hidden shape {shape} != {expected_shape}")
    assert expected_shape is not None
    return ordered, expected_shape


def build_emotion_directions(
    vectors_by_emotion: Mapping[str, Any],
    topic_ids_by_emotion: Mapping[str, Sequence[int]],
    emotions: Sequence[str],
    *,
    split_a_topics: Sequence[int],
    split_b_topics: Sequence[int],
) -> EmotionDirectionSet:
    """Build full and topic-split directions from per-example activations."""

    ordered, _ = _validate_examples(vectors_by_emotion, topic_ids_by_emotion, emotions)
    split_a = frozenset(int(value) for value in split_a_topics)
    split_b = frozenset(int(value) for value in split_b_topics)
    if not split_a or not split_b or split_a & split_b:
        raise ValueError("topic splits must be non-empty and disjoint")

    full_means: list[Float64Array] = []
    a_means: list[Float64Array] = []
    b_means: list[Float64Array] = []
    counts: list[int] = []
    a_counts: list[int] = []
    b_counts: list[int] = []
    for emotion in ordered:
        values = _finite_float(vectors_by_emotion[emotion], name=emotion)
        topics = np.asarray(topic_ids_by_emotion[emotion], dtype=np.int64)
        mask_a = np.isin(topics, tuple(split_a))
        mask_b = np.isin(topics, tuple(split_b))
        if not bool(mask_a.any()) or not bool(mask_b.any()):
            raise ValueError(f"{emotion}: both topic splits require at least one example")
        outside = ~(mask_a | mask_b)
        if bool(outside.any()):
            raise ValueError(f"{emotion}: examples contain topic IDs outside both splits")
        full_means.append(values.mean(axis=0, dtype=np.float64))
        a_means.append(values[mask_a].mean(axis=0, dtype=np.float64))
        b_means.append(values[mask_b].mean(axis=0, dtype=np.float64))
        counts.append(int(values.shape[0]))
        a_counts.append(int(mask_a.sum()))
        b_counts.append(int(mask_b.sum()))

    return EmotionDirectionSet(
        emotions=ordered,
        full=difference_from_other_emotions(np.stack(full_means)),
        split_a=difference_from_other_emotions(np.stack(a_means)),
        split_b=difference_from_other_emotions(np.stack(b_means)),
        counts=np.asarray(counts, dtype=np.int64),
        split_a_counts=np.asarray(a_counts, dtype=np.int64),
        split_b_counts=np.asarray(b_counts, dtype=np.int64),
    )


def fit_neutral_pca(neutral_activations: Any, *, variance_threshold: float = 0.5) -> NeutralPCA:
    """Fit a layer-specific PCA basis and retain the 50%-variance prefix."""

    neutral = np.asarray(neutral_activations, dtype=np.float32)
    if neutral.ndim != 3 or neutral.shape[0] < 2:
        raise ValueError("neutral_activations must be [example, layer, hidden] with >=2 examples")
    if not np.isfinite(neutral).all():
        raise ValueError("neutral_activations contain non-finite values")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")

    components: list[FloatArray] = []
    ratios: list[FloatArray] = []
    retained: list[int] = []
    means = neutral.mean(axis=0, dtype=np.float64).astype(np.float32)
    for layer in range(neutral.shape[1]):
        samples = neutral[:, layer, :].astype(np.float64)
        centered = samples - samples.mean(axis=0, keepdims=True)
        _left, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        variances = np.square(singular_values) / float(samples.shape[0] - 1)
        total = float(variances.sum())
        if total <= np.finfo(np.float64).eps:
            layer_ratios = np.zeros_like(variances)
            count = 0
        else:
            layer_ratios = variances / total
            count = int(
                np.searchsorted(np.cumsum(layer_ratios), variance_threshold, side="left") + 1
            )
        count = min(count, int(vh.shape[0]))
        components.append(np.ascontiguousarray(vh[:count], dtype=np.float32))
        ratios.append(np.asarray(layer_ratios, dtype=np.float32))
        retained.append(count)
    return NeutralPCA(
        components=tuple(components),
        explained_variance_ratio=tuple(ratios),
        retained_components=np.asarray(retained, dtype=np.int64),
        neutral_means=means,
    )


def project_out_neutral_components(vectors: Any, pca: NeutralPCA) -> FloatArray:
    """Apply one common per-layer complement projection to any vector collection.

    Input may be ``[layer, hidden]`` or have arbitrary leading axes ending in
    ``[layer, hidden]``.  The same fitted basis can therefore clean both emotion
    and EM directions without mixing projections.
    """

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("vectors must end in [layer, hidden]")
    n_layers, hidden = values.shape[-2:]
    if len(pca.components) != n_layers:
        raise ValueError("PCA layer count does not match vectors")
    cleaned = values.astype(np.float64, copy=True)
    for layer, raw_components in enumerate(pca.components):
        pcs = np.asarray(raw_components, dtype=np.float64)
        if pcs.ndim != 2 or pcs.shape[1] != hidden:
            raise ValueError(f"layer {layer}: PCA components have invalid shape {pcs.shape}")
        if pcs.shape[0] == 0:
            continue
        selected = cleaned[..., layer, :]
        cleaned[..., layer, :] = selected - (selected @ pcs.T) @ pcs
    return cleaned.astype(np.float32)


__all__ = [
    "EmotionDirectionSet",
    "NeutralPCA",
    "build_emotion_directions",
    "difference_from_other_emotions",
    "fit_neutral_pca",
    "project_out_neutral_components",
]
