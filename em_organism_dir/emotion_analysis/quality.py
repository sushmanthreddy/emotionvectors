"""Quality gates for reduced RQ1 emotion directions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReliabilityGate:
    """Split-half and cross-model evidence at preregistered layers."""

    aligned_reliability: FloatArray
    misaligned_reliability: FloatArray
    cross_model_stability: FloatArray
    aligned_median_by_layer: FloatArray
    misaligned_median_by_layer: FloatArray
    cross_model_median_by_layer: FloatArray
    preregistered_layers: tuple[int, ...]
    threshold: float
    layer_passes: tuple[bool, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Emotion-label retrieval metrics over independent examples."""

    top1_accuracy: float
    top5_accuracy: float
    mean_reciprocal_rank: float
    ranks: npt.NDArray[np.int64]


def rowwise_cosine(left: Any, right: Any) -> FloatArray:
    """Cosine over the final axis for identically shaped arrays."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1:
        raise ValueError(f"cosine inputs must have the same non-scalar shape: {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine inputs contain non-finite values")
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(denominator <= np.finfo(np.float64).eps):
        raise ValueError("cosine input contains a zero vector")
    return np.clip(np.sum(a * b, axis=-1) / denominator, -1.0, 1.0)


def evaluate_reliability_gate(
    aligned_split_a: Any,
    aligned_split_b: Any,
    misaligned_split_a: Any,
    misaligned_split_b: Any,
    aligned_full: Any,
    misaligned_full: Any,
    *,
    preregistered_layers: tuple[int, ...] = (24, 32),
    threshold: float = 0.7,
) -> ReliabilityGate:
    """Require both models to be split-half reliable at one primary layer."""

    aligned_reliability = rowwise_cosine(aligned_split_a, aligned_split_b)
    misaligned_reliability = rowwise_cosine(misaligned_split_a, misaligned_split_b)
    stability = rowwise_cosine(aligned_full, misaligned_full)
    if aligned_reliability.ndim != 2:
        raise ValueError("emotion directions must have shape [emotion, layer, hidden]")
    n_layers = aligned_reliability.shape[1]
    layers = tuple(int(layer) for layer in preregistered_layers)
    if not layers or any(layer < 0 or layer >= n_layers for layer in layers):
        raise ValueError("preregistered layer is outside the direction arrays")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("reliability threshold must be in [0, 1]")

    aligned_median = np.median(aligned_reliability, axis=0)
    misaligned_median = np.median(misaligned_reliability, axis=0)
    stability_median = np.median(stability, axis=0)
    layer_passes = tuple(
        bool(aligned_median[layer] >= threshold and misaligned_median[layer] >= threshold)
        for layer in layers
    )
    return ReliabilityGate(
        aligned_reliability=aligned_reliability,
        misaligned_reliability=misaligned_reliability,
        cross_model_stability=stability,
        aligned_median_by_layer=aligned_median,
        misaligned_median_by_layer=misaligned_median,
        cross_model_median_by_layer=stability_median,
        preregistered_layers=layers,
        threshold=float(threshold),
        layer_passes=layer_passes,
        passed=any(layer_passes),
    )


def retrieval_metrics(scores: Any, true_label_indices: Any, *, top_k: int = 5) -> RetrievalMetrics:
    """Score emotion retrieval from an ``[example, emotion]`` projection matrix."""

    matrix = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(true_label_indices, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("scores must be a non-empty [example, emotion] matrix")
    if labels.shape != (matrix.shape[0],):
        raise ValueError("true_label_indices must contain one label per example")
    if not np.isfinite(matrix).all():
        raise ValueError("scores contain non-finite values")
    if np.any((labels < 0) | (labels >= matrix.shape[1])):
        raise ValueError("true label index is outside the score matrix")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Stable sorting makes ties deterministic and therefore reproducible.
    order = np.argsort(-matrix, axis=1, kind="stable")
    ranks = np.empty(matrix.shape[0], dtype=np.int64)
    for row, label in enumerate(labels):
        ranks[row] = int(np.flatnonzero(order[row] == label)[0]) + 1
    effective_k = min(int(top_k), matrix.shape[1])
    return RetrievalMetrics(
        top1_accuracy=float(np.mean(ranks == 1)),
        top5_accuracy=float(np.mean(ranks <= effective_k)),
        mean_reciprocal_rank=float(np.mean(1.0 / ranks)),
        ranks=ranks,
    )


__all__ = [
    "ReliabilityGate",
    "RetrievalMetrics",
    "evaluate_reliability_gate",
    "retrieval_metrics",
    "rowwise_cosine",
]
