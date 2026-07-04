from __future__ import annotations

import numpy as np

from em_organism_dir.emotion_analysis.quality import (
    evaluate_reliability_gate,
    retrieval_metrics,
    rowwise_cosine,
)


def test_rowwise_cosine_signed_values() -> None:
    left = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    right = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(rowwise_cosine(left, right), [1.0, -1.0, 0.0])


def test_reliability_gate_requires_both_models_at_same_layer() -> None:
    # Two emotions, two layers, two hidden dimensions.
    aligned_a = np.asarray([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=float)
    aligned_b = aligned_a.copy()
    misaligned_a = aligned_a.copy()
    misaligned_b = np.asarray([[[1, 0], [-1, 0]], [[0, 1], [0, -1]]], dtype=float)
    result = evaluate_reliability_gate(
        aligned_a,
        aligned_b,
        misaligned_a,
        misaligned_b,
        aligned_a,
        misaligned_a,
        preregistered_layers=(0, 1),
        threshold=0.7,
    )
    assert result.layer_passes == (True, False)
    assert result.passed


def test_retrieval_metrics_known_ranks() -> None:
    scores = np.asarray([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
    result = retrieval_metrics(scores, [0, 2, 2], top_k=2)
    assert result.ranks.tolist() == [1, 2, 3]
    assert result.top1_accuracy == 1 / 3
    assert result.top5_accuracy == 2 / 3
    assert result.mean_reciprocal_rank == (1.0 + 0.5 + 1 / 3) / 3
