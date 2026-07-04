from __future__ import annotations

import numpy as np
import pytest

from em_organism_dir.emotion_analysis.geometry import (
    cosine_similarity,
    layerwise_normalized_centroid,
    layerwise_subspace_projection,
    normalized_centroid,
    project_onto_subspace,
)
from em_organism_dir.emotion_analysis.statistics import (
    TopicBootstrapDraws,
    benjamini_hochberg,
    draw_topic_bootstrap,
    empirical_p_value,
    empirical_p_values,
    matched_random_centroid_null,
    matched_random_subspace_null,
    matched_random_subspace_null_layerwise,
    max_statistic_p_value,
    percentile_interval,
    permute_labels_within_blocks,
    topic_bootstrap_means,
    topic_cluster_means,
)


def test_cosine_values_for_identical_opposite_and_orthogonal_vectors() -> None:
    vector = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, vector * 4.0) == pytest.approx(1.0)
    assert cosine_similarity(vector, -vector) == pytest.approx(-1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 2.0]) == pytest.approx(0.0)


def test_normalized_centroid_prevents_vector_norms_from_dominating() -> None:
    centroid = normalized_centroid([[100.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(centroid, [2**-0.5, 2**-0.5])

    layers = np.array(
        [
            [[100.0, 0.0], [0.0, 2.0]],
            [[0.0, 1.0], [3.0, 0.0]],
        ]
    )
    layer_centroids = layerwise_normalized_centroid(layers)
    np.testing.assert_allclose(
        layer_centroids,
        [[2**-0.5, 2**-0.5], [2**-0.5, 2**-0.5]],
    )


def test_svd_projection_reports_known_in_span_and_out_of_span_components() -> None:
    target = np.array([3.0, 4.0, 5.0])
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = project_onto_subspace(target, basis)

    np.testing.assert_allclose(result.projected, [3.0, 4.0, 0.0])
    assert result.explained_fraction == pytest.approx(0.5)
    assert result.cosine == pytest.approx(2**-0.5)
    assert result.effective_rank == 2
    assert result.condition_number == pytest.approx(1.0)


def test_svd_projection_cosine_is_scale_invariant_for_non_unit_target() -> None:
    basis = np.array([[1.0, 0.0, 0.0]])
    unit_scale = project_onto_subspace([1.0, 1.0, 0.0], basis)
    large_scale = project_onto_subspace([100.0, 100.0, 0.0], basis)
    assert large_scale.cosine == pytest.approx(unit_scale.cosine)
    assert large_scale.cosine == pytest.approx(2**-0.5)


def test_svd_projection_discards_numerically_dependent_vectors() -> None:
    result = project_onto_subspace(
        [1.0, 1.0, 0.0],
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    assert result.effective_rank == 1
    assert result.condition_number == pytest.approx(1.0)
    assert result.explained_fraction == pytest.approx(0.5)


def test_layerwise_subspace_projection_preserves_layer_axis() -> None:
    targets = np.array([[1.0, 1.0], [1.0, 1.0]])
    emotions = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [0.0, 1.0]],
        ]
    )
    result = layerwise_subspace_projection(targets, emotions)
    np.testing.assert_allclose(result.explained_fraction, [0.5, 1.0])
    np.testing.assert_array_equal(result.effective_rank, [1, 2])
    assert result.projected.shape == targets.shape


def test_empirical_p_values_use_add_one_correction() -> None:
    null = np.array([0.1, 0.2, 0.3, 0.4])
    assert empirical_p_value(0.35, null) == pytest.approx(2 / 5)
    assert empirical_p_value(-0.35, -null, alternative="less") == pytest.approx(2 / 5)
    assert empirical_p_value(
        0.35, [-0.4, -0.2, 0.1, 0.3], alternative="two-sided"
    ) == pytest.approx(2 / 5)

    observed = np.array([0.35, 0.15])
    matrix = np.array([[0.1, 0.2], [0.4, 0.1], [0.2, 0.3], [0.3, 0.0]])
    np.testing.assert_allclose(empirical_p_values(observed, matrix), [2 / 5, 3 / 5])


def test_max_statistic_uses_joint_iteration_maxima() -> None:
    observed = np.array([0.5, 0.2])
    null = np.array([[0.1, 0.2], [0.4, 0.3], [0.6, 0.1]])
    assert max_statistic_p_value(observed, null) == pytest.approx(2 / 4)


def test_benjamini_hochberg_is_monotone_and_preserves_nan() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_topic_bootstrap_uses_topic_means_not_individual_replicas() -> None:
    values = np.array([0.0, 2.0, 10.0])
    topic_ids = ["a", "a", "b"]
    topics, means = topic_cluster_means(values, topic_ids)
    assert topics == ("a", "b")
    np.testing.assert_allclose(means, [1.0, 10.0])

    draws = TopicBootstrapDraws(
        topics=("a", "b"),
        sampled_topic_positions=np.array([[0, 1], [0, 0]], dtype=np.int64),
    )
    bootstrapped = topic_bootstrap_means(values, topic_ids, draws=draws)
    np.testing.assert_allclose(bootstrapped, [5.5, 1.0])


def test_topic_bootstrap_draws_are_reproducible() -> None:
    first = draw_topic_bootstrap([0, 0, 1, 1, 2, 2], n_iterations=8, seed=7)
    second = draw_topic_bootstrap([0, 0, 1, 1, 2, 2], n_iterations=8, seed=7)
    np.testing.assert_array_equal(first.sampled_topic_positions, second.sampled_topic_positions)


def test_label_permutations_are_reproducible_and_block_preserving() -> None:
    labels = np.array(["angry", "calm", "sad", "angry", "calm", "sad"])
    blocks = [(0, 0)] * 3 + [(1, 0)] * 3
    first = permute_labels_within_blocks(labels, blocks, n_iterations=10, seed=11)
    second = permute_labels_within_blocks(labels, blocks, n_iterations=10, seed=11)
    np.testing.assert_array_equal(first, second)
    for row in first:
        assert sorted(row[:3]) == sorted(labels[:3])
        assert sorted(row[3:]) == sorted(labels[3:])


def test_matched_random_subspace_control_is_reproducible_and_rank_matched() -> None:
    target = np.array([1.0, 2.0, 3.0, 4.0])
    first = matched_random_subspace_null(target, 1, n_iterations=500, seed=19)
    second = matched_random_subspace_null(target, 1, n_iterations=500, seed=19)
    np.testing.assert_array_equal(first, second)
    assert first.mean() == pytest.approx(0.25, abs=0.05)
    np.testing.assert_array_equal(
        matched_random_subspace_null(target, 4, n_iterations=5, seed=19),
        np.ones(5),
    )

    layerwise = matched_random_subspace_null_layerwise(
        np.stack([target, target]), [1, 4], n_iterations=20, seed=3
    )
    assert layerwise.shape == (20, 2)
    np.testing.assert_array_equal(layerwise[:, 1], np.ones(20))


def test_random_centroid_control_normalizes_emotion_vectors() -> None:
    target = np.array([1.0, 0.0])
    emotions = np.array([[100.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    first = matched_random_centroid_null(target, emotions, 2, n_iterations=20, seed=5)
    second = matched_random_centroid_null(target, emotions, 2, n_iterations=20, seed=5)
    np.testing.assert_array_equal(first, second)


def test_percentile_interval_returns_equal_tailed_bounds() -> None:
    low, high = percentile_interval(np.arange(101), confidence=0.80)
    assert float(low) == pytest.approx(10.0)
    assert float(high) == pytest.approx(90.0)
