from __future__ import annotations

import numpy as np
import pytest

from em_organism_dir.emotion_analysis.vectors import (
    build_emotion_directions,
    difference_from_other_emotions,
    fit_neutral_pca,
    project_out_neutral_components,
)


def test_difference_from_other_emotions_matches_known_means() -> None:
    means = np.asarray([[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    result = difference_from_other_emotions(means)
    expected = np.asarray([[3.0, -1.5], [-1.5, 3.0], [-1.5, -1.5]])
    np.testing.assert_allclose(result, expected)


def test_build_emotion_directions_keeps_topic_splits_separate() -> None:
    vectors = {
        "a": np.asarray([[[2.0, 0.0]], [[4.0, 0.0]]], dtype=np.float32),
        "b": np.asarray([[[0.0, 2.0]], [[0.0, 4.0]]], dtype=np.float32),
    }
    topics = {"a": [0, 1], "b": [0, 1]}
    result = build_emotion_directions(
        vectors,
        topics,
        ("a", "b"),
        split_a_topics=(0,),
        split_b_topics=(1,),
    )
    np.testing.assert_allclose(result.full[:, 0], [[3.0, -3.0], [-3.0, 3.0]])
    np.testing.assert_allclose(result.split_a[:, 0], [[2.0, -2.0], [-2.0, 2.0]])
    np.testing.assert_allclose(result.split_b[:, 0], [[4.0, -4.0], [-4.0, 4.0]])
    assert result.counts.tolist() == [2, 2]


def test_common_pca_projection_removes_component_from_em_and_em_direction() -> None:
    neutral = np.asarray(
        [
            [[-3.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    pca = fit_neutral_pca(neutral, variance_threshold=0.5)
    emotion = np.asarray([[[2.0, 4.0, 0.0]]], dtype=np.float32)
    em = np.asarray([[5.0, 0.0, 6.0]], dtype=np.float32)
    clean_emotion = project_out_neutral_components(emotion, pca)
    clean_em = project_out_neutral_components(em, pca)
    np.testing.assert_allclose(clean_emotion[..., 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(clean_em[..., 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(clean_emotion[0, 0, 1:], [4.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(clean_em[0, 1:], [0.0, 6.0], atol=1e-6)


def test_vector_builder_rejects_missing_split() -> None:
    vectors = {
        "a": np.ones((1, 1, 2), dtype=np.float32),
        "b": np.ones((1, 1, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="both topic splits"):
        build_emotion_directions(
            vectors,
            {"a": [0], "b": [0]},
            ("a", "b"),
            split_a_topics=(0,),
            split_b_topics=(1,),
        )
