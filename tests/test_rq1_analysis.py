from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from em_organism_dir.emotion_analysis.analysis import (
    METRIC_COLUMNS,
    _matched_subspace_beta_null,
    _permutation_assignments,
    run_rq1_analysis,
)
from em_organism_dir.emotion_analysis.vectors import NeutralPCA

ALL_EMOTIONS = ("positive_a", "positive_b", "negative_a", "negative_b")
REPORTED = ("negative_a", "negative_b")
GROUPS = {
    "negative_a_only": ("negative_a",),
    "all_negative": REPORTED,
}


def _toy_inputs(*, misaligned_reliable: bool = True, n_layers: int = 3) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    list[int],
    list[tuple[int, int]],
]:
    signals = {
        "positive_a": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "positive_b": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "negative_a": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "negative_b": np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
    }
    topic_ids = [0, 0, 0, 1, 1, 1]
    block_ids = [(topic, story) for topic in (0, 1) for story in range(3)]
    aligned: dict[str, np.ndarray] = {}
    misaligned: dict[str, np.ndarray] = {}
    for emotion in ALL_EMOTIONS:
        signal = signals[emotion]
        aligned_rows = []
        misaligned_rows = []
        for topic in topic_ids:
            aligned_rows.append(np.stack([signal] * n_layers))
            multiplier = 1.1 if misaligned_reliable or topic == 0 else -0.5
            misaligned_rows.append(np.stack([signal * multiplier] * n_layers))
        aligned[emotion] = np.asarray(aligned_rows, dtype=np.float32)
        misaligned[emotion] = np.asarray(misaligned_rows, dtype=np.float32)
    em_direction = np.stack([np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)] * n_layers)
    return aligned, misaligned, em_direction, topic_ids, block_ids


def _run(
    *,
    misaligned_reliable: bool = True,
    aligned_neutral_pca: NeutralPCA | None = None,
    misaligned_neutral_pca: NeutralPCA | None = None,
    seed: int = 0,
):
    aligned, misaligned, em_direction, topic_ids, block_ids = _toy_inputs(
        misaligned_reliable=misaligned_reliable
    )
    return run_rq1_analysis(
        aligned,
        misaligned,
        topic_ids,
        block_ids,
        em_direction,
        all_emotions=ALL_EMOTIONS,
        reported_emotions=REPORTED,
        groups=GROUPS,
        split_a_topics=(0,),
        split_b_topics=(1,),
        preregistered_layers=(1, 2),
        reliability_threshold=0.7,
        permutation_iterations=19,
        bootstrap_iterations=19,
        matched_control_iterations=19,
        svd_relative_tolerance=1e-6,
        seed=seed,
        expected_examples_per_emotion=6,
        aligned_neutral_pca=aligned_neutral_pca,
        misaligned_neutral_pca=misaligned_neutral_pca,
        control_batch_size=4,
    )


def _nuisance_pca(n_layers: int = 3) -> NeutralPCA:
    component = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    return NeutralPCA(
        components=tuple(component.copy() for _ in range(n_layers)),
        explained_variance_ratio=tuple(np.array([1.0], dtype=np.float32) for _ in range(n_layers)),
        retained_components=np.ones(n_layers, dtype=np.int64),
        neutral_means=np.zeros((n_layers, 4), dtype=np.float32),
    )


def test_analysis_uses_all_emotions_but_reports_only_negative_scope() -> None:
    result = _run()

    assert not result.inconclusive
    assert result.reliability_gate.passed
    assert result.reported_emotions == REPORTED
    assert result.aligned_directions.emotions == ALL_EMOTIONS
    assert len(result.geometries) == 2

    aligned = result.comparison("aligned")
    assert aligned.individual.emotions == REPORTED
    assert aligned.individual.cosine.shape == (2, 3)
    assert aligned.centroids.group_names == tuple(GROUPS)
    assert aligned.subspace.emotions == REPORTED
    np.testing.assert_array_equal(aligned.subspace.effective_rank, [2, 2, 2])
    assert np.isfinite(aligned.individual.permutation_p_value).all()
    assert np.isfinite(aligned.centroids.matched_centroid_p_value).all()
    assert np.isfinite(aligned.subspace.random_subspace_p_value).all()
    assert aligned.inference_performed


def test_analysis_controls_and_primary_decision_are_reproducible() -> None:
    first = _run(seed=17)
    second = _run(seed=17)
    first_aligned = first.comparison("aligned")
    second_aligned = second.comparison("aligned")

    np.testing.assert_array_equal(
        first_aligned.individual.permutation_p_value,
        second_aligned.individual.permutation_p_value,
    )
    np.testing.assert_array_equal(
        first_aligned.subspace.ci_low,
        second_aligned.subspace.ci_low,
    )
    assert first.primary_evidence.as_dict() == second.primary_evidence.as_dict()
    assert first.primary_evidence.verdict in {"positive", "null"}
    assert first.primary_evidence.quality_gate_passed


def test_failed_reliability_gate_is_explicit_and_skips_inference() -> None:
    result = _run(misaligned_reliable=False)

    assert result.inconclusive
    assert not result.reliability_gate.passed
    assert result.primary_evidence.verdict == "inconclusive"
    assert not result.primary_evidence.positive
    assert "failed split-half" in result.gate_message
    for geometry in result.geometries:
        assert not geometry.inference_performed
        assert np.isfinite(geometry.individual.cosine).all()
        assert np.isnan(geometry.individual.permutation_p_value).all()
        assert np.isnan(geometry.subspace.random_subspace_p_value).all()


def test_cross_model_instability_makes_primary_inherited_comparison_inconclusive() -> None:
    aligned, _misaligned, em_direction, topic_ids, block_ids = _toy_inputs()
    rotated = {emotion: np.roll(values, 1, axis=-1) for emotion, values in aligned.items()}
    result = run_rq1_analysis(
        aligned,
        rotated,
        topic_ids,
        block_ids,
        em_direction,
        all_emotions=ALL_EMOTIONS,
        reported_emotions=REPORTED,
        groups=GROUPS,
        split_a_topics=(0,),
        split_b_topics=(1,),
        preregistered_layers=(1, 2),
        permutation_iterations=9,
        bootstrap_iterations=9,
        matched_control_iterations=9,
        control_batch_size=3,
    )
    assert result.reliability_gate.passed
    assert not any(result.cross_model_preregistered_passes)
    assert result.primary_evidence.verdict == "inconclusive"
    assert "cross-model stability failed" in result.primary_evidence.verdict_reason


def test_optional_pca_comparisons_clean_em_and_emotions_with_same_basis() -> None:
    pca = _nuisance_pca()
    result = _run(aligned_neutral_pca=pca, misaligned_neutral_pca=pca)

    assert len(result.geometries) == 4
    raw = result.comparison("aligned", "raw")
    cleaned = result.comparison("aligned", "cleaned_aligned_neutral")
    negative_a = cleaned.individual.emotions.index("negative_a")
    assert raw.individual.cosine[negative_a, 0] == pytest.approx(2**-0.5)
    assert cleaned.individual.cosine[negative_a, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(cleaned.em_direction[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(cleaned.directions[:, :, 2], 0.0, atol=1e-6)


def test_metric_records_follow_exact_contract_and_mark_confirmatory_rows() -> None:
    result = _run()
    records = result.metric_records()

    assert records
    assert all(tuple(record) == METRIC_COLUMNS for record in records)
    individual_labels = {
        record["emotion_or_group"]
        for record in records
        if record["comparison_type"] == "individual_emotion"
    }
    assert individual_labels == set(REPORTED)
    confirmatory = [record for record in records if record["confirmatory"]]
    assert confirmatory
    assert all(record["model_source"] == "aligned" for record in confirmatory)
    assert all(record["pca_status"] == "raw" for record in confirmatory)
    centroid_estimates = {
        record["estimate_name"]
        for record in records
        if record["comparison_type"] == "negative_centroid"
    }
    assert centroid_estimates == {
        "centroid_cosine_label_permutation",
        "centroid_cosine_matched_random",
    }


def test_permutation_assignments_preserve_one_of_each_label_per_block() -> None:
    assignments = _permutation_assignments(12, 7, 4, seed=5)
    assert assignments.shape == (12, 7, 4)
    for iteration in assignments:
        for block in iteration:
            np.testing.assert_array_equal(np.sort(block), np.arange(4))


def test_exact_beta_random_subspace_null_is_rank_matched_and_reproducible() -> None:
    ranks = np.array([1, 2, 4], dtype=np.int64)
    first = _matched_subspace_beta_null(ranks, 4, 2000, seed=9)
    second = _matched_subspace_beta_null(ranks, 4, 2000, seed=9)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(axis=0), [0.25, 0.5, 1.0], atol=0.03)


def test_analysis_rejects_unbalanced_or_missing_confirmatory_design() -> None:
    aligned, misaligned, em_direction, topic_ids, block_ids = _toy_inputs()
    aligned["negative_a"] = aligned["negative_a"][:-1]
    common: Mapping[str, object] = {
        "all_emotions": ALL_EMOTIONS,
        "reported_emotions": REPORTED,
        "groups": GROUPS,
        "split_a_topics": (0,),
        "split_b_topics": (1,),
        "preregistered_layers": (1, 2),
        "permutation_iterations": 3,
        "bootstrap_iterations": 3,
        "matched_control_iterations": 3,
    }
    with pytest.raises(ValueError, match=r"shapes differ|count does not match"):
        run_rq1_analysis(
            aligned,
            misaligned,
            topic_ids,
            block_ids,
            em_direction,
            **common,
        )

    aligned, misaligned, em_direction, topic_ids, block_ids = _toy_inputs()
    with pytest.raises(ValueError, match="confirmatory centroid"):
        run_rq1_analysis(
            aligned,
            misaligned,
            topic_ids,
            block_ids,
            em_direction,
            **{**common, "groups": {"negative_a_only": ("negative_a",)}},
        )
