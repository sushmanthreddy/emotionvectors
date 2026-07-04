"""In-memory orchestration of the reduced negative-emotion RQ1 analysis.

This module deliberately has no file, CLI, model-loading, or reporting
dependencies.  It accepts balanced complete-case per-example activations,
constructs all-emotion directions, applies the preregistered quality gate, and
returns typed numerical results for downstream artifact/report code.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .geometry import layerwise_subspace_projection
from .quality import ReliabilityGate, evaluate_reliability_gate
from .statistics import (
    benjamini_hochberg,
    empirical_p_values,
    max_statistic_p_value,
    percentile_interval,
)
from .vectors import (
    EmotionDirectionSet,
    NeutralPCA,
    build_emotion_directions,
    project_out_neutral_components,
)

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
METRIC_COLUMNS = (
    "comparison_type",
    "model_source",
    "layer",
    "emotion_or_group",
    "estimate_name",
    "cosine",
    "explained_fraction",
    "ci_low",
    "ci_high",
    "p_value",
    "q_value",
    "max_stat_p_value",
    "null_p95",
    "reliability",
    "pca_status",
    "effective_rank",
    "condition_number",
    "gate_passed",
    "confirmatory",
    "notes",
)


@dataclass(frozen=True, slots=True)
class IndividualAnalysis:
    """Per-emotion cosine estimates and label-permutation inference."""

    emotions: tuple[str, ...]
    cosine: FloatArray
    ci_low: FloatArray
    ci_high: FloatArray
    permutation_p_value: FloatArray
    permutation_q_value: FloatArray
    permutation_null_p95: FloatArray
    max_stat_p_value: FloatArray
    family_max_stat_p_value: float


@dataclass(frozen=True, slots=True)
class CentroidAnalysis:
    """Fixed normalized-centroid estimates and both matched controls."""

    group_names: tuple[str, ...]
    group_members: tuple[tuple[str, ...], ...]
    cosine: FloatArray
    ci_low: FloatArray
    ci_high: FloatArray
    label_permutation_p_value: FloatArray
    label_permutation_q_value: FloatArray
    label_permutation_null_p95: FloatArray
    label_max_stat_p_value: FloatArray
    label_family_max_stat_p_value: float
    matched_centroid_p_value: FloatArray
    matched_centroid_q_value: FloatArray
    matched_centroid_null_p95: FloatArray
    matched_max_stat_p_value: FloatArray
    matched_family_max_stat_p_value: float


@dataclass(frozen=True, slots=True)
class SubspaceAnalysis:
    """Six-negative emotion-span reconstruction and matched-rank control."""

    emotions: tuple[str, ...]
    explained_fraction: FloatArray
    projection_cosine: FloatArray
    effective_rank: npt.NDArray[np.int64]
    condition_number: FloatArray
    ci_low: FloatArray
    ci_high: FloatArray
    random_subspace_p_value: FloatArray
    random_subspace_q_value: FloatArray
    random_subspace_null_p95: FloatArray
    max_stat_p_value: float
    random_subspace_method: str


@dataclass(frozen=True, slots=True)
class GeometryAnalysis:
    """All geometric outputs for one emotion source and PCA status."""

    model_source: str
    pca_status: str
    all_emotions: tuple[str, ...]
    directions: Float32Array
    em_direction: Float32Array
    individual: IndividualAnalysis
    centroids: CentroidAnalysis
    subspace: SubspaceAnalysis
    inference_performed: bool


@dataclass(frozen=True, slots=True)
class PrimaryEvidence:
    """Predeclared decision criteria for the reduced six-negative study."""

    verdict: str
    positive: bool
    quality_gate_passed: bool
    confirmatory_centroid_group: str
    centroid_passing_layers: tuple[int, ...]
    subspace_passing_layers: tuple[int, ...]
    cross_model_preregistered_passes: tuple[bool, ...]
    verdict_reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe summary for workflow/report code."""

        return {
            "verdict": self.verdict,
            "positive": self.positive,
            "quality_gate_passed": self.quality_gate_passed,
            "confirmatory_centroid_group": self.confirmatory_centroid_group,
            "centroid_passing_layers": list(self.centroid_passing_layers),
            "subspace_passing_layers": list(self.subspace_passing_layers),
            "cross_model_preregistered_passes": list(self.cross_model_preregistered_passes),
            "verdict_reason": self.verdict_reason,
        }


@dataclass(frozen=True, slots=True)
class RQ1AnalysisResult:
    """Complete in-memory output of the reduced RQ1 experiment."""

    all_emotions: tuple[str, ...]
    reported_emotions: tuple[str, ...]
    aligned_directions: EmotionDirectionSet
    misaligned_directions: EmotionDirectionSet
    reliability_gate: ReliabilityGate
    cross_model_preregistered_passes: tuple[bool, ...]
    geometries: tuple[GeometryAnalysis, ...]
    inconclusive: bool
    gate_message: str
    primary_evidence: PrimaryEvidence

    def comparison(self, model_source: str, pca_status: str = "raw") -> GeometryAnalysis:
        """Select one result, failing clearly if it was not requested."""

        for geometry in self.geometries:
            if geometry.model_source == model_source and geometry.pca_status == pca_status:
                return geometry
        raise KeyError((model_source, pca_status))

    def metric_records(self) -> tuple[dict[str, Any], ...]:
        """Flatten results to the fixed 20-column workflow metrics contract."""

        records: list[dict[str, Any]] = []
        emotion_lookup = {emotion: index for index, emotion in enumerate(self.all_emotions)}
        gate_passed = not self.inconclusive

        for model_source, reliability in (
            ("aligned", self.reliability_gate.aligned_reliability),
            ("misaligned", self.reliability_gate.misaligned_reliability),
        ):
            for emotion_index, emotion in enumerate(self.all_emotions):
                for layer in range(reliability.shape[1]):
                    records.append(
                        _metric_record(
                            comparison_type="quality_gate",
                            model_source=model_source,
                            layer=layer,
                            emotion_or_group=emotion,
                            estimate_name="split_half_reliability",
                            cosine=float(reliability[emotion_index, layer]),
                            reliability=float(reliability[emotion_index, layer]),
                            pca_status="raw",
                            gate_passed=gate_passed,
                            confirmatory=False,
                            notes="topic_split_a_vs_b",
                        )
                    )
        stability = self.reliability_gate.cross_model_stability
        for emotion_index, emotion in enumerate(self.all_emotions):
            for layer in range(stability.shape[1]):
                records.append(
                    _metric_record(
                        comparison_type="cross_model_stability",
                        model_source="aligned_vs_misaligned",
                        layer=layer,
                        emotion_or_group=emotion,
                        estimate_name="cosine",
                        cosine=float(stability[emotion_index, layer]),
                        reliability=float(stability[emotion_index, layer]),
                        pca_status="raw",
                        gate_passed=gate_passed,
                        confirmatory=False,
                        notes="same_complete_case_stories",
                    )
                )

        for geometry in self.geometries:
            if geometry.model_source == "aligned":
                reliability = self.reliability_gate.aligned_reliability
                median_reliability = self.reliability_gate.aligned_median_by_layer
            else:
                reliability = self.reliability_gate.misaligned_reliability
                median_reliability = self.reliability_gate.misaligned_median_by_layer
            for row, emotion in enumerate(geometry.individual.emotions):
                emotion_index = emotion_lookup[emotion]
                for layer in range(geometry.individual.cosine.shape[1]):
                    records.append(
                        _metric_record(
                            comparison_type="individual_emotion",
                            model_source=geometry.model_source,
                            layer=layer,
                            emotion_or_group=emotion,
                            estimate_name="cosine",
                            cosine=float(geometry.individual.cosine[row, layer]),
                            ci_low=float(geometry.individual.ci_low[row, layer]),
                            ci_high=float(geometry.individual.ci_high[row, layer]),
                            p_value=float(geometry.individual.permutation_p_value[row, layer]),
                            q_value=float(geometry.individual.permutation_q_value[row, layer]),
                            max_stat_p_value=float(geometry.individual.max_stat_p_value[row]),
                            null_p95=float(geometry.individual.permutation_null_p95[row, layer]),
                            reliability=float(reliability[emotion_index, layer]),
                            pca_status=geometry.pca_status,
                            gate_passed=gate_passed,
                            confirmatory=False,
                            notes="block_preserving_label_permutation",
                        )
                    )
            for row, group_name in enumerate(geometry.centroids.group_names):
                confirmatory = (
                    geometry.model_source == "aligned"
                    and geometry.pca_status == "raw"
                    and group_name == self.primary_evidence.confirmatory_centroid_group
                )
                for layer in range(geometry.centroids.cosine.shape[1]):
                    common = {
                        "comparison_type": "negative_centroid",
                        "model_source": geometry.model_source,
                        "layer": layer,
                        "emotion_or_group": group_name,
                        "cosine": float(geometry.centroids.cosine[row, layer]),
                        "ci_low": float(geometry.centroids.ci_low[row, layer]),
                        "ci_high": float(geometry.centroids.ci_high[row, layer]),
                        "reliability": float(median_reliability[layer]),
                        "pca_status": geometry.pca_status,
                        "gate_passed": gate_passed,
                        "confirmatory": confirmatory,
                    }
                    records.append(
                        _metric_record(
                            **common,
                            estimate_name="centroid_cosine_label_permutation",
                            p_value=float(geometry.centroids.label_permutation_p_value[row, layer]),
                            q_value=float(geometry.centroids.label_permutation_q_value[row, layer]),
                            max_stat_p_value=float(geometry.centroids.label_max_stat_p_value[row]),
                            null_p95=float(
                                geometry.centroids.label_permutation_null_p95[row, layer]
                            ),
                            notes="block_preserving_label_permutation",
                        )
                    )
                    records.append(
                        _metric_record(
                            **common,
                            estimate_name="centroid_cosine_matched_random",
                            p_value=float(geometry.centroids.matched_centroid_p_value[row, layer]),
                            q_value=float(geometry.centroids.matched_centroid_q_value[row, layer]),
                            max_stat_p_value=float(
                                geometry.centroids.matched_max_stat_p_value[row]
                            ),
                            null_p95=float(
                                geometry.centroids.matched_centroid_null_p95[row, layer]
                            ),
                            notes="matched_size_random_emotion_groups",
                        )
                    )
            confirmatory_subspace = (
                geometry.model_source == "aligned" and geometry.pca_status == "raw"
            )
            for layer in range(geometry.subspace.explained_fraction.shape[0]):
                records.append(
                    _metric_record(
                        comparison_type="negative_subspace",
                        model_source=geometry.model_source,
                        layer=layer,
                        emotion_or_group="six_negative_span",
                        estimate_name="explained_fraction",
                        cosine=float(geometry.subspace.projection_cosine[layer]),
                        explained_fraction=float(geometry.subspace.explained_fraction[layer]),
                        ci_low=float(geometry.subspace.ci_low[layer]),
                        ci_high=float(geometry.subspace.ci_high[layer]),
                        p_value=float(geometry.subspace.random_subspace_p_value[layer]),
                        q_value=float(geometry.subspace.random_subspace_q_value[layer]),
                        max_stat_p_value=float(geometry.subspace.max_stat_p_value),
                        null_p95=float(geometry.subspace.random_subspace_null_p95[layer]),
                        reliability=float(median_reliability[layer]),
                        pca_status=geometry.pca_status,
                        effective_rank=int(geometry.subspace.effective_rank[layer]),
                        condition_number=float(geometry.subspace.condition_number[layer]),
                        gate_passed=gate_passed,
                        confirmatory=confirmatory_subspace,
                        notes=geometry.subspace.random_subspace_method,
                    )
                )
        assert all(tuple(record) == METRIC_COLUMNS for record in records)
        return tuple(records)


@dataclass(frozen=True, slots=True)
class _ObservedGeometry:
    individual: FloatArray
    centroids: FloatArray
    subspace_fraction: FloatArray
    subspace_cosine: FloatArray
    subspace_rank: npt.NDArray[np.int64]
    subspace_condition: FloatArray


@dataclass(frozen=True, slots=True)
class _ControlDistributions:
    bootstrap_individual: FloatArray
    bootstrap_centroids: FloatArray
    bootstrap_subspace: FloatArray
    permutation_individual: FloatArray
    permutation_centroids: FloatArray
    matched_centroids: FloatArray
    matched_subspace: FloatArray


def _metric_record(
    *,
    comparison_type: str,
    model_source: str,
    layer: int,
    emotion_or_group: str,
    estimate_name: str,
    cosine: float = float("nan"),
    explained_fraction: float = float("nan"),
    ci_low: float = float("nan"),
    ci_high: float = float("nan"),
    p_value: float = float("nan"),
    q_value: float = float("nan"),
    max_stat_p_value: float = float("nan"),
    null_p95: float = float("nan"),
    reliability: float = float("nan"),
    pca_status: str,
    effective_rank: int | float = float("nan"),
    condition_number: float = float("nan"),
    gate_passed: bool,
    confirmatory: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "comparison_type": comparison_type,
        "model_source": model_source,
        "layer": int(layer),
        "emotion_or_group": emotion_or_group,
        "estimate_name": estimate_name,
        "cosine": cosine,
        "explained_fraction": explained_fraction,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "q_value": q_value,
        "max_stat_p_value": max_stat_p_value,
        "null_p95": null_p95,
        "reliability": reliability,
        "pca_status": pca_status,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "gate_passed": gate_passed,
        "confirmatory": confirmatory,
        "notes": notes,
    }


def _positive_integer(value: int, name: str) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _hashable(value: Any) -> Hashable:
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(value)
    if not isinstance(value, Hashable):
        raise TypeError(f"block ID must be hashable, got {type(value).__name__}")
    return value


def _validate_and_stack(
    aligned_vectors_by_emotion: Mapping[str, Any],
    misaligned_vectors_by_emotion: Mapping[str, Any],
    all_emotions: Sequence[str],
    topic_ids: Sequence[int],
    block_ids: Sequence[Hashable],
    *,
    expected_examples_per_emotion: int | None,
) -> tuple[tuple[str, ...], Float32Array, Float32Array, npt.NDArray[np.int64]]:
    emotions = tuple(all_emotions)
    if len(emotions) < 2 or len(set(emotions)) != len(emotions):
        raise ValueError("all_emotions must contain at least two unique labels")
    if len(topic_ids) == 0 or len(topic_ids) != len(block_ids):
        raise ValueError("topic_ids and block_ids must have equal non-zero length")
    if expected_examples_per_emotion is not None and len(topic_ids) != int(
        expected_examples_per_emotion
    ):
        raise ValueError(
            f"complete-case analysis has {len(topic_ids)} examples per emotion, "
            f"expected {expected_examples_per_emotion}"
        )
    normalized_blocks = tuple(_hashable(value) for value in block_ids)
    if len(set(normalized_blocks)) != len(normalized_blocks):
        raise ValueError("block_ids must be unique within each emotion")

    expected_shape: tuple[int, int, int] | None = None
    aligned: list[Float32Array] = []
    misaligned: list[Float32Array] = []
    for emotion in emotions:
        if emotion not in aligned_vectors_by_emotion:
            raise KeyError(f"missing aligned activations for {emotion!r}")
        if emotion not in misaligned_vectors_by_emotion:
            raise KeyError(f"missing misaligned activations for {emotion!r}")
        aligned_values = np.asarray(aligned_vectors_by_emotion[emotion], dtype=np.float32)
        misaligned_values = np.asarray(misaligned_vectors_by_emotion[emotion], dtype=np.float32)
        if aligned_values.ndim != 3 or misaligned_values.ndim != 3:
            raise ValueError(f"{emotion}: activations must have shape [example, layer, hidden]")
        if aligned_values.shape != misaligned_values.shape:
            raise ValueError(f"{emotion}: aligned and misaligned activation shapes differ")
        if aligned_values.shape[0] != len(topic_ids):
            raise ValueError(f"{emotion}: activation count does not match complete-case cells")
        if not np.isfinite(aligned_values).all() or not np.isfinite(misaligned_values).all():
            raise ValueError(f"{emotion}: activations contain non-finite values")
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in aligned_values.shape)
        elif aligned_values.shape != expected_shape:
            raise ValueError(
                f"{emotion}: activation shape {aligned_values.shape} differs from "
                f"{expected_shape}"
            )
        aligned.append(aligned_values)
        misaligned.append(misaligned_values)
    topics = np.asarray(topic_ids, dtype=np.int64)
    return emotions, np.stack(aligned), np.stack(misaligned), topics


def _validate_hypotheses(
    all_emotions: tuple[str, ...],
    reported_emotions: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    reported = tuple(reported_emotions)
    if not reported or len(set(reported)) != len(reported):
        raise ValueError("reported_emotions must contain unique labels")
    unknown = sorted(set(reported) - set(all_emotions))
    if unknown:
        raise ValueError(f"reported emotions are absent from all_emotions: {unknown}")
    if not groups:
        raise ValueError("at least one fixed centroid group is required")
    group_names = tuple(groups)
    members = tuple(tuple(groups[name]) for name in group_names)
    for name, group in zip(group_names, members, strict=True):
        if not group or len(set(group)) != len(group):
            raise ValueError(f"group {name!r} must contain unique emotions")
        outside = sorted(set(group) - set(reported))
        if outside:
            raise ValueError(f"group {name!r} contains emotions outside reported scope: {outside}")
    return reported, group_names, members


def _unit_last(values: Any, *, strict: bool) -> Float32Array:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    invalid = ~np.isfinite(norms) | (norms <= np.finfo(np.float32).eps)
    if strict and bool(invalid.any()):
        raise ValueError("geometry contains a non-finite or zero vector")
    safe = np.where(invalid, 1.0, norms)
    unit = array / safe
    return np.where(invalid, 0.0, unit).astype(np.float32, copy=False)


def _difference_from_other_batch(means: Float32Array) -> Float32Array:
    if means.ndim != 3 or means.shape[1] < 2:
        raise ValueError("means must have shape [sample, emotion, hidden]")
    grand = means.mean(axis=1, keepdims=True, dtype=np.float32)
    factor = float(means.shape[1]) / float(means.shape[1] - 1)
    return ((means - grand) * factor).astype(np.float32, copy=False)


def _project_layer(values: Float32Array, components: Any | None) -> Float32Array:
    if components is None:
        return values
    pcs = np.asarray(components, dtype=np.float32)
    if pcs.ndim != 2 or pcs.shape[1] != values.shape[-1]:
        raise ValueError("PCA components do not match the direction hidden size")
    if pcs.shape[0] == 0:
        return values
    return (values - (values @ pcs.T) @ pcs).astype(np.float32, copy=False)


def _indices(labels: Sequence[str], ordered: tuple[str, ...]) -> npt.NDArray[np.int64]:
    lookup = {emotion: index for index, emotion in enumerate(ordered)}
    return np.asarray([lookup[label] for label in labels], dtype=np.int64)


def _batch_individual_centroids(
    directions: Float32Array,
    em_direction: Float32Array,
    reported_indices: npt.NDArray[np.int64],
    group_indices: tuple[npt.NDArray[np.int64], ...],
    *,
    strict: bool,
) -> tuple[FloatArray, FloatArray]:
    units = _unit_last(directions, strict=strict)
    em_unit = _unit_last(em_direction, strict=True)
    individual = np.einsum("brh,h->br", units[:, reported_indices, :], em_unit, optimize=True)
    centroid_values: list[FloatArray] = []
    for indices in group_indices:
        mean = units[:, indices, :].mean(axis=1, dtype=np.float32)
        centroid = _unit_last(mean, strict=strict)
        centroid_values.append(np.einsum("bh,h->b", centroid, em_unit, optimize=True))
    return (
        np.asarray(individual, dtype=np.float64),
        np.stack(centroid_values, axis=1).astype(np.float64, copy=False),
    )


def _subspace_fraction_batch(
    directions: Float32Array,
    em_direction: Float32Array,
    subspace_indices: npt.NDArray[np.int64],
    *,
    relative_tolerance: float,
) -> FloatArray:
    basis = _unit_last(directions[:, subspace_indices, :], strict=False).astype(np.float64)
    target = np.asarray(em_direction, dtype=np.float64)
    target_norm_sq = float(np.dot(target, target))
    if target_norm_sq <= np.finfo(np.float64).eps:
        raise ValueError("EM direction is zero after projection")
    gram = np.einsum("bkh,bjh->bkj", basis, basis, optimize=True)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    maximum = np.maximum(eigenvalues[:, -1:], 0.0)
    retained = eigenvalues > (relative_tolerance**2) * maximum
    projections = np.einsum("bkh,h->bk", basis, target, optimize=True)
    coordinates = np.einsum("bki,bk->bi", eigenvectors, projections, optimize=True)
    contributions = np.zeros_like(coordinates)
    np.divide(
        np.square(coordinates),
        eigenvalues,
        out=contributions,
        where=retained & (eigenvalues > np.finfo(np.float64).eps),
    )
    explained = contributions.sum(axis=1) / target_norm_sq
    return np.clip(explained, 0.0, 1.0)


def _observed_geometry(
    directions: Float32Array,
    em_direction: Float32Array,
    reported_indices: npt.NDArray[np.int64],
    group_indices: tuple[npt.NDArray[np.int64], ...],
    *,
    relative_tolerance: float,
) -> _ObservedGeometry:
    if directions.ndim != 3 or em_direction.shape != directions.shape[1:]:
        raise ValueError("directions must be [emotion, layer, hidden] and match EM")
    units = _unit_last(directions, strict=True)
    em_units = _unit_last(em_direction, strict=True)
    individual = np.einsum("rlh,lh->rl", units[reported_indices], em_units, optimize=True).astype(
        np.float64
    )

    centroids: list[FloatArray] = []
    for indices in group_indices:
        means = units[indices].mean(axis=0, dtype=np.float32)
        centroid_units = _unit_last(means, strict=True)
        centroids.append(
            np.einsum("lh,lh->l", centroid_units, em_units, optimize=True).astype(np.float64)
        )
    subspace = layerwise_subspace_projection(
        em_direction,
        directions[reported_indices],
        relative_tolerance=relative_tolerance,
        normalize_basis=True,
    )
    return _ObservedGeometry(
        individual=individual,
        centroids=np.stack(centroids, axis=0),
        subspace_fraction=subspace.explained_fraction,
        subspace_cosine=subspace.cosine,
        subspace_rank=subspace.effective_rank,
        subspace_condition=subspace.condition_number,
    )


def _permutation_assignments(
    n_iterations: int, n_blocks: int, n_emotions: int, seed: int
) -> npt.NDArray[np.int16]:
    rng = np.random.default_rng(seed)
    random_keys = rng.random((n_iterations, n_blocks, n_emotions), dtype=np.float32)
    return np.argsort(random_keys, axis=2).astype(np.int16)


def _permutation_nulls(
    values: Float32Array,
    em_direction: Float32Array,
    assignments: npt.NDArray[np.int16],
    reported_indices: npt.NDArray[np.int64],
    group_indices: tuple[npt.NDArray[np.int64], ...],
    *,
    pca: NeutralPCA | None,
    batch_size: int,
) -> tuple[FloatArray, FloatArray]:
    n_iterations, n_blocks, n_emotions = assignments.shape
    if values.shape[:2] != (n_emotions, n_blocks):
        raise ValueError("permutation assignments do not match example activations")
    individual = np.empty((n_iterations, len(reported_indices), values.shape[2]), dtype=np.float64)
    centroids = np.empty((n_iterations, len(group_indices), values.shape[2]), dtype=np.float64)
    block_positions = np.arange(n_blocks, dtype=np.int64)[None, :, None]
    for layer in range(values.shape[2]):
        source = np.transpose(values[:, :, layer, :], (1, 0, 2))
        components = None if pca is None else pca.components[layer]
        for start in range(0, n_iterations, batch_size):
            stop = min(start + batch_size, n_iterations)
            selected = source[block_positions, assignments[start:stop]]
            means = selected.mean(axis=1, dtype=np.float32)
            directions = _project_layer(_difference_from_other_batch(means), components)
            batch_individual, batch_centroids = _batch_individual_centroids(
                directions,
                em_direction[layer],
                reported_indices,
                group_indices,
                strict=False,
            )
            individual[start:stop, :, layer] = batch_individual
            centroids[start:stop, :, layer] = batch_centroids
    return individual, centroids


def _topic_sums(
    values: Float32Array, topic_ids: npt.NDArray[np.int64]
) -> tuple[Float32Array, npt.NDArray[np.int64]]:
    topics = tuple(dict.fromkeys(int(value) for value in topic_ids))
    sums = np.empty(
        (values.shape[0], len(topics), values.shape[2], values.shape[3]),
        dtype=np.float32,
    )
    counts = np.empty(len(topics), dtype=np.int64)
    for index, topic in enumerate(topics):
        mask = topic_ids == topic
        counts[index] = int(mask.sum())
        sums[:, index] = values[:, mask].sum(axis=1, dtype=np.float32)
    return sums, counts


def _bootstrap_nulls(
    values: Float32Array,
    topic_ids: npt.NDArray[np.int64],
    em_direction: Float32Array,
    topic_draws: npt.NDArray[np.int64],
    reported_indices: npt.NDArray[np.int64],
    group_indices: tuple[npt.NDArray[np.int64], ...],
    *,
    pca: NeutralPCA | None,
    relative_tolerance: float,
    batch_size: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    topic_sums, topic_counts = _topic_sums(values, topic_ids)
    if topic_draws.ndim != 2 or topic_draws.shape[1] != topic_sums.shape[1]:
        raise ValueError("topic bootstrap draws have the wrong shape")
    n_iterations = topic_draws.shape[0]
    individual = np.empty((n_iterations, len(reported_indices), values.shape[2]), dtype=np.float64)
    centroids = np.empty((n_iterations, len(group_indices), values.shape[2]), dtype=np.float64)
    subspace = np.empty((n_iterations, values.shape[2]), dtype=np.float64)
    for layer in range(values.shape[2]):
        source = topic_sums[:, :, layer, :]
        components = None if pca is None else pca.components[layer]
        for start in range(0, n_iterations, batch_size):
            stop = min(start + batch_size, n_iterations)
            draws = topic_draws[start:stop]
            selected_sums = source[:, draws, :].sum(axis=2, dtype=np.float32)
            denominators = topic_counts[draws].sum(axis=1, dtype=np.int64)
            means = np.transpose(selected_sums, (1, 0, 2)) / denominators[:, None, None]
            directions = _project_layer(
                _difference_from_other_batch(means.astype(np.float32)), components
            )
            batch_individual, batch_centroids = _batch_individual_centroids(
                directions,
                em_direction[layer],
                reported_indices,
                group_indices,
                strict=False,
            )
            individual[start:stop, :, layer] = batch_individual
            centroids[start:stop, :, layer] = batch_centroids
            subspace[start:stop, layer] = _subspace_fraction_batch(
                directions,
                em_direction[layer],
                reported_indices,
                relative_tolerance=relative_tolerance,
            )
    return individual, centroids, subspace


def _matched_centroid_selections(
    n_emotions: int,
    group_indices: tuple[npt.NDArray[np.int64], ...],
    n_iterations: int,
    seed: int,
) -> tuple[npt.NDArray[np.int64], ...]:
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(group_indices))
    selections: list[npt.NDArray[np.int64]] = []
    for indices, child in zip(group_indices, children, strict=True):
        rng = np.random.default_rng(child)
        keys = rng.random((n_iterations, n_emotions), dtype=np.float32)
        selections.append(np.argsort(keys, axis=1)[:, : len(indices)].astype(np.int64))
    return tuple(selections)


def _matched_centroid_nulls(
    directions: Float32Array,
    em_direction: Float32Array,
    selections: tuple[npt.NDArray[np.int64], ...],
    *,
    batch_size: int,
) -> FloatArray:
    n_iterations = selections[0].shape[0]
    null = np.empty((n_iterations, len(selections), directions.shape[1]), dtype=np.float64)
    units = _unit_last(directions, strict=True)
    em_units = _unit_last(em_direction, strict=True)
    for layer in range(directions.shape[1]):
        layer_units = units[:, layer, :]
        for group_index, selected_indices in enumerate(selections):
            for start in range(0, n_iterations, batch_size):
                stop = min(start + batch_size, n_iterations)
                means = layer_units[selected_indices[start:stop]].mean(axis=1, dtype=np.float32)
                centroids = _unit_last(means, strict=False)
                null[start:stop, group_index, layer] = np.einsum(
                    "bh,h->b", centroids, em_units[layer], optimize=True
                )
    return null


def _matched_subspace_beta_null(
    effective_ranks: npt.NDArray[np.int64],
    hidden_size: int,
    n_iterations: int,
    seed: int,
) -> FloatArray:
    """Exact rotational null equivalent to Gaussian-subspace QR sampling.

    For a fixed unit vector and a Haar-uniform rank-r subspace of R^d, the
    squared projection has distribution Beta(r/2, (d-r)/2).  Gaussian columns
    followed by QR generate exactly that Haar measure, so direct beta draws
    remove tens of thousands of unnecessary 5120-by-r decompositions.
    """

    rng = np.random.default_rng(seed)
    null = np.empty((n_iterations, len(effective_ranks)), dtype=np.float64)
    for layer, raw_rank in enumerate(effective_ranks):
        rank = int(raw_rank)
        if not 0 < rank <= hidden_size:
            raise ValueError(f"invalid effective subspace rank {rank} at layer {layer}")
        if rank == hidden_size:
            null[:, layer] = 1.0
        else:
            null[:, layer] = rng.beta(
                rank / 2.0,
                (hidden_size - rank) / 2.0,
                size=n_iterations,
            )
    return null


def _nan_geometry_result(
    model_source: str,
    pca_status: str,
    all_emotions: tuple[str, ...],
    reported: tuple[str, ...],
    group_names: tuple[str, ...],
    group_members: tuple[tuple[str, ...], ...],
    directions: Float32Array,
    em_direction: Float32Array,
    observed: _ObservedGeometry,
) -> GeometryAnalysis:
    individual_nan = np.full_like(observed.individual, np.nan)
    centroid_nan = np.full_like(observed.centroids, np.nan)
    subspace_nan = np.full_like(observed.subspace_fraction, np.nan)
    return GeometryAnalysis(
        model_source=model_source,
        pca_status=pca_status,
        all_emotions=all_emotions,
        directions=directions,
        em_direction=em_direction,
        individual=IndividualAnalysis(
            emotions=reported,
            cosine=observed.individual,
            ci_low=individual_nan.copy(),
            ci_high=individual_nan.copy(),
            permutation_p_value=individual_nan.copy(),
            permutation_q_value=individual_nan.copy(),
            permutation_null_p95=individual_nan.copy(),
            max_stat_p_value=np.full(len(reported), np.nan),
            family_max_stat_p_value=float("nan"),
        ),
        centroids=CentroidAnalysis(
            group_names=group_names,
            group_members=group_members,
            cosine=observed.centroids,
            ci_low=centroid_nan.copy(),
            ci_high=centroid_nan.copy(),
            label_permutation_p_value=centroid_nan.copy(),
            label_permutation_q_value=centroid_nan.copy(),
            label_permutation_null_p95=centroid_nan.copy(),
            label_max_stat_p_value=np.full(len(group_names), np.nan),
            label_family_max_stat_p_value=float("nan"),
            matched_centroid_p_value=centroid_nan.copy(),
            matched_centroid_q_value=centroid_nan.copy(),
            matched_centroid_null_p95=centroid_nan.copy(),
            matched_max_stat_p_value=np.full(len(group_names), np.nan),
            matched_family_max_stat_p_value=float("nan"),
        ),
        subspace=SubspaceAnalysis(
            emotions=reported,
            explained_fraction=observed.subspace_fraction,
            projection_cosine=observed.subspace_cosine,
            effective_rank=observed.subspace_rank,
            condition_number=observed.subspace_condition,
            ci_low=subspace_nan.copy(),
            ci_high=subspace_nan.copy(),
            random_subspace_p_value=subspace_nan.copy(),
            random_subspace_q_value=subspace_nan.copy(),
            random_subspace_null_p95=subspace_nan.copy(),
            max_stat_p_value=float("nan"),
            random_subspace_method="exact_rotational_beta",
        ),
        inference_performed=False,
    )


def _build_geometry_result(
    model_source: str,
    pca_status: str,
    all_emotions: tuple[str, ...],
    reported: tuple[str, ...],
    group_names: tuple[str, ...],
    group_members: tuple[tuple[str, ...], ...],
    directions: Float32Array,
    em_direction: Float32Array,
    observed: _ObservedGeometry,
    controls: _ControlDistributions,
) -> GeometryAnalysis:
    individual_low, individual_high = percentile_interval(controls.bootstrap_individual)
    centroid_low, centroid_high = percentile_interval(controls.bootstrap_centroids)
    subspace_low, subspace_high = percentile_interval(controls.bootstrap_subspace)

    individual_p = empirical_p_values(
        observed.individual, controls.permutation_individual, alternative="greater"
    )
    individual_max = np.asarray(
        [
            max_statistic_p_value(
                observed.individual[index],
                controls.permutation_individual[:, index, :],
                alternative="greater",
            )
            for index in range(len(reported))
        ],
        dtype=np.float64,
    )
    centroid_label_p = empirical_p_values(
        observed.centroids, controls.permutation_centroids, alternative="greater"
    )
    centroid_matched_p = empirical_p_values(
        observed.centroids, controls.matched_centroids, alternative="greater"
    )
    centroid_label_max = np.asarray(
        [
            max_statistic_p_value(
                observed.centroids[index],
                controls.permutation_centroids[:, index, :],
                alternative="greater",
            )
            for index in range(len(group_names))
        ]
    )
    centroid_matched_max = np.asarray(
        [
            max_statistic_p_value(
                observed.centroids[index],
                controls.matched_centroids[:, index, :],
                alternative="greater",
            )
            for index in range(len(group_names))
        ]
    )
    subspace_p = empirical_p_values(
        observed.subspace_fraction, controls.matched_subspace, alternative="greater"
    )
    return GeometryAnalysis(
        model_source=model_source,
        pca_status=pca_status,
        all_emotions=all_emotions,
        directions=directions,
        em_direction=em_direction,
        individual=IndividualAnalysis(
            emotions=reported,
            cosine=observed.individual,
            ci_low=individual_low,
            ci_high=individual_high,
            permutation_p_value=individual_p,
            permutation_q_value=benjamini_hochberg(individual_p),
            permutation_null_p95=np.percentile(controls.permutation_individual, 95.0, axis=0),
            max_stat_p_value=individual_max,
            family_max_stat_p_value=max_statistic_p_value(
                observed.individual,
                controls.permutation_individual,
                alternative="greater",
            ),
        ),
        centroids=CentroidAnalysis(
            group_names=group_names,
            group_members=group_members,
            cosine=observed.centroids,
            ci_low=centroid_low,
            ci_high=centroid_high,
            label_permutation_p_value=centroid_label_p,
            label_permutation_q_value=benjamini_hochberg(centroid_label_p),
            label_permutation_null_p95=np.percentile(controls.permutation_centroids, 95.0, axis=0),
            label_max_stat_p_value=centroid_label_max,
            label_family_max_stat_p_value=max_statistic_p_value(
                observed.centroids,
                controls.permutation_centroids,
                alternative="greater",
            ),
            matched_centroid_p_value=centroid_matched_p,
            matched_centroid_q_value=benjamini_hochberg(centroid_matched_p),
            matched_centroid_null_p95=np.percentile(controls.matched_centroids, 95.0, axis=0),
            matched_max_stat_p_value=centroid_matched_max,
            matched_family_max_stat_p_value=max_statistic_p_value(
                observed.centroids,
                controls.matched_centroids,
                alternative="greater",
            ),
        ),
        subspace=SubspaceAnalysis(
            emotions=reported,
            explained_fraction=observed.subspace_fraction,
            projection_cosine=observed.subspace_cosine,
            effective_rank=observed.subspace_rank,
            condition_number=observed.subspace_condition,
            ci_low=subspace_low,
            ci_high=subspace_high,
            random_subspace_p_value=subspace_p,
            random_subspace_q_value=benjamini_hochberg(subspace_p),
            random_subspace_null_p95=np.percentile(controls.matched_subspace, 95.0, axis=0),
            max_stat_p_value=max_statistic_p_value(
                observed.subspace_fraction,
                controls.matched_subspace,
                alternative="greater",
            ),
            random_subspace_method="exact_rotational_beta",
        ),
        inference_performed=True,
    )


def _adjacent_layers(layer: int, n_layers: int) -> tuple[int, ...]:
    return tuple(candidate for candidate in (layer - 1, layer + 1) if 0 <= candidate < n_layers)


def _evaluate_primary_evidence(
    *,
    gate: ReliabilityGate,
    preregistered_layers: tuple[int, ...],
    cross_model_passes: tuple[bool, ...],
    confirmatory_group_name: str,
    confirmatory_group_index: int,
    aligned_raw: GeometryAnalysis,
    misaligned_raw: GeometryAnalysis,
) -> PrimaryEvidence:
    if not gate.passed:
        return PrimaryEvidence(
            verdict="inconclusive",
            positive=False,
            quality_gate_passed=False,
            confirmatory_centroid_group=confirmatory_group_name,
            centroid_passing_layers=(),
            subspace_passing_layers=(),
            cross_model_preregistered_passes=cross_model_passes,
            verdict_reason=(
                "Both preregistered layers failed split-half reliability; "
                "confirmatory geometry cannot be evaluated."
            ),
        )

    usable_preregistered = tuple(
        layer
        for index, layer in enumerate(preregistered_layers)
        if gate.layer_passes[index] and cross_model_passes[index]
    )
    if not usable_preregistered:
        return PrimaryEvidence(
            verdict="inconclusive",
            positive=False,
            quality_gate_passed=True,
            confirmatory_centroid_group=confirmatory_group_name,
            centroid_passing_layers=(),
            subspace_passing_layers=(),
            cross_model_preregistered_passes=cross_model_passes,
            verdict_reason=(
                "Emotion vectors passed split-half reliability, but cross-model "
                "stability failed at every otherwise usable preregistered layer; "
                "the inherited-base comparison is unreliable."
            ),
        )

    centroid_passes: list[int] = []
    subspace_passes: list[int] = []
    n_layers = aligned_raw.individual.cosine.shape[1]
    for gate_index, layer in enumerate(preregistered_layers):
        reliability_ok = gate.layer_passes[gate_index]
        cross_model_ok = cross_model_passes[gate_index]
        adjacent = _adjacent_layers(layer, n_layers)

        centroid = aligned_raw.centroids
        centroid_adjacent_ok = all(
            centroid.cosine[confirmatory_group_index, neighbor] > 0.0 for neighbor in adjacent
        )
        centroid_same_model_ok = (
            misaligned_raw.centroids.cosine[confirmatory_group_index, layer] >= 0.0
        )
        centroid_ok = bool(
            reliability_ok
            and cross_model_ok
            and centroid.cosine[confirmatory_group_index, layer] > 0.0
            and centroid.label_permutation_q_value[confirmatory_group_index, layer] <= 0.05
            and centroid.matched_centroid_q_value[confirmatory_group_index, layer] <= 0.05
            and centroid.cosine[confirmatory_group_index, layer]
            > centroid.label_permutation_null_p95[confirmatory_group_index, layer]
            and centroid.cosine[confirmatory_group_index, layer]
            > centroid.matched_centroid_null_p95[confirmatory_group_index, layer]
            and centroid_adjacent_ok
            and centroid_same_model_ok
        )
        if centroid_ok:
            centroid_passes.append(layer)

        subspace = aligned_raw.subspace
        # A projection fraction has no sign.  Requiring an adjacent layer to
        # also exceed its matched null is the non-trivial analogue of adjacent
        # sign consistency for this statistic.
        subspace_adjacent_ok = bool(adjacent) and any(
            subspace.explained_fraction[neighbor] > subspace.random_subspace_null_p95[neighbor]
            for neighbor in adjacent
        )
        subspace_same_model_ok = (
            np.isfinite(misaligned_raw.subspace.explained_fraction[layer])
            and misaligned_raw.subspace.explained_fraction[layer] > 0.0
        )
        subspace_ok = bool(
            reliability_ok
            and cross_model_ok
            and subspace.random_subspace_q_value[layer] <= 0.05
            and subspace.explained_fraction[layer] > subspace.random_subspace_null_p95[layer]
            and subspace_adjacent_ok
            and subspace_same_model_ok
        )
        if subspace_ok:
            subspace_passes.append(layer)

    positive = bool(centroid_passes or subspace_passes)
    if positive:
        reason = (
            "At least one preregistered centroid or negative-subspace test passed "
            "correction, its matched null, adjacent-layer consistency, reliability, "
            "and cross/same-model consistency checks."
        )
        verdict = "positive"
    else:
        reason = (
            "Quality gates passed, but no preregistered centroid or negative-subspace "
            "test satisfied every positive-evidence criterion."
        )
        verdict = "null"
    return PrimaryEvidence(
        verdict=verdict,
        positive=positive,
        quality_gate_passed=True,
        confirmatory_centroid_group=confirmatory_group_name,
        centroid_passing_layers=tuple(centroid_passes),
        subspace_passing_layers=tuple(subspace_passes),
        cross_model_preregistered_passes=cross_model_passes,
        verdict_reason=reason,
    )


def run_rq1_analysis(
    aligned_vectors_by_emotion: Mapping[str, Any],
    misaligned_vectors_by_emotion: Mapping[str, Any],
    topic_ids: Sequence[int],
    block_ids: Sequence[Hashable],
    em_direction: Any,
    *,
    all_emotions: Sequence[str],
    reported_emotions: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    split_a_topics: Sequence[int],
    split_b_topics: Sequence[int],
    preregistered_layers: Sequence[int],
    reliability_threshold: float = 0.7,
    permutation_iterations: int = 1000,
    bootstrap_iterations: int = 1000,
    matched_control_iterations: int = 1000,
    svd_relative_tolerance: float = 1e-6,
    seed: int = 0,
    expected_examples_per_emotion: int | None = None,
    aligned_neutral_pca: NeutralPCA | None = None,
    misaligned_neutral_pca: NeutralPCA | None = None,
    control_batch_size: int = 8,
) -> RQ1AnalysisResult:
    """Run raw and optional jointly PCA-cleaned RQ1 geometry in memory.

    The input must already be the balanced complete-case design: every emotion
    has one aligned and one misaligned activation for each common ``block_id``.
    All emotions participate in difference-from-other-emotions construction and
    label permutations; only ``reported_emotions`` appear in EM comparisons.
    """

    permutations = _positive_integer(permutation_iterations, "permutation_iterations")
    bootstraps = _positive_integer(bootstrap_iterations, "bootstrap_iterations")
    matched_iterations = _positive_integer(matched_control_iterations, "matched_control_iterations")
    batch_size = _positive_integer(control_batch_size, "control_batch_size")
    if not 0.0 <= reliability_threshold <= 1.0:
        raise ValueError("reliability_threshold must be in [0, 1]")
    if not 0.0 < svd_relative_tolerance < 1.0:
        raise ValueError("svd_relative_tolerance must be in (0, 1)")

    emotions, aligned_values, misaligned_values, topics = _validate_and_stack(
        aligned_vectors_by_emotion,
        misaligned_vectors_by_emotion,
        all_emotions,
        topic_ids,
        block_ids,
        expected_examples_per_emotion=expected_examples_per_emotion,
    )
    reported, group_names, group_members = _validate_hypotheses(emotions, reported_emotions, groups)
    confirmatory_candidates = [
        index
        for index, members in enumerate(group_members)
        if len(members) == len(reported) and set(members) == set(reported)
    ]
    if len(confirmatory_candidates) != 1:
        raise ValueError(
            "groups must contain exactly one confirmatory centroid with all " "reported emotions"
        )
    confirmatory_group_index = confirmatory_candidates[0]
    confirmatory_group_name = group_names[confirmatory_group_index]
    em = np.asarray(em_direction, dtype=np.float32)
    if em.shape != aligned_values.shape[2:]:
        raise ValueError(f"EM direction must have shape {aligned_values.shape[2:]}, got {em.shape}")
    if not np.isfinite(em).all():
        raise ValueError("EM direction contains non-finite values")

    topic_mapping = {emotion: topics for emotion in emotions}
    aligned_mapping = {emotion: aligned_values[index] for index, emotion in enumerate(emotions)}
    misaligned_mapping = {
        emotion: misaligned_values[index] for index, emotion in enumerate(emotions)
    }
    aligned_directions = build_emotion_directions(
        aligned_mapping,
        topic_mapping,
        emotions,
        split_a_topics=split_a_topics,
        split_b_topics=split_b_topics,
    )
    misaligned_directions = build_emotion_directions(
        misaligned_mapping,
        topic_mapping,
        emotions,
        split_a_topics=split_a_topics,
        split_b_topics=split_b_topics,
    )
    preregistered = tuple(int(layer) for layer in preregistered_layers)
    gate = evaluate_reliability_gate(
        aligned_directions.split_a,
        aligned_directions.split_b,
        misaligned_directions.split_a,
        misaligned_directions.split_b,
        aligned_directions.full,
        misaligned_directions.full,
        preregistered_layers=preregistered,
        threshold=reliability_threshold,
    )
    cross_model_passes = tuple(
        bool(gate.cross_model_median_by_layer[layer] >= reliability_threshold)
        for layer in preregistered
    )
    inconclusive = not gate.passed
    gate_message = (
        "Both preregistered layers failed split-half extraction reliability; "
        "RQ1 is inconclusive and inferential controls were not run."
        if inconclusive
        else "At least one preregistered layer passed split-half extraction reliability."
    )

    reported_indices = _indices(reported, emotions)
    group_indices = tuple(_indices(group, emotions) for group in group_members)
    seed_children = np.random.SeedSequence(seed).spawn(4)
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_children]
    assignments: npt.NDArray[np.int16] | None = None
    topic_draws: npt.NDArray[np.int64] | None = None
    centroid_selections: tuple[npt.NDArray[np.int64], ...] | None = None
    if not inconclusive:
        assignments = _permutation_assignments(
            permutations, aligned_values.shape[1], aligned_values.shape[0], seeds[0]
        )
        n_topics = len(tuple(dict.fromkeys(int(value) for value in topics)))
        topic_draws = np.random.default_rng(seeds[1]).integers(
            0, n_topics, size=(bootstraps, n_topics), dtype=np.int64
        )
        centroid_selections = _matched_centroid_selections(
            len(emotions), group_indices, matched_iterations, seeds[2]
        )

    specifications: list[tuple[str, str, Float32Array, Float32Array, NeutralPCA | None]] = [
        ("aligned", "raw", aligned_directions.full, em, None),
        ("misaligned", "raw", misaligned_directions.full, em, None),
    ]
    if aligned_neutral_pca is not None:
        specifications.append(
            (
                "aligned",
                "cleaned_aligned_neutral",
                project_out_neutral_components(aligned_directions.full, aligned_neutral_pca),
                project_out_neutral_components(em, aligned_neutral_pca),
                aligned_neutral_pca,
            )
        )
    if misaligned_neutral_pca is not None:
        specifications.append(
            (
                "misaligned",
                "cleaned_misaligned_neutral",
                project_out_neutral_components(misaligned_directions.full, misaligned_neutral_pca),
                project_out_neutral_components(em, misaligned_neutral_pca),
                misaligned_neutral_pca,
            )
        )

    geometries: list[GeometryAnalysis] = []
    for specification_index, (
        model_source,
        pca_status,
        directions,
        comparison_em,
        pca,
    ) in enumerate(specifications):
        observed = _observed_geometry(
            directions,
            comparison_em,
            reported_indices,
            group_indices,
            relative_tolerance=svd_relative_tolerance,
        )
        if inconclusive:
            geometries.append(
                _nan_geometry_result(
                    model_source,
                    pca_status,
                    emotions,
                    reported,
                    group_names,
                    group_members,
                    directions,
                    comparison_em,
                    observed,
                )
            )
            continue

        assert assignments is not None
        assert topic_draws is not None
        assert centroid_selections is not None
        source_values = aligned_values if model_source == "aligned" else misaligned_values
        permutation_individual, permutation_centroids = _permutation_nulls(
            source_values,
            comparison_em,
            assignments,
            reported_indices,
            group_indices,
            pca=pca,
            batch_size=batch_size,
        )
        bootstrap_individual, bootstrap_centroids, bootstrap_subspace = _bootstrap_nulls(
            source_values,
            topics,
            comparison_em,
            topic_draws,
            reported_indices,
            group_indices,
            pca=pca,
            relative_tolerance=svd_relative_tolerance,
            batch_size=batch_size,
        )
        matched_centroids = _matched_centroid_nulls(
            directions,
            comparison_em,
            centroid_selections,
            batch_size=batch_size,
        )
        matched_subspace = _matched_subspace_beta_null(
            observed.subspace_rank,
            directions.shape[2],
            matched_iterations,
            seeds[3] + specification_index,
        )
        controls = _ControlDistributions(
            bootstrap_individual=bootstrap_individual,
            bootstrap_centroids=bootstrap_centroids,
            bootstrap_subspace=bootstrap_subspace,
            permutation_individual=permutation_individual,
            permutation_centroids=permutation_centroids,
            matched_centroids=matched_centroids,
            matched_subspace=matched_subspace,
        )
        geometries.append(
            _build_geometry_result(
                model_source,
                pca_status,
                emotions,
                reported,
                group_names,
                group_members,
                directions,
                comparison_em,
                observed,
                controls,
            )
        )

    aligned_raw = next(
        geometry
        for geometry in geometries
        if geometry.model_source == "aligned" and geometry.pca_status == "raw"
    )
    misaligned_raw = next(
        geometry
        for geometry in geometries
        if geometry.model_source == "misaligned" and geometry.pca_status == "raw"
    )
    primary_evidence = _evaluate_primary_evidence(
        gate=gate,
        preregistered_layers=preregistered,
        cross_model_passes=cross_model_passes,
        confirmatory_group_name=confirmatory_group_name,
        confirmatory_group_index=confirmatory_group_index,
        aligned_raw=aligned_raw,
        misaligned_raw=misaligned_raw,
    )
    return RQ1AnalysisResult(
        all_emotions=emotions,
        reported_emotions=reported,
        aligned_directions=aligned_directions,
        misaligned_directions=misaligned_directions,
        reliability_gate=gate,
        cross_model_preregistered_passes=cross_model_passes,
        geometries=tuple(geometries),
        inconclusive=inconclusive,
        gate_message=gate_message,
        primary_evidence=primary_evidence,
    )


__all__ = [
    "METRIC_COLUMNS",
    "CentroidAnalysis",
    "GeometryAnalysis",
    "IndividualAnalysis",
    "PrimaryEvidence",
    "RQ1AnalysisResult",
    "SubspaceAnalysis",
    "run_rq1_analysis",
]
