"""Reproducible statistical controls for the reduced RQ1 study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .geometry import cosine_similarity, unit_vector

Alternative = Literal["greater", "less", "two-sided"]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TopicBootstrapDraws:
    """Cluster-level bootstrap draws shared across all emotions and layers."""

    topics: tuple[Hashable, ...]
    sampled_topic_positions: npt.NDArray[np.int64]


def _validate_iterations(n_iterations: int) -> int:
    value = int(n_iterations)
    if value <= 0 or value != n_iterations:
        raise ValueError("n_iterations must be a positive integer")
    return value


def empirical_p_value(
    observed: float,
    null_distribution: Any,
    *,
    alternative: Alternative = "greater",
) -> float:
    """Compute an add-one corrected empirical p-value."""

    null = np.asarray(null_distribution, dtype=np.float64)
    if null.ndim != 1 or null.size == 0:
        raise ValueError("null_distribution must be a non-empty one-dimensional array")
    if not np.isfinite(observed) or not np.isfinite(null).all():
        raise ValueError("observed and null values must be finite")
    if alternative == "greater":
        exceedances = int(np.count_nonzero(null >= observed))
    elif alternative == "less":
        exceedances = int(np.count_nonzero(null <= observed))
    elif alternative == "two-sided":
        exceedances = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    else:
        raise ValueError(f"unsupported alternative: {alternative!r}")
    return float((exceedances + 1) / (null.size + 1))


def empirical_p_values(
    observed: Any,
    null_distributions: Any,
    *,
    alternative: Alternative = "greater",
) -> FloatArray:
    """Vectorized empirical p-values for null arrays shaped ``[iteration, ...]``."""

    values = np.asarray(observed, dtype=np.float64)
    nulls = np.asarray(null_distributions, dtype=np.float64)
    if nulls.ndim != values.ndim + 1 or nulls.shape[1:] != values.shape:
        raise ValueError(
            "null_distributions must have shape [iteration, *observed.shape], got "
            f"{nulls.shape} for observed {values.shape}"
        )
    if nulls.shape[0] == 0 or not np.isfinite(values).all() or not np.isfinite(nulls).all():
        raise ValueError("observed and null distributions must be finite and non-empty")
    if alternative == "greater":
        exceedances = np.count_nonzero(nulls >= values[None, ...], axis=0)
    elif alternative == "less":
        exceedances = np.count_nonzero(nulls <= values[None, ...], axis=0)
    elif alternative == "two-sided":
        exceedances = np.count_nonzero(np.abs(nulls) >= np.abs(values)[None, ...], axis=0)
    else:
        raise ValueError(f"unsupported alternative: {alternative!r}")
    return (exceedances + 1).astype(np.float64) / float(nulls.shape[0] + 1)


def max_statistic_p_value(
    observed: Any,
    null_distributions: Any,
    *,
    alternative: Alternative = "greater",
) -> float:
    """Compare an all-tests observed maximum with iteration-wise null maxima."""

    values = np.asarray(observed, dtype=np.float64)
    nulls = np.asarray(null_distributions, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("observed must contain at least one test")
    if nulls.ndim != values.ndim + 1 or nulls.shape[1:] != values.shape:
        raise ValueError("null_distributions must have shape [iteration, *observed.shape]")
    if not np.isfinite(values).all() or not np.isfinite(nulls).all():
        raise ValueError("observed and null values must be finite")
    axes = tuple(range(1, nulls.ndim))
    if alternative == "greater":
        observed_stat = float(np.max(values))
        null_stats = np.max(nulls, axis=axes)
        return empirical_p_value(observed_stat, null_stats, alternative="greater")
    if alternative == "less":
        observed_stat = float(np.min(values))
        null_stats = np.min(nulls, axis=axes)
        return empirical_p_value(observed_stat, null_stats, alternative="less")
    if alternative == "two-sided":
        observed_stat = float(np.max(np.abs(values)))
        null_stats = np.max(np.abs(nulls), axis=axes)
        return empirical_p_value(observed_stat, null_stats, alternative="greater")
    raise ValueError(f"unsupported alternative: {alternative!r}")


def benjamini_hochberg(p_values: Any) -> FloatArray:
    """Return Benjamini-Hochberg q-values, preserving NaN positions."""

    values = np.asarray(p_values, dtype=np.float64)
    flat = values.reshape(-1)
    valid = ~np.isnan(flat)
    finite = flat[valid]
    if np.any(~np.isfinite(finite)) or np.any((finite < 0.0) | (finite > 1.0)):
        raise ValueError("p-values must be in [0, 1] or NaN")
    adjusted = np.full(flat.shape, np.nan, dtype=np.float64)
    if finite.size == 0:
        return adjusted.reshape(values.shape)

    order = np.argsort(finite, kind="stable")
    ranked = finite[order]
    ranks = np.arange(1, finite.size + 1, dtype=np.float64)
    raw_adjusted = ranked * float(finite.size) / ranks
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty_like(monotone)
    restored[order] = monotone
    adjusted[np.flatnonzero(valid)] = restored
    return adjusted.reshape(values.shape)


def percentile_interval(
    samples: Any,
    *,
    confidence: float = 0.95,
    axis: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Return an equal-tailed percentile interval."""

    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("samples must be finite and non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = 50.0 * (1.0 - confidence)
    low, high = np.percentile(values, [tail, 100.0 - tail], axis=axis)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def _hashable_block(value: Any) -> Hashable:
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(value)
    if not isinstance(value, Hashable):
        raise TypeError(f"block identifier must be hashable, got {type(value).__name__}")
    return value


def draw_topic_bootstrap(
    topic_ids: Sequence[Hashable],
    *,
    n_iterations: int = 1000,
    seed: int = 0,
) -> TopicBootstrapDraws:
    """Draw topics with replacement, never individual same-topic replicas."""

    iterations = _validate_iterations(n_iterations)
    if len(topic_ids) == 0:
        raise ValueError("topic_ids must be non-empty")
    # Preserve first-appearance order, which matches canonical dataset order.
    topics = tuple(dict.fromkeys(_hashable_block(topic) for topic in topic_ids))
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(topics), size=(iterations, len(topics)), dtype=np.int64)
    return TopicBootstrapDraws(topics, positions)


def topic_cluster_means(
    values: Any,
    topic_ids: Sequence[Hashable],
) -> tuple[tuple[Hashable, ...], FloatArray]:
    """Collapse duplicated seeds/stories to one equally weighted mean per topic."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] != len(topic_ids) or array.shape[0] == 0:
        raise ValueError("values' first axis must match non-empty topic_ids")
    if not np.isfinite(array).all():
        raise ValueError("values contain non-finite entries")

    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, raw_topic in enumerate(topic_ids):
        groups[_hashable_block(raw_topic)].append(index)
    topics = tuple(groups)
    means = np.stack([array[indices].mean(axis=0) for indices in groups.values()], axis=0)
    return topics, means


def topic_bootstrap_means(
    values: Any,
    topic_ids: Sequence[Hashable],
    *,
    n_iterations: int = 1000,
    seed: int = 0,
    draws: TopicBootstrapDraws | None = None,
) -> FloatArray:
    """Bootstrap an overall mean with topics, rather than stories, as units."""

    topics, cluster_means = topic_cluster_means(values, topic_ids)
    selected = draws or draw_topic_bootstrap(topic_ids, n_iterations=n_iterations, seed=seed)
    if selected.topics != topics:
        raise ValueError("bootstrap draws do not match the supplied topic ordering")
    positions = selected.sampled_topic_positions
    if positions.ndim != 2 or positions.shape[1] != len(topics):
        raise ValueError("bootstrap draw matrix has an invalid shape")
    if np.any((positions < 0) | (positions >= len(topics))):
        raise ValueError("bootstrap draws contain an invalid topic position")
    return cluster_means[positions].mean(axis=1)


def permute_labels_within_blocks(
    labels: Sequence[Any],
    block_ids: Sequence[Hashable],
    *,
    n_iterations: int = 1000,
    seed: int = 0,
) -> npt.NDArray[Any]:
    """Shuffle labels independently within shared topic/story blocks."""

    iterations = _validate_iterations(n_iterations)
    if len(labels) == 0 or len(labels) != len(block_ids):
        raise ValueError("labels and block_ids must have equal non-zero length")
    label_array = np.asarray(labels)
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, block in enumerate(block_ids):
        groups[_hashable_block(block)].append(index)
    rng = np.random.default_rng(seed)
    permutations = np.empty((iterations, len(labels)), dtype=label_array.dtype)
    for iteration in range(iterations):
        row = label_array.copy()
        for indices in groups.values():
            positions = np.asarray(indices, dtype=np.int64)
            row[positions] = label_array[rng.permutation(positions)]
        permutations[iteration] = row
    return permutations


def matched_random_subspace_null(
    target: Any,
    effective_rank: int,
    *,
    n_iterations: int = 1000,
    seed: int = 0,
) -> FloatArray:
    """Sample Gaussian random subspaces and return projection fractions."""

    iterations = _validate_iterations(n_iterations)
    direction = unit_vector(target, name="target")
    rank = int(effective_rank)
    if rank <= 0 or rank != effective_rank or rank > direction.size:
        raise ValueError("effective_rank must be an integer in [1, hidden_size]")
    if rank == direction.size:
        return np.ones(iterations, dtype=np.float64)

    rng = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        gaussian = rng.standard_normal((direction.size, rank))
        orthonormal, _ = np.linalg.qr(gaussian, mode="reduced")
        coefficients = orthonormal.T @ direction
        null[iteration] = float(np.dot(coefficients, coefficients))
    return np.clip(null, 0.0, 1.0)


def matched_random_subspace_null_layerwise(
    targets: Any,
    effective_ranks: Sequence[int],
    *,
    n_iterations: int = 1000,
    seed: int = 0,
) -> FloatArray:
    """Generate a joint ``[iteration, layer]`` null for max-statistic tests."""

    iterations = _validate_iterations(n_iterations)
    directions = np.asarray(targets, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[0] != len(effective_ranks):
        raise ValueError("targets must be [layer, hidden] and have one rank per layer")
    if not np.isfinite(directions).all():
        raise ValueError("targets contain non-finite values")
    # One RNG stream keeps iteration rows joint across layers for max statistics.
    rng = np.random.default_rng(seed)
    null = np.empty((iterations, directions.shape[0]), dtype=np.float64)
    for layer, raw_rank in enumerate(effective_ranks):
        direction = unit_vector(directions[layer], name=f"target layer {layer}")
        rank = int(raw_rank)
        if rank <= 0 or rank != raw_rank or rank > direction.size:
            raise ValueError(f"effective rank at layer {layer} must be in [1, hidden_size]")
        if rank == direction.size:
            null[:, layer] = 1.0
            continue
        for iteration in range(iterations):
            gaussian = rng.standard_normal((direction.size, rank))
            orthonormal, _ = np.linalg.qr(gaussian, mode="reduced")
            coefficients = orthonormal.T @ direction
            null[iteration, layer] = float(np.dot(coefficients, coefficients))
    return np.clip(null, 0.0, 1.0)


def matched_random_centroid_null(
    target: Any,
    emotion_vectors: Any,
    group_size: int,
    *,
    n_iterations: int = 1000,
    seed: int = 0,
) -> FloatArray:
    """Compare a fixed centroid with random matched-size emotion groups."""

    iterations = _validate_iterations(n_iterations)
    vectors = np.asarray(emotion_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("emotion_vectors must have shape [emotion, hidden]")
    if not np.isfinite(vectors).all():
        raise ValueError("emotion_vectors contain non-finite values")
    size = int(group_size)
    if size <= 0 or size != group_size or size > vectors.shape[0]:
        raise ValueError("group_size must be an integer in [1, n_emotions]")
    rng = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("emotion_vectors contain a zero vector")
    unit_emotions = vectors / norms[:, None]
    for iteration in range(iterations):
        selected = rng.choice(vectors.shape[0], size=size, replace=False)
        mean = unit_emotions[selected].mean(axis=0)
        if np.linalg.norm(mean) <= np.finfo(np.float64).eps:
            # An exactly cancelling group has no directional overlap.  Keeping
            # it at zero is preferable to silently resampling the null group.
            null[iteration] = 0.0
        else:
            null[iteration] = cosine_similarity(target, mean)
    return null


__all__ = [
    "Alternative",
    "TopicBootstrapDraws",
    "benjamini_hochberg",
    "draw_topic_bootstrap",
    "empirical_p_value",
    "empirical_p_values",
    "matched_random_centroid_null",
    "matched_random_subspace_null",
    "matched_random_subspace_null_layerwise",
    "max_statistic_p_value",
    "percentile_interval",
    "permute_labels_within_blocks",
    "topic_bootstrap_means",
    "topic_cluster_means",
]
