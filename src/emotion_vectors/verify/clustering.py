"""V4: cosine geometry, k-means clusters, and a two-dimensional embedding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import (
    EmotionVectorArtifact,
    VerificationResult,
    _coerce_vector_mapping,
    _config_value,
    _plot_options,
    _resolve_output_dir,
    _save_figure,
    _write_csv,
    _write_manifest_if_supported,
    _write_report,
)

DEFAULT_SYNONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("excited", "enthusiastic"),
    ("elated", "ecstatic"),
    ("content", "calm"),
    ("calm", "serene"),
    ("angry", "furious"),
    ("terrified", "panicked"),
    ("anxious", "panicked"),
    ("sad", "gloomy"),
    ("depressed", "miserable"),
    ("guilty", "ashamed"),
)

DEFAULT_OPPOSITE_PAIRS: tuple[tuple[str, str], ...] = (
    ("joyful", "sad"),
    ("elated", "depressed"),
    ("excited", "bored"),
    ("calm", "anxious"),
    ("serene", "angry"),
    ("content", "miserable"),
    ("grateful", "jealous"),
    ("hopeful", "gloomy"),
    ("relaxed", "panicked"),
    ("proud", "ashamed"),
)


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Numerical V4 result independent of plotting."""

    emotions: tuple[str, ...]
    cosine_similarity: np.ndarray
    cluster_labels: np.ndarray
    embedding: np.ndarray
    embedding_method: str
    hierarchical_order: np.ndarray


def cosine_similarity_matrix(vectors: Any) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("vectors must be a non-empty [emotions, d_model] array")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("cosine similarity is undefined for zero vectors")
    normalized = matrix / norms
    similarity = normalized @ normalized.T
    similarity = np.clip((similarity + similarity.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _numpy_kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    seed: int,
    max_iterations: int = 300,
) -> np.ndarray:
    """Small deterministic fallback used only when scikit-learn is unavailable."""

    random = np.random.default_rng(seed)
    centers = np.empty((k, vectors.shape[1]), dtype=np.float64)
    first = int(random.integers(vectors.shape[0]))
    centers[0] = vectors[first]
    nearest = np.sum((vectors - centers[0]) ** 2, axis=1)
    for center_index in range(1, k):
        total = float(nearest.sum())
        chosen = (
            int(random.integers(vectors.shape[0]))
            if total == 0
            else int(random.choice(vectors.shape[0], p=nearest / total))
        )
        centers[center_index] = vectors[chosen]
        nearest = np.minimum(nearest, np.sum((vectors - centers[center_index]) ** 2, axis=1))
    labels = np.full(vectors.shape[0], -1, dtype=np.int64)
    for _ in range(max_iterations):
        distances = np.sum((vectors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = vectors[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
    return labels


def kmeans_labels(vectors: Any, k: int, *, seed: int = 0) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("vectors must be a non-empty rank-two array")
    if not 1 <= k <= matrix.shape[0]:
        raise ValueError("k must be between one and the number of emotions")
    try:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(matrix)
    except (ImportError, RecursionError, RuntimeError, TypeError, ValueError):
        return _numpy_kmeans(matrix, k, seed=seed)


def hierarchical_order(similarity: Any) -> np.ndarray:
    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if matrix.shape[0] < 2:
        return np.arange(matrix.shape[0])
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        return leaves_list(linkage(squareform(distance, checks=False), method="average"))
    except (ImportError, RecursionError, RuntimeError, TypeError, ValueError):
        # A deterministic spectral-style order is preferable to an unordered heatmap.
        eigenvalues, eigenvectors = np.linalg.eigh(distance)
        return np.argsort(eigenvectors[:, np.argmax(eigenvalues)])


def two_dimensional_embedding(
    vectors: Any,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, str]:
    """Use UMAP when installed, otherwise emit a deterministic PCA fallback."""

    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("vectors must be a non-empty rank-two array")
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64), "PCA fallback (one sample)"
    try:
        import umap

        neighbors = min(15, matrix.shape[0] - 1)
        embedding = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, neighbors),
            metric="cosine",
            n_jobs=1,
            random_state=seed,
        ).fit_transform(matrix)
        return np.asarray(embedding, dtype=np.float64), "UMAP"
    except (ImportError, RecursionError, RuntimeError, TypeError, ValueError):
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        components = min(2, right.shape[0])
        coordinates = centered @ right[:components].T
        if components < 2:
            coordinates = np.pad(coordinates, ((0, 0), (0, 2 - components)))
        return coordinates, "PCA fallback (UMAP unavailable)"


def compute_clustering(
    vectors: Any,
    emotions: Sequence[str],
    *,
    k: int,
    seed: int = 0,
) -> ClusteringResult:
    matrix = np.asarray(vectors, dtype=np.float64)
    if len(emotions) != matrix.shape[0]:
        raise ValueError("emotion labels do not match vector rows")
    similarity = cosine_similarity_matrix(matrix)
    labels = kmeans_labels(matrix, k, seed=seed)
    embedding, method = two_dimensional_embedding(matrix, seed=seed)
    order = hierarchical_order(similarity)
    return ClusteringResult(
        emotions=tuple(map(str, emotions)),
        cosine_similarity=similarity,
        cluster_labels=labels,
        embedding=embedding,
        embedding_method=method,
        hierarchical_order=order,
    )


def _valid_pairs(
    pairs: Sequence[tuple[str, str]], labels: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    available = set(labels)
    return tuple((left, right) for left, right in pairs if left in available and right in available)


def run_clustering(
    emotion_vectors: Mapping[str, Any] | np.ndarray | EmotionVectorArtifact,
    *,
    emotions: Sequence[str] | None = None,
    layer: int | None = None,
    k: int | None = None,
    synonym_pairs: Sequence[tuple[str, str]] | None = None,
    opposite_pairs: Sequence[tuple[str, str]] | None = None,
    output_dir: str | Path | None = None,
    config: object | None = None,
    seed: int | None = None,
    plot_formats: Sequence[str] | None = None,
    dpi: int | None = None,
) -> VerificationResult:
    """Run cosine, synonym/opposite, k-means, hierarchy, and UMAP/PCA checks."""

    vector_map = _coerce_vector_mapping(emotion_vectors, emotions, layer=layer)
    labels = tuple(vector_map)
    vectors = np.stack([vector_map[label] for label in labels])
    resolved_k = int(k if k is not None else _config_value(config, "kmeans_k", 10))
    resolved_k = min(resolved_k, len(labels))
    resolved_seed = int(seed if seed is not None else _config_value(config, "seed", 0))
    numerical = compute_clustering(vectors, labels, k=resolved_k, seed=resolved_seed)
    destination = _resolve_output_dir(output_dir, config, "V4_clustering")
    formats, resolved_dpi = _plot_options(config, plot_formats, dpi)

    index = {emotion: position for position, emotion in enumerate(labels)}
    synonyms = _valid_pairs(
        synonym_pairs if synonym_pairs is not None else DEFAULT_SYNONYM_PAIRS, labels
    )
    opposites = _valid_pairs(
        opposite_pairs if opposite_pairs is not None else DEFAULT_OPPOSITE_PAIRS, labels
    )
    pair_rows: list[dict[str, object]] = []
    for relation, pairs in (("synonym", synonyms), ("opposite", opposites)):
        for left, right in pairs:
            value = float(numerical.cosine_similarity[index[left], index[right]])
            pair_rows.append(
                {
                    "relation": relation,
                    "emotion_a": left,
                    "emotion_b": right,
                    "cosine_similarity": value,
                    "pair_passed": value > 0.0 if relation == "synonym" else value < 0.0,
                }
            )
    off_diagonal = numerical.cosine_similarity[~np.eye(len(labels), dtype=bool)]
    background_mean = float(off_diagonal.mean()) if off_diagonal.size else 0.0
    synonym_values = [
        float(numerical.cosine_similarity[index[left], index[right]]) for left, right in synonyms
    ]
    opposite_values = [
        float(numerical.cosine_similarity[index[left], index[right]]) for left, right in opposites
    ]
    synonym_mean = float(np.mean(synonym_values)) if synonym_values else float("nan")
    opposite_mean = float(np.mean(opposite_values)) if opposite_values else float("nan")
    synonyms_passed = bool(synonym_values and synonym_mean > background_mean and synonym_mean > 0.0)
    opposites_passed = bool(opposite_values and opposite_mean < 0.0)
    passed = synonyms_passed and opposites_passed

    matrix_rows = [
        {
            "emotion": emotion,
            **{
                other: float(numerical.cosine_similarity[row, column])
                for column, other in enumerate(labels)
            },
        }
        for row, emotion in enumerate(labels)
    ]
    similarity_table = _write_csv(
        destination / "cosine_similarity.csv", ("emotion", *labels), matrix_rows
    )
    long_rows = [
        {
            "emotion_a": left,
            "emotion_b": right,
            "cosine_similarity": float(numerical.cosine_similarity[row, column]),
        }
        for row, left in enumerate(labels)
        for column, right in enumerate(labels)
    ]
    long_table = _write_csv(
        destination / "cosine_similarity_long.csv",
        ("emotion_a", "emotion_b", "cosine_similarity"),
        long_rows,
    )
    cluster_rows = [
        {"emotion": emotion, "cluster": int(numerical.cluster_labels[position])}
        for position, emotion in enumerate(labels)
    ]
    cluster_table = _write_csv(
        destination / "cluster_membership.csv", ("emotion", "cluster"), cluster_rows
    )
    embedding_rows = [
        {
            "emotion": emotion,
            "x": float(numerical.embedding[position, 0]),
            "y": float(numerical.embedding[position, 1]),
            "cluster": int(numerical.cluster_labels[position]),
            "method": numerical.embedding_method,
        }
        for position, emotion in enumerate(labels)
    ]
    embedding_table = _write_csv(
        destination / "embedding.csv",
        ("emotion", "x", "y", "cluster", "method"),
        embedding_rows,
    )
    pair_table = _write_csv(
        destination / "semantic_pair_checks.csv",
        ("relation", "emotion_a", "emotion_b", "cosine_similarity", "pair_passed"),
        pair_rows,
    )
    heatmap_source = _write_csv(
        destination / "cosine_similarity_heatmap.csv",
        ("emotion", *labels),
        matrix_rows,
    )
    scatter_source = _write_csv(
        destination / "umap_scatter.csv",
        ("emotion", "x", "y", "cluster", "method"),
        embedding_rows,
    )
    tables = (
        similarity_table,
        long_table,
        cluster_table,
        embedding_table,
        pair_table,
        heatmap_source,
        scatter_source,
    )

    figures: list[Path] = []
    order = numerical.hierarchical_order
    ordered_similarity = numerical.cosine_similarity[np.ix_(order, order)]
    ordered_labels = [labels[int(position)] for position in order]
    figure, axis = plt.subplots(
        figsize=(max(7.0, 0.38 * len(labels)), max(6.0, 0.36 * len(labels)))
    )
    image = axis.imshow(ordered_similarity, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(np.arange(len(labels)), ordered_labels, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(labels)), ordered_labels)
    axis.set_title("Hierarchically ordered cosine similarity")
    figure.colorbar(image, ax=axis, label="cosine similarity")
    figures.extend(
        _save_figure(
            figure,
            destination / "cosine_similarity_heatmap",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    scatter = axis.scatter(
        numerical.embedding[:, 0],
        numerical.embedding[:, 1],
        c=numerical.cluster_labels,
        cmap="tab10",
        s=70,
    )
    for position, emotion in enumerate(labels):
        axis.annotate(
            emotion, numerical.embedding[position], xytext=(4, 4), textcoords="offset points"
        )
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.set_title(f"{numerical.embedding_method} embedding; k-means k={resolved_k}")
    figure.colorbar(scatter, ax=axis, label="cluster")
    figures.extend(
        _save_figure(
            figure,
            destination / "umap_scatter",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    report = _write_report(
        destination / "report.md",
        title="V4 — Cosine similarity and clustering",
        passed=passed,
        summary=(
            f"Synonym mean cosine: {synonym_mean:.4f}; background mean: {background_mean:.4f}; "
            f"check passed: {synonyms_passed}.",
            f"Opposite-pair mean cosine: {opposite_mean:.4f}; anti-correlation check passed: "
            f"{opposites_passed}.",
            f"K-means used k={resolved_k}; two-dimensional method: {numerical.embedding_method}.",
            "When UMAP is unavailable, the required scatter is produced with a deterministic "
            "PCA fallback and labeled as such.",
        ),
        figures=figures,
        tables=tables,
    )
    _write_manifest_if_supported(
        destination,
        config,
        "V4_clustering",
        {
            "emotions": len(labels),
            "clusters": resolved_k,
            "synonym_pairs": len(synonyms),
            "opposite_pairs": len(opposites),
            "passed": passed,
        },
    )
    return VerificationResult(
        name="V4_clustering",
        passed=passed,
        output_dir=destination,
        report=report,
        tables=tables,
        figures=tuple(figures),
        metrics={
            "n_emotions": len(labels),
            "kmeans_k": resolved_k,
            "synonym_mean_cosine": synonym_mean,
            "opposite_mean_cosine": opposite_mean,
            "embedding_method": numerical.embedding_method,
        },
    )


__all__ = [
    "DEFAULT_OPPOSITE_PAIRS",
    "DEFAULT_SYNONYM_PAIRS",
    "ClusteringResult",
    "compute_clustering",
    "cosine_similarity_matrix",
    "hierarchical_order",
    "kmeans_labels",
    "run_clustering",
    "two_dimensional_embedding",
]
