"""V5: test whether vector geometry recovers valence and arousal structure."""

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

# Approximate affective-circumplex coordinates used only when the caller does not
# supply ratings. Values encode the intended balanced-set quadrants, not results.
DEFAULT_AFFECT_RATINGS: Mapping[str, tuple[float, float]] = {
    "excited": (0.80, 0.85),
    "elated": (0.90, 0.82),
    "ecstatic": (0.95, 0.95),
    "enthusiastic": (0.82, 0.80),
    "joyful": (0.92, 0.72),
    "content": (0.72, 0.20),
    "calm": (0.65, 0.08),
    "serene": (0.78, 0.06),
    "grateful": (0.82, 0.25),
    "relaxed": (0.68, 0.12),
    "angry": (-0.78, 0.82),
    "furious": (-0.90, 0.95),
    "terrified": (-0.92, 0.96),
    "anxious": (-0.66, 0.75),
    "panicked": (-0.88, 0.98),
    "outraged": (-0.82, 0.90),
    "sad": (-0.78, 0.28),
    "depressed": (-0.94, 0.18),
    "gloomy": (-0.78, 0.22),
    "lonely": (-0.72, 0.26),
    "miserable": (-0.90, 0.32),
    "bored": (-0.48, 0.08),
    "surprised": (0.05, 0.88),
    "proud": (0.68, 0.55),
    "hopeful": (0.72, 0.50),
    "nostalgic": (0.10, 0.25),
    "guilty": (-0.62, 0.42),
    "ashamed": (-0.78, 0.48),
    "jealous": (-0.58, 0.68),
    "disgusted": (-0.76, 0.62),
}


@dataclass(frozen=True, slots=True)
class PCAStructureResult:
    """Numerical PCA and proxy correlations."""

    emotions: tuple[str, ...]
    scores: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    valence: np.ndarray
    arousal: np.ndarray
    valence_correlation: float
    arousal_factor: np.ndarray
    arousal_correlation: float
    arousal_component_weights: np.ndarray


def fit_pca(vectors: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via deterministic SVD, returning scores, components, variance ratios."""

    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("PCA needs [at least 2 emotions, non-empty d_model]")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain a non-finite value")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    scores = left * singular_values
    variances = singular_values**2 / max(1, matrix.shape[0] - 1)
    total = float(variances.sum())
    ratios = variances / total if total > 0.0 else np.zeros_like(variances)
    return scores, right, ratios


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    x = left[valid]
    y = right[valid]
    if float(x.std()) <= np.finfo(np.float64).eps or float(y.std()) <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ratings(
    emotions: Sequence[str],
    affect_ratings: Mapping[str, tuple[float, float]] | None,
    valence: Mapping[str, float] | Sequence[float] | None,
    arousal: Mapping[str, float] | Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    source = affect_ratings or DEFAULT_AFFECT_RATINGS
    if valence is None:
        valence_array = np.asarray([source.get(label, (np.nan, np.nan))[0] for label in emotions])
    elif isinstance(valence, Mapping):
        valence_array = np.asarray(
            [valence.get(label, np.nan) for label in emotions], dtype=np.float64
        )
    else:
        valence_array = np.asarray(valence, dtype=np.float64)
    if arousal is None:
        arousal_array = np.asarray([source.get(label, (np.nan, np.nan))[1] for label in emotions])
    elif isinstance(arousal, Mapping):
        arousal_array = np.asarray(
            [arousal.get(label, np.nan) for label in emotions], dtype=np.float64
        )
    else:
        arousal_array = np.asarray(arousal, dtype=np.float64)
    if valence_array.shape != (len(emotions),) or arousal_array.shape != (len(emotions),):
        raise ValueError("valence/arousal ratings must match the number of emotions")
    return valence_array, arousal_array


def compute_pca_structure(
    vectors: Any,
    emotions: Sequence[str],
    *,
    affect_ratings: Mapping[str, tuple[float, float]] | None = None,
    valence: Mapping[str, float] | Sequence[float] | None = None,
    arousal: Mapping[str, float] | Sequence[float] | None = None,
) -> PCAStructureResult:
    matrix = np.asarray(vectors, dtype=np.float64)
    if len(emotions) != matrix.shape[0]:
        raise ValueError("emotion labels do not match vector rows")
    scores, components, ratios = fit_pca(matrix)
    valence_values, arousal_values = _ratings(emotions, affect_ratings, valence, arousal)
    valence_correlation = _correlation(scores[:, 0], valence_values)

    # The paper permits arousal to rotate within the PC2/PC3 subspace. Fit the
    # least-squares direction in that subspace, only on emotions with ratings.
    arousal_stop = min(3, scores.shape[1])
    arousal_basis = scores[:, 1:arousal_stop]
    valid = np.isfinite(arousal_values)
    if arousal_basis.shape[1] and int(valid.sum()) >= 3:
        weights, *_ = np.linalg.lstsq(arousal_basis[valid], arousal_values[valid], rcond=None)
        arousal_factor = arousal_basis @ weights
        arousal_correlation = _correlation(arousal_factor, arousal_values)
    else:
        weights = np.empty((0,), dtype=np.float64)
        arousal_factor = np.full(len(emotions), np.nan)
        arousal_correlation = float("nan")
    return PCAStructureResult(
        emotions=tuple(map(str, emotions)),
        scores=scores,
        components=components,
        explained_variance_ratio=ratios,
        valence=valence_values,
        arousal=arousal_values,
        valence_correlation=valence_correlation,
        arousal_factor=arousal_factor,
        arousal_correlation=arousal_correlation,
        arousal_component_weights=weights,
    )


def run_pca_structure(
    emotion_vectors: Mapping[str, Any] | np.ndarray | EmotionVectorArtifact,
    *,
    emotions: Sequence[str] | None = None,
    layer: int | None = None,
    affect_ratings: Mapping[str, tuple[float, float]] | None = None,
    valence: Mapping[str, float] | Sequence[float] | None = None,
    arousal: Mapping[str, float] | Sequence[float] | None = None,
    minimum_absolute_correlation: float | None = None,
    output_dir: str | Path | None = None,
    config: object | None = None,
    plot_formats: Sequence[str] | None = None,
    dpi: int | None = None,
) -> VerificationResult:
    """Run affective-circumplex PCA and write all numerical/visual artifacts."""

    vector_map = _coerce_vector_mapping(emotion_vectors, emotions, layer=layer)
    labels = tuple(vector_map)
    vectors = np.stack([vector_map[label] for label in labels])
    numerical = compute_pca_structure(
        vectors,
        labels,
        affect_ratings=affect_ratings,
        valence=valence,
        arousal=arousal,
    )
    resolved_minimum_correlation = float(
        minimum_absolute_correlation
        if minimum_absolute_correlation is not None
        else _config_value(config, "v5_min_absolute_correlation", 0.5)
    )
    if not 0.0 <= resolved_minimum_correlation <= 1.0:
        raise ValueError("minimum_absolute_correlation must be in [0, 1]")
    valence_passed = bool(
        np.isfinite(numerical.valence_correlation)
        and abs(numerical.valence_correlation) >= resolved_minimum_correlation
    )
    arousal_passed = bool(
        np.isfinite(numerical.arousal_correlation)
        and abs(numerical.arousal_correlation) >= resolved_minimum_correlation
    )
    passed = valence_passed and arousal_passed
    destination = _resolve_output_dir(output_dir, config, "V5_pca_structure")
    formats, resolved_dpi = _plot_options(config, plot_formats, dpi)

    score_fields = (
        ["emotion"]
        + [f"PC{component + 1}" for component in range(numerical.scores.shape[1])]
        + ["valence", "arousal", "arousal_factor"]
    )
    score_rows: list[dict[str, object]] = []
    for row, emotion in enumerate(labels):
        values: dict[str, object] = {"emotion": emotion}
        values.update(
            {
                f"PC{component + 1}": float(numerical.scores[row, component])
                for component in range(numerical.scores.shape[1])
            }
        )
        values.update(
            {
                "valence": float(numerical.valence[row]),
                "arousal": float(numerical.arousal[row]),
                "arousal_factor": float(numerical.arousal_factor[row]),
            }
        )
        score_rows.append(values)
    score_table = _write_csv(destination / "pca_scores.csv", score_fields, score_rows)
    variance_rows = [
        {
            "component": component + 1,
            "explained_variance_ratio": float(ratio),
            "cumulative_explained_variance": float(
                numerical.explained_variance_ratio[: component + 1].sum()
            ),
        }
        for component, ratio in enumerate(numerical.explained_variance_ratio)
    ]
    variance_table = _write_csv(
        destination / "pca_variance.csv", tuple(variance_rows[0]), variance_rows
    )
    correlation_rows = (
        {
            "factor": "valence",
            "components": "PC1",
            "correlation": numerical.valence_correlation,
            "absolute_correlation": abs(numerical.valence_correlation),
            "passed": valence_passed,
        },
        {
            "factor": "arousal",
            "components": "PC2/PC3 least-squares factor",
            "correlation": numerical.arousal_correlation,
            "absolute_correlation": abs(numerical.arousal_correlation),
            "passed": arousal_passed,
        },
    )
    correlation_table = _write_csv(
        destination / "affect_correlations.csv", tuple(correlation_rows[0]), correlation_rows
    )
    circumplex_source = _write_csv(
        destination / "affective_circumplex.csv", score_fields, score_rows
    )
    scree_source = _write_csv(destination / "pca_scree.csv", tuple(variance_rows[0]), variance_rows)
    pc1_valence_rows = [
        {
            "emotion": emotion,
            "valence": float(numerical.valence[row]),
            "PC1": float(numerical.scores[row, 0]),
        }
        for row, emotion in enumerate(labels)
    ]
    pc1_valence_source = _write_csv(
        destination / "pc1_valence.csv",
        ("emotion", "valence", "PC1"),
        pc1_valence_rows,
    )
    tables = (
        score_table,
        variance_table,
        correlation_table,
        circumplex_source,
        scree_source,
        pc1_valence_source,
    )

    figures: list[Path] = []
    pc2 = numerical.scores[:, 1] if numerical.scores.shape[1] > 1 else np.zeros(len(labels))
    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    scatter = axis.scatter(
        numerical.scores[:, 0],
        pc2,
        c=numerical.valence,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        s=75,
    )
    for row, emotion in enumerate(labels):
        axis.annotate(
            emotion, (numerical.scores[row, 0], pc2[row]), xytext=(4, 4), textcoords="offset points"
        )
    axis.axhline(0.0, color="0.8", linewidth=0.8)
    axis.axvline(0.0, color="0.8", linewidth=0.8)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("Emotion-vector affective circumplex")
    figure.colorbar(scatter, ax=axis, label="valence proxy")
    figures.extend(
        _save_figure(
            figure,
            destination / "affective_circumplex",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    components = np.arange(1, len(numerical.explained_variance_ratio) + 1)
    axis.plot(components, numerical.explained_variance_ratio, marker="o")
    axis.set_xlabel("principal component")
    axis.set_ylabel("explained variance ratio")
    axis.set_title("Emotion-vector PCA scree plot")
    axis.set_xticks(components[: min(15, len(components))])
    figures.extend(
        _save_figure(
            figure,
            destination / "pca_scree",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 5.0))
    valid = np.isfinite(numerical.valence)
    axis.scatter(numerical.valence[valid], numerical.scores[valid, 0], color="#4c78a8")
    for row in np.flatnonzero(valid):
        axis.annotate(
            labels[int(row)],
            (numerical.valence[row], numerical.scores[row, 0]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    if int(valid.sum()) >= 2:
        slope, intercept = np.polyfit(numerical.valence[valid], numerical.scores[valid, 0], 1)
        line_x = np.linspace(
            float(numerical.valence[valid].min()), float(numerical.valence[valid].max()), 100
        )
        axis.plot(line_x, slope * line_x + intercept, color="#d62728", linestyle="--")
    axis.set_xlabel("valence proxy")
    axis.set_ylabel("PC1 score")
    axis.set_title(f"PC1 vs valence (r={numerical.valence_correlation:.3f})")
    figures.extend(
        _save_figure(
            figure,
            destination / "pc1_valence",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    report = _write_report(
        destination / "report.md",
        title="V5 — PCA valence/arousal structure",
        passed=passed,
        summary=(
            f"PC1/valence correlation: {numerical.valence_correlation:.4f}; "
            f"absolute-threshold check passed: {valence_passed}.",
            f"PC2/PC3 arousal-factor correlation: {numerical.arousal_correlation:.4f}; "
            f"absolute-threshold check passed: {arousal_passed}.",
            f"Required absolute correlation: {resolved_minimum_correlation:.3f}.",
            "Absolute correlations are used because PCA component signs are arbitrary.",
        ),
        figures=figures,
        tables=tables,
    )
    _write_manifest_if_supported(
        destination,
        config,
        "V5_pca_structure",
        {"emotions": len(labels), "passed": passed},
    )
    return VerificationResult(
        name="V5_pca_structure",
        passed=passed,
        output_dir=destination,
        report=report,
        tables=tables,
        figures=tuple(figures),
        metrics={
            "n_emotions": len(labels),
            "valence_correlation": numerical.valence_correlation,
            "arousal_correlation": numerical.arousal_correlation,
            "minimum_absolute_correlation": resolved_minimum_correlation,
        },
    )


__all__ = [
    "DEFAULT_AFFECT_RATINGS",
    "PCAStructureResult",
    "compute_pca_structure",
    "fit_pca",
    "run_pca_structure",
]
