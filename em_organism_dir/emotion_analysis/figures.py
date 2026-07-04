"""Auditable figures for the reduced RQ1 geometric analysis.

Every rendered PNG/SVG pair has one sibling long-form CSV containing exactly
the values used by that figure.  Plot selection is preregistered here rather
than inferred from whichever rows happen to have the largest effects.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_METRIC_COLUMNS = (
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
COMPARISON_TYPES = frozenset(
    {"individual", "centroid", "subspace", "split_reliability", "cross_model_stability"}
)
MODEL_SOURCES = frozenset({"aligned", "misaligned", "cross_model"})
ESTIMATE_NAMES = frozenset({"cosine", "explained_fraction", "reliability"})
PCA_STATUSES = frozenset({"raw", "cleaned_aligned_basis", "cleaned_misaligned_basis"})

_COMPARISON_ALIASES = {
    "individual_emotion": "individual",
    "negative_centroid": "centroid",
    "negative_subspace": "subspace",
}
_PCA_ALIASES = {
    "cleaned_aligned_neutral": "cleaned_aligned_basis",
    "cleaned_misaligned_neutral": "cleaned_misaligned_basis",
}


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    """Paths produced for one requested visualization."""

    name: str
    png_path: Path
    svg_path: Path
    csv_path: Path


def _as_bool(value: Any, *, column: str, row: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"metrics row {row}: {column} must be boolean, got {value!r}")


def _finite_when_present(series: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise").astype(float)
    present = numeric.notna()
    if not np.isfinite(numeric[present].to_numpy()).all():
        raise ValueError(f"metrics column {column!r} contains a non-finite value")
    return numeric


def validate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the analysis-to-report CSV contract."""

    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    missing = [column for column in REQUIRED_METRIC_COLUMNS if column not in metrics]
    if missing:
        raise ValueError(f"rq1_metrics.csv is missing required columns: {missing}")
    if metrics.empty:
        raise ValueError("rq1_metrics.csv contains no rows")

    frame = metrics.loc[:, REQUIRED_METRIC_COLUMNS].copy()
    for column in ("comparison_type", "model_source", "emotion_or_group", "estimate_name"):
        if frame[column].isna().any():
            raise ValueError(f"metrics column {column!r} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"metrics column {column!r} contains an empty value")
    frame["comparison_type"] = frame["comparison_type"].replace(_COMPARISON_ALIASES)
    frame["pca_status"] = frame["pca_status"].replace(_PCA_ALIASES)

    invalid = sorted(set(frame["comparison_type"]) - COMPARISON_TYPES)
    if invalid:
        raise ValueError(f"unsupported comparison_type values: {invalid}")
    invalid = sorted(set(frame["model_source"]) - MODEL_SOURCES)
    if invalid:
        raise ValueError(f"unsupported model_source values: {invalid}")
    invalid = sorted(set(frame["estimate_name"]) - ESTIMATE_NAMES)
    if invalid:
        raise ValueError(f"unsupported estimate_name values: {invalid}")
    if frame["pca_status"].isna().any():
        raise ValueError("metrics column 'pca_status' contains missing values")
    frame["pca_status"] = frame["pca_status"].astype(str).str.strip()
    invalid = sorted(set(frame["pca_status"]) - PCA_STATUSES)
    if invalid:
        raise ValueError(f"unsupported pca_status values: {invalid}")

    raw_layers = pd.to_numeric(frame["layer"], errors="raise")
    if (
        raw_layers.isna().any()
        or (raw_layers < 0).any()
        or not np.equal(raw_layers, np.floor(raw_layers)).all()
    ):
        raise ValueError("metrics layer values must be non-negative integers")
    frame["layer"] = raw_layers.astype(int)

    numeric_columns = (
        "cosine",
        "explained_fraction",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "max_stat_p_value",
        "null_p95",
        "reliability",
        "effective_rank",
        "condition_number",
    )
    for column in numeric_columns:
        frame[column] = _finite_when_present(frame[column], column=column)
    for column in ("p_value", "q_value", "max_stat_p_value"):
        present = frame[column].dropna()
        if ((present < 0.0) | (present > 1.0)).any():
            raise ValueError(f"metrics column {column!r} must be in [0, 1]")
    for column in ("cosine", "reliability"):
        present = frame[column].dropna()
        if ((present < -1.0) | (present > 1.0)).any():
            raise ValueError(f"metrics column {column!r} must be in [-1, 1]")
    present_fraction = frame["explained_fraction"].dropna()
    if ((present_fraction < 0.0) | (present_fraction > 1.0)).any():
        raise ValueError("metrics explained_fraction must be in [0, 1]")

    one_ci_missing = frame["ci_low"].isna() ^ frame["ci_high"].isna()
    if one_ci_missing.any():
        raise ValueError("ci_low and ci_high must either both be present or both be missing")
    ci_present = ~frame["ci_low"].isna()
    if (frame.loc[ci_present, "ci_low"] > frame.loc[ci_present, "ci_high"]).any():
        raise ValueError("metrics contain an interval with ci_low > ci_high")

    for row_index, row in frame.iterrows():
        estimate_name = row["estimate_name"]
        if estimate_name == "cosine" and pd.isna(row["cosine"]):
            raise ValueError(f"metrics row {row_index}: cosine estimate is missing")
        if estimate_name == "explained_fraction" and pd.isna(row["explained_fraction"]):
            raise ValueError(f"metrics row {row_index}: explained_fraction is missing")
        if estimate_name == "reliability" and pd.isna(row["reliability"]):
            raise ValueError(f"metrics row {row_index}: reliability is missing")
        if row["comparison_type"] == "subspace":
            if pd.isna(row["effective_rank"]) or row["effective_rank"] <= 0:
                raise ValueError(f"metrics row {row_index}: subspace rank must be positive")
            if pd.isna(row["condition_number"]) or row["condition_number"] < 1:
                raise ValueError(f"metrics row {row_index}: subspace condition number must be >= 1")

    frame["gate_passed"] = [
        _as_bool(value, column="gate_passed", row=index)
        for index, value in frame["gate_passed"].items()
    ]
    frame["confirmatory"] = [
        _as_bool(value, column="confirmatory", row=index)
        for index, value in frame["confirmatory"].items()
    ]
    frame["notes"] = frame["notes"].fillna("").astype(str)
    return frame.sort_values(
        ["comparison_type", "model_source", "pca_status", "emotion_or_group", "layer"],
        kind="stable",
    ).reset_index(drop=True)


def load_metrics(path: str | Path) -> pd.DataFrame:
    """Read and validate the canonical metrics CSV."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"could not read metrics CSV {source}") from exc
    return validate_metrics(frame)


def _config_sequence(config: Any, path: str, default: Sequence[Any] = ()) -> tuple[str, ...]:
    value = config
    for component in path.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                return tuple(str(item) for item in default)
            value = value[component]
        elif hasattr(value, component):
            value = getattr(value, component)
        else:
            return tuple(str(item) for item in default)
    return tuple(str(item) for item in value)


def _reported_emotions(config: Any, frame: pd.DataFrame) -> tuple[str, ...]:
    configured = _config_sequence(config, "emotions.reported_negative")
    available = tuple(
        dict.fromkeys(
            frame.loc[frame["comparison_type"] == "individual", "emotion_or_group"].tolist()
        )
    )
    selected = tuple(emotion for emotion in configured if emotion in available)
    if configured and selected != configured:
        missing = [emotion for emotion in configured if emotion not in selected]
        raise ValueError(f"individual metrics are missing reported emotions: {missing}")
    return selected or available


def _require_rows(frame: pd.DataFrame, mask: pd.Series, description: str) -> pd.DataFrame:
    rows = frame.loc[mask].copy()
    if rows.empty:
        raise ValueError(f"metrics contain no rows for {description}")
    return rows


def _assert_unique(rows: pd.DataFrame, keys: Sequence[str], description: str) -> None:
    duplicated = rows.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = rows.loc[duplicated, list(keys)].head().to_dict(orient="records")
        raise ValueError(f"duplicate {description} rows for keys {list(keys)}: {examples}")


def _collapse_equal_estimates(
    rows: pd.DataFrame,
    *,
    keys: Sequence[str],
    value_column: str,
    description: str,
) -> pd.DataFrame:
    """Collapse control-specific rows only when their plotted estimate agrees."""

    collapsed: list[pd.Series] = []
    for _key, group in rows.groupby(list(keys), sort=False, dropna=False):
        values = group[value_column].to_numpy(dtype=float)
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"duplicate {description} rows disagree on {value_column}")
        row = group.iloc[0].copy()
        row["notes"] = " | ".join(dict.fromkeys(group["notes"].astype(str)))
        collapsed.append(row)
    return pd.DataFrame(collapsed, columns=rows.columns)


def _atomic_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            data.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp"
    extension = path.suffix.lstrip(".")
    metadata: dict[str, Any]
    if extension == "svg":
        metadata = {"Creator": "model-organisms-for-EM RQ1", "Date": None}
    else:
        metadata = {"Software": "model-organisms-for-EM RQ1"}
    try:
        figure.savefig(
            temporary,
            format=extension,
            dpi=160,
            bbox_inches="tight",
            metadata=metadata,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_artifact(
    output_dir: Path,
    name: str,
    source_data: pd.DataFrame,
    draw: Callable[[], plt.Figure],
) -> FigureArtifact:
    csv_path = output_dir / f"{name}.csv"
    png_path = output_dir / f"{name}.png"
    svg_path = output_dir / f"{name}.svg"
    _atomic_csv(csv_path, source_data)
    figure = draw()
    try:
        _atomic_figure(figure, png_path)
        _atomic_figure(figure, svg_path)
    finally:
        plt.close(figure)
    return FigureArtifact(name, png_path, svg_path, csv_path)


def _draw_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
) -> plt.Figure:
    width = max(7.0, min(16.0, 0.22 * pivot.shape[1] + 4.0))
    height = max(3.2, min(14.0, 0.36 * pivot.shape[0] + 2.2))
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("Layer")
    axis.set_ylabel("")
    axis.set_xticks(np.arange(pivot.shape[1]))
    axis.set_xticklabels([str(value) for value in pivot.columns], rotation=90)
    axis.set_yticks(np.arange(pivot.shape[0]))
    axis.set_yticklabels([str(value) for value in pivot.index])
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label(colorbar_label)
    return figure


def individual_emotion_heatmap(
    metrics: pd.DataFrame, config: Any, output_dir: Path
) -> FigureArtifact:
    emotions = _reported_emotions(config, metrics)
    rows = _require_rows(
        metrics,
        (metrics["comparison_type"] == "individual")
        & (metrics["model_source"] == "aligned")
        & (metrics["estimate_name"] == "cosine")
        & (metrics["pca_status"] == "raw")
        & metrics["emotion_or_group"].isin(emotions),
        "the primary aligned/raw individual-emotion heatmap",
    )
    _assert_unique(rows, ("emotion_or_group", "layer"), "individual heatmap")
    pivot = rows.pivot(index="emotion_or_group", columns="layer", values="cosine")
    pivot = pivot.reindex(index=emotions, columns=sorted(pivot.columns))
    if pivot.isna().any().any():
        raise ValueError("individual heatmap is missing an emotion-by-layer cell")
    source = rows.sort_values(["emotion_or_group", "layer"], kind="stable")
    return _save_artifact(
        output_dir,
        "individual_emotion_layer_cosine_heatmap",
        source,
        lambda: _draw_heatmap(
            pivot,
            title="EM overlap with aligned-model negative-emotion directions",
            colorbar_label="Cosine similarity",
        ),
    )


def centroid_overlap_figure(metrics: pd.DataFrame, output_dir: Path) -> FigureArtifact:
    rows = _require_rows(
        metrics,
        (metrics["comparison_type"] == "centroid")
        & (metrics["estimate_name"] == "cosine")
        & (metrics["pca_status"] == "raw")
        & metrics["model_source"].isin(("aligned", "misaligned")),
        "raw negative-centroid overlap",
    )
    rows = _collapse_equal_estimates(
        rows,
        keys=("model_source", "emotion_or_group", "layer"),
        value_column="cosine",
        description="centroid",
    )
    source = rows.sort_values(["model_source", "emotion_or_group", "layer"], kind="stable")

    def draw() -> plt.Figure:
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        for (model_source, group), group_rows in source.groupby(
            ["model_source", "emotion_or_group"], sort=False
        ):
            group_rows = group_rows.sort_values("layer")
            axis.plot(
                group_rows["layer"],
                group_rows["cosine"],
                label=f"{model_source}: {group}",
                linewidth=1.8,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set(title="Preregistered negative-centroid overlap", xlabel="Layer")
        axis.set_ylabel("Cosine similarity")
        axis.set_ylim(-1.0, 1.0)
        axis.legend(fontsize="small", ncol=2)
        axis.grid(alpha=0.2)
        return figure

    return _save_artifact(output_dir, "centroid_overlap_across_layers", source, draw)


def subspace_explained_fraction_figure(metrics: pd.DataFrame, output_dir: Path) -> FigureArtifact:
    rows = _require_rows(
        metrics,
        (metrics["comparison_type"] == "subspace")
        & (metrics["estimate_name"] == "explained_fraction")
        & (metrics["pca_status"] == "raw")
        & metrics["model_source"].isin(("aligned", "misaligned")),
        "raw negative-subspace reconstruction",
    )
    _assert_unique(rows, ("model_source", "emotion_or_group", "layer"), "subspace")
    source = rows.sort_values(["model_source", "emotion_or_group", "layer"], kind="stable")

    def draw() -> plt.Figure:
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        for (model_source, group), group_rows in source.groupby(
            ["model_source", "emotion_or_group"], sort=False
        ):
            group_rows = group_rows.sort_values("layer")
            axis.plot(
                group_rows["layer"],
                group_rows["explained_fraction"],
                label=f"{model_source}: {group}",
                linewidth=2.0,
            )
        axis.set(
            title="Fraction of the EM direction in the negative-emotion span",
            xlabel="Layer",
            ylabel="Explained fraction",
            ylim=(0.0, 1.0),
        )
        axis.legend(fontsize="small")
        axis.grid(alpha=0.2)
        return figure

    return _save_artifact(output_dir, "negative_subspace_explained_fraction", source, draw)


def split_half_reliability_figure(metrics: pd.DataFrame, output_dir: Path) -> FigureArtifact:
    rows = _require_rows(
        metrics,
        (metrics["comparison_type"] == "split_reliability")
        & (metrics["estimate_name"] == "reliability")
        & (metrics["pca_status"] == "raw")
        & metrics["model_source"].isin(("aligned", "misaligned")),
        "emotion-vector split-half reliability",
    )
    _assert_unique(rows, ("model_source", "emotion_or_group", "layer"), "reliability")
    source = rows.sort_values(["model_source", "emotion_or_group", "layer"], kind="stable").copy()
    source["row_label"] = source["model_source"] + ": " + source["emotion_or_group"]
    pivot = source.pivot(index="row_label", columns="layer", values="reliability")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    if pivot.isna().any().any():
        raise ValueError("split-half reliability is missing a model/emotion/layer cell")
    return _save_artifact(
        output_dir,
        "split_half_reliability",
        source,
        lambda: _draw_heatmap(
            pivot,
            title="Emotion-vector split-half reliability",
            colorbar_label="Split-A / split-B cosine",
        ),
    )


def cross_model_stability_figure(metrics: pd.DataFrame, output_dir: Path) -> FigureArtifact:
    rows = _require_rows(
        metrics,
        (metrics["comparison_type"] == "cross_model_stability")
        & (metrics["model_source"] == "cross_model")
        & metrics["estimate_name"].isin(("cosine", "reliability"))
        & (metrics["pca_status"] == "raw"),
        "base-versus-misaligned emotion-vector stability",
    )
    _assert_unique(rows, ("emotion_or_group", "layer"), "cross-model stability")
    source = rows.sort_values(["emotion_or_group", "layer"], kind="stable").copy()
    source["stability"] = np.where(
        source["estimate_name"] == "reliability", source["reliability"], source["cosine"]
    )
    pivot = source.pivot(index="emotion_or_group", columns="layer", values="stability")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    if pivot.isna().any().any():
        raise ValueError("cross-model stability is missing an emotion-by-layer cell")
    return _save_artifact(
        output_dir,
        "base_vs_misaligned_stability",
        source,
        lambda: _draw_heatmap(
            pivot,
            title="Aligned-versus-misaligned emotion-vector stability",
            colorbar_label="Cross-model cosine",
        ),
    )


def raw_vs_pca_robustness_figure(
    metrics: pd.DataFrame, config: Any, output_dir: Path
) -> FigureArtifact:
    emotions = _reported_emotions(config, metrics)
    rows = _require_rows(
        metrics,
        (metrics["comparison_type"] == "individual")
        & (metrics["model_source"] == "aligned")
        & (metrics["estimate_name"] == "cosine")
        & metrics["emotion_or_group"].isin(emotions)
        & metrics["pca_status"].isin(("raw", "cleaned_aligned_basis")),
        "raw-versus-PCA-cleaned robustness",
    )
    statuses = set(rows["pca_status"])
    if statuses != {"raw", "cleaned_aligned_basis"}:
        raise ValueError("raw-versus-PCA figure requires both raw and cleaned_aligned_basis rows")
    _assert_unique(rows, ("pca_status", "emotion_or_group", "layer"), "PCA robustness")
    expected_per_cell = len(emotions)
    source = (
        rows.groupby(["pca_status", "layer"], sort=False)["cosine"]
        .agg(median_cosine="median", n_emotions="count")
        .reset_index()
        .sort_values(["pca_status", "layer"], kind="stable")
    )
    if (source["n_emotions"] != expected_per_cell).any():
        raise ValueError("raw-versus-PCA metrics do not cover every reported emotion")

    def draw() -> plt.Figure:
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        labels = {
            "raw": "Raw vectors",
            "cleaned_aligned_basis": "Jointly cleaned (aligned neutral basis)",
        }
        for status, group_rows in source.groupby("pca_status", sort=False):
            group_rows = group_rows.sort_values("layer")
            axis.plot(
                group_rows["layer"],
                group_rows["median_cosine"],
                linewidth=2.0,
                label=labels[str(status)],
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set(
            title="Raw versus jointly PCA-cleaned EM overlap",
            xlabel="Layer",
            ylabel="Median negative-emotion cosine",
            ylim=(-1.0, 1.0),
        )
        axis.legend()
        axis.grid(alpha=0.2)
        return figure

    return _save_artifact(output_dir, "raw_vs_pca_robustness", source, draw)


def create_all_figures(
    metrics: pd.DataFrame,
    config: Any,
    output_dir: str | Path,
    *,
    allow_partial: bool = False,
) -> tuple[FigureArtifact, ...]:
    """Create all six preregistered figures from validated metrics."""

    frame = validate_metrics(metrics)
    destination = Path(output_dir)
    builders: tuple[Callable[[], FigureArtifact], ...] = (
        lambda: individual_emotion_heatmap(frame, config, destination),
        lambda: centroid_overlap_figure(frame, destination),
        lambda: subspace_explained_fraction_figure(frame, destination),
        lambda: split_half_reliability_figure(frame, destination),
        lambda: cross_model_stability_figure(frame, destination),
        lambda: raw_vs_pca_robustness_figure(frame, config, destination),
    )
    artifacts: list[FigureArtifact] = []
    errors: list[str] = []
    for build in builders:
        try:
            artifacts.append(build())
        except ValueError as exc:
            if not allow_partial:
                raise
            errors.append(str(exc))
    if allow_partial and errors:
        error_path = destination / "omitted_figures.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
    return tuple(artifacts)


__all__ = [
    "COMPARISON_TYPES",
    "ESTIMATE_NAMES",
    "MODEL_SOURCES",
    "PCA_STATUSES",
    "REQUIRED_METRIC_COLUMNS",
    "FigureArtifact",
    "centroid_overlap_figure",
    "create_all_figures",
    "cross_model_stability_figure",
    "individual_emotion_heatmap",
    "load_metrics",
    "raw_vs_pca_robustness_figure",
    "split_half_reliability_figure",
    "subspace_explained_fraction_figure",
    "validate_metrics",
]
