"""Create reader-facing RQ1 figures directly from the recorded metrics.

The scientific analysis already writes detailed figures.  This script produces three
compact summaries for the repository README and writes the exact plotted rows beside
each PNG/SVG pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

EMOTIONS = (
    "excited",
    "enthusiastic",
    "joyful",
    "content",
    "calm",
    "serene",
    "angry",
    "furious",
    "anxious",
    "sad",
    "gloomy",
    "miserable",
)
NEGATIVE_EMOTIONS = ("angry", "furious", "anxious", "sad", "gloomy", "miserable")
LAYERS = (24, 32)
BLUE = "#35618D"
ORANGE = "#E07A2D"
RED = "#B23A48"
GRAY = "#667085"


def _load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "comparison_type",
        "model_source",
        "layer",
        "emotion_or_group",
        "cosine",
        "explained_fraction",
        "ci_low",
        "ci_high",
        "q_value",
        "null_p95",
        "reliability",
        "pca_status",
        "notes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metrics file is missing columns: {missing}")
    return frame


def _require_unique(frame: pd.DataFrame, keys: list[str], expected: int, label: str) -> None:
    if len(frame) != expected:
        raise ValueError(f"expected {expected} {label} rows, found {len(frame)}")
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        bad = frame.loc[duplicates, keys].to_dict(orient="records")
        raise ValueError(f"duplicate {label} rows: {bad[:4]}")


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "emotionvectors RQ1 README plotter"},
    )
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "emotionvectors RQ1 README plotter", "Date": None},
    )
    svg_text = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(f"{svg_text}\n", encoding="utf-8")
    plt.close(figure)


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    show_y_labels: bool,
) -> matplotlib.image.AxesImage:
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=11, fontweight="bold", pad=10)
    axis.set_xticks(range(len(LAYERS)), [f"Layer {layer}" for layer in LAYERS])
    axis.set_yticks(range(len(EMOTIONS)))
    axis.set_yticklabels([emotion.title() for emotion in EMOTIONS] if show_y_labels else [])
    axis.tick_params(length=0)
    axis.axhline(5.5, color="white", linewidth=2.5)
    midpoint = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value > midpoint else "#182230"
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    return image


def plot_emotion_vectors(frame: pd.DataFrame, output_dir: Path) -> None:
    reliability = frame[
        (frame["comparison_type"] == "split_reliability")
        & (frame["pca_status"] == "raw")
        & frame["layer"].isin(LAYERS)
        & frame["emotion_or_group"].isin(EMOTIONS)
        & frame["model_source"].isin(("aligned", "misaligned"))
    ].copy()
    stability = frame[
        (frame["comparison_type"] == "cross_model_stability")
        & (frame["pca_status"] == "raw")
        & frame["layer"].isin(LAYERS)
        & frame["emotion_or_group"].isin(EMOTIONS)
    ].copy()
    _require_unique(
        reliability,
        ["model_source", "emotion_or_group", "layer"],
        48,
        "split-half reliability",
    )
    _require_unique(stability, ["emotion_or_group", "layer"], 24, "cross-model stability")

    def matrix(rows: pd.DataFrame, value: str) -> np.ndarray:
        return (
            rows.assign(
                emotion_or_group=pd.Categorical(
                    rows["emotion_or_group"], categories=EMOTIONS, ordered=True
                )
            )
            .pivot(index="emotion_or_group", columns="layer", values=value)
            .reindex(index=EMOTIONS, columns=LAYERS)
            .to_numpy(dtype=float)
        )

    aligned = matrix(reliability[reliability["model_source"] == "aligned"], "reliability")
    misaligned = matrix(reliability[reliability["model_source"] == "misaligned"], "reliability")
    cross_model = matrix(stability, "reliability")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 7.0),
        gridspec_kw={"width_ratios": (1.2, 1.0, 1.0)},
    )
    figure.suptitle(
        "The same 12 emotion directions are recovered in both models",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    _annotated_heatmap(
        axes[0],
        aligned,
        title="Aligned model\nrepeatability",
        cmap="YlGnBu",
        vmin=0.50,
        vmax=0.85,
        show_y_labels=True,
    )
    _annotated_heatmap(
        axes[1],
        misaligned,
        title="Misaligned model\nrepeatability",
        cmap="YlGnBu",
        vmin=0.50,
        vmax=0.85,
        show_y_labels=False,
    )
    _annotated_heatmap(
        axes[2],
        cross_model,
        title="Aligned ↔ misaligned\ndirection similarity",
        cmap="Blues",
        vmin=0.98,
        vmax=1.00,
        show_y_labels=False,
    )
    figure.text(
        0.5,
        0.015,
        "Repeatability compares two disjoint topic halves. Direction similarity compares the full aligned and misaligned vectors. Higher is better.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.10, wspace=0.30)
    _save_figure(figure, output_dir, "readme_emotion_vectors")

    reliability_out = reliability[
        ["model_source", "emotion_or_group", "layer", "reliability"]
    ].copy()
    reliability_out.insert(0, "panel", "split_half_repeatability")
    stability_out = stability[["model_source", "emotion_or_group", "layer", "reliability"]].copy()
    stability_out.insert(0, "panel", "cross_model_direction_similarity")
    pd.concat([reliability_out, stability_out], ignore_index=True).to_csv(
        output_dir / "readme_emotion_vectors.csv", index=False, lineterminator="\n"
    )


def plot_individual_emotions(frame: pd.DataFrame, output_dir: Path) -> None:
    rows = frame[
        (frame["comparison_type"] == "individual")
        & (frame["model_source"] == "aligned")
        & (frame["pca_status"] == "raw")
        & frame["layer"].isin(LAYERS)
        & frame["emotion_or_group"].isin(NEGATIVE_EMOTIONS)
    ].copy()
    _require_unique(rows, ["emotion_or_group", "layer"], 12, "individual-emotion")

    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    y_base = np.arange(len(NEGATIVE_EMOTIONS), dtype=float)
    colors = {24: BLUE, 32: ORANGE}
    offsets = {24: -0.13, 32: 0.13}
    for layer in LAYERS:
        layer_rows = (
            rows[rows["layer"] == layer].set_index("emotion_or_group").reindex(NEGATIVE_EMOTIONS)
        )
        y = y_base + offsets[layer]
        values = layer_rows["cosine"].to_numpy(dtype=float)
        lower = values - layer_rows["ci_low"].to_numpy(dtype=float)
        upper = layer_rows["ci_high"].to_numpy(dtype=float) - values
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=7,
            capsize=3,
            linewidth=1.8,
            color=colors[layer],
            label=f"Layer {layer}",
        )
        axis.scatter(
            layer_rows["null_p95"].to_numpy(dtype=float),
            y,
            marker="|",
            s=170,
            linewidths=2,
            color=colors[layer],
            alpha=0.75,
        )
        for index, (value, q_value) in enumerate(
            zip(values, layer_rows["q_value"].to_numpy(dtype=float), strict=True)
        ):
            if q_value < 0.05:
                axis.text(
                    value + 0.014,
                    y[index],
                    "★",
                    va="center",
                    fontsize=11,
                    color=colors[layer],
                )

    axis.axvline(0, color="#98A2B3", linewidth=1)
    axis.set_yticks(y_base, [emotion.title() for emotion in NEGATIVE_EMOTIONS])
    axis.invert_yaxis()
    axis.set_xlim(-0.08, 0.37)
    axis.set_xlabel("cosine with recreated EM direction")
    axis.set_title(
        "RQ1: anger and fury are the individual emotions that align",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    axis.grid(axis="x", color="#E4E7EC", linewidth=0.8)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="|",
            linestyle="none",
            markeredgewidth=2,
            markersize=12,
            color=GRAY,
        )
    )
    labels.append("95% shuffled-label cutoff")
    handles.append(Line2D([0], [0], marker="$★$", linestyle="none", markersize=9, color=GRAY))
    labels.append("FDR q < 0.05")
    axis.legend(handles, labels, loc="lower right", frameon=False, ncol=2, fontsize=9)
    figure.text(
        0.5,
        0.015,
        "Primary comparison uses emotion vectors extracted from the aligned model; intervals bootstrap topics.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _save_figure(figure, output_dir, "readme_individual_emotions")

    rows[
        [
            "model_source",
            "emotion_or_group",
            "layer",
            "cosine",
            "ci_low",
            "ci_high",
            "q_value",
            "null_p95",
        ]
    ].sort_values(["emotion_or_group", "layer"]).to_csv(
        output_dir / "readme_individual_emotions.csv", index=False, lineterminator="\n"
    )


def plot_group_geometry(frame: pd.DataFrame, output_dir: Path) -> None:
    centroid = frame[
        (frame["comparison_type"] == "centroid")
        & (frame["model_source"] == "aligned")
        & (frame["pca_status"] == "raw")
        & (frame["emotion_or_group"] == "all_six_negative")
        & frame["layer"].isin(LAYERS)
        & frame["notes"].fillna("").str.contains("control=matched_random_centroid")
    ].copy()
    subspace = frame[
        (frame["comparison_type"] == "subspace")
        & (frame["pca_status"] == "raw")
        & (frame["emotion_or_group"] == "all_six_negative")
        & frame["layer"].isin(LAYERS)
        & frame["model_source"].isin(("aligned", "misaligned"))
    ].copy()
    _require_unique(centroid, ["layer"], 2, "matched-centroid")
    _require_unique(subspace, ["model_source", "layer"], 4, "emotion-subspace")

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    figure.suptitle(
        "RQ1: a shared negative direction fails, but the six-emotion span passes",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    centroid = centroid.set_index("layer").reindex(LAYERS)
    x = np.arange(len(LAYERS))
    bars = axes[0].bar(x, centroid["cosine"], width=0.58, color=BLUE, label="observed cosine")
    axes[0].scatter(
        x,
        centroid["null_p95"],
        marker="D",
        s=55,
        color=RED,
        zorder=3,
        label="matched-random 95%",
    )
    for bar, q_value in zip(bars, centroid["q_value"], strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"q={q_value:.3f}",
            ha="center",
            fontsize=9,
        )
    axes[0].set_xticks(x, [f"Layer {layer}" for layer in LAYERS])
    axes[0].set_ylim(0, 0.275)
    axes[0].set_ylabel("cosine with EM direction")
    axes[0].set_title(
        "One average of all six negatives\ndoes not beat its matched control",
        fontsize=11,
    )
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].grid(axis="y", color="#E4E7EC", linewidth=0.8)
    axes[0].set_axisbelow(True)

    colors = {"aligned": BLUE, "misaligned": ORANGE}
    offsets = {"aligned": -0.18, "misaligned": 0.18}
    width = 0.33
    for model in ("aligned", "misaligned"):
        model_rows = subspace[subspace["model_source"] == model].set_index("layer").reindex(LAYERS)
        positions = x + offsets[model]
        values = model_rows["explained_fraction"].to_numpy(dtype=float) * 100
        bars = axes[1].bar(
            positions,
            values,
            width=width,
            color=colors[model],
            label=f"{model.title()} emotion span",
        )
        axes[1].scatter(
            positions,
            model_rows["null_p95"].to_numpy(dtype=float) * 100,
            marker="D",
            s=42,
            color=RED,
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.25,
                f"{value:.1f}%",
                ha="center",
                fontsize=9,
            )
    axes[1].scatter([], [], marker="D", s=42, color=RED, label="random-subspace 95%")
    axes[1].set_xticks(x, [f"Layer {layer}" for layer in LAYERS])
    axes[1].set_ylim(0, 15.5)
    axes[1].set_ylabel("EM direction inside emotion span")
    axes[1].set_title(
        "The six directions together explain\nabout 10-12% of the EM direction",
        fontsize=11,
    )
    axes[1].legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    axes[1].grid(axis="y", color="#E4E7EC", linewidth=0.8)
    axes[1].set_axisbelow(True)

    figure.text(
        0.5,
        0.005,
        "Centroid asks whether all six point one way. Subspace asks whether the EM direction can be reconstructed from their distinct directions.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save_figure(figure, output_dir, "readme_group_geometry")

    centroid_out = centroid.reset_index()[
        ["model_source", "emotion_or_group", "layer", "cosine", "q_value", "null_p95"]
    ].copy()
    centroid_out.insert(0, "panel", "matched_negative_centroid")
    subspace_out = subspace[
        [
            "model_source",
            "emotion_or_group",
            "layer",
            "explained_fraction",
            "q_value",
            "null_p95",
        ]
    ].copy()
    subspace_out.insert(0, "panel", "negative_emotion_subspace")
    pd.concat([centroid_out, subspace_out], ignore_index=True).to_csv(
        output_dir / "readme_group_geometry.csv", index=False, lineterminator="\n"
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=project_root / "results/rq1_exploratory_haiku/rq1_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results/rq1_exploratory_haiku/figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _load_metrics(args.metrics)
    plot_emotion_vectors(frame, args.output_dir)
    plot_individual_emotions(frame, args.output_dir)
    plot_group_geometry(frame, args.output_dir)
    print(f"Wrote README figures and source CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
