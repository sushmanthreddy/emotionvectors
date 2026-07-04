"""V1: localize emotion-vector activation within its own training stories."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import (
    EmotionVectorArtifact,
    VerificationResult,
    _as_numpy,
    _coerce_vector_mapping,
    _config_value,
    _plot_options,
    _resolve_output_dir,
    _save_figure,
    _slug,
    _write_csv,
    _write_manifest_if_supported,
    _write_report,
)


@dataclass(frozen=True, slots=True)
class LocalizationStory:
    """A story and its per-token primary-layer residual activations."""

    emotion: str
    activations: Any | None = None
    tokens: Sequence[str] | None = None
    story_id: str = ""
    text: str | None = None
    attention_mask: Any | None = None


ActivationProvider = Callable[[LocalizationStory], Any]


def project_token_activations(activations: Any, vector: Any) -> np.ndarray:
    """Project ``[tokens, d_model]`` activations on a unit emotion vector."""

    matrix = _as_numpy(activations, dtype=np.float64)
    direction = _as_numpy(vector, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"activations must be [tokens, d_model], got {matrix.shape}")
    if matrix.shape[1] != direction.size:
        raise ValueError(f"activation width {matrix.shape[1]} != vector width {direction.size}")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("the projection vector must have finite non-zero norm")
    scores = matrix @ (direction / norm)
    if not np.isfinite(scores).all():
        raise ValueError("token projection produced a non-finite score")
    return scores


def concentration_metrics(
    scores: Any,
    *,
    top_fraction: float = 0.10,
) -> dict[str, float | int]:
    """Compute top-k mass and peak-to-mean after a non-negative shift.

    Shifting by the minimum preserves token ordering, makes the mass statistic
    meaningful for signed projections, and gives a uniform sequence the expected
    top-k fraction and peak-to-mean of one.
    """

    values = _as_numpy(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one token score is required")
    if not np.isfinite(values).all():
        raise ValueError("scores contain a non-finite value")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    k = max(1, int(np.ceil(values.size * top_fraction)))
    shifted = values - float(values.min())
    epsilon = np.finfo(np.float64).eps
    if float(shifted.sum()) <= epsilon:
        top_mass = k / values.size
        peak_to_mean = 1.0
    else:
        top_mass = float(np.partition(shifted, -k)[-k:].sum() / shifted.sum())
        peak_to_mean = float(shifted.max() / max(float(shifted.mean()), epsilon))
    return {
        "n_tokens": int(values.size),
        "top_k": int(k),
        "top_k_mass_fraction": top_mass,
        "uniform_top_k_fraction": float(k / values.size),
        "peak_to_mean": peak_to_mean,
        "score_mean": float(values.mean()),
        "score_std": float(values.std()),
        "score_max": float(values.max()),
    }


def _coerce_story(value: LocalizationStory | Mapping[str, Any], index: int) -> LocalizationStory:
    if isinstance(value, LocalizationStory):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("stories must contain LocalizationStory objects or mappings")
    if "emotion" not in value:
        raise ValueError("each localization story needs an emotion label")
    return LocalizationStory(
        emotion=str(value["emotion"]),
        activations=value.get("activations", value.get("token_activations")),
        tokens=value.get("tokens"),
        story_id=str(value.get("story_id", value.get("id", index))),
        text=None if value.get("text") is None else str(value["text"]),
        attention_mask=value.get("attention_mask"),
    )


def _call_provider(
    provider: ActivationProvider, story: LocalizationStory
) -> tuple[Any, Sequence[str] | None]:
    output = provider(story)
    if isinstance(output, LocalizationStory):
        return output.activations, output.tokens
    if isinstance(output, Mapping):
        return output.get("activations", output.get("token_activations")), output.get("tokens")
    if isinstance(output, tuple) and len(output) == 2:
        return output[1], output[0]
    return output, None


def _bundle_provider(model_bundle: object, layer: int | None) -> ActivationProvider:
    for name in ("token_activations", "capture_token_activations", "get_token_activations"):
        method = getattr(model_bundle, name, None)
        if callable(method):
            try:
                parameters = inspect.signature(method).parameters.values()
                accepts_layer = any(
                    parameter.name == "layer" or parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                accepts_layer = False

            def provider(
                story: LocalizationStory,
                *,
                method: Any = method,
                accepts_layer: bool = accepts_layer,
            ) -> Any:
                if accepts_layer and layer is not None:
                    return method(story.text, layer=layer)
                return method(story.text)

            return provider
    raise TypeError(
        "model_bundle must expose token_activations/capture_token_activations, "
        "or activation_provider must be supplied"
    )


def _select_layer(activations: Any, *, layer: int | None, width: int) -> np.ndarray:
    array = _as_numpy(activations, dtype=np.float64)
    if array.ndim == 2:
        return array
    if array.ndim != 3 or layer is None:
        raise ValueError(
            "provider activations must be [tokens, d_model], or layer must be "
            "given for [layers, tokens, d_model]"
        )
    if array.shape[-1] != width:
        raise ValueError(f"activation tensor has no trailing d_model={width} dimension")
    if not 0 <= layer < array.shape[0]:
        raise IndexError(f"layer {layer} is outside [0, {array.shape[0]})")
    return array[layer]


def run_localization(
    stories: Iterable[LocalizationStory | Mapping[str, Any]],
    emotion_vectors: Mapping[str, Any] | np.ndarray | EmotionVectorArtifact,
    *,
    emotions: Sequence[str] | None = None,
    layer: int | None = None,
    activation_provider: ActivationProvider | None = None,
    model_bundle: object | None = None,
    output_dir: str | Path | None = None,
    config: object | None = None,
    top_fraction: float | None = None,
    concentration_factor: float | None = None,
    minimum_emotion_pass_rate: float | None = None,
    minimum_control_percentile: float | None = None,
    sample_stories_per_emotion: int | None = None,
    plot_formats: Sequence[str] | None = None,
    dpi: int | None = None,
) -> VerificationResult:
    """Run V1 from cached token activations or a caller-provided capture function."""

    vectors = _coerce_vector_mapping(emotion_vectors, emotions, layer=layer)
    effective_layer = (
        emotion_vectors.primary_layer
        if layer is None and isinstance(emotion_vectors, EmotionVectorArtifact)
        else layer
    )
    destination = _resolve_output_dir(output_dir, config, "V1_localization")
    formats, resolved_dpi = _plot_options(config, plot_formats, dpi)
    resolved_top_fraction = float(
        top_fraction if top_fraction is not None else _config_value(config, "v1_top_fraction", 0.10)
    )
    resolved_concentration_factor = float(
        concentration_factor
        if concentration_factor is not None
        else _config_value(config, "v1_concentration_factor", 1.5)
    )
    resolved_minimum_pass_rate = float(
        minimum_emotion_pass_rate
        if minimum_emotion_pass_rate is not None
        else _config_value(config, "v1_min_emotion_pass_rate", 0.5)
    )
    resolved_control_percentile = float(
        minimum_control_percentile
        if minimum_control_percentile is not None
        else _config_value(config, "v1_min_control_percentile", 0.75)
    )
    resolved_sample_count = int(
        sample_stories_per_emotion
        if sample_stories_per_emotion is not None
        else _config_value(config, "v1_sample_stories_per_emotion", 1)
    )
    if not 0.0 <= resolved_minimum_pass_rate <= 1.0:
        raise ValueError("minimum_emotion_pass_rate must be in [0, 1]")
    if not 0.0 <= resolved_control_percentile <= 1.0:
        raise ValueError("minimum_control_percentile must be in [0, 1]")
    if resolved_concentration_factor <= 1.0:
        raise ValueError("concentration_factor must be greater than one")
    if resolved_sample_count < 0:
        raise ValueError("sample_stories_per_emotion cannot be negative")
    if activation_provider is None and model_bundle is not None:
        activation_provider = _bundle_provider(model_bundle, effective_layer)

    token_rows: list[dict[str, object]] = []
    story_rows: list[dict[str, object]] = []
    plot_payloads: list[tuple[str, str, list[str], np.ndarray]] = []
    sampled: dict[str, int] = {}

    for index, raw_story in enumerate(stories):
        story = _coerce_story(raw_story, index)
        if story.emotion not in vectors:
            raise KeyError(f"no emotion vector exists for story label {story.emotion!r}")
        raw_activations = story.activations
        tokens = story.tokens
        if raw_activations is None:
            if activation_provider is None:
                raise ValueError(
                    f"story {story.story_id or index!r} has no activations and no provider"
                )
            if story.text is None:
                raise ValueError("an activation provider requires story text")
            raw_activations, provider_tokens = _call_provider(activation_provider, story)
            tokens = tokens if tokens is not None else provider_tokens
        matrix = _select_layer(
            raw_activations,
            layer=effective_layer,
            width=vectors[story.emotion].size,
        )
        if story.attention_mask is not None:
            mask = _as_numpy(story.attention_mask).astype(bool).reshape(-1)
            if mask.size != matrix.shape[0]:
                raise ValueError("attention_mask length does not match token activations")
            matrix = matrix[mask]
            if tokens is not None:
                tokens = [str(token) for token, keep in zip(tokens, mask, strict=True) if keep]
        scores = project_token_activations(matrix, vectors[story.emotion])
        if tokens is None:
            tokens = [f"token_{token_index}" for token_index in range(scores.size)]
        if len(tokens) != scores.size:
            raise ValueError("token labels do not match token activation rows")

        metrics = concentration_metrics(scores, top_fraction=resolved_top_fraction)
        threshold = float(metrics["uniform_top_k_fraction"]) * resolved_concentration_factor
        matching_score = max(
            float(metrics["top_k_mass_fraction"])
            / max(float(metrics["uniform_top_k_fraction"]), np.finfo(np.float64).eps),
            float(metrics["peak_to_mean"]),
        )
        control_scores: list[float] = []
        for control_emotion, control_vector in vectors.items():
            if control_emotion == story.emotion:
                continue
            control_metrics = concentration_metrics(
                project_token_activations(matrix, control_vector),
                top_fraction=resolved_top_fraction,
            )
            control_scores.append(
                max(
                    float(control_metrics["top_k_mass_fraction"])
                    / max(
                        float(control_metrics["uniform_top_k_fraction"]),
                        np.finfo(np.float64).eps,
                    ),
                    float(control_metrics["peak_to_mean"]),
                )
            )
        control_percentile = (
            float(
                np.mean(
                    [
                        (
                            1.0
                            if matching_score > control
                            else 0.5 if matching_score == control else 0.0
                        )
                        for control in control_scores
                    ]
                )
            )
            if control_scores
            else 0.0
        )
        absolute_concentration_passed = bool(
            float(metrics["top_k_mass_fraction"]) >= threshold
            or float(metrics["peak_to_mean"]) >= resolved_concentration_factor
        )
        story_passed = bool(
            absolute_concentration_passed and control_percentile >= resolved_control_percentile
        )
        story_id = story.story_id or str(index)
        story_rows.append(
            {
                "emotion": story.emotion,
                "story_id": story_id,
                **metrics,
                "concentration_threshold": threshold,
                "matching_concentration_score": matching_score,
                "nonmatching_control_count": len(control_scores),
                "nonmatching_control_mean": (
                    float(np.mean(control_scores)) if control_scores else float("nan")
                ),
                "matching_control_percentile": control_percentile,
                "minimum_control_percentile": resolved_control_percentile,
                "absolute_concentration_passed": absolute_concentration_passed,
                "passed": story_passed,
            }
        )
        k = int(metrics["top_k"])
        top_indices = set(np.argpartition(scores, -k)[-k:].tolist())
        is_heatmap_sample = sampled.get(story.emotion, 0) < resolved_sample_count
        if is_heatmap_sample:
            for token_index, (token, score) in enumerate(zip(tokens, scores, strict=True)):
                token_rows.append(
                    {
                        "emotion": story.emotion,
                        "story_id": story_id,
                        "token_index": token_index,
                        "token": str(token),
                        "projection": float(score),
                        "top_k": token_index in top_indices,
                    }
                )
            plot_payloads.append((story.emotion, story_id, list(map(str, tokens)), scores))
            sampled[story.emotion] = sampled.get(story.emotion, 0) + 1

    if not story_rows:
        raise ValueError("localization requires at least one story")

    emotion_rows: list[dict[str, object]] = []
    for emotion in vectors:
        group = [row for row in story_rows if row["emotion"] == emotion]
        if not group:
            continue
        emotion_rows.append(
            {
                "emotion": emotion,
                "n_stories": len(group),
                "mean_top_k_mass_fraction": float(
                    np.mean([float(row["top_k_mass_fraction"]) for row in group])
                ),
                "mean_uniform_top_k_fraction": float(
                    np.mean([float(row["uniform_top_k_fraction"]) for row in group])
                ),
                "mean_peak_to_mean": float(np.mean([float(row["peak_to_mean"]) for row in group])),
                "mean_matching_control_percentile": float(
                    np.mean([float(row["matching_control_percentile"]) for row in group])
                ),
                "story_pass_rate": float(np.mean([bool(row["passed"]) for row in group])),
                "passed": float(np.mean([bool(row["passed"]) for row in group])) >= 0.5,
            }
        )
    emotion_pass_rate = float(np.mean([bool(row["passed"]) for row in emotion_rows]))
    passed = emotion_pass_rate >= resolved_minimum_pass_rate

    story_table = _write_csv(
        destination / "story_concentration.csv", tuple(story_rows[0]), story_rows
    )
    emotion_table = _write_csv(
        destination / "emotion_concentration.csv", tuple(emotion_rows[0]), emotion_rows
    )
    token_table = _write_csv(
        destination / "token_projections.csv",
        ("emotion", "story_id", "token_index", "token", "projection", "top_k"),
        token_rows,
    )
    tables: list[Path] = [story_table, emotion_table, token_table]

    concentration_source = _write_csv(
        destination / "concentration_by_emotion.csv",
        tuple(emotion_rows[0]),
        emotion_rows,
    )
    tables.append(concentration_source)

    figures: list[Path] = []
    figure, axis = plt.subplots(figsize=(max(6.0, 0.45 * len(emotion_rows)), 4.2))
    x = np.arange(len(emotion_rows))
    axis.bar(x, [float(row["mean_peak_to_mean"]) for row in emotion_rows], color="#4472c4")
    axis.axhline(
        resolved_concentration_factor,
        color="#b22222",
        linestyle="--",
        label="absolute threshold",
    )
    axis.set_xticks(x, [str(row["emotion"]) for row in emotion_rows], rotation=45, ha="right")
    axis.set_ylabel("mean peak / shifted mean")
    axis.set_title("Within-story emotion-vector concentration")
    axis.legend(frameon=False)
    figures.extend(
        _save_figure(
            figure,
            destination / "concentration_by_emotion",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    for emotion, story_id, tokens, scores in plot_payloads:
        heatmap_stem = destination / f"heatmap_{_slug(emotion)}_{_slug(story_id)}"
        tables.append(
            _write_csv(
                heatmap_stem.with_suffix(".csv"),
                ("emotion", "story_id", "token_index", "token", "projection"),
                [
                    {
                        "emotion": emotion,
                        "story_id": story_id,
                        "token_index": token_index,
                        "token": token,
                        "projection": float(scores[token_index]),
                    }
                    for token_index, token in enumerate(tokens)
                ],
            )
        )
        width = min(20.0, max(7.0, 0.35 * len(tokens)))
        figure, axis = plt.subplots(figsize=(width, 2.6))
        image = axis.imshow(scores[np.newaxis, :], aspect="auto", cmap="coolwarm")
        step = max(1, int(np.ceil(len(tokens) / 40)))
        positions = np.arange(0, len(tokens), step)
        axis.set_xticks(
            positions, [tokens[position] for position in positions], rotation=60, ha="right"
        )
        axis.set_yticks([0], [emotion])
        axis.set_title(f"{emotion}: story {story_id}")
        figure.colorbar(image, ax=axis, label="projection")
        figures.extend(
            _save_figure(
                figure,
                heatmap_stem,
                formats=formats,
                dpi=resolved_dpi,
            )
        )
        plt.close(figure)

    report = _write_report(
        destination / "report.md",
        title="V1 — Within-story localization",
        passed=passed,
        summary=(
            f"Evaluated {len(story_rows)} stories across {len(emotion_rows)} emotions.",
            f"Emotion pass rate: {emotion_pass_rate:.3f}; required: "
            f"{resolved_minimum_pass_rate:.3f}.",
            "A story passes when its shifted top-k mass or peak-to-mean exceeds "
            f"the uniform baseline by a factor of {resolved_concentration_factor:.2f}, and "
            f"the matching vector ranks at or above the {resolved_control_percentile:.2f} "
            "quantile of non-matching emotion-vector controls.",
        ),
        figures=figures,
        tables=tables,
    )
    _write_manifest_if_supported(
        destination,
        config,
        "V1_localization",
        {
            "stories": len(story_rows),
            "emotions": len(emotion_rows),
            "passed": passed,
        },
    )
    return VerificationResult(
        name="V1_localization",
        passed=passed,
        output_dir=destination,
        report=report,
        tables=tuple(tables),
        figures=tuple(figures),
        metrics={
            "n_stories": len(story_rows),
            "n_emotions": len(emotion_rows),
            "emotion_pass_rate": emotion_pass_rate,
            "minimum_control_percentile": resolved_control_percentile,
        },
    )


__all__ = [
    "ActivationProvider",
    "LocalizationStory",
    "concentration_metrics",
    "project_token_activations",
    "run_localization",
]
