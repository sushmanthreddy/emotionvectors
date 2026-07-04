"""V3: project emotion directions through the model unembedding."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
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

DEFAULT_RELATED_TERMS: Mapping[str, tuple[str, ...]] = {
    "excited": ("excited", "thrill", "eager", "anticipat"),
    "elated": ("elated", "delight", "overjoy", "exhilar"),
    "ecstatic": ("ecstatic", "euphor", "raptur", "bliss"),
    "enthusiastic": ("enthusias", "eager", "zeal", "passion"),
    "joyful": ("joy", "happy", "glad", "delight"),
    "content": ("content", "satisf", "fulfilled", "ease"),
    "calm": ("calm", "peace", "quiet", "tranquil"),
    "serene": ("serene", "tranquil", "peace", "still"),
    "grateful": ("grateful", "thank", "appreciat", "gratitude"),
    "relaxed": ("relax", "ease", "unwind", "rest"),
    "angry": ("angry", "anger", "mad", "rage"),
    "furious": ("furious", "fury", "rage", "irate"),
    "terrified": ("terr", "fear", "fright", "horror"),
    "anxious": ("anx", "worr", "uneasy", "nervous"),
    "panicked": ("panic", "alarm", "frantic", "fear"),
    "outraged": ("outrag", "indignan", "furious", "offend"),
    "sad": ("sad", "grief", "tear", "sorrow", "unhappy"),
    "depressed": ("depress", "hopeless", "despair", "bleak"),
    "gloomy": ("gloom", "bleak", "somber", "dreary"),
    "lonely": ("lonely", "alone", "isolat", "lonesome"),
    "miserable": ("miser", "wretched", "unhappy", "suffer"),
    "bored": ("bored", "tedious", "dull", "monoton"),
    "surprised": ("surpris", "astonish", "startl", "unexpected"),
    "proud": ("proud", "pride", "accomplish", "honor"),
    "hopeful": ("hope", "optim", "promise", "expect"),
    "nostalgic": ("nostalg", "memory", "reminisc", "past"),
    "guilty": ("guilt", "remorse", "regret", "culpab"),
    "ashamed": ("shame", "embarrass", "humiliat", "disgrace"),
    "jealous": ("jealous", "envy", "resent", "possess"),
    "disgusted": ("disgust", "revuls", "repuls", "nause"),
    "desperate": ("desper", "urgent", "bankrupt", "dire"),
}


@dataclass(frozen=True, slots=True)
class LogitLensResult:
    """Numerical V3 projection before report rendering."""

    emotions: tuple[str, ...]
    tokens: tuple[str, ...]
    scores: np.ndarray
    top_up_indices: np.ndarray
    top_down_indices: np.ndarray


RelevanceScorer = Callable[[str, str], bool]


def project_unembedding(emotion_vectors: Any, unembedding: Any) -> np.ndarray:
    """Return ``[emotion, vocabulary]`` logit changes.

    The unembedding may use either Hugging Face's ``[vocab, d_model]`` layout
    or TransformerLens' ``[d_model, vocab]`` layout.
    """

    vectors = _as_numpy(emotion_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("emotion_vectors and unembedding must be rank two")

    # Keep a live model's (potentially multi-gigabyte, bf16) unembedding on its
    # current device. Moving it through NumPy would widen it and can exhaust host
    # memory for a 32B model; only the compact [emotion, vocabulary] result moves.
    if hasattr(unembedding, "detach") and hasattr(unembedding, "device"):
        import torch

        matrix_tensor = unembedding.detach()
        if matrix_tensor.ndim != 2:
            raise ValueError("emotion_vectors and unembedding must be rank two")
        vector_tensor = torch.as_tensor(
            vectors,
            device=matrix_tensor.device,
            dtype=matrix_tensor.dtype,
        )
        width = vectors.shape[1]
        with torch.inference_mode():
            if matrix_tensor.shape[1] == width:
                score_tensor = vector_tensor @ matrix_tensor.transpose(0, 1)
            elif matrix_tensor.shape[0] == width:
                score_tensor = vector_tensor @ matrix_tensor
            else:
                raise ValueError(
                    f"unembedding shape {tuple(matrix_tensor.shape)} is incompatible "
                    f"with d_model={width}"
                )
        scores = _as_numpy(score_tensor.float(), dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("unembedding projection produced a non-finite score")
        return scores

    matrix = _as_numpy(unembedding, dtype=np.float32)
    if vectors.ndim != 2 or matrix.ndim != 2:
        raise ValueError("emotion_vectors and unembedding must be rank two")
    width = vectors.shape[1]
    if matrix.shape[1] == width:
        scores = vectors @ matrix.T
    elif matrix.shape[0] == width:
        scores = vectors @ matrix
    else:
        raise ValueError(f"unembedding shape {matrix.shape} is incompatible with d_model={width}")
    if not np.isfinite(scores).all():
        raise ValueError("unembedding projection produced a non-finite score")
    return scores


def compute_logit_lens(
    emotion_vectors: Any,
    unembedding: Any,
    emotions: Sequence[str],
    tokens: Sequence[str],
    *,
    top_k: int = 20,
) -> LogitLensResult:
    scores = project_unembedding(emotion_vectors, unembedding)
    if len(emotions) != scores.shape[0] or len(tokens) != scores.shape[1]:
        raise ValueError("emotion/token labels do not match the projected score matrix")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(top_k, scores.shape[1])
    up = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    down = np.argsort(scores, axis=1)[:, :k]
    return LogitLensResult(
        emotions=tuple(map(str, emotions)),
        tokens=tuple(map(str, tokens)),
        scores=scores,
        top_up_indices=up,
        top_down_indices=down,
    )


def _normalize_token(token: str) -> str:
    token = token.replace("Ġ", " ").replace("▁", " ").strip().lower()
    return re.sub(r"[^a-z]+", "", token)


def token_is_emotion_related(
    emotion: str,
    token: str,
    related_terms: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    normalized = _normalize_token(token)
    if len(normalized) < 3:
        return False
    terms = (related_terms or DEFAULT_RELATED_TERMS).get(emotion, (emotion,))
    return any(
        len(term_normalized) >= 3
        and (term_normalized in normalized or normalized in term_normalized)
        for term in terms
        if (term_normalized := _normalize_token(str(term)))
    )


def _resolve_unembedding(model_bundle: object) -> Any:
    for candidate in (
        getattr(model_bundle, "unembedding", None),
        getattr(model_bundle, "unembedding_matrix", None),
        getattr(model_bundle, "W_U", None),
    ):
        if candidate is not None:
            return getattr(candidate, "weight", candidate)
    model = getattr(model_bundle, "model", model_bundle)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        return lm_head.weight
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        embeddings = getter()
        if embeddings is not None and getattr(embeddings, "weight", None) is not None:
            return embeddings.weight
    raise TypeError("could not resolve an unembedding matrix from model_bundle")


def _resolve_tokens(
    vocabulary_size: int,
    vocabulary_tokens: Sequence[str] | None,
    tokenizer: object | None,
) -> tuple[str, ...]:
    if vocabulary_tokens is not None:
        if len(vocabulary_tokens) != vocabulary_size:
            raise ValueError("vocabulary_tokens length does not match the unembedding")
        return tuple(map(str, vocabulary_tokens))
    if tokenizer is None:
        return tuple(f"token_{index}" for index in range(vocabulary_size))
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(converter):
        converted = converter(list(range(vocabulary_size)))
        if converted is not None and len(converted) == vocabulary_size:
            return tuple(map(str, converted))
    decoder = getattr(tokenizer, "decode", None)
    if callable(decoder):
        return tuple(str(decoder([index])) for index in range(vocabulary_size))
    raise TypeError("tokenizer cannot convert vocabulary ids to token strings")


def run_logit_lens(
    emotion_vectors: Mapping[str, Any] | np.ndarray | EmotionVectorArtifact,
    unembedding: Any | None = None,
    *,
    emotions: Sequence[str] | None = None,
    layer: int | None = None,
    vocabulary_tokens: Sequence[str] | None = None,
    tokenizer: object | None = None,
    model_bundle: object | None = None,
    related_terms: Mapping[str, Sequence[str]] | None = None,
    relevance_scorer: RelevanceScorer | None = None,
    top_k: int | None = None,
    minimum_related_hit_rate: float | None = None,
    output_dir: str | Path | None = None,
    config: object | None = None,
    plot_formats: Sequence[str] | None = None,
    dpi: int | None = None,
) -> VerificationResult:
    """Run V3 from an explicit unembedding or a caller-owned model bundle."""

    vector_map = _coerce_vector_mapping(emotion_vectors, emotions, layer=layer)
    labels = tuple(vector_map)
    vectors = np.stack([vector_map[label] for label in labels])
    if unembedding is None:
        if model_bundle is None:
            raise ValueError("unembedding or model_bundle is required")
        unembedding = _resolve_unembedding(model_bundle)
    matrix_shape = tuple(unembedding.shape)
    if len(matrix_shape) != 2:
        raise ValueError("unembedding must be rank two")
    vocabulary_size = matrix_shape[0] if matrix_shape[1] == vectors.shape[1] else matrix_shape[1]
    if tokenizer is None and model_bundle is not None:
        tokenizer = getattr(model_bundle, "tokenizer", None)
    tokens = _resolve_tokens(vocabulary_size, vocabulary_tokens, tokenizer)
    resolved_top_k = int(top_k if top_k is not None else _config_value(config, "v3_top_k", 20))
    resolved_minimum_hit_rate = float(
        minimum_related_hit_rate
        if minimum_related_hit_rate is not None
        else _config_value(config, "v3_min_related_hit_rate", 0.5)
    )
    numerical = compute_logit_lens(
        vectors,
        unembedding,
        labels,
        tokens,
        top_k=resolved_top_k,
    )
    destination = _resolve_output_dir(output_dir, config, "V3_logit_lens")
    formats, resolved_dpi = _plot_options(config, plot_formats, dpi)
    if not 0.0 <= resolved_minimum_hit_rate <= 1.0:
        raise ValueError("minimum_related_hit_rate must be in [0, 1]")

    scorer = relevance_scorer or (
        lambda emotion, token: token_is_emotion_related(emotion, token, related_terms)
    )
    token_rows: list[dict[str, object]] = []
    emotion_rows: list[dict[str, object]] = []
    for emotion_index, emotion in enumerate(labels):
        related_up = 0
        for direction, indices in (
            ("upweighted", numerical.top_up_indices[emotion_index]),
            ("downweighted", numerical.top_down_indices[emotion_index]),
        ):
            for rank, token_index in enumerate(indices, start=1):
                token = tokens[int(token_index)]
                related = bool(scorer(emotion, token))
                related_up += int(direction == "upweighted" and related)
                token_rows.append(
                    {
                        "emotion": emotion,
                        "direction": direction,
                        "rank": rank,
                        "token_id": int(token_index),
                        "token": token,
                        "score": float(numerical.scores[emotion_index, token_index]),
                        "emotion_related": related,
                    }
                )
        emotion_rows.append(
            {
                "emotion": emotion,
                "n_related_in_top_upweighted": related_up,
                "top_k": numerical.top_up_indices.shape[1],
                "related_fraction": related_up / numerical.top_up_indices.shape[1],
                "passed": related_up > 0,
            }
        )
    related_hit_rate = float(np.mean([bool(row["passed"]) for row in emotion_rows]))
    passed = related_hit_rate >= resolved_minimum_hit_rate

    token_table = _write_csv(
        destination / "top_unembedding_tokens.csv", tuple(token_rows[0]), token_rows
    )
    emotion_table = _write_csv(
        destination / "emotion_related_summary.csv", tuple(emotion_rows[0]), emotion_rows
    )

    selected_indices: list[int] = []
    compact_per_emotion = min(3, numerical.top_up_indices.shape[1])
    for indices in numerical.top_up_indices[:, :compact_per_emotion]:
        for index in indices:
            if int(index) not in selected_indices:
                selected_indices.append(int(index))
    compact_rows = [
        {
            "emotion": emotion,
            "token_id": token_index,
            "token": tokens[token_index],
            "score": float(numerical.scores[emotion_index, token_index]),
        }
        for emotion_index, emotion in enumerate(labels)
        for token_index in selected_indices
    ]
    compact_table = _write_csv(
        destination / "compact_token_matrix.csv",
        ("emotion", "token_id", "token", "score"),
        compact_rows,
    )
    tables: list[Path] = [token_table, emotion_table, compact_table]

    figures: list[Path] = []
    for emotion_index, emotion in enumerate(labels):
        token_figure_stem = destination / f"tokens_{_slug(emotion)}"
        emotion_token_rows = [row for row in token_rows if row["emotion"] == emotion]
        tables.append(
            _write_csv(
                token_figure_stem.with_suffix(".csv"),
                tuple(emotion_token_rows[0]),
                emotion_token_rows,
            )
        )
        k_plot = min(10, numerical.top_up_indices.shape[1])
        up = numerical.top_up_indices[emotion_index, :k_plot]
        down = numerical.top_down_indices[emotion_index, :k_plot]
        indices = np.concatenate((down[::-1], up))
        values = numerical.scores[emotion_index, indices]
        names = [tokens[int(index)] for index in indices]
        colors = ["#d65f5f" if value < 0 else "#4c78a8" for value in values]
        figure, axis = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(indices))))
        axis.barh(np.arange(len(indices)), values, color=colors)
        axis.set_yticks(np.arange(len(indices)), names)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("logit change")
        axis.set_title(f"Unembedding projection: {emotion}")
        figures.extend(
            _save_figure(
                figure,
                token_figure_stem,
                formats=formats,
                dpi=resolved_dpi,
            )
        )
        plt.close(figure)

    if selected_indices:
        tables.append(
            _write_csv(
                destination / "top_token_heatmap.csv",
                ("emotion", "token_id", "token", "score"),
                compact_rows,
            )
        )
        compact = numerical.scores[:, selected_indices]
        figure, axis = plt.subplots(
            figsize=(max(7.0, 0.35 * len(selected_indices)), max(4.5, 0.3 * len(labels)))
        )
        image = axis.imshow(compact, aspect="auto", cmap="coolwarm")
        axis.set_xticks(
            np.arange(len(selected_indices)),
            [tokens[index] for index in selected_indices],
            rotation=60,
            ha="right",
        )
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_title("Top upweighted tokens across emotions")
        figure.colorbar(image, ax=axis, label="logit change")
        figures.extend(
            _save_figure(
                figure,
                destination / "top_token_heatmap",
                formats=formats,
                dpi=resolved_dpi,
            )
        )
        plt.close(figure)

    report = _write_report(
        destination / "report.md",
        title="V3 — Unembedding projection",
        passed=passed,
        summary=(
            f"Projected {len(labels)} emotion vectors over {len(tokens)} vocabulary tokens.",
            f"Emotions with at least one related top-{numerical.top_up_indices.shape[1]} "
            f"upweighted token: {related_hit_rate:.3f}; required: "
            f"{resolved_minimum_hit_rate:.3f}.",
            "Relatedness uses the supplied lexical references (or the balanced-set defaults).",
        ),
        figures=figures,
        tables=tables,
    )
    _write_manifest_if_supported(
        destination,
        config,
        "V3_logit_lens",
        {
            "emotions": len(labels),
            "vocabulary_tokens": len(tokens),
            "passed": passed,
        },
    )
    return VerificationResult(
        name="V3_logit_lens",
        passed=passed,
        output_dir=destination,
        report=report,
        tables=tuple(tables),
        figures=tuple(figures),
        metrics={
            "n_emotions": len(labels),
            "vocabulary_size": len(tokens),
            "related_hit_rate": related_hit_rate,
        },
    )


__all__ = [
    "DEFAULT_RELATED_TERMS",
    "LogitLensResult",
    "compute_logit_lens",
    "project_unembedding",
    "run_logit_lens",
    "token_is_emotion_related",
]
