"""V2: surface emotion-vector activation in a distinct held-out corpus."""

from __future__ import annotations

import csv
import errno
import inspect
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
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
    _write_csv,
    _write_manifest_if_supported,
    _write_report,
)
from .logit_lens import DEFAULT_RELATED_TERMS


@dataclass(frozen=True, slots=True)
class HeldOutDocument:
    """Held-out text with optional activations or on-device projections."""

    text: str
    activations: Any | None = None
    projections: Any | None = None
    tokens: Sequence[str] | None = None
    document_id: str = ""
    dataset: str = "held_out"
    attention_mask: Any | None = None


@dataclass(frozen=True, slots=True)
class _TopDocument:
    score: float
    document_index: int
    dataset: str
    document_id: str
    text: str
    tokens: tuple[str, ...]
    values: np.ndarray


ActivationProvider = Callable[[HeldOutDocument], Any]


def project_held_out_activations(
    activations: Any,
    vectors: Any,
    *,
    normalize_vectors: bool = True,
) -> np.ndarray:
    """Project ``[tokens, d_model]`` activations onto ``[emotions, d_model]``."""

    matrix = _as_numpy(activations, dtype=np.float32)
    directions = _as_numpy(vectors, dtype=np.float32)
    if matrix.ndim != 2 or directions.ndim != 2:
        raise ValueError("activations and vectors must both be rank-two arrays")
    if matrix.shape[1] != directions.shape[1]:
        raise ValueError("activation and vector dimensions do not agree")
    if normalize_vectors:
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError("emotion vectors must have non-zero norm")
        directions = directions / norms
    projections = matrix @ directions.T
    if not np.isfinite(projections).all():
        raise ValueError("held-out projection produced a non-finite value")
    return projections


def percentile_thresholds(projections: Sequence[np.ndarray], percentile: float) -> np.ndarray:
    """Compute one dataset-wide token threshold per emotion."""

    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if not projections:
        raise ValueError("at least one projection array is required")
    widths = {array.shape[1] for array in projections if array.ndim == 2}
    if len(widths) != 1 or len(projections) != sum(array.ndim == 2 for array in projections):
        raise ValueError("all projection arrays must be [tokens, emotions] with equal width")
    concatenated = np.concatenate(projections, axis=0)
    if concatenated.shape[0] == 0:
        raise ValueError("the held-out corpus contains no unmasked tokens")
    return np.percentile(concatenated, percentile, axis=0)


def highlight_tokens_html(
    tokens: Sequence[str],
    scores: Any,
    threshold: float,
) -> str:
    """Render tokens with strictly-above-threshold values in ``<mark>`` tags."""

    values = _as_numpy(scores, dtype=np.float64).reshape(-1)
    if len(tokens) != values.size:
        raise ValueError("tokens and projection scores must have equal length")
    rendered = []
    for token, score in zip(tokens, values, strict=True):
        safe = escape(str(token))
        rendered.append(f"<mark>{safe}</mark>" if float(score) > threshold else safe)
    return " ".join(rendered)


def _coerce_document(
    value: HeldOutDocument | Mapping[str, Any] | str,
    index: int,
) -> HeldOutDocument:
    if isinstance(value, HeldOutDocument):
        return value
    if isinstance(value, str):
        return HeldOutDocument(text=value, document_id=str(index))
    if not isinstance(value, Mapping):
        raise TypeError("documents must be strings, HeldOutDocument objects, or mappings")
    return HeldOutDocument(
        text=str(value.get("text", "")),
        activations=value.get("activations", value.get("token_activations")),
        projections=value.get("projections", value.get("token_projections")),
        tokens=value.get("tokens"),
        document_id=str(value.get("document_id", value.get("id", index))),
        dataset=str(value.get("dataset", value.get("source", "held_out"))),
        attention_mask=value.get("attention_mask"),
    )


def _call_provider(
    provider: ActivationProvider, document: HeldOutDocument
) -> tuple[Any, Sequence[str] | None]:
    output = provider(document)
    if isinstance(output, HeldOutDocument):
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
                document: HeldOutDocument,
                *,
                method: Any = method,
                accepts_layer: bool = accepts_layer,
            ) -> Any:
                if accepts_layer and layer is not None:
                    return method(document.text, layer=layer)
                return method(document.text)

            return provider
    raise TypeError(
        "model_bundle must expose token_activations/capture_token_activations, "
        "or activation_provider must be supplied"
    )


def _select_layer(activations: Any, layer: int | None, width: int) -> np.ndarray:
    array = _as_numpy(activations, dtype=np.float32)
    if array.ndim == 2:
        return array
    if array.ndim != 3 or layer is None:
        raise ValueError(
            "activations must be [tokens, d_model], or layer must select a "
            "[layers, tokens, d_model] tensor"
        )
    if array.shape[-1] != width or not 0 <= layer < array.shape[0]:
        raise ValueError("invalid layer or activation shape")
    return array[layer]


def _kernel_projection(
    matrix: np.ndarray,
    directions: np.ndarray,
    percentile: float,
) -> np.ndarray | None:
    """Use the optional fused projection dispatcher when its dependencies exist."""

    try:
        import torch

        from ..kernels.project_threshold import project_threshold

        activation_tensor = torch.as_tensor(matrix, dtype=torch.float32)
        vector_tensor = torch.as_tensor(directions, dtype=torch.float32)
        output = project_threshold(
            activation_tensor,
            vector_tensor,
            percentile=percentile,
            thresholds=torch.zeros(vector_tensor.shape[0], dtype=torch.float32),
            use_kernel=True,
        )
        projected = output[0] if isinstance(output, tuple) else output.projections
        return _as_numpy(projected, dtype=np.float32)
    except (ImportError, AttributeError, TypeError, RuntimeError, ValueError) as error:
        warnings.warn(
            f"fused project-threshold unavailable; using numpy projection ({error})",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _exact_disk_percentiles(
    projections: np.memmap,
    percentile: float,
    temporary_dir: Path,
    *,
    chunk_rows: int = 8192,
) -> np.ndarray:
    """Calculate NumPy's linear percentile exactly, one disk-backed column at a time."""

    n_tokens, n_emotions = projections.shape
    if n_tokens <= 0:
        raise ValueError("cannot calculate percentiles without token projections")
    rank = (n_tokens - 1) * (float(percentile) / 100.0)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    fraction = rank - lower
    thresholds = np.empty(n_emotions, dtype=np.float64)
    column_path = temporary_dir / "percentile-column.f32"
    for emotion_index in range(n_emotions):
        column = np.memmap(column_path, mode="w+", dtype=np.float32, shape=(n_tokens,))
        for start in range(0, n_tokens, chunk_rows):
            end = min(start + chunk_rows, n_tokens)
            column[start:end] = projections[start:end, emotion_index]
        column.flush()
        column.partition(lower if lower == upper else (lower, upper))
        thresholds[emotion_index] = float(column[lower]) + fraction * (
            float(column[upper]) - float(column[lower])
        )
        column._mmap.close()
        del column
    column_path.unlink(missing_ok=True)
    return thresholds


def _histogram_edges(minimum: float, maximum: float, bins: int) -> np.ndarray:
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    return np.linspace(minimum, maximum, bins + 1, dtype=np.float64)


def _histogram_rows(
    counts: np.ndarray,
    edges: Sequence[np.ndarray],
    emotions: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column, emotion in enumerate(emotions):
        for index, count in enumerate(counts[column]):
            rows.append(
                {
                    "emotion": emotion,
                    "bin_left": float(edges[column][index]),
                    "bin_right": float(edges[column][index + 1]),
                    "count": int(count),
                }
            )
    return rows


def _write_csv_streaming(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    """Atomically write rows without materializing a document-by-emotion table."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _cleanup_temporary_workspace(workspace: tempfile.TemporaryDirectory[str]) -> None:
    """Remove disk-backed V2 state, tolerating delayed NFS directory updates."""

    for attempt in range(5):
        try:
            workspace.cleanup()
            return
        except OSError as error:
            if error.errno != errno.ENOTEMPTY:
                raise
            time.sleep(0.05 * (attempt + 1))
    # Cleanup must never turn a completed verification into a failed run. A
    # final best-effort removal also handles a stale empty directory on NFS.
    shutil.rmtree(workspace.name, ignore_errors=True)


def _close_memmap(value: object) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None and not mmap.closed:
        mmap.close()


def run_held_out(
    documents: Iterable[HeldOutDocument | Mapping[str, Any] | str],
    emotion_vectors: Mapping[str, Any] | np.ndarray | EmotionVectorArtifact,
    *,
    emotions: Sequence[str] | None = None,
    layer: int | None = None,
    activation_provider: ActivationProvider | None = None,
    model_bundle: object | None = None,
    output_dir: str | Path | None = None,
    config: object | None = None,
    percentile: float | None = None,
    max_documents: int | None = None,
    top_documents: int | None = None,
    expected_terms: Mapping[str, Sequence[str]] | None = None,
    minimum_emotion_pass_rate: float | None = None,
    use_kernels: bool | None = None,
    plot_formats: Sequence[str] | None = None,
    dpi: int | None = None,
) -> VerificationResult:
    """Run V2 while retaining projections, never the full activation corpus."""

    vector_map = _coerce_vector_mapping(emotion_vectors, emotions, layer=layer)
    effective_layer = (
        emotion_vectors.primary_layer
        if layer is None and isinstance(emotion_vectors, EmotionVectorArtifact)
        else layer
    )
    labels = tuple(vector_map)
    directions = np.stack([vector_map[label] for label in labels]).astype(np.float32, copy=False)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    destination = _resolve_output_dir(output_dir, config, "V2_held_out")
    # Remove the obsolete dense token-level artifact from pre-streaming runs.
    (destination / "token_projections.npz").unlink(missing_ok=True)
    formats, resolved_dpi = _plot_options(config, plot_formats, dpi)
    resolved_percentile = float(
        percentile
        if percentile is not None
        else _config_value(config, "activation_percentile", 90.0)
    )
    resolved_max = int(
        max_documents
        if max_documents is not None
        else _config_value(config, "heldout_max_docs", 50_000)
    )
    resolved_kernels = bool(
        use_kernels if use_kernels is not None else _config_value(config, "use_kernels", True)
    )
    resolved_top_documents = int(
        top_documents if top_documents is not None else _config_value(config, "v2_top_documents", 3)
    )
    resolved_minimum_pass_rate = float(
        minimum_emotion_pass_rate
        if minimum_emotion_pass_rate is not None
        else _config_value(config, "v2_min_emotion_pass_rate", 0.5)
    )
    if resolved_max <= 0 or resolved_top_documents <= 0:
        raise ValueError("max_documents and top_documents must be positive")
    if not 0.0 <= resolved_percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if not 0.0 <= resolved_minimum_pass_rate <= 1.0:
        raise ValueError("minimum_emotion_pass_rate must be in [0, 1]")
    if activation_provider is None and model_bundle is not None:
        activation_provider = _bundle_provider(model_bundle, effective_layer)

    # The token-level matrix is needed twice (exact quantiles, then sparse hits),
    # but must never become a permanent dense artifact. Append it to ordinary
    # disk and keep only bounded top-example payloads in memory.
    # The configured output can be NFS and the dense temporary sweep can be
    # several GiB. Use node-local scratch; only compact permanent artifacts are
    # copied into the output directory.
    temporary_workspace = tempfile.TemporaryDirectory(prefix="emotion-vectors-v2-")
    temporary_dir = Path(temporary_workspace.name)
    projection_path = temporary_dir / "token-projections.f32"
    maxima_path = temporary_dir / "document-maxima.f32"
    document_metadata: list[tuple[str, str, int, int]] = []
    offsets_list = [0]
    top_candidates: list[list[_TopDocument]] = [[] for _ in labels]
    projection_sum = np.zeros(len(labels), dtype=np.float64)
    projection_sum_squares = np.zeros(len(labels), dtype=np.float64)
    projection_minimum = np.full(len(labels), np.inf, dtype=np.float64)
    projection_maximum = np.full(len(labels), -np.inf, dtype=np.float64)
    kernel_used = False
    with projection_path.open("wb") as projection_handle, maxima_path.open("wb") as maxima_handle:
        for index, value in enumerate(documents):
            if index >= resolved_max:
                break
            document = _coerce_document(value, index)
            activations = document.activations
            projected = (
                None
                if document.projections is None
                else _as_numpy(document.projections, dtype=np.float32)
            )
            if projected is not None and (projected.ndim != 2 or projected.shape[1] != len(labels)):
                raise ValueError(f"precomputed projections must be [tokens, {len(labels)}]")
            tokens = document.tokens
            if projected is None and activations is None:
                if activation_provider is None:
                    raise ValueError(
                        f"document {document.document_id or index!r} has no activations and no provider"
                    )
                activations, supplied_tokens = _call_provider(activation_provider, document)
                tokens = tokens if tokens is not None else supplied_tokens
            matrix = (
                None
                if projected is not None
                else _select_layer(activations, effective_layer, directions.shape[1])
            )
            if document.attention_mask is not None:
                mask = _as_numpy(document.attention_mask).astype(bool).reshape(-1)
                row_count = projected.shape[0] if projected is not None else matrix.shape[0]
                if mask.size != row_count:
                    raise ValueError("attention_mask length does not match token rows")
                if projected is not None:
                    projected = projected[mask]
                else:
                    matrix = matrix[mask]
                if tokens is not None:
                    tokens = [str(token) for token, keep in zip(tokens, mask, strict=True) if keep]
            row_count = projected.shape[0] if projected is not None else matrix.shape[0]
            if row_count == 0:
                continue
            if tokens is None or len(tokens) != row_count:
                tokens = [f"token_{token_index}" for token_index in range(row_count)]
            token_tuple = tuple(map(str, tokens))
            if projected is None:
                projected = (
                    _kernel_projection(matrix, directions, resolved_percentile)
                    if resolved_kernels
                    else None
                )
            if projected is None:
                projected = project_held_out_activations(
                    matrix, directions, normalize_vectors=False
                )
            # The dispatcher currently selects its exact torch fallback; do not
            # label that path as a custom fused-kernel execution.
            projected = np.ascontiguousarray(projected, dtype=np.float32)
            if projected.shape != (row_count, len(labels)):
                raise ValueError(
                    f"projection has shape {projected.shape}; expected "
                    f"{(row_count, len(labels))}"
                )

            document_index = len(document_metadata)
            token_start = offsets_list[-1]
            token_end = token_start + row_count
            document_id = document.document_id or str(document_index)
            maxima = projected.max(axis=0)
            projected.tofile(projection_handle)
            maxima.astype(np.float32, copy=False).tofile(maxima_handle)
            document_metadata.append((document.dataset, document_id, token_start, token_end))
            offsets_list.append(token_end)
            projection_sum += projected.sum(axis=0, dtype=np.float64)
            projection_sum_squares += np.square(projected, dtype=np.float64).sum(axis=0)
            projection_minimum = np.minimum(projection_minimum, projected.min(axis=0))
            projection_maximum = np.maximum(projection_maximum, projected.max(axis=0))

            for emotion_index, maximum in enumerate(maxima):
                bucket = top_candidates[emotion_index]
                candidate = _TopDocument(
                    score=float(maximum),
                    document_index=document_index,
                    dataset=document.dataset,
                    document_id=document_id,
                    text=document.text,
                    tokens=token_tuple,
                    values=np.array(projected[:, emotion_index], copy=True),
                )
                bucket.append(candidate)
                bucket.sort(key=lambda item: (item.score, item.document_index), reverse=True)
                del bucket[resolved_top_documents:]

    if not document_metadata:
        _cleanup_temporary_workspace(temporary_workspace)
        raise ValueError("the held-out sweep produced no documents with tokens")

    offsets = np.asarray(offsets_list, dtype=np.int64)
    n_documents = len(document_metadata)
    n_tokens = int(offsets[-1])
    projection_memmap = np.memmap(
        projection_path,
        mode="r",
        dtype=np.float32,
        shape=(n_tokens, len(labels)),
    )
    document_maxima = np.memmap(
        maxima_path,
        mode="r",
        dtype=np.float32,
        shape=(n_documents, len(labels)),
    )
    thresholds = _exact_disk_percentiles(projection_memmap, resolved_percentile, temporary_dir)

    # Sparse threshold hits and histogram counts are produced in one bounded
    # pass over the temporary projection matrix.
    histogram_edges = tuple(
        _histogram_edges(float(projection_minimum[index]), float(projection_maximum[index]), 30)
        for index in range(len(labels))
    )
    histogram_counts = np.zeros((len(labels), 30), dtype=np.int64)
    highlight_counts = np.zeros(len(labels), dtype=np.int64)
    hit_token_path = temporary_dir / "above-token.i64"
    hit_emotion_path = temporary_dir / "above-emotion.u16"
    with hit_token_path.open("wb") as token_handle, hit_emotion_path.open("wb") as emotion_handle:
        for start in range(0, n_tokens, 8192):
            end = min(start + 8192, n_tokens)
            block = np.asarray(projection_memmap[start:end])
            local_token, emotion_index = np.nonzero(block > thresholds[None, :])
            (local_token.astype(np.int64) + start).tofile(token_handle)
            emotion_index.astype(np.uint16).tofile(emotion_handle)
            highlight_counts += np.bincount(emotion_index, minlength=len(labels)).astype(np.int64)
            for column in range(len(labels)):
                histogram_counts[column] += np.histogram(
                    block[:, column], bins=histogram_edges[column]
                )[0]

    n_sparse_hits = int(highlight_counts.sum())
    threshold_archive = destination / "above_threshold_indices.npz"
    if n_sparse_hits:
        hit_tokens = np.memmap(hit_token_path, mode="r", dtype=np.int64, shape=(n_sparse_hits,))
        hit_emotions = np.memmap(
            hit_emotion_path, mode="r", dtype=np.uint16, shape=(n_sparse_hits,)
        )
    else:
        hit_tokens = np.empty(0, dtype=np.int64)
        hit_emotions = np.empty(0, dtype=np.uint16)
    np.savez_compressed(
        threshold_archive,
        global_token_index=hit_tokens,
        emotion_index=hit_emotions,
    )
    _close_memmap(hit_tokens)
    _close_memmap(hit_emotions)
    del hit_tokens, hit_emotions

    # The permanent dense artifact is document-level only. Offsets map sparse
    # global token indices back to documents without retaining token projections.
    projection_archive = destination / "document_projections.npz"
    np.savez_compressed(
        projection_archive,
        max_projections=document_maxima,
        document_offsets=offsets,
        thresholds=thresholds,
        emotions=np.asarray(labels, dtype=np.str_),
    )

    def document_index_rows() -> Iterable[Mapping[str, object]]:
        for document_index, (dataset, document_id, token_start, token_end) in enumerate(
            document_metadata
        ):
            yield {
                "document_index": document_index,
                "dataset": dataset,
                "document_id": document_id,
                "token_start": token_start,
                "token_end": token_end,
            }

    document_index_table = _write_csv_streaming(
        destination / "document_index.csv",
        ("document_index", "dataset", "document_id", "token_start", "token_end"),
        document_index_rows(),
    )
    reference_terms: Mapping[str, Sequence[str]] = expected_terms or {
        emotion: DEFAULT_RELATED_TERMS.get(emotion, (emotion,)) for emotion in labels
    }

    highlighted_rows: list[dict[str, object]] = []

    top_rows: list[dict[str, object]] = []
    html_sections: list[str] = [
        "<!doctype html><meta charset='utf-8'><title>V2 highlighted snippets</title>",
        "<style>body{font-family:sans-serif;max-width:1000px;margin:auto} mark{background:#ffd54f}</style>",
        "<h1>Held-out snippets above the activation percentile</h1>",
    ]
    emotion_rows: list[dict[str, object]] = []
    projection_variance = np.maximum(
        projection_sum_squares / n_tokens - (projection_sum / n_tokens) ** 2,
        0.0,
    )
    projection_std = np.sqrt(projection_variance)
    for column, emotion in enumerate(labels):
        candidates = top_candidates[column]
        relevance_flags: list[bool] = []
        html_sections.append(f"<h2>{escape(emotion)}</h2>")
        for rank, candidate in enumerate(candidates, start=1):
            tokens = candidate.tokens
            values = candidate.values
            rendered = highlight_tokens_html(tokens, values, float(thresholds[column]))
            terms = tuple(term.lower() for term in reference_terms.get(emotion, (emotion,)))
            searchable_text = f"{candidate.text} {' '.join(tokens)}".lower()
            relevant = any(term in searchable_text for term in terms)
            relevance_flags.append(relevant)
            hit_indices = np.flatnonzero(values > thresholds[column])
            for token_index in hit_indices:
                highlighted_rows.append(
                    {
                        "dataset": candidate.dataset,
                        "document_id": candidate.document_id,
                        "emotion": emotion,
                        "token_index": int(token_index),
                        "token": tokens[int(token_index)],
                        "projection": float(values[int(token_index)]),
                        "threshold": float(thresholds[column]),
                    }
                )
            top_rows.append(
                {
                    "emotion": emotion,
                    "rank": rank,
                    "dataset": candidate.dataset,
                    "document_id": candidate.document_id,
                    "max_projection": candidate.score,
                    "threshold": float(thresholds[column]),
                    "expected_term_match": relevant,
                    "text": candidate.text,
                }
            )
            html_sections.append(
                f"<h3>#{rank} — {escape(candidate.dataset)} / "
                f"{escape(candidate.document_id)}</h3><p>{rendered}</p>"
            )
        has_variation = float(projection_std[column]) > np.finfo(np.float64).eps
        n_highlights = int(highlight_counts[column])
        relevance_rate = float(np.mean(relevance_flags)) if relevance_flags else 0.0
        emotion_passed = has_variation and n_highlights > 0 and relevance_rate >= 0.5
        emotion_rows.append(
            {
                "emotion": emotion,
                "threshold": float(thresholds[column]),
                "projection_std": float(projection_std[column]),
                "n_highlighted": n_highlights,
                "top_document_expected_term_rate": relevance_rate,
                "passed": emotion_passed,
            }
        )

    snippets_path = destination / "highlighted_snippets.html"
    snippets_path.write_text("\n".join(html_sections) + "\n", encoding="utf-8")

    def document_projection_rows(
        projection_data: np.ndarray = projection_memmap,
        maxima_data: np.ndarray = document_maxima,
    ) -> Iterable[Mapping[str, object]]:
        for document_index, (dataset, document_id, token_start, token_end) in enumerate(
            document_metadata
        ):
            document_hits = np.sum(
                projection_data[token_start:token_end] > thresholds[None, :],
                axis=0,
            )
            for emotion_index, emotion in enumerate(labels):
                yield {
                    "document_index": document_index,
                    "dataset": dataset,
                    "document_id": document_id,
                    "emotion": emotion,
                    "max_projection": float(maxima_data[document_index, emotion_index]),
                    "threshold": float(thresholds[emotion_index]),
                    "n_highlighted": int(document_hits[emotion_index]),
                }

    document_table = _write_csv_streaming(
        destination / "document_projections.csv",
        (
            "document_index",
            "dataset",
            "document_id",
            "emotion",
            "max_projection",
            "threshold",
            "n_highlighted",
        ),
        document_projection_rows(),
    )
    highlighted_fields = (
        "dataset",
        "document_id",
        "emotion",
        "token_index",
        "token",
        "projection",
        "threshold",
    )
    highlighted_table = _write_csv(
        destination / "highlighted_tokens.csv", highlighted_fields, highlighted_rows
    )
    top_table = _write_csv(destination / "top_snippets.csv", tuple(top_rows[0]), top_rows)
    emotion_table = _write_csv(
        destination / "emotion_summary.csv", tuple(emotion_rows[0]), emotion_rows
    )
    histogram_rows = _histogram_rows(histogram_counts, histogram_edges, labels)
    histogram_table = _write_csv(
        destination / "projection_histogram.csv", tuple(histogram_rows[0]), histogram_rows
    )
    histogram_source = _write_csv(
        destination / "projection_histograms.csv",
        tuple(histogram_rows[0]),
        histogram_rows,
    )
    top_examples_source = _write_csv(
        destination / "top_activating_examples.csv", tuple(top_rows[0]), top_rows
    )
    tables = (
        document_table,
        highlighted_table,
        top_table,
        emotion_table,
        histogram_table,
        histogram_source,
        top_examples_source,
        document_index_table,
        projection_archive,
        threshold_archive,
    )

    figures: list[Path] = []
    columns = min(4, len(labels))
    rows = int(np.ceil(len(labels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.8 * rows), squeeze=False)
    for column, emotion in enumerate(labels):
        axis = axes.flat[column]
        edges = histogram_edges[column]
        axis.bar(
            (edges[:-1] + edges[1:]) / 2.0,
            histogram_counts[column],
            width=np.diff(edges),
            color="#4c78a8",
            alpha=0.85,
            align="center",
        )
        axis.axvline(thresholds[column], color="#d62728", linestyle="--")
        axis.set_title(emotion)
        axis.set_xlabel("projection")
    for axis in axes.flat[len(labels) :]:
        axis.set_visible(False)
    figure.suptitle(f"Held-out projection distributions ({resolved_percentile:g}th percentile)")
    figure.tight_layout()
    figures.extend(
        _save_figure(
            figure,
            destination / "projection_histograms",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.8 * rows), squeeze=False)
    for column, emotion in enumerate(labels):
        axis = axes.flat[column]
        rows_for_emotion = [row for row in top_rows if row["emotion"] == emotion]
        axis.barh(
            [f"#{int(row['rank'])}" for row in rows_for_emotion][::-1],
            [float(row["max_projection"]) for row in rows_for_emotion][::-1],
            color="#59a14f",
        )
        axis.axvline(thresholds[column], color="#d62728", linestyle="--")
        axis.set_title(emotion)
        axis.set_xlabel("document maximum")
    for axis in axes.flat[len(labels) :]:
        axis.set_visible(False)
    figure.suptitle("Top activating held-out documents")
    figure.tight_layout()
    figures.extend(
        _save_figure(
            figure,
            destination / "top_activating_examples",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    emotion_pass_rate = float(np.mean([bool(row["passed"]) for row in emotion_rows]))
    passed = emotion_pass_rate >= resolved_minimum_pass_rate
    report = _write_report(
        destination / "report.md",
        title="V2 — Expected contexts in held-out data",
        passed=passed,
        summary=(
            f"Swept {n_documents} held-out documents; retained dense document maxima and "
            "sparse above-threshold token indices only.",
            f"Highlighted tokens strictly above the dataset-wide {resolved_percentile:g}th percentile.",
            f"Emotion pass rate: {emotion_pass_rate:.3f}; required: "
            f"{resolved_minimum_pass_rate:.3f}.",
            "Expected-context relevance is scored by supplied lexical references, or by "
            "the balanced-emotion reference lexicon when none is supplied.",
            f"Fused project-threshold path used: {kernel_used}.",
            f"Rendered snippets: [{snippets_path.name}]({snippets_path.name}).",
        ),
        figures=figures,
        tables=tables,
    )
    _write_manifest_if_supported(
        destination,
        config,
        "V2_held_out",
        {
            "documents": n_documents,
            "emotions": len(labels),
            "highlighted_tokens": n_sparse_hits,
            "passed": passed,
        },
    )
    del document_projection_rows
    if "block" in locals():
        del block
    _close_memmap(projection_memmap)
    _close_memmap(document_maxima)
    del projection_memmap, document_maxima
    _cleanup_temporary_workspace(temporary_workspace)
    return VerificationResult(
        name="V2_held_out",
        passed=passed,
        output_dir=destination,
        report=report,
        tables=tables,
        figures=tuple(figures),
        metrics={
            "n_documents": n_documents,
            "n_tokens": n_tokens,
            "n_emotions": len(labels),
            "percentile": resolved_percentile,
            "emotion_pass_rate": emotion_pass_rate,
            "n_above_threshold": n_sparse_hits,
            "kernel_used": kernel_used,
        },
    )


__all__ = [
    "ActivationProvider",
    "HeldOutDocument",
    "highlight_tokens_html",
    "percentile_thresholds",
    "project_held_out_activations",
    "run_held_out",
]
