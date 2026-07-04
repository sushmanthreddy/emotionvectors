"""Stage 3: construct, denoise, and persist emotion vectors."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .config import ExperimentConfig

LOGGER = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.floating]
_VECTOR_IMPLEMENTATION_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PCAProjection:
    """Per-layer neutral PCA fit used by A5."""

    components: tuple[npt.NDArray[np.float64], ...]
    explained_variance_ratio: tuple[npt.NDArray[np.float64], ...]
    n_components: npt.NDArray[np.int64]


@dataclass(frozen=True)
class VectorBuildResult:
    output_path: Path
    raw: npt.NDArray[np.float32]
    denoised: npt.NDArray[np.float32]
    emotions: tuple[str, ...]
    primary_layer: int
    pca: PCAProjection


def mean_center_emotion_means(story_vectors: FloatArray) -> npt.NDArray[np.float32]:
    """A4: average stories, then center over the emotion axis.

    ``story_vectors`` has shape ``[emotion, story, layer, d_model]``.  For
    unequal story counts use :func:`mean_center_emotion_mapping` instead.
    """

    array = np.asarray(story_vectors)
    if array.ndim != 4:
        raise ValueError(
            "story_vectors must have shape [emotion, story, layer, d_model], " f"got {array.shape}"
        )
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("At least two emotions and one story per emotion are required")
    means = array.astype(np.float64, copy=False).mean(axis=1)
    means -= means.mean(axis=0, keepdims=True)
    return means.astype(np.float32)


def mean_center_emotion_mapping(
    vectors_by_emotion: Mapping[str, FloatArray], emotions: Sequence[str]
) -> npt.NDArray[np.float32]:
    """A4 with independently sized per-emotion story arrays."""

    if len(emotions) < 2:
        raise ValueError("Mean-centering requires at least two emotions")
    means: list[npt.NDArray[np.float64]] = []
    reference_shape: tuple[int, int] | None = None
    for emotion in emotions:
        if emotion not in vectors_by_emotion:
            raise KeyError(f"Missing story activations for emotion {emotion!r}")
        array = np.asarray(vectors_by_emotion[emotion])
        if array.ndim != 3 or array.shape[0] == 0:
            raise ValueError(
                f"{emotion}: expected non-empty [story, layer, d_model], got {array.shape}"
            )
        if reference_shape is None:
            reference_shape = (array.shape[1], array.shape[2])
        elif array.shape[1:] != reference_shape:
            raise ValueError(
                f"{emotion}: layer/model shape {array.shape[1:]} does not match {reference_shape}"
            )
        means.append(array.astype(np.float64, copy=False).mean(axis=0))
    stacked = np.stack(means, axis=0)
    stacked -= stacked.mean(axis=0, keepdims=True)
    return stacked.astype(np.float32)


def fit_neutral_pca(neutral_vectors: FloatArray, variance_threshold: float) -> PCAProjection:
    """Fit a separate centered PCA per layer and retain the 50%-variance prefix."""

    neutral = np.asarray(neutral_vectors)
    if neutral.ndim != 3:
        raise ValueError(
            "neutral_vectors must have shape [sample, layer, d_model], " f"got {neutral.shape}"
        )
    if neutral.shape[0] < 1:
        raise ValueError("At least one neutral activation is required")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")

    all_components: list[npt.NDArray[np.float64]] = []
    all_ratios: list[npt.NDArray[np.float64]] = []
    counts: list[int] = []
    for layer in range(neutral.shape[1]):
        samples = neutral[:, layer, :].astype(np.float64, copy=False)
        centered = samples - samples.mean(axis=0, keepdims=True)
        if samples.shape[0] < 2 or not np.any(centered):
            all_components.append(np.empty((0, samples.shape[1]), dtype=np.float64))
            all_ratios.append(np.empty((0,), dtype=np.float64))
            counts.append(0)
            continue

        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        variances = singular_values**2 / float(samples.shape[0] - 1)
        total = float(variances.sum())
        if total <= np.finfo(np.float64).eps:
            all_components.append(np.empty((0, samples.shape[1]), dtype=np.float64))
            all_ratios.append(np.empty((0,), dtype=np.float64))
            counts.append(0)
            continue
        ratios = variances / total
        count = min(
            int(np.searchsorted(np.cumsum(ratios), variance_threshold, side="left") + 1),
            vh.shape[0],
        )
        all_components.append(np.ascontiguousarray(vh[:count]))
        all_ratios.append(ratios)
        counts.append(count)

    return PCAProjection(
        components=tuple(all_components),
        explained_variance_ratio=tuple(all_ratios),
        n_components=np.asarray(counts, dtype=np.int64),
    )


def project_out_components(
    emotion_vectors: FloatArray, components: Sequence[FloatArray]
) -> npt.NDArray[np.float32]:
    """A5: remove each layer's neutral-PC subspace from every emotion vector."""

    vectors = np.asarray(emotion_vectors)
    if vectors.ndim != 3:
        raise ValueError(
            "emotion_vectors must have shape [emotion, layer, d_model], " f"got {vectors.shape}"
        )
    if len(components) != vectors.shape[1]:
        raise ValueError(f"Received {len(components)} component sets for {vectors.shape[1]} layers")
    cleaned = vectors.astype(np.float64, copy=True)
    for layer, layer_components in enumerate(components):
        pcs = np.asarray(layer_components, dtype=np.float64)
        if pcs.size == 0:
            continue
        if pcs.ndim != 2 or pcs.shape[1] != vectors.shape[2]:
            raise ValueError(
                f"Layer {layer}: components must be [component, d_model], got {pcs.shape}"
            )
        # PCA rows are orthonormal. This is v - sum_k (v . pc_k) pc_k.
        projections = cleaned[:, layer, :] @ pcs.T
        cleaned[:, layer, :] -= projections @ pcs
    return cleaned.astype(np.float32)


def fit_and_project_neutral_pca(
    emotion_vectors: FloatArray,
    neutral_vectors: FloatArray,
    variance_threshold: float,
    *,
    device: str = "auto",
) -> tuple[npt.NDArray[np.float32], PCAProjection]:
    """Fit/project one layer at a time, using CUDA when requested and available.

    Retained PCs for all 64 default layers can occupy multiple GiB. This
    Stage-3 path discards each layer's PCs immediately after projection and
    retains only the scree diagnostics needed downstream.
    """

    raw = np.asarray(emotion_vectors, dtype=np.float32)
    neutral = np.asarray(neutral_vectors, dtype=np.float32)
    if raw.ndim != 3 or neutral.ndim != 3 or raw.shape[1:] != neutral.shape[1:]:
        raise ValueError("emotion and neutral arrays must agree on [layer, d_model]")
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("PCA device must be auto, cpu, or cuda")

    import torch

    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("PCA device cuda was requested but CUDA is unavailable")
    target = torch.device("cuda" if use_cuda else "cpu")
    cleaned = np.empty_like(raw)
    ratios_by_layer: list[npt.NDArray[np.float64]] = []
    component_counts: list[int] = []
    empty_components: list[npt.NDArray[np.float64]] = []
    for layer in range(raw.shape[1]):
        samples = torch.as_tensor(neutral[:, layer, :], dtype=torch.float32, device=target)
        centered = samples - samples.mean(dim=0, keepdim=True)
        if samples.shape[0] < 2 or not bool(torch.any(centered)):
            ratios = np.empty((0,), dtype=np.float64)
            count = 0
            cleaned[:, layer, :] = raw[:, layer, :]
        else:
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            variances = singular_values.square() / float(samples.shape[0] - 1)
            total = variances.sum()
            if not bool(total > torch.finfo(variances.dtype).eps):
                ratios = np.empty((0,), dtype=np.float64)
                count = 0
                cleaned[:, layer, :] = raw[:, layer, :]
            else:
                ratio_tensor = variances / total
                cumulative = torch.cumsum(ratio_tensor, dim=0)
                count = int(
                    torch.searchsorted(
                        cumulative,
                        torch.tensor(variance_threshold, device=target),
                        right=False,
                    ).item()
                    + 1
                )
                count = min(count, int(vh.shape[0]))
                pcs = vh[:count]
                layer_vectors = torch.as_tensor(
                    raw[:, layer, :], dtype=torch.float32, device=target
                )
                layer_cleaned = layer_vectors - (layer_vectors @ pcs.T) @ pcs
                cleaned[:, layer, :] = layer_cleaned.cpu().numpy()
                ratios = ratio_tensor.double().cpu().numpy()
        ratios_by_layer.append(ratios)
        component_counts.append(count)
        empty_components.append(np.empty((0, raw.shape[2]), dtype=np.float64))

    return cleaned, PCAProjection(
        components=tuple(empty_components),
        explained_variance_ratio=tuple(ratios_by_layer),
        n_components=np.asarray(component_counts, dtype=np.int64),
    )


def choose_primary_layer(n_layers: int, fraction: float) -> int:
    """A6, expressed as a zero-based transformer-block index."""

    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("primary layer fraction must be in [0, 1]")
    return min(n_layers - 1, max(0, round(fraction * n_layers)))


def build_vectors(config: ExperimentConfig) -> VectorBuildResult:
    """Read Stage-2 artifacts, construct A4-A6 vectors, and write Stage 3."""

    activations_dir = config.resolve_path(config.paths.activations)
    vectors_dir = config.resolve_path(config.paths.vectors)
    output_path = vectors_dir / "emotion_vectors.npz"
    neutral_path = activations_dir / "neutral.npz"
    story_paths = [activations_dir / "stories" / f"{emotion}.npz" for emotion in config.emotions]
    for path in story_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage-2 artifact: {path}")
    if not neutral_path.exists():
        raise FileNotFoundError(f"Missing Stage-2 artifact: {neutral_path}")

    input_paths = [*story_paths, neutral_path]
    cache_key = _vector_cache_key(config, input_paths)
    cached = _load_cached_result(config, output_path, cache_key)
    if cached is not None:
        if not _diagnostics_valid(config):
            _write_diagnostics(config, cached.raw, cached.denoised, cached.pca)
        LOGGER.info("resumed_emotion_vectors", extra={"output": str(output_path)})
        return cached

    # A full run contains roughly 47 GiB of Stage-2 story arrays. Compute and
    # release one per-emotion mean at a time; retaining every story array here
    # would violate the pipeline's streaming memory contract.
    emotion_means: list[npt.NDArray[np.float64]] = []
    reference_shape: tuple[int, int] | None = None
    for emotion, path in zip(config.emotions, story_paths, strict=True):
        with np.load(path, allow_pickle=False) as archive:
            if "vectors" not in archive:
                raise ValueError(f"{path} does not contain a 'vectors' array")
            vectors = _validated_float_array(archive["vectors"], path)
            if reference_shape is None:
                reference_shape = (vectors.shape[1], vectors.shape[2])
            elif vectors.shape[1:] != reference_shape:
                raise ValueError(
                    f"{emotion}: layer/model shape {vectors.shape[1:]} "
                    f"does not match {reference_shape}"
                )
            emotion_means.append(vectors.mean(axis=0, dtype=np.float64))
        del vectors

    raw64 = np.stack(emotion_means, axis=0)
    raw64 -= raw64.mean(axis=0, keepdims=True)
    raw = raw64.astype(np.float32)
    del raw64, emotion_means

    with np.load(neutral_path, allow_pickle=False) as archive:
        if "vectors" not in archive:
            raise ValueError(f"{neutral_path} does not contain a 'vectors' array")
        neutral = _validated_float_array(archive["vectors"], neutral_path)

    if neutral.shape[1:] != raw.shape[1:]:
        raise ValueError(
            f"Neutral activation shape {neutral.shape[1:]} does not match stories {raw.shape[1:]}"
        )
    denoised, pca = fit_and_project_neutral_pca(
        raw,
        neutral,
        float(config.pca_variance),
        device=str(getattr(config, "pca_device", "auto")),
    )
    primary_layer = choose_primary_layer(raw.shape[1], float(config.primary_layer_frac))

    vectors_dir.mkdir(parents=True, exist_ok=True)
    _save_vector_archive(
        output_path,
        raw=raw,
        denoised=denoised,
        emotions=config.emotions,
        primary_layer=primary_layer,
    )
    _write_diagnostics(config, raw, denoised, pca)
    pca_summary_path = vectors_dir / "pca_summary.npz"
    _write_pca_summary(pca_summary_path, pca)
    _write_cache_metadata(
        vectors_dir / "emotion_vectors.cache.json",
        cache_key,
        output_path=output_path,
        pca_summary_path=pca_summary_path,
    )
    result_pca = _pca_summary_view(pca, raw.shape[2])
    LOGGER.info(
        "built_emotion_vectors",
        extra={
            "output": str(output_path),
            "emotions": len(config.emotions),
            "layers": raw.shape[1],
            "d_model": raw.shape[2],
            "primary_layer": primary_layer,
        },
    )
    return VectorBuildResult(
        output_path=output_path,
        raw=raw,
        denoised=denoised,
        emotions=tuple(config.emotions),
        primary_layer=primary_layer,
        pca=result_pca,
    )


def _vector_cache_key(config: ExperimentConfig, inputs: Sequence[Path]) -> str:
    """Hash every Stage-3 computation input without rereading multi-GiB arrays."""

    payload = {
        "schema": 2,
        "implementation_sha256": _VECTOR_IMPLEMENTATION_HASH,
        "emotions": list(config.emotions),
        "pca_variance": float(config.pca_variance),
        "pca_device": str(getattr(config, "pca_device", "auto")),
        "primary_layer_frac": float(config.primary_layer_frac),
        "plot_format": list(config.plot_format),
        "dpi": int(config.dpi),
        "inputs": [_input_identity(path) for path in inputs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_identity(path: Path) -> dict[str, object]:
    """Use Stage-2's semantic cache key, or hash an external artifact directly."""

    sidecar = path.with_suffix(".cache.json")
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") == 1
                and isinstance(payload.get("cache_key"), str)
                and isinstance(payload.get("archive_sha256"), str)
                and len(payload["archive_sha256"]) == 64
            ):
                actual_digest = _sha256_path(path)
                if actual_digest != payload["archive_sha256"]:
                    raise ValueError(f"Stage-2 artifact digest does not match its sidecar: {path}")
                return {
                    "path": str(path.resolve()),
                    "stage2_cache_key": payload["cache_key"],
                    "archive_sha256": actual_digest,
                }
        except (OSError, TypeError, json.JSONDecodeError):
            pass
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _pca_summary_view(pca: PCAProjection, d_model: int) -> PCAProjection:
    """Drop potentially multi-GiB PCs after projection while retaining diagnostics."""

    return PCAProjection(
        components=tuple(
            np.empty((0, d_model), dtype=np.float64) for _ in pca.explained_variance_ratio
        ),
        explained_variance_ratio=tuple(ratios.copy() for ratios in pca.explained_variance_ratio),
        n_components=pca.n_components.copy(),
    )


def _load_cached_result(
    config: ExperimentConfig, output_path: Path, cache_key: str
) -> VectorBuildResult | None:
    if not bool(config.resume) or not output_path.is_file():
        return None
    metadata_path = output_path.parent / "emotion_vectors.cache.json"
    summary_path = output_path.parent / "pca_summary.npz"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != 2
            or metadata.get("cache_key") != cache_key
            or metadata.get("output_sha256") != _sha256_path(output_path)
            or metadata.get("pca_summary_sha256") != _sha256_path(summary_path)
        ):
            return None
        with np.load(output_path, allow_pickle=False) as archive:
            if set(archive.files) != {"raw", "denoised", "emotions", "primary_layer"}:
                return None
            raw = np.asarray(archive["raw"])
            denoised = np.asarray(archive["denoised"])
            emotions = tuple(str(value) for value in archive["emotions"].tolist())
            primary_layer = int(np.asarray(archive["primary_layer"]).reshape(()))
        if (
            raw.ndim != 3
            or denoised.shape != raw.shape
            or raw.dtype != np.float32
            or denoised.dtype != np.float32
            or not np.isfinite(raw).all()
            or not np.isfinite(denoised).all()
            or emotions != tuple(config.emotions)
            or not 0 <= primary_layer < raw.shape[1]
        ):
            return None
        with np.load(summary_path, allow_pickle=False) as summary:
            counts = np.asarray(summary["n_components"], dtype=np.int64)
            ratios = np.asarray(summary["explained_variance_ratio"], dtype=np.float64)
            lengths = np.asarray(summary["ratio_lengths"], dtype=np.int64)
        if counts.shape != (raw.shape[1],) or lengths.shape != counts.shape:
            return None
        ratio_tuple = tuple(ratios[layer, : lengths[layer]].copy() for layer in range(raw.shape[1]))
        pca = PCAProjection(
            components=tuple(
                np.empty((0, raw.shape[2]), dtype=np.float64) for _ in range(raw.shape[1])
            ),
            explained_variance_ratio=ratio_tuple,
            n_components=counts,
        )
        return VectorBuildResult(
            output_path=output_path,
            raw=raw,
            denoised=denoised,
            emotions=emotions,
            primary_layer=primary_layer,
            pca=pca,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_pca_summary(path: Path, pca: PCAProjection) -> None:
    lengths = np.asarray([ratios.size for ratios in pca.explained_variance_ratio], dtype=np.int64)
    width = int(lengths.max(initial=0))
    padded = np.zeros((len(lengths), width), dtype=np.float64)
    for layer, ratios in enumerate(pca.explained_variance_ratio):
        padded[layer, : ratios.size] = ratios
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(
                handle,
                n_components=pca.n_components,
                explained_variance_ratio=padded,
                ratio_lengths=lengths,
            )
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _write_cache_metadata(
    path: Path,
    cache_key: str,
    *,
    output_path: Path,
    pca_summary_path: Path,
) -> None:
    payload = {
        "schema_version": 2,
        "cache_key": cache_key,
        "output_sha256": _sha256_path(output_path),
        "pca_summary_sha256": _sha256_path(pca_summary_path),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostics_valid(config: ExperimentConfig) -> bool:
    output_dir = config.resolve_path(config.paths.outputs) / "diagnostics"
    for stem in ("neutral_pca_scree", "raw_vs_denoised_cosine"):
        if not (output_dir / f"{stem}.csv").is_file():
            return False
        if any(
            not (output_dir / f"{stem}.{file_format}").is_file()
            for file_format in config.plot_format
        ):
            return False
    return True


def _validated_float_array(array: npt.NDArray[np.generic], path: Path) -> npt.NDArray[np.float32]:
    if array.ndim != 3:
        raise ValueError(f"{path}: vectors must be [sample, layer, d_model], got {array.shape}")
    if array.shape[0] == 0 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path}: vectors must be a non-empty floating-point array")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: vectors contain NaN or infinity")
    return np.asarray(array, dtype=np.float32)


def _save_vector_archive(
    path: Path,
    *,
    raw: npt.NDArray[np.float32],
    denoised: npt.NDArray[np.float32],
    emotions: Sequence[str],
    primary_layer: int,
) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                raw=np.asarray(raw, dtype=np.float32),
                denoised=np.asarray(denoised, dtype=np.float32),
                emotions=np.asarray(emotions, dtype=np.str_),
                primary_layer=np.asarray(primary_layer, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _cosine_rows(left: npt.NDArray[np.float32], right: npt.NDArray[np.float32]) -> np.ndarray:
    numerator = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _write_diagnostics(
    config: ExperimentConfig,
    raw: npt.NDArray[np.float32],
    denoised: npt.NDArray[np.float32],
    pca: PCAProjection,
) -> None:
    """Write the two diagnostics that are defined by Stage 3."""

    import matplotlib.pyplot as plt

    from .plotting import save_figure

    output_dir = config.resolve_path(config.paths.outputs) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    scree_rows: list[dict[str, object]] = []
    for layer, ratios in enumerate(pca.explained_variance_ratio):
        cumulative = np.cumsum(ratios)
        for component, (ratio, cumulative_ratio) in enumerate(
            zip(ratios, cumulative, strict=True), 1
        ):
            scree_rows.append(
                {
                    "layer": layer,
                    "component": component,
                    "explained_variance_ratio": float(ratio),
                    "cumulative_variance": float(cumulative_ratio),
                    "retained": component <= int(pca.n_components[layer]),
                }
            )
    _write_csv(output_dir / "neutral_pca_scree.csv", scree_rows)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for _layer, ratios in enumerate(pca.explained_variance_ratio):
        if ratios.size:
            axis.plot(np.arange(1, ratios.size + 1), np.cumsum(ratios), alpha=0.16, color="C0")
    axis.axhline(float(config.pca_variance), color="C3", linestyle="--", label="cutoff")
    axis.set(xlabel="Principal component", ylabel="Cumulative explained variance", ylim=(0, 1.01))
    axis.legend()
    save_figure(figure, output_dir / "neutral_pca_scree", config)
    plt.close(figure)

    similarities = _cosine_rows(raw, denoised)
    similarity_rows = [
        {"emotion": emotion, "layer": layer, "cosine_similarity": float(similarities[index, layer])}
        for index, emotion in enumerate(config.emotions)
        for layer in range(raw.shape[1])
    ]
    _write_csv(output_dir / "raw_vs_denoised_cosine.csv", similarity_rows)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(np.arange(raw.shape[1]), similarities.mean(axis=0), color="C0")
    axis.fill_between(
        np.arange(raw.shape[1]),
        np.quantile(similarities, 0.1, axis=0),
        np.quantile(similarities, 0.9, axis=0),
        color="C0",
        alpha=0.2,
        label="10th-90th percentile",
    )
    axis.set(xlabel="Layer", ylabel="cosine(raw, denoised)", ylim=(-1.0, 1.0))
    axis.legend()
    save_figure(figure, output_dir / "raw_vs_denoised_cosine", config)
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
