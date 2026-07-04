"""Computational verification suite for extracted emotion vectors.

The six verification modules expose array-first APIs. A caller may therefore use
cached activations, a light-weight test double, or a live model bundle without
any verifier loading a model or a dataset on import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

MetricValue: TypeAlias = bool | int | float | str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Files and headline metrics emitted by one verification."""

    name: str
    passed: bool
    output_dir: Path
    report: Path
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    metrics: Mapping[str, MetricValue]


@dataclass(frozen=True, slots=True)
class EmotionVectorArtifact:
    """Primary-layer view of the Stage-3 vector artifact."""

    emotions: tuple[str, ...]
    vectors: np.ndarray
    primary_layer: int
    all_layers: np.ndarray | None = None


def load_emotion_vectors(
    path: str | Path,
    *,
    kind: str = "denoised",
    layer: int | None = None,
) -> EmotionVectorArtifact:
    """Load and validate ``emotion_vectors.npz`` from the Stage-3 contract."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        missing = {kind, "emotions", "primary_layer"}.difference(data.files)
        if missing:
            raise ValueError(f"{source} is missing keys: {sorted(missing)}")
        all_layers = np.asarray(data[kind], dtype=np.float32)
        emotions = tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in np.asarray(data["emotions"]).tolist()
        )
        primary_values = np.asarray(data["primary_layer"]).reshape(-1)
        if primary_values.size != 1:
            raise ValueError("primary_layer must contain exactly one integer")
        primary = int(primary_values[0])

    if all_layers.ndim != 3:
        raise ValueError(
            f"{kind!r} must have shape [n_emotions, n_layers, d_model], " f"got {all_layers.shape}"
        )
    if len(emotions) != all_layers.shape[0]:
        raise ValueError("emotion labels do not match the vector array")
    chosen = primary if layer is None else int(layer)
    if not 0 <= chosen < all_layers.shape[1]:
        raise IndexError(f"layer {chosen} is outside [0, {all_layers.shape[1]})")
    return EmotionVectorArtifact(
        emotions=emotions,
        vectors=np.array(all_layers[:, chosen, :], copy=True),
        primary_layer=chosen,
        all_layers=np.array(all_layers, copy=True),
    )


def _as_numpy(value: Any, *, dtype: np.dtype[Any] | type[Any] | None = None) -> np.ndarray:
    """Convert numpy/torch-like data without importing torch."""

    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        try:
            candidate = candidate.numpy()
        except (RuntimeError, TypeError):
            # NumPy has no bfloat16 dtype; torch tensors can be safely widened
            # for verification calculations after leaving the accelerator.
            if not hasattr(candidate, "float"):
                raise
            candidate = candidate.float().numpy()
    return np.asarray(candidate, dtype=dtype)


def _config_value(config: object | None, key: str, default: Any = None) -> Any:
    """Read a dotted value from either a mapping or a dataclass-like config."""

    if config is None:
        return default
    value: Any = config
    for part in key.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return default
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return default
    return value


def _resolve_output_dir(
    output_dir: str | Path | None,
    config: object | None,
    leaf: str,
) -> Path:
    if output_dir is not None:
        destination = Path(output_dir)
    else:
        root = _config_value(config, "paths.outputs")
        if root is None:
            raise ValueError("output_dir is required when config does not define paths.outputs")
        destination = Path(root) / leaf
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _plot_options(
    config: object | None,
    formats: Sequence[str] | None,
    dpi: int | None,
) -> tuple[tuple[str, ...], int]:
    from ..plotting import apply_plot_style

    apply_plot_style()
    raw_formats = (
        formats if formats is not None else _config_value(config, "plot_format", ("png", "svg"))
    )
    if isinstance(raw_formats, str):
        raw_formats = (raw_formats,)
    cleaned = tuple(dict.fromkeys(str(item).lower().lstrip(".") for item in raw_formats))
    unsupported = set(cleaned).difference({"png", "svg"})
    if unsupported:
        raise ValueError(f"unsupported plot formats: {sorted(unsupported)}")
    if not cleaned:
        raise ValueError("at least one plot format is required")
    resolved_dpi = int(dpi if dpi is not None else _config_value(config, "dpi", 150))
    return cleaned, resolved_dpi


def _save_figure(
    figure: Any,
    stem: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    from ..plotting import save_figure

    return save_figure(figure, stem, formats, dpi)


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]] | Any,
) -> Path:
    """Write a deterministic RFC-4180 table without requiring pandas."""

    from ..plotting import save_csv

    normalized_rows = ({name: row.get(name, "") for name in fieldnames} for row in rows)
    return save_csv(path, normalized_rows, fieldnames)


def _write_report(
    path: Path,
    *,
    title: str,
    passed: bool,
    summary: Sequence[str],
    figures: Sequence[Path] = (),
    tables: Sequence[Path] = (),
) -> Path:
    verdict = "PASS" if passed else "FAIL"
    lines = [f"# {title}", "", f"VERDICT: **{verdict}**", ""]
    lines.extend(summary)
    if figures:
        lines.extend(("", "## Figures", ""))
        selected: dict[str, Path] = {}
        for figure in figures:
            key = str(figure.with_suffix(""))
            if key not in selected or figure.suffix.lower() == ".png":
                selected[key] = figure
        for figure in selected.values():
            lines.append(f"![{figure.stem}]({figure.name})")
    if tables:
        lines.extend(("", "## Source tables", ""))
        for table in tables:
            lines.append(f"- [{table.name}]({table.name})")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_manifest_if_supported(
    output_dir: Path,
    config: object | None,
    stage: str,
    counts: Mapping[str, MetricValue],
) -> Path | None:
    """Write the canonical manifest for real configs, tolerate tiny test configs."""

    if config is None or not all(
        hasattr(config, attribute)
        for attribute in (
            "as_dict",
            "config_hash",
            "project_root",
            "resolve_path",
            "seed",
        )
    ):
        return None
    from ..logging_utils import write_run_manifest

    return write_run_manifest(output_dir, config, stage, counts=counts)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "item"


def _coerce_vector_mapping(
    vectors: Mapping[str, Any] | np.ndarray | EmotionVectorArtifact,
    emotions: Sequence[str] | None = None,
    *,
    layer: int | None = None,
) -> dict[str, np.ndarray]:
    """Return label -> one-dimensional vector with strict shape validation."""

    if isinstance(vectors, EmotionVectorArtifact):
        labels = vectors.emotions
        array = vectors.vectors if layer is None else vectors.all_layers
        if array is None:
            raise ValueError("the artifact does not retain all layers")
        if layer is not None:
            array = array[:, int(layer), :]
        return _coerce_vector_mapping(array, labels)

    if isinstance(vectors, Mapping):
        result = {
            str(label): _as_numpy(vector, dtype=np.float64).reshape(-1)
            for label, vector in vectors.items()
        }
    else:
        array = _as_numpy(vectors, dtype=np.float64)
        if array.ndim == 3:
            if layer is None:
                raise ValueError("layer is required for a [emotion, layer, d_model] array")
            array = array[:, int(layer), :]
        if array.ndim != 2:
            raise ValueError(f"vectors must be rank two after layer selection, got {array.shape}")
        if emotions is None:
            raise ValueError("emotions are required when vectors is an array")
        if len(emotions) != array.shape[0]:
            raise ValueError("emotion labels do not match vector rows")
        result = {
            str(label): np.array(array[index], dtype=np.float64, copy=True)
            for index, label in enumerate(emotions)
        }

    if not result:
        raise ValueError("at least one emotion vector is required")
    dimensions = {vector.shape for vector in result.values()}
    if len(dimensions) != 1 or next(iter(dimensions))[0] == 0:
        raise ValueError("all emotion vectors must be non-empty and share one dimension")
    for label, vector in result.items():
        if not np.isfinite(vector).all():
            raise ValueError(f"vector for {label!r} contains a non-finite value")
        if float(np.linalg.norm(vector)) <= np.finfo(np.float64).eps:
            raise ValueError(f"vector for {label!r} has zero norm")
    return result


def _iter_training_stories(
    raw_root: str | Path,
    emotions: Sequence[str],
) -> Any:
    """Stream Stage-1 stories for V1 without accumulating their text in memory."""

    story_root = Path(raw_root) / "stories"
    for emotion in emotions:
        path = story_root / f"{emotion}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"V1 story artifact does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                if not isinstance(row, Mapping) or not isinstance(row.get("text"), str):
                    raise ValueError(f"invalid Stage-1 story schema at {path}:{line_number}")
                label = str(row.get("emotion", emotion))
                if label != emotion:
                    raise ValueError(
                        f"story label {label!r} does not match shard {emotion!r} "
                        f"at {path}:{line_number}"
                    )
                yield {
                    "emotion": emotion,
                    "story_id": f"{row.get('topic_id', 'unknown')}-{row.get('story_idx', line_number - 1)}",
                    "text": row["text"],
                }


def _iter_heldout_from_config(config: object) -> Any:
    """Adapt the central streaming dataset loader to V2's input contract."""

    from ..datasets import iter_heldout_documents

    for document in iter_heldout_documents(config):
        yield {
            "dataset": document.dataset,
            "document_id": document.document_id,
            "text": document.text,
        }


def _with_batched_token_activations(
    records: Any,
    model_bundle: object,
    *,
    batch_size: int,
    max_length: int | None = None,
) -> Any:
    """Attach token activations while preserving streaming at batch granularity."""

    if batch_size <= 0:
        raise ValueError("verification batch_size must be positive")
    batch_method = getattr(model_bundle, "capture_token_activations_batch", None)
    single_method = getattr(model_bundle, "capture_token_activations", None)
    if not callable(batch_method) and not callable(single_method):
        raise TypeError("model bundle does not expose token-activation capture")

    batch: list[Mapping[str, Any]] = []

    def flush(items: list[Mapping[str, Any]]) -> Any:
        texts = [str(item["text"]) for item in items]
        if callable(batch_method):
            captures = batch_method(texts, max_length=max_length)
        else:
            captures = [single_method(text) for text in texts]
        if len(captures) != len(items):
            raise RuntimeError("token-activation capture returned the wrong batch size")
        for item, capture in zip(items, captures, strict=True):
            if not isinstance(capture, tuple) or len(capture) != 2:
                raise TypeError("token-activation capture must return (tokens, activations)")
            tokens, activations = capture
            yield {**item, "tokens": tokens, "activations": activations}

    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield from flush(batch)
            batch = []
    if batch:
        yield from flush(batch)


def _with_batched_token_projections(
    records: Any,
    model_bundle: object,
    vectors: np.ndarray,
    *,
    batch_size: int,
    max_length: int,
) -> Any:
    """Project V2 batches in the layer hook and retain no raw activations."""

    method = getattr(model_bundle, "capture_token_projections_batch", None)
    if not callable(method):
        yield from _with_batched_token_activations(
            records,
            model_bundle,
            batch_size=batch_size,
            max_length=max_length,
        )
        return
    batch: list[Mapping[str, Any]] = []

    def flush(items: list[Mapping[str, Any]]) -> Any:
        captures = method(
            [str(item["text"]) for item in items],
            vectors,
            max_length=max_length,
        )
        if len(captures) != len(items):
            raise RuntimeError("token-projection capture returned the wrong batch size")
        for item, capture in zip(items, captures, strict=True):
            tokens, projections = capture
            yield {**item, "tokens": tokens, "projections": projections}

    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield from flush(batch)
            batch = []
    if batch:
        yield from flush(batch)


def _topic_content_terms(config: object) -> tuple[str, ...]:
    """Derive salient topic bigrams for V6's story-content sanity check."""

    stop = {
        "a",
        "an",
        "the",
        "their",
        "they",
        "them",
        "someone",
        "someones",
        "person",
        "persons",
        "two",
        "out",
        "finds",
        "find",
        "discovers",
        "learns",
        "receives",
        "gets",
        "has",
        "have",
        "is",
        "was",
        "are",
        "to",
        "of",
        "for",
        "in",
        "on",
        "from",
        "with",
        "that",
    }
    terms: set[str] = set()
    for topic in _config_value(config, "topics", ()):
        words = [word for word in re.findall(r"[a-z]+", str(topic).lower()) if word not in stop]
        terms.update(" ".join(words[index : index + 2]) for index in range(len(words) - 1))
    return tuple(sorted(term for term in terms if len(term) >= 8))


def _file_identity(path: Path) -> Mapping[str, object]:
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": _sha256_path(path),
    }
    sidecar = path.with_suffix(".cache.json")
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(payload.get("cache_key"), str):
                identity["upstream_cache_key"] = payload["cache_key"]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return identity


def _verification_cache_key(
    key: str,
    config: object,
    vector_path: Path | None,
) -> str:
    config_snapshot = (
        config.as_dict(include_assets=True)
        if hasattr(config, "as_dict")
        else {
            name: _config_value(config, name)
            for name in (
                "model_name",
                "model_revision",
                "emotions",
                "activation_percentile",
                "heldout_datasets",
                "heldout_revisions",
                "heldout_max_docs",
                "steering_strength",
                "kmeans_k",
                "seed",
                "plot_format",
                "dpi",
            )
        }
    )
    input_identities: list[Mapping[str, object]] = []
    if vector_path is not None and vector_path.is_file():
        input_identities.append(_file_identity(vector_path))
    if key == "V1":
        raw_root = _config_value(config, "paths.raw")
        for emotion in _config_value(config, "emotions", ()):
            path = Path(raw_root) / "stories" / f"{emotion}.jsonl"
            if path.is_file():
                input_identities.append(_file_identity(path))
    if key == "V6":
        activation_root = _config_value(config, "paths.activations")
        if activation_root is not None:
            path = Path(activation_root) / "layer_norms.npy"
            if path.is_file():
                input_identities.append(_file_identity(path))
    payload = {
        "schema": 2,
        "verification": key,
        "implementation_sha256": _verification_implementation_hash(key),
        "config": config_snapshot,
        "inputs": input_identities,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verification_implementation_hash(key: str) -> str:
    module_names = {
        "V1": "localization.py",
        "V2": "held_out.py",
        "V3": "logit_lens.py",
        "V4": "clustering.py",
        "V5": "pca_structure.py",
        "V6": "steering_probe.py",
    }
    verify_root = Path(__file__).parent
    package_root = verify_root.parent
    paths = [Path(__file__), verify_root / module_names[key], package_root / "plotting.py"]
    if key in {"V1", "V2", "V3", "V6"}:
        paths.append(package_root / "model.py")
    if key == "V2":
        paths.append(package_root / "kernels" / "project_threshold.py")
        paths.append(package_root / "datasets.py")
    if key == "V6":
        paths.append(package_root / "steering.py")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_cached_verification(
    output_dir: Path,
    key: str,
    cache_key: str,
    config: object,
) -> VerificationResult | None:
    metadata_path = output_dir / "verification.cache.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 2
            or payload.get("verification") != key
            or payload.get("cache_key") != cache_key
            or not isinstance(payload.get("passed"), bool)
        ):
            return None

        artifact_records = payload.get("artifacts")
        if not isinstance(artifact_records, list) or not artifact_records:
            return None
        validated: dict[str, Path] = {}
        for record in artifact_records:
            if not isinstance(record, Mapping):
                return None
            relative = record.get("path")
            expected_size = record.get("size")
            expected_digest = record.get("sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or not isinstance(expected_size, int)
                or expected_size < 0
                or not isinstance(expected_digest, str)
                or len(expected_digest) != 64
            ):
                return None
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return None
            path = output_dir / relative_path
            if (
                not path.is_file()
                or path.stat().st_size != expected_size
                or _sha256_path(path) != expected_digest
            ):
                return None
            validated[relative_path.as_posix()] = path

        report_name = payload.get("report")
        table_names = payload.get("tables")
        figure_names = payload.get("figures")
        if (
            not isinstance(report_name, str)
            or not isinstance(table_names, list)
            or not table_names
            or not all(isinstance(name, str) for name in table_names)
            or not isinstance(figure_names, list)
            or not figure_names
            or not all(isinstance(name, str) for name in figure_names)
        ):
            return None
        report = validated.get(report_name)
        tables = tuple(validated[name] for name in table_names if name in validated)
        figures = tuple(validated[name] for name in figure_names if name in validated)
        if report is None or len(tables) != len(table_names) or len(figures) != len(figure_names):
            return None

        verdict_match = re.search(
            r"^VERDICT: \*\*(PASS|FAIL)\*\*$",
            report.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        expected_verdict = "PASS" if payload["passed"] else "FAIL"
        if verdict_match is None or verdict_match.group(1) != expected_verdict:
            return None

        required_formats = tuple(_config_value(config, "plot_format", ("png", "svg")))
        figure_set = set(figures)
        figure_stems = {figure.with_suffix("") for figure in figures}
        table_set = set(tables)
        if any(
            stem.with_suffix(f".{fmt}") not in figure_set
            for stem in figure_stems
            for fmt in required_formats
        ) or any(stem.with_suffix(".csv") not in table_set for stem in figure_stems):
            return None
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        return VerificationResult(
            name=str(payload.get("name", key)),
            passed=payload["passed"],
            output_dir=output_dir,
            report=report,
            tables=tables,
            figures=figures,
            metrics={**metrics, "resumed": True},
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_verification_cache(
    result: VerificationResult,
    key: str,
    cache_key: str,
) -> None:
    unique_artifacts = tuple(dict.fromkeys((result.report, *result.tables, *result.figures)))
    artifact_records = [
        {
            "path": path.relative_to(result.output_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_path(path),
        }
        for path in unique_artifacts
    ]
    payload = {
        "schema_version": 2,
        "verification": key,
        "cache_key": cache_key,
        "name": result.name,
        "passed": result.passed,
        "metrics": _strict_json_value(dict(result.metrics)),
        "report": result.report.relative_to(result.output_dir).as_posix(),
        "tables": [path.relative_to(result.output_dir).as_posix() for path in result.tables],
        "figures": [path.relative_to(result.output_dir).as_posix() for path in result.figures],
        "artifacts": artifact_records,
    }
    path = result.output_dir / "verification.cache.json"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
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


def _strict_json_value(value: Any) -> Any:
    """Return a JSON-safe value without non-standard NaN/Infinity literals."""

    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strict_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _strict_json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Path):
        return str(value)
    return value


def run_verifications(
    config: object,
    model_bundle: object | None = None,
    checks: Sequence[str] | None = None,
    *,
    inputs: Mapping[str, Mapping[str, Any]] | None = None,
    output_root: str | Path | None = None,
) -> dict[str, VerificationResult]:
    """Run selected verifications using explicit keyword bundles per verifier.

    ``inputs`` is keyed by ``V1``...``V6`` (descriptive names are also
    accepted). The dispatcher injects the config and, for model-aware checks,
    ``model_bundle``. It also loads the Stage-3 vector artifact when its standard
    configured path exists and the caller did not provide vectors explicitly.
    """

    from .clustering import run_clustering
    from .held_out import run_held_out
    from .localization import run_localization
    from .logit_lens import run_logit_lens
    from .pca_structure import run_pca_structure
    from .steering_probe import run_steering_probe

    registry = {
        "V1": ("localization", "V1_localization", run_localization),
        "V2": ("held_out", "V2_held_out", run_held_out),
        "V3": ("logit_lens", "V3_logit_lens", run_logit_lens),
        "V4": ("clustering", "V4_clustering", run_clustering),
        "V5": ("pca_structure", "V5_pca_structure", run_pca_structure),
        "V6": ("steering_probe", "V6_steering_probe", run_steering_probe),
    }
    aliases = {name: key for key, (name, _, _) in registry.items()}
    requested = (checks,) if isinstance(checks, str) else tuple(checks or registry.keys())
    provided_inputs = inputs or {}
    root = (
        Path(output_root)
        if output_root is not None
        else (
            Path(_config_value(config, "paths.outputs"))
            if _config_value(config, "paths.outputs") is not None
            else None
        )
    )
    configured_vectors = _config_value(config, "paths.vectors")
    vector_artifact: EmotionVectorArtifact | None = None
    if configured_vectors is not None:
        vector_path = Path(configured_vectors)
        if vector_path.is_dir() or vector_path.suffix != ".npz":
            vector_path = vector_path / "emotion_vectors.npz"
        if vector_path.exists():
            vector_artifact = load_emotion_vectors(vector_path)
    else:
        vector_path = None
    results: dict[str, VerificationResult] = {}
    for requested_name in requested:
        key = aliases.get(requested_name.lower(), requested_name.upper())
        if key not in registry:
            raise KeyError(f"unknown verification {requested_name!r}")
        descriptive, leaf, runner = registry[key]
        kwargs = dict(provided_inputs.get(key, provided_inputs.get(descriptive, {})))
        kwargs.setdefault("config", config)
        if root is not None:
            kwargs.setdefault("output_dir", root / leaf)
        if vector_artifact is not None:
            kwargs.setdefault("emotion_vectors", vector_artifact)
        if model_bundle is not None and key in {"V1", "V2", "V3", "V6"}:
            kwargs.setdefault("model_bundle", model_bundle)
        if key == "V6" and "layer_norms" not in kwargs:
            activation_root = _config_value(config, "paths.activations")
            if activation_root is not None:
                norm_path = Path(activation_root) / "layer_norms.npy"
                if norm_path.exists():
                    kwargs["layer_norms"] = np.load(norm_path, allow_pickle=False)
        if key == "V6":
            kwargs.setdefault("story_content_terms", _topic_content_terms(config))
        if key == "V1" and "stories" not in kwargs:
            raw_root = _config_value(config, "paths.raw")
            if raw_root is None:
                raise ValueError("V1 requires inputs.stories or config.paths.raw")
            labels = (
                vector_artifact.emotions
                if vector_artifact is not None
                else tuple(_config_value(config, "emotions", ()))
            )
            stories = _iter_training_stories(raw_root, labels)
            kwargs["stories"] = (
                _with_batched_token_activations(
                    stories,
                    model_bundle,
                    batch_size=int(_config_value(config, "batch_size", 1)),
                )
                if model_bundle is not None
                else stories
            )
        if key == "V2" and "documents" not in kwargs:
            documents = _iter_heldout_from_config(config)
            kwargs["documents"] = (
                _with_batched_token_projections(
                    documents,
                    model_bundle,
                    vector_artifact.vectors,
                    batch_size=int(_config_value(config, "batch_size", 1)),
                    max_length=int(_config_value(config, "heldout_max_tokens", 512)),
                )
                if model_bundle is not None and vector_artifact is not None
                else documents
            )
        has_explicit_inputs = key in provided_inputs or descriptive in provided_inputs
        cache_key = _verification_cache_key(key, config, vector_path)
        cached = (
            _load_cached_verification(Path(kwargs["output_dir"]), key, cache_key, config)
            if bool(_config_value(config, "resume", False)) and not has_explicit_inputs
            else None
        )
        if cached is not None:
            results[key] = cached
            continue
        if model_bundle is None and not has_explicit_inputs and key in {"V1", "V2", "V3", "V6"}:
            from ..model import load_model

            model_bundle = load_model(config)
            kwargs.setdefault("model_bundle", model_bundle)
            if key == "V1":
                kwargs["stories"] = _with_batched_token_activations(
                    kwargs["stories"],
                    model_bundle,
                    batch_size=int(_config_value(config, "batch_size", 1)),
                )
            if key == "V2":
                if vector_artifact is None:
                    raise FileNotFoundError("V2 requires the Stage-3 emotion-vector artifact")
                kwargs["documents"] = _with_batched_token_projections(
                    kwargs["documents"],
                    model_bundle,
                    vector_artifact.vectors,
                    batch_size=int(_config_value(config, "batch_size", 1)),
                    max_length=int(_config_value(config, "heldout_max_tokens", 512)),
                )
        result = runner(**kwargs)
        if not has_explicit_inputs:
            _write_verification_cache(result, key, cache_key)
        results[key] = result
    return results


from .clustering import ClusteringResult, run_clustering  # noqa: E402
from .held_out import HeldOutDocument, run_held_out  # noqa: E402
from .localization import LocalizationStory, run_localization  # noqa: E402
from .logit_lens import LogitLensResult, run_logit_lens  # noqa: E402
from .pca_structure import PCAStructureResult, run_pca_structure  # noqa: E402
from .steering_probe import (  # noqa: E402
    PROBE_HE,
    PROBE_I,
    ProbeEvaluation,
    SteeringProbeResult,
    run_steering_probe,
)

__all__ = [
    "PROBE_HE",
    "PROBE_I",
    "ClusteringResult",
    "EmotionVectorArtifact",
    "HeldOutDocument",
    "LocalizationStory",
    "LogitLensResult",
    "PCAStructureResult",
    "ProbeEvaluation",
    "SteeringProbeResult",
    "VerificationResult",
    "load_emotion_vectors",
    "run_clustering",
    "run_held_out",
    "run_localization",
    "run_logit_lens",
    "run_pca_structure",
    "run_steering_probe",
    "run_verifications",
]
