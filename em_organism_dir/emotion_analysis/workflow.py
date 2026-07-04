"""Strict disk orchestration for the reduced RQ1 analysis stage.

This module is the boundary between versioned disk artifacts and the pure
in-memory analysis.  It never substitutes evidence: a missing normalized EM
artifact produces a durable blocked summary and no metric rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import RQ1AnalysisResult, run_rq1_analysis
from .config import RQ1Config
from .data import ReuseDataset, build_reuse_dataset, sha256_file
from .em_direction import (
    CANONICAL_ADAPTER_RANK,
    CANONICAL_DEFINITION,
    CANONICAL_DIRECTION_SIGN,
    CANONICAL_SOURCE_SCRIPT,
    CANONICAL_STORAGE_DTYPE,
    CANONICAL_TARGET_ID,
    CANONICAL_TOKEN_AGGREGATION,
    SUPPORTED_SUBTRACTION_DTYPES,
)
from .quality import ReliabilityGate, evaluate_reliability_gate
from .vectors import EmotionDirectionSet, build_emotion_directions, fit_neutral_pca

WORKFLOW_SCHEMA_VERSION = 1
DIRECTION_ARTIFACT_SCHEMA_VERSION = 1
NORMALIZED_EM_FILENAME = "em_direction.normalized.npz"

METRIC_COLUMNS = (
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

_NUMERIC_COLUMNS = (
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


class RQ1WorkflowError(RuntimeError):
    """Raised when disk artifacts violate the RQ1 workflow contract."""


class MissingNormalizedEMArtifact(RQ1WorkflowError):
    """The normalized EM artifact is absent; the written summary is blocked."""

    def __init__(self, message: str, summary_path: Path):
        super().__init__(message)
        self.summary_path = summary_path


@dataclass(frozen=True, slots=True)
class WorkflowInputs:
    dataset: ReuseDataset
    aligned_by_emotion: Mapping[str, np.ndarray[Any, Any]]
    misaligned_by_emotion: Mapping[str, np.ndarray[Any, Any]]
    aligned_neutral: np.ndarray[Any, Any]
    misaligned_neutral: np.ndarray[Any, Any]
    topic_ids: tuple[int, ...]
    block_ids: tuple[tuple[int, int], ...]
    em_direction: np.ndarray[Any, Any] | None
    em_metadata: Mapping[str, Any]
    em_path: Path
    em_sha256: str | None
    misaligned_manifest_path: Path
    misaligned_manifest_sha256: str
    activation_hashes: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    status: str
    conclusion: str
    summary_path: Path
    metrics_csv: Path | None
    metrics_parquet: Path | None
    direction_artifacts: tuple[Path, ...]
    analysis: RQ1AnalysisResult | None
    blocker: str | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RQ1WorkflowError(f"cannot read {context} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RQ1WorkflowError(f"invalid JSON in {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RQ1WorkflowError(f"{context} {path} must contain a JSON object")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
    _atomic_write_text(path, f"{text}\n")


def _expected_model(config: RQ1Config) -> dict[str, Any]:
    return {
        "model_source": "misaligned",
        "model_id": config.models.misaligned_adapter_id,
        "model_revision": config.models.misaligned_adapter_revision,
        "base_model_id": config.models.audited_parent_model_id,
        "base_model_revision": config.models.audited_parent_revision,
        "tokenizer_id": config.models.tokenizer_id,
        "tokenizer_revision": config.models.tokenizer_revision,
        "adapter_id": config.models.misaligned_adapter_id,
        "adapter_revision": config.models.misaligned_adapter_revision,
        "adapter_merged": False,
        "extraction_dtype": config.analysis.extraction_dtype,
        "attention_backend": config.analysis.attention_implementation,
        "activation_site": config.analysis.activation_site,
        "n_layers": len(config.analysis.layers),
        "hidden_size": config.analysis.hidden_size,
    }


def _dataset_source_hash(config: RQ1Config) -> str:
    sources = (
        *(
            (emotion, config.paths.raw_dir / "stories" / f"{emotion}.jsonl")
            for emotion in config.emotions.reference
        ),
        ("neutral", config.paths.raw_dir / "neutral.jsonl"),
    )
    payload: dict[str, Any] = {}
    for name, path in sources:
        if not path.is_file():
            raise RQ1WorkflowError(f"dataset source is missing: {path}")
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, json.JSONDecodeError) as exc:
            raise RQ1WorkflowError(f"dataset source is invalid: {path}") from exc
        payload[name] = {
            "relative_name": path.name,
            "sha256": sha256_file(path),
            "records": len(records),
        }
    return _json_hash(payload)


def _validate_dataset(config: RQ1Config, dataset: ReuseDataset) -> None:
    if dataset.config != config:
        raise RQ1WorkflowError("reuse dataset was built from a different RQ1 configuration")
    if (len(dataset.rows), len(dataset.same_example_rows), len(dataset.confirmatory_rows)) != (
        960,
        959,
        948,
    ):
        raise RQ1WorkflowError(
            "reuse dataset must contain 960/959/948 raw/sensitivity/confirmatory rows"
        )
    counts = Counter(row.emotion for row in dataset.confirmatory_rows)
    if tuple(counts.get(emotion, 0) for emotion in config.emotions.reference) != (79,) * 12:
        raise RQ1WorkflowError("confirmatory dataset must contain exactly 79 rows per emotion")
    if dataset.manifest.get("counts", {}).get("confirmatory_rows") != 948:
        raise RQ1WorkflowError("reuse manifest confirmatory count mismatch")
    actual_index_hash = hashlib.sha256(dataset.index_jsonl().encode("utf-8")).hexdigest()
    if dataset.manifest.get("index", {}).get("sha256") != actual_index_hash:
        raise RQ1WorkflowError("reuse dataset index hash does not match its manifest")


def _validate_misaligned_manifest(
    config: RQ1Config, dataset_sha256: str
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    path = config.paths.misaligned_activations_dir / "manifest.v1.json"
    manifest = _read_json(path, "misaligned activation manifest")
    expected_top = {
        "schema_version": 1,
        "kind": "rq1_activation_manifest",
        "artifact_kind": "rq1_per_example_residual_activations",
        "study_id": config.study_id,
        "status": "extracted_or_resumed",
        "model_source": "misaligned",
        "activation_site": config.analysis.activation_site,
        "token_aggregation": (
            f"mean_nonpadding_tokens_from_ordinal_{config.analysis.token_start}_to_last"
        ),
        "token_start": config.analysis.token_start,
        "reduction_dtype": config.analysis.reduction_dtype,
        "seed": config.analysis.seed,
        "dataset_sha256": dataset_sha256,
        "config_sha256": config.source_sha256,
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected_top.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RQ1WorkflowError(f"misaligned activation manifest mismatch: {mismatches}")
    if manifest.get("model") != _expected_model(config):
        raise RQ1WorkflowError("misaligned activation model provenance mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise RQ1WorkflowError("misaligned activation manifest shards must be a list")
    expected_names = (*config.emotions.reference, "neutral")
    if len(shards) != len(expected_names):
        raise RQ1WorkflowError("misaligned activation manifest must contain 13 shards")
    by_name: dict[str, dict[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("name"), str):
            raise RQ1WorkflowError("misaligned activation manifest has an invalid shard")
        if shard["name"] in by_name:
            raise RQ1WorkflowError(f"duplicate misaligned shard {shard['name']!r}")
        by_name[shard["name"]] = shard
    if set(by_name) != set(expected_names):
        raise RQ1WorkflowError("misaligned activation manifest shard names are incomplete")
    return path, manifest, by_name


def _load_activation_npz(
    path: Path,
    config: RQ1Config,
    *,
    require_meta: bool,
    expected_meta_fields: tuple[str, ...] | None,
    expected_archive_sha256: str | None,
    expected_cache_fields: Mapping[str, Any] | None,
) -> tuple[np.ndarray[Any, Any], tuple[tuple[int, int], ...] | None, str, dict[str, Any]]:
    if not path.is_file():
        raise RQ1WorkflowError(f"activation archive is missing: {path}")
    archive_sha256 = sha256_file(path)
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise RQ1WorkflowError(f"activation archive hash mismatch: {path}")
    cache_path = path.with_suffix(".cache.json")
    cache = _read_json(cache_path, "activation cache")
    cache_expected = {
        "schema_version": 1,
        "archive_sha256": archive_sha256,
        "archive_size": path.stat().st_size,
        "vector_dtype": "float32",
        "has_meta": require_meta,
        **dict(expected_cache_fields or {}),
    }
    cache_mismatches = {
        key: {"actual": cache.get(key), "expected": value}
        for key, value in cache_expected.items()
        if cache.get(key) != value
    }
    if cache_mismatches:
        raise RQ1WorkflowError(f"activation cache mismatch for {path}: {cache_mismatches}")

    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_files = {"vectors", "meta"} if require_meta else {"vectors"}
            if set(archive.files) != expected_files:
                raise RQ1WorkflowError(
                    f"{path}: arrays {archive.files!r} do not match {sorted(expected_files)!r}"
                )
            vectors = np.asarray(archive["vectors"])
            metadata = np.asarray(archive["meta"]) if require_meta else None
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, RQ1WorkflowError):
            raise
        raise RQ1WorkflowError(f"could not load activation archive {path}: {exc}") from exc

    expected_tail = (len(config.analysis.layers), config.analysis.hidden_size)
    if vectors.ndim != 3 or vectors.shape[1:] != expected_tail:
        raise RQ1WorkflowError(
            f"{path}: activation shape {vectors.shape} does not end in {expected_tail}"
        )
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise RQ1WorkflowError(f"{path}: activations must be finite float32")
    if cache.get("vector_shape") != list(vectors.shape):
        raise RQ1WorkflowError(f"{path}: cache vector shape mismatch")

    rows: tuple[tuple[int, int], ...] | None = None
    if require_meta:
        assert metadata is not None
        if metadata.ndim != 1 or metadata.shape[0] != vectors.shape[0]:
            raise RQ1WorkflowError(f"{path}: metadata row count mismatch")
        if metadata.dtype.names != expected_meta_fields:
            raise RQ1WorkflowError(
                f"{path}: metadata fields {metadata.dtype.names} != {expected_meta_fields}"
            )
        assert expected_meta_fields is not None
        rows = tuple(
            (int(row[expected_meta_fields[0]]), int(row[expected_meta_fields[1]]))
            for row in metadata
        )
        if len(rows) != len(set(rows)):
            raise RQ1WorkflowError(f"{path}: metadata contains duplicate row identities")
    return vectors, rows, archive_sha256, cache


def _misaligned_cache_fields(
    config: RQ1Config,
    *,
    source_sha256: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    return {
        "study_id": config.study_id,
        "model_source": "misaligned",
        "model": _expected_model(config),
        "activation_site": config.analysis.activation_site,
        "token_aggregation": (
            f"mean_nonpadding_tokens_from_ordinal_{config.analysis.token_start}_to_last"
        ),
        "token_start": config.analysis.token_start,
        "source_sha256": source_sha256,
        "dataset_sha256": dataset_sha256,
        "reduction_dtype": config.analysis.reduction_dtype,
        "seed": config.analysis.seed,
        "config_sha256": config.source_sha256,
    }


def _source_hashes_from_reuse_manifest(dataset: ReuseDataset) -> dict[str, str]:
    sources = dataset.manifest.get("sources", {}).get("raw_jsonl", [])
    if not isinstance(sources, list):
        raise RQ1WorkflowError("reuse manifest raw_jsonl sources must be a list")
    result: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise RQ1WorkflowError("reuse manifest contains an invalid raw source")
        emotion = source.get("emotion")
        digest = source.get("sha256")
        if not isinstance(emotion, str) or not isinstance(digest, str):
            raise RQ1WorkflowError("reuse manifest raw source lacks emotion/hash")
        result[emotion] = digest
    return result


def _load_normalized_em(
    config: RQ1Config,
) -> tuple[np.ndarray[Any, Any], dict[str, Any], Path, str]:
    path = config.paths.results_dir / "artifacts" / NORMALIZED_EM_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"raw", "metadata", "legacy_unverified", "source_sha256"}:
                raise RQ1WorkflowError(
                    f"normalized EM artifact has unexpected arrays: {archive.files}"
                )
            vectors = np.asarray(archive["raw"])
            metadata_value = np.asarray(archive["metadata"])
            legacy_value = np.asarray(archive["legacy_unverified"])
            source_sha_value = np.asarray(archive["source_sha256"])
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, RQ1WorkflowError):
            raise
        raise RQ1WorkflowError(f"could not load normalized EM artifact {path}: {exc}") from exc
    expected_shape = (len(config.analysis.layers), config.analysis.hidden_size)
    if vectors.shape != expected_shape or vectors.dtype != np.float32:
        raise RQ1WorkflowError(
            f"normalized EM shape/dtype is {vectors.shape}/{vectors.dtype}, "
            f"expected {expected_shape}/float32"
        )
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise RQ1WorkflowError("normalized EM direction contains non-finite or zero layers")
    if metadata_value.shape != () or legacy_value.shape != () or source_sha_value.shape != ():
        raise RQ1WorkflowError("normalized EM scalar metadata arrays are malformed")
    try:
        metadata = json.loads(str(metadata_value.item()))
    except json.JSONDecodeError as exc:
        raise RQ1WorkflowError("normalized EM metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise RQ1WorkflowError("normalized EM metadata must be an object")
    expected_metadata = {
        "model_id": config.models.misaligned_adapter_id,
        "model_revision": config.models.misaligned_adapter_revision,
        "hook_site": config.analysis.activation_site,
        "target_id": CANONICAL_TARGET_ID,
        "source_script": CANONICAL_SOURCE_SCRIPT,
        "adapter_rank": CANONICAL_ADAPTER_RANK,
        "token_aggregation": CANONICAL_TOKEN_AGGREGATION,
        "storage_dtype": CANONICAL_STORAGE_DTYPE,
        "direction_sign": CANONICAL_DIRECTION_SIGN,
        "definition": CANONICAL_DEFINITION,
        "n_layers": len(config.analysis.layers),
        "hidden_size": config.analysis.hidden_size,
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RQ1WorkflowError(f"normalized EM provenance mismatch: {mismatches}")
    if metadata.get("subtraction_dtype") not in SUPPORTED_SUBTRACTION_DTYPES:
        raise RQ1WorkflowError(
            "normalized EM provenance must record bfloat16 or float32 subtraction"
        )
    if bool(legacy_value.item()) or config.em.allow_unverified_legacy:
        raise RQ1WorkflowError("unverified legacy EM directions are forbidden for RQ1 analysis")
    source_sha256 = str(source_sha_value.item())
    if len(source_sha256) != 64:
        raise RQ1WorkflowError("normalized EM artifact lacks a SHA-256 source identity")
    if config.em.artifact_path.is_file() and sha256_file(config.em.artifact_path) != source_sha256:
        raise RQ1WorkflowError("normalized EM source hash does not match the canonical artifact")
    return vectors, metadata, path, sha256_file(path)


def load_workflow_inputs(
    config: RQ1Config,
    dataset: ReuseDataset | None = None,
    *,
    require_em: bool = True,
) -> WorkflowInputs:
    """Load and exactly align all confirmatory arrays from validated disk artifacts."""

    reuse = build_reuse_dataset(config) if dataset is None else dataset
    _validate_dataset(config, reuse)
    dataset_sha256 = _dataset_source_hash(config)
    manifest_path, _manifest, manifest_shards = _validate_misaligned_manifest(
        config, dataset_sha256
    )
    raw_hashes = _source_hashes_from_reuse_manifest(reuse)
    activation_metadata = {item.emotion: item for item in reuse.activation_metadata}

    aligned_by_emotion: dict[str, np.ndarray[Any, Any]] = {}
    misaligned_by_emotion: dict[str, np.ndarray[Any, Any]] = {}
    activation_hashes: dict[str, dict[str, str]] = {}
    common_cells: tuple[tuple[int, int], ...] | None = None

    for emotion in config.emotions.reference:
        canonical_rows = reuse.rows_for_emotion(emotion, confirmatory=True)
        cells = tuple(row.cell for row in canonical_rows)
        if common_cells is None:
            common_cells = cells
        elif cells != common_cells:
            raise RQ1WorkflowError("confirmatory cell order differs between emotions")

        aligned_metadata = activation_metadata.get(emotion)
        if aligned_metadata is None:
            raise RQ1WorkflowError(f"reuse dataset lacks aligned metadata for {emotion}")
        aligned_vectors, aligned_rows, aligned_sha, _aligned_cache = _load_activation_npz(
            aligned_metadata.archive_path,
            config,
            require_meta=True,
            expected_meta_fields=("topic_id", "story_idx"),
            expected_archive_sha256=aligned_metadata.archive_sha256,
            expected_cache_fields=None,
        )
        assert aligned_rows is not None
        if aligned_rows != aligned_metadata.rows:
            raise RQ1WorkflowError(f"{emotion}: aligned NPZ row order changed after indexing")

        source_path = config.paths.raw_dir / "stories" / f"{emotion}.jsonl"
        source_sha = sha256_file(source_path)
        if raw_hashes.get(emotion) != source_sha:
            raise RQ1WorkflowError(f"{emotion}: raw source hash differs from reuse manifest")
        misaligned_path = config.paths.misaligned_activations_dir / "stories" / f"{emotion}.npz"
        shard = manifest_shards[emotion]
        misaligned_vectors, misaligned_rows, misaligned_sha, _misaligned_cache = (
            _load_activation_npz(
                misaligned_path,
                config,
                require_meta=True,
                expected_meta_fields=("topic_id", "story_idx"),
                expected_archive_sha256=shard.get("artifact_sha256"),
                expected_cache_fields=_misaligned_cache_fields(
                    config,
                    source_sha256=source_sha,
                    dataset_sha256=dataset_sha256,
                ),
            )
        )
        assert misaligned_rows is not None
        if misaligned_rows != aligned_rows:
            raise RQ1WorkflowError(
                f"{emotion}: aligned and misaligned NPZ row identities/order differ"
            )
        if shard.get("source_sha256") != source_sha or shard.get("shape") != list(
            misaligned_vectors.shape
        ):
            raise RQ1WorkflowError(f"{emotion}: manifest shard source hash/shape mismatch")

        lookup = {cell: index for index, cell in enumerate(aligned_rows)}
        try:
            selected_indices = np.asarray([lookup[cell] for cell in cells], dtype=np.int64)
        except KeyError as exc:
            raise RQ1WorkflowError(
                f"{emotion}: confirmatory row {exc.args[0]} is absent from activation archives"
            ) from exc
        aligned_selected = np.ascontiguousarray(aligned_vectors[selected_indices])
        misaligned_selected = np.ascontiguousarray(misaligned_vectors[selected_indices])
        if aligned_selected.shape[0] != 79 or misaligned_selected.shape != aligned_selected.shape:
            raise RQ1WorkflowError(
                f"{emotion}: balanced activation selection is not 79 matched rows"
            )
        aligned_by_emotion[emotion] = aligned_selected
        misaligned_by_emotion[emotion] = misaligned_selected
        activation_hashes[emotion] = {
            "aligned": aligned_sha,
            "misaligned": misaligned_sha,
            "source": source_sha,
        }

    assert common_cells is not None
    aligned_neutral, _aligned_meta, aligned_neutral_sha, _ = _load_activation_npz(
        config.paths.neutral_activations_path,
        config,
        require_meta=False,
        expected_meta_fields=None,
        expected_archive_sha256=None,
        expected_cache_fields=None,
    )
    neutral_source = config.paths.raw_dir / "neutral.jsonl"
    neutral_source_sha = sha256_file(neutral_source)
    neutral_shard = manifest_shards["neutral"]
    misaligned_neutral_path = config.paths.misaligned_activations_dir / "neutral.npz"
    misaligned_neutral, neutral_meta, misaligned_neutral_sha, _ = _load_activation_npz(
        misaligned_neutral_path,
        config,
        require_meta=True,
        expected_meta_fields=("topic_id", "dialogue_idx"),
        expected_archive_sha256=neutral_shard.get("artifact_sha256"),
        expected_cache_fields=_misaligned_cache_fields(
            config,
            source_sha256=neutral_source_sha,
            dataset_sha256=dataset_sha256,
        ),
    )
    if neutral_meta is None or neutral_shard.get("source_sha256") != neutral_source_sha:
        raise RQ1WorkflowError("misaligned neutral source identity is invalid")
    if neutral_shard.get("shape") != list(misaligned_neutral.shape):
        raise RQ1WorkflowError("misaligned neutral manifest shape mismatch")
    if aligned_neutral.shape != misaligned_neutral.shape:
        raise RQ1WorkflowError("aligned and misaligned neutral activation shapes differ")
    activation_hashes["neutral"] = {
        "aligned": aligned_neutral_sha,
        "misaligned": misaligned_neutral_sha,
        "source": neutral_source_sha,
    }

    em_path = config.paths.results_dir / "artifacts" / NORMALIZED_EM_FILENAME
    if require_em:
        em_direction, em_metadata, em_path, em_sha256 = _load_normalized_em(config)
    else:
        em_direction = None
        em_metadata = {}
        em_sha256 = None

    return WorkflowInputs(
        dataset=reuse,
        aligned_by_emotion=aligned_by_emotion,
        misaligned_by_emotion=misaligned_by_emotion,
        aligned_neutral=aligned_neutral,
        misaligned_neutral=misaligned_neutral,
        topic_ids=tuple(cell[0] for cell in common_cells),
        block_ids=common_cells,
        em_direction=em_direction,
        em_metadata=em_metadata,
        em_path=em_path,
        em_sha256=em_sha256,
        misaligned_manifest_path=manifest_path,
        misaligned_manifest_sha256=sha256_file(manifest_path),
        activation_hashes=activation_hashes,
    )


def _nan() -> float:
    return float("nan")


def _metric_row(**values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "comparison_type": "",
        "model_source": "",
        "layer": -1,
        "emotion_or_group": "",
        "estimate_name": "",
        "cosine": _nan(),
        "explained_fraction": _nan(),
        "ci_low": _nan(),
        "ci_high": _nan(),
        "p_value": _nan(),
        "q_value": _nan(),
        "max_stat_p_value": _nan(),
        "null_p95": _nan(),
        "reliability": _nan(),
        "pca_status": "raw",
        "effective_rank": _nan(),
        "condition_number": _nan(),
        "gate_passed": False,
        "confirmatory": True,
        "notes": "",
    }
    unknown = set(values) - set(row)
    if unknown:
        raise RQ1WorkflowError(f"unknown metric fields: {sorted(unknown)}")
    row.update(values)
    return {column: row[column] for column in METRIC_COLUMNS}


def _pca_label(value: str) -> str:
    labels = {
        "raw": "raw",
        "cleaned_aligned_neutral": "cleaned_aligned_basis",
        "cleaned_misaligned_neutral": "cleaned_misaligned_basis",
    }
    try:
        return labels[value]
    except KeyError as exc:
        raise RQ1WorkflowError(f"unknown analysis PCA status {value!r}") from exc


def _reliability_for_source(result: RQ1AnalysisResult, model_source: str) -> np.ndarray[Any, Any]:
    if model_source == "aligned":
        return result.reliability_gate.aligned_reliability
    if model_source == "misaligned":
        return result.reliability_gate.misaligned_reliability
    raise RQ1WorkflowError(f"unknown model source {model_source!r}")


def flatten_analysis_metrics(result: RQ1AnalysisResult) -> pd.DataFrame:
    """Flatten typed analysis outputs into the fixed 20-column metrics schema."""

    rows: list[dict[str, Any]] = []
    gate = result.reliability_gate
    emotion_index = {emotion: index for index, emotion in enumerate(result.all_emotions)}
    n_layers = gate.aligned_reliability.shape[1]

    for model_source, values in (
        ("aligned", gate.aligned_reliability),
        ("misaligned", gate.misaligned_reliability),
    ):
        for emotion, index in emotion_index.items():
            for layer in range(n_layers):
                rows.append(
                    _metric_row(
                        comparison_type="split_reliability",
                        model_source=model_source,
                        layer=layer,
                        emotion_or_group=emotion,
                        estimate_name="reliability",
                        reliability=float(values[index, layer]),
                        gate_passed=gate.passed,
                        notes="topic split A (0-9) versus split B (10-19)",
                    )
                )
    for emotion, index in emotion_index.items():
        for layer in range(n_layers):
            rows.append(
                _metric_row(
                    comparison_type="cross_model_stability",
                    model_source="cross_model",
                    layer=layer,
                    emotion_or_group=emotion,
                    estimate_name="reliability",
                    reliability=float(gate.cross_model_stability[index, layer]),
                    gate_passed=gate.passed,
                    notes="aligned versus misaligned full emotion direction",
                )
            )

    for geometry in result.geometries:
        pca_status = _pca_label(geometry.pca_status)
        reliability = _reliability_for_source(result, geometry.model_source)
        individual = geometry.individual
        for local_index, emotion in enumerate(individual.emotions):
            global_index = emotion_index[emotion]
            for layer in range(n_layers):
                rows.append(
                    _metric_row(
                        comparison_type="individual",
                        model_source=geometry.model_source,
                        layer=layer,
                        emotion_or_group=emotion,
                        estimate_name="cosine",
                        cosine=float(individual.cosine[local_index, layer]),
                        ci_low=float(individual.ci_low[local_index, layer]),
                        ci_high=float(individual.ci_high[local_index, layer]),
                        p_value=float(individual.permutation_p_value[local_index, layer]),
                        q_value=float(individual.permutation_q_value[local_index, layer]),
                        max_stat_p_value=float(individual.max_stat_p_value[local_index]),
                        null_p95=float(individual.permutation_null_p95[local_index, layer]),
                        reliability=float(reliability[global_index, layer]),
                        pca_status=pca_status,
                        gate_passed=gate.passed,
                        notes=(
                            "control=emotion_label_permutation;"
                            f"family_max_stat_p={individual.family_max_stat_p_value}"
                        ),
                    )
                )

        centroids = geometry.centroids
        for group_index, (group_name, members) in enumerate(
            zip(centroids.group_names, centroids.group_members, strict=True)
        ):
            member_indices = [emotion_index[emotion] for emotion in members]
            for layer in range(n_layers):
                group_reliability = float(np.median(reliability[member_indices, layer]))
                common = {
                    "comparison_type": "centroid",
                    "model_source": geometry.model_source,
                    "layer": layer,
                    "emotion_or_group": group_name,
                    "estimate_name": "cosine",
                    "cosine": float(centroids.cosine[group_index, layer]),
                    "ci_low": float(centroids.ci_low[group_index, layer]),
                    "ci_high": float(centroids.ci_high[group_index, layer]),
                    "reliability": group_reliability,
                    "pca_status": pca_status,
                    "gate_passed": gate.passed,
                }
                rows.append(
                    _metric_row(
                        **common,
                        p_value=float(centroids.label_permutation_p_value[group_index, layer]),
                        q_value=float(centroids.label_permutation_q_value[group_index, layer]),
                        max_stat_p_value=float(centroids.label_max_stat_p_value[group_index]),
                        null_p95=float(centroids.label_permutation_null_p95[group_index, layer]),
                        notes=(
                            "control=emotion_label_permutation;"
                            f"members={','.join(members)};"
                            f"family_max_stat_p={centroids.label_family_max_stat_p_value}"
                        ),
                    )
                )
                rows.append(
                    _metric_row(
                        **common,
                        p_value=float(centroids.matched_centroid_p_value[group_index, layer]),
                        q_value=float(centroids.matched_centroid_q_value[group_index, layer]),
                        max_stat_p_value=float(centroids.matched_max_stat_p_value[group_index]),
                        null_p95=float(centroids.matched_centroid_null_p95[group_index, layer]),
                        notes=(
                            "control=matched_random_centroid;"
                            f"members={','.join(members)};"
                            f"family_max_stat_p={centroids.matched_family_max_stat_p_value}"
                        ),
                    )
                )

        subspace = geometry.subspace
        reported_indices = [emotion_index[emotion] for emotion in subspace.emotions]
        for layer in range(n_layers):
            rows.append(
                _metric_row(
                    comparison_type="subspace",
                    model_source=geometry.model_source,
                    layer=layer,
                    emotion_or_group="all_six_negative",
                    estimate_name="explained_fraction",
                    cosine=float(subspace.projection_cosine[layer]),
                    explained_fraction=float(subspace.explained_fraction[layer]),
                    ci_low=float(subspace.ci_low[layer]),
                    ci_high=float(subspace.ci_high[layer]),
                    p_value=float(subspace.random_subspace_p_value[layer]),
                    q_value=float(subspace.random_subspace_q_value[layer]),
                    max_stat_p_value=float(subspace.max_stat_p_value),
                    null_p95=float(subspace.random_subspace_null_p95[layer]),
                    reliability=float(np.median(reliability[reported_indices, layer])),
                    pca_status=pca_status,
                    effective_rank=float(subspace.effective_rank[layer]),
                    condition_number=float(subspace.condition_number[layer]),
                    gate_passed=gate.passed,
                    notes=f"control=matched_random_subspace;method={subspace.random_subspace_method}",
                )
            )

    frame = pd.DataFrame.from_records(rows, columns=METRIC_COLUMNS)
    if tuple(frame.columns) != METRIC_COLUMNS or frame.empty:
        raise RQ1WorkflowError("analysis did not produce the fixed metrics schema")
    if set(frame["comparison_type"]) != {
        "individual",
        "centroid",
        "subspace",
        "split_reliability",
        "cross_model_stability",
    }:
        raise RQ1WorkflowError("flattened metrics are missing a comparison family")
    if not set(frame["model_source"]).issubset({"aligned", "misaligned", "cross_model"}):
        raise RQ1WorkflowError("flattened metrics contain an invalid model source")
    if not set(frame["pca_status"]).issubset(
        {"raw", "cleaned_aligned_basis", "cleaned_misaligned_basis"}
    ):
        raise RQ1WorkflowError("flattened metrics contain an invalid PCA status")
    return frame


def _save_direction_artifact(
    config: RQ1Config,
    *,
    model_source: str,
    directions: EmotionDirectionSet,
    inputs: WorkflowInputs,
) -> tuple[Path, str]:
    destination = (
        config.paths.results_dir
        / "artifacts"
        / f"emotion_directions.{model_source}.v{DIRECTION_ARTIFACT_SCHEMA_VERSION}.npz"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if model_source == "aligned":
        model = {
            "model_id": config.models.base_model_id,
            "model_revision": config.models.base_model_revision,
        }
    elif model_source == "misaligned":
        model = {
            "model_id": config.models.misaligned_adapter_id,
            "model_revision": config.models.misaligned_adapter_revision,
            "base_model_id": config.models.audited_parent_model_id,
            "base_model_revision": config.models.audited_parent_revision,
            "adapter_merged": False,
        }
    else:
        raise RQ1WorkflowError(f"invalid direction model source {model_source!r}")
    metadata = {
        "schema_version": DIRECTION_ARTIFACT_SCHEMA_VERSION,
        "kind": "rq1_emotion_directions",
        "study_id": config.study_id,
        "model_source": model_source,
        "model": model,
        "tokenizer_id": config.models.tokenizer_id,
        "tokenizer_revision": config.models.tokenizer_revision,
        "activation_site": config.analysis.activation_site,
        "token_aggregation": (
            f"mean_nonpadding_tokens_from_ordinal_{config.analysis.token_start}_to_last"
        ),
        "direction_definition": "emotion_mean_minus_mean_of_other_11_emotions",
        "selection": "balanced_confirmatory_complete_cell",
        "selection_rows": 948,
        "rows_per_emotion": 79,
        "dataset_index_sha256": inputs.dataset.manifest["index"]["sha256"],
        "selection_sha256": inputs.dataset.manifest["selection"]["confirmatory"]["sha256"],
        "config_sha256": config.source_sha256,
        "layers": list(config.analysis.layers),
        "hidden_size": config.analysis.hidden_size,
        "dtype": "float32",
        "source_hashes": {
            emotion: dict(inputs.activation_hashes[emotion])
            for emotion in config.emotions.reference
        },
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                raw=np.asarray(directions.full, dtype=np.float32),
                split_a=np.asarray(directions.split_a, dtype=np.float32),
                split_b=np.asarray(directions.split_b, dtype=np.float32),
                emotions=np.asarray(directions.emotions),
                counts=np.asarray(directions.counts, dtype=np.int64),
                split_a_counts=np.asarray(directions.split_a_counts, dtype=np.int64),
                split_b_counts=np.asarray(directions.split_b_counts, dtype=np.int64),
                metadata=np.asarray(_canonical_json(metadata)),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, sha256_file(destination)


def _write_metrics(frame: pd.DataFrame, results_dir: Path) -> tuple[Path, Path]:
    csv_path = results_dir / "rq1_metrics.csv"
    parquet_path = results_dir / "rq1_metrics.parquet"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_temp = results_dir / f".{csv_path.name}.{uuid.uuid4().hex}.tmp"
    parquet_temp = results_dir / f".{parquet_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(csv_temp, index=False, columns=list(METRIC_COLUMNS))
        frame.to_parquet(parquet_temp, index=False, engine="pyarrow")
        reloaded = pd.read_parquet(parquet_temp, engine="pyarrow")
        if tuple(reloaded.columns) != METRIC_COLUMNS or len(reloaded) != len(frame):
            raise RQ1WorkflowError("temporary Parquet metrics failed schema verification")
        os.replace(csv_temp, csv_path)
        os.replace(parquet_temp, parquet_path)
    finally:
        csv_temp.unlink(missing_ok=True)
        parquet_temp.unlink(missing_ok=True)
    return csv_path, parquet_path


def _quality_summary(result: RQ1AnalysisResult) -> dict[str, Any]:
    gate = result.reliability_gate
    layers: list[dict[str, Any]] = []
    for index, layer in enumerate(gate.preregistered_layers):
        layers.append(
            {
                "layer": layer,
                "aligned_median_split_reliability": float(gate.aligned_median_by_layer[layer]),
                "misaligned_median_split_reliability": float(
                    gate.misaligned_median_by_layer[layer]
                ),
                "cross_model_median_stability": float(gate.cross_model_median_by_layer[layer]),
                "split_reliability_passed": bool(gate.layer_passes[index]),
                "cross_model_stability_passed": bool(
                    result.cross_model_preregistered_passes[index]
                ),
            }
        )
    return {
        "evaluated": True,
        "passed": bool(gate.passed),
        "threshold": gate.threshold,
        "preregistered_layers": layers,
        "message": result.gate_message,
    }


def _derive_primary_evidence(result: RQ1AnalysisResult) -> dict[str, Any]:
    """Use the analysis module's single preregistered decision implementation."""

    evidence = result.primary_evidence.as_dict()
    required = {
        "verdict",
        "positive",
        "quality_gate_passed",
        "confirmatory_centroid_group",
        "centroid_passing_layers",
        "subspace_passing_layers",
        "cross_model_preregistered_passes",
        "verdict_reason",
    }
    if set(evidence) != required:
        raise RQ1WorkflowError(
            f"primary evidence schema mismatch: {sorted(set(evidence) ^ required)}"
        )
    if evidence["verdict"] not in {"positive", "null", "inconclusive"}:
        raise RQ1WorkflowError(f"invalid primary verdict {evidence['verdict']!r}")
    if bool(evidence["positive"]) != (evidence["verdict"] == "positive"):
        raise RQ1WorkflowError("primary positive flag contradicts the verdict")
    if bool(evidence["quality_gate_passed"]) != bool(result.reliability_gate.passed):
        raise RQ1WorkflowError("primary evidence contradicts the reliability gate")
    if tuple(evidence["cross_model_preregistered_passes"]) != tuple(
        result.cross_model_preregistered_passes
    ):
        raise RQ1WorkflowError("primary evidence cross-model flags do not match analysis")
    if (
        evidence["verdict"] == "positive"
        and not evidence["centroid_passing_layers"]
        and evidence["subspace_passing_layers"]
    ):
        raise RQ1WorkflowError(
            "subspace-only positive evidence is unsupported until the analysis records both "
            "emotion-label-permutation and matched-random-subspace controls"
        )
    return evidence


def _build_directions_without_em(
    inputs: WorkflowInputs, config: RQ1Config
) -> tuple[EmotionDirectionSet, EmotionDirectionSet, ReliabilityGate, tuple[bool, ...]]:
    topic_mapping = {emotion: inputs.topic_ids for emotion in config.emotions.reference}
    kwargs = {
        "topic_ids_by_emotion": topic_mapping,
        "emotions": config.emotions.reference,
        "split_a_topics": config.analysis.topic_split_a,
        "split_b_topics": config.analysis.topic_split_b,
    }
    aligned = build_emotion_directions(
        inputs.aligned_by_emotion,
        **kwargs,
    )
    misaligned = build_emotion_directions(
        inputs.misaligned_by_emotion,
        **kwargs,
    )
    gate = evaluate_reliability_gate(
        aligned.split_a,
        aligned.split_b,
        misaligned.split_a,
        misaligned.split_b,
        aligned.full,
        misaligned.full,
        preregistered_layers=config.analysis.preregistered_layers,
        threshold=config.analysis.reliability_threshold,
    )
    cross_model_passes = tuple(
        bool(gate.cross_model_median_by_layer[layer] >= gate.threshold)
        for layer in gate.preregistered_layers
    )
    return aligned, misaligned, gate, cross_model_passes


def _quality_summary_from_gate(
    gate: ReliabilityGate, cross_model_passes: Sequence[bool], message: str
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for index, layer in enumerate(gate.preregistered_layers):
        layers.append(
            {
                "layer": layer,
                "aligned_median_split_reliability": float(gate.aligned_median_by_layer[layer]),
                "misaligned_median_split_reliability": float(
                    gate.misaligned_median_by_layer[layer]
                ),
                "cross_model_median_stability": float(gate.cross_model_median_by_layer[layer]),
                "split_reliability_passed": bool(gate.layer_passes[index]),
                "cross_model_stability_passed": bool(cross_model_passes[index]),
            }
        )
    return {
        "evaluated": True,
        "passed": bool(gate.passed),
        "threshold": gate.threshold,
        "preregistered_layers": layers,
        "message": message,
    }


def _limitations(config: RQ1Config) -> list[str]:
    limitations = [
        "Reduced reuse study: individual claims are limited to six captured negative emotions.",
        "Hostility, contempt, cruelty, desperation, and other uncaptured concepts were not tested.",
        "The original 36-emotion implicit-validation retrieval gate is unavailable.",
        "The 959-row unbalanced intersection is sensitivity-only; primary inference uses 948 balanced rows.",
    ]
    if config.em.aligned_response_path is None or config.em.misaligned_response_path is None:
        limitations.append(
            "Per-example EM response activations and the aligned-model text-semantic control are unavailable."
        )
    return limitations


def _counts(config: RQ1Config, dataset: ReuseDataset) -> dict[str, Any]:
    return {
        "raw_rows": len(dataset.rows),
        "same_example_sensitivity_rows": len(dataset.same_example_rows),
        "balanced_confirmatory_rows": len(dataset.confirmatory_rows),
        "rows_per_emotion": 79,
        "reference_emotions": len(config.emotions.reference),
        "reported_negative_emotions": len(config.emotions.reported_negative),
        "layers": len(config.analysis.layers),
        "hidden_size": config.analysis.hidden_size,
    }


def _blocked_summary(
    config: RQ1Config,
    inputs: WorkflowInputs,
    *,
    quality_gate: Mapping[str, Any],
    direction_artifacts: Sequence[tuple[Path, str]],
    blocker: str,
) -> dict[str, Any]:
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "study_id": config.study_id,
        "status": "blocked",
        "conclusion": "inconclusive",
        "quality_gate": dict(quality_gate),
        "primary_evidence": {
            "evaluated": False,
            "positive": False,
            "reason": "Canonical EM geometry is unavailable; no overlap metric was computed.",
            "direction_artifacts": [
                {"path": str(path), "sha256": digest} for path, digest in direction_artifacts
            ],
        },
        "counts": _counts(config, inputs.dataset),
        "models": asdict(config.models),
        "em": {
            "status": "missing_normalized_artifact",
            "required_path": str(inputs.em_path),
            "target_id": CANONICAL_TARGET_ID,
            "source_script": CANONICAL_SOURCE_SCRIPT,
            "adapter_rank": CANONICAL_ADAPTER_RANK,
            "token_aggregation": CANONICAL_TOKEN_AGGREGATION,
            "direction_sign": config.em.direction_sign,
        },
        "limitations": _limitations(config),
        "metrics_csv": None,
        "metrics_parquet": None,
        "verdict_reason": blocker,
    }


def _complete_summary(
    config: RQ1Config,
    inputs: WorkflowInputs,
    result: RQ1AnalysisResult,
    frame: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path,
    direction_artifacts: Sequence[tuple[Path, str]],
) -> dict[str, Any]:
    primary = _derive_primary_evidence(result)
    conclusion = str(primary.get("verdict", "inconclusive"))
    if conclusion not in {"positive", "null", "inconclusive"}:
        raise RQ1WorkflowError(f"analysis returned invalid primary verdict {conclusion!r}")
    verdict_reason = str(primary.get("verdict_reason", result.gate_message))
    primary = {
        **primary,
        "direction_artifacts": [
            {"path": str(path), "sha256": digest} for path, digest in direction_artifacts
        ],
    }
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "study_id": config.study_id,
        "status": "complete",
        "conclusion": conclusion,
        "quality_gate": _quality_summary(result),
        "primary_evidence": primary,
        "counts": {
            **_counts(config, inputs.dataset),
            "metric_rows": len(frame),
            "aligned_neutral_rows": int(inputs.aligned_neutral.shape[0]),
            "misaligned_neutral_rows": int(inputs.misaligned_neutral.shape[0]),
        },
        "models": asdict(config.models),
        "em": {
            "status": "validated",
            "normalized_path": str(inputs.em_path),
            "normalized_sha256": inputs.em_sha256,
            "metadata": dict(inputs.em_metadata),
        },
        "limitations": _limitations(config),
        "metrics_csv": {
            "path": str(csv_path),
            "sha256": sha256_file(csv_path),
            "rows": len(frame),
        },
        "metrics_parquet": {
            "path": str(parquet_path),
            "sha256": sha256_file(parquet_path),
            "rows": len(frame),
        },
        "verdict_reason": verdict_reason,
    }


def run_analysis_workflow(
    config: RQ1Config,
    *,
    dataset: ReuseDataset | None = None,
    raise_on_blocked: bool = True,
) -> WorkflowRun:
    """Run strict RQ1 analysis, persist artifacts, or durably report an EM blocker."""

    reuse = build_reuse_dataset(config) if dataset is None else dataset
    _validate_dataset(config, reuse)
    summary_path = config.paths.results_dir / "analysis_summary.json"
    normalized_em_path = config.paths.results_dir / "artifacts" / NORMALIZED_EM_FILENAME

    if not normalized_em_path.is_file():
        inputs = load_workflow_inputs(config, reuse, require_em=False)
        aligned, misaligned, gate, cross_model_passes = _build_directions_without_em(inputs, config)
        artifacts = (
            _save_direction_artifact(
                config,
                model_source="aligned",
                directions=aligned,
                inputs=inputs,
            ),
            _save_direction_artifact(
                config,
                model_source="misaligned",
                directions=misaligned,
                inputs=inputs,
            ),
        )
        message = (
            "Canonical repository-script q14b_bad_med_32 rank-32-adapter normalized EM "
            "artifact is missing. Emotion-vector reliability was "
            "evaluated, but no emotion/EM overlap metric was computed."
        )
        quality = _quality_summary_from_gate(gate, cross_model_passes, message)
        summary = _blocked_summary(
            config,
            inputs,
            quality_gate=quality,
            direction_artifacts=artifacts,
            blocker=message,
        )
        _atomic_write_json(summary_path, summary)
        run = WorkflowRun(
            status="blocked",
            conclusion="inconclusive",
            summary_path=summary_path,
            metrics_csv=None,
            metrics_parquet=None,
            direction_artifacts=tuple(path for path, _digest in artifacts),
            analysis=None,
            blocker=message,
        )
        if raise_on_blocked:
            raise MissingNormalizedEMArtifact(message, summary_path)
        return run

    inputs = load_workflow_inputs(config, reuse, require_em=True)
    assert inputs.em_direction is not None
    aligned_pca = fit_neutral_pca(
        inputs.aligned_neutral,
        variance_threshold=config.analysis.pca_variance_threshold,
    )
    misaligned_pca = fit_neutral_pca(
        inputs.misaligned_neutral,
        variance_threshold=config.analysis.pca_variance_threshold,
    )
    result = run_rq1_analysis(
        inputs.aligned_by_emotion,
        inputs.misaligned_by_emotion,
        inputs.topic_ids,
        inputs.block_ids,
        inputs.em_direction,
        all_emotions=config.emotions.reference,
        reported_emotions=config.emotions.reported_negative,
        groups={group.name: group.emotions for group in config.emotions.preregistered_groups},
        split_a_topics=config.analysis.topic_split_a,
        split_b_topics=config.analysis.topic_split_b,
        preregistered_layers=config.analysis.preregistered_layers,
        reliability_threshold=config.analysis.reliability_threshold,
        permutation_iterations=config.analysis.permutation_iterations,
        bootstrap_iterations=config.analysis.bootstrap_iterations,
        matched_control_iterations=config.analysis.random_subspace_iterations,
        svd_relative_tolerance=config.analysis.svd_relative_tolerance,
        seed=config.analysis.seed,
        expected_examples_per_emotion=79,
        aligned_neutral_pca=aligned_pca,
        misaligned_neutral_pca=misaligned_pca,
    )
    frame = flatten_analysis_metrics(result)
    artifacts = (
        _save_direction_artifact(
            config,
            model_source="aligned",
            directions=result.aligned_directions,
            inputs=inputs,
        ),
        _save_direction_artifact(
            config,
            model_source="misaligned",
            directions=result.misaligned_directions,
            inputs=inputs,
        ),
    )
    csv_path, parquet_path = _write_metrics(frame, config.paths.results_dir)
    summary = _complete_summary(
        config,
        inputs,
        result,
        frame,
        csv_path,
        parquet_path,
        artifacts,
    )
    _atomic_write_json(summary_path, summary)
    return WorkflowRun(
        status="complete",
        conclusion=summary["conclusion"],
        summary_path=summary_path,
        metrics_csv=csv_path,
        metrics_parquet=parquet_path,
        direction_artifacts=tuple(path for path, _digest in artifacts),
        analysis=result,
    )


run_rq1_workflow = run_analysis_workflow


__all__ = [
    "DIRECTION_ARTIFACT_SCHEMA_VERSION",
    "METRIC_COLUMNS",
    "MissingNormalizedEMArtifact",
    "RQ1WorkflowError",
    "WorkflowInputs",
    "WorkflowRun",
    "flatten_analysis_metrics",
    "load_workflow_inputs",
    "run_analysis_workflow",
    "run_rq1_workflow",
]
