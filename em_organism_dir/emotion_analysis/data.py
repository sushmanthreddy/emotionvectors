"""Validated, deterministic reuse layer for the reduced RQ1 emotion corpus.

The existing aligned activation archives are compressed NPZ files. This module
reads only their NPY headers and small structured metadata members while building
the index; it never materializes the large ``vectors`` arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format

from .config import RQ1Config

INDEX_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 8 * 1024 * 1024


class RQ1DataError(ValueError):
    """Raised when a reuse source violates the preregistered dataset contract."""


@dataclass(frozen=True, slots=True)
class AlignedActivationMetadata:
    emotion: str
    archive_path: Path
    cache_path: Path
    archive_sha256: str
    archive_size: int
    cache_sha256: str
    cache_key: str
    vector_shape: tuple[int, int, int]
    vector_dtype: str
    rows: tuple[tuple[int, int], ...]
    rows_sha256: str

    def as_manifest_record(self, config: RQ1Config) -> dict[str, Any]:
        return {
            "emotion": self.emotion,
            "archive_path": _display_path(self.archive_path, config),
            "cache_path": _display_path(self.cache_path, config),
            "archive_sha256": self.archive_sha256,
            "archive_size": self.archive_size,
            "cache_sha256": self.cache_sha256,
            "cache_key": self.cache_key,
            "vector_shape": list(self.vector_shape),
            "vector_dtype": self.vector_dtype,
            "metadata_rows": len(self.rows),
            "metadata_sha256": self.rows_sha256,
        }


@dataclass(frozen=True, slots=True)
class StoryRecord:
    canonical_row: int
    example_id: str
    emotion: str
    emotion_index: int
    topic_id: int
    story_idx: int
    topic: str
    split: str
    text: str
    text_sha256: str
    raw_source: str
    raw_line_number: int
    aligned_source: str
    aligned_row: int | None
    same_example_eligible: bool
    confirmatory_eligible: bool
    exclusion_reason: str | None

    @property
    def cell(self) -> tuple[int, int]:
        return (self.topic_id, self.story_idx)

    def as_index_record(self) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "canonical_row": self.canonical_row,
            "example_id": self.example_id,
            "emotion": self.emotion,
            "emotion_index": self.emotion_index,
            "topic_id": self.topic_id,
            "story_idx": self.story_idx,
            "topic": self.topic,
            "split": self.split,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "raw_source": self.raw_source,
            "raw_line_number": self.raw_line_number,
            "aligned_source": self.aligned_source,
            "aligned_row": self.aligned_row,
            "same_example_eligible": self.same_example_eligible,
            "confirmatory_eligible": self.confirmatory_eligible,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class ReuseDataset:
    config: RQ1Config
    rows: tuple[StoryRecord, ...]
    activation_metadata: tuple[AlignedActivationMetadata, ...]
    manifest: Mapping[str, Any]

    @property
    def same_example_rows(self) -> tuple[StoryRecord, ...]:
        """All 959 rows available in both aligned and future misaligned passes."""

        return tuple(row for row in self.rows if row.same_example_eligible)

    @property
    def confirmatory_rows(self) -> tuple[StoryRecord, ...]:
        """Balanced 948-row complete-cell set used for confirmatory inference."""

        return tuple(row for row in self.rows if row.confirmatory_eligible)

    @property
    def sensitivity_rows(self) -> tuple[StoryRecord, ...]:
        """Alias documenting that the 959-row intersection is sensitivity-only."""

        return self.same_example_rows

    def rows_for_emotion(
        self, emotion: str, *, confirmatory: bool = True
    ) -> tuple[StoryRecord, ...]:
        if emotion not in self.config.emotions.reference:
            raise KeyError(emotion)
        source = self.confirmatory_rows if confirmatory else self.same_example_rows
        return tuple(row for row in source if row.emotion == emotion)

    def index_jsonl(self) -> str:
        return _index_jsonl(self.rows)


def sha256_file(path: str | Path) -> str:
    """Hash a file by streaming fixed-size chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _display_path(path: Path, config: RQ1Config) -> str:
    try:
        return path.relative_to(config.paths.project_root).as_posix()
    except ValueError:
        return str(path)


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RQ1DataError(f"cannot read {context} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RQ1DataError(f"invalid JSON in {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RQ1DataError(f"{context} {path} must contain a JSON object")
    return value


def _read_npy_header(
    archive: zipfile.ZipFile, member: str
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    try:
        with archive.open(member) as handle:
            version = npy_format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
            else:
                raise RQ1DataError(f"unsupported NPY version {version} in {member}")
    except KeyError as exc:
        raise RQ1DataError(f"NPZ archive is missing {member}") from exc
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype)


def _read_small_npy_member(archive: zipfile.ZipFile, member: str) -> np.ndarray[Any, Any]:
    try:
        with archive.open(member) as handle:
            return npy_format.read_array(handle, allow_pickle=False)
    except KeyError as exc:
        raise RQ1DataError(f"NPZ archive is missing {member}") from exc


def _validate_cache(
    *,
    archive_path: Path,
    cache_path: Path,
    actual_sha256: str,
    vector_shape: tuple[int, ...],
    vector_dtype: str,
) -> tuple[dict[str, Any], str]:
    cache = _read_json_object(cache_path, "activation cache metadata")
    cache_sha256 = sha256_file(cache_path)
    expected = {
        "archive_sha256": actual_sha256,
        "archive_size": archive_path.stat().st_size,
        "vector_shape": list(vector_shape),
        "vector_dtype": vector_dtype,
    }
    for key, expected_value in expected.items():
        if cache.get(key) != expected_value:
            raise RQ1DataError(
                f"{cache_path}: {key}={cache.get(key)!r}, expected {expected_value!r}"
            )
    if cache.get("schema_version") != 1:
        raise RQ1DataError(f"{cache_path}: unsupported cache schema")
    if not isinstance(cache.get("cache_key"), str) or not cache["cache_key"]:
        raise RQ1DataError(f"{cache_path}: cache_key must be non-empty")
    return cache, cache_sha256


def inspect_aligned_activation(config: RQ1Config, emotion: str) -> AlignedActivationMetadata:
    """Validate one aligned archive without loading its activation tensor."""

    if emotion not in config.emotions.reference:
        raise RQ1DataError(f"unknown reference emotion {emotion!r}")
    archive_path = config.paths.aligned_activations_dir / "stories" / f"{emotion}.npz"
    cache_path = archive_path.with_suffix(".cache.json")
    if not archive_path.is_file():
        raise RQ1DataError(f"missing aligned activation archive: {archive_path}")
    archive_sha256 = sha256_file(archive_path)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            vector_shape, fortran_order, vector_dtype = _read_npy_header(archive, "vectors.npy")
            meta = _read_small_npy_member(archive, "meta.npy")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RQ1DataError(f"invalid aligned activation archive {archive_path}: {exc}") from exc

    if fortran_order:
        raise RQ1DataError(f"{archive_path}: vectors must be C-order")
    if len(vector_shape) != 3:
        raise RQ1DataError(f"{archive_path}: vectors must have three dimensions")
    if vector_shape[1:] != (len(config.analysis.layers), config.analysis.hidden_size):
        raise RQ1DataError(
            f"{archive_path}: vector shape {vector_shape} does not match "
            f"48 layers x {config.analysis.hidden_size} hidden units"
        )
    dtype_name = vector_dtype.name
    if dtype_name != "float32":
        raise RQ1DataError(f"{archive_path}: vectors must be float32, got {dtype_name}")
    if meta.ndim != 1 or meta.dtype.names != ("topic_id", "story_idx"):
        raise RQ1DataError(
            f"{archive_path}: meta must be a 1D structured array with topic_id/story_idx"
        )
    rows = tuple((int(row["topic_id"]), int(row["story_idx"])) for row in meta)
    if len(rows) != vector_shape[0]:
        raise RQ1DataError(f"{archive_path}: vector and metadata row counts differ")
    if len(rows) != len(set(rows)):
        raise RQ1DataError(f"{archive_path}: duplicate topic/story metadata rows")
    unknown_cells = set(rows) - set(config.analysis.expected_cells)
    if unknown_cells:
        raise RQ1DataError(f"{archive_path}: unknown metadata cells {sorted(unknown_cells)}")

    cache, cache_sha256 = _validate_cache(
        archive_path=archive_path,
        cache_path=cache_path,
        actual_sha256=archive_sha256,
        vector_shape=vector_shape,
        vector_dtype=dtype_name,
    )
    return AlignedActivationMetadata(
        emotion=emotion,
        archive_path=archive_path,
        cache_path=cache_path,
        archive_sha256=archive_sha256,
        archive_size=archive_path.stat().st_size,
        cache_sha256=cache_sha256,
        cache_key=cache["cache_key"],
        vector_shape=(vector_shape[0], vector_shape[1], vector_shape[2]),
        vector_dtype=dtype_name,
        rows=rows,
        rows_sha256=_sha256_json([list(row) for row in rows]),
    )


def _inspect_vector_archive(config: RQ1Config) -> dict[str, Any]:
    path = config.paths.aligned_vectors_path
    if not path.is_file():
        raise RQ1DataError(f"missing aligned emotion-vector archive: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            raw_shape, raw_fortran, raw_dtype = _read_npy_header(archive, "raw.npy")
            clean_shape, clean_fortran, clean_dtype = _read_npy_header(archive, "denoised.npy")
            emotions = _read_small_npy_member(archive, "emotions.npy")
            primary_layer = _read_small_npy_member(archive, "primary_layer.npy")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RQ1DataError(f"invalid aligned emotion-vector archive {path}: {exc}") from exc

    expected_shape = (
        len(config.emotions.reference),
        len(config.analysis.layers),
        config.analysis.hidden_size,
    )
    if raw_shape != expected_shape or clean_shape != expected_shape:
        raise RQ1DataError(
            f"{path}: raw/denoised shapes must both be {expected_shape}, "
            f"got {raw_shape}/{clean_shape}"
        )
    if raw_fortran or clean_fortran or raw_dtype.name != "float32" or clean_dtype.name != "float32":
        raise RQ1DataError(f"{path}: raw/denoised vectors must be C-order float32")
    if tuple(str(value) for value in emotions.tolist()) != config.emotions.reference:
        raise RQ1DataError(f"{path}: emotion order does not match the preregistered reference")
    if primary_layer.shape != () or int(primary_layer) != 32:
        raise RQ1DataError(f"{path}: original vector artifact must record primary layer 32")

    archive_sha256 = sha256_file(path)
    cache_path = path.with_suffix(".cache.json")
    cache = _read_json_object(cache_path, "emotion-vector cache metadata")
    if cache.get("schema_version") != 2 or cache.get("output_sha256") != archive_sha256:
        raise RQ1DataError(f"{cache_path}: emotion-vector output hash/schema mismatch")
    return {
        "path": _display_path(path, config),
        "sha256": archive_sha256,
        "size": path.stat().st_size,
        "cache_path": _display_path(cache_path, config),
        "cache_sha256": sha256_file(cache_path),
        "shape": list(raw_shape),
        "dtype": raw_dtype.name,
        "emotions": list(config.emotions.reference),
        "recorded_primary_layer": 32,
        "use_policy": "reuse raw vectors at every layer; do not reuse denoised vectors as primary",
    }


def _inspect_neutral_archive(config: RQ1Config) -> dict[str, Any]:
    path = config.paths.neutral_activations_path
    if not path.is_file():
        raise RQ1DataError(f"missing neutral activation archive: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            shape, fortran_order, dtype = _read_npy_header(archive, "vectors.npy")
    except (OSError, zipfile.BadZipFile) as exc:
        raise RQ1DataError(f"invalid neutral activation archive {path}: {exc}") from exc
    if len(shape) != 3 or shape[1:] != (
        len(config.analysis.layers),
        config.analysis.hidden_size,
    ):
        raise RQ1DataError(f"{path}: unexpected neutral activation shape {shape}")
    if fortran_order or dtype.name != "float32":
        raise RQ1DataError(f"{path}: neutral vectors must be C-order float32")
    archive_sha256 = sha256_file(path)
    cache_path = path.with_suffix(".cache.json")
    cache, cache_sha256 = _validate_cache(
        archive_path=path,
        cache_path=cache_path,
        actual_sha256=archive_sha256,
        vector_shape=shape,
        vector_dtype=dtype.name,
    )
    if cache.get("has_meta") is not False:
        raise RQ1DataError(f"{cache_path}: neutral archive must record has_meta=false")
    return {
        "path": _display_path(path, config),
        "sha256": archive_sha256,
        "size": path.stat().st_size,
        "cache_path": _display_path(cache_path, config),
        "cache_sha256": cache_sha256,
        "cache_key": cache["cache_key"],
        "shape": list(shape),
        "dtype": dtype.name,
        "use_policy": "optional joint PCA-complement robustness only",
    }


def _load_raw_stories(
    config: RQ1Config,
) -> tuple[dict[str, tuple[dict[str, Any], ...]], tuple[dict[str, Any], ...]]:
    by_emotion: dict[str, tuple[dict[str, Any], ...]] = {}
    sources: list[dict[str, Any]] = []
    shared_topics: dict[int, str] = {}
    expected_cells = set(config.analysis.expected_cells)

    for emotion in config.emotions.reference:
        path = config.paths.raw_dir / "stories" / f"{emotion}.jsonl"
        try:
            raw_lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RQ1DataError(f"cannot read story source {path}: {exc}") from exc
        if len(raw_lines) != config.analysis.expected_stories_per_emotion:
            raise RQ1DataError(
                f"{path}: expected {config.analysis.expected_stories_per_emotion} rows, "
                f"got {len(raw_lines)}"
            )
        records: list[dict[str, Any]] = []
        cells: set[tuple[int, int]] = set()
        for line_number, line in enumerate(raw_lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RQ1DataError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise RQ1DataError(f"{path}:{line_number}: row must be a JSON object")
            required = {"emotion", "story_idx", "text", "topic", "topic_id"}
            if not required.issubset(record):
                raise RQ1DataError(
                    f"{path}:{line_number}: missing fields {sorted(required - set(record))}"
                )
            if record["emotion"] != emotion:
                raise RQ1DataError(f"{path}:{line_number}: emotion does not match filename")
            if type(record["topic_id"]) is not int or type(record["story_idx"]) is not int:
                raise RQ1DataError(f"{path}:{line_number}: topic_id/story_idx must be integers")
            if not isinstance(record["text"], str) or not record["text"].strip():
                raise RQ1DataError(f"{path}:{line_number}: text must be non-empty")
            if not isinstance(record["topic"], str) or not record["topic"]:
                raise RQ1DataError(f"{path}:{line_number}: topic must be non-empty")
            cell = (record["topic_id"], record["story_idx"])
            if cell not in expected_cells or cell in cells:
                raise RQ1DataError(f"{path}:{line_number}: invalid or duplicate cell {cell}")
            cells.add(cell)
            prior_topic = shared_topics.setdefault(record["topic_id"], record["topic"])
            if prior_topic != record["topic"]:
                raise RQ1DataError(
                    f"{path}:{line_number}: topic text differs across reference emotions"
                )
            records.append({**record, "raw_line_number": line_number})
        if cells != expected_cells:
            raise RQ1DataError(f"{path}: topic/story grid is incomplete")
        by_emotion[emotion] = tuple(
            sorted(records, key=lambda row: (row["topic_id"], row["story_idx"]))
        )
        sources.append(
            {
                "emotion": emotion,
                "path": _display_path(path, config),
                "sha256": sha256_file(path),
                "rows": len(records),
            }
        )
    if set(shared_topics) != set(range(20)):
        raise RQ1DataError("raw stories must share exactly topic IDs 0 through 19")
    return by_emotion, tuple(sources)


def _validate_run_manifests(config: RQ1Config) -> tuple[dict[str, Any], ...]:
    paths = (
        ("generation", config.paths.raw_dir / "run_manifest.json"),
        ("aligned_activation", config.paths.aligned_activations_dir / "run_manifest.json"),
        ("aligned_vectors", config.paths.aligned_vectors_path.parent / "run_manifest.json"),
    )
    results: list[dict[str, Any]] = []
    for name, path in paths:
        manifest = _read_json_object(path, f"{name} run manifest")
        run_config = manifest.get("config")
        if not isinstance(run_config, dict):
            raise RQ1DataError(f"{path}: config must be an object")
        if tuple(run_config.get("emotions", ())) != config.emotions.reference:
            raise RQ1DataError(f"{path}: reference emotion order mismatch")
        if run_config.get("model_name") != config.models.base_model_id:
            raise RQ1DataError(f"{path}: base model ID mismatch")
        if run_config.get("model_revision") != config.models.base_model_revision:
            raise RQ1DataError(f"{path}: base model revision mismatch")
        if run_config.get("token_start") != config.analysis.token_start:
            raise RQ1DataError(f"{path}: token_start mismatch")
        if len(run_config.get("topics", ())) != 20:
            raise RQ1DataError(f"{path}: expected 20 shared topics")
        results.append(
            {
                "kind": name,
                "path": _display_path(path, config),
                "sha256": sha256_file(path),
                "stage": manifest.get("stage"),
                "config_hash": manifest.get("config_hash"),
            }
        )
    return tuple(results)


def _index_jsonl(rows: tuple[StoryRecord, ...]) -> str:
    return "".join(f"{_canonical_json(row.as_index_record())}\n" for row in rows)


def _selection_hash(rows: tuple[StoryRecord, ...]) -> str:
    return _sha256_json([row.example_id for row in rows])


def _build_manifest(
    dataset_rows: tuple[StoryRecord, ...],
    activation_metadata: tuple[AlignedActivationMetadata, ...],
    raw_sources: tuple[dict[str, Any], ...],
    run_manifests: tuple[dict[str, Any], ...],
    aligned_vectors: dict[str, Any],
    neutral_activations: dict[str, Any],
    config: RQ1Config,
) -> dict[str, Any]:
    same_rows = tuple(row for row in dataset_rows if row.same_example_eligible)
    confirmatory_rows = tuple(row for row in dataset_rows if row.confirmatory_eligible)
    same_counts = Counter(row.emotion for row in same_rows)
    confirmatory_counts = Counter(row.emotion for row in confirmatory_rows)
    split_counts = Counter(row.split for row in confirmatory_rows)
    missing = tuple(row for row in dataset_rows if not row.same_example_eligible)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_id": config.study_id,
        "config": {
            "path": _display_path(config.source_path, config),
            "sha256": config.source_sha256,
        },
        "models": asdict(config.models),
        "equality_provenance": asdict(config.equality_provenance),
        "selection": {
            "confirmatory": {
                "description": (
                    "balanced complete-cell set; remove topic_id=2/story_idx=3 from every emotion"
                ),
                "rows": len(confirmatory_rows),
                "rows_per_emotion": dict(sorted(confirmatory_counts.items())),
                "rows_per_split": dict(sorted(split_counts.items())),
                "sha256": _selection_hash(confirmatory_rows),
                "use": "primary cross-model comparisons and all label permutations",
            },
            "full_intersection_sensitivity": {
                "description": "all rows with existing aligned activations",
                "rows": len(same_rows),
                "rows_per_emotion": dict(sorted(same_counts.items())),
                "sha256": _selection_hash(same_rows),
                "use": "sensitivity analysis only because emotion groups are unbalanced",
            },
            "all_raw": {
                "rows": len(dataset_rows),
                "sha256": _selection_hash(dataset_rows),
            },
        },
        "counts": {
            "all_raw_rows": len(dataset_rows),
            "same_example_rows": len(same_rows),
            "confirmatory_rows": len(confirmatory_rows),
            "reference_emotions": len(config.emotions.reference),
            "reported_negative_emotions": len(config.emotions.reported_negative),
            "topics": 20,
            "stories_per_topic": config.analysis.stories_per_topic,
        },
        "exclusions": {
            "missing_aligned_rows": [
                {
                    "example_id": row.example_id,
                    "emotion": row.emotion,
                    "topic_id": row.topic_id,
                    "story_idx": row.story_idx,
                    "reason": row.exclusion_reason,
                }
                for row in missing
            ],
            "balanced_cell_exclusion": {
                "topic_id": config.analysis.expected_missing_topic_id,
                "story_idx": config.analysis.expected_missing_story_idx,
                "observed_missing_emotion": config.analysis.expected_missing_emotion,
                "drop_from_every_reference_emotion": True,
                "dropped_raw_rows": len(config.emotions.reference),
            },
        },
        "index": {
            "path": _display_path(config.paths.dataset_index_path, config),
            "sha256": hashlib.sha256(_index_jsonl(dataset_rows).encode("utf-8")).hexdigest(),
            "schema_version": INDEX_SCHEMA_VERSION,
        },
        "sources": {
            "raw_jsonl": list(raw_sources),
            "aligned_activation_npz": [
                metadata.as_manifest_record(config) for metadata in activation_metadata
            ],
            "aligned_emotion_vectors": aligned_vectors,
            "neutral_activations": neutral_activations,
            "run_manifests": list(run_manifests),
        },
        "topic_splits": {
            "A": list(config.analysis.topic_split_a),
            "B": list(config.analysis.topic_split_b),
            "rule": "all four stories sharing a topic remain in the same split",
        },
        "reporting_scope": {
            "reference_emotions_used_to_construct_vectors": list(config.emotions.reference),
            "emotions_reported": list(config.emotions.reported_negative),
            "preregistered_groups": {
                group.name: list(group.emotions) for group in config.emotions.preregistered_groups
            },
        },
    }


def build_reuse_dataset(config: RQ1Config) -> ReuseDataset:
    """Validate all reuse sources and build the deterministic 960-row index."""

    raw_by_emotion, raw_sources = _load_raw_stories(config)
    activation_metadata = tuple(
        inspect_aligned_activation(config, emotion) for emotion in config.emotions.reference
    )
    metadata_by_emotion = {item.emotion: item for item in activation_metadata}

    rows: list[StoryRecord] = []
    missing_cell = config.analysis.balanced_excluded_cell
    for emotion_index, emotion in enumerate(config.emotions.reference):
        metadata = metadata_by_emotion[emotion]
        aligned_lookup = {cell: row for row, cell in enumerate(metadata.rows)}
        for raw in raw_by_emotion[emotion]:
            topic_id = int(raw["topic_id"])
            story_idx = int(raw["story_idx"])
            cell = (topic_id, story_idx)
            aligned_row = aligned_lookup.get(cell)
            same_example = aligned_row is not None
            confirmatory = same_example and cell != missing_cell
            reason = None
            if not same_example:
                reason = "aligned_activation_missing_token_count_below_50"
            text = str(raw["text"])
            rows.append(
                StoryRecord(
                    canonical_row=len(rows),
                    example_id=f"{emotion}:t{topic_id:02d}:s{story_idx:02d}",
                    emotion=emotion,
                    emotion_index=emotion_index,
                    topic_id=topic_id,
                    story_idx=story_idx,
                    topic=str(raw["topic"]),
                    split="A" if topic_id in config.analysis.topic_split_a else "B",
                    text=text,
                    text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    raw_source=f"stories/{emotion}.jsonl",
                    raw_line_number=int(raw["raw_line_number"]),
                    aligned_source=f"stories/{emotion}.npz",
                    aligned_row=aligned_row,
                    same_example_eligible=same_example,
                    confirmatory_eligible=confirmatory,
                    exclusion_reason=reason,
                )
            )
    row_tuple = tuple(rows)
    _validate_selection_contract(row_tuple, activation_metadata, config)

    run_manifests = _validate_run_manifests(config)
    aligned_vectors = _inspect_vector_archive(config)
    neutral_activations = _inspect_neutral_archive(config)
    manifest = _build_manifest(
        row_tuple,
        activation_metadata,
        raw_sources,
        run_manifests,
        aligned_vectors,
        neutral_activations,
        config,
    )
    return ReuseDataset(
        config=config,
        rows=row_tuple,
        activation_metadata=activation_metadata,
        manifest=manifest,
    )


def _validate_selection_contract(
    rows: tuple[StoryRecord, ...],
    activation_metadata: tuple[AlignedActivationMetadata, ...],
    config: RQ1Config,
) -> None:
    if len(rows) != 960:
        raise RQ1DataError(f"expected 960 canonical raw rows, got {len(rows)}")
    same = tuple(row for row in rows if row.same_example_eligible)
    confirmatory = tuple(row for row in rows if row.confirmatory_eligible)
    if len(same) != 959:
        raise RQ1DataError(f"expected 959 same-example rows, got {len(same)}")
    missing = tuple(row for row in rows if not row.same_example_eligible)
    expected_missing = (
        config.analysis.expected_missing_emotion,
        config.analysis.expected_missing_topic_id,
        config.analysis.expected_missing_story_idx,
    )
    observed_missing = tuple((row.emotion, row.topic_id, row.story_idx) for row in missing)
    if observed_missing != (expected_missing,):
        raise RQ1DataError(
            f"expected only missing aligned row {expected_missing}, got {observed_missing}"
        )
    if len(confirmatory) != 948:
        raise RQ1DataError(f"expected 948 balanced confirmatory rows, got {len(confirmatory)}")
    confirmatory_counts = Counter(row.emotion for row in confirmatory)
    if set(confirmatory_counts.values()) != {79}:
        raise RQ1DataError(
            f"confirmatory set must contain 79 rows per emotion: {confirmatory_counts}"
        )
    expected_shapes = {
        emotion: 79 if emotion == config.analysis.expected_missing_emotion else 80
        for emotion in config.emotions.reference
    }
    actual_shapes = {item.emotion: item.vector_shape[0] for item in activation_metadata}
    if actual_shapes != expected_shapes:
        raise RQ1DataError(f"aligned activation row counts differ from expected: {actual_shapes}")


def write_dataset_artifacts(dataset: ReuseDataset) -> tuple[Path, Path]:
    """Write the deterministic small index and manifest; never copy activations."""

    index_path = dataset.config.paths.dataset_index_path
    manifest_path = dataset.config.paths.dataset_manifest_path
    index_text = dataset.index_jsonl()
    actual_index_hash = hashlib.sha256(index_text.encode("utf-8")).hexdigest()
    if dataset.manifest["index"]["sha256"] != actual_index_hash:
        raise RQ1DataError("in-memory index hash does not match manifest")
    _atomic_write_text(index_path, index_text)
    manifest_text = json.dumps(
        dataset.manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    _atomic_write_text(manifest_path, f"{manifest_text}\n")
    return index_path, manifest_path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_dataset_index(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load a previously written small canonical index with basic integrity checks."""

    records: list[dict[str, Any]] = []
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RQ1DataError(f"cannot read canonical index {source}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RQ1DataError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict) or record.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise RQ1DataError(f"{source}:{line_number}: unsupported index record")
        if record.get("canonical_row") != line_number - 1:
            raise RQ1DataError(f"{source}:{line_number}: non-canonical row order")
        records.append(record)
    return tuple(records)


@contextmanager
def open_aligned_activations(config: RQ1Config, emotion: str) -> Iterator[np.lib.npyio.NpzFile]:
    """Open one aligned NPZ lazily; arrays are loaded only when explicitly indexed."""

    if emotion not in config.emotions.reference:
        raise KeyError(emotion)
    path = config.paths.aligned_activations_dir / "stories" / f"{emotion}.npz"
    with np.load(path, allow_pickle=False) as archive:
        yield archive


@contextmanager
def open_aligned_vectors(config: RQ1Config) -> Iterator[np.lib.npyio.NpzFile]:
    """Open the small aligned vector archive without eagerly copying its arrays."""

    with np.load(config.paths.aligned_vectors_path, allow_pickle=False) as archive:
        yield archive
