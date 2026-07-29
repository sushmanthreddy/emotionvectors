#!/usr/bin/env python3
"""Model-agnostic emotional-story activation and raw-vector extraction.

The extraction contract is intentionally separate from neutral activation
extraction and neutral PCA.  Accepted emotional stories are treated as plain
text, transformer-layer outputs are averaged over valid tokens beginning at a
one-based position, and one story-level vector is retained for every layer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..generation.emotion_config import (
    EmotionSpec,
    emotion_slug,
    load_emotion_config,
)


SCHEMA_VERSION = 1
DATASET_FINGERPRINT_SCHEMA_VERSION = 1
SHARD_SCHEMA_VERSION = 1
CONSOLIDATED_SCHEMA_VERSION = 1
INPUT_TEXT_FIELD = "story"
TOKEN_POSITION_INDEXING = "one-based"
HIDDEN_STATE_MAPPING = "saved layer l equals outputs.hidden_states[l + 1]"
RESIDUAL_STREAM_DEFINITION = "output after each transformer layer"
NORMALIZATION_EPSILON = 1e-12
DEFAULT_START_TOKEN_POSITION = 50
DEFAULT_RECORDS_PER_SHARD = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_BATCH_TOKENS = 8192
DEFAULT_PAD_TO_MULTIPLE_OF = 8
DEFAULT_ACTIVATION_DTYPE = "float16"
CANONICAL_14B_MODEL = "Qwen/Qwen2.5-14B-Instruct"
CANONICAL_14B_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
CANONICAL_14B_EMOTIONS = 45
CANONICAL_14B_LAYERS = 48
CANONICAL_14B_HIDDEN_SIZE = 5120


@dataclass(frozen=True)
class StoryRecord:
    """One strict accepted-story record in canonical dataset order."""

    record_id: str
    emotion: str
    emotion_slug: str
    story: str
    source_file: Path
    source_line: int
    topic_id: Any
    sample_index: Any
    payload: dict[str, Any]


@dataclass(frozen=True)
class DatasetInspection:
    """Validated dataset state and its immutable fingerprint."""

    emotion_specs: tuple[EmotionSpec, ...]
    records_by_emotion: dict[str, tuple[StoryRecord, ...]]
    records_dir: Path
    emotion_config_path: Path
    generation_manifest_path: Path
    generation_config_path: Path
    generation_manifest: dict[str, Any]
    generation_config: dict[str, Any]
    fingerprint: dict[str, Any]

    @property
    def emotion_order(self) -> tuple[str, ...]:
        return tuple(spec.emotion for spec in self.emotion_specs)

    @property
    def slug_by_emotion(self) -> dict[str, str]:
        return {spec.emotion: spec.slug for spec in self.emotion_specs}

    @property
    def fingerprint_sha256(self) -> str:
        value = self.fingerprint.get("dataset_fingerprint_sha256")
        if not isinstance(value, str):
            raise ValueError("Dataset fingerprint has no canonical SHA-256")
        return value

    @property
    def total_records(self) -> int:
        return sum(len(rows) for rows in self.records_by_emotion.values())

    def iter_records(self) -> Iterable[StoryRecord]:
        for emotion in self.emotion_order:
            yield from self.records_by_emotion[emotion]


@dataclass(frozen=True)
class TokenIndexRecord:
    """Precomputed plain-text tokenizer length for one story."""

    record_id: str
    emotion: str
    emotion_slug: str
    source_file: str
    source_line: int
    token_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "emotion": self.emotion,
            "emotion_slug": self.emotion_slug,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "token_count": self.token_count,
        }


@dataclass
class TokenizerInspection:
    """Tokenizer preflight report plus canonical token index."""

    records: tuple[TokenIndexRecord, ...]
    report: dict[str, Any]
    tokenizer_bundle: dict[str, Any] | None = None

    @property
    def by_record_id(self) -> dict[str, TokenIndexRecord]:
        return {row.record_id: row for row in self.records}


@dataclass(frozen=True)
class ActivationShardPlan:
    """One fixed canonical shard for one configured emotion."""

    emotion: str
    emotion_slug: str
    shard_index: int
    records: tuple[StoryRecord, ...]
    token_counts: tuple[int, ...]

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(record.record_id for record in self.records)


@dataclass(frozen=True)
class ActivationShardExpectation:
    """Immutable identity and tensor contract for a shard."""

    plan: ActivationShardPlan
    model_name: str
    model_revision: str
    dataset_fingerprint: str
    start_token_position: int
    num_layers: int
    hidden_size: int
    activation_dtype: str


@dataclass(frozen=True)
class ShardInspection:
    """Verified and corrupt shard accounting for resumable extraction."""

    valid: dict[tuple[str, int], Path]
    corrupt: dict[tuple[str, int], str]


class ExtractionOutOfMemory(RuntimeError):
    """A resumable CUDA OOM with the failed batch context."""

    def __init__(
        self,
        *,
        emotion: str,
        shard_index: int,
        record_ids: Sequence[str],
        token_counts: Sequence[int],
        batch_size: int,
        original_error: BaseException,
    ) -> None:
        super().__init__(str(original_error))
        self.emotion = emotion
        self.shard_index = shard_index
        self.record_ids = list(record_ids)
        self.token_counts = [int(value) for value in token_counts]
        self.batch_size = int(batch_size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "story_activation_extraction",
            "emotion": self.emotion,
            "shard_index": self.shard_index,
            "record_ids": self.record_ids,
            "token_counts": self.token_counts,
            "batch_size": self.batch_size,
            "error": str(self),
            "recommendation": (
                "Rerun with --resume and a smaller --batch-size or "
                "--max-batch-tokens."
            ),
        }


class ShortStoryError(ValueError):
    """Tokenizer preflight found stories outside the scientific contract."""


class IncompatibleCheckpointError(ValueError):
    """A checkpoint belongs to a different immutable extraction contract."""


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit extraction and vector-computation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Preflight accepted emotional stories, index local tokenizer lengths, "
            "and extract layer-preserving raw emotion vectors."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="Strict CPU-only dataset validation and fingerprinting",
    )
    _add_dataset_arguments(preflight, require_expected_counts=True)
    preflight.add_argument("--output-dir", type=Path, required=True)

    tokenize = subparsers.add_parser(
        "tokenize-preflight",
        help="Pinned local-tokenizer length indexing without model weights",
    )
    _add_tokenizer_arguments(tokenize)
    tokenize.add_argument("--records-dir", type=Path, required=True)
    tokenize.add_argument("--emotion-config", type=Path, required=True)
    tokenize.add_argument(
        "--start-token-position",
        type=_positive_int,
        default=DEFAULT_START_TOKEN_POSITION,
    )
    tokenize.add_argument("--output-dir", type=Path, required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Resumable story activations and raw one-versus-rest vectors",
    )
    _add_dataset_arguments(extract, require_expected_counts=False)
    _add_tokenizer_arguments(extract)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--emotions", nargs="+", default=None)
    extract.add_argument("--max-records-per-emotion", type=_positive_int, default=None)
    extract.add_argument(
        "--start-token-position",
        type=_positive_int,
        default=DEFAULT_START_TOKEN_POSITION,
    )
    extract.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
    )
    extract.add_argument(
        "--max-batch-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_BATCH_TOKENS,
    )
    extract.add_argument(
        "--pad-to-multiple-of",
        type=_positive_int,
        default=DEFAULT_PAD_TO_MULTIPLE_OF,
    )
    extract.add_argument(
        "--records-per-shard",
        type=_positive_int,
        default=DEFAULT_RECORDS_PER_SHARD,
    )
    extract.add_argument(
        "--activation-dtype",
        choices=("float16", "float32"),
        default=DEFAULT_ACTIVATION_DTYPE,
    )
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--expected-layers", type=_positive_int, default=None)
    extract.add_argument("--expected-hidden-size", type=_positive_int, default=None)
    extract.add_argument("--resume", action="store_true")

    compute = subparsers.add_parser(
        "compute-vectors",
        help=(
            "Compute one-versus-rest vectors from saved consolidated story "
            "activations without loading a model"
        ),
    )
    compute.add_argument("--activation-run", type=Path, required=True)
    compute.add_argument("--story-activations-dir", type=Path, required=True)
    compute.add_argument("--emotion-config", type=Path, required=True)
    compute.add_argument("--expected-layers", type=_positive_int, required=True)
    compute.add_argument(
        "--expected-hidden-size",
        type=_positive_int,
        required=True,
    )
    compute.add_argument(
        "--accumulation-dtype",
        choices=("float64", "float32"),
        default="float64",
    )
    compute.add_argument(
        "--compute-device",
        default="cpu",
        help="Reduction device, for example cpu or cuda:0",
    )
    return parser


def _add_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_expected_counts: bool,
) -> None:
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--emotion-config", type=Path, required=True)
    parser.add_argument(
        "--expected-emotions",
        type=_positive_int,
        required=require_expected_counts,
        default=None,
    )
    parser.add_argument(
        "--expected-records-per-emotion",
        type=_positive_int,
        required=require_expected_counts,
        default=None,
    )
    parser.add_argument(
        "--expected-total-records",
        type=_positive_int,
        required=require_expected_counts,
        default=None,
    )


def _add_tokenizer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "bf16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CLI stage and emit clean, resumable failure codes."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            run_dataset_preflight(args)
        elif args.command == "tokenize-preflight":
            run_tokenizer_preflight_command(args)
        elif args.command == "extract":
            run_extraction(args)
        elif args.command == "compute-vectors":
            run_compute_vectors(args)
        else:  # pragma: no cover - argparse enforces this
            raise ValueError(f"Unsupported command: {args.command!r}")
    except KeyboardInterrupt:
        print("Story activation extraction interrupted", file=sys.stderr)
        return 130
    except ExtractionOutOfMemory as error:
        output_dir = getattr(args, "output_dir", None)
        if isinstance(output_dir, Path):
            mark_progress_failure(output_dir, "out_of_memory", error.as_dict())
        print(
            "GPU out of memory. Verified shards remain resumable; rerun with "
            "--resume and smaller batch limits.",
            file=sys.stderr,
        )
        return 2
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        output_dir = getattr(args, "output_dir", None)
        if getattr(args, "command", None) == "extract" and isinstance(output_dir, Path):
            mark_progress_failure(
                output_dir,
                "failed",
                {"error": safe_error_message(error)},
            )
        print(f"Story activation stage failed: {error}", file=sys.stderr)
        return 1
    return 0


def run_dataset_preflight(args: argparse.Namespace) -> DatasetInspection:
    """Run strict dataset preflight without importing Transformers."""

    inspection = inspect_dataset(
        records_dir=args.records_dir,
        generation_manifest_path=args.generation_manifest,
        emotion_config_path=args.emotion_config,
        expected_emotions=args.expected_emotions,
        expected_records_per_emotion=args.expected_records_per_emotion,
        expected_total_records=args.expected_total_records,
    )
    write_dataset_preflight_outputs(args.output_dir, inspection)
    print(
        "DATASET_PREFLIGHT_PASSED",
        f"emotions={len(inspection.emotion_specs)}",
        f"records_per_emotion={_uniform_record_count(inspection)}",
        f"total_records={inspection.total_records}",
        f"fingerprint={inspection.fingerprint_sha256}",
    )
    return inspection


def inspect_dataset(
    *,
    records_dir: Path,
    generation_manifest_path: Path,
    emotion_config_path: Path,
    expected_emotions: int | None,
    expected_records_per_emotion: int | None,
    expected_total_records: int | None,
) -> DatasetInspection:
    """Strictly validate every accepted JSONL row and build a fingerprint."""

    records_root = records_dir.expanduser().resolve()
    manifest_path = generation_manifest_path.expanduser().resolve()
    emotion_path = emotion_config_path.expanduser().resolve()
    if not records_root.is_dir():
        raise NotADirectoryError(f"Records directory does not exist: {records_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generation manifest does not exist: {manifest_path}")
    if not emotion_path.is_file():
        raise FileNotFoundError(f"Emotion config does not exist: {emotion_path}")

    emotion_specs = load_emotion_config(emotion_path)
    if expected_emotions is not None and len(emotion_specs) != expected_emotions:
        raise ValueError(
            f"Expected {expected_emotions} configured emotions; "
            f"found {len(emotion_specs)}"
        )

    file_by_emotion = discover_emotion_record_files(
        records_dir=records_root,
        emotion_specs=emotion_specs,
    )
    manifest = read_json_object(manifest_path)
    generation_config_path = manifest_path.parent / "config.json"
    if not generation_config_path.is_file():
        raise FileNotFoundError(
            "Strict generation compatibility requires the sibling config.json: "
            f"{generation_config_path}"
        )
    generation_config = read_json_object(generation_config_path)
    emotion_config_sha256 = sha256_file(emotion_path)
    _validate_generation_compatibility(
        manifest=manifest,
        generation_config=generation_config,
        emotion_specs=emotion_specs,
        emotion_config_sha256=emotion_config_sha256,
        expected_records_per_emotion=expected_records_per_emotion,
        expected_total_records=expected_total_records,
    )

    records_by_emotion: dict[str, tuple[StoryRecord, ...]] = {}
    record_file_fingerprints: dict[str, dict[str, Any]] = {}
    seen_record_ids: dict[str, tuple[Path, int]] = {}
    manifest_model = _required_nonempty_string(
        manifest,
        "generator_model",
        context="generation manifest",
    )
    manifest_revision = _required_nonempty_string(
        manifest,
        "model_revision",
        context="generation manifest",
    )

    for spec in emotion_specs:
        path = file_by_emotion[spec.emotion]
        before = path.stat()
        records: list[StoryRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for source_line, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise ValueError(f"Blank JSONL line: {path}:{source_line}")
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path}:{source_line}: {error.msg}"
                    ) from error
                record = _strict_story_record(
                    row=row,
                    emotion_spec=spec,
                    source_file=path,
                    source_line=source_line,
                )
                previous = seen_record_ids.get(record.record_id)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate record_id {record.record_id!r}: "
                        f"{previous[0]}:{previous[1]} and {path}:{source_line}"
                    )
                seen_record_ids[record.record_id] = (path, source_line)
                row_model = row.get("generator_model")
                if row_model is not None and row_model != manifest_model:
                    raise ValueError(
                        f"Generator model mismatch in {path}:{source_line}: "
                        f"{row_model!r} != {manifest_model!r}"
                    )
                row_revision = row.get("model_revision")
                if row_revision is not None and row_revision != manifest_revision:
                    raise ValueError(
                        f"Model revision mismatch in {path}:{source_line}: "
                        f"{row_revision!r} != {manifest_revision!r}"
                    )
                records.append(record)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"Record file changed during preflight: {path}")
        if (
            expected_records_per_emotion is not None
            and len(records) != expected_records_per_emotion
        ):
            raise ValueError(
                f"Emotion {spec.emotion!r} has {len(records)} records; "
                f"expected {expected_records_per_emotion}"
            )
        records_by_emotion[spec.emotion] = tuple(records)
        record_file_fingerprints[spec.emotion] = {
            "slug": spec.slug,
            "path": str(path),
            "sha256": sha256_file(path),
            "record_count": len(records),
        }

    total_records = sum(len(rows) for rows in records_by_emotion.values())
    if expected_total_records is not None and total_records != expected_total_records:
        raise ValueError(
            f"Dataset has {total_records} records; expected {expected_total_records}"
        )
    if total_records != len(seen_record_ids):
        raise RuntimeError("Global record-ID accounting is inconsistent")
    manifest_accepted = manifest.get("accepted_stories")
    if manifest_accepted != total_records:
        raise ValueError(
            f"Generation manifest accepted_stories is {manifest_accepted!r}; "
            f"validated dataset contains {total_records}"
        )

    fingerprint_core: dict[str, Any] = {
        "schema_version": DATASET_FINGERPRINT_SCHEMA_VERSION,
        "emotion_config_path": str(emotion_path),
        "emotion_config_sha256": emotion_config_sha256,
        "generation_manifest_path": str(manifest_path),
        "generation_manifest_sha256": sha256_file(manifest_path),
        "generation_config_path": str(generation_config_path),
        "generation_config_sha256": sha256_file(generation_config_path),
        "emotion_order": [spec.emotion for spec in emotion_specs],
        "emotion_slugs": [spec.slug for spec in emotion_specs],
        "record_files": record_file_fingerprints,
        "total_emotions": len(emotion_specs),
        "records_per_emotion": _uniform_count_from_mapping(records_by_emotion),
        "total_records": total_records,
    }
    fingerprint = {
        **fingerprint_core,
        "dataset_fingerprint_sha256": canonical_json_sha256(fingerprint_core),
    }
    return DatasetInspection(
        emotion_specs=emotion_specs,
        records_by_emotion=records_by_emotion,
        records_dir=records_root,
        emotion_config_path=emotion_path,
        generation_manifest_path=manifest_path,
        generation_config_path=generation_config_path,
        generation_manifest=manifest,
        generation_config=generation_config,
        fingerprint=fingerprint,
    )


def discover_emotion_record_files(
    *,
    records_dir: Path,
    emotion_specs: Sequence[EmotionSpec],
) -> dict[str, Path]:
    """Resolve exactly one shared-slug JSONL file for every configured label."""

    expected_by_name = {
        f"{spec.slug}.jsonl": spec for spec in emotion_specs
    }
    actual_by_name = {
        path.name: path.resolve()
        for path in records_dir.iterdir()
        if path.is_file() and path.suffix == ".jsonl"
    }
    missing = sorted(set(expected_by_name) - set(actual_by_name))
    unexpected = sorted(set(actual_by_name) - set(expected_by_name))
    if missing or unexpected:
        raise ValueError(
            "Emotion record-file set mismatch: "
            f"missing={missing!r} unexpected={unexpected!r}"
        )
    return {
        spec.emotion: actual_by_name[f"{spec.slug}.jsonl"]
        for spec in emotion_specs
    }


def _strict_story_record(
    *,
    row: Any,
    emotion_spec: EmotionSpec,
    source_file: Path,
    source_line: int,
) -> StoryRecord:
    if not isinstance(row, dict):
        raise ValueError(f"JSON value is not an object: {source_file}:{source_line}")
    for field in ("record_id", "emotion", "story"):
        if field not in row:
            raise ValueError(
                f"Missing {field!r} in {source_file}:{source_line}"
            )
    record_id = row["record_id"]
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(
            f"record_id is not a nonempty string: {source_file}:{source_line}"
        )
    if row["emotion"] != emotion_spec.emotion:
        raise ValueError(
            f"Emotion mismatch in {source_file}:{source_line}: "
            f"{row['emotion']!r} != {emotion_spec.emotion!r}"
        )
    story = row["story"]
    if not isinstance(story, str) or not story.strip():
        raise ValueError(
            f"story is not a nonempty string: {source_file}:{source_line}"
        )
    return StoryRecord(
        record_id=record_id,
        emotion=emotion_spec.emotion,
        emotion_slug=emotion_spec.slug,
        story=story,
        source_file=source_file,
        source_line=source_line,
        topic_id=row.get("topic_id"),
        sample_index=row.get("sample_index"),
        payload=row,
    )


def _validate_generation_compatibility(
    *,
    manifest: dict[str, Any],
    generation_config: dict[str, Any],
    emotion_specs: Sequence[EmotionSpec],
    emotion_config_sha256: str,
    expected_records_per_emotion: int | None,
    expected_total_records: int | None,
) -> None:
    """Cross-check aggregate manifest state with its immutable run config."""

    if manifest.get("status") != "completed":
        raise ValueError(
            f"Generation manifest status is {manifest.get('status')!r}; "
            "expected 'completed'"
        )
    if manifest.get("failed_stories") != 0:
        raise ValueError(
            f"Generation manifest failed_stories is "
            f"{manifest.get('failed_stories')!r}; expected 0"
        )
    if generation_config.get("schema_version") != 2:
        raise ValueError("Generation config must use schema_version 2")
    emotion_order = [spec.emotion for spec in emotion_specs]
    slug_mapping = {spec.emotion: spec.slug for spec in emotion_specs}
    if generation_config.get("emotions") != emotion_order:
        raise ValueError("Generation config emotion order differs from emotion config")
    if generation_config.get("emotion_record_stems") != slug_mapping:
        raise ValueError("Generation config label-to-slug mapping is incompatible")
    if generation_config.get("emotion_config_sha256") != emotion_config_sha256:
        raise ValueError("Generation config emotion-config SHA-256 is incompatible")

    paired_fields = (
        ("generator_model", "generator_model"),
        ("model_revision", "model_revision"),
        ("prompt_version", "prompt_version"),
        ("base_seed", "base_seed"),
        ("samples_per_pair", "samples_per_pair"),
        ("max_attempts", "max_attempts"),
        ("generation_parameters", "generation_parameters"),
    )
    for manifest_field, config_field in paired_fields:
        if manifest.get(manifest_field) != generation_config.get(config_field):
            raise ValueError(
                f"Generation manifest/config mismatch for {manifest_field}"
            )
    if manifest.get("number_of_emotions") != len(emotion_specs):
        raise ValueError("Generation manifest has an incompatible emotion count")
    topics = generation_config.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("Generation config contains no ordered topics")
    if manifest.get("number_of_topics") != len(topics):
        raise ValueError("Generation manifest/config topic count mismatch")
    samples_per_pair = generation_config.get("samples_per_pair")
    if isinstance(samples_per_pair, bool) or not isinstance(samples_per_pair, int):
        raise ValueError("Generation config samples_per_pair is invalid")
    intended = len(emotion_specs) * len(topics) * samples_per_pair
    if manifest.get("intended_stories") != intended:
        raise ValueError("Generation manifest intended-story arithmetic is invalid")
    if manifest.get("accepted_stories") != intended:
        raise ValueError("Generation manifest is not a complete accepted dataset")
    if (
        expected_records_per_emotion is not None
        and len(topics) * samples_per_pair != expected_records_per_emotion
    ):
        raise ValueError(
            "Expected records per emotion are incompatible with topics × samples"
        )
    if expected_total_records is not None and intended != expected_total_records:
        raise ValueError(
            f"Generation config resolves {intended} records; "
            f"expected {expected_total_records}"
        )


def write_dataset_preflight_outputs(
    output_dir: Path,
    inspection: DatasetInspection,
) -> None:
    """Persist the immutable fingerprint and a human-readable preflight report."""

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint_path = destination / "dataset_fingerprint.json"
    if fingerprint_path.exists():
        previous = read_json_object(fingerprint_path)
        if previous != inspection.fingerprint:
            raise IncompatibleCheckpointError(
                f"Existing dataset fingerprint differs: {fingerprint_path}"
            )
    atomic_write_json(fingerprint_path, inspection.fingerprint)
    counts = {
        emotion: len(inspection.records_by_emotion[emotion])
        for emotion in inspection.emotion_order
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "checked_at": utc_now(),
        "dataset_fingerprint": inspection.fingerprint_sha256,
        "records_directory": str(inspection.records_dir),
        "generation_manifest": str(inspection.generation_manifest_path),
        "generation_config": str(inspection.generation_config_path),
        "emotion_config": str(inspection.emotion_config_path),
        "number_of_emotions": len(inspection.emotion_specs),
        "emotion_order": list(inspection.emotion_order),
        "label_to_slug": inspection.slug_by_emotion,
        "records_per_emotion": counts,
        "total_records": inspection.total_records,
        "generation_status": inspection.generation_manifest.get("status"),
        "failed_stories": inspection.generation_manifest.get("failed_stories"),
        "transformers_model_loaded": False,
    }
    atomic_write_json(destination / "preflight_report.json", report)


def load_inspection_from_existing_fingerprint(
    *,
    records_dir: Path,
    emotion_config_path: Path,
    output_dir: Path,
) -> DatasetInspection:
    """Revalidate the exact dataset referenced by a persisted preflight."""

    fingerprint_path = output_dir.expanduser().resolve() / "dataset_fingerprint.json"
    if not fingerprint_path.is_file():
        raise FileNotFoundError(
            f"Run dataset preflight first; missing {fingerprint_path}"
        )
    persisted = read_json_object(fingerprint_path)
    manifest_raw = persisted.get("generation_manifest_path")
    if not isinstance(manifest_raw, str):
        raise ValueError("Persisted fingerprint has no generation manifest path")
    expected_emotions = persisted.get("total_emotions")
    expected_per_emotion = persisted.get("records_per_emotion")
    expected_total = persisted.get("total_records")
    for field, value in (
        ("total_emotions", expected_emotions),
        ("records_per_emotion", expected_per_emotion),
        ("total_records", expected_total),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Persisted fingerprint {field} is invalid")
    inspection = inspect_dataset(
        records_dir=records_dir,
        generation_manifest_path=Path(manifest_raw),
        emotion_config_path=emotion_config_path,
        expected_emotions=expected_emotions,
        expected_records_per_emotion=expected_per_emotion,
        expected_total_records=expected_total,
    )
    if inspection.fingerprint != persisted:
        raise IncompatibleCheckpointError(
            "Current dataset bytes differ from the persisted fingerprint"
        )
    return inspection


def _uniform_record_count(inspection: DatasetInspection) -> int:
    return _uniform_count_from_mapping(inspection.records_by_emotion)


def _uniform_count_from_mapping(
    records_by_emotion: Mapping[str, Sequence[Any]],
) -> int:
    counts = {len(rows) for rows in records_by_emotion.values()}
    if len(counts) != 1:
        raise ValueError(f"Per-emotion record counts are not uniform: {sorted(counts)}")
    return next(iter(counts))


def run_tokenizer_preflight_command(
    args: argparse.Namespace,
) -> TokenizerInspection:
    """Run the local tokenizer preflight after exact dataset preflight."""

    if not args.local_files_only:
        raise ValueError(
            "Tokenizer preflight requires --local-files-only to prevent downloads"
        )
    if not _is_immutable_revision(args.model_revision):
        raise ValueError("--model-revision must be a 40-character commit hash")
    inspection = load_inspection_from_existing_fingerprint(
        records_dir=args.records_dir,
        emotion_config_path=args.emotion_config,
        output_dir=args.output_dir,
    )
    result = create_tokenizer_preflight(
        inspection=inspection,
        model_name=args.model,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        start_token_position=args.start_token_position,
        output_dir=args.output_dir,
    )
    stats = result.report["token_count_statistics"]
    print(
        "TOKENIZER_PREFLIGHT_PASSED",
        f"records={stats['count']}",
        f"min={stats['minimum']}",
        f"max={stats['maximum']}",
        f"short={stats['shorter_than_start_position']}",
        f"layers={result.report['number_of_layers']}",
        f"hidden_size={result.report['hidden_size']}",
    )
    return result


def create_tokenizer_preflight(
    *,
    inspection: DatasetInspection,
    model_name: str,
    model_revision: str,
    cache_dir: Path,
    local_files_only: bool,
    start_token_position: int,
    output_dir: Path,
    tokenizer_bundle: dict[str, Any] | None = None,
) -> TokenizerInspection:
    """Index exact plain-story token counts without loading model weights."""

    if not _is_immutable_revision(model_revision):
        raise ValueError("Model revision must be a 40-character commit hash")
    if not local_files_only:
        raise ValueError("This extraction preflight requires local_files_only=True")
    bundle = tokenizer_bundle or load_local_tokenizer(
        model_name=model_name,
        model_revision=model_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    _validate_canonical_architecture(
        model_name=model_name,
        model_revision=model_revision,
        num_layers=int(bundle["num_layers"]),
        hidden_size=int(bundle["hidden_size"]),
    )
    tokenizer = bundle["tokenizer"]
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"

    token_rows: list[TokenIndexRecord] = []
    short_rows: list[dict[str, Any]] = []
    token_counts: list[int] = []
    for record in inspection.iter_records():
        encoded = tokenizer(
            record.story,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        token_count = _encoded_token_count(encoded)
        row = TokenIndexRecord(
            record_id=record.record_id,
            emotion=record.emotion,
            emotion_slug=record.emotion_slug,
            source_file=str(record.source_file),
            source_line=record.source_line,
            token_count=token_count,
        )
        token_rows.append(row)
        token_counts.append(token_count)
        if token_count < start_token_position:
            short_rows.append(
                {
                    **row.as_dict(),
                    "required_start_token_position": start_token_position,
                    "reason": (
                        f"story contains only {token_count} valid tokens; "
                        f"position {start_token_position} does not exist"
                    ),
                }
            )

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    token_index_path = destination / "story_token_index.jsonl"
    atomic_write_jsonl(
        token_index_path,
        (row.as_dict() for row in token_rows),
    )
    atomic_write_jsonl(destination / "short_stories.jsonl", short_rows)
    token_index_sha256 = sha256_file(token_index_path)
    statistics_payload = token_count_statistics(
        token_counts,
        start_token_position=start_token_position,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed_short_stories" if short_rows else "passed",
        "checked_at": utc_now(),
        "dataset_fingerprint": inspection.fingerprint_sha256,
        "model": model_name,
        "model_revision": model_revision,
        "requested_cache_dir": str(cache_dir.expanduser().resolve()),
        "resolved_hub_cache_dir": str(bundle["resolved_hub_cache_dir"]),
        "resolved_snapshot_dir": str(bundle["resolved_snapshot_dir"]),
        "local_files_only": bool(local_files_only),
        "tokenizer_class": bundle["tokenizer_class"],
        "model_config_class": bundle["model_config_class"],
        "number_of_layers": int(bundle["num_layers"]),
        "hidden_size": int(bundle["hidden_size"]),
        "model_config_dtype": str(bundle["model_config_dtype"]),
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "padding_side": "right",
        "start_token_position": start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "story_token_index": str(token_index_path),
        "story_token_index_sha256": token_index_sha256,
        "short_stories": str(destination / "short_stories.jsonl"),
        "token_count_statistics": statistics_payload,
        "model_weights_loaded": False,
        "transformers_version": bundle["transformers_version"],
    }
    atomic_write_json(destination / "tokenizer_preflight_report.json", report)
    result = TokenizerInspection(
        records=tuple(token_rows),
        report=report,
        tokenizer_bundle=bundle,
    )
    if short_rows:
        raise ShortStoryError(
            f"Tokenizer preflight found {len(short_rows)} stories shorter than "
            f"position {start_token_position}; see "
            f"{destination / 'short_stories.jsonl'}"
        )
    return result


def load_local_tokenizer(
    *,
    model_name: str,
    model_revision: str,
    cache_dir: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    """Load only pinned config/tokenizer objects from a resolved local Hub cache."""

    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for tokenizer preflight"
        ) from error
    resolved_hub = resolve_hub_cache_dir(
        cache_dir=cache_dir,
        model_name=model_name,
        model_revision=model_revision,
    )
    kwargs = {
        "revision": model_revision,
        "cache_dir": str(resolved_hub),
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    num_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    if num_layers < 1 or hidden_size < 1:
        raise ValueError(
            f"Invalid cached model dimensions: layers={num_layers} "
            f"hidden_size={hidden_size}"
        )
    return {
        "tokenizer": tokenizer,
        "config": config,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_config_class": config.__class__.__name__,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "model_config_dtype": getattr(config, "torch_dtype", None),
        "resolved_hub_cache_dir": resolved_hub,
        "resolved_snapshot_dir": resolve_snapshot_dir(
            hub_cache_dir=resolved_hub,
            model_name=model_name,
            model_revision=model_revision,
        ),
        "transformers_version": transformers.__version__,
    }


def resolve_hub_cache_dir(
    *,
    cache_dir: Path,
    model_name: str,
    model_revision: str,
) -> Path:
    """Normalize a user-facing cache root to its actual Hugging Face Hub root."""

    requested = cache_dir.expanduser().resolve()
    candidates = (
        requested / "huggingface" / "hub",
        requested / "hub",
        requested,
    )
    repository = f"models--{model_name.replace('/', '--')}"
    for candidate in candidates:
        snapshot = candidate / repository / "snapshots" / model_revision
        if snapshot.is_dir():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Pinned local snapshot {model_name}@{model_revision} was not found "
        f"under cache candidates: {rendered}"
    )


def resolve_snapshot_dir(
    *,
    hub_cache_dir: Path,
    model_name: str,
    model_revision: str,
) -> Path:
    repository = f"models--{model_name.replace('/', '--')}"
    snapshot = (
        hub_cache_dir.expanduser().resolve()
        / repository
        / "snapshots"
        / model_revision
    )
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned model snapshot does not exist: {snapshot}")
    return snapshot


def _encoded_token_count(encoded: Any) -> int:
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("Tokenizer output contains no input_ids")
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            count = int(input_ids.shape[0])
        elif input_ids.ndim == 2 and input_ids.shape[0] == 1:
            count = int(input_ids.shape[1])
        else:
            raise ValueError(
                f"Unexpected tokenizer tensor shape: {tuple(input_ids.shape)}"
            )
    elif isinstance(input_ids, Sequence) and not isinstance(
        input_ids, (str, bytes)
    ):
        if input_ids and isinstance(input_ids[0], Sequence):
            if len(input_ids) != 1:
                raise ValueError("Tokenizer returned multiple sequences for one story")
            count = len(input_ids[0])
        else:
            count = len(input_ids)
    else:
        raise ValueError("Tokenizer input_ids have an unsupported type")
    if count < 1:
        raise ValueError("Tokenizer produced an empty story sequence")
    return count


def token_count_statistics(
    token_counts: Sequence[int],
    *,
    start_token_position: int,
) -> dict[str, Any]:
    """Return deterministic linear-interpolation percentiles and short count."""

    if not token_counts:
        raise ValueError("At least one token count is required")
    counts = [int(value) for value in token_counts]
    if any(value < 1 for value in counts):
        raise ValueError("Token counts must all be positive")
    ordered = sorted(counts)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p90": _linear_percentile(ordered, 0.90),
        "p95": _linear_percentile(ordered, 0.95),
        "p99": _linear_percentile(ordered, 0.99),
        "shorter_than_start_position": sum(
            value < start_token_position for value in ordered
        ),
    }


def _linear_percentile(ordered: Sequence[int], fraction: float) -> float:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Percentile fraction must lie in [0,1]")
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        value = float(ordered[lower])
    else:
        weight = index - lower
        value = float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
    return int(value) if value.is_integer() else value


def load_tokenizer_inspection(
    *,
    inspection: DatasetInspection,
    model_name: str,
    model_revision: str,
    start_token_position: int,
    output_dir: Path,
) -> TokenizerInspection:
    """Load and strictly validate a persisted tokenizer preflight."""

    root = output_dir.expanduser().resolve()
    report_path = root / "tokenizer_preflight_report.json"
    index_path = root / "story_token_index.jsonl"
    if not report_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Tokenizer preflight artifacts are incomplete")
    report = read_json_object(report_path)
    expected_report = {
        "status": "passed",
        "dataset_fingerprint": inspection.fingerprint_sha256,
        "model": model_name,
        "model_revision": model_revision,
        "local_files_only": True,
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "padding_side": "right",
        "start_token_position": start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
    }
    for field, expected in expected_report.items():
        if report.get(field) != expected:
            raise IncompatibleCheckpointError(
                f"Tokenizer preflight mismatch for {field}: "
                f"{report.get(field)!r} != {expected!r}"
            )
    observed_sha = sha256_file(index_path)
    if report.get("story_token_index_sha256") != observed_sha:
        raise IncompatibleCheckpointError(
            "Persisted story token index SHA-256 does not match its report"
        )
    expected_records = list(inspection.iter_records())
    token_rows: list[TokenIndexRecord] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank token-index line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Token-index line {line_number} is not an object")
            try:
                token_row = TokenIndexRecord(
                    record_id=str(row["record_id"]),
                    emotion=str(row["emotion"]),
                    emotion_slug=str(row["emotion_slug"]),
                    source_file=str(row["source_file"]),
                    source_line=int(row["source_line"]),
                    token_count=int(row["token_count"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid token-index line {line_number}: {error}"
                ) from error
            token_rows.append(token_row)
    if len(token_rows) != len(expected_records):
        raise ValueError(
            f"Token index has {len(token_rows)} rows; "
            f"expected {len(expected_records)}"
        )
    for expected, observed in zip(expected_records, token_rows, strict=True):
        required = (
            expected.record_id,
            expected.emotion,
            expected.emotion_slug,
            str(expected.source_file),
            expected.source_line,
        )
        actual = (
            observed.record_id,
            observed.emotion,
            observed.emotion_slug,
            observed.source_file,
            observed.source_line,
        )
        if actual != required or observed.token_count < 1:
            raise ValueError(
                f"Token-index row is incompatible for {expected.record_id!r}"
            )
        if observed.token_count < start_token_position:
            raise ShortStoryError(
                f"Story {observed.record_id!r} has only "
                f"{observed.token_count} tokens"
            )
    return TokenizerInspection(records=tuple(token_rows), report=report)


def build_selected_token_mask(
    attention_mask: torch.Tensor,
    *,
    start_token_position: int,
) -> torch.Tensor:
    """Select valid tokens at or after a one-based content position."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch,sequence]")
    if start_token_position < 1:
        raise ValueError("start_token_position must be at least 1")
    content_positions = attention_mask.long().cumsum(dim=1)
    return attention_mask.bool() & (content_positions >= start_token_position)


def mean_transformer_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    *,
    attention_mask: torch.Tensor,
    start_token_position: int,
    num_layers: int,
    hidden_size: int,
) -> torch.Tensor:
    """Exclude embeddings and compute FP32 token means for every layer."""

    if len(hidden_states) != num_layers + 1:
        raise ValueError(
            f"Hidden-state tuple has {len(hidden_states)} entries; "
            f"expected {num_layers + 1}"
        )
    selected_mask = build_selected_token_mask(
        attention_mask,
        start_token_position=start_token_position,
    )
    selected_counts = selected_mask.sum(dim=1, keepdim=True)
    if bool(torch.any(selected_counts <= 0)):
        raise ValueError(
            "At least one story has no valid token at the configured start position"
        )
    batch_size, sequence_length = attention_mask.shape
    means: list[torch.Tensor] = []
    for layer_index in range(num_layers):
        hidden = hidden_states[layer_index + 1]
        expected_shape = (batch_size, sequence_length, hidden_size)
        if tuple(hidden.shape) != expected_shape:
            raise ValueError(
                f"Layer {layer_index} hidden state has shape "
                f"{tuple(hidden.shape)}; expected {expected_shape}"
            )
        mask = selected_mask.to(hidden.device).unsqueeze(-1)
        counts = selected_counts.to(hidden.device, dtype=torch.float32)
        hidden_fp32 = hidden.float()
        summed = (hidden_fp32 * mask).sum(dim=1)
        mean = summed / counts.clamp_min(1)
        means.append(mean.to(device="cpu", dtype=torch.float32))
        del hidden_fp32, summed, mean, mask, counts
    result = torch.stack(means, dim=1)
    expected_result = (batch_size, num_layers, hidden_size)
    if tuple(result.shape) != expected_result:
        raise RuntimeError(
            f"Story activation batch has shape {tuple(result.shape)}; "
            f"expected {expected_result}"
        )
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise ValueError("Story activation batch is not finite float32")
    return result


def build_length_aware_batches(
    token_counts: Sequence[int],
    *,
    batch_size: int,
    max_batch_tokens: int,
    pad_to_multiple_of: int,
) -> list[tuple[int, ...]]:
    """Length-sort deterministically while enforcing both batch limits."""

    if batch_size < 1 or max_batch_tokens < 1 or pad_to_multiple_of < 1:
        raise ValueError("Batch constraints must all be positive")
    counts = [int(value) for value in token_counts]
    if any(value < 1 for value in counts):
        raise ValueError("Token counts must all be positive")
    order = sorted(range(len(counts)), key=lambda index: (counts[index], index))
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        token_count = counts[index]
        singleton_padded = round_up(token_count, pad_to_multiple_of)
        if singleton_padded > max_batch_tokens:
            raise ValueError(
                f"Record index {index} needs padded length {singleton_padded}, "
                f"exceeding max_batch_tokens={max_batch_tokens}"
            )
        proposed_max = max(current_max, token_count)
        proposed_size = len(current) + 1
        proposed_padded = round_up(proposed_max, pad_to_multiple_of)
        if current and (
            proposed_size > batch_size
            or proposed_size * proposed_padded > max_batch_tokens
        ):
            batches.append(tuple(current))
            current = [index]
            current_max = token_count
        else:
            current.append(index)
            current_max = proposed_max
    if current:
        batches.append(tuple(current))
    flattened = [index for batch in batches for index in batch]
    if sorted(flattened) != list(range(len(counts))):
        raise RuntimeError("Length-aware batching lost or duplicated record indices")
    for batch in batches:
        padded = round_up(
            max(counts[index] for index in batch),
            pad_to_multiple_of,
        )
        if len(batch) > batch_size or len(batch) * padded > max_batch_tokens:
            raise RuntimeError("Constructed batch violates its declared limits")
    return batches


def restore_canonical_order(
    batch_results: Sequence[tuple[Sequence[int], torch.Tensor]],
    *,
    total_records: int,
) -> torch.Tensor:
    """Restore length-sorted batch results to fixed canonical record order."""

    if total_records < 1:
        raise ValueError("total_records must be positive")
    result: torch.Tensor | None = None
    seen: set[int] = set()
    for indices, values in batch_results:
        if values.ndim < 1 or values.shape[0] != len(indices):
            raise ValueError("Batch result count does not match its indices")
        if result is None:
            result = torch.empty(
                (total_records, *values.shape[1:]),
                dtype=values.dtype,
                device=values.device,
            )
        elif (
            values.dtype != result.dtype
            or values.device != result.device
            or tuple(values.shape[1:]) != tuple(result.shape[1:])
        ):
            raise ValueError("Batch result tensor contracts are inconsistent")
        for offset, raw_index in enumerate(indices):
            index = int(raw_index)
            if not 0 <= index < total_records or index in seen:
                raise ValueError(f"Invalid or duplicate canonical index: {index}")
            result[index] = values[offset]
            seen.add(index)
    if result is None or seen != set(range(total_records)):
        raise ValueError("Batch results do not cover every canonical record")
    return result


def round_up(value: int, multiple: int) -> int:
    if value < 0 or multiple < 1:
        raise ValueError("round_up requires a nonnegative value and positive multiple")
    return ((value + multiple - 1) // multiple) * multiple


def compute_emotion_sums_and_means(
    activations_by_emotion: Mapping[str, torch.Tensor],
    *,
    emotion_order: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    """Accumulate story means in float64 and save means in float32."""

    sums: dict[str, torch.Tensor] = {}
    means: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    expected_shape: tuple[int, int] | None = None
    for emotion in emotion_order:
        activations = activations_by_emotion[emotion]
        if activations.ndim != 3 or activations.shape[0] < 1:
            raise ValueError(f"Invalid story activations for {emotion!r}")
        shape = (int(activations.shape[1]), int(activations.shape[2]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("Emotion activation layer/hidden shapes differ")
        if not bool(torch.isfinite(activations).all()):
            raise ValueError(f"Non-finite story activation for {emotion!r}")
        count = int(activations.shape[0])
        emotion_sum = activations.to(torch.float64).sum(dim=0)
        sums[emotion] = emotion_sum
        means[emotion] = (emotion_sum / count).to(torch.float32)
        counts[emotion] = count
    return sums, means, counts


def compute_weighted_one_vs_rest(
    emotion_sums: Mapping[str, torch.Tensor],
    emotion_counts: Mapping[str, int],
    *,
    emotion_order: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute story-weighted other means and raw vectors in configured order."""

    if len(emotion_order) < 2:
        raise ValueError("At least two emotions are required for one-versus-rest")
    first_sum = emotion_sums[emotion_order[0]]
    total_sum = torch.zeros_like(first_sum, dtype=torch.float64)
    total_count = 0
    for emotion in emotion_order:
        value = emotion_sums[emotion]
        count = emotion_counts[emotion]
        if value.dtype != torch.float64 or value.shape != first_sum.shape:
            raise ValueError("Emotion sums must be shape-compatible float64 tensors")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid emotion count for {emotion!r}")
        total_sum += value
        total_count += count
    means: dict[str, torch.Tensor] = {}
    raw_vectors: dict[str, torch.Tensor] = {}
    for emotion in emotion_order:
        count = emotion_counts[emotion]
        other_count = total_count - count
        if other_count < 1:
            raise ValueError(f"No valid comparison stories for {emotion!r}")
        emotion_mean64 = emotion_sums[emotion] / count
        other_mean64 = (total_sum - emotion_sums[emotion]) / other_count
        means[emotion] = emotion_mean64.to(torch.float32)
        raw_vectors[emotion] = (emotion_mean64 - other_mean64).to(torch.float32)
    return means, raw_vectors


def normalize_layers(
    tensor: torch.Tensor,
    *,
    epsilon: float = NORMALIZATION_EPSILON,
) -> torch.Tensor:
    """Normalize each layer independently while keeping zero layers finite."""

    if tensor.ndim != 2:
        raise ValueError("Layer-wise normalization expects [layers,hidden]")
    if epsilon <= 0:
        raise ValueError("Normalization epsilon must be positive")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Cannot normalize a non-finite tensor")
    norms = tensor.norm(p=2, dim=-1, keepdim=True)
    normalized = tensor / norms.clamp_min(epsilon)
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("Layer normalization produced NaN or infinity")
    return normalized


def run_compute_vectors(args: argparse.Namespace) -> dict[str, Any]:
    """Compute `[emotion, layer, hidden]` vectors from saved activations only."""

    started_at = utc_now()
    started_monotonic = time.monotonic()
    activation_run = args.activation_run.expanduser().resolve()
    story_activations_dir = args.story_activations_dir.expanduser().resolve()
    emotion_config_path = args.emotion_config.expanduser().resolve()
    if not activation_run.is_dir():
        raise NotADirectoryError(
            f"Activation-run directory does not exist: {activation_run}"
        )
    if not story_activations_dir.is_dir():
        raise NotADirectoryError(
            "Story-activations directory does not exist: "
            f"{story_activations_dir}"
        )
    if not story_activations_dir.is_relative_to(activation_run):
        raise ValueError(
            "The story-activations directory must be inside the activation run"
        )
    if not emotion_config_path.is_file():
        raise FileNotFoundError(
            f"Emotion configuration does not exist: {emotion_config_path}"
        )

    emotion_specs = load_emotion_config(emotion_config_path)
    if len(emotion_specs) != CANONICAL_14B_EMOTIONS:
        raise ValueError(
            f"Expected {CANONICAL_14B_EMOTIONS} configured emotions; "
            f"found {len(emotion_specs)}"
        )
    activation_files = discover_consolidated_activation_files(
        story_activations_dir=story_activations_dir,
        emotion_specs=emotion_specs,
    )
    source_metadata = validate_vector_source_metadata(
        activation_run=activation_run,
        emotion_specs=emotion_specs,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
    )
    print(
        "VECTOR_DEVICE_REQUEST",
        f"value={args.compute_device!r}",
    )
    compute_device = resolve_single_device(args.compute_device)
    accumulation_dtype = vector_accumulation_dtype(args.accumulation_dtype)
    cuda_index: int | None = None
    if compute_device.type == "cuda":
        cuda_index = (
            compute_device.index if compute_device.index is not None else 0
        )
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    print(
        "VECTOR_DEVICE_READY",
        f"device={compute_device}",
        f"accumulation_dtype={accumulation_dtype}",
    )

    emotion_sums, emotion_counts, input_shapes, input_dtypes = (
        compute_emotion_sums_from_activation_files(
            emotion_specs=emotion_specs,
            activation_files=activation_files,
            expected_layers=args.expected_layers,
            expected_hidden_size=args.expected_hidden_size,
            accumulation_dtype=accumulation_dtype,
            compute_device=compute_device,
            expected_model=source_metadata["model"],
            expected_model_revision=source_metadata["model_revision"],
        )
    )
    (
        emotion_means,
        other_emotion_means,
        raw_vectors,
        unit_vectors,
    ) = compute_story_weighted_vector_outputs(
        emotion_sums=emotion_sums,
        emotion_counts=emotion_counts,
        emotion_order=[spec.emotion for spec in emotion_specs],
    )
    stacked_outputs = stack_vector_outputs(
        emotion_specs=emotion_specs,
        emotion_means=emotion_means,
        raw_vectors=raw_vectors,
        unit_vectors=unit_vectors,
    )
    validate_computed_vector_outputs(
        emotion_specs=emotion_specs,
        emotion_counts=emotion_counts,
        emotion_sums=emotion_sums,
        emotion_means=emotion_means,
        other_emotion_means=other_emotion_means,
        raw_vectors=raw_vectors,
        unit_vectors=unit_vectors,
        stacked_outputs=stacked_outputs,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
    )

    cpu_sums = ordered_tensor_mapping_to_cpu(emotion_sums, emotion_specs)
    cpu_means = ordered_tensor_mapping_to_cpu(emotion_means, emotion_specs)
    cpu_other_means = ordered_tensor_mapping_to_cpu(
        other_emotion_means,
        emotion_specs,
    )
    cpu_raw_vectors = ordered_tensor_mapping_to_cpu(raw_vectors, emotion_specs)
    cpu_unit_vectors = ordered_tensor_mapping_to_cpu(unit_vectors, emotion_specs)
    cpu_stacked_outputs = {
        name: {
            **payload,
            payload["tensor_key"]: payload[payload["tensor_key"]]
            .detach()
            .to(device="cpu")
            .contiguous(),
        }
        for name, payload in stacked_outputs.items()
    }
    for payload in cpu_stacked_outputs.values():
        payload.pop("tensor_key")
    norms_payload = {
        spec.emotion: {
            str(layer): float(value)
            for layer, value in enumerate(
                cpu_raw_vectors[spec.emotion]
                .norm(p=2, dim=-1)
                .tolist()
            )
        }
        for spec in emotion_specs
    }

    files_created = save_vector_computation_outputs(
        activation_run=activation_run,
        emotion_specs=emotion_specs,
        emotion_counts=emotion_counts,
        emotion_sums=cpu_sums,
        emotion_means=cpu_means,
        other_emotion_means=cpu_other_means,
        raw_vectors=cpu_raw_vectors,
        unit_vectors=cpu_unit_vectors,
        stacked_outputs=cpu_stacked_outputs,
        norms_payload=norms_payload,
    )
    completed_at = utc_now()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "model": source_metadata["model"],
        "model_revision": source_metadata["model_revision"],
        "activation_run": str(activation_run),
        "story_activations_dir": str(story_activations_dir),
        "emotion_config": str(emotion_config_path),
        "emotion_config_sha256": sha256_file(emotion_config_path),
        "number_of_emotions": len(emotion_specs),
        "emotion_order": [spec.emotion for spec in emotion_specs],
        "emotion_slugs": [spec.slug for spec in emotion_specs],
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "actual_story_counts": {
            spec.emotion: int(emotion_counts[spec.emotion])
            for spec in emotion_specs
        },
        "total_stories": int(sum(emotion_counts.values())),
        "input_shapes": {
            spec.emotion: list(input_shapes[spec.emotion])
            for spec in emotion_specs
        },
        "input_activation_dtypes": {
            spec.emotion: input_dtypes[spec.emotion]
            for spec in emotion_specs
        },
        "sum_source": "quantized consolidated story activations",
        "accumulation_dtype": str(accumulation_dtype),
        "output_dtype": "torch.float32",
        "compute_device": str(compute_device),
        "gpu_name": (
            torch.cuda.get_device_name(cuda_index)
            if cuda_index is not None
            else None
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if cuda_index is not None
            else None
        ),
        "emotion_mean_definition": (
            "mean across stories corresponding to the target emotion at every "
            "layer"
        ),
        "other_mean_definition": (
            "story-weighted mean across stories corresponding to all different "
            "emotions at every layer"
        ),
        "raw_vector_definition": (
            "target emotion mean minus different-emotions mean"
        ),
        "layers_averaged_together": False,
        "emotion_vector_shape": [
            args.expected_layers,
            args.expected_hidden_size,
        ],
        "stacked_vector_shape": [
            len(emotion_specs),
            args.expected_layers,
            args.expected_hidden_size,
        ],
        "files_created": files_created
        + ["vector_computation_metadata.json"],
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_loaded": False,
        "inference_performed": False,
    }
    atomic_write_json(
        activation_run / "vector_computation_metadata.json",
        metadata,
    )
    print(
        "VECTOR_COMPUTATION_COMPLETED",
        f"emotions={len(emotion_specs)}",
        f"stories={sum(emotion_counts.values())}",
        f"shape={[len(emotion_specs), args.expected_layers, args.expected_hidden_size]}",
        f"device={compute_device}",
        f"output={activation_run}",
    )
    return metadata


def discover_consolidated_activation_files(
    *,
    story_activations_dir: Path,
    emotion_specs: Sequence[EmotionSpec],
) -> dict[str, Path]:
    """Map configured labels to canonical consolidated `<slug>.pt` files."""

    result: dict[str, Path] = {}
    missing: list[str] = []
    for spec in emotion_specs:
        path = story_activations_dir / f"{spec.slug}.pt"
        if not path.is_file():
            missing.append(str(path))
        else:
            result[spec.emotion] = path.resolve()
    if missing:
        raise FileNotFoundError(
            "Missing configured consolidated activation files: "
            f"{missing!r}"
        )
    return result


def validate_vector_source_metadata(
    *,
    activation_run: Path,
    emotion_specs: Sequence[EmotionSpec],
    expected_layers: int,
    expected_hidden_size: int,
) -> dict[str, Any]:
    """Validate the minimal completed-run identity used in output metadata."""

    metadata_path = activation_run / "metadata.json"
    summary_path = activation_run / "extraction_summary.json"
    if not metadata_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "The completed activation run must contain metadata.json and "
            "extraction_summary.json"
        )
    metadata = read_json_object(metadata_path)
    summary = read_json_object(summary_path)
    required = {
        "model": CANONICAL_14B_MODEL,
        "model_revision": CANONICAL_14B_REVISION,
        "number_of_layers": expected_layers,
        "hidden_size": expected_hidden_size,
    }
    for field, expected in required.items():
        if metadata.get(field) != expected or summary.get(field) != expected:
            raise IncompatibleCheckpointError(
                f"Activation-run {field} is incompatible with {expected!r}"
            )
    if summary.get("status") != "completed":
        raise IncompatibleCheckpointError(
            f"Activation extraction status is {summary.get('status')!r}"
        )
    emotion_order = [spec.emotion for spec in emotion_specs]
    emotion_slugs = [spec.slug for spec in emotion_specs]
    if (
        metadata.get("ordered_emotions") != emotion_order
        or metadata.get("emotion_slugs") != emotion_slugs
        or summary.get("emotion_order") != emotion_order
        or summary.get("emotion_slugs") != emotion_slugs
    ):
        raise IncompatibleCheckpointError(
            "Activation-run emotion order or slugs differ from the configuration"
        )
    return {
        "model": metadata["model"],
        "model_revision": metadata["model_revision"],
    }


def vector_accumulation_dtype(name: str) -> torch.dtype:
    mapping = {
        "float64": torch.float64,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported vector accumulation dtype: {name!r}"
        ) from error


def compute_emotion_sums_from_activation_files(
    *,
    emotion_specs: Sequence[EmotionSpec],
    activation_files: Mapping[str, Path],
    expected_layers: int,
    expected_hidden_size: int,
    accumulation_dtype: torch.dtype,
    compute_device: torch.device,
    expected_model: str,
    expected_model_revision: str,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, tuple[int, int, int]],
    dict[str, str],
]:
    """Sum `[stories,layers,hidden]` tensors over stories only."""

    emotion_sums: dict[str, torch.Tensor] = {}
    emotion_counts: dict[str, int] = {}
    input_shapes: dict[str, tuple[int, int, int]] = {}
    input_dtypes: dict[str, str] = {}
    for index, spec in enumerate(emotion_specs, start=1):
        path = activation_files[spec.emotion]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Activation payload is not a mapping: {path}")
        if payload.get("emotion") != spec.emotion:
            raise ValueError(
                f"Activation label mismatch in {path}: "
                f"{payload.get('emotion')!r} != {spec.emotion!r}"
            )
        if payload.get("emotion_slug") != spec.slug:
            raise ValueError(
                f"Activation slug mismatch in {path}: "
                f"{payload.get('emotion_slug')!r} != {spec.slug!r}"
            )
        if (
            payload.get("model_name") != expected_model
            or payload.get("model_revision") != expected_model_revision
        ):
            raise IncompatibleCheckpointError(
                f"Activation model identity is incompatible in {path}"
            )
        activations = payload.get("activations")
        if not isinstance(activations, torch.Tensor):
            raise ValueError(f"Activation tensor is missing from {path}")
        if activations.ndim != 3:
            raise ValueError(
                f"Activation tensor in {path} has {activations.ndim} dimensions; "
                "expected [stories,layers,hidden]"
            )
        count, layers, hidden_size = (
            int(activations.shape[0]),
            int(activations.shape[1]),
            int(activations.shape[2]),
        )
        if count < 1:
            raise ValueError(f"Emotion {spec.emotion!r} has no story activations")
        if layers != expected_layers or hidden_size != expected_hidden_size:
            raise ValueError(
                f"Activation shape for {spec.emotion!r} is "
                f"{tuple(activations.shape)}; expected "
                f"[N,{expected_layers},{expected_hidden_size}]"
            )
        if not activations.is_floating_point():
            raise ValueError(f"Activation tensor is not floating point: {path}")
        converted = activations.to(
            device=compute_device,
            dtype=accumulation_dtype,
        )
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(
                f"Activation tensor contains NaN or infinity: {path}"
            )
        emotion_sum = converted.sum(dim=0)
        expected_sum_shape = (expected_layers, expected_hidden_size)
        if tuple(emotion_sum.shape) != expected_sum_shape:
            raise RuntimeError(
                f"Story-only sum destroyed a tensor dimension for "
                f"{spec.emotion!r}"
            )
        emotion_sums[spec.emotion] = emotion_sum
        emotion_counts[spec.emotion] = count
        input_shapes[spec.emotion] = (count, layers, hidden_size)
        input_dtypes[spec.emotion] = str(activations.dtype)
        print(
            "VECTOR_INPUT_SUMMED",
            f"emotion={spec.emotion}",
            f"index={index}/{len(emotion_specs)}",
            f"stories={count}",
            f"shape={tuple(activations.shape)}",
            f"dtype={activations.dtype}",
        )
        del converted, activations, payload
    return emotion_sums, emotion_counts, input_shapes, input_dtypes


def compute_story_weighted_vector_outputs(
    *,
    emotion_sums: Mapping[str, torch.Tensor],
    emotion_counts: Mapping[str, int],
    emotion_order: Sequence[str],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Compute float32 target, weighted-other, raw, and per-layer unit vectors."""

    if len(emotion_order) < 2:
        raise ValueError("At least two emotions are required")
    total_sum = torch.stack(
        [emotion_sums[emotion] for emotion in emotion_order],
        dim=0,
    ).sum(dim=0)
    total_count = sum(emotion_counts[emotion] for emotion in emotion_order)
    emotion_means: dict[str, torch.Tensor] = {}
    other_emotion_means: dict[str, torch.Tensor] = {}
    raw_vectors: dict[str, torch.Tensor] = {}
    unit_vectors: dict[str, torch.Tensor] = {}
    for emotion in emotion_order:
        emotion_count = emotion_counts[emotion]
        other_count = total_count - emotion_count
        if emotion_count < 1:
            raise ValueError(f"Emotion {emotion!r} has no target stories")
        if other_count < 1:
            raise ValueError(
                f"Emotion {emotion!r} has no different-emotion stories"
            )
        emotion_mean_accumulation = emotion_sums[emotion] / emotion_count
        other_mean_accumulation = (
            total_sum - emotion_sums[emotion]
        ) / other_count
        emotion_mean = emotion_mean_accumulation.to(torch.float32)
        other_mean = other_mean_accumulation.to(torch.float32)
        raw_vector = emotion_mean - other_mean
        unit_vector = normalize_layers(raw_vector)
        emotion_means[emotion] = emotion_mean
        other_emotion_means[emotion] = other_mean
        raw_vectors[emotion] = raw_vector
        unit_vectors[emotion] = unit_vector
    return (
        emotion_means,
        other_emotion_means,
        raw_vectors,
        unit_vectors,
    )


def stack_vector_outputs(
    *,
    emotion_specs: Sequence[EmotionSpec],
    emotion_means: Mapping[str, torch.Tensor],
    raw_vectors: Mapping[str, torch.Tensor],
    unit_vectors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Stack ordered `[layers,hidden]` tensors as `[emotions,layers,hidden]`."""

    emotion_order = [spec.emotion for spec in emotion_specs]
    emotion_slugs = [spec.slug for spec in emotion_specs]
    return {
        "emotion_means_stacked.pt": {
            "emotion_order": emotion_order,
            "emotion_slugs": emotion_slugs,
            "means": torch.stack(
                [emotion_means[emotion] for emotion in emotion_order],
                dim=0,
            ),
            "tensor_key": "means",
        },
        "emotion_vectors_stacked.pt": {
            "emotion_order": emotion_order,
            "emotion_slugs": emotion_slugs,
            "vectors": torch.stack(
                [raw_vectors[emotion] for emotion in emotion_order],
                dim=0,
            ),
            "tensor_key": "vectors",
        },
        "emotion_vectors_unit_stacked.pt": {
            "emotion_order": emotion_order,
            "emotion_slugs": emotion_slugs,
            "vectors": torch.stack(
                [unit_vectors[emotion] for emotion in emotion_order],
                dim=0,
            ),
            "tensor_key": "vectors",
        },
    }


def validate_computed_vector_outputs(
    *,
    emotion_specs: Sequence[EmotionSpec],
    emotion_counts: Mapping[str, int],
    emotion_sums: Mapping[str, torch.Tensor],
    emotion_means: Mapping[str, torch.Tensor],
    other_emotion_means: Mapping[str, torch.Tensor],
    raw_vectors: Mapping[str, torch.Tensor],
    unit_vectors: Mapping[str, torch.Tensor],
    stacked_outputs: Mapping[str, Mapping[str, Any]],
    expected_layers: int,
    expected_hidden_size: int,
) -> None:
    """Check all vector outputs retain `[layers,hidden]` dimensions."""

    emotion_order = [spec.emotion for spec in emotion_specs]
    expected_shape = (expected_layers, expected_hidden_size)
    collections = (
        ("emotion sums", emotion_sums),
        ("emotion means", emotion_means),
        ("other-emotions means", other_emotion_means),
        ("raw vectors", raw_vectors),
        ("unit vectors", unit_vectors),
    )
    for name, collection in collections:
        if list(collection) != emotion_order:
            raise ValueError(f"{name} do not preserve configured emotion order")
        for emotion in emotion_order:
            tensor = collection[emotion]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} shape for {emotion!r} is {tuple(tensor.shape)}; "
                    f"expected {expected_shape}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"{name} contain NaN or infinity for {emotion!r}"
                )
    total_count = sum(emotion_counts.values())
    for emotion in emotion_order:
        if emotion_counts[emotion] < 1:
            raise ValueError(f"Emotion {emotion!r} has no stories")
        if total_count - emotion_counts[emotion] < 1:
            raise ValueError(
                f"Emotion {emotion!r} has no comparison stories"
            )
        expected_raw = (
            emotion_means[emotion] - other_emotion_means[emotion]
        )
        if not torch.allclose(
            raw_vectors[emotion],
            expected_raw,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                f"Raw vector subtraction is inconsistent for {emotion!r}"
            )
    expected_stacked_shape = (
        len(emotion_specs),
        expected_layers,
        expected_hidden_size,
    )
    for filename, payload in stacked_outputs.items():
        tensor_key = payload["tensor_key"]
        tensor = payload[tensor_key]
        if tuple(tensor.shape) != expected_stacked_shape:
            raise ValueError(
                f"{filename} has shape {tuple(tensor.shape)}; "
                f"expected {expected_stacked_shape}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{filename} contains NaN or infinity")


def ordered_tensor_mapping_to_cpu(
    tensors: Mapping[str, torch.Tensor],
    emotion_specs: Sequence[EmotionSpec],
) -> dict[str, torch.Tensor]:
    """Copy an ordered emotion mapping to contiguous CPU tensors."""

    return {
        spec.emotion: tensors[spec.emotion]
        .detach()
        .to(device="cpu")
        .contiguous()
        for spec in emotion_specs
    }


def save_vector_computation_outputs(
    *,
    activation_run: Path,
    emotion_specs: Sequence[EmotionSpec],
    emotion_counts: Mapping[str, int],
    emotion_sums: Mapping[str, torch.Tensor],
    emotion_means: Mapping[str, torch.Tensor],
    other_emotion_means: Mapping[str, torch.Tensor],
    raw_vectors: Mapping[str, torch.Tensor],
    unit_vectors: Mapping[str, torch.Tensor],
    stacked_outputs: Mapping[str, Mapping[str, Any]],
    norms_payload: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Atomically write every required vector-computation artifact."""

    emotion_order = [spec.emotion for spec in emotion_specs]
    torch_artifacts: dict[str, Any] = {
        "emotion_sums.pt": dict(emotion_sums),
        "emotion_means.pt": dict(emotion_means),
        "other_emotion_means.pt": dict(other_emotion_means),
        "emotion_vectors_raw.pt": dict(raw_vectors),
        "emotion_vectors_unit.pt": dict(unit_vectors),
        **{
            filename: dict(payload)
            for filename, payload in stacked_outputs.items()
        },
    }
    files_created: list[str] = []
    for filename, payload in torch_artifacts.items():
        atomic_torch_save(activation_run / filename, payload)
        files_created.append(filename)
    atomic_write_json(
        activation_run / "emotion_counts.json",
        {
            emotion: int(emotion_counts[emotion])
            for emotion in emotion_order
        },
    )
    files_created.append("emotion_counts.json")
    atomic_write_json(
        activation_run / "emotion_vector_norms.json",
        norms_payload,
    )
    files_created.append("emotion_vector_norms.json")
    return files_created


def run_extraction(args: argparse.Namespace) -> dict[str, Any]:
    """Extract all selected story activations and finish vector artifacts."""

    if not args.local_files_only:
        raise ValueError("Extraction requires --local-files-only")
    if not _is_immutable_revision(args.model_revision):
        raise ValueError("--model-revision must be a 40-character commit hash")
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_extraction_logger(output_dir / "extraction.log")
    extraction_started_at = utc_now()

    inspection = inspect_dataset(
        records_dir=args.records_dir,
        generation_manifest_path=args.generation_manifest,
        emotion_config_path=args.emotion_config,
        expected_emotions=args.expected_emotions,
        expected_records_per_emotion=args.expected_records_per_emotion,
        expected_total_records=args.expected_total_records,
    )
    write_dataset_preflight_outputs(output_dir, inspection)
    tokenizer_inspection = ensure_tokenizer_preflight(
        inspection=inspection,
        model_name=args.model,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        start_token_position=args.start_token_position,
        output_dir=output_dir,
    )
    num_layers = int(tokenizer_inspection.report["number_of_layers"])
    hidden_size = int(tokenizer_inspection.report["hidden_size"])
    if args.expected_layers is not None and num_layers != args.expected_layers:
        raise ValueError(
            f"Cached model reports {num_layers} layers; "
            f"expected {args.expected_layers}"
        )
    if (
        args.expected_hidden_size is not None
        and hidden_size != args.expected_hidden_size
    ):
        raise ValueError(
            f"Cached model reports hidden size {hidden_size}; "
            f"expected {args.expected_hidden_size}"
        )
    _validate_canonical_architecture(
        model_name=args.model,
        model_revision=args.model_revision,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )

    selected_specs = select_emotion_specs(
        inspection.emotion_specs,
        requested=args.emotions,
    )
    selected_records = select_records_by_emotion(
        inspection=inspection,
        selected_specs=selected_specs,
        max_records_per_emotion=args.max_records_per_emotion,
    )
    token_by_id = tokenizer_inspection.by_record_id
    plans = build_activation_shard_plans(
        selected_specs=selected_specs,
        records_by_emotion=selected_records,
        token_by_record_id=token_by_id,
        records_per_shard=args.records_per_shard,
    )
    compatibility_signature = build_extraction_compatibility_signature(
        inspection=inspection,
        selected_specs=selected_specs,
        selected_records=selected_records,
        token_index_sha256=str(
            tokenizer_inspection.report["story_token_index_sha256"]
        ),
        model_name=args.model,
        model_revision=args.model_revision,
        start_token_position=args.start_token_position,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation_dtype=args.activation_dtype,
        records_per_shard=args.records_per_shard,
    )
    config = prepare_extraction_config(
        args=args,
        output_dir=output_dir,
        compatibility_signature=compatibility_signature,
        extraction_started_at=extraction_started_at,
    )
    expectations = tuple(
        ActivationShardExpectation(
            plan=plan,
            model_name=args.model,
            model_revision=args.model_revision,
            dataset_fingerprint=inspection.fingerprint_sha256,
            start_token_position=args.start_token_position,
            num_layers=num_layers,
            hidden_size=hidden_size,
            activation_dtype=args.activation_dtype,
        )
        for plan in plans
    )
    shard_root = output_dir / "story_activation_shards"
    shard_inspection = inspect_activation_shards(shard_root, expectations)
    if (shard_inspection.valid or shard_inspection.corrupt) and not args.resume:
        raise ValueError(
            "Existing extraction shards require --resume; refusing to overwrite "
            "without explicit resume"
        )
    valid_shards = dict(shard_inspection.valid)
    total_records = sum(len(plan.records) for plan in plans)
    completed_records = sum(
        len(expectation.plan.records)
        for expectation in expectations
        if _expectation_key(expectation) in valid_shards
    )
    update_extraction_progress(
        output_dir,
        status="running",
        stage="inspecting_activation_shards",
        total_shards=len(expectations),
        completed_shards=len(valid_shards),
        corrupt_shards=len(shard_inspection.corrupt),
        total_records=total_records,
        completed_records=completed_records,
    )
    logger.info(
        "shard_preflight total=%d valid=%d corrupt=%d selected_records=%d",
        len(expectations),
        len(valid_shards),
        len(shard_inspection.corrupt),
        total_records,
    )

    missing = [
        expectation
        for expectation in expectations
        if _expectation_key(expectation) not in valid_shards
    ]
    model_bundle: dict[str, Any] | None = None
    if missing:
        update_extraction_progress(
            output_dir,
            status="running",
            stage="loading_model",
            total_shards=len(expectations),
            completed_shards=len(valid_shards),
            total_records=total_records,
            completed_records=completed_records,
        )
        model_bundle = load_model_and_tokenizer(
            model_name=args.model,
            model_revision=args.model_revision,
            dtype_name=args.dtype,
            device=args.device,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            expected_layers=num_layers,
            expected_hidden_size=hidden_size,
        )
        if (
            model_bundle["resolved_revision"] != args.model_revision
            or model_bundle["tokenizer_revision"] != args.model_revision
        ):
            raise IncompatibleCheckpointError(
                "Loaded model/tokenizer revision does not equal the requested "
                "immutable revision"
            )
        metadata = build_extraction_metadata(
            args=args,
            inspection=inspection,
            tokenizer_inspection=tokenizer_inspection,
            selected_specs=selected_specs,
            selected_records=selected_records,
            model_bundle=model_bundle,
            compatibility_signature=compatibility_signature,
            output_dir=output_dir,
            extraction_started_at=extraction_started_at,
            extraction_completed_at=None,
        )
        atomic_write_json(output_dir / "metadata.json", metadata)
        if model_bundle["input_device"].type == "cuda":
            torch.cuda.reset_peak_memory_stats(model_bundle["input_device"])

        for expectation in missing:
            path = activation_shard_path(shard_root, expectation.plan)
            update_extraction_progress(
                output_dir,
                status="running",
                stage="extracting_story_activation_shards",
                active_emotion=expectation.plan.emotion,
                active_emotion_slug=expectation.plan.emotion_slug,
                active_shard=expectation.plan.shard_index,
                total_shards=len(expectations),
                completed_shards=len(valid_shards),
                total_records=total_records,
                completed_records=completed_records,
            )
            payload = extract_activation_shard(
                expectation=expectation,
                model=model_bundle["model"],
                tokenizer=model_bundle["tokenizer"],
                model_input_device=model_bundle["input_device"],
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                pad_to_multiple_of=args.pad_to_multiple_of,
                logger=logger,
            )
            write_verified_activation_shard(path, payload, expectation)
            valid_shards[_expectation_key(expectation)] = path
            completed_records += len(expectation.plan.records)
            update_extraction_progress(
                output_dir,
                status="running",
                stage="extracting_story_activation_shards",
                total_shards=len(expectations),
                completed_shards=len(valid_shards),
                total_records=total_records,
                completed_records=completed_records,
                last_completed_shard=str(path),
            )
            logger.info(
                "shard_complete emotion=%s shard=%05d records=%d path=%s",
                expectation.plan.emotion,
                expectation.plan.shard_index,
                len(expectation.plan.records),
                path,
            )

    runtime_model_metadata: dict[str, Any]
    if model_bundle is not None:
        runtime_model_metadata = {
            key: model_bundle.get(key)
            for key in (
                "resolved_dtype",
                "resolved_revision",
                "tokenizer_revision",
                "tokenizer_class",
                "model_class",
                "model_config_class",
                "transformers_version",
                "gpu_name",
                "resolved_hub_cache_dir",
                "resolved_snapshot_dir",
            )
        }
        runtime_model_metadata["peak_gpu_memory_bytes"] = peak_gpu_memory_bytes(
            model_bundle["input_device"]
        )
        del model_bundle["model"]
        model_bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        existing_metadata = read_json_object(output_dir / "metadata.json")
        runtime_model_metadata = {
            key: existing_metadata.get(key)
            for key in (
                "resolved_dtype",
                "resolved_revision",
                "tokenizer_revision",
                "tokenizer_class",
                "model_class",
                "model_config_class",
                "transformers_version",
                "gpu_name",
                "resolved_hub_cache_dir",
                "resolved_snapshot_dir",
                "peak_gpu_memory_bytes",
            )
        }
        logger.info("all_shards_verified skipping_model_load=true")

    update_extraction_progress(
        output_dir,
        status="running",
        stage="consolidating_story_activations",
        total_shards=len(expectations),
        completed_shards=len(valid_shards),
        total_records=total_records,
        completed_records=completed_records,
    )
    aggregate = consolidate_all_emotions(
        output_dir=output_dir,
        selected_specs=selected_specs,
        expectations=expectations,
        valid_shards=valid_shards,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation_dtype=args.activation_dtype,
        dataset_fingerprint=inspection.fingerprint_sha256,
        model_name=args.model,
        model_revision=args.model_revision,
        start_token_position=args.start_token_position,
        logger=logger,
    )
    final_artifacts = save_final_vector_artifacts(
        output_dir=output_dir,
        selected_specs=selected_specs,
        emotion_sums=aggregate["emotion_sums"],
        emotion_counts=aggregate["emotion_counts"],
        num_layers=num_layers,
        hidden_size=hidden_size,
    )
    validate_final_outputs(
        output_dir=output_dir,
        selected_specs=selected_specs,
        emotion_counts=aggregate["emotion_counts"],
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation_dtype=args.activation_dtype,
    )
    extraction_completed_at = utc_now()
    metadata = build_extraction_metadata(
        args=args,
        inspection=inspection,
        tokenizer_inspection=tokenizer_inspection,
        selected_specs=selected_specs,
        selected_records=selected_records,
        model_bundle=runtime_model_metadata,
        compatibility_signature=compatibility_signature,
        output_dir=output_dir,
        extraction_started_at=config["extraction_started_at"],
        extraction_completed_at=extraction_completed_at,
    )
    atomic_write_json(output_dir / "metadata.json", metadata)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "dataset_fingerprint": inspection.fingerprint_sha256,
        "extraction_fingerprint": compatibility_signature[
            "extraction_fingerprint_sha256"
        ],
        "model": args.model,
        "model_revision": args.model_revision,
        "number_of_emotions": len(selected_specs),
        "emotion_order": [spec.emotion for spec in selected_specs],
        "emotion_slugs": [spec.slug for spec in selected_specs],
        "records_per_emotion": aggregate["emotion_counts"],
        "total_records": total_records,
        "number_of_layers": num_layers,
        "hidden_size": hidden_size,
        "start_token_position": args.start_token_position,
        "story_activation_dtype": args.activation_dtype,
        "emotion_mean_dtype": "float32",
        "emotion_vector_dtype": "float32",
        "total_shards": len(expectations),
        "records_per_shard": args.records_per_shard,
        "output_directory": str(output_dir),
        "artifacts": final_artifacts,
        "extraction_started_at": config["extraction_started_at"],
        "extraction_completed_at": extraction_completed_at,
        "peak_gpu_memory_bytes": metadata.get("peak_gpu_memory_bytes"),
    }
    atomic_write_json(output_dir / "extraction_summary.json", summary)
    update_extraction_progress(
        output_dir,
        status="completed",
        stage="completed",
        total_shards=len(expectations),
        completed_shards=len(expectations),
        total_records=total_records,
        completed_records=total_records,
        extraction_completed_at=extraction_completed_at,
    )
    logger.info(
        "extraction_complete emotions=%d records=%d output=%s",
        len(selected_specs),
        total_records,
        output_dir,
    )
    print(
        "STORY_EXTRACTION_COMPLETED",
        f"emotions={len(selected_specs)}",
        f"records={total_records}",
        f"output={output_dir}",
    )
    return summary


def ensure_tokenizer_preflight(
    *,
    inspection: DatasetInspection,
    model_name: str,
    model_revision: str,
    cache_dir: Path,
    local_files_only: bool,
    start_token_position: int,
    output_dir: Path,
) -> TokenizerInspection:
    """Reuse an exact token index, or create it locally when absent."""

    try:
        return load_tokenizer_inspection(
            inspection=inspection,
            model_name=model_name,
            model_revision=model_revision,
            start_token_position=start_token_position,
            output_dir=output_dir,
        )
    except FileNotFoundError:
        return create_tokenizer_preflight(
            inspection=inspection,
            model_name=model_name,
            model_revision=model_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            start_token_position=start_token_position,
            output_dir=output_dir,
        )


def select_emotion_specs(
    emotion_specs: Sequence[EmotionSpec],
    *,
    requested: Sequence[str] | None,
) -> tuple[EmotionSpec, ...]:
    """Select exact configured labels while preserving configured order."""

    configured = {spec.emotion: spec for spec in emotion_specs}
    if requested is None:
        selected = tuple(emotion_specs)
    else:
        if len(set(requested)) != len(requested):
            raise ValueError("--emotions cannot contain duplicate labels")
        unknown = [emotion for emotion in requested if emotion not in configured]
        if unknown:
            raise ValueError(
                f"Unknown emotion label(s): {unknown!r}; labels must exactly match "
                "the emotion configuration"
            )
        requested_set = set(requested)
        selected = tuple(
            spec for spec in emotion_specs if spec.emotion in requested_set
        )
    if len(selected) < 2:
        raise ValueError(
            "Raw one-versus-rest extraction requires at least two emotions"
        )
    return selected


def select_records_by_emotion(
    *,
    inspection: DatasetInspection,
    selected_specs: Sequence[EmotionSpec],
    max_records_per_emotion: int | None,
) -> dict[str, tuple[StoryRecord, ...]]:
    """Take a deterministic canonical prefix for every selected emotion."""

    result: dict[str, tuple[StoryRecord, ...]] = {}
    for spec in selected_specs:
        records = inspection.records_by_emotion[spec.emotion]
        selected = (
            records
            if max_records_per_emotion is None
            else records[:max_records_per_emotion]
        )
        if not selected:
            raise ValueError(f"No selected records for {spec.emotion!r}")
        result[spec.emotion] = tuple(selected)
    return result


def build_activation_shard_plans(
    *,
    selected_specs: Sequence[EmotionSpec],
    records_by_emotion: Mapping[str, Sequence[StoryRecord]],
    token_by_record_id: Mapping[str, TokenIndexRecord],
    records_per_shard: int,
) -> tuple[ActivationShardPlan, ...]:
    """Partition canonical emotion records into immutable fixed-size shards."""

    if records_per_shard < 1:
        raise ValueError("records_per_shard must be positive")
    plans: list[ActivationShardPlan] = []
    for spec in selected_specs:
        records = tuple(records_by_emotion[spec.emotion])
        for shard_index, start in enumerate(
            range(0, len(records), records_per_shard)
        ):
            shard_records = records[start : start + records_per_shard]
            token_counts: list[int] = []
            for record in shard_records:
                try:
                    token_row = token_by_record_id[record.record_id]
                except KeyError as error:
                    raise ValueError(
                        f"No tokenizer index for {record.record_id!r}"
                    ) from error
                if (
                    token_row.emotion != record.emotion
                    or token_row.source_line != record.source_line
                ):
                    raise ValueError(
                        f"Tokenizer index identity mismatch for "
                        f"{record.record_id!r}"
                    )
                token_counts.append(token_row.token_count)
            plans.append(
                ActivationShardPlan(
                    emotion=spec.emotion,
                    emotion_slug=spec.slug,
                    shard_index=shard_index,
                    records=tuple(shard_records),
                    token_counts=tuple(token_counts),
                )
            )
    return tuple(plans)


def build_extraction_compatibility_signature(
    *,
    inspection: DatasetInspection,
    selected_specs: Sequence[EmotionSpec],
    selected_records: Mapping[str, Sequence[StoryRecord]],
    token_index_sha256: str,
    model_name: str,
    model_revision: str,
    start_token_position: int,
    num_layers: int,
    hidden_size: int,
    activation_dtype: str,
    records_per_shard: int,
) -> dict[str, Any]:
    """Build the immutable scientific identity used by every checkpoint."""

    selection: dict[str, Any] = {}
    for spec in selected_specs:
        records = selected_records[spec.emotion]
        identity_rows = [
            {
                "record_id": row.record_id,
                "source_line": row.source_line,
            }
            for row in records
        ]
        selection[spec.emotion] = {
            "emotion_slug": spec.slug,
            "record_count": len(records),
            "record_identity_sha256": canonical_json_sha256(identity_rows),
            "first_record_id": records[0].record_id,
            "last_record_id": records[-1].record_id,
        }
    core = {
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint": inspection.fingerprint_sha256,
        "story_token_index_sha256": token_index_sha256,
        "model_name": model_name,
        "model_revision": model_revision,
        "emotion_order": [spec.emotion for spec in selected_specs],
        "emotion_slugs": [spec.slug for spec in selected_specs],
        "selection": selection,
        "start_token_position": start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "number_of_layers": num_layers,
        "hidden_size": hidden_size,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "saved_activation_dtype": activation_dtype,
        "records_per_shard": records_per_shard,
    }
    return {
        **core,
        "extraction_fingerprint_sha256": canonical_json_sha256(core),
    }


def prepare_extraction_config(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    compatibility_signature: dict[str, Any],
    extraction_started_at: str,
) -> dict[str, Any]:
    """Create or validate the immutable run configuration."""

    path = output_dir / "config.json"
    execution = {
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "device": args.device,
        "dtype": args.dtype,
    }
    if path.is_file():
        existing = read_json_object(path)
        if existing.get("compatibility_signature") != compatibility_signature:
            raise IncompatibleCheckpointError(
                f"Existing extraction config is incompatible: {path}"
            )
        if not args.resume:
            raise ValueError(
                f"Existing extraction config requires --resume: {path}"
            )
        existing["execution_settings"] = execution
        existing["last_resumed_at"] = utc_now()
        existing["cli_arguments"] = json_safe_cli_arguments(args)
        atomic_write_json(path, existing)
        return existing
    payload = {
        "schema_version": SCHEMA_VERSION,
        "compatibility_signature": compatibility_signature,
        "execution_settings": execution,
        "cli_arguments": json_safe_cli_arguments(args),
        "output_directory": str(output_dir),
        "extraction_started_at": extraction_started_at,
    }
    atomic_write_json(path, payload)
    return payload


def activation_shard_path(
    shard_root: Path,
    plan: ActivationShardPlan,
) -> Path:
    return (
        shard_root
        / plan.emotion_slug
        / f"shard_{plan.shard_index:05d}.pt"
    )


def _expectation_key(
    expectation: ActivationShardExpectation,
) -> tuple[str, int]:
    return (
        expectation.plan.emotion_slug,
        expectation.plan.shard_index,
    )


def inspect_activation_shards(
    root: Path,
    expectations: Sequence[ActivationShardExpectation],
) -> ShardInspection:
    """Verify checkpoint files, hard-reject foreign state, flag corruption."""

    shard_root = root.expanduser().resolve()
    expected_by_path = {
        activation_shard_path(shard_root, expectation.plan): expectation
        for expectation in expectations
    }
    if shard_root.exists():
        actual_paths = {
            path.resolve()
            for path in shard_root.rglob("*.pt")
            if path.is_file()
        }
        unexpected = sorted(actual_paths - set(expected_by_path), key=str)
        if unexpected:
            raise IncompatibleCheckpointError(
                "Unexpected activation shard files are present: "
                f"{[str(path) for path in unexpected]!r}"
            )
    valid: dict[tuple[str, int], Path] = {}
    corrupt: dict[tuple[str, int], str] = {}
    for path, expectation in expected_by_path.items():
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_activation_shard(payload, expectation)
        except IncompatibleCheckpointError:
            raise
        except Exception as error:
            corrupt[_expectation_key(expectation)] = safe_error_message(error)
            continue
        valid[_expectation_key(expectation)] = path
    return ShardInspection(valid=valid, corrupt=corrupt)


def validate_activation_shard(
    payload: Any,
    expectation: ActivationShardExpectation,
) -> None:
    """Validate shard identity, canonical rows, tensor shape/dtype, and sums."""

    if not isinstance(payload, dict):
        raise ValueError("Activation shard payload is not a mapping")
    immutable_fields = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "model_name": expectation.model_name,
        "model_revision": expectation.model_revision,
        "dataset_fingerprint": expectation.dataset_fingerprint,
        "start_token_position": expectation.start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "number_of_layers": expectation.num_layers,
        "hidden_size": expectation.hidden_size,
        "activation_dtype": expectation.activation_dtype,
    }
    for field, expected in immutable_fields.items():
        if payload.get(field) != expected:
            raise IncompatibleCheckpointError(
                f"Activation shard {field} mismatch: "
                f"{payload.get(field)!r} != {expected!r}"
            )
    plan = expectation.plan
    local_fields = {
        "emotion": plan.emotion,
        "emotion_slug": plan.emotion_slug,
        "shard_index": plan.shard_index,
        "record_ids": list(plan.record_ids),
        "source_lines": [record.source_line for record in plan.records],
        "topic_ids": [record.topic_id for record in plan.records],
        "sample_indices": [record.sample_index for record in plan.records],
        "token_counts": list(plan.token_counts),
    }
    for field, expected in local_fields.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"Activation shard canonical field {field!r} is corrupt"
            )
    activations = payload.get("activations")
    expected_shape = (
        len(plan.records),
        expectation.num_layers,
        expectation.hidden_size,
    )
    expected_dtype = activation_torch_dtype(expectation.activation_dtype)
    if not isinstance(activations, torch.Tensor):
        raise ValueError("Activation shard has no tensor")
    if tuple(activations.shape) != expected_shape:
        raise ValueError(
            f"Activation shard tensor shape {tuple(activations.shape)} != "
            f"{expected_shape}"
        )
    if activations.dtype != expected_dtype:
        raise ValueError(
            f"Activation shard tensor dtype {activations.dtype} != "
            f"{expected_dtype}"
        )
    if activations.device.type != "cpu":
        raise ValueError("Persisted activation shard must be on CPU")
    if not bool(torch.isfinite(activations).all()):
        raise ValueError("Activation shard contains NaN or infinity")
    activation_sum = payload.get("activation_sum_float64")
    sum_shape = (expectation.num_layers, expectation.hidden_size)
    if (
        not isinstance(activation_sum, torch.Tensor)
        or tuple(activation_sum.shape) != sum_shape
        or activation_sum.dtype != torch.float64
        or activation_sum.device.type != "cpu"
        or not bool(torch.isfinite(activation_sum).all())
    ):
        raise ValueError("Activation shard has an invalid float64 activation sum")


def build_activation_shard_payload(
    *,
    expectation: ActivationShardExpectation,
    activations_fp32: torch.Tensor,
) -> dict[str, Any]:
    """Build a shard while retaining the pre-quantization float64 sum."""

    plan = expectation.plan
    expected_shape = (
        len(plan.records),
        expectation.num_layers,
        expectation.hidden_size,
    )
    if (
        tuple(activations_fp32.shape) != expected_shape
        or activations_fp32.dtype != torch.float32
        or activations_fp32.device.type != "cpu"
        or not bool(torch.isfinite(activations_fp32).all())
    ):
        raise ValueError(
            "Canonical shard activations must be finite CPU float32 with shape "
            f"{expected_shape}"
        )
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "model_name": expectation.model_name,
        "model_revision": expectation.model_revision,
        "dataset_fingerprint": expectation.dataset_fingerprint,
        "emotion": plan.emotion,
        "emotion_slug": plan.emotion_slug,
        "shard_index": plan.shard_index,
        "start_token_position": expectation.start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "number_of_layers": expectation.num_layers,
        "hidden_size": expectation.hidden_size,
        "activation_dtype": expectation.activation_dtype,
        "record_ids": list(plan.record_ids),
        "source_lines": [record.source_line for record in plan.records],
        "topic_ids": [record.topic_id for record in plan.records],
        "sample_indices": [record.sample_index for record in plan.records],
        "token_counts": list(plan.token_counts),
        "activation_sum_float64": activations_fp32.to(torch.float64).sum(dim=0),
        "activations": activations_fp32.to(
            dtype=activation_torch_dtype(expectation.activation_dtype)
        ),
    }
    validate_activation_shard(payload, expectation)
    return payload


def write_verified_activation_shard(
    path: Path,
    payload: Any,
    expectation: ActivationShardExpectation,
) -> None:
    """Publish a shard only after reloading and fully validating its temp file."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_activation_shard(payload, expectation)
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = torch.load(
            temporary,
            map_location="cpu",
            weights_only=False,
        )
        validate_activation_shard(reloaded, expectation)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_activation_shard(
    *,
    expectation: ActivationShardExpectation,
    model: Any,
    tokenizer: Any,
    model_input_device: torch.device,
    batch_size: int,
    max_batch_tokens: int,
    pad_to_multiple_of: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run deterministic length-aware batches for one fixed shard."""

    plan = expectation.plan
    batches = build_length_aware_batches(
        plan.token_counts,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    batch_results: list[tuple[Sequence[int], torch.Tensor]] = []
    for batch_number, indices in enumerate(batches):
        records = [plan.records[index] for index in indices]
        expected_counts = [plan.token_counts[index] for index in indices]
        started = time.monotonic()
        try:
            encoded = tokenizer(
                [record.story for record in records],
                add_special_tokens=True,
                truncation=False,
                padding=True,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            observed_counts = [
                int(value)
                for value in attention_mask.sum(dim=1).tolist()
            ]
            if observed_counts != expected_counts:
                raise ValueError(
                    "Extraction token counts differ from tokenizer preflight: "
                    f"{observed_counts!r} != {expected_counts!r}"
                )
            input_ids = input_ids.to(model_input_device)
            attention_mask = attention_mask.to(model_input_device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise ValueError("Model did not return hidden states")
            means = mean_transformer_hidden_states(
                hidden_states,
                attention_mask=attention_mask,
                start_token_position=expectation.start_token_position,
                num_layers=expectation.num_layers,
                hidden_size=expectation.hidden_size,
            )
        except BaseException as error:
            if is_out_of_memory(error):
                raise ExtractionOutOfMemory(
                    emotion=plan.emotion,
                    shard_index=plan.shard_index,
                    record_ids=[record.record_id for record in records],
                    token_counts=expected_counts,
                    batch_size=len(records),
                    original_error=error,
                ) from error
            raise
        padded_sequence_length = int(attention_mask.shape[1])
        valid_tokens = int(attention_mask.sum().item())
        padded_tokens = int(attention_mask.numel())
        padding_fraction = (
            float(padded_tokens - valid_tokens) / padded_tokens
            if padded_tokens
            else 0.0
        )
        logger.info(
            "batch emotion=%s shard=%05d batch=%d/%d size=%d "
            "min_tokens=%d max_tokens=%d padded_sequence_length=%d "
            "valid_tokens=%d padded_tokens=%d padding_fraction=%.6f "
            "elapsed_seconds=%.3f",
            plan.emotion,
            plan.shard_index,
            batch_number + 1,
            len(batches),
            len(indices),
            min(expected_counts),
            max(expected_counts),
            padded_sequence_length,
            valid_tokens,
            padded_tokens,
            padding_fraction,
            time.monotonic() - started,
        )
        batch_results.append((indices, means))
        del outputs, hidden_states, input_ids, attention_mask, encoded, means
    canonical = restore_canonical_order(
        batch_results,
        total_records=len(plan.records),
    )
    payload = build_activation_shard_payload(
        expectation=expectation,
        activations_fp32=canonical,
    )
    del canonical, batch_results
    return payload


def consolidate_all_emotions(
    *,
    output_dir: Path,
    selected_specs: Sequence[EmotionSpec],
    expectations: Sequence[ActivationShardExpectation],
    valid_shards: Mapping[tuple[str, int], Path],
    num_layers: int,
    hidden_size: int,
    activation_dtype: str,
    dataset_fingerprint: str,
    model_name: str,
    model_revision: str,
    start_token_position: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Consolidate verified shards in canonical order and stream float64 sums."""

    expectations_by_emotion: dict[str, list[ActivationShardExpectation]] = {
        spec.emotion: [] for spec in selected_specs
    }
    for expectation in expectations:
        expectations_by_emotion[expectation.plan.emotion].append(expectation)
    destination_root = output_dir / "story_activations"
    destination_root.mkdir(parents=True, exist_ok=True)
    emotion_sums: dict[str, torch.Tensor] = {}
    emotion_counts: dict[str, int] = {}
    saved_dtype = activation_torch_dtype(activation_dtype)

    for spec in selected_specs:
        emotion_expectations = sorted(
            expectations_by_emotion[spec.emotion],
            key=lambda item: item.plan.shard_index,
        )
        count = sum(
            len(expectation.plan.records)
            for expectation in emotion_expectations
        )
        if count < 1:
            raise ValueError(f"No activation shards for {spec.emotion!r}")
        activations = torch.empty(
            (count, num_layers, hidden_size),
            dtype=saved_dtype,
            device="cpu",
        )
        record_ids: list[str] = []
        source_lines: list[int] = []
        topic_ids: list[Any] = []
        sample_indices: list[Any] = []
        token_counts: list[int] = []
        activation_sum = torch.zeros(
            (num_layers, hidden_size),
            dtype=torch.float64,
            device="cpu",
        )
        offset = 0
        for expectation in emotion_expectations:
            key = _expectation_key(expectation)
            try:
                path = valid_shards[key]
            except KeyError as error:
                raise RuntimeError(
                    f"Missing verified shard for {key!r}"
                ) from error
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_activation_shard(payload, expectation)
            shard_activations = payload["activations"]
            shard_count = len(expectation.plan.records)
            activations[offset : offset + shard_count].copy_(shard_activations)
            activation_sum += payload["activation_sum_float64"]
            record_ids.extend(payload["record_ids"])
            source_lines.extend(payload["source_lines"])
            topic_ids.extend(payload["topic_ids"])
            sample_indices.extend(payload["sample_indices"])
            token_counts.extend(payload["token_counts"])
            offset += shard_count
            del payload, shard_activations
        if offset != count:
            raise RuntimeError(
                f"Consolidation count mismatch for {spec.emotion!r}"
            )
        payload = {
            "schema_version": CONSOLIDATED_SCHEMA_VERSION,
            "model_name": model_name,
            "model_revision": model_revision,
            "dataset_fingerprint": dataset_fingerprint,
            "emotion": spec.emotion,
            "emotion_slug": spec.slug,
            "start_token_position": start_token_position,
            "token_position_indexing": TOKEN_POSITION_INDEXING,
            "hidden_state_mapping": HIDDEN_STATE_MAPPING,
            "record_ids": record_ids,
            "source_lines": source_lines,
            "topic_ids": topic_ids,
            "sample_indices": sample_indices,
            "token_counts": token_counts,
            "activation_dtype": activation_dtype,
            "activations": activations,
        }
        validate_consolidated_activation(
            payload,
            emotion_spec=spec,
            expected_count=count,
            num_layers=num_layers,
            hidden_size=hidden_size,
            activation_dtype=activation_dtype,
            expected_record_ids=[
                record_id
                for expectation in emotion_expectations
                for record_id in expectation.plan.record_ids
            ],
        )
        destination = destination_root / f"{spec.slug}.pt"
        atomic_torch_save(destination, payload)
        emotion_sums[spec.emotion] = activation_sum
        emotion_counts[spec.emotion] = count
        logger.info(
            "emotion_consolidated emotion=%s records=%d shape=%s path=%s",
            spec.emotion,
            count,
            tuple(activations.shape),
            destination,
        )
        del payload, activations, activation_sum
        gc.collect()
    return {
        "emotion_sums": emotion_sums,
        "emotion_counts": emotion_counts,
    }


def validate_consolidated_activation(
    payload: Any,
    *,
    emotion_spec: EmotionSpec,
    expected_count: int,
    num_layers: int,
    hidden_size: int,
    activation_dtype: str,
    expected_record_ids: Sequence[str] | None = None,
) -> None:
    """Validate one per-emotion consolidated activation artifact."""

    if not isinstance(payload, dict):
        raise ValueError("Consolidated activation payload is not a mapping")
    if payload.get("emotion") != emotion_spec.emotion:
        raise ValueError("Consolidated activation emotion label is incompatible")
    if payload.get("emotion_slug") != emotion_spec.slug:
        raise ValueError("Consolidated activation emotion slug is incompatible")
    record_ids = payload.get("record_ids")
    if (
        not isinstance(record_ids, list)
        or len(record_ids) != expected_count
        or len(set(record_ids)) != expected_count
    ):
        raise ValueError("Consolidated activation record IDs are invalid")
    if (
        expected_record_ids is not None
        and record_ids != list(expected_record_ids)
    ):
        raise ValueError("Consolidated activation record order is incompatible")
    for field in ("source_lines", "topic_ids", "sample_indices", "token_counts"):
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(
                f"Consolidated activation {field} length is invalid"
            )
    activations = payload.get("activations")
    expected_shape = (expected_count, num_layers, hidden_size)
    if (
        not isinstance(activations, torch.Tensor)
        or tuple(activations.shape) != expected_shape
        or activations.dtype != activation_torch_dtype(activation_dtype)
        or activations.device.type != "cpu"
        or not bool(torch.isfinite(activations).all())
    ):
        raise ValueError(
            "Consolidated activation tensor is invalid; expected "
            f"shape={expected_shape} dtype={activation_dtype}"
        )


def save_final_vector_artifacts(
    *,
    output_dir: Path,
    selected_specs: Sequence[EmotionSpec],
    emotion_sums: Mapping[str, torch.Tensor],
    emotion_counts: Mapping[str, int],
    num_layers: int,
    hidden_size: int,
) -> dict[str, str]:
    """Compute and atomically persist sums, means, raw, unit, and stacks."""

    emotion_order = [spec.emotion for spec in selected_specs]
    emotion_slugs = [spec.slug for spec in selected_specs]
    means, raw_vectors = compute_weighted_one_vs_rest(
        emotion_sums,
        emotion_counts,
        emotion_order=emotion_order,
    )
    unit_vectors = {
        emotion: normalize_layers(raw_vectors[emotion])
        for emotion in emotion_order
    }
    expected_shape = (num_layers, hidden_size)
    for collection_name, collection, expected_dtype in (
        ("emotion sums", emotion_sums, torch.float64),
        ("emotion means", means, torch.float32),
        ("raw emotion vectors", raw_vectors, torch.float32),
        ("unit emotion vectors", unit_vectors, torch.float32),
    ):
        if list(collection) != emotion_order:
            raise ValueError(f"{collection_name} order differs from configuration")
        for emotion in emotion_order:
            tensor = collection[emotion]
            if (
                tuple(tensor.shape) != expected_shape
                or tensor.dtype != expected_dtype
                or tensor.device.type != "cpu"
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(
                    f"Invalid {collection_name} tensor for {emotion!r}"
                )
    stacked = torch.stack(
        [raw_vectors[emotion] for emotion in emotion_order],
        dim=0,
    )
    unit_stacked = torch.stack(
        [unit_vectors[emotion] for emotion in emotion_order],
        dim=0,
    )
    expected_stacked_shape = (
        len(emotion_order),
        num_layers,
        hidden_size,
    )
    if (
        tuple(stacked.shape) != expected_stacked_shape
        or tuple(unit_stacked.shape) != expected_stacked_shape
    ):
        raise RuntimeError("Stacked vector shape is inconsistent")
    norms_payload: dict[str, list[float]] = {}
    for emotion in emotion_order:
        norms = raw_vectors[emotion].norm(p=2, dim=-1)
        norms_payload[emotion] = [float(value) for value in norms.tolist()]
        unit_norms = unit_vectors[emotion].norm(p=2, dim=-1)
        nonzero = norms > NORMALIZATION_EPSILON
        if bool(nonzero.any()):
            ones = torch.ones_like(unit_norms[nonzero])
            if not torch.allclose(
                unit_norms[nonzero],
                ones,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    f"Unit-vector norms are invalid for {emotion!r}"
                )

    artifacts: dict[str, Any] = {
        "emotion_sums.pt": dict(emotion_sums),
        "emotion_means.pt": means,
        "emotion_vectors_raw.pt": raw_vectors,
        "emotion_vectors_unit.pt": unit_vectors,
        "emotion_vectors_stacked.pt": {
            "emotion_order": emotion_order,
            "emotion_slugs": emotion_slugs,
            "vectors": stacked,
        },
        "emotion_vectors_unit_stacked.pt": {
            "emotion_order": emotion_order,
            "emotion_slugs": emotion_slugs,
            "vectors": unit_stacked,
        },
    }
    paths: dict[str, str] = {}
    for filename, payload in artifacts.items():
        path = output_dir / filename
        atomic_torch_save(path, payload)
        paths[filename] = str(path)
    counts_path = output_dir / "emotion_counts.json"
    atomic_write_json(
        counts_path,
        {emotion: int(emotion_counts[emotion]) for emotion in emotion_order},
    )
    paths[counts_path.name] = str(counts_path)
    norms_path = output_dir / "emotion_vector_norms.json"
    atomic_write_json(norms_path, norms_payload)
    paths[norms_path.name] = str(norms_path)
    return paths


def validate_final_outputs(
    *,
    output_dir: Path,
    selected_specs: Sequence[EmotionSpec],
    emotion_counts: Mapping[str, int],
    num_layers: int,
    hidden_size: int,
    activation_dtype: str,
) -> None:
    """Reload final artifacts and enforce all documented tensor contracts."""

    emotion_order = [spec.emotion for spec in selected_specs]
    expected_vector_shape = (num_layers, hidden_size)
    for spec in selected_specs:
        path = output_dir / "story_activations" / f"{spec.slug}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_consolidated_activation(
            payload,
            emotion_spec=spec,
            expected_count=int(emotion_counts[spec.emotion]),
            num_layers=num_layers,
            hidden_size=hidden_size,
            activation_dtype=activation_dtype,
        )
        del payload
    artifact_contracts = (
        ("emotion_sums.pt", torch.float64),
        ("emotion_means.pt", torch.float32),
        ("emotion_vectors_raw.pt", torch.float32),
        ("emotion_vectors_unit.pt", torch.float32),
    )
    for filename, expected_dtype in artifact_contracts:
        payload = torch.load(
            output_dir / filename,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict) or list(payload) != emotion_order:
            raise ValueError(f"{filename} has an invalid emotion mapping")
        for emotion in emotion_order:
            value = payload[emotion]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected_vector_shape
                or value.dtype != expected_dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{filename} tensor is invalid for {emotion!r}")
        del payload
    expected_stacked_shape = (
        len(selected_specs),
        num_layers,
        hidden_size,
    )
    for filename in (
        "emotion_vectors_stacked.pt",
        "emotion_vectors_unit_stacked.pt",
    ):
        payload = torch.load(
            output_dir / filename,
            map_location="cpu",
            weights_only=False,
        )
        if (
            payload.get("emotion_order") != emotion_order
            or payload.get("emotion_slugs")
            != [spec.slug for spec in selected_specs]
        ):
            raise ValueError(f"{filename} has invalid ordered labels")
        vectors = payload.get("vectors")
        if (
            not isinstance(vectors, torch.Tensor)
            or tuple(vectors.shape) != expected_stacked_shape
            or vectors.dtype != torch.float32
            or not bool(torch.isfinite(vectors).all())
        ):
            raise ValueError(f"{filename} has an invalid stacked tensor")
    counts = read_json_object(output_dir / "emotion_counts.json")
    expected_counts = {
        emotion: int(emotion_counts[emotion])
        for emotion in emotion_order
    }
    if counts != expected_counts:
        raise ValueError("emotion_counts.json is incompatible")


def build_extraction_metadata(
    *,
    args: argparse.Namespace,
    inspection: DatasetInspection,
    tokenizer_inspection: TokenizerInspection,
    selected_specs: Sequence[EmotionSpec],
    selected_records: Mapping[str, Sequence[StoryRecord]],
    model_bundle: Mapping[str, Any],
    compatibility_signature: Mapping[str, Any],
    output_dir: Path,
    extraction_started_at: str,
    extraction_completed_at: str | None,
) -> dict[str, Any]:
    """Build complete scientific and runtime provenance."""

    counts = {
        spec.emotion: len(selected_records[spec.emotion])
        for spec in selected_specs
    }
    report = tokenizer_inspection.report
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "model_revision": args.model_revision,
        "resolved_revision": model_bundle.get("resolved_revision"),
        "tokenizer_revision": model_bundle.get("tokenizer_revision"),
        "requested_cache_dir": str(args.cache_dir.expanduser().resolve()),
        "resolved_hub_cache_dir": str(
            model_bundle.get(
                "resolved_hub_cache_dir",
                report["resolved_hub_cache_dir"],
            )
        ),
        "resolved_snapshot_dir": str(
            model_bundle.get(
                "resolved_snapshot_dir",
                report["resolved_snapshot_dir"],
            )
        ),
        "local_files_only": bool(args.local_files_only),
        "tokenizer_class": model_bundle.get(
            "tokenizer_class",
            report["tokenizer_class"],
        ),
        "model_class": model_bundle.get("model_class"),
        "model_config_class": model_bundle.get(
            "model_config_class",
            report["model_config_class"],
        ),
        "emotion_configuration_path": str(inspection.emotion_config_path),
        "emotion_configuration_sha256": inspection.fingerprint[
            "emotion_config_sha256"
        ],
        "ordered_emotions": [spec.emotion for spec in selected_specs],
        "emotion_slugs": [spec.slug for spec in selected_specs],
        "label_to_slug": {
            spec.emotion: spec.slug for spec in selected_specs
        },
        "records_directory": str(inspection.records_dir),
        "generation_manifest_path": str(inspection.generation_manifest_path),
        "generation_manifest_sha256": inspection.fingerprint[
            "generation_manifest_sha256"
        ],
        "generation_config_path": str(inspection.generation_config_path),
        "generation_config_sha256": inspection.fingerprint[
            "generation_config_sha256"
        ],
        "dataset_fingerprint": inspection.fingerprint_sha256,
        "extraction_fingerprint": compatibility_signature[
            "extraction_fingerprint_sha256"
        ],
        "total_input_records": inspection.total_records,
        "number_of_selected_emotions": len(selected_specs),
        "selected_records_per_emotion": counts,
        "total_selected_records": sum(counts.values()),
        "number_of_layers": int(report["number_of_layers"]),
        "hidden_size": int(report["hidden_size"]),
        "start_token_position": args.start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "residual_stream_definition": RESIDUAL_STREAM_DEFINITION,
        "layers_averaged_together": False,
        "model_dtype": model_bundle.get("resolved_dtype", str(args.dtype)),
        "token_averaging_dtype": "float32",
        "saved_activation_dtype": args.activation_dtype,
        "emotion_sum_dtype": "float64",
        "emotion_mean_dtype": "float32",
        "emotion_vector_dtype": "float32",
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "records_per_shard": args.records_per_shard,
        "output_directory": str(output_dir),
        "cli_arguments": json_safe_cli_arguments(args),
        "extraction_started_at": extraction_started_at,
        "extraction_completed_at": extraction_completed_at,
        "torch_version": torch.__version__,
        "transformers_version": model_bundle.get(
            "transformers_version",
            report["transformers_version"],
        ),
        "cuda_version": torch.version.cuda,
        "gpu_name": model_bundle.get("gpu_name"),
        "peak_gpu_memory_bytes": model_bundle.get("peak_gpu_memory_bytes"),
    }
    return payload


def load_model_and_tokenizer(
    *,
    model_name: str,
    model_revision: str | None,
    dtype_name: str,
    device: str,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    expected_layers: int | None = None,
    expected_hidden_size: int | None = None,
) -> dict[str, Any]:
    """Load one unquantized checkpoint wholly on one requested device.

    Optional cache and architecture arguments preserve the legacy neutral-PCA
    caller while the emotional-story extractor supplies the strict values.
    """

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for activation extraction"
        ) from error
    resolved_dtype = resolve_torch_dtype(dtype_name)
    common_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    resolved_hub: Path | None = None
    resolved_snapshot: Path | None = None
    if cache_dir is not None:
        if model_revision is None:
            raise ValueError(
                "A model revision is required with an explicit local cache"
            )
        resolved_hub = resolve_hub_cache_dir(
            cache_dir=cache_dir,
            model_name=model_name,
            model_revision=model_revision,
        )
        resolved_snapshot = resolve_snapshot_dir(
            hub_cache_dir=resolved_hub,
            model_name=model_name,
            model_revision=model_revision,
        )
        common_kwargs["cache_dir"] = str(resolved_hub)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    target_device = resolve_single_device(device)
    model_kwargs = {
        **common_kwargs,
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
    }
    if target_device.type == "cuda":
        cuda_index = target_device.index if target_device.index is not None else 0
        model_kwargs["device_map"] = {"": cuda_index}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if target_device.type != "cuda":
        model.to(target_device)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    if expected_layers is not None and num_layers != expected_layers:
        raise ValueError(
            f"Loaded model reports {num_layers} layers; expected {expected_layers}"
        )
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise ValueError(
            f"Loaded model reports hidden size {hidden_size}; "
            f"expected {expected_hidden_size}"
        )
    ensure_no_cpu_or_disk_offload(model, requested_device=target_device)
    input_device = model_input_device(model)
    if input_device.type != target_device.type:
        raise RuntimeError(
            f"Model input device {input_device} differs from requested "
            f"{target_device}"
        )
    if target_device.type == "cuda":
        requested_index = target_device.index if target_device.index is not None else 0
        input_index = input_device.index if input_device.index is not None else 0
        if input_index != requested_index:
            raise RuntimeError(
                f"Model loaded on CUDA device {input_index}; "
                f"requested {requested_index}"
            )
    resolved_revision = resolve_model_revision(
        model=model,
        tokenizer=tokenizer,
        requested_revision=model_revision,
    )
    tokenizer_revision = resolve_tokenizer_revision(
        tokenizer=tokenizer,
        requested_revision=model_revision,
    )
    gpu_name: str | None = None
    if input_device.type == "cuda":
        gpu_index = input_device.index if input_device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(gpu_index)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "input_device": input_device,
        "resolved_dtype": str(model.dtype),
        "resolved_revision": resolved_revision,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "model_config_class": model.config.__class__.__name__,
        "transformers_version": transformers.__version__,
        "gpu_name": gpu_name,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "resolved_hub_cache_dir": resolved_hub,
        "resolved_snapshot_dir": resolved_snapshot,
    }


def resolve_single_device(device: str) -> torch.device:
    """Resolve ``auto`` without permitting automatic multi-device offload."""

    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
        index = resolved.index if resolved.index is not None else 0
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is unavailable; "
                f"visible devices={torch.cuda.device_count()}"
            )
        return torch.device("cuda", index)
    if resolved.type != "cpu":
        raise ValueError(
            f"Only one CUDA device or CPU is supported; got {device!r}"
        )
    return resolved


def ensure_no_cpu_or_disk_offload(
    model: Any,
    *,
    requested_device: torch.device,
) -> None:
    """Reject Accelerate maps that silently offload a GPU extraction."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return
    mapped: set[str] = set()
    for raw_device in device_map.values():
        if isinstance(raw_device, int):
            mapped.add(f"cuda:{raw_device}")
        else:
            mapped.add(str(raw_device))
    if "cpu" in mapped or "disk" in mapped:
        raise RuntimeError(
            f"CPU/disk model offload is forbidden: hf_device_map={device_map!r}"
        )
    if requested_device.type == "cuda":
        expected_index = (
            requested_device.index if requested_device.index is not None else 0
        )
        allowed = {
            str(expected_index),
            f"cuda:{expected_index}",
            "cuda",
        }
        unexpected = {
            value
            for value in mapped
            if value not in allowed
        }
        if unexpected:
            raise RuntimeError(
                "Model spans unexpected devices despite a single-GPU request: "
                f"{sorted(unexpected)!r}"
            )


def resolve_torch_dtype(name: str) -> torch.dtype | str:
    normalized = name.strip().lower().replace("torch.", "")
    aliases: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def activation_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"Unsupported activation dtype: {name!r}") from error


def model_input_device(model: Any) -> torch.device:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        weight = getattr(get_embeddings(), "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for mapped_device in device_map.values():
            if mapped_device not in {"cpu", "disk"}:
                if isinstance(mapped_device, int):
                    return torch.device("cuda", mapped_device)
                return torch.device(mapped_device)
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def resolve_model_revision(
    *,
    model: Any,
    tokenizer: Any,
    requested_revision: str | None,
) -> str | None:
    candidates = (
        getattr(model.config, "_commit_hash", None),
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        requested_revision,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def resolve_tokenizer_revision(
    *,
    tokenizer: Any,
    requested_revision: str | None,
) -> str | None:
    candidates = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        requested_revision,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def peak_gpu_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated(device))


def configure_extraction_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(
        f"emotionvectors.story_raw_vectors.{path.parent}"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def update_extraction_progress(
    output_dir: Path,
    *,
    status: str,
    stage: str,
    **fields: Any,
) -> None:
    """Atomically update progress while retaining stable run context."""

    path = output_dir.expanduser().resolve() / "progress.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = read_json_object(path)
        except ValueError:
            previous = {}
    payload = {
        **previous,
        **fields,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "stage": stage,
        "updated_at": utc_now(),
    }
    atomic_write_json(path, payload)


def mark_progress_failure(
    output_dir: Path,
    status: str,
    detail: Mapping[str, Any],
) -> None:
    try:
        update_extraction_progress(
            output_dir,
            status=status,
            stage=status,
            failure=dict(detail),
        )
    except Exception:
        # Never mask the primary extraction error with a progress-write failure.
        pass


def is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(
        error
    ).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, value: Any) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON object from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def json_safe_cli_arguments(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value.expanduser().resolve())
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_error_message(error: BaseException) -> str:
    text = str(error).strip() or error.__class__.__name__
    return f"{error.__class__.__name__}: {text}"


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _is_immutable_revision(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _required_nonempty_string(
    mapping: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} {field!r} is not a nonempty string")
    return value


def _validate_canonical_architecture(
    *,
    model_name: str,
    model_revision: str,
    num_layers: int,
    hidden_size: int,
) -> None:
    """Apply exact guards for the pinned 14B checkpoint without gating others."""

    if (
        model_name == CANONICAL_14B_MODEL
        and model_revision == CANONICAL_14B_REVISION
        and (
            num_layers != CANONICAL_14B_LAYERS
            or hidden_size != CANONICAL_14B_HIDDEN_SIZE
        )
    ):
        raise ValueError(
            "Pinned Qwen2.5-14B architecture mismatch: "
            f"layers={num_layers} hidden_size={hidden_size}; expected "
            f"{CANONICAL_14B_LAYERS}×{CANONICAL_14B_HIDDEN_SIZE}"
        )


__all__ = [
    "ActivationShardExpectation",
    "ActivationShardPlan",
    "DatasetInspection",
    "ExtractionOutOfMemory",
    "IncompatibleCheckpointError",
    "ShardInspection",
    "ShortStoryError",
    "StoryRecord",
    "TokenIndexRecord",
    "TokenizerInspection",
    "activation_shard_path",
    "activation_torch_dtype",
    "atomic_torch_save",
    "atomic_write_json",
    "atomic_write_jsonl",
    "build_activation_shard_payload",
    "build_activation_shard_plans",
    "build_length_aware_batches",
    "build_parser",
    "build_selected_token_mask",
    "canonical_json_sha256",
    "compute_emotion_sums_and_means",
    "compute_emotion_sums_from_activation_files",
    "compute_story_weighted_vector_outputs",
    "compute_weighted_one_vs_rest",
    "create_tokenizer_preflight",
    "discover_emotion_record_files",
    "discover_consolidated_activation_files",
    "emotion_slug",
    "inspect_activation_shards",
    "inspect_dataset",
    "is_out_of_memory",
    "load_model_and_tokenizer",
    "load_tokenizer_inspection",
    "main",
    "mean_transformer_hidden_states",
    "normalize_layers",
    "read_json_object",
    "resolve_hub_cache_dir",
    "resolve_torch_dtype",
    "restore_canonical_order",
    "round_up",
    "run_dataset_preflight",
    "run_compute_vectors",
    "run_extraction",
    "run_tokenizer_preflight_command",
    "select_emotion_specs",
    "sha256_file",
    "token_count_statistics",
    "validate_computed_vector_outputs",
    "validate_activation_shard",
    "validate_consolidated_activation",
    "write_verified_activation_shard",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
