#!/usr/bin/env python3
"""Remove layer-wise neutral PCA directions from raw emotion vectors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..constants import (
    EMOTIONS,
    HIDDEN_SIZE,
    HIDDEN_STATE_MAPPING,
    MODEL_ID,
    MODEL_REVISION,
    NUM_LAYERS,
    RESIDUAL_STREAM_DEFINITION,
)
from .raw_vectors import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_sha256,
    is_out_of_memory,
    load_model_and_tokenizer,
    read_json_object,
    resolve_torch_dtype,
    sha256_file,
)


TARGET_MODEL = MODEL_ID
EXPECTED_MODEL_REVISION = MODEL_REVISION
EXPECTED_NEUTRAL_RECORDS = 1200
EXPECTED_STORIES_PER_EMOTION = 1200
EMOTIONAL_STORY_START_POSITION = 40
TOKEN_POSITION_INDEXING = "one-based"
NEUTRAL_SAMPLE_UNIT = "one token-averaged activation per transcript"
NEUTRAL_TOKEN_AGGREGATION = "mean across all valid non-padding tokens"
PUBLIC_METHOD_DETAIL_STATUS = (
    "The Anthropic write-up specifies neutral-transcript PCA but does not "
    "explicitly state the neutral-token aggregation rule."
)
EPSILON = 1e-12
DEFAULT_NEUTRAL_JSONL = Path("data/neutral_dialogues/neutral_dialogues.jsonl")
DEFAULT_EMOTION_OUTPUT_BASE = Path("artifacts/Qwen2.5-7B-Instruct")
DEFAULT_STORY_RECORDS_DIR = Path("data/emotional_stories")
DEFAULT_OUTPUT_DIR = Path("outputs/Qwen2.5-7B-Instruct/neutral_pca_cleaning")


@dataclass(frozen=True)
class NeutralRecord:
    """One valid neutral JSONL row and its one-based source line number."""

    payload: dict[str, Any]
    line_number: int

    @property
    def record_id(self) -> str:
        return str(self.payload["record_id"])

    @property
    def transcript(self) -> str:
        return str(self.payload["transcript"])


@dataclass(frozen=True)
class LoadedNeutralRecords:
    """Validated neutral records and basic input accounting."""

    records: list[NeutralRecord]
    invalid_records: list[dict[str, Any]]
    total_lines: int


@dataclass(frozen=True)
class EmotionArtifacts:
    """Validated emotional-vector inputs reused by the cleaning stage."""

    base_dir: Path
    raw_path: Path
    unit_path: Path
    stacked_path: Path
    metadata_path: Path
    story_activations_dir: Path
    story_records_dir: Path
    metadata: dict[str, Any]
    raw_vectors: dict[str, torch.Tensor]
    raw_unit_vectors: dict[str, torch.Tensor]
    model_revision: str


class NeutralExtractionOutOfMemory(RuntimeError):
    """CUDA OOM information needed for a clean resumable exit."""

    def __init__(
        self,
        *,
        record_ids: list[str],
        sequence_lengths: list[int],
        batch_size: int,
        original_error: BaseException,
    ) -> None:
        super().__init__(str(original_error))
        self.record_ids = record_ids
        self.sequence_lengths = sequence_lengths
        self.batch_size = batch_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "neutral_activation_extraction",
            "record_ids": self.record_ids,
            "sequence_lengths": self.sequence_lengths,
            "batch_size": self.batch_size,
            "error": str(self),
            "recommendation": "Rerun with --resume and a smaller --batch-size.",
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the full cleaning pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract neutral transcript activations, compute independent PCA at "
            "each transformer layer, and clean raw emotion vectors."
        )
    )
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--neutral-jsonl", type=Path, default=DEFAULT_NEUTRAL_JSONL)
    parser.add_argument("--neutral-text-field", default="transcript")
    parser.add_argument("--raw-emotion-vectors", type=Path, default=None)
    parser.add_argument("--emotion-metadata", type=Path, default=None)
    parser.add_argument("--story-activations-dir", type=Path, default=None)
    parser.add_argument("--emotional-stories-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--neutral-start-token-position", type=_positive_int, default=1
    )
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--neutral-activation-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--max-sequence-length", type=_positive_int, default=None)
    parser.add_argument("--save-every", type=_positive_int, default=100)
    parser.add_argument("--max-neutral-records", type=_positive_int, default=None)
    parser.add_argument(
        "--neutral-explained-variance-threshold",
        type=_variance_threshold,
        default=0.50,
    )
    parser.add_argument(
        "--pca-device", choices=("auto", "cpu", "cuda"), default="cuda"
    )
    parser.add_argument(
        "--pca-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--eigenvalue-tolerance", type=_nonnegative_float, default=1e-10
    )
    parser.add_argument(
        "--component-orthonormality-tolerance",
        type=_positive_float,
        default=1e-4,
    )
    parser.add_argument(
        "--projection-orthogonality-tolerance",
        type=_positive_float,
        default=1e-4,
    )
    parser.add_argument(
        "--tokenwise-stories-per-emotion", type=_positive_int, default=10
    )
    parser.add_argument("--tokenwise-validation-seed", type=int, default=20260722)
    parser.add_argument(
        "--tokenwise-validation-layer", type=_nonnegative_int, default=None
    )
    parser.add_argument(
        "--story-validation-batch-size", type=_positive_int, default=64
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and convert expected runtime errors into clean exit codes."""

    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        mark_run_status(output_dir, "interrupted", {"error": "KeyboardInterrupt"})
        print("Neutral PCA cleaning interrupted; rerun with --resume.", file=sys.stderr)
        return 130
    except NeutralExtractionOutOfMemory as error:
        mark_run_status(output_dir, "out_of_memory", error.as_dict())
        print(
            "GPU out of memory during neutral extraction. Saved shards remain "
            "resumable; rerun with --resume and a smaller --batch-size.",
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        mark_run_status(output_dir, "failed", {"error": safe_error_message(error)})
        print(f"Neutral PCA cleaning failed: {error}", file=sys.stderr)
        return 1


def run_pipeline(args: argparse.Namespace) -> int:
    """Execute extraction, 28 independent PCAs, cleaning, and validation."""

    validate_cli_constraints(args)
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_directory(output_dir, resume=args.resume)
    logger = configure_logging(output_dir / "neutral_pca_cleaning.log")
    start_timestamp = utc_now()

    neutral_path = args.neutral_jsonl.expanduser().resolve()
    loaded = load_neutral_records(neutral_path)
    atomic_write_jsonl(
        output_dir / "skipped_invalid_neutral_records.jsonl",
        loaded.invalid_records,
    )
    selected_records = select_neutral_records(
        loaded=loaded,
        max_neutral_records=args.max_neutral_records,
    )
    partial = len(selected_records) < len(loaded.records)
    if not partial:
        if loaded.total_lines != EXPECTED_NEUTRAL_RECORDS:
            raise ValueError(
                f"Full run requires exactly {EXPECTED_NEUTRAL_RECORDS} neutral "
                f"JSONL lines; found {loaded.total_lines}"
            )
        if loaded.invalid_records:
            raise ValueError(
                "Full PCA requires valid activations for every accepted neutral "
                f"record; found {len(loaded.invalid_records)} invalid records"
            )
        if len(selected_records) != EXPECTED_NEUTRAL_RECORDS:
            raise ValueError(
                f"Expected {EXPECTED_NEUTRAL_RECORDS} valid neutral records; "
                f"found {len(selected_records)}"
            )

    artifacts = discover_and_validate_emotion_artifacts(args)
    save_verified_raw_vector_copy(
        source_vectors=artifacts.raw_vectors,
        destination=output_dir / "emotion_vectors_raw_original.pt",
        resume=args.resume,
    )
    requested_revision = args.model_revision or artifacts.model_revision
    if requested_revision != artifacts.model_revision:
        raise ValueError(
            "Requested model revision differs from emotional activation metadata: "
            f"{requested_revision!r} != {artifacts.model_revision!r}"
        )

    logger.info(
        "inputs_ready neutral_total=%d valid=%d invalid=%d selected=%d partial=%s "
        "raw_vectors=%s story_activations=%s",
        loaded.total_lines,
        len(loaded.records),
        len(loaded.invalid_records),
        len(selected_records),
        partial,
        artifacts.raw_path,
        artifacts.story_activations_dir,
    )
    shard_dir = output_dir / "neutral_activation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    expected_record_ids = {record.record_id for record in selected_records}
    existing_shards: list[Path] = []
    completed_ids: set[str] = set()
    model_bundle: dict[str, Any] | None = None
    fingerprint: dict[str, Any] | None = None
    shard_fingerprint: dict[str, Any] | None = None
    neutral_model_loaded = False

    if args.resume and any(shard_dir.glob("shard_*.pt")):
        unloaded_model_bundle = build_unloaded_model_bundle_metadata(
            args=args,
            artifacts=artifacts,
        )
        candidate_fingerprint = build_compatibility_signature(
            args=args,
            neutral_path=neutral_path,
            selected_records=selected_records,
            artifacts=artifacts,
            model_bundle=unloaded_model_bundle,
            partial=partial,
        )
        candidate_shard_fingerprint = build_neutral_extraction_fingerprint(
            candidate_fingerprint
        )
        existing_shards, completed_ids = load_existing_activation_shards(
            shard_dir=shard_dir,
            fingerprint=candidate_shard_fingerprint,
            expected_record_ids=expected_record_ids,
            activation_dtype=args.neutral_activation_dtype,
        )
        if completed_ids == expected_record_ids:
            model_bundle = unloaded_model_bundle
            fingerprint = candidate_fingerprint
            shard_fingerprint = candidate_shard_fingerprint
            logger.info(
                "resume_neutral_shards_complete shards=%d completed_records=%d "
                "skipping_neutral_model_load=true",
                len(existing_shards),
                len(completed_ids),
            )
            update_progress(
                output_dir,
                status="running",
                stage="combining_completed_neutral_activation_shards",
                total_neutral_records=loaded.total_lines,
                valid_neutral_records=len(loaded.records),
                invalid_neutral_records=len(loaded.invalid_records),
                selected_neutral_records=len(selected_records),
                partial_neutral_dataset=partial,
                completed_neutral_records=len(completed_ids),
                total_selected_neutral_records=len(selected_records),
            )

    if model_bundle is None:
        update_progress(
            output_dir,
            status="running",
            stage="loading_model_for_neutral_activations",
            total_neutral_records=loaded.total_lines,
            valid_neutral_records=len(loaded.records),
            invalid_neutral_records=len(loaded.invalid_records),
            selected_neutral_records=len(selected_records),
            partial_neutral_dataset=partial,
        )
        model_bundle = load_model_and_tokenizer(
            model_name=args.model,
            model_revision=requested_revision,
            dtype_name=args.dtype,
            device=args.device,
        )
        validate_loaded_model(model_bundle, artifacts.metadata)
        neutral_model_loaded = True
        fingerprint = build_compatibility_signature(
            args=args,
            neutral_path=neutral_path,
            selected_records=selected_records,
            artifacts=artifacts,
            model_bundle=model_bundle,
            partial=partial,
        )
        shard_fingerprint = build_neutral_extraction_fingerprint(fingerprint)
        existing_shards, completed_ids = load_existing_activation_shards(
            shard_dir=shard_dir,
            fingerprint=shard_fingerprint,
            expected_record_ids=expected_record_ids,
            activation_dtype=args.neutral_activation_dtype,
        )

    if fingerprint is None or shard_fingerprint is None:
        raise RuntimeError("Neutral activation compatibility state is incomplete")
    config = prepare_config(
        output_dir=output_dir,
        args=args,
        fingerprint=fingerprint,
        created_at=start_timestamp,
        resume=args.resume,
    )
    metadata = build_running_metadata(
        args=args,
        config=config,
        fingerprint=fingerprint,
        model_bundle=model_bundle,
        neutral_path=neutral_path,
        loaded=loaded,
        selected_records=selected_records,
        artifacts=artifacts,
        partial=partial,
    )
    atomic_write_json(output_dir / "metadata.json", metadata)

    if completed_ids:
        logger.info(
            "resume_neutral_shards shards=%d completed_records=%d",
            len(existing_shards),
            len(completed_ids),
        )
    if neutral_model_loaded:
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        extract_neutral_activations(
            records=selected_records,
            completed_ids=completed_ids,
            existing_shards=existing_shards,
            shard_dir=shard_dir,
            fingerprint=shard_fingerprint,
            model=model,
            tokenizer=tokenizer,
            model_input_device=model_bundle["input_device"],
            batch_size=args.batch_size,
            save_every=args.save_every,
            max_sequence_length=args.max_sequence_length,
            start_token_position=args.neutral_start_token_position,
            storage_dtype_name=args.neutral_activation_dtype,
            output_dir=output_dir,
            logger=logger,
        )
    neutral_payload = combine_activation_shards(
        records=selected_records,
        shard_dir=shard_dir,
        fingerprint=shard_fingerprint,
        activation_dtype=args.neutral_activation_dtype,
        output_path=output_dir / "neutral_activations.pt",
    )
    neutral_activations = neutral_payload["activations"]
    truncated_neutral_records = sum(
        bool(value) for value in neutral_payload["metadata"]["truncated"]
    )
    logger.info(
        "neutral_activations_complete shape=%s dtype=%s truncated=%d",
        tuple(neutral_activations.shape),
        neutral_activations.dtype,
        truncated_neutral_records,
    )
    update_progress(
        output_dir,
        status="running",
        stage="releasing_model_before_pca",
        neutral_activation_shape=list(neutral_activations.shape),
        completed_neutral_records=len(selected_records),
        truncated_neutral_records=truncated_neutral_records,
    )

    if neutral_model_loaded:
        del model, tokenizer
        model_bundle["model"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pca_device = resolve_pca_device(args.pca_device, args.pca_dtype)
    pca_dtype = torch.float32 if args.pca_dtype == "float32" else torch.float64
    update_progress(
        output_dir,
        status="running",
        stage="layerwise_neutral_pca",
        pca_device=str(pca_device),
        pca_dtype=str(pca_dtype),
        pca_svd_driver="gesvd" if pca_device.type == "cuda" else "default",
    )
    pca_payload, pca_summary, neutral_layer_means = compute_layerwise_pca(
        neutral_activations=neutral_activations,
        variance_threshold=args.neutral_explained_variance_threshold,
        eigenvalue_tolerance=args.eigenvalue_tolerance,
        orthonormality_tolerance=args.component_orthonormality_tolerance,
        pca_device=pca_device,
        pca_dtype=pca_dtype,
        logger=logger,
    )
    atomic_torch_save(output_dir / "neutral_layer_means.pt", neutral_layer_means)
    atomic_torch_save(output_dir / "neutral_pca_components.pt", pca_payload)
    atomic_write_json(output_dir / "neutral_pca_summary.json", pca_summary)

    cleaning = clean_emotion_vectors(
        raw_vectors=artifacts.raw_vectors,
        pca_payload=pca_payload,
        projection_tolerance=args.projection_orthogonality_tolerance,
    )
    save_cleaning_outputs(output_dir=output_dir, artifacts=artifacts, cleaning=cleaning)
    logger.info(
        "emotion_vectors_cleaned raw_shape=%s clean_stacked_shape=%s",
        (NUM_LAYERS, HIDDEN_SIZE),
        tuple(cleaning["stacked_clean"]["vectors"].shape),
    )

    update_progress(
        output_dir,
        status="running",
        stage="story_activation_validation",
    )
    story_validation = validate_clean_vectors_on_story_activations(
        story_activations_dir=artifacts.story_activations_dir,
        emotion_metadata=artifacts.metadata,
        raw_unit_vectors=artifacts.raw_unit_vectors,
        clean_unit_vectors=cleaning["clean_unit_vectors"],
        batch_size=args.story_validation_batch_size,
        compute_device=pca_device,
        logger=logger,
    )
    atomic_write_json(
        output_dir / "clean_vector_story_validation.json", story_validation
    )

    update_progress(
        output_dir,
        status="running",
        stage="loading_model_for_tokenwise_validation",
    )
    tokenwise_bundle = load_model_and_tokenizer(
        model_name=args.model,
        model_revision=requested_revision,
        dtype_name=args.dtype,
        device=args.device,
    )
    validate_loaded_model(tokenwise_bundle, artifacts.metadata)
    validation_layer = resolve_tokenwise_layer(args.tokenwise_validation_layer)
    tokenwise_counts = export_tokenwise_story_validation(
        story_records_dir=artifacts.story_records_dir,
        story_activations_dir=artifacts.story_activations_dir,
        output_dir=output_dir / "tokenwise_story_validation",
        model=tokenwise_bundle["model"],
        tokenizer=tokenwise_bundle["tokenizer"],
        model_input_device=tokenwise_bundle["input_device"],
        raw_unit_vectors=artifacts.raw_unit_vectors,
        clean_unit_vectors=cleaning["clean_unit_vectors"],
        stories_per_emotion=args.tokenwise_stories_per_emotion,
        selection_seed=args.tokenwise_validation_seed,
        validation_layer=validation_layer,
        resume=args.resume,
        logger=logger,
    )
    del tokenwise_bundle["model"]
    gc.collect()

    end_timestamp = utc_now()
    metadata = finalize_metadata(
        metadata=metadata,
        pca_summary=pca_summary,
        neutral_activations=neutral_activations,
        loaded=loaded,
        selected_records=selected_records,
        truncated_neutral_records=truncated_neutral_records,
        cleaning=cleaning,
        story_validation=story_validation,
        tokenwise_counts=tokenwise_counts,
        validation_layer=validation_layer,
        pca_device=pca_device,
        pca_dtype=pca_dtype,
        end_timestamp=end_timestamp,
    )
    atomic_write_json(output_dir / "metadata.json", metadata)
    update_progress(
        output_dir,
        status="completed",
        stage="completed",
        completed_neutral_records=len(selected_records),
        neutral_activation_shape=list(neutral_activations.shape),
        component_counts={
            layer: values["num_components"]
            for layer, values in pca_summary["layers"].items()
        },
        final_artifacts_complete=True,
        completed_at=end_timestamp,
    )
    logger.info(
        "neutral_pca_cleaning_complete neutral_records=%d layers=%d emotions=%d "
        "tokenwise_records=%d",
        len(selected_records),
        NUM_LAYERS,
        len(EMOTIONS),
        sum(tokenwise_counts.values()),
    )
    return 0


def validate_cli_constraints(args: argparse.Namespace) -> None:
    """Reject settings that would change the requested experiment definition."""

    if args.model != TARGET_MODEL:
        raise ValueError(f"--model must be exactly {TARGET_MODEL}")
    if args.neutral_text_field != "transcript":
        raise ValueError("--neutral-text-field must be exactly 'transcript'")
    if args.neutral_start_token_position < 1:
        raise ValueError("--neutral-start-token-position must be at least 1")
    if (
        args.max_sequence_length is not None
        and args.max_sequence_length < args.neutral_start_token_position
    ):
        raise ValueError(
            "--max-sequence-length must be at least the neutral start position"
        )
    if (
        args.max_neutral_records is not None
        and args.output_dir.expanduser().resolve() == DEFAULT_OUTPUT_DIR.resolve()
    ):
        raise ValueError(
            "--max-neutral-records requires a separate --output-dir so partial "
            "artifacts cannot mix with full cleaning outputs"
        )


def prepare_output_directory(output_dir: Path, *, resume: bool) -> None:
    """Create a fresh output directory or authorize a compatible resume."""

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --resume or use "
            "a separate directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_neutral_records(path: Path) -> LoadedNeutralRecords:
    """Load only basic-valid neutral records; emotion words are never inspected."""

    if not path.is_file():
        raise FileNotFoundError(f"Neutral JSONL does not exist: {path}")
    records: list[NeutralRecord] = []
    invalid: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_lines += 1
            row: Any = None
            reason: str | None = None
            if not line.strip():
                reason = "blank line is not valid JSON"
            else:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    reason = f"invalid JSON: {error.msg}"
            if reason is None:
                if not isinstance(row, dict):
                    reason = "JSON value is not an object"
                elif "record_id" not in row:
                    reason = "missing record_id"
                elif row.get("label") != "neutral":
                    reason = "label is not neutral"
                elif "transcript" not in row:
                    reason = "missing transcript"
                elif not isinstance(row["transcript"], str) or not row[
                    "transcript"
                ].strip():
                    reason = "transcript is not a nonempty string"
            if reason is not None:
                invalid.append(
                    invalid_neutral_entry(
                        line_number=line_number,
                        row=row,
                        reason=reason,
                    )
                )
                continue
            record_id = str(row["record_id"])
            if record_id in seen_ids:
                raise ValueError(f"Duplicate valid neutral record_id: {record_id}")
            seen_ids.add(record_id)
            records.append(NeutralRecord(payload=row, line_number=line_number))
    return LoadedNeutralRecords(
        records=records,
        invalid_records=invalid,
        total_lines=total_lines,
    )


def invalid_neutral_entry(
    *, line_number: int, row: Any, reason: str
) -> dict[str, Any]:
    """Create the requested compact invalid-neutral audit row."""

    entry: dict[str, Any] = {
        "input_line_number": line_number,
        "record_id": row.get("record_id") if isinstance(row, dict) else None,
        "topic_id": row.get("topic_id") if isinstance(row, dict) else None,
        "failure_reason": reason,
    }
    return entry


def select_neutral_records(
    *, loaded: LoadedNeutralRecords, max_neutral_records: int | None
) -> list[NeutralRecord]:
    """Select the original input prefix for an explicitly partial manual run."""

    if max_neutral_records is None:
        return list(loaded.records)
    return list(loaded.records[:max_neutral_records])


def discover_and_validate_emotion_artifacts(
    args: argparse.Namespace,
) -> EmotionArtifacts:
    """Discover and validate raw vectors, metadata, and saved story activations."""

    explicit_paths = [
        args.raw_emotion_vectors,
        args.emotion_metadata,
        args.story_activations_dir,
    ]
    candidate_dirs: list[Path] = []
    for value in explicit_paths:
        if value is None:
            continue
        resolved = value.expanduser().resolve()
        candidate_dirs.append(resolved if resolved.is_dir() else resolved.parent)
    candidate_dirs.extend(
        [
            DEFAULT_EMOTION_OUTPUT_BASE,
            Path("outputs/Qwen2.5-7B-Instruct/raw_extraction").resolve(),
        ]
    )
    base_dir: Path | None = None
    for candidate in candidate_dirs:
        required = (
            candidate / "emotion_vectors_raw.pt",
            candidate / "emotion_vectors_unit.pt",
            candidate / "emotion_vectors_stacked.pt",
            candidate / "metadata.json",
            candidate / "story_activations",
        )
        if all(path.exists() for path in required):
            base_dir = candidate
            break
    if base_dir is None:
        raise FileNotFoundError(
            "Could not discover a completed emotion-vector output directory"
        )

    raw_path = (
        args.raw_emotion_vectors.expanduser().resolve()
        if args.raw_emotion_vectors is not None
        else base_dir / "emotion_vectors_raw.pt"
    )
    metadata_path = (
        args.emotion_metadata.expanduser().resolve()
        if args.emotion_metadata is not None
        else base_dir / "metadata.json"
    )
    story_activations_dir = (
        args.story_activations_dir.expanduser().resolve()
        if args.story_activations_dir is not None
        else base_dir / "story_activations"
    )
    story_records_dir = (
        args.emotional_stories_dir.expanduser().resolve()
        if args.emotional_stories_dir is not None
        else DEFAULT_STORY_RECORDS_DIR
    )
    unit_path = raw_path.parent / "emotion_vectors_unit.pt"
    stacked_path = raw_path.parent / "emotion_vectors_stacked.pt"
    for path in (raw_path, unit_path, stacked_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required emotion artifact is missing: {path}")
    if not story_activations_dir.is_dir():
        raise NotADirectoryError(
            f"Story activation directory is missing: {story_activations_dir}"
        )
    if not story_records_dir.is_dir():
        raise NotADirectoryError(
            f"Original emotional-story directory is missing: {story_records_dir}"
        )

    metadata = read_json_object(metadata_path)
    validate_emotion_metadata(metadata)
    raw_vectors = load_vector_dictionary(raw_path, "raw emotion vectors")
    raw_unit_vectors = load_vector_dictionary(unit_path, "raw unit emotion vectors")
    validate_raw_and_unit_vectors(raw_vectors, raw_unit_vectors)
    validate_stacked_raw_vectors(stacked_path, raw_vectors)
    validate_story_activation_headers(story_activations_dir, metadata)
    validate_story_record_files(story_records_dir, metadata=metadata)
    revision = str(metadata["resolved_model_revision"])
    return EmotionArtifacts(
        base_dir=base_dir,
        raw_path=raw_path,
        unit_path=unit_path,
        stacked_path=stacked_path,
        metadata_path=metadata_path,
        story_activations_dir=story_activations_dir,
        story_records_dir=story_records_dir,
        metadata=metadata,
        raw_vectors=raw_vectors,
        raw_unit_vectors=raw_unit_vectors,
        model_revision=revision,
    )


def validate_emotion_metadata(metadata: dict[str, Any]) -> None:
    """Require exact compatibility with the completed emotional extraction."""

    expected = {
        "status": "completed",
        "target_model": TARGET_MODEL,
        "resolved_model_revision": EXPECTED_MODEL_REVISION,
        "resolved_tokenizer_revision": EXPECTED_MODEL_REVISION,
        "tokenizer_class": "Qwen2TokenizerFast",
        "model_class": "Qwen2ForCausalLM",
        "number_of_transformer_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "start_token_position": EMOTIONAL_STORY_START_POSITION,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "chat_template_used": False,
        "embedding_hidden_state_included": False,
        "residual_stream_definition": RESIDUAL_STREAM_DEFINITION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "layers_averaged_together": False,
    }
    for field, expected_value in expected.items():
        if metadata.get(field) != expected_value:
            raise ValueError(
                f"Emotion metadata mismatch for {field}: "
                f"{metadata.get(field)!r} != {expected_value!r}"
            )
    if metadata.get("valid_story_activations") != len(EMOTIONS) * EXPECTED_STORIES_PER_EMOTION:
        raise ValueError("Emotion metadata has an unexpected valid-story count")


def load_vector_dictionary(path: Path, description: str) -> dict[str, torch.Tensor]:
    """Load a 12-emotion dictionary of finite float32 [28,3584] tensors."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not load {description} from {path}: {error}") from error
    if not isinstance(value, dict) or set(value) != set(EMOTIONS):
        raise ValueError(f"{description} has unexpected emotion keys")
    for emotion in EMOTIONS:
        tensor = value[emotion]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (NUM_LAYERS, HIDDEN_SIZE)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"Invalid {description} tensor for {emotion}")
    return value


def validate_raw_and_unit_vectors(
    raw_vectors: dict[str, torch.Tensor],
    unit_vectors: dict[str, torch.Tensor],
) -> None:
    """Verify existing unit vectors are the per-layer normalization of raw vectors."""

    for emotion in EMOTIONS:
        raw = raw_vectors[emotion]
        expected = raw / raw.norm(p=2, dim=-1, keepdim=True).clamp_min(EPSILON)
        if not torch.allclose(
            unit_vectors[emotion], expected, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(
                f"Existing raw unit vectors do not exactly match raw vectors: {emotion}"
            )


def validate_stacked_raw_vectors(
    path: Path, raw_vectors: dict[str, torch.Tensor]
) -> None:
    """Validate the existing stacked raw representation [12,28,3584]."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("emotion_order") != list(EMOTIONS):
        raise ValueError(f"Invalid stacked raw-vector metadata: {path}")
    vectors = payload.get("vectors")
    expected = torch.stack([raw_vectors[emotion] for emotion in EMOTIONS], dim=0)
    if (
        not isinstance(vectors, torch.Tensor)
        or vectors.dtype != torch.float32
        or tuple(vectors.shape) != (len(EMOTIONS), NUM_LAYERS, HIDDEN_SIZE)
        or not torch.equal(vectors, expected)
    ):
        raise ValueError("Stacked raw vectors do not exactly match the raw dictionary")


def validate_story_activation_headers(
    directory: Path, emotion_metadata: dict[str, Any]
) -> None:
    """Validate each saved story tensor's model, mapping, shape, and finiteness."""

    expected_fingerprint = emotion_metadata.get("extraction_configuration_sha256")
    for emotion in EMOTIONS:
        path = directory / f"{emotion}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing story activation file: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Story activation payload is not a dictionary: {path}")
        expected_fields = {
            "emotion": emotion,
            "target_model": TARGET_MODEL,
            "model_revision": EXPECTED_MODEL_REVISION,
            "start_token_position": EMOTIONAL_STORY_START_POSITION,
            "token_position_indexing": TOKEN_POSITION_INDEXING,
            "input_text_field": "story",
            "chat_template_used": False,
            "embedding_hidden_state_included": False,
            "hidden_state_mapping": HIDDEN_STATE_MAPPING,
            "num_transformer_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "story_activation_dtype": "float16",
            "extraction_configuration_sha256": expected_fingerprint,
        }
        for field, expected in expected_fields.items():
            if payload.get(field) != expected:
                raise ValueError(f"Story activation mismatch in {path}: {field}")
        activations = payload.get("activations")
        if (
            not isinstance(activations, torch.Tensor)
            or activations.dtype != torch.float16
            or tuple(activations.shape)
            != (EXPECTED_STORIES_PER_EMOTION, NUM_LAYERS, HIDDEN_SIZE)
            or not bool(torch.isfinite(activations).all())
        ):
            raise ValueError(f"Invalid saved story activation tensor: {path}")
        record_ids = payload.get("record_ids")
        if not isinstance(record_ids, list) or len(record_ids) != EXPECTED_STORIES_PER_EMOTION:
            raise ValueError(f"Invalid record IDs in story activations: {path}")
        activation_sum = payload.get("activation_sum_float64")
        if (
            not isinstance(activation_sum, torch.Tensor)
            or activation_sum.dtype != torch.float64
            or tuple(activation_sum.shape) != (NUM_LAYERS, HIDDEN_SIZE)
            or not bool(torch.isfinite(activation_sum).all())
        ):
            raise ValueError(f"Invalid activation sum in story activations: {path}")
        parallel_fields = (
            "emotion_groups",
            "topic_ids",
            "topics",
            "sample_indices",
            "prompt_versions",
            "generator_models",
            "accepted_seeds",
            "attempt_counts",
            "emotion_word_present",
            "non_latin_letter_present",
            "accepted_after_max_attempts",
            "created_at_values",
            "original_token_counts",
            "processed_token_counts",
            "truncated",
            "input_filenames",
            "input_line_numbers",
            "source_metadata",
        )
        for field in parallel_fields:
            values = payload.get(field)
            if not isinstance(values, list) or len(values) != EXPECTED_STORIES_PER_EMOTION:
                raise ValueError(
                    f"Misaligned {field!r} metadata in story activations: {path}"
                )
        del payload, activations


def validate_story_record_files(
    directory: Path, *, metadata: dict[str, Any]
) -> None:
    """Bind every original story JSONL to the hashes used for extraction."""

    extraction = metadata.get("extraction_configuration")
    source_files = extraction.get("source_files") if isinstance(extraction, dict) else None
    if not isinstance(source_files, list):
        raise ValueError("Emotion metadata does not contain source-file fingerprints")
    expected_by_name: dict[str, dict[str, Any]] = {}
    for value in source_files:
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise ValueError("Emotion metadata has an invalid source-file fingerprint")
        name = Path(value["path"]).name
        if name in expected_by_name:
            raise ValueError(f"Duplicate source-file fingerprint for {name}")
        expected_by_name[name] = value

    for emotion in EMOTIONS:
        path = directory / f"{emotion}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing original emotional stories: {path}")
        expected = expected_by_name.get(path.name)
        if expected is None:
            raise ValueError(f"No extraction fingerprint exists for {path.name}")
        if path.stat().st_size != int(expected.get("size_bytes", -1)):
            raise ValueError(f"Original emotional-story size mismatch: {path}")
        if sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"Original emotional-story SHA-256 mismatch: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        if count != EXPECTED_STORIES_PER_EMOTION:
            raise ValueError(f"Expected 1200 stories in {path}; found {count}")


def validate_loaded_model(
    model_bundle: dict[str, Any], emotion_metadata: dict[str, Any]
) -> None:
    """Validate exact model, tokenizer, dimensions, revision, and mapping context."""

    model = model_bundle["model"]
    checks = {
        "model revision": (
            model_bundle["resolved_revision"],
            emotion_metadata["resolved_model_revision"],
        ),
        "tokenizer revision": (
            model_bundle["tokenizer_revision"],
            emotion_metadata["resolved_tokenizer_revision"],
        ),
        "tokenizer class": (
            model_bundle["tokenizer_class"],
            emotion_metadata["tokenizer_class"],
        ),
        "model class": (model_bundle["model_class"], emotion_metadata["model_class"]),
        "number of layers": (int(model.config.num_hidden_layers), NUM_LAYERS),
        "hidden size": (int(model.config.hidden_size), HIDDEN_SIZE),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"Loaded {name} mismatch: {observed!r} != {expected!r}")
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False


def build_unloaded_model_bundle_metadata(
    *, args: argparse.Namespace, artifacts: EmotionArtifacts
) -> dict[str, Any]:
    """Rebuild immutable model metadata without loading weights for a full resume."""

    try:
        import transformers
    except ImportError as error:
        raise RuntimeError("transformers is required for resume validation") from error

    resolved_dtype = resolve_torch_dtype(args.dtype)
    resolved_dtype_name = (
        artifacts.metadata["resolved_model_dtype"]
        if resolved_dtype == "auto"
        else str(resolved_dtype)
    )
    gpu_name: str | None = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    bundle = {
        "resolved_dtype": resolved_dtype_name,
        "resolved_revision": artifacts.model_revision,
        "tokenizer_revision": artifacts.metadata["resolved_tokenizer_revision"],
        "tokenizer_class": artifacts.metadata["tokenizer_class"],
        "model_class": artifacts.metadata["model_class"],
        "model_config_class": artifacts.metadata["model_configuration_class"],
        "transformers_version": transformers.__version__,
        "gpu_name": gpu_name,
    }
    checks = {
        "requested model revision": (
            args.model_revision or artifacts.model_revision,
            artifacts.model_revision,
        ),
        "tokenizer revision": (
            bundle["tokenizer_revision"],
            artifacts.model_revision,
        ),
        "Transformers version": (
            bundle["transformers_version"],
            artifacts.metadata["transformers_version"],
        ),
        "PyTorch version": (torch.__version__, artifacts.metadata["pytorch_version"]),
        "resolved model dtype": (
            bundle["resolved_dtype"],
            artifacts.metadata["resolved_model_dtype"],
        ),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(
                f"Unloaded-resume {name} mismatch: {observed!r} != {expected!r}"
            )
    return bundle


def build_compatibility_signature(
    *,
    args: argparse.Namespace,
    neutral_path: Path,
    selected_records: Sequence[NeutralRecord],
    artifacts: EmotionArtifacts,
    model_bundle: dict[str, Any],
    partial: bool,
) -> dict[str, Any]:
    """Build the immutable run fingerprint used to guard resume compatibility."""

    story_sources = []
    for emotion in EMOTIONS:
        path = artifacts.story_records_dir / f"{emotion}.jsonl"
        story_sources.append(
            {
                "emotion": emotion,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "neutral_input": {
            "path": str(neutral_path),
            "size_bytes": neutral_path.stat().st_size,
            "sha256": sha256_file(neutral_path),
            "text_field": args.neutral_text_field,
            "label": "neutral",
            "selected_record_ids_sha256": canonical_json_sha256(
                [record.record_id for record in selected_records]
            ),
            "selected_record_count": len(selected_records),
            "partial_neutral_dataset": partial,
        },
        "raw_emotion_vectors": {
            "path": str(artifacts.raw_path),
            "size_bytes": artifacts.raw_path.stat().st_size,
            "sha256": sha256_file(artifacts.raw_path),
        },
        "emotion_metadata": {
            "path": str(artifacts.metadata_path),
            "size_bytes": artifacts.metadata_path.stat().st_size,
            "sha256": sha256_file(artifacts.metadata_path),
            "extraction_configuration_sha256": artifacts.metadata.get(
                "extraction_configuration_sha256"
            ),
        },
        "story_activations_directory": str(artifacts.story_activations_dir),
        "story_record_sources": story_sources,
        "model": {
            "name": args.model,
            "requested_revision": args.model_revision,
            "resolved_revision": model_bundle["resolved_revision"],
            "tokenizer_name": args.model,
            "tokenizer_revision": model_bundle["tokenizer_revision"],
            "tokenizer_class": model_bundle["tokenizer_class"],
            "model_class": model_bundle["model_class"],
            "model_configuration_class": model_bundle["model_config_class"],
            "transformers_version": model_bundle["transformers_version"],
            "pytorch_version": torch.__version__,
            "num_transformer_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "hidden_state_mapping": HIDDEN_STATE_MAPPING,
            "residual_stream_definition": RESIDUAL_STREAM_DEFINITION,
            "embedding_hidden_state_included": False,
            "chat_template_used": False,
            "model_dtype": args.dtype,
            "resolved_model_dtype": model_bundle["resolved_dtype"],
        },
        "neutral_extraction": {
            "sample_unit": NEUTRAL_SAMPLE_UNIT,
            "token_aggregation": NEUTRAL_TOKEN_AGGREGATION,
            "start_token_position": args.neutral_start_token_position,
            "token_position_indexing": TOKEN_POSITION_INDEXING,
            "add_special_tokens": True,
            "padding_side": "right",
            "truncation_side": "right",
            "truncation_enabled": args.max_sequence_length is not None,
            "maximum_sequence_length": args.max_sequence_length,
            "activation_storage_dtype": args.neutral_activation_dtype,
            "token_averaging_dtype": "float32",
        },
        "pca": {
            "solver": "torch.linalg.svd(full_matrices=False)",
            "requested_device": args.pca_device,
            "dtype": args.pca_dtype,
            "explained_variance_threshold": (
                args.neutral_explained_variance_threshold
            ),
            "eigenvalue_tolerance": args.eigenvalue_tolerance,
            "component_orthonormality_tolerance": (
                args.component_orthonormality_tolerance
            ),
            "performed_separately_per_layer": True,
            "neutral_matrix_centered": True,
        },
        "cleaning": {
            "input": "raw unnormalized emotion vectors",
            "projection_orthogonality_tolerance": (
                args.projection_orthogonality_tolerance
            ),
            "layers_averaged_together": False,
        },
        "validation": {
            "story_validation_batch_size": args.story_validation_batch_size,
            "tokenwise_stories_per_emotion": args.tokenwise_stories_per_emotion,
            "tokenwise_validation_seed": args.tokenwise_validation_seed,
            "tokenwise_validation_layer": resolve_tokenwise_layer(
                args.tokenwise_validation_layer
            ),
        },
    }


def build_neutral_extraction_fingerprint(
    full_signature: dict[str, Any]
) -> dict[str, Any]:
    """Select only fields that determine transcript-level activation shards."""

    return {
        "schema_version": full_signature["schema_version"],
        "neutral_input": full_signature["neutral_input"],
        "model": full_signature["model"],
        "neutral_extraction": full_signature["neutral_extraction"],
    }


def prepare_config(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    fingerprint: dict[str, Any],
    created_at: str,
    resume: bool,
) -> dict[str, Any]:
    """Create complete configuration metadata or verify a resumed run."""

    path = output_dir / "config.json"
    signature_hash = canonical_json_sha256(fingerprint)
    invocation = {
        "timestamp": created_at,
        "batch_size": args.batch_size,
        "device": args.device,
        "resume": bool(args.resume),
    }
    if path.exists():
        if not resume:
            raise FileExistsError(f"Existing config requires --resume: {path}")
        config = read_json_object(path)
        if config.get("compatibility_signature") != fingerprint:
            raise ValueError(
                "Requested settings do not match the existing neutral-PCA run"
            )
        if config.get("compatibility_signature_sha256") != signature_hash:
            raise ValueError("Existing neutral-PCA configuration fingerprint mismatch")
        invocations = config.setdefault("invocations", [])
        if not isinstance(invocations, list):
            raise ValueError("Existing config has invalid invocation metadata")
        invocations.append(invocation)
        config["last_cli_arguments"] = json_safe_cli_arguments(args)
        config["pca_solver"] = "torch.linalg.svd(full_matrices=False)"
        config["pca_svd_driver_policy"] = (
            "gesvd on CUDA for accuracy; PyTorch default on CPU"
        )
        atomic_write_json(path, config)
        return config

    config = {
        "neutral_jsonl_path": fingerprint["neutral_input"]["path"],
        "raw_emotion_vector_path": fingerprint["raw_emotion_vectors"]["path"],
        "emotion_metadata_path": fingerprint["emotion_metadata"]["path"],
        "story_activations_directory": fingerprint[
            "story_activations_directory"
        ],
        "emotional_stories_directory": str(
            args.emotional_stories_dir.expanduser().resolve()
            if args.emotional_stories_dir is not None
            else DEFAULT_STORY_RECORDS_DIR
        ),
        "model_name": args.model,
        "model_revision": fingerprint["model"]["resolved_revision"],
        "tokenizer": {
            "name": args.model,
            "revision": fingerprint["model"]["tokenizer_revision"],
            "class": fingerprint["model"]["tokenizer_class"],
        },
        "neutral_input_field": args.neutral_text_field,
        "neutral_start_token_position": args.neutral_start_token_position,
        "emotional_story_start_token_position": EMOTIONAL_STORY_START_POSITION,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "batch_size": args.batch_size,
        "model_dtype": args.dtype,
        "neutral_activation_storage_dtype": args.neutral_activation_dtype,
        "pca_dtype": args.pca_dtype,
        "pca_requested_device": args.pca_device,
        "pca_solver": "torch.linalg.svd(full_matrices=False)",
        "pca_svd_driver_policy": (
            "gesvd on CUDA for accuracy; PyTorch default on CPU"
        ),
        "explained_variance_threshold": args.neutral_explained_variance_threshold,
        "eigenvalue_tolerance": args.eigenvalue_tolerance,
        "component_orthonormality_tolerance": (
            args.component_orthonormality_tolerance
        ),
        "projection_orthogonality_tolerance": (
            args.projection_orthogonality_tolerance
        ),
        "truncation_enabled": args.max_sequence_length is not None,
        "maximum_sequence_length": args.max_sequence_length,
        "tokenwise_validation_layer": resolve_tokenwise_layer(
            args.tokenwise_validation_layer
        ),
        "tokenwise_sample_count": args.tokenwise_stories_per_emotion,
        "tokenwise_validation_seed": args.tokenwise_validation_seed,
        "output_directory": str(output_dir),
        "complete_cli_arguments": json_safe_cli_arguments(args),
        "creation_timestamp": created_at,
        "compatibility_signature": fingerprint,
        "compatibility_signature_sha256": signature_hash,
        "invocations": [invocation],
    }
    atomic_write_json(path, config)
    return config


def build_running_metadata(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    fingerprint: dict[str, Any],
    model_bundle: dict[str, Any],
    neutral_path: Path,
    loaded: LoadedNeutralRecords,
    selected_records: Sequence[NeutralRecord],
    artifacts: EmotionArtifacts,
    partial: bool,
) -> dict[str, Any]:
    """Build result metadata before extraction begins."""

    existing_path = args.output_dir.expanduser().resolve() / "metadata.json"
    existing = read_json_object(existing_path) if existing_path.exists() else {}
    metadata = {
        **existing,
        "status": "running",
        "target_model": args.model,
        "exact_model_revision": model_bundle["resolved_revision"],
        "resolved_tokenizer_revision": model_bundle["tokenizer_revision"],
        "tokenizer_class": model_bundle["tokenizer_class"],
        "model_class": model_bundle["model_class"],
        "model_configuration_class": model_bundle["model_config_class"],
        "neutral_input_file": str(neutral_path),
        "neutral_input_field": "transcript",
        "neutral_label": "neutral",
        "expected_neutral_records": EXPECTED_NEUTRAL_RECORDS,
        "neutral_pca_sample_unit": NEUTRAL_SAMPLE_UNIT,
        "neutral_token_aggregation": NEUTRAL_TOKEN_AGGREGATION,
        "neutral_start_token_position": args.neutral_start_token_position,
        "emotional_story_start_token_position": EMOTIONAL_STORY_START_POSITION,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "public_method_detail_status": PUBLIC_METHOD_DETAIL_STATUS,
        "chat_template_used": False,
        "embedding_hidden_state_included": False,
        "number_of_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "residual_stream_definition": RESIDUAL_STREAM_DEFINITION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "neutral_matrix_centered_before_pca": True,
        "pca_performed_separately_per_layer": True,
        "neutral_explained_variance_threshold": (
            args.neutral_explained_variance_threshold
        ),
        "cleaning_definition": (
            "raw emotion vector minus its orthogonal projection onto the retained "
            "neutral PCA basis at the matching layer"
        ),
        "layers_averaged_together": False,
        "resolved_model_dtype": model_bundle["resolved_dtype"],
        "pytorch_version": torch.__version__,
        "transformers_version": fingerprint["model"].get(
            "transformers_version", model_bundle.get("transformers_version")
        ),
        "cuda_version": torch.version.cuda,
        "gpu_name": model_bundle["gpu_name"],
        "total_neutral_records": loaded.total_lines,
        "valid_neutral_records": len(loaded.records),
        "invalid_neutral_records": len(loaded.invalid_records),
        "selected_neutral_records": len(selected_records),
        "partial_neutral_dataset": partial,
        "truncated_neutral_records": existing.get("truncated_neutral_records", 0),
        "raw_emotion_vector_path": str(artifacts.raw_path),
        "emotion_metadata_path": str(artifacts.metadata_path),
        "story_activations_directory": str(artifacts.story_activations_dir),
        "original_emotional_stories_directory": str(artifacts.story_records_dir),
        "raw_emotion_vector_shape": [len(EMOTIONS), NUM_LAYERS, HIDDEN_SIZE],
        "cleaning_start_timestamp": existing.get(
            "cleaning_start_timestamp", config["creation_timestamp"]
        ),
        "cleaning_end_timestamp": None,
        "compatibility_signature_sha256": canonical_json_sha256(fingerprint),
    }
    metadata.pop("failure", None)
    return metadata


def load_existing_activation_shards(
    *,
    shard_dir: Path,
    fingerprint: dict[str, Any],
    expected_record_ids: set[str],
    activation_dtype: str,
) -> tuple[list[Path], set[str]]:
    """Validate resumable shard files and return their unique completed IDs."""

    paths = sorted(shard_dir.glob("shard_*.pt"))
    completed: set[str] = set()
    for path in paths:
        payload = load_and_validate_activation_shard(
            path=path,
            fingerprint=fingerprint,
            activation_dtype=activation_dtype,
        )
        for record_id in payload["record_ids"]:
            record_id = str(record_id)
            if record_id not in expected_record_ids:
                raise ValueError(f"Shard record is outside this run: {record_id}")
            if record_id in completed:
                raise ValueError(f"Duplicate record across neutral shards: {record_id}")
            completed.add(record_id)
    return paths, completed


def load_and_validate_activation_shard(
    *, path: Path, fingerprint: dict[str, Any], activation_dtype: str
) -> dict[str, Any]:
    """Validate one shard with activations [records,28,3584]."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not load neutral activation shard {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Neutral activation shard is not a dictionary: {path}")
    if payload.get("extraction_fingerprint") != fingerprint:
        raise ValueError(f"Neutral extraction fingerprint mismatch: {path}")
    if payload.get("extraction_fingerprint_sha256") != canonical_json_sha256(
        fingerprint
    ):
        raise ValueError(f"Neutral extraction fingerprint hash mismatch: {path}")
    if payload.get("target_model") != TARGET_MODEL:
        raise ValueError(f"Unexpected target model in shard: {path}")
    if payload.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise ValueError(f"Unexpected model revision in shard: {path}")
    if payload.get("input_text_field") != "transcript":
        raise ValueError(f"Unexpected neutral text field in shard: {path}")
    if payload.get("hidden_state_mapping") != HIDDEN_STATE_MAPPING:
        raise ValueError(f"Unexpected hidden-state mapping in shard: {path}")

    activations = payload.get("activations")
    expected_dtype = torch.float16 if activation_dtype == "float16" else torch.float32
    if (
        not isinstance(activations, torch.Tensor)
        or activations.ndim != 3
        or tuple(activations.shape[1:]) != (NUM_LAYERS, HIDDEN_SIZE)
        or activations.dtype != expected_dtype
        or not bool(torch.isfinite(activations).all())
    ):
        raise ValueError(f"Invalid activation tensor in shard: {path}")
    count = int(activations.shape[0])
    list_fields = (
        "record_ids",
        "topic_ids",
        "topics",
        "sample_indices",
        "dialogue_types",
        "system_instruction_included",
        "prompt_versions",
        "accepted_seeds",
        "attempt_counts",
        "original_token_counts",
        "processed_token_counts",
        "truncated",
    )
    for field in list_fields:
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Shard metadata {field!r} is misaligned in {path}")
    if len(set(str(value) for value in payload["record_ids"])) != count:
        raise ValueError(f"Duplicate record ID inside neutral shard: {path}")
    return payload


def extract_neutral_activations(
    *,
    records: Sequence[NeutralRecord],
    completed_ids: set[str],
    existing_shards: Sequence[Path],
    shard_dir: Path,
    fingerprint: dict[str, Any],
    model: Any,
    tokenizer: Any,
    model_input_device: torch.device,
    batch_size: int,
    save_every: int,
    max_sequence_length: int | None,
    start_token_position: int,
    storage_dtype_name: str,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Extract CPU transcript means [batch,28,3584] and save atomic shards."""

    remaining = [record for record in records if record.record_id not in completed_ids]
    if not remaining:
        return
    next_shard_index = next_available_shard_index(existing_shards)
    pending_activations: list[torch.Tensor] = []
    pending_records: list[NeutralRecord] = []
    pending_token_metadata: list[tuple[int, int, bool]] = []
    storage_dtype = (
        torch.float16 if storage_dtype_name == "float16" else torch.float32
    )

    try:
        for batch_start in range(0, len(remaining), batch_size):
            batch_records = list(remaining[batch_start : batch_start + batch_size])
            transcripts = [record.transcript for record in batch_records]
            encoded, original_counts, processed_counts = tokenize_plain_text_batch(
                tokenizer=tokenizer,
                texts=transcripts,
                max_sequence_length=max_sequence_length,
            )
            try:
                input_ids = encoded["input_ids"].to(model_input_device)
                attention_mask = encoded["attention_mask"].to(model_input_device)
                content_positions = attention_mask.long().cumsum(dim=1)
                selected_token_mask = attention_mask.bool() & (
                    content_positions >= start_token_position
                )
                selected_counts = selected_token_mask.sum(dim=1, keepdim=True)
                if bool(torch.any(selected_counts <= 0)):
                    raise ValueError(
                        "Neutral token-selection mask produced an empty transcript mean"
                    )
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states
                    if hidden_states is None or len(hidden_states) != NUM_LAYERS + 1:
                        observed = None if hidden_states is None else len(hidden_states)
                        raise ValueError(
                            "Hidden-state tuple length mismatch: "
                            f"expected 29, received {observed}"
                        )
                    layer_means: list[torch.Tensor] = []
                    for layer_index in range(NUM_LAYERS):
                        hidden = hidden_states[layer_index + 1]
                        expected_shape = (
                            len(batch_records),
                            int(input_ids.shape[1]),
                            HIDDEN_SIZE,
                        )
                        if tuple(hidden.shape) != expected_shape:
                            raise ValueError(
                                f"Layer {layer_index} hidden shape {tuple(hidden.shape)} "
                                f"does not match {expected_shape}"
                            )
                        mask = selected_token_mask.to(hidden.device).unsqueeze(-1)
                        counts = selected_counts.to(
                            hidden.device, dtype=torch.float32
                        ).clamp_min(1)
                        hidden_fp32 = hidden.float()
                        summed = (hidden_fp32 * mask).sum(dim=1)
                        layer_mean = (summed / counts).to(
                            device="cpu", dtype=torch.float32
                        )
                        layer_means.append(layer_mean)
                        del hidden_fp32, summed, mask, counts, layer_mean
                    batch_activations = torch.stack(layer_means, dim=1)
            except RuntimeError as error:
                if is_out_of_memory(error):
                    if pending_records:
                        destination = flush_pending_neutral_shard(
                            shard_dir=shard_dir,
                            shard_index=next_shard_index,
                            records=pending_records,
                            activations=pending_activations,
                            token_metadata=pending_token_metadata,
                            fingerprint=fingerprint,
                            storage_dtype=storage_dtype,
                        )
                        for record in pending_records:
                            completed_ids.add(record.record_id)
                        update_progress(
                            output_dir,
                            status="running",
                            stage="neutral_activation_extraction",
                            completed_neutral_records=len(completed_ids),
                            total_selected_neutral_records=len(records),
                            last_saved_shard=str(destination),
                        )
                    logger.error(
                        "neutral_extraction_out_of_memory record_ids=%s "
                        "sequence_lengths=%s batch_size=%d recommendation='rerun "
                        "with --resume and a smaller --batch-size'",
                        [record.record_id for record in batch_records],
                        processed_counts,
                        len(batch_records),
                    )
                    raise NeutralExtractionOutOfMemory(
                        record_ids=[record.record_id for record in batch_records],
                        sequence_lengths=processed_counts,
                        batch_size=len(batch_records),
                        original_error=error,
                    ) from error
                raise
            finally:
                if "outputs" in locals():
                    del outputs
                if "hidden_states" in locals():
                    del hidden_states

            expected_batch_shape = (
                len(batch_records),
                NUM_LAYERS,
                HIDDEN_SIZE,
            )
            if tuple(batch_activations.shape) != expected_batch_shape:
                raise ValueError(
                    f"Neutral activation batch shape {tuple(batch_activations.shape)} "
                    f"does not match {expected_batch_shape}"
                )
            if not bool(torch.isfinite(batch_activations).all()):
                raise ValueError("Non-finite neutral transcript activation detected")
            pending_activations.append(batch_activations.to(storage_dtype))
            pending_records.extend(batch_records)
            pending_token_metadata.extend(
                (
                    original_count,
                    processed_count,
                    original_count > processed_count,
                )
                for original_count, processed_count in zip(
                    original_counts, processed_counts, strict=True
                )
            )
            completed_count = len(completed_ids) + len(pending_records)
            logger.info(
                "neutral_batch_complete completed_or_pending=%d/%d batch_size=%d "
                "sequence_length=%d",
                completed_count,
                len(records),
                len(batch_records),
                int(input_ids.shape[1]),
            )

            if len(pending_records) >= save_every:
                destination = flush_pending_neutral_shard(
                    shard_dir=shard_dir,
                    shard_index=next_shard_index,
                    records=pending_records,
                    activations=pending_activations,
                    token_metadata=pending_token_metadata,
                    fingerprint=fingerprint,
                    storage_dtype=storage_dtype,
                )
                for record in pending_records:
                    completed_ids.add(record.record_id)
                next_shard_index += 1
                pending_records = []
                pending_activations = []
                pending_token_metadata = []
                update_progress(
                    output_dir,
                    status="running",
                    stage="neutral_activation_extraction",
                    completed_neutral_records=len(completed_ids),
                    total_selected_neutral_records=len(records),
                    last_saved_shard=str(destination),
                )

            del (
                batch_activations,
                layer_means,
                input_ids,
                attention_mask,
                content_positions,
                selected_token_mask,
                selected_counts,
                hidden,
            )

        if pending_records:
            destination = flush_pending_neutral_shard(
                shard_dir=shard_dir,
                shard_index=next_shard_index,
                records=pending_records,
                activations=pending_activations,
                token_metadata=pending_token_metadata,
                fingerprint=fingerprint,
                storage_dtype=storage_dtype,
            )
            for record in pending_records:
                completed_ids.add(record.record_id)
            update_progress(
                output_dir,
                status="running",
                stage="neutral_activation_extraction",
                completed_neutral_records=len(completed_ids),
                total_selected_neutral_records=len(records),
                last_saved_shard=str(destination),
            )
    except KeyboardInterrupt:
        if pending_records:
            destination = flush_pending_neutral_shard(
                shard_dir=shard_dir,
                shard_index=next_shard_index,
                records=pending_records,
                activations=pending_activations,
                token_metadata=pending_token_metadata,
                fingerprint=fingerprint,
                storage_dtype=storage_dtype,
            )
            for record in pending_records:
                completed_ids.add(record.record_id)
            update_progress(
                output_dir,
                status="interrupted",
                stage="neutral_activation_extraction",
                completed_neutral_records=len(completed_ids),
                total_selected_neutral_records=len(records),
                last_saved_shard=str(destination),
            )
        raise


def tokenize_plain_text_batch(
    *, tokenizer: Any, texts: Sequence[str], max_sequence_length: int | None
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    """Tokenize stored plain text without applying a chat template."""

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_sequence_length is not None,
        "add_special_tokens": True,
        "return_attention_mask": True,
    }
    if max_sequence_length is not None:
        kwargs["max_length"] = max_sequence_length
    encoded_raw = tokenizer(list(texts), **kwargs)
    input_ids = encoded_raw["input_ids"]
    attention_mask = encoded_raw.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    processed_counts = [int(value) for value in attention_mask.sum(dim=1).tolist()]
    if max_sequence_length is None:
        original_counts = list(processed_counts)
    else:
        original = tokenizer(
            list(texts),
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
        original_counts = [len(row) for row in original]
    return (
        {"input_ids": input_ids, "attention_mask": attention_mask},
        original_counts,
        processed_counts,
    )


def next_available_shard_index(paths: Sequence[Path]) -> int:
    """Return one more than the largest existing zero-padded shard index."""

    indexes: list[int] = []
    for path in paths:
        try:
            indexes.append(int(path.stem.split("_")[-1]))
        except ValueError as error:
            raise ValueError(f"Invalid neutral shard filename: {path.name}") from error
    return max(indexes, default=-1) + 1


def flush_pending_neutral_shard(
    *,
    shard_dir: Path,
    shard_index: int,
    records: Sequence[NeutralRecord],
    activations: Sequence[torch.Tensor],
    token_metadata: Sequence[tuple[int, int, bool]],
    fingerprint: dict[str, Any],
    storage_dtype: torch.dtype,
) -> Path:
    """Atomically save pending [records,28,3584] means and aligned metadata."""

    destination = shard_dir / f"shard_{shard_index:05d}.pt"
    if not records:
        return destination
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite neutral shard: {destination}")
    tensor = torch.cat(list(activations), dim=0)
    if (
        tuple(tensor.shape) != (len(records), NUM_LAYERS, HIDDEN_SIZE)
        or tensor.dtype != storage_dtype
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError("Pending neutral shard activation tensor is invalid")
    if len(token_metadata) != len(records):
        raise ValueError("Pending neutral token metadata is misaligned")
    payload = {
        "target_model": TARGET_MODEL,
        "model_revision": EXPECTED_MODEL_REVISION,
        "input_text_field": "transcript",
        "neutral_start_token_position": fingerprint["neutral_extraction"][
            "start_token_position"
        ],
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "neutral_pca_sample_unit": NEUTRAL_SAMPLE_UNIT,
        "neutral_token_aggregation": NEUTRAL_TOKEN_AGGREGATION,
        "chat_template_used": False,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "num_transformer_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "activation_dtype": str(storage_dtype),
        "extraction_fingerprint": fingerprint,
        "extraction_fingerprint_sha256": canonical_json_sha256(fingerprint),
        "record_ids": [record.record_id for record in records],
        "topic_ids": [record.payload.get("topic_id") for record in records],
        "topics": [record.payload.get("topic") for record in records],
        "sample_indices": [record.payload.get("sample_index") for record in records],
        "dialogue_types": [
            record.payload.get("dialogue_type") for record in records
        ],
        "system_instruction_included": [
            record.payload.get("system_instruction_included") for record in records
        ],
        "prompt_versions": [
            record.payload.get("prompt_version") for record in records
        ],
        "accepted_seeds": [
            record.payload.get("accepted_seed") for record in records
        ],
        "attempt_counts": [
            record.payload.get("attempt_count") for record in records
        ],
        "original_token_counts": [value[0] for value in token_metadata],
        "processed_token_counts": [value[1] for value in token_metadata],
        "truncated": [value[2] for value in token_metadata],
        "activations": tensor,
    }
    atomic_torch_save(destination, payload)
    load_and_validate_activation_shard(
        path=destination,
        fingerprint=fingerprint,
        activation_dtype="float16" if storage_dtype == torch.float16 else "float32",
    )
    return destination


def combine_activation_shards(
    *,
    records: Sequence[NeutralRecord],
    shard_dir: Path,
    fingerprint: dict[str, Any],
    activation_dtype: str,
    output_path: Path,
) -> dict[str, Any]:
    """Combine shards into original JSONL order: [N,28,3584]."""

    expected_dtype = torch.float16 if activation_dtype == "float16" else torch.float32
    index_by_id = {record.record_id: index for index, record in enumerate(records)}
    activations = torch.empty(
        (len(records), NUM_LAYERS, HIDDEN_SIZE), dtype=expected_dtype
    )
    metadata_fields = {
        "topic_ids": [None] * len(records),
        "topics": [None] * len(records),
        "sample_indices": [None] * len(records),
        "dialogue_types": [None] * len(records),
        "system_instruction_included": [None] * len(records),
        "prompt_versions": [None] * len(records),
        "accepted_seeds": [None] * len(records),
        "attempt_counts": [None] * len(records),
        "original_token_counts": [None] * len(records),
        "processed_token_counts": [None] * len(records),
        "truncated": [None] * len(records),
    }
    filled: set[str] = set()
    paths = sorted(shard_dir.glob("shard_*.pt"))
    if not paths:
        raise ValueError("No neutral activation shards are available")
    for path in paths:
        payload = load_and_validate_activation_shard(
            path=path,
            fingerprint=fingerprint,
            activation_dtype=activation_dtype,
        )
        for row_index, record_id_value in enumerate(payload["record_ids"]):
            record_id = str(record_id_value)
            if record_id not in index_by_id:
                raise ValueError(f"Unexpected record in neutral shard: {record_id}")
            if record_id in filled:
                raise ValueError(f"Duplicate neutral activation: {record_id}")
            destination_index = index_by_id[record_id]
            activations[destination_index] = payload["activations"][row_index]
            for field, values in metadata_fields.items():
                values[destination_index] = payload[field][row_index]
            filled.add(record_id)
        del payload
    expected_ids = set(index_by_id)
    if filled != expected_ids:
        missing = sorted(expected_ids - filled)
        raise ValueError(
            f"PCA cannot begin; missing {len(missing)} neutral activations: "
            f"{missing[:5]}"
        )
    if (
        tuple(activations.shape) != (len(records), NUM_LAYERS, HIDDEN_SIZE)
        or activations.dtype != expected_dtype
        or not bool(torch.isfinite(activations).all())
    ):
        raise ValueError("Combined neutral activation tensor is invalid")
    payload = {
        "record_ids": [record.record_id for record in records],
        "topic_ids": metadata_fields["topic_ids"],
        "sample_indices": metadata_fields["sample_indices"],
        "metadata": {
            "target_model": TARGET_MODEL,
            "model_revision": EXPECTED_MODEL_REVISION,
            "input_text_field": "transcript",
            "neutral_start_token_position": fingerprint["neutral_extraction"][
                "start_token_position"
            ],
            "token_position_indexing": TOKEN_POSITION_INDEXING,
            "neutral_pca_sample_unit": NEUTRAL_SAMPLE_UNIT,
            "neutral_token_aggregation": NEUTRAL_TOKEN_AGGREGATION,
            "chat_template_used": False,
            "embedding_hidden_state_included": False,
            "hidden_state_mapping": HIDDEN_STATE_MAPPING,
            "activation_dtype": str(expected_dtype),
            "extraction_fingerprint": fingerprint,
            "extraction_fingerprint_sha256": canonical_json_sha256(fingerprint),
            **metadata_fields,
        },
        "activations": activations,
    }
    atomic_torch_save(output_path, payload)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(reloaded, dict)
        or reloaded.get("record_ids") != payload["record_ids"]
        or not isinstance(reloaded.get("activations"), torch.Tensor)
        or not torch.equal(reloaded["activations"], activations)
    ):
        raise ValueError("Combined neutral activation artifact failed verification")
    return payload


def resolve_pca_device(requested: str, dtype_name: str) -> torch.device:
    """Resolve the actual exact-SVD device and avoid float64 CUDA by policy."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--pca-device cuda requested, but CUDA is unavailable")
    if dtype_name == "float64" and requested == "cuda":
        return torch.device("cpu")
    return torch.device(requested)


def compute_layerwise_pca(
    *,
    neutral_activations: torch.Tensor,
    variance_threshold: float,
    eigenvalue_tolerance: float,
    orthonormality_tolerance: float,
    pca_device: torch.device,
    pca_dtype: torch.dtype,
    logger: logging.Logger,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], torch.Tensor]:
    """Run 28 exact centered SVDs on independent [N,3584] layer matrices."""

    if (
        neutral_activations.ndim != 3
        or tuple(neutral_activations.shape[1:]) != (NUM_LAYERS, HIDDEN_SIZE)
        or not bool(torch.isfinite(neutral_activations).all())
    ):
        raise ValueError("Neutral activations are invalid before PCA")
    num_records = int(neutral_activations.shape[0])
    if num_records < 2:
        raise ValueError("At least two neutral transcripts are required for PCA")
    layer_means = torch.empty((NUM_LAYERS, HIDDEN_SIZE), dtype=torch.float32)
    pca_payload: dict[int, dict[str, Any]] = {}
    summary_layers: dict[str, dict[str, Any]] = {}
    svd_driver: str | None = "gesvd" if pca_device.type == "cuda" else None
    pca_solver = (
        "torch.linalg.svd(full_matrices=False, driver='gesvd')"
        if svd_driver is not None
        else "torch.linalg.svd(full_matrices=False)"
    )

    for layer_index in range(NUM_LAYERS):
        matrix = neutral_activations[:, layer_index, :].to(
            device=pca_device, dtype=pca_dtype
        )
        if tuple(matrix.shape) != (num_records, HIDDEN_SIZE):
            raise ValueError(f"Layer {layer_index} PCA matrix has an invalid shape")
        layer_mean = matrix.mean(dim=0, keepdim=True)
        centered = matrix - layer_mean
        centered_column_mean = centered.mean(dim=0)
        centered_max_abs_mean = float(centered_column_mean.abs().max().item())
        scale = max(float(matrix.abs().max().item()), 1.0)
        dtype_epsilon = torch.finfo(pca_dtype).eps
        centering_tolerance = max(1e-7, 128.0 * dtype_epsilon * scale)
        if centered_max_abs_mean > centering_tolerance:
            raise ValueError(
                f"Layer {layer_index} is not numerically centered: "
                f"max_abs_mean={centered_max_abs_mean} "
                f"tolerance={centering_tolerance}"
            )

        if svd_driver is None:
            u, singular_values, vh = torch.linalg.svd(
                centered,
                full_matrices=False,
            )
        else:
            u, singular_values, vh = torch.linalg.svd(
                centered,
                full_matrices=False,
                driver=svd_driver,
            )
        expected_rank_dimension = min(num_records, HIDDEN_SIZE)
        if tuple(u.shape) != (num_records, expected_rank_dimension):
            raise ValueError(f"Layer {layer_index} SVD U shape is invalid")
        if tuple(singular_values.shape) != (expected_rank_dimension,):
            raise ValueError(f"Layer {layer_index} singular-value shape is invalid")
        if tuple(vh.shape) != (expected_rank_dimension, HIDDEN_SIZE):
            raise ValueError(f"Layer {layer_index} SVD Vh shape is invalid")
        if not bool(torch.isfinite(singular_values).all()) or not bool(
            torch.isfinite(vh).all()
        ):
            raise ValueError(f"Layer {layer_index} PCA produced non-finite values")
        if singular_values.numel() > 1 and bool(
            torch.any(singular_values[1:] > singular_values[:-1])
        ):
            raise ValueError(f"Layer {layer_index} singular values are not descending")
        eigenvalues_all = singular_values.square() / max(num_records - 1, 1)
        if bool(torch.any(eigenvalues_all < 0)):
            raise ValueError(f"Layer {layer_index} PCA produced a negative eigenvalue")
        usable_mask = eigenvalues_all > eigenvalue_tolerance
        usable_eigenvalues = eigenvalues_all[usable_mask]
        usable_components = vh[usable_mask]
        total_variance_tensor = eigenvalues_all.sum()
        total_variance = float(total_variance_tensor.item())
        discarded_numerical_variance = float(
            eigenvalues_all[~usable_mask].sum().item()
        )

        if usable_eigenvalues.numel() == 0 or total_variance <= EPSILON:
            components = torch.empty(
                (0, HIDDEN_SIZE), dtype=torch.float32, device="cpu"
            )
            retained_eigenvalues = torch.empty(0, dtype=torch.float32)
            retained_ratios = torch.empty(0, dtype=torch.float32)
            retained_cumulative = torch.empty(0, dtype=torch.float32)
            num_components = 0
            achieved = 0.0
            previous: float | None = None
            orthonormality_error = 0.0
        else:
            ratios_all = usable_eigenvalues / total_variance_tensor.clamp_min(EPSILON)
            cumulative_all = torch.cumsum(ratios_all, dim=0)
            if float(cumulative_all[-1].item()) < variance_threshold:
                raise ValueError(
                    f"Layer {layer_index} cannot reach variance threshold "
                    f"{variance_threshold} after removing eigenvalues at or below "
                    f"{eigenvalue_tolerance}"
                )
            threshold_tensor = torch.tensor(
                variance_threshold,
                device=cumulative_all.device,
                dtype=cumulative_all.dtype,
            )
            threshold_index = torch.searchsorted(cumulative_all, threshold_tensor)
            num_components = int(threshold_index.item()) + 1
            if num_components > int(usable_eigenvalues.numel()):
                num_components = int(usable_eigenvalues.numel())
            components = usable_components[:num_components].to(
                device="cpu", dtype=torch.float32
            )
            retained_eigenvalues = usable_eigenvalues[:num_components].to(
                device="cpu", dtype=torch.float32
            )
            retained_ratios = ratios_all[:num_components].to(
                device="cpu", dtype=torch.float32
            )
            retained_cumulative = cumulative_all[:num_components].to(
                device="cpu", dtype=torch.float32
            )
            achieved = float(cumulative_all[num_components - 1].item())
            previous = (
                float(cumulative_all[num_components - 2].item())
                if num_components > 1
                else None
            )
            if achieved < variance_threshold:
                raise ValueError(
                    f"Layer {layer_index} retained variance {achieved} is below "
                    f"threshold {variance_threshold}"
                )
            if previous is not None and previous >= variance_threshold:
                raise ValueError(
                    f"Layer {layer_index} did not select the minimum component count"
                )
            orthonormality_error = component_orthonormality_error(components)
            if orthonormality_error > orthonormality_tolerance:
                raise ValueError(
                    f"Layer {layer_index} retained PCA basis is not orthonormal: "
                    f"error={orthonormality_error} "
                    f"tolerance={orthonormality_tolerance}"
                )

        neutral_mean_cpu = layer_mean.squeeze(0).to(
            device="cpu", dtype=torch.float32
        )
        layer_means[layer_index] = neutral_mean_cpu
        pca_payload[layer_index] = {
            "components": components,
            "eigenvalues": retained_eigenvalues,
            "explained_variance_ratio": retained_ratios,
            "cumulative_explained_variance": retained_cumulative,
            "num_components": num_components,
            "neutral_mean": neutral_mean_cpu,
            "total_variance": total_variance,
            "num_neutral_transcripts": num_records,
            "orthonormality_error": orthonormality_error,
            "centered_max_absolute_column_mean": centered_max_abs_mean,
            "numerically_nonzero_eigenvalue_count": int(
                usable_eigenvalues.numel()
            ),
            "discarded_numerical_variance": discarded_numerical_variance,
            "svd_driver": svd_driver or "default",
        }
        summary_layers[str(layer_index)] = {
            "num_components": num_components,
            "achieved_cumulative_variance": achieved,
            "previous_cumulative_variance": previous,
            "total_variance": total_variance,
            "orthonormality_error": orthonormality_error,
            "centered_max_absolute_column_mean": centered_max_abs_mean,
            "centering_tolerance": centering_tolerance,
            "numerically_nonzero_eigenvalue_count": int(
                usable_eigenvalues.numel()
            ),
            "discarded_numerical_variance": discarded_numerical_variance,
        }
        logger.info(
            "pca_layer_complete layer=%d components=%d achieved_variance=%.8f "
            "total_variance=%.8f orthonormality_error=%.3e",
            layer_index,
            num_components,
            achieved,
            total_variance,
            orthonormality_error,
        )
        del (
            matrix,
            layer_mean,
            centered,
            centered_column_mean,
            u,
            singular_values,
            vh,
            eigenvalues_all,
            usable_mask,
            usable_eigenvalues,
            usable_components,
            total_variance_tensor,
        )

    if not bool(torch.isfinite(layer_means).all()):
        raise ValueError("Neutral layer means contain NaN or infinity")
    summary = {
        "variance_threshold": variance_threshold,
        "pca_sample_unit": "transcript mean activation",
        "neutral_pca_sample_unit": NEUTRAL_SAMPLE_UNIT,
        "neutral_token_aggregation": NEUTRAL_TOKEN_AGGREGATION,
        "pca_solver": pca_solver,
        "pca_svd_driver": svd_driver or "default",
        "pca_device": str(pca_device),
        "pca_dtype": str(pca_dtype),
        "eigenvalue_tolerance": eigenvalue_tolerance,
        "component_orthonormality_tolerance": orthonormality_tolerance,
        "num_neutral_transcripts": num_records,
        "layers": summary_layers,
    }
    return pca_payload, summary, layer_means


def component_orthonormality_error(components: torch.Tensor) -> float:
    """Return max|P P^T - I| for retained rows P [K,3584]."""

    if components.shape[0] == 0:
        return 0.0
    gram = components @ components.T
    identity = torch.eye(components.shape[0], dtype=gram.dtype, device=gram.device)
    return float((gram - identity).abs().max().item())


def clean_emotion_vectors(
    *,
    raw_vectors: dict[str, torch.Tensor],
    pca_payload: dict[int, dict[str, Any]],
    projection_tolerance: float,
) -> dict[str, Any]:
    """Compute v_clean=v_raw-P^T(Pv_raw) independently at every layer."""

    projections: dict[str, torch.Tensor] = {}
    clean_vectors: dict[str, torch.Tensor] = {}
    clean_unit_vectors: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, dict[str, float | bool]]] = {}
    near_zero_vectors: list[dict[str, Any]] = []

    for emotion in EMOTIONS:
        raw = raw_vectors[emotion].to(torch.float32)
        projection_layers: list[torch.Tensor] = []
        clean_layers: list[torch.Tensor] = []
        emotion_metrics: dict[str, dict[str, float | bool]] = {}
        for layer_index in range(NUM_LAYERS):
            raw_vector = raw[layer_index]
            components = pca_payload[layer_index]["components"].to(torch.float32)
            if tuple(components.shape[1:]) != (HIDDEN_SIZE,):
                raise ValueError(f"Layer {layer_index} PCA component shape is invalid")
            if components.shape[0] == 0:
                coefficients = torch.empty(0, dtype=torch.float32)
                neutral_projection = torch.zeros_like(raw_vector)
            else:
                coefficients = components @ raw_vector
                neutral_projection = components.T @ coefficients
            clean_vector = raw_vector - neutral_projection
            residual_coefficients = components @ clean_vector
            residual_norm = float(residual_coefficients.norm(p=2).item())
            maximum_absolute_residual = (
                float(residual_coefficients.abs().max().item())
                if residual_coefficients.numel()
                else 0.0
            )
            raw_norm = float(raw_vector.norm(p=2).item())
            removed_norm = float(neutral_projection.norm(p=2).item())
            clean_norm = float(clean_vector.norm(p=2).item())
            relative_residual_raw = residual_norm / max(raw_norm, EPSILON)
            relative_residual_clean = residual_norm / max(clean_norm, EPSILON)
            fraction_squared_removed = (
                removed_norm**2 / (raw_norm**2 + EPSILON)
            )
            fraction_norm_retained = clean_norm / (raw_norm + EPSILON)
            raw_clean_cosine = float(
                torch.dot(raw_vector, clean_vector).item()
                / max(raw_norm * clean_norm, EPSILON)
            )
            reconstruction_error = float(
                (raw_vector - (neutral_projection + clean_vector)).abs().max().item()
            )
            projection_clean_dot = float(
                torch.dot(neutral_projection, clean_vector).item()
            )
            projection_clean_relative_dot = abs(projection_clean_dot) / max(
                removed_norm * clean_norm, EPSILON
            )
            projection_residual_within_tolerance = (
                maximum_absolute_residual <= projection_tolerance
                and relative_residual_raw <= projection_tolerance
            )
            if not projection_residual_within_tolerance:
                raise ValueError(
                    f"Projection residual too large for {emotion}, layer "
                    f"{layer_index}: max={maximum_absolute_residual}, "
                    f"relative={relative_residual_raw}"
                )
            if reconstruction_error > max(1e-6, projection_tolerance * raw_norm):
                raise ValueError(
                    f"Projection reconstruction failed for {emotion}, layer "
                    f"{layer_index}: {reconstruction_error}"
                )
            if not bool(torch.isfinite(clean_vector).all()) or not bool(
                torch.isfinite(neutral_projection).all()
            ):
                raise ValueError(
                    f"Non-finite cleaned vector for {emotion}, layer {layer_index}"
                )
            if clean_norm <= EPSILON:
                near_zero_vectors.append(
                    {"emotion": emotion, "layer_index": layer_index, "norm": clean_norm}
                )
            projection_layers.append(neutral_projection)
            clean_layers.append(clean_vector)
            emotion_metrics[str(layer_index)] = {
                "raw_vector_norm": raw_norm,
                "removed_neutral_component_norm": removed_norm,
                "clean_vector_norm": clean_norm,
                "fraction_squared_norm_removed": fraction_squared_removed,
                "fraction_norm_retained": fraction_norm_retained,
                "raw_clean_cosine_similarity": raw_clean_cosine,
                "residual_projection_norm": residual_norm,
                "maximum_absolute_residual": maximum_absolute_residual,
                "relative_residual_projection_norm_to_raw": relative_residual_raw,
                "relative_residual_projection_norm_to_clean": relative_residual_clean,
                "projection_reconstruction_max_absolute_error": reconstruction_error,
                "projection_clean_dot_product": projection_clean_dot,
                "projection_clean_relative_dot": projection_clean_relative_dot,
                "projection_residual_within_tolerance": (
                    projection_residual_within_tolerance
                ),
            }
        projection_tensor = torch.stack(projection_layers, dim=0).to(torch.float32)
        clean_tensor = torch.stack(clean_layers, dim=0).to(torch.float32)
        norms = clean_tensor.norm(p=2, dim=-1, keepdim=True)
        clean_unit = clean_tensor / norms.clamp_min(EPSILON)
        nonzero = norms.squeeze(-1) > EPSILON
        if bool(torch.any(nonzero)):
            observed = clean_unit.norm(p=2, dim=-1)[nonzero]
            if not torch.allclose(
                observed,
                torch.ones_like(observed),
                rtol=1e-4,
                atol=1e-5,
            ):
                raise ValueError(f"Clean unit-vector norms are invalid for {emotion}")
        for name, tensor in (
            ("neutral projection", projection_tensor),
            ("clean vector", clean_tensor),
            ("clean unit vector", clean_unit),
        ):
            if (
                tensor.dtype != torch.float32
                or tuple(tensor.shape) != (NUM_LAYERS, HIDDEN_SIZE)
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"Invalid {name} tensor for {emotion}")
        projections[emotion] = projection_tensor
        clean_vectors[emotion] = clean_tensor
        clean_unit_vectors[emotion] = clean_unit
        metrics[emotion] = emotion_metrics

    stacked_clean = {
        "emotion_order": list(EMOTIONS),
        "vectors": torch.stack([clean_vectors[e] for e in EMOTIONS], dim=0),
    }
    stacked_unit = {
        "emotion_order": list(EMOTIONS),
        "vectors": torch.stack([clean_unit_vectors[e] for e in EMOTIONS], dim=0),
    }
    expected_shape = (len(EMOTIONS), NUM_LAYERS, HIDDEN_SIZE)
    for payload in (stacked_clean, stacked_unit):
        if (
            payload["vectors"].dtype != torch.float32
            or tuple(payload["vectors"].shape) != expected_shape
            or not bool(torch.isfinite(payload["vectors"]).all())
        ):
            raise ValueError("Invalid stacked clean-vector artifact")
    return {
        "neutral_projections": projections,
        "clean_vectors": clean_vectors,
        "clean_unit_vectors": clean_unit_vectors,
        "stacked_clean": stacked_clean,
        "stacked_unit": stacked_unit,
        "metrics": {
            "projection_orthogonality_tolerance": projection_tolerance,
            "normalization_epsilon": EPSILON,
            "near_zero_clean_layer_vectors": near_zero_vectors,
            "emotions": metrics,
        },
    }


def save_cleaning_outputs(
    *, output_dir: Path, artifacts: EmotionArtifacts, cleaning: dict[str, Any]
) -> None:
    """Save verified raw copy, projections, clean vectors, units, stacks, metrics."""

    raw_copy = load_vector_dictionary(
        output_dir / "emotion_vectors_raw_original.pt", "verified raw-vector copy"
    )
    for emotion in EMOTIONS:
        if not torch.equal(raw_copy[emotion], artifacts.raw_vectors[emotion]):
            raise ValueError(f"Raw-vector copy changed source values for {emotion}")
    atomic_torch_save(
        output_dir / "emotion_vectors_neutral_projection.pt",
        cleaning["neutral_projections"],
    )
    atomic_torch_save(
        output_dir / "emotion_vectors_clean.pt", cleaning["clean_vectors"]
    )
    atomic_torch_save(
        output_dir / "emotion_vectors_clean_unit.pt",
        cleaning["clean_unit_vectors"],
    )
    atomic_torch_save(
        output_dir / "emotion_vectors_clean_stacked.pt",
        cleaning["stacked_clean"],
    )
    atomic_torch_save(
        output_dir / "emotion_vectors_clean_unit_stacked.pt",
        cleaning["stacked_unit"],
    )
    atomic_write_json(
        output_dir / "emotion_vector_cleaning_metrics.json", cleaning["metrics"]
    )


def save_verified_raw_vector_copy(
    *,
    source_vectors: dict[str, torch.Tensor],
    destination: Path,
    resume: bool,
) -> None:
    """Save and verify the immutable raw-vector input before any cleaning work."""

    if destination.exists() and not resume:
        raise FileExistsError(f"Unexpected existing raw-vector copy: {destination}")
    if not destination.exists():
        atomic_torch_save(destination, source_vectors)
    copied = load_vector_dictionary(destination, "verified raw-vector copy")
    for emotion in EMOTIONS:
        if not torch.equal(copied[emotion], source_vectors[emotion]):
            raise ValueError(f"Raw-vector copy mismatch for {emotion}")


def validate_clean_vectors_on_story_activations(
    *,
    story_activations_dir: Path,
    emotion_metadata: dict[str, Any],
    raw_unit_vectors: dict[str, torch.Tensor],
    clean_unit_vectors: dict[str, torch.Tensor],
    batch_size: int,
    compute_device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Compare raw/clean probes on saved [1200,28,3584] story means."""

    raw_probe_stack = torch.stack(
        [raw_unit_vectors[emotion] for emotion in EMOTIONS], dim=0
    )
    clean_probe_stack = torch.stack(
        [clean_unit_vectors[emotion] for emotion in EMOTIONS], dim=0
    )
    probes = torch.stack([raw_probe_stack, clean_probe_stack], dim=0).to(
        device=compute_device, dtype=torch.float32
    )
    # [probe_kind=2, target_emotion=12, layer=28]
    target_sum = torch.zeros((2, len(EMOTIONS), NUM_LAYERS), dtype=torch.float64)
    target_sumsq = torch.zeros_like(target_sum)
    other_sum = torch.zeros_like(target_sum)
    other_sumsq = torch.zeros_like(target_sum)
    target_counts = torch.zeros(len(EMOTIONS), dtype=torch.int64)
    other_counts = torch.zeros(len(EMOTIONS), dtype=torch.int64)

    for source_index, source_emotion in enumerate(EMOTIONS):
        path = story_activations_dir / f"{source_emotion}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_story_payload_for_scoring(
            payload=payload,
            path=path,
            emotion=source_emotion,
            emotion_metadata=emotion_metadata,
        )
        activations = payload["activations"]
        source_count = int(activations.shape[0])
        for batch_start in range(0, source_count, batch_size):
            activation_batch = activations[batch_start : batch_start + batch_size].to(
                device=compute_device, dtype=torch.float32
            )
            # [2, batch, 12 target probes, 28 layers]
            scores = torch.einsum("blh,pelh->pbel", activation_batch, probes)
            score_sum = scores.to(torch.float64).sum(dim=1).cpu()
            score_sumsq = scores.to(torch.float64).square().sum(dim=1).cpu()
            count = int(scores.shape[1])
            for target_index in range(len(EMOTIONS)):
                if target_index == source_index:
                    target_sum[:, target_index] += score_sum[:, target_index]
                    target_sumsq[:, target_index] += score_sumsq[:, target_index]
                    target_counts[target_index] += count
                else:
                    other_sum[:, target_index] += score_sum[:, target_index]
                    other_sumsq[:, target_index] += score_sumsq[:, target_index]
                    other_counts[target_index] += count
            del activation_batch, scores, score_sum, score_sumsq
        logger.info(
            "story_validation_source_complete emotion=%s stories=%d",
            source_emotion,
            source_count,
        )
        del payload, activations

    expected_target = EXPECTED_STORIES_PER_EMOTION
    expected_other = (len(EMOTIONS) - 1) * EXPECTED_STORIES_PER_EMOTION
    if not bool(torch.all(target_counts == expected_target)):
        raise ValueError(f"Unexpected story-validation target counts: {target_counts}")
    if not bool(torch.all(other_counts == expected_other)):
        raise ValueError(f"Unexpected story-validation other counts: {other_counts}")

    result_emotions: dict[str, dict[str, Any]] = {}
    raw_standardized: list[float] = []
    clean_standardized: list[float] = []
    clean_greater_count = 0
    for emotion_index, emotion in enumerate(EMOTIONS):
        layers: dict[str, Any] = {}
        for layer_index in range(NUM_LAYERS):
            layer_payload: dict[str, Any] = {}
            layer_values: list[float] = []
            for probe_index, probe_name in enumerate(("raw", "clean")):
                stats = score_distribution_statistics(
                    target_sum=float(
                        target_sum[probe_index, emotion_index, layer_index].item()
                    ),
                    target_sumsq=float(
                        target_sumsq[probe_index, emotion_index, layer_index].item()
                    ),
                    target_count=int(target_counts[emotion_index].item()),
                    other_sum=float(
                        other_sum[probe_index, emotion_index, layer_index].item()
                    ),
                    other_sumsq=float(
                        other_sumsq[probe_index, emotion_index, layer_index].item()
                    ),
                    other_count=int(other_counts[emotion_index].item()),
                )
                layer_payload[probe_name] = stats
                layer_values.append(float(stats["standardized_difference"]))
                if probe_name == "raw":
                    raw_standardized.append(layer_values[-1])
                else:
                    clean_standardized.append(layer_values[-1])
            if layer_values[1] > layer_values[0]:
                clean_greater_count += 1
            layers[str(layer_index)] = layer_payload
        result_emotions[emotion] = {"layers": layers}

    summary = {
        "mean_raw_standardized_difference": mean_float(raw_standardized),
        "mean_clean_standardized_difference": mean_float(clean_standardized),
        "mean_clean_minus_raw_standardized_difference": mean_float(
            [
                clean - raw
                for raw, clean in zip(
                    raw_standardized, clean_standardized, strict=True
                )
            ]
        ),
        "emotion_layer_pairs_with_larger_clean_standardized_difference": (
            clean_greater_count
        ),
        "total_emotion_layer_pairs": len(EMOTIONS) * NUM_LAYERS,
    }
    return {
        "target_story_count_per_emotion": expected_target,
        "other_story_count_per_emotion": expected_other,
        "standard_deviation_definition": "sample standard deviation (ddof=1)",
        "pooled_standardized_mean_difference_definition": (
            "(target_mean - other_mean) divided by the pooled sample standard "
            "deviation"
        ),
        "compute_device": str(compute_device),
        "summary": summary,
        "emotions": result_emotions,
    }


def validate_story_payload_for_scoring(
    *,
    payload: Any,
    path: Path,
    emotion: str,
    emotion_metadata: dict[str, Any],
) -> None:
    """Validate one saved story payload immediately before score accumulation."""

    if not isinstance(payload, dict):
        raise ValueError(f"Story activation payload is invalid: {path}")
    expected = {
        "emotion": emotion,
        "target_model": TARGET_MODEL,
        "model_revision": emotion_metadata["resolved_model_revision"],
        "start_token_position": EMOTIONAL_STORY_START_POSITION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "num_transformer_layers": NUM_LAYERS,
        "hidden_size": HIDDEN_SIZE,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Story scoring compatibility mismatch: {path}:{field}")
    activations = payload.get("activations")
    if (
        not isinstance(activations, torch.Tensor)
        or activations.dtype != torch.float16
        or tuple(activations.shape)
        != (EXPECTED_STORIES_PER_EMOTION, NUM_LAYERS, HIDDEN_SIZE)
        or not bool(torch.isfinite(activations).all())
    ):
        raise ValueError(f"Story scoring tensor is invalid: {path}")


def score_distribution_statistics(
    *,
    target_sum: float,
    target_sumsq: float,
    target_count: int,
    other_sum: float,
    other_sumsq: float,
    other_count: int,
) -> dict[str, float | int]:
    """Compute means, sample SDs, and pooled standardized mean difference."""

    target_mean = target_sum / target_count
    other_mean = other_sum / other_count
    target_variance = max(
        (target_sumsq - target_sum * target_sum / target_count)
        / max(target_count - 1, 1),
        0.0,
    )
    other_variance = max(
        (other_sumsq - other_sum * other_sum / other_count)
        / max(other_count - 1, 1),
        0.0,
    )
    target_std = math.sqrt(target_variance)
    other_std = math.sqrt(other_variance)
    pooled_variance = (
        (target_count - 1) * target_variance
        + (other_count - 1) * other_variance
    ) / max(target_count + other_count - 2, 1)
    pooled_std = math.sqrt(max(pooled_variance, 0.0))
    difference = target_mean - other_mean
    standardized = difference / max(pooled_std, EPSILON)
    return {
        "target_mean": target_mean,
        "other_mean": other_mean,
        "difference": difference,
        "target_standard_deviation": target_std,
        "other_standard_deviation": other_std,
        "pooled_standard_deviation": pooled_std,
        "standardized_difference": standardized,
        "target_story_count": target_count,
        "other_story_count": other_count,
    }


def export_tokenwise_story_validation(
    *,
    story_records_dir: Path,
    story_activations_dir: Path,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    model_input_device: torch.device,
    raw_unit_vectors: dict[str, torch.Tensor],
    clean_unit_vectors: dict[str, torch.Tensor],
    stories_per_emotion: int,
    selection_seed: int,
    validation_layer: int,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, int]:
    """Export unmerged token fragments and raw/clean layer-18 probe scores."""

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for emotion in EMOTIONS:
        activation_payload = torch.load(
            story_activations_dir / f"{emotion}.pt",
            map_location="cpu",
            weights_only=True,
        )
        eligible_ids = set(str(value) for value in activation_payload["record_ids"])
        del activation_payload
        story_records = load_original_story_records(
            story_records_dir / f"{emotion}.jsonl", emotion=emotion
        )
        eligible_records = [
            record for record in story_records if str(record["record_id"]) in eligible_ids
        ]
        if len(eligible_records) != len(eligible_ids):
            raise ValueError(
                f"Original story join is incomplete for {emotion}: "
                f"{len(eligible_records)} != {len(eligible_ids)}"
            )
        selected = deterministic_story_sample(
            records=eligible_records,
            emotion=emotion,
            sample_count=stories_per_emotion,
            seed=selection_seed,
        )
        destination = output_dir / f"{emotion}.jsonl"
        selected_ids = [str(record["record_id"]) for record in selected]
        if destination.exists() and resume:
            validate_existing_tokenwise_file(
                path=destination,
                emotion=emotion,
                expected_records=selected,
                validation_layer=validation_layer,
            )
            counts[emotion] = len(selected_ids)
            logger.info(
                "resume_tokenwise_complete emotion=%s records=%d",
                emotion,
                len(selected_ids),
            )
            continue
        if destination.exists():
            raise FileExistsError(f"Unexpected tokenwise output: {destination}")

        raw_probe = raw_unit_vectors[emotion][validation_layer].to(
            model_input_device, dtype=torch.float32
        )
        clean_probe = clean_unit_vectors[emotion][validation_layer].to(
            model_input_device, dtype=torch.float32
        )
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(selected, start=1):
            story = str(record["story"])
            encoded, _, processed_counts = tokenize_plain_text_batch(
                tokenizer=tokenizer,
                texts=[story],
                max_sequence_length=None,
            )
            input_ids = encoded["input_ids"].to(model_input_device)
            attention_mask = encoded["attention_mask"].to(model_input_device)
            try:
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states
                    if hidden_states is None or len(hidden_states) != NUM_LAYERS + 1:
                        raise ValueError("Tokenwise hidden-state tuple length is not 29")
                    hidden = hidden_states[validation_layer + 1][0]
                    valid_mask = attention_mask[0].bool().to(hidden.device)
                    valid_hidden = hidden[valid_mask].float()
                    valid_ids = input_ids[0][attention_mask[0].bool()]
                    raw_scores = valid_hidden @ raw_probe.to(valid_hidden.device)
                    clean_scores = valid_hidden @ clean_probe.to(valid_hidden.device)
            except RuntimeError as error:
                if is_out_of_memory(error):
                    logger.error(
                        "tokenwise_out_of_memory emotion=%s record_id=%s "
                        "sequence_length=%d batch_size=1",
                        emotion,
                        record["record_id"],
                        processed_counts[0],
                    )
                    raise RuntimeError(
                        "GPU out of memory during tokenwise validation for "
                        f"{record['record_id']}; completed emotion files remain "
                        "resumable."
                    ) from error
                raise
            token_ids = [int(value) for value in valid_ids.detach().cpu().tolist()]
            token_strings = tokenizer.convert_ids_to_tokens(token_ids)
            if not isinstance(token_strings, list) or len(token_strings) != len(token_ids):
                raise ValueError("Tokenizer fragments are misaligned in tokenwise export")
            raw_values = [float(value) for value in raw_scores.detach().cpu().tolist()]
            clean_values = [
                float(value) for value in clean_scores.detach().cpu().tolist()
            ]
            if not (
                len(token_ids)
                == len(token_strings)
                == len(raw_values)
                == len(clean_values)
            ):
                raise ValueError("Tokenwise score arrays are not aligned")
            if not all(math.isfinite(value) for value in raw_values + clean_values):
                raise ValueError("Tokenwise validation produced non-finite scores")
            rows.append(
                {
                    "record_id": str(record["record_id"]),
                    "emotion": emotion,
                    "layer_index_zero_based": validation_layer,
                    "layer_number_one_based": validation_layer + 1,
                    "token_ids": token_ids,
                    "token_strings": token_strings,
                    "raw_probe_scores": raw_values,
                    "clean_probe_scores": clean_values,
                    "story_text": story,
                }
            )
            logger.info(
                "tokenwise_story_complete emotion=%s record=%d/%d record_id=%s "
                "tokens=%d",
                emotion,
                index,
                len(selected),
                record["record_id"],
                len(token_ids),
            )
            del (
                outputs,
                hidden_states,
                hidden,
                valid_hidden,
                valid_ids,
                raw_scores,
                clean_scores,
                input_ids,
                attention_mask,
            )
        atomic_write_jsonl(destination, rows)
        validate_existing_tokenwise_file(
            path=destination,
            emotion=emotion,
            expected_records=selected,
            validation_layer=validation_layer,
        )
        counts[emotion] = len(rows)
    return counts


def load_original_story_records(path: Path, *, emotion: str) -> list[dict[str, Any]]:
    """Load original accepted stories needed only for deterministic token export."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid story JSON {path}:{line_number}") from error
            if (
                not isinstance(row, dict)
                or row.get("emotion") != emotion
                or "record_id" not in row
                or not isinstance(row.get("story"), str)
                or not row["story"].strip()
            ):
                raise ValueError(f"Invalid emotional story {path}:{line_number}")
            record_id = str(row["record_id"])
            if record_id in seen:
                raise ValueError(f"Duplicate emotional story ID: {record_id}")
            seen.add(record_id)
            records.append(row)
    return records


def deterministic_story_sample(
    *, records: Sequence[dict[str, Any]], emotion: str, sample_count: int, seed: int
) -> list[dict[str, Any]]:
    """Select stable story IDs with SHA-256-derived per-emotion RNG state."""

    ordered = sorted(records, key=lambda record: str(record["record_id"]))
    if len(ordered) < sample_count:
        raise ValueError(
            f"Not enough eligible tokenwise stories for {emotion}: "
            f"{len(ordered)} < {sample_count}"
        )
    material = f"{seed}|tokenwise-validation|{emotion}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    random.Random(derived_seed).shuffle(ordered)
    return ordered[:sample_count]


def validate_existing_tokenwise_file(
    *,
    path: Path,
    emotion: str,
    expected_records: Sequence[dict[str, Any]],
    validation_layer: int,
) -> None:
    """Validate a complete per-emotion tokenwise JSONL export for resume."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid tokenwise JSON {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Invalid tokenwise row {path}:{line_number}")
            rows.append(row)
    expected_record_ids = [str(record["record_id"]) for record in expected_records]
    expected_story_text = {
        str(record["record_id"]): str(record["story"])
        for record in expected_records
    }
    if [str(row.get("record_id")) for row in rows] != expected_record_ids:
        raise ValueError(f"Tokenwise record IDs do not match deterministic sample: {path}")
    for row in rows:
        if row.get("emotion") != emotion:
            raise ValueError(f"Tokenwise emotion mismatch: {path}")
        if row.get("layer_index_zero_based") != validation_layer:
            raise ValueError(f"Tokenwise zero-based layer mismatch: {path}")
        if row.get("layer_number_one_based") != validation_layer + 1:
            raise ValueError(f"Tokenwise one-based layer mismatch: {path}")
        record_id = str(row.get("record_id"))
        if row.get("story_text") != expected_story_text[record_id]:
            raise ValueError(f"Tokenwise story text mismatch: {path}:{record_id}")
        arrays = (
            row.get("token_ids"),
            row.get("token_strings"),
            row.get("raw_probe_scores"),
            row.get("clean_probe_scores"),
        )
        if not all(isinstance(value, list) for value in arrays):
            raise ValueError(f"Tokenwise arrays are invalid: {path}")
        if len({len(value) for value in arrays}) != 1 or len(arrays[0]) == 0:
            raise ValueError(f"Tokenwise arrays are misaligned: {path}")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in arrays[0]):
            raise ValueError(f"Tokenwise token IDs are invalid: {path}")
        if not all(isinstance(value, str) for value in arrays[1]):
            raise ValueError(f"Tokenwise token strings are invalid: {path}")
        for scores in arrays[2:]:
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in scores
            ):
                raise ValueError(f"Tokenwise probe scores are invalid: {path}")


def finalize_metadata(
    *,
    metadata: dict[str, Any],
    pca_summary: dict[str, Any],
    neutral_activations: torch.Tensor,
    loaded: LoadedNeutralRecords,
    selected_records: Sequence[NeutralRecord],
    truncated_neutral_records: int,
    cleaning: dict[str, Any],
    story_validation: dict[str, Any],
    tokenwise_counts: dict[str, int],
    validation_layer: int,
    pca_device: torch.device,
    pca_dtype: torch.dtype,
    end_timestamp: str,
) -> dict[str, Any]:
    """Attach actual shapes, PCA results, diagnostics, and completion timestamps."""

    component_counts = {
        layer: values["num_components"]
        for layer, values in pca_summary["layers"].items()
    }
    achieved_variance = {
        layer: values["achieved_cumulative_variance"]
        for layer, values in pca_summary["layers"].items()
    }
    orthonormality_errors = {
        layer: values["orthonormality_error"]
        for layer, values in pca_summary["layers"].items()
    }
    maximum_projection_residual = max(
        float(layer["maximum_absolute_residual"])
        for emotion in EMOTIONS
        for layer in cleaning["metrics"]["emotions"][emotion].values()
    )
    maximum_relative_projection_residual = max(
        float(layer["relative_residual_projection_norm_to_raw"])
        for emotion in EMOTIONS
        for layer in cleaning["metrics"]["emotions"][emotion].values()
    )
    metadata.update(
        {
            "status": "completed",
            "total_neutral_records": loaded.total_lines,
            "valid_neutral_records": len(loaded.records),
            "invalid_neutral_records": len(loaded.invalid_records),
            "selected_neutral_records": len(selected_records),
            "truncated_neutral_records": truncated_neutral_records,
            "neutral_activation_tensor_shape": list(neutral_activations.shape),
            "neutral_activation_dtype": str(neutral_activations.dtype),
            "pca_solver": pca_summary["pca_solver"],
            "pca_svd_driver": pca_summary["pca_svd_driver"],
            "actual_pca_device": str(pca_device),
            "actual_pca_dtype": str(pca_dtype),
            "component_count_per_layer": component_counts,
            "achieved_cumulative_variance_per_layer": achieved_variance,
            "component_orthonormality_error_per_layer": orthonormality_errors,
            "raw_emotion_vector_shape": [
                len(EMOTIONS),
                NUM_LAYERS,
                HIDDEN_SIZE,
            ],
            "clean_emotion_vector_shape": [NUM_LAYERS, HIDDEN_SIZE],
            "stacked_clean_emotion_vector_shape": [
                len(EMOTIONS),
                NUM_LAYERS,
                HIDDEN_SIZE,
            ],
            "maximum_absolute_projection_residual": maximum_projection_residual,
            "maximum_relative_projection_residual_to_raw": (
                maximum_relative_projection_residual
            ),
            "near_zero_clean_layer_vectors": cleaning["metrics"][
                "near_zero_clean_layer_vectors"
            ],
            "story_validation_summary": story_validation["summary"],
            "tokenwise_validation_layer_index_zero_based": validation_layer,
            "tokenwise_validation_layer_number_one_based": validation_layer + 1,
            "tokenwise_records_per_emotion": tokenwise_counts,
            "cleaning_end_timestamp": end_timestamp,
        }
    )
    metadata.pop("failure", None)
    return metadata


def resolve_tokenwise_layer(explicit_layer: int | None) -> int:
    """Resolve zero-based layer 18 (one-based transformer layer 19) by default."""

    layer = (
        round((NUM_LAYERS - 1) * 2 / 3)
        if explicit_layer is None
        else explicit_layer
    )
    if not 0 <= layer < NUM_LAYERS:
        raise ValueError(
            f"Tokenwise validation layer must be in [0,{NUM_LAYERS - 1}]"
        )
    return layer


def update_progress(
    output_dir: Path,
    *,
    status: str,
    stage: str,
    **values: Any,
) -> None:
    """Atomically update resumable stage progress without discarding prior fields."""

    path = output_dir / "progress.json"
    progress = read_json_object(path) if path.exists() else {}
    progress.update(values)
    progress["status"] = status
    progress["stage"] = stage
    if status in {"running", "completed"}:
        progress.pop("failure", None)
    progress["updated_at"] = utc_now()
    atomic_write_json(path, progress)


def mark_run_status(
    output_dir: Path, status: str, failure: dict[str, Any]
) -> None:
    """Best-effort status persistence for interruption or a normal runtime error."""

    if not output_dir.exists() or not output_dir.is_dir():
        return
    try:
        update_progress(
            output_dir,
            status=status,
            stage=status,
            failure=failure,
        )
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            metadata = read_json_object(metadata_path)
            metadata["status"] = status
            metadata["failure"] = failure
            metadata["cleaning_end_timestamp"] = utc_now()
            atomic_write_json(metadata_path, metadata)
    except (OSError, ValueError):
        pass


def configure_logging(path: Path) -> logging.Logger:
    """Configure repository-style file and stderr logging."""

    logger = logging.getLogger("neutral_pca_cleaning")
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


def json_safe_cli_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Convert Paths and sequences in argparse state to JSON-safe values."""

    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value.expanduser().resolve())
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def safe_error_message(error: BaseException) -> str:
    """Bound error text and redact token-like Hugging Face credentials."""

    message = str(error) or error.__class__.__name__
    import re

    message = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_HF_TOKEN]", message)
    message = re.sub(
        r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        message,
    )
    return message[:4000]


def mean_float(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a required nonempty sequence."""

    if not values:
        raise ValueError("Cannot compute the mean of an empty sequence")
    return math.fsum(values) / len(values)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def _variance_threshold(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("value must be in the interval (0,1]")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
