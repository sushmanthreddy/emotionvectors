#!/usr/bin/env python3
"""Extract neutral transcript activations and fit one PCA basis per layer.

This module intentionally stops after neutral activation extraction and PCA. It
does not read emotional stories or emotion vectors and does not project any PCA
directions out of another representation.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .story_raw_vectors import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    build_length_aware_batches,
    canonical_json_sha256,
    is_out_of_memory,
    load_model_and_tokenizer,
    mean_transformer_hidden_states,
    read_json_object,
    restore_canonical_order,
    sha256_file,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
INPUT_TEXT_FIELD = "transcript"
TOKEN_POSITION_INDEXING = "one-based"
HIDDEN_STATE_MAPPING = "saved layer l equals outputs.hidden_states[l + 1]"
TOKEN_AGGREGATION = "mean across all valid non-padding tokens"
PCA_SAMPLE_UNIT = "one token-averaged activation per transcript"
PCA_METHOD = "torch.linalg.svd(full_matrices=False), independently per layer"
SCHEMA_VERSION = 1
EPSILON = 1e-12


@dataclass(frozen=True)
class NeutralRecord:
    """One accepted record in canonical JSONL order."""

    source_line: int
    record_id: str
    transcript: str
    topic_id: Any
    sample_index: Any
    dialogue_type: Any


@dataclass(frozen=True)
class LoadedNeutralRecords:
    """Valid records plus invalid-line diagnostics."""

    records: tuple[NeutralRecord, ...]
    invalid_records: tuple[dict[str, Any], ...]
    total_lines: int


@dataclass(frozen=True)
class ShardPlan:
    """Canonical fixed-boundary activation shard."""

    index: int
    records: tuple[NeutralRecord, ...]

    @property
    def path_name(self) -> str:
        return f"shard_{self.index:05d}.pt"


class IncompatibleCheckpointError(RuntimeError):
    """A published artifact belongs to a different immutable run."""


class CorruptCheckpointError(RuntimeError):
    """A run-compatible artifact cannot be trusted and must be recomputed."""


def build_parser() -> argparse.ArgumentParser:
    """Build the neutral extraction and layer-wise PCA CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract token-averaged neutral transcript activations and run "
            "independent layer-wise PCA. No emotional data is read."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "extract", "pca"),
        default="all",
        help=(
            "Run both stages, activation extraction only, or PCA only. "
            "The pca stage reads neutral_activations.pt and never loads a model."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--neutral-jsonl", type=Path, required=True)
    parser.add_argument("--neutral-manifest", type=Path, required=True)
    parser.add_argument("--neutral-text-field", default=INPUT_TEXT_FIELD)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--neutral-start-token-position", type=_positive_int, default=1
    )
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--records-per-shard", type=_positive_int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "bf16"), default="bfloat16"
    )
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--pca-device", default="cpu")
    parser.add_argument(
        "--pca-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--explained-variance-threshold",
        type=_variance_threshold,
        default=0.50,
    )
    parser.add_argument("--expected-records", type=_positive_int, default=1200)
    parser.add_argument("--expected-layers", type=_positive_int, default=48)
    parser.add_argument(
        "--expected-hidden-size", type=_positive_int, default=5120
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested stage and return a shell-friendly exit code."""

    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        run_pipeline(args)
        return 0
    except KeyboardInterrupt:
        _mark_failed(output_dir, "interrupted", "KeyboardInterrupt")
        print(
            "Neutral activation/PCA run interrupted; compatible activation "
            "shards remain resumable.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        out_of_memory = is_out_of_memory(error)
        _mark_failed(
            output_dir,
            "out_of_memory" if out_of_memory else "failed",
            f"{error.__class__.__name__}: {str(error).strip()}",
        )
        print(f"Neutral activation/PCA run failed: {error}", file=sys.stderr)
        return 2 if out_of_memory else 1


def run_pipeline(args: argparse.Namespace) -> None:
    """Validate input, run the selected stages, and persist final metadata."""

    validate_cli_arguments(args)
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_directory(output_dir, resume=args.resume)
    invocation_started_at = utc_now()

    neutral_jsonl = args.neutral_jsonl.expanduser().resolve()
    neutral_manifest_path = args.neutral_manifest.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    loaded = load_neutral_records(
        neutral_jsonl,
        text_field=args.neutral_text_field,
    )
    if len(loaded.records) != args.expected_records:
        raise ValueError(
            f"Expected exactly {args.expected_records} valid neutral records; "
            f"found {len(loaded.records)} valid and "
            f"{len(loaded.invalid_records)} invalid"
        )
    if loaded.total_lines != args.expected_records:
        raise ValueError(
            f"Expected exactly {args.expected_records} JSONL lines; "
            f"found {loaded.total_lines}"
        )
    validate_unique_record_ids(loaded.records)
    manifest = validate_neutral_manifest(
        neutral_manifest_path,
        model_name=args.model,
        model_revision=args.model_revision,
        expected_records=args.expected_records,
    )

    dataset_identity = build_dataset_identity(
        args=args,
        records=loaded.records,
        neutral_jsonl=neutral_jsonl,
        neutral_manifest=neutral_manifest_path,
    )
    config = prepare_run_config(
        args=args,
        output_dir=output_dir,
        identity=dataset_identity,
        neutral_jsonl=neutral_jsonl,
        neutral_manifest=neutral_manifest_path,
        cache_dir=cache_dir,
        valid_records=len(loaded.records),
        invalid_records=len(loaded.invalid_records),
    )
    logger = configure_logging(output_dir / "neutral_pca.log")
    atomic_write_jsonl(
        output_dir / "skipped_invalid_neutral_records.jsonl",
        loaded.invalid_records,
    )
    logger.info(
        "input_ready valid=%d invalid=%d fingerprint=%s",
        len(loaded.records),
        len(loaded.invalid_records),
        dataset_identity["fingerprint_sha256"],
    )
    update_progress(
        output_dir,
        status="running",
        stage=args.stage,
        valid_records=len(loaded.records),
        invalid_records=len(loaded.invalid_records),
    )
    write_metadata(
        output_dir=output_dir,
        args=args,
        config=config,
        manifest=manifest,
        status="running",
        invocation_started_at=invocation_started_at,
        valid_records=len(loaded.records),
        invalid_records=len(loaded.invalid_records),
    )

    consolidated: dict[str, Any] | None = None
    extraction_runtime: dict[str, Any] = {}
    if args.stage in {"all", "extract"}:
        consolidated, extraction_runtime = extract_neutral_activations(
            records=loaded.records,
            args=args,
            output_dir=output_dir,
            identity=dataset_identity,
            logger=logger,
        )
        update_progress(
            output_dir,
            status="running" if args.stage == "all" else "completed",
            stage="activations_completed",
            completed_records=len(loaded.records),
            completed_shards=len(
                build_shard_plans(loaded.records, args.records_per_shard)
            ),
        )
        write_metadata(
            output_dir=output_dir,
            args=args,
            config=config,
            manifest=manifest,
            status=(
                "activations_completed"
                if args.stage == "extract"
                else "running"
            ),
            invocation_started_at=invocation_started_at,
            valid_records=len(loaded.records),
            invalid_records=len(loaded.invalid_records),
            extraction_runtime=extraction_runtime,
            neutral_activation_shape=list(
                consolidated["activations"].shape
            ),
        )

    if args.stage in {"all", "pca"}:
        if consolidated is None:
            consolidated = load_and_validate_consolidated(
                output_dir / "neutral_activations.pt",
                records=loaded.records,
                identity=dataset_identity,
                expected_layers=args.expected_layers,
                expected_hidden_size=args.expected_hidden_size,
                expected_dtype=activation_dtype(args.activation_dtype),
                mmap=True,
            )
        pca_summary = run_layerwise_pca(
            consolidated=consolidated,
            args=args,
            output_dir=output_dir,
            logger=logger,
        )
        update_progress(
            output_dir,
            status="completed",
            stage="pca_completed",
            completed_records=len(loaded.records),
            completed_pca_layers=args.expected_layers,
        )
        write_metadata(
            output_dir=output_dir,
            args=args,
            config=config,
            manifest=manifest,
            status="completed",
            invocation_started_at=invocation_started_at,
            valid_records=len(loaded.records),
            invalid_records=len(loaded.invalid_records),
            extraction_runtime=extraction_runtime,
            neutral_activation_shape=list(
                consolidated["activations"].shape
            ),
            pca_summary=pca_summary,
        )
        logger.info(
            "run_complete records=%d pca_layers=%d",
            len(loaded.records),
            args.expected_layers,
        )
    elif args.stage == "extract":
        logger.info(
            "extraction_complete records=%d; model memory released",
            len(loaded.records),
        )


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Reject settings that could violate the requested scientific contract."""

    if args.neutral_text_field != INPUT_TEXT_FIELD:
        raise ValueError(
            "--neutral-text-field must be 'transcript'; generation prompts and "
            "alternate dialogue fields are forbidden"
        )
    if args.neutral_start_token_position != 1:
        raise ValueError(
            "--neutral-start-token-position must be 1 to average all valid tokens"
        )
    if not args.local_files_only:
        raise ValueError("--local-files-only is required; downloads are forbidden")
    if not _immutable_revision(args.model_revision):
        raise ValueError(
            "--model-revision must be an exact 40-character hexadecimal commit"
        )
    if args.dtype not in {"bfloat16", "bf16"}:
        raise ValueError("The model must be loaded in bfloat16")
    if args.expected_records < 2:
        raise ValueError("At least two neutral transcripts are required for PCA")
    if args.stage == "pca" and not args.resume:
        raise ValueError(
            "PCA-only execution must use --resume with an existing compatible run"
        )


def load_neutral_records(
    path: Path,
    *,
    text_field: str = INPUT_TEXT_FIELD,
) -> LoadedNeutralRecords:
    """Load valid neutral records without inspecting generation attempts."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Neutral JSONL does not exist: {source}")
    records: list[NeutralRecord] = []
    invalid: list[dict[str, Any]] = []
    total_lines = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_lines += 1
            text = raw_line.rstrip("\n")
            try:
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError("line is not a JSON object")
                record_id = payload.get("record_id")
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError("record_id is missing or empty")
                if payload.get("label") != "neutral":
                    raise ValueError("label is not 'neutral'")
                transcript = payload.get(text_field)
                if not isinstance(transcript, str) or not transcript.strip():
                    raise ValueError(f"{text_field} is missing or empty")
            except (json.JSONDecodeError, ValueError) as error:
                invalid.append(
                    {
                        "source_line": line_number,
                        "error": str(error),
                        "raw_line": text,
                    }
                )
                continue
            records.append(
                NeutralRecord(
                    source_line=line_number,
                    record_id=record_id,
                    transcript=transcript,
                    topic_id=payload.get("topic_id"),
                    sample_index=payload.get("sample_index"),
                    dialogue_type=payload.get("dialogue_type"),
                )
            )
    return LoadedNeutralRecords(
        records=tuple(records),
        invalid_records=tuple(invalid),
        total_lines=total_lines,
    )


def validate_unique_record_ids(records: Sequence[NeutralRecord]) -> None:
    """Require one stable identity per transcript."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        if record.record_id in seen:
            duplicates.append(record.record_id)
        seen.add(record.record_id)
    if duplicates:
        raise ValueError(
            f"Neutral record IDs are not unique: {duplicates[:5]!r}"
        )


def validate_neutral_manifest(
    path: Path,
    *,
    model_name: str,
    model_revision: str,
    expected_records: int,
) -> dict[str, Any]:
    """Validate completed accepted-data provenance without reading attempts."""

    manifest = read_json_object(path.expanduser().resolve())
    expected_values = {
        "status": "completed",
        "label": "neutral",
        "generator_model": model_name,
        "model_revision": model_revision,
        "intended_dialogues": expected_records,
        "accepted_dialogues": expected_records,
        "failed_dialogues": 0,
    }
    mismatches = {
        field: {"observed": manifest.get(field), "expected": expected}
        for field, expected in expected_values.items()
        if manifest.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"Neutral manifest is incompatible: {mismatches!r}")
    topics = manifest.get("number_of_topics")
    samples = manifest.get("samples_per_topic")
    if (
        topics != 100
        or samples != 12
        or topics * samples != expected_records
    ):
        raise ValueError(
            "Neutral manifest must describe exactly 100 topics with 12 "
            f"transcripts each; observed topics={topics!r}, samples={samples!r}"
        )
    return manifest


def build_dataset_identity(
    *,
    args: argparse.Namespace,
    records: Sequence[NeutralRecord],
    neutral_jsonl: Path,
    neutral_manifest: Path,
) -> dict[str, Any]:
    """Build the immutable identity used by shards and consolidated output."""

    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "neutral_jsonl_sha256": sha256_file(neutral_jsonl),
        "neutral_manifest_sha256": sha256_file(neutral_manifest),
        "ordered_record_ids_sha256": canonical_json_sha256(
            [record.record_id for record in records]
        ),
        "model_name": args.model,
        "model_revision": args.model_revision,
        "input_text_field": args.neutral_text_field,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "padding_side": "right",
        "neutral_start_token_position": args.neutral_start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "embedding_hidden_state_included": False,
        "number_of_records": len(records),
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "activation_dtype": args.activation_dtype,
        "records_per_shard": args.records_per_shard,
    }
    identity["fingerprint_sha256"] = canonical_json_sha256(identity)
    return identity


def prepare_output_directory(path: Path, *, resume: bool) -> None:
    """Create a new run root or require explicit resume for existing content."""

    destination = path.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Output path is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Output directory is not empty; use --resume only for the same "
                f"immutable run: {destination}"
            )
        if not (destination / "config.json").is_file():
            raise IncompatibleCheckpointError(
                "A nonempty resume directory must already contain config.json"
            )
    destination.mkdir(parents=True, exist_ok=True)


def prepare_run_config(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    identity: Mapping[str, Any],
    neutral_jsonl: Path,
    neutral_manifest: Path,
    cache_dir: Path,
    valid_records: int,
    invalid_records: int,
) -> dict[str, Any]:
    """Create or validate immutable run configuration."""

    config_path = output_dir / "config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"Existing config requires --resume: {config_path}")
        existing = read_json_object(config_path)
        if existing.get("run_identity") != dict(identity):
            raise IncompatibleCheckpointError(
                "Existing config.json has a different dataset, model, "
                "tokenisation, shape, shard, or PCA identity"
            )
        return existing
    if args.resume and args.stage == "pca":
        raise FileNotFoundError(
            f"PCA resume requires an existing config.json: {config_path}"
        )
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "neutral_jsonl": str(neutral_jsonl),
        "neutral_manifest": str(neutral_manifest),
        "model": args.model,
        "model_revision": args.model_revision,
        "cache_dir": str(cache_dir),
        "local_files_only": True,
        "neutral_input_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "tokenisation": {
            "padding_side": "right",
            "add_special_tokens": True,
            "truncation": False,
        },
        "neutral_start_token_position": 1,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "neutral_token_aggregation": TOKEN_AGGREGATION,
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "embedding_hidden_state_included": False,
        "model_dtype": "bfloat16",
        "token_averaging_dtype": "float32",
        "neutral_activation_storage_dtype": args.activation_dtype,
        "records_per_shard": args.records_per_shard,
        "pca_dtype": args.pca_dtype,
        "pca_device": args.pca_device,
        "pca_method": PCA_METHOD,
        "explained_variance_threshold": args.explained_variance_threshold,
        "total_input_records": valid_records + invalid_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "output_directory": str(output_dir),
        "initial_cli_arguments": json_safe_cli_arguments(args),
        "run_identity": dict(identity),
    }
    atomic_write_json(config_path, config)
    return config


def build_shard_plans(
    records: Sequence[NeutralRecord],
    records_per_shard: int,
) -> tuple[ShardPlan, ...]:
    """Split canonical JSONL order into fixed activation-shard boundaries."""

    plans: list[ShardPlan] = []
    for start in range(0, len(records), records_per_shard):
        plans.append(
            ShardPlan(
                index=len(plans),
                records=tuple(records[start : start + records_per_shard]),
            )
        )
    return tuple(plans)


def inspect_activation_shards(
    *,
    shard_dir: Path,
    plans: Sequence[ShardPlan],
    identity: Mapping[str, Any],
    expected_layers: int,
    expected_hidden_size: int,
    expected_dtype: torch.dtype,
    logger: logging.Logger,
) -> tuple[dict[int, dict[str, Any]], tuple[ShardPlan, ...]]:
    """Validate completed shards and identify missing/corrupt plans."""

    shard_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {plan.path_name for plan in plans}
    observed_names = {
        path.name for path in shard_dir.glob("shard_*.pt") if path.is_file()
    }
    unexpected = sorted(observed_names - expected_names)
    if unexpected:
        raise IncompatibleCheckpointError(
            f"Unexpected published activation shards: {unexpected!r}"
        )
    valid: dict[int, dict[str, Any]] = {}
    missing_or_corrupt: list[ShardPlan] = []
    for plan in plans:
        path = shard_dir / plan.path_name
        if not path.exists():
            missing_or_corrupt.append(plan)
            continue
        try:
            payload = torch_load(path, mmap=True)
            validate_activation_shard(
                payload,
                plan=plan,
                identity=identity,
                expected_layers=expected_layers,
                expected_hidden_size=expected_hidden_size,
                expected_dtype=expected_dtype,
            )
        except IncompatibleCheckpointError:
            raise
        except Exception as error:
            logger.warning(
                "recompute_corrupt_shard path=%s error=%s",
                path,
                error,
            )
            missing_or_corrupt.append(plan)
            continue
        valid[plan.index] = payload
    return valid, tuple(missing_or_corrupt)


def validate_activation_shard(
    payload: Any,
    *,
    plan: ShardPlan,
    identity: Mapping[str, Any],
    expected_layers: int,
    expected_hidden_size: int,
    expected_dtype: torch.dtype,
) -> None:
    """Validate one shard with activations shaped ``[records,layers,hidden]``."""

    if not isinstance(payload, dict):
        raise CorruptCheckpointError("activation shard is not a dictionary")
    global_identity = {
        "schema_version": SCHEMA_VERSION,
        "model_name": identity["model_name"],
        "model_revision": identity["model_revision"],
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "padding_side": "right",
        "neutral_start_token_position": 1,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "neutral_token_aggregation": TOKEN_AGGREGATION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "embedding_hidden_state_included": False,
        "dataset_fingerprint": identity["fingerprint_sha256"],
        "number_of_layers": expected_layers,
        "hidden_size": expected_hidden_size,
        "activation_dtype": dtype_name(expected_dtype),
    }
    global_mismatches = {
        field: {"observed": payload.get(field), "expected": expected}
        for field, expected in global_identity.items()
        if payload.get(field) != expected
    }
    if global_mismatches:
        raise IncompatibleCheckpointError(
            f"Shard {plan.index} global identity mismatch: "
            f"{global_mismatches!r}"
        )
    local_identity = {
        "shard_index": plan.index,
        "record_ids": [record.record_id for record in plan.records],
        "topic_ids": [record.topic_id for record in plan.records],
        "sample_indices": [record.sample_index for record in plan.records],
        "dialogue_types": [record.dialogue_type for record in plan.records],
        "source_lines": [record.source_line for record in plan.records],
    }
    mismatches = {
        field: {"observed": payload.get(field), "expected": expected}
        for field, expected in local_identity.items()
        if payload.get(field) != expected
    }
    if mismatches:
        raise CorruptCheckpointError(
            f"Shard {plan.index} local metadata mismatch: {mismatches!r}"
        )
    token_counts = payload.get("token_counts")
    if (
        not isinstance(token_counts, list)
        or len(token_counts) != len(plan.records)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in token_counts
        )
    ):
        raise CorruptCheckpointError("invalid shard token_counts")
    activations = payload.get("activations")
    expected_shape = (
        len(plan.records),
        expected_layers,
        expected_hidden_size,
    )
    if not isinstance(activations, torch.Tensor):
        raise CorruptCheckpointError("shard activations are not a tensor")
    if tuple(activations.shape) != expected_shape:
        raise CorruptCheckpointError(
            f"shard activation shape {tuple(activations.shape)} != "
            f"{expected_shape}"
        )
    if activations.dtype != expected_dtype:
        raise CorruptCheckpointError(
            f"shard activation dtype {activations.dtype} != {expected_dtype}"
        )
    if not bool(torch.isfinite(activations).all()):
        raise CorruptCheckpointError("shard activations contain nonfinite values")


def extract_neutral_activations(
    *,
    records: Sequence[NeutralRecord],
    args: argparse.Namespace,
    output_dir: Path,
    identity: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract/resume activations and consolidate to ``[N,L,D]`` on CPU."""

    shard_dir = output_dir / "neutral_activation_shards"
    plans = build_shard_plans(records, args.records_per_shard)
    storage_dtype = activation_dtype(args.activation_dtype)
    valid, pending = inspect_activation_shards(
        shard_dir=shard_dir,
        plans=plans,
        identity=identity,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        expected_dtype=storage_dtype,
        logger=logger,
    )
    logger.info(
        "shard_resume valid=%d pending=%d total=%d",
        len(valid),
        len(pending),
        len(plans),
    )
    model_bundle: dict[str, Any] | None = None
    token_count_by_id: dict[str, int] = {}
    runtime: dict[str, Any] = {
        "model_weights_loaded": False,
        "resumed_shards": len(valid),
        "computed_shards": 0,
    }
    try:
        if pending:
            model_bundle = load_model_and_tokenizer(
                model_name=args.model,
                model_revision=args.model_revision,
                dtype_name=args.dtype,
                device=args.device,
                cache_dir=args.cache_dir.expanduser().resolve(),
                local_files_only=True,
                expected_layers=args.expected_layers,
                expected_hidden_size=args.expected_hidden_size,
            )
            validate_loaded_model_bundle(model_bundle, args)
            runtime.update(
                {
                    "model_weights_loaded": True,
                    "resolved_model_revision": model_bundle["resolved_revision"],
                    "tokenizer_revision": model_bundle["tokenizer_revision"],
                    "model_class": model_bundle["model_class"],
                    "tokenizer_class": model_bundle["tokenizer_class"],
                    "transformers_version": model_bundle[
                        "transformers_version"
                    ],
                    "resolved_model_dtype": model_bundle["resolved_dtype"],
                    "resolved_hub_cache_dir": str(
                        model_bundle["resolved_hub_cache_dir"]
                    ),
                    "resolved_snapshot_dir": str(
                        model_bundle["resolved_snapshot_dir"]
                    ),
                    "gpu_name": model_bundle["gpu_name"],
                }
            )
            for plan_number, plan in enumerate(pending, start=1):
                payload = extract_one_shard(
                    plan=plan,
                    model=model_bundle["model"],
                    tokenizer=model_bundle["tokenizer"],
                    input_device=model_bundle["input_device"],
                    batch_size=args.batch_size,
                    start_token_position=1,
                    expected_layers=args.expected_layers,
                    expected_hidden_size=args.expected_hidden_size,
                    storage_dtype=storage_dtype,
                    identity=identity,
                    logger=logger,
                )
                write_verified_activation_shard(
                    shard_dir / plan.path_name,
                    payload=payload,
                    plan=plan,
                    identity=identity,
                    expected_layers=args.expected_layers,
                    expected_hidden_size=args.expected_hidden_size,
                    expected_dtype=storage_dtype,
                )
                valid[plan.index] = payload
                token_count_by_id.update(
                    zip(
                        payload["record_ids"],
                        payload["token_counts"],
                        strict=True,
                    )
                )
                runtime["computed_shards"] = plan_number
                update_progress(
                    output_dir,
                    status="running",
                    stage="neutral_activation_extraction",
                    completed_shards=len(valid),
                    total_shards=len(plans),
                    completed_records=sum(
                        len(plans[index].records) for index in valid
                    ),
                    total_records=len(records),
                )
                logger.info(
                    "shard_complete index=%d records=%d completed=%d/%d",
                    plan.index,
                    len(plan.records),
                    len(valid),
                    len(plans),
                )
        if len(valid) != len(plans):
            raise RuntimeError(
                f"Only {len(valid)} of {len(plans)} activation shards validated"
            )
    finally:
        if model_bundle is not None:
            model = model_bundle.pop("model", None)
            tokenizer = model_bundle.pop("tokenizer", None)
            del model, tokenizer, model_bundle
        gc.collect()
        if runtime["model_weights_loaded"] and torch.cuda.is_available():
            torch.cuda.empty_cache()
            runtime["peak_gpu_memory_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )

    # Reload every shard after model release; do not trust in-memory payloads.
    verified: dict[int, dict[str, Any]] = {}
    for plan in plans:
        payload = torch_load(shard_dir / plan.path_name, mmap=True)
        validate_activation_shard(
            payload,
            plan=plan,
            identity=identity,
            expected_layers=args.expected_layers,
            expected_hidden_size=args.expected_hidden_size,
            expected_dtype=storage_dtype,
        )
        verified[plan.index] = payload
    consolidated_path = output_dir / "neutral_activations.pt"
    if not pending and consolidated_path.exists():
        try:
            consolidated = load_and_validate_consolidated(
                consolidated_path,
                records=records,
                identity=identity,
                expected_layers=args.expected_layers,
                expected_hidden_size=args.expected_hidden_size,
                expected_dtype=storage_dtype,
                mmap=True,
            )
            runtime["consolidated_reused"] = True
            runtime["token_count_statistics"] = token_count_statistics(
                consolidated["token_counts"]
            )
            return consolidated, runtime
        except IncompatibleCheckpointError:
            raise
        except Exception as error:
            logger.warning(
                "rebuild_corrupt_consolidated path=%s error=%s",
                consolidated_path,
                error,
            )
    consolidated = combine_activation_shards(
        plans=plans,
        shards=verified,
        records=records,
        identity=identity,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        storage_dtype=storage_dtype,
    )
    atomic_torch_save(consolidated_path, consolidated)
    consolidated = load_and_validate_consolidated(
        consolidated_path,
        records=records,
        identity=identity,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        expected_dtype=storage_dtype,
        mmap=True,
    )
    runtime["consolidated_reused"] = False
    runtime["token_count_statistics"] = token_count_statistics(
        consolidated["token_counts"]
    )
    return consolidated, runtime


def validate_loaded_model_bundle(
    bundle: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """Validate the pinned local model before the first forward pass."""

    if bundle.get("resolved_revision") != args.model_revision:
        raise ValueError(
            "Loaded model revision differs from the requested revision: "
            f"{bundle.get('resolved_revision')!r}"
        )
    if bundle.get("tokenizer_revision") != args.model_revision:
        raise ValueError(
            "Loaded tokenizer revision differs from the requested revision: "
            f"{bundle.get('tokenizer_revision')!r}"
        )
    if int(bundle.get("num_layers", -1)) != args.expected_layers:
        raise ValueError("Loaded model layer count is incompatible")
    if int(bundle.get("hidden_size", -1)) != args.expected_hidden_size:
        raise ValueError("Loaded model hidden size is incompatible")
    model = bundle.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Model loader returned no torch module")
    if model.dtype != torch.bfloat16:
        raise ValueError(f"Model dtype is {model.dtype}; expected torch.bfloat16")


def extract_one_shard(
    *,
    plan: ShardPlan,
    model: Any,
    tokenizer: Any,
    input_device: torch.device,
    batch_size: int,
    start_token_position: int,
    expected_layers: int,
    expected_hidden_size: int,
    storage_dtype: torch.dtype,
    identity: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Extract one fixed shard as ``[records,layers,hidden]``."""

    transcripts = [record.transcript for record in plan.records]
    tokenised_without_padding = tokenizer(
        transcripts,
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    token_counts = [
        len(input_ids)
        for input_ids in tokenised_without_padding["input_ids"]
    ]
    batches = build_length_aware_batches(
        token_counts,
        batch_size=batch_size,
        max_batch_tokens=batch_size * max(token_counts),
        pad_to_multiple_of=1,
    )
    batch_results: list[tuple[Sequence[int], torch.Tensor]] = []
    for batch_number, record_indices in enumerate(batches):
        batch_records = [plan.records[index] for index in record_indices]
        transcripts = [record.transcript for record in batch_records]
        encoded = tokenizer(
            transcripts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids_cpu = encoded["input_ids"]
        attention_mask_cpu = encoded["attention_mask"]
        batch_token_counts = [
            int(value)
            for value in attention_mask_cpu.sum(dim=1).tolist()
        ]
        expected_batch_counts = [
            token_counts[index] for index in record_indices
        ]
        if batch_token_counts != expected_batch_counts:
            raise ValueError(
                "Batched token counts differ from the no-padding preflight"
            )
        if any(value < 1 for value in batch_token_counts):
            raise ValueError("Tokenizer produced an empty neutral transcript")
        input_ids = input_ids_cpu.to(input_device)
        attention_mask = attention_mask_cpu.to(input_device)
        outputs: Any = None
        hidden_states: Any = None
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise ValueError("Model returned no hidden states")
                if len(hidden_states) != expected_layers + 1:
                    raise ValueError(
                        f"Hidden-state tuple length {len(hidden_states)} != "
                        f"{expected_layers + 1}"
                    )
                activations = mean_transformer_hidden_states(
                    hidden_states,
                    attention_mask=attention_mask,
                    start_token_position=start_token_position,
                    num_layers=expected_layers,
                    hidden_size=expected_hidden_size,
                )
        except Exception as error:
            if is_out_of_memory(error):
                record_ids = [record.record_id for record in batch_records]
                raise RuntimeError(
                    "CUDA out of memory for neutral records "
                    f"{record_ids!r}; rerun --resume with a smaller --batch-size"
                ) from error
            raise
        batch_results.append((record_indices, activations))
        logger.info(
            "batch_complete shard=%d batch_index=%d batch=%d max_tokens=%d",
            plan.index,
            batch_number,
            len(batch_records),
            max(batch_token_counts),
        )
        del (
            outputs,
            activations,
            encoded,
            input_ids_cpu,
            attention_mask_cpu,
            input_ids,
            attention_mask,
            hidden_states,
        )
    shard_activations = restore_canonical_order(
        batch_results,
        total_records=len(plan.records),
    ).to(dtype=storage_dtype)
    del (
        batch_results,
        tokenised_without_padding,
    )
    expected_shape = (
        len(plan.records),
        expected_layers,
        expected_hidden_size,
    )
    if tuple(shard_activations.shape) != expected_shape:
        raise RuntimeError(
            f"Extracted shard shape {tuple(shard_activations.shape)} != "
            f"{expected_shape}"
        )
    if not bool(torch.isfinite(shard_activations).all()):
        raise ValueError("Extracted neutral activations contain nonfinite values")
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": identity["model_name"],
        "model_revision": identity["model_revision"],
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "padding_side": "right",
        "neutral_start_token_position": 1,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "neutral_token_aggregation": TOKEN_AGGREGATION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "embedding_hidden_state_included": False,
        "dataset_fingerprint": identity["fingerprint_sha256"],
        "number_of_layers": expected_layers,
        "hidden_size": expected_hidden_size,
        "activation_dtype": dtype_name(storage_dtype),
        "shard_index": plan.index,
        "record_ids": [record.record_id for record in plan.records],
        "topic_ids": [record.topic_id for record in plan.records],
        "sample_indices": [record.sample_index for record in plan.records],
        "dialogue_types": [record.dialogue_type for record in plan.records],
        "source_lines": [record.source_line for record in plan.records],
        "token_counts": token_counts,
        "activations": shard_activations,
    }


def write_verified_activation_shard(
    path: Path,
    *,
    payload: dict[str, Any],
    plan: ShardPlan,
    identity: Mapping[str, Any],
    expected_layers: int,
    expected_hidden_size: int,
    expected_dtype: torch.dtype,
) -> None:
    """Validate a temporary shard before atomically publishing it."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_raw)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = torch_load(temporary)
        validate_activation_shard(
            reloaded,
            plan=plan,
            identity=identity,
            expected_layers=expected_layers,
            expected_hidden_size=expected_hidden_size,
            expected_dtype=expected_dtype,
        )
        del reloaded
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def combine_activation_shards(
    *,
    plans: Sequence[ShardPlan],
    shards: Mapping[int, Mapping[str, Any]],
    records: Sequence[NeutralRecord],
    identity: Mapping[str, Any],
    expected_layers: int,
    expected_hidden_size: int,
    storage_dtype: torch.dtype,
) -> dict[str, Any]:
    """Combine fixed shards in JSONL order into ``[N,layers,hidden]``."""

    combined = torch.empty(
        (len(records), expected_layers, expected_hidden_size),
        dtype=storage_dtype,
        device="cpu",
    )
    record_ids: list[str] = []
    topic_ids: list[Any] = []
    sample_indices: list[Any] = []
    dialogue_types: list[Any] = []
    source_lines: list[int] = []
    token_counts: list[int] = []
    offset = 0
    for plan in plans:
        payload = shards[plan.index]
        activations = payload["activations"]
        count = len(plan.records)
        combined[offset : offset + count].copy_(activations)
        offset += count
        record_ids.extend(payload["record_ids"])
        topic_ids.extend(payload["topic_ids"])
        sample_indices.extend(payload["sample_indices"])
        dialogue_types.extend(payload["dialogue_types"])
        source_lines.extend(payload["source_lines"])
        token_counts.extend(payload["token_counts"])
    if offset != len(records):
        raise RuntimeError(
            f"Consolidation copied {offset} records; expected {len(records)}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": identity["model_name"],
        "model_revision": identity["model_revision"],
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "padding_side": "right",
        "neutral_start_token_position": 1,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "neutral_token_aggregation": TOKEN_AGGREGATION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "embedding_hidden_state_included": False,
        "dataset_fingerprint": identity["fingerprint_sha256"],
        "number_of_layers": expected_layers,
        "hidden_size": expected_hidden_size,
        "activation_dtype": dtype_name(storage_dtype),
        "record_ids": record_ids,
        "topic_ids": topic_ids,
        "sample_indices": sample_indices,
        "dialogue_types": dialogue_types,
        "source_lines": source_lines,
        "token_counts": token_counts,
        "activations": combined,
    }


def load_and_validate_consolidated(
    path: Path,
    *,
    records: Sequence[NeutralRecord],
    identity: Mapping[str, Any],
    expected_layers: int,
    expected_hidden_size: int,
    expected_dtype: torch.dtype,
    mmap: bool,
) -> dict[str, Any]:
    """Load and validate consolidated neutral activations ``[N,L,D]``."""

    if not path.is_file():
        raise FileNotFoundError(f"Consolidated activations do not exist: {path}")
    payload = torch_load(path, mmap=mmap)
    if not isinstance(payload, dict):
        raise CorruptCheckpointError("neutral_activations.pt is not a dictionary")
    immutable = {
        "schema_version": SCHEMA_VERSION,
        "model_name": identity["model_name"],
        "model_revision": identity["model_revision"],
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "add_special_tokens": True,
        "truncation": False,
        "padding_side": "right",
        "neutral_start_token_position": 1,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "neutral_token_aggregation": TOKEN_AGGREGATION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "embedding_hidden_state_included": False,
        "dataset_fingerprint": identity["fingerprint_sha256"],
        "number_of_layers": expected_layers,
        "hidden_size": expected_hidden_size,
        "activation_dtype": dtype_name(expected_dtype),
        "record_ids": [record.record_id for record in records],
        "topic_ids": [record.topic_id for record in records],
        "sample_indices": [record.sample_index for record in records],
        "dialogue_types": [record.dialogue_type for record in records],
        "source_lines": [record.source_line for record in records],
    }
    mismatches = {
        field: {"observed": payload.get(field), "expected": expected}
        for field, expected in immutable.items()
        if payload.get(field) != expected
    }
    if mismatches:
        raise IncompatibleCheckpointError(
            f"Consolidated activation metadata mismatch: {mismatches!r}"
        )
    token_counts = payload.get("token_counts")
    if (
        not isinstance(token_counts, list)
        or len(token_counts) != len(records)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in token_counts
        )
    ):
        raise CorruptCheckpointError(
            "Consolidated activation token_counts are invalid"
        )
    activations = payload.get("activations")
    expected_shape = (
        len(records),
        expected_layers,
        expected_hidden_size,
    )
    if not isinstance(activations, torch.Tensor):
        raise CorruptCheckpointError(
            "Consolidated activations field is not a tensor"
        )
    if tuple(activations.shape) != expected_shape:
        raise CorruptCheckpointError(
            f"Consolidated shape {tuple(activations.shape)} != {expected_shape}"
        )
    if activations.dtype != expected_dtype:
        raise CorruptCheckpointError(
            f"Consolidated dtype {activations.dtype} != {expected_dtype}"
        )
    if not bool(torch.isfinite(activations).all()):
        raise CorruptCheckpointError(
            "Consolidated activations contain nonfinite values"
        )
    return payload


def run_layerwise_pca(
    *,
    consolidated: Mapping[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Fit exactly one centred PCA to each ``[N,hidden]`` layer matrix."""

    activations = consolidated["activations"]
    expected_shape = (
        args.expected_records,
        args.expected_layers,
        args.expected_hidden_size,
    )
    if tuple(activations.shape) != expected_shape:
        raise ValueError(
            f"PCA source shape {tuple(activations.shape)} != {expected_shape}"
        )
    pca_dtype_value = pca_dtype(args.pca_dtype)
    pca_device_value = resolve_pca_device(args.pca_device)
    neutral_activations_sha256 = sha256_file(
        output_dir / "neutral_activations.pt"
    )
    pca_identity = {
        "dataset_fingerprint": consolidated["dataset_fingerprint"],
        "neutral_activations_sha256": neutral_activations_sha256,
        "model": args.model,
        "model_revision": args.model_revision,
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "number_of_neutral_transcripts": args.expected_records,
        "pca_method": PCA_METHOD,
        "pca_dtype": args.pca_dtype,
        "pca_device": str(pca_device_value),
        "explained_variance_threshold": args.explained_variance_threshold,
    }
    pca_fingerprint = canonical_json_sha256(pca_identity)
    layer_means: list[torch.Tensor] = []
    results: dict[int, dict[str, Any]] = {}
    summary_layers: dict[str, dict[str, Any]] = {}
    for layer_index in range(args.expected_layers):
        layer_matrix = activations[:, layer_index, :].to(
            device=pca_device_value,
            dtype=pca_dtype_value,
        ).contiguous()
        if tuple(layer_matrix.shape) != (
            args.expected_records,
            args.expected_hidden_size,
        ):
            raise RuntimeError(
                f"Layer {layer_index} PCA matrix shape is "
                f"{tuple(layer_matrix.shape)}"
            )
        result = compute_layer_pca(
            layer_matrix,
            explained_variance_threshold=args.explained_variance_threshold,
        )
        layer_means.append(result["neutral_mean"])
        results[layer_index] = result
        achieved = float(
            result["cumulative_explained_variance"][
                result["num_components"] - 1
            ].item()
        )
        previous = (
            None
            if result["num_components"] == 1
            else float(
                result["cumulative_explained_variance"][
                    result["num_components"] - 2
                ].item()
            )
        )
        summary_layers[str(layer_index)] = {
            "num_components": int(result["num_components"]),
            "achieved_cumulative_variance": achieved,
            "previous_cumulative_variance": previous,
            "total_variance": float(result["total_variance"]),
            "num_numerically_nonzero_components": int(
                result["num_numerically_nonzero_components"]
            ),
            "centred_column_mean_max_abs": float(
                result["centred_column_mean_max_abs"]
            ),
        }
        logger.info(
            "pca_layer_complete layer=%d components=%d achieved=%.9f "
            "previous=%s total_variance=%.9f",
            layer_index,
            result["num_components"],
            achieved,
            "null" if previous is None else f"{previous:.9f}",
            result["total_variance"],
        )
        update_progress(
            output_dir,
            status="running",
            stage="neutral_layer_pca",
            completed_pca_layers=layer_index + 1,
            total_pca_layers=args.expected_layers,
        )
        del layer_matrix
        if pca_device_value.type == "cuda":
            # Per-layer tensors are released naturally; no per-layer cache purge.
            pass

    neutral_layer_means = torch.stack(layer_means, dim=0)
    if tuple(neutral_layer_means.shape) != (
        args.expected_layers,
        args.expected_hidden_size,
    ):
        raise RuntimeError(
            f"Neutral layer means shape {tuple(neutral_layer_means.shape)} is "
            "incompatible"
        )
    if neutral_layer_means.dtype != torch.float32:
        neutral_layer_means = neutral_layer_means.float()
    if not bool(torch.isfinite(neutral_layer_means).all()):
        raise ValueError("Neutral layer means contain nonfinite values")

    summary: dict[str, Any] = {
        "model": args.model,
        "model_revision": args.model_revision,
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "number_of_neutral_transcripts": args.expected_records,
        "dataset_fingerprint": consolidated["dataset_fingerprint"],
        "neutral_activations_sha256": neutral_activations_sha256,
        "pca_fingerprint_sha256": pca_fingerprint,
        "explained_variance_threshold": args.explained_variance_threshold,
        "pca_method": PCA_METHOD,
        "pca_dtype": args.pca_dtype,
        "pca_device": str(pca_device_value),
        "pca_performed_separately_per_layer": True,
        "layers_averaged_together": False,
        "layers": summary_layers,
    }
    atomic_torch_save(
        output_dir / "neutral_layer_means.pt",
        neutral_layer_means,
    )
    atomic_torch_save(
        output_dir / "neutral_pca_components.pt",
        results,
    )
    atomic_write_json(output_dir / "neutral_pca_summary.json", summary)
    validate_saved_pca_outputs(
        output_dir=output_dir,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        expected_records=args.expected_records,
        threshold=args.explained_variance_threshold,
    )
    del neutral_layer_means, layer_means, results
    gc.collect()
    if pca_device_value.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def centre_layer_activations(
    layer_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Centre ``[N,hidden]`` and return mean ``[hidden]`` plus diagnostics."""

    if layer_matrix.ndim != 2:
        raise ValueError("PCA input must have shape [records,hidden]")
    if not layer_matrix.is_floating_point():
        raise ValueError("PCA input must be floating point")
    if not bool(torch.isfinite(layer_matrix).all()):
        raise ValueError("PCA input contains nonfinite values")
    layer_mean = layer_matrix.mean(dim=0, keepdim=True)
    centred = layer_matrix - layer_mean
    centred_max_abs_mean = float(
        centred.mean(dim=0).abs().max().item()
    )
    input_scale = float(layer_matrix.abs().max().item())
    tolerance = max(
        1e-6,
        128.0
        * torch.finfo(layer_matrix.dtype).eps
        * max(1.0, input_scale),
    )
    if centred_max_abs_mean > tolerance:
        raise ValueError(
            "Centred PCA matrix has nonzero column mean: "
            f"max_abs={centred_max_abs_mean} tolerance={tolerance}"
        )
    return centred, layer_mean.squeeze(0), centred_max_abs_mean, tolerance


def compute_layer_pca(
    layer_matrix: torch.Tensor,
    *,
    explained_variance_threshold: float,
) -> dict[str, Any]:
    """Compute exact economy PCA for one ``[N,hidden]`` layer matrix."""

    centred, layer_mean, centred_error, centring_tolerance = (
        centre_layer_activations(layer_matrix)
    )
    num_records, hidden_size = centred.shape
    if num_records < 2:
        raise ValueError("PCA requires at least two neutral transcripts")
    left_vectors, singular_values, vh = torch.linalg.svd(
        centred,
        full_matrices=False,
    )
    del left_vectors
    if not bool(torch.isfinite(singular_values).all()) or not bool(
        torch.isfinite(vh).all()
    ):
        raise ValueError("PCA decomposition contains nonfinite values")
    largest = singular_values[0]
    singular_tolerance = (
        max(num_records, hidden_size)
        * torch.finfo(singular_values.dtype).eps
        * largest
    )
    usable = singular_values > singular_tolerance
    num_usable = int(usable.sum().item())
    if num_usable < 1:
        raise ValueError("Neutral layer has no numerically nonzero PCA component")
    eigenvalues64 = (
        singular_values.double().square() / max(num_records - 1, 1)
    )
    total_variance64 = eigenvalues64.sum()
    if (
        not bool(torch.isfinite(total_variance64))
        or float(total_variance64.item()) <= EPSILON
    ):
        raise ValueError("Neutral layer total variance is zero or nonfinite")
    ratios64 = eigenvalues64 / total_variance64
    cumulative64 = torch.cumsum(ratios64, dim=0)
    num_components = select_components_for_variance_threshold(
        cumulative64[:num_usable],
        explained_variance_threshold,
    )
    # Clone the retained rows so torch.save does not serialize Vh's complete
    # [min(N,D), D] backing storage for a small [K,D] view.
    retained = vh[:num_components].clone().float().cpu()
    canonicalise_component_signs(retained)
    gram = retained @ retained.T
    identity = torch.eye(num_components, dtype=gram.dtype)
    orthonormality_error = float((gram - identity).abs().max().item())
    if orthonormality_error > 1e-4:
        raise ValueError(
            f"Retained PCA components are not orthonormal: "
            f"max_error={orthonormality_error}"
        )
    achieved = float(cumulative64[num_components - 1].item())
    previous = (
        None
        if num_components == 1
        else float(cumulative64[num_components - 2].item())
    )
    if achieved + 1e-12 < explained_variance_threshold:
        raise RuntimeError("Retained PCA basis does not reach the threshold")
    if (
        previous is not None
        and previous >= explained_variance_threshold
    ):
        raise RuntimeError("Retained PCA component count is not minimal")
    result = {
        "components": retained,
        "eigenvalues": eigenvalues64.cpu(),
        "explained_variance_ratio": ratios64.cpu(),
        "cumulative_explained_variance": cumulative64.cpu(),
        "num_components": num_components,
        "neutral_mean": layer_mean.float().cpu(),
        "total_variance": float(total_variance64.item()),
        "num_neutral_transcripts": num_records,
        "num_numerically_nonzero_components": num_usable,
        "singular_value_tolerance": float(singular_tolerance.item()),
        "discarded_numerical_variance": float(
            eigenvalues64[num_usable:].sum().item()
        ),
        "centred_column_mean_max_abs": centred_error,
        "centring_tolerance": centring_tolerance,
        "component_orthonormality_max_error": orthonormality_error,
        "component_sign_convention": (
            "largest-absolute loading is nonnegative"
        ),
    }
    del centred, vh, singular_values, eigenvalues64, ratios64, cumulative64
    return result


def select_components_for_variance_threshold(
    cumulative_variance: torch.Tensor,
    explained_variance_threshold: float,
) -> int:
    """Return the smallest one-based K whose cumulative variance reaches target."""

    if cumulative_variance.ndim != 1 or cumulative_variance.numel() < 1:
        raise ValueError("Cumulative explained variance must be a nonempty vector")
    threshold = torch.tensor(
        explained_variance_threshold,
        dtype=cumulative_variance.dtype,
        device=cumulative_variance.device,
    )
    threshold_index = torch.searchsorted(
        cumulative_variance,
        threshold,
        right=False,
    )
    if int(threshold_index.item()) >= cumulative_variance.numel():
        raise ValueError(
            "Numerically nonzero PCA components do not reach the variance "
            "threshold"
        )
    return int(threshold_index.item()) + 1


def canonicalise_component_signs(components: torch.Tensor) -> None:
    """Choose deterministic signs without changing PCA directions."""

    for row_index in range(components.shape[0]):
        row = components[row_index]
        pivot = int(row.abs().argmax().item())
        if float(row[pivot].item()) < 0.0:
            row.neg_()


def validate_saved_pca_outputs(
    *,
    output_dir: Path,
    expected_layers: int,
    expected_hidden_size: int,
    expected_records: int,
    threshold: float,
) -> None:
    """Reload and validate the three final PCA artifacts."""

    means = torch_load(output_dir / "neutral_layer_means.pt")
    if not isinstance(means, torch.Tensor) or tuple(means.shape) != (
        expected_layers,
        expected_hidden_size,
    ):
        raise CorruptCheckpointError("Saved neutral_layer_means.pt has wrong shape")
    if means.dtype != torch.float32 or not bool(torch.isfinite(means).all()):
        raise CorruptCheckpointError("Saved neutral layer means are invalid")
    summary = read_json_object(output_dir / "neutral_pca_summary.json")
    if (
        summary.get("number_of_layers") != expected_layers
        or summary.get("hidden_size") != expected_hidden_size
        or summary.get("number_of_neutral_transcripts") != expected_records
        or summary.get("explained_variance_threshold") != threshold
        or summary.get("pca_performed_separately_per_layer") is not True
        or summary.get("layers_averaged_together") is not False
    ):
        raise CorruptCheckpointError("Saved PCA summary contract is invalid")
    summary_layers = summary.get("layers")
    if (
        not isinstance(summary_layers, dict)
        or set(summary_layers) != {
            str(index) for index in range(expected_layers)
        }
    ):
        raise CorruptCheckpointError(
            "Saved PCA summary does not contain every layer exactly once"
        )
    results = torch_load(output_dir / "neutral_pca_components.pt")
    if not isinstance(results, dict) or set(results) != set(range(expected_layers)):
        raise CorruptCheckpointError(
            "Saved PCA components do not contain every layer exactly once"
        )
    for layer_index in range(expected_layers):
        layer = results[layer_index]
        if not isinstance(layer, dict):
            raise CorruptCheckpointError(f"Layer {layer_index} PCA entry is invalid")
        count = layer.get("num_components")
        components = layer.get("components")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or not isinstance(components, torch.Tensor)
            or tuple(components.shape) != (count, expected_hidden_size)
            or components.dtype != torch.float32
            or not bool(torch.isfinite(components).all())
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} retained PCA components are invalid"
            )
        if layer.get("num_neutral_transcripts") != expected_records:
            raise CorruptCheckpointError(
                f"Layer {layer_index} neutral record count is invalid"
            )
        neutral_mean = layer.get("neutral_mean")
        if (
            not isinstance(neutral_mean, torch.Tensor)
            or tuple(neutral_mean.shape) != (expected_hidden_size,)
            or neutral_mean.dtype != torch.float32
            or not bool(torch.isfinite(neutral_mean).all())
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} neutral mean is invalid"
            )
        if not torch.equal(neutral_mean, means[layer_index]):
            raise CorruptCheckpointError(
                f"Layer {layer_index} mean differs between saved artifacts"
            )
        eigenvalues = layer.get("eigenvalues")
        ratios = layer.get("explained_variance_ratio")
        cumulative = layer.get("cumulative_explained_variance")
        if (
            not isinstance(eigenvalues, torch.Tensor)
            or eigenvalues.ndim != 1
            or eigenvalues.numel() < count
            or eigenvalues.dtype != torch.float64
            or not bool(torch.isfinite(eigenvalues).all())
            or bool(torch.any(eigenvalues < 0))
            or not isinstance(ratios, torch.Tensor)
            or ratios.shape != eigenvalues.shape
            or ratios.dtype != torch.float64
            or not bool(torch.isfinite(ratios).all())
            or bool(torch.any(ratios < 0))
            or not isinstance(cumulative, torch.Tensor)
            or cumulative.ndim != 1
            or cumulative.shape != eigenvalues.shape
            or cumulative.dtype != torch.float64
            or not bool(torch.isfinite(cumulative).all())
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} saved PCA spectrum is invalid"
            )
        if eigenvalues.numel() > 1 and bool(
            torch.any(eigenvalues[1:] > eigenvalues[:-1])
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} eigenvalues are not descending"
            )
        total_variance = layer.get("total_variance")
        if (
            isinstance(total_variance, bool)
            or not isinstance(total_variance, (int, float))
            or not torch.isfinite(torch.tensor(float(total_variance)))
            or float(total_variance) <= 0.0
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} total variance is invalid"
            )
        eigenvalue_sum = float(eigenvalues.sum().item())
        if abs(eigenvalue_sum - float(total_variance)) > max(
            1e-10,
            1e-8 * abs(float(total_variance)),
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} eigenvalues disagree with total variance"
            )
        reconstructed_ratios = eigenvalues / float(total_variance)
        if not torch.allclose(
            ratios,
            reconstructed_ratios,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} variance ratios are inconsistent"
            )
        reconstructed_cumulative = torch.cumsum(ratios, dim=0)
        if not torch.allclose(
            cumulative,
            reconstructed_cumulative,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} cumulative variance is inconsistent"
            )
        component_gram = components @ components.T
        component_identity = torch.eye(count, dtype=components.dtype)
        orthonormality_error = float(
            (component_gram - component_identity).abs().max().item()
        )
        if orthonormality_error > 1e-4:
            raise CorruptCheckpointError(
                f"Layer {layer_index} components are not orthonormal"
            )
        usable = layer.get("num_numerically_nonzero_components")
        if (
            isinstance(usable, bool)
            or not isinstance(usable, int)
            or usable < count
            or usable > eigenvalues.numel()
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} numerical rank is invalid"
            )
        achieved = float(cumulative[count - 1].item())
        if achieved < threshold:
            raise CorruptCheckpointError(
                f"Layer {layer_index} retained variance {achieved} < {threshold}"
            )
        if count > 1 and float(cumulative[count - 2].item()) >= threshold:
            raise CorruptCheckpointError(
                f"Layer {layer_index} retained count is not minimal"
            )
        layer_summary = summary_layers[str(layer_index)]
        if (
            not isinstance(layer_summary, dict)
            or layer_summary.get("num_components") != count
            or layer_summary.get(
                "num_numerically_nonzero_components"
            ) != usable
            or abs(
                float(
                    layer_summary.get(
                        "achieved_cumulative_variance",
                        float("nan"),
                    )
                )
                - achieved
            )
            > 1e-12
            or (
                count == 1
                and layer_summary.get("previous_cumulative_variance")
                is not None
            )
            or (
                count > 1
                and abs(
                    float(
                        layer_summary.get(
                            "previous_cumulative_variance",
                            float("nan"),
                        )
                    )
                    - float(cumulative[count - 2].item())
                )
                > 1e-12
            )
            or abs(
                float(layer_summary.get("total_variance", float("nan")))
                - float(total_variance)
            )
            > max(1e-10, 1e-8 * abs(float(total_variance)))
        ):
            raise CorruptCheckpointError(
                f"Layer {layer_index} summary disagrees with tensor payload"
            )


def write_metadata(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    status: str,
    invocation_started_at: str,
    valid_records: int,
    invalid_records: int,
    extraction_runtime: Mapping[str, Any] | None = None,
    neutral_activation_shape: list[int] | None = None,
    pca_summary: Mapping[str, Any] | None = None,
) -> None:
    """Write current provenance and completion metadata atomically."""

    existing_path = output_dir / "metadata.json"
    existing: dict[str, Any] = {}
    if existing_path.exists():
        try:
            existing = read_json_object(existing_path)
        except ValueError:
            existing = {}
    latest_runtime = dict(extraction_runtime or {})
    runtime = dict(existing.get("extraction_runtime", {}))
    runtime.update(latest_runtime)
    retained_counts: dict[str, int] = {}
    achieved_variance: dict[str, float] = {}
    if pca_summary is not None:
        layers = pca_summary.get("layers", {})
        retained_counts = {
            str(layer): int(detail["num_components"])
            for layer, detail in layers.items()
        }
        achieved_variance = {
            str(layer): float(detail["achieved_cumulative_variance"])
            for layer, detail in layers.items()
        }
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "model": args.model,
        "model_revision": args.model_revision,
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "neutral_jsonl": str(args.neutral_jsonl.expanduser().resolve()),
        "neutral_manifest": str(
            args.neutral_manifest.expanduser().resolve()
        ),
        "neutral_jsonl_sha256": config["run_identity"][
            "neutral_jsonl_sha256"
        ],
        "neutral_manifest_sha256": config["run_identity"][
            "neutral_manifest_sha256"
        ],
        "manifest_status": manifest.get("status"),
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "local_files_only": True,
        "neutral_input_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "tokenisation": {
            "padding_side": "right",
            "add_special_tokens": True,
            "truncation": False,
        },
        "neutral_start_token_position": 1,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "neutral_token_aggregation": TOKEN_AGGREGATION,
        "neutral_pca_sample_unit": PCA_SAMPLE_UNIT,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "model_dtype": "bfloat16",
        "token_averaging_dtype": "float32",
        "neutral_activation_storage_dtype": args.activation_dtype,
        "pca_dtype": args.pca_dtype,
        "pca_device": args.pca_device,
        "pca_method": PCA_METHOD,
        "pca_performed_separately_per_layer": True,
        "explained_variance_threshold": args.explained_variance_threshold,
        "layers_averaged_together": False,
        "total_input_records": valid_records + invalid_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "neutral_activation_shape": neutral_activation_shape,
        "neutral_activations_sha256": (
            pca_summary.get("neutral_activations_sha256")
            if pca_summary is not None
            else existing.get("neutral_activations_sha256")
        ),
        "pca_fingerprint_sha256": (
            pca_summary.get("pca_fingerprint_sha256")
            if pca_summary is not None
            else existing.get("pca_fingerprint_sha256")
        ),
        "retained_component_count_per_layer": retained_counts,
        "achieved_cumulative_variance_per_layer": achieved_variance,
        "output_directory": str(output_dir),
        "initial_cli_arguments": config.get("initial_cli_arguments"),
        "latest_cli_arguments": json_safe_cli_arguments(args),
        "run_created_at": config.get("created_at"),
        "invocation_started_at": invocation_started_at,
        "updated_at": utc_now(),
        "completed_at": (
            utc_now()
            if status in {"completed", "activations_completed"}
            else None
        ),
        "torch_version": torch.__version__,
        "transformers_version": runtime.get(
            "transformers_version",
            existing.get("transformers_version"),
        ),
        "cuda_version": torch.version.cuda,
        "gpu_name": runtime.get("gpu_name", existing.get("gpu_name")),
        "peak_gpu_memory_bytes": runtime.get(
            "peak_gpu_memory_bytes",
            existing.get("peak_gpu_memory_bytes"),
        ),
        "model_weights_loaded_in_latest_invocation": runtime.get(
            "model_weights_loaded", False
        )
        if latest_runtime
        else False,
        "extraction_runtime": runtime,
        "pca_cpu_threads": (
            torch.get_num_threads() if args.pca_device == "cpu" else None
        ),
    }
    atomic_write_json(existing_path, metadata)


def update_progress(output_dir: Path, **fields: Any) -> None:
    """Update the small resumable progress marker atomically."""

    path = output_dir / "progress.json"
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = read_json_object(path)
        except ValueError:
            current = {}
    current.update(fields)
    if current.get("status") in {"running", "completed"}:
        current.pop("error", None)
        current.pop("failure", None)
    current["updated_at"] = utc_now()
    atomic_write_json(path, current)


def configure_logging(path: Path) -> logging.Logger:
    """Log to both the run file and standard output."""

    logger = logging.getLogger(
        f"emotionvectors.neutral_pca.{path.parent.name}"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)sZ %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def torch_load(path: Path, *, mmap: bool = False) -> Any:
    """Load trusted local run artifacts on CPU across supported Torch versions."""

    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def activation_dtype(name: str) -> torch.dtype:
    """Resolve the requested persisted activation dtype."""

    return {
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def pca_dtype(name: str) -> torch.dtype:
    """Resolve a non-half PCA dtype."""

    return {
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def dtype_name(value: torch.dtype) -> str:
    """Render a Torch dtype without its module prefix."""

    return str(value).removeprefix("torch.")


def resolve_pca_device(raw: str) -> torch.device:
    """Resolve one CPU or CUDA device for one-layer-at-a-time PCA."""

    device = torch.device(raw)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"PCA CUDA device is unavailable: {raw}")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"PCA CUDA device index is unavailable: {raw}")
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise ValueError(f"PCA supports only CPU or one CUDA device: {raw}")
    return torch.device("cpu")


def token_count_statistics(values: Sequence[int]) -> dict[str, Any]:
    """Summarise actual plain-text tokenizer counts."""

    counts = sorted(int(value) for value in values)
    return {
        "count": len(counts),
        "minimum": min(counts),
        "maximum": max(counts),
        "mean": sum(counts) / len(counts),
    }


def json_safe_cli_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse values to JSON-compatible provenance."""

    return {
        key: (
            str(value.expanduser().resolve())
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }


def fsync_directory(path: Path) -> None:
    """Best-effort durability for an atomic directory entry replacement."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mark_failed(output_dir: Path, status: str, error: str) -> None:
    try:
        progress_path = output_dir / "progress.json"
        if not progress_path.is_file():
            return
        progress = read_json_object(progress_path)
        if progress.get("status") != "running":
            return
        update_progress(
            output_dir,
            status=status,
            stage=status,
            error=error,
        )
        metadata_path = output_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = read_json_object(metadata_path)
            metadata.update(
                {
                    "status": status,
                    "error": error,
                    "updated_at": utc_now(),
                    "completed_at": None,
                }
            )
            atomic_write_json(metadata_path, metadata)
    except Exception:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _variance_threshold(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return value


def _immutable_revision(value: str) -> bool:
    return (
        len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


if __name__ == "__main__":
    raise SystemExit(main())
