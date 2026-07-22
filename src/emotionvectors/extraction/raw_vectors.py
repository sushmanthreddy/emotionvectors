#!/usr/bin/env python3
"""Extract per-layer story activations and compute one-vs-all emotion vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..constants import (
    EMOTIONS,
    HIDDEN_STATE_MAPPING,
    MODEL_ID,
    MODEL_REVISION,
    RESIDUAL_STREAM_DEFINITION,
)

TARGET_MODEL = MODEL_ID
START_TOKEN_POSITION_DEFAULT = 40
ANTHROPIC_ORIGINAL_START_POSITION = 50
TOKEN_POSITION_INDEXING = "one-based"
INPUT_TEXT_FIELD = "story"
NORMALIZATION_EPSILON = 1e-12


@dataclass(frozen=True)
class SourceRecord:
    """A validated input record and its source location."""

    payload: dict[str, Any]
    input_file: str
    line_number: int

    @property
    def record_id(self) -> Any:
        return self.payload["record_id"]

    @property
    def emotion(self) -> str:
        return str(self.payload["emotion"])

    @property
    def story(self) -> str:
        return str(self.payload["story"])


@dataclass(frozen=True)
class LoadedRecords:
    """Validated and selected records plus input accounting."""

    by_emotion: dict[str, list[SourceRecord]]
    invalid_records: list[dict[str, Any]]
    total_input_records: int
    basic_valid_records: int
    unrequested_valid_records: int
    records_excluded_by_limit: int
    source_files: list[dict[str, Any]]

    @property
    def selected_valid_records(self) -> int:
        return sum(len(records) for records in self.by_emotion.values())


class ExtractionOutOfMemory(RuntimeError):
    """An OOM that must stop the run without silently skipping a batch."""

    def __init__(
        self,
        *,
        emotion: str,
        record_ids: list[str],
        batch_size: int,
        sequence_lengths: list[int],
        original_error: BaseException,
    ) -> None:
        super().__init__(str(original_error))
        self.emotion = emotion
        self.record_ids = record_ids
        self.batch_size = batch_size
        self.sequence_lengths = sequence_lengths

    def as_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion,
            "record_ids": self.record_ids,
            "batch_size": self.batch_size,
            "sequence_lengths": self.sequence_lengths,
            "error": str(self),
            "recommendation": "Rerun with --resume and a smaller --batch-size.",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Qwen2.5-7B-Instruct residual-stream activations from cleaned "
            "stories and compute layer-preserving emotion vectors."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Directory whose top level contains one or more JSONL files",
    )
    source.add_argument(
        "--input-file",
        type=Path,
        help="One combined JSONL input file",
    )
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument(
        "--model-revision",
        default=MODEL_REVISION,
        help="Pinned Hugging Face commit for the model and tokenizer",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=EMOTIONS,
        default=list(EMOTIONS),
    )
    parser.add_argument(
        "--start-token-position",
        type=_positive_int,
        default=START_TOKEN_POSITION_DEFAULT,
        help="One-based first valid token included in each mean",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--story-activation-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=_positive_int,
        default=None,
        help="Enable truncation only when explicitly supplied",
    )
    parser.add_argument(
        "--max-records-per-emotion",
        type=_positive_int,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/Qwen2.5-7B-Instruct/raw_extraction"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_extraction(args)
    except KeyboardInterrupt:
        print("Activation extraction interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Activation extraction failed: {error}", file=sys.stderr)
        return 1


def run_extraction(args: argparse.Namespace) -> int:
    if args.model != TARGET_MODEL:
        raise ValueError(
            f"This experiment requires exactly {TARGET_MODEL!r}; received {args.model!r}"
        )
    if args.model_revision != MODEL_REVISION:
        raise ValueError(
            "This release requires exactly model revision "
            f"{MODEL_REVISION!r}; received {args.model_revision!r}"
        )

    requested_emotions = _canonical_requested_emotions(args.emotions)
    if len(requested_emotions) < 2:
        raise ValueError(
            "At least two requested emotions are required to compute other-emotions means"
        )
    if args.start_token_position != START_TOKEN_POSITION_DEFAULT:
        raise ValueError(
            "This experiment requires --start-token-position 40 exactly"
        )
    if (
        args.max_sequence_length is not None
        and args.max_sequence_length < args.start_token_position
    ):
        raise ValueError(
            "--max-sequence-length must be at least --start-token-position"
        )

    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else None
    input_file = args.input_file.expanduser().resolve() if args.input_file else None
    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output_directory(output_dir, resume=args.resume)
    story_activation_dir = output_dir / "story_activations"
    story_activation_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "extraction.log")
    extraction_invocation_timestamp = utc_now()

    input_paths = discover_input_paths(input_dir=input_dir, input_file=input_file)
    source_files = describe_source_files(input_paths)
    loaded = load_input_records(
        input_paths=input_paths,
        requested_emotions=requested_emotions,
        max_records_per_emotion=args.max_records_per_emotion,
        source_files=source_files,
        logger=logger,
    )
    atomic_write_jsonl(
        output_dir / "skipped_invalid_records.jsonl",
        loaded.invalid_records,
    )
    for emotion in requested_emotions:
        if not loaded.by_emotion[emotion]:
            raise ValueError(f"No valid input stories available for emotion: {emotion}")

    logger.info(
        "input_loaded total=%d selected_valid=%d malformed=%d files=%d",
        loaded.total_input_records,
        loaded.selected_valid_records,
        len(loaded.invalid_records),
        len(input_paths),
    )

    model_bundle = load_model_and_tokenizer(
        model_name=args.model,
        model_revision=args.model_revision,
        dtype_name=args.dtype,
        device=args.device,
    )
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    if num_layers <= 0 or hidden_size <= 0:
        raise ValueError(
            f"Invalid model dimensions: layers={num_layers}, hidden_size={hidden_size}"
        )

    signature = build_extraction_signature(
        args=args,
        requested_emotions=requested_emotions,
        input_dir=input_dir,
        input_file=input_file,
        source_files=source_files,
        model_bundle=model_bundle,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )
    config = prepare_config(
        output_dir=output_dir,
        args=args,
        signature=signature,
        created_at=extraction_invocation_timestamp,
        resume=args.resume,
    )
    extraction_start = str(config["created_at"])

    metadata = prepare_metadata(
        output_dir=output_dir,
        signature=signature,
        model_bundle=model_bundle,
        args=args,
        loaded=loaded,
        extraction_start=extraction_start,
    )
    atomic_write_json(output_dir / "metadata.json", metadata)

    completed_payloads: dict[str, dict[str, Any]] = {}
    completed_emotions: list[str] = []
    for emotion in requested_emotions:
        path = story_activation_dir / f"{emotion}.pt"
        if not path.exists():
            continue
        if not args.resume:
            raise FileExistsError(f"Unexpected existing activation file: {path}")
        payload = load_and_validate_story_activation(
            path=path,
            expected_emotion=emotion,
            expected_signature=signature,
            num_layers=num_layers,
            hidden_size=hidden_size,
            expected_dtype=args.story_activation_dtype,
        )
        completed_payloads[emotion] = payload
        completed_emotions.append(emotion)
        logger.info(
            "resume_completed_emotion emotion=%s valid_stories=%d",
            emotion,
            int(payload["activations"].shape[0]),
        )

    write_short_story_log(
        output_dir / "skipped_short_stories.jsonl",
        completed_payloads,
        requested_emotions,
    )
    progress = build_progress(
        requested_emotions=requested_emotions,
        completed_emotions=completed_emotions,
        completed_payloads=completed_payloads,
        status="running",
    )
    atomic_write_json(output_dir / "progress.json", progress)

    try:
        for emotion in requested_emotions:
            if emotion in completed_payloads:
                continue
            payload = extract_one_emotion(
                emotion=emotion,
                records=loaded.by_emotion[emotion],
                model=model,
                tokenizer=tokenizer,
                model_input_device=model_bundle["input_device"],
                num_layers=num_layers,
                hidden_size=hidden_size,
                start_token_position=args.start_token_position,
                batch_size=args.batch_size,
                max_sequence_length=args.max_sequence_length,
                story_activation_dtype=args.story_activation_dtype,
                extraction_signature=signature,
                model_name=args.model,
                resolved_model_revision=model_bundle["resolved_revision"],
                logger=logger,
            )
            destination = story_activation_dir / f"{emotion}.pt"
            atomic_torch_save(destination, payload)
            payload = load_and_validate_story_activation(
                path=destination,
                expected_emotion=emotion,
                expected_signature=signature,
                num_layers=num_layers,
                hidden_size=hidden_size,
                expected_dtype=args.story_activation_dtype,
            )
            completed_payloads[emotion] = payload
            completed_emotions.append(emotion)
            write_short_story_log(
                output_dir / "skipped_short_stories.jsonl",
                completed_payloads,
                requested_emotions,
            )
            progress = build_progress(
                requested_emotions=requested_emotions,
                completed_emotions=completed_emotions,
                completed_payloads=completed_payloads,
                status="running",
            )
            atomic_write_json(output_dir / "progress.json", progress)
            logger.info(
                "emotion_complete emotion=%s valid_stories=%d short_stories=%d shape=%s",
                emotion,
                int(payload["activations"].shape[0]),
                len(payload["skipped_short_records"]),
                tuple(payload["activations"].shape),
            )
    except ExtractionOutOfMemory as error:
        logger.error(
            "out_of_memory emotion=%s record_ids=%s batch_size=%d sequence_lengths=%s "
            "recommendation='rerun with --resume and a smaller --batch-size'",
            error.emotion,
            error.record_ids,
            error.batch_size,
            error.sequence_lengths,
        )
        progress = build_progress(
            requested_emotions=requested_emotions,
            completed_emotions=completed_emotions,
            completed_payloads=completed_payloads,
            status="out_of_memory",
            failure=error.as_dict(),
        )
        atomic_write_json(output_dir / "progress.json", progress)
        metadata["status"] = "out_of_memory"
        metadata["extraction_end_timestamp"] = utc_now()
        metadata["failure"] = error.as_dict()
        atomic_write_json(output_dir / "metadata.json", metadata)
        print(
            "GPU out of memory. Completed emotions were saved. "
            "Rerun with --resume and a smaller --batch-size.",
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        progress = build_progress(
            requested_emotions=requested_emotions,
            completed_emotions=completed_emotions,
            completed_payloads=completed_payloads,
            status="failed",
            failure={"error": str(error)},
        )
        atomic_write_json(output_dir / "progress.json", progress)
        metadata["status"] = "failed"
        metadata["extraction_end_timestamp"] = utc_now()
        metadata["failure"] = {"error": str(error)}
        atomic_write_json(output_dir / "metadata.json", metadata)
        raise

    del model
    model_bundle["model"] = None

    try:
        final_summary = compute_and_save_emotion_vectors(
            output_dir=output_dir,
            requested_emotions=requested_emotions,
            extraction_signature=signature,
            num_layers=num_layers,
            hidden_size=hidden_size,
            story_activation_dtype=args.story_activation_dtype,
        )
    except Exception as error:
        progress = build_progress(
            requested_emotions=requested_emotions,
            completed_emotions=completed_emotions,
            completed_payloads=completed_payloads,
            status="failed_during_finalization",
            failure={"error": str(error)},
        )
        atomic_write_json(output_dir / "progress.json", progress)
        metadata["status"] = "failed_during_finalization"
        metadata["extraction_end_timestamp"] = utc_now()
        metadata["failure"] = {"error": str(error)}
        atomic_write_json(output_dir / "metadata.json", metadata)
        raise
    completed_payloads = final_summary["payloads"]
    progress = build_progress(
        requested_emotions=requested_emotions,
        completed_emotions=list(requested_emotions),
        completed_payloads=completed_payloads,
        status="completed",
    )
    progress["final_artifacts_complete"] = True
    atomic_write_json(output_dir / "progress.json", progress)

    short_count = sum(
        len(payload["skipped_short_records"])
        for payload in completed_payloads.values()
    )
    truncation_count = sum(
        sum(bool(value) for value in payload["truncated"])
        for payload in completed_payloads.values()
    )
    valid_counts = final_summary["emotion_counts"]
    metadata.update(
        {
            "status": "completed",
            "total_input_records": loaded.total_input_records,
            "valid_input_records": loaded.selected_valid_records,
            "malformed_records": len(loaded.invalid_records),
            "stories_shorter_than_start_position": short_count,
            "valid_count_per_emotion": valid_counts,
            "valid_story_activations": sum(valid_counts.values()),
            "truncation_count": truncation_count,
            "extraction_end_timestamp": utc_now(),
            "one_story_activation_shape": [num_layers, hidden_size],
            "emotion_mean_shape": [num_layers, hidden_size],
            "other_emotions_mean_shape": [num_layers, hidden_size],
            "emotion_vector_shape": [num_layers, hidden_size],
            "stacked_emotion_vector_shape": [
                len(requested_emotions),
                num_layers,
                hidden_size,
            ],
        }
    )
    atomic_write_json(output_dir / "metadata.json", metadata)
    logger.info(
        "extraction_complete valid_story_activations=%d malformed=%d short=%d "
        "stacked_shape=%s",
        sum(valid_counts.values()),
        len(loaded.invalid_records),
        short_count,
        (len(requested_emotions), num_layers, hidden_size),
    )
    return 0


def discover_input_paths(
    *, input_dir: Path | None, input_file: Path | None
) -> list[Path]:
    if input_dir is not None:
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
        paths = sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())
        if not paths:
            raise FileNotFoundError(f"No top-level *.jsonl files found in {input_dir}")
        return paths
    if input_file is None:
        raise ValueError("Exactly one of --input-dir or --input-file is required")
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    return [input_file]


def describe_source_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        descriptions.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return descriptions


def load_input_records(
    *,
    input_paths: Sequence[Path],
    requested_emotions: tuple[str, ...],
    max_records_per_emotion: int | None,
    source_files: list[dict[str, Any]],
    logger: logging.Logger,
) -> LoadedRecords:
    by_emotion = {emotion: [] for emotion in requested_emotions}
    invalid: list[dict[str, Any]] = []
    total_input_records = 0
    basic_valid_records = 0
    unrequested_valid_records = 0

    for path in input_paths:
        filename_emotion = path.stem if path.stem in EMOTIONS else None
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                total_input_records += 1
                record: Any = None
                if not line.strip():
                    invalid.append(
                        invalid_record_entry(
                            path=path,
                            line_number=line_number,
                            record=None,
                            reason="blank line is not valid JSON",
                        )
                    )
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    invalid.append(
                        invalid_record_entry(
                            path=path,
                            line_number=line_number,
                            record=None,
                            reason=f"invalid JSON: {error.msg}",
                        )
                    )
                    continue

                reason = validate_basic_record(record)
                if reason is not None:
                    invalid.append(
                        invalid_record_entry(
                            path=path,
                            line_number=line_number,
                            record=record,
                            reason=reason,
                        )
                    )
                    continue
                emotion = str(record["emotion"])
                if emotion not in EMOTIONS:
                    invalid.append(
                        invalid_record_entry(
                            path=path,
                            line_number=line_number,
                            record=record,
                            reason=f"unsupported emotion label: {emotion}",
                        )
                    )
                    continue
                basic_valid_records += 1
                if filename_emotion is not None and filename_emotion != emotion:
                    logger.warning(
                        "filename_emotion_mismatch file=%s line=%d filename_emotion=%s "
                        "record_emotion=%s using_record_emotion=true",
                        path,
                        line_number,
                        filename_emotion,
                        emotion,
                    )
                if emotion not in by_emotion:
                    unrequested_valid_records += 1
                    continue
                by_emotion[emotion].append(
                    SourceRecord(
                        payload=record,
                        input_file=str(path),
                        line_number=line_number,
                    )
                )

    records_excluded_by_limit = 0
    if max_records_per_emotion is not None:
        for emotion in requested_emotions:
            records = by_emotion[emotion]
            records_excluded_by_limit += max(0, len(records) - max_records_per_emotion)
            by_emotion[emotion] = records[:max_records_per_emotion]

    return LoadedRecords(
        by_emotion=by_emotion,
        invalid_records=invalid,
        total_input_records=total_input_records,
        basic_valid_records=basic_valid_records,
        unrequested_valid_records=unrequested_valid_records,
        records_excluded_by_limit=records_excluded_by_limit,
        source_files=source_files,
    )


def validate_basic_record(record: Any) -> str | None:
    if not isinstance(record, dict):
        return "JSON value is not an object"
    if "record_id" not in record:
        return "missing record_id"
    if "emotion" not in record:
        return "missing emotion"
    if "story" not in record:
        return "missing story"
    if not isinstance(record["story"], str) or not record["story"].strip():
        return "story is not a nonempty string"
    return None


def invalid_record_entry(
    *, path: Path, line_number: int, record: Any, reason: str
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "input_filename": str(path),
        "line_number": line_number,
        "reason": reason,
    }
    if isinstance(record, dict):
        if "record_id" in record:
            entry["record_id"] = record["record_id"]
        if "emotion" in record:
            entry["emotion"] = record["emotion"]
    return entry


def load_model_and_tokenizer(
    *, model_name: str, model_revision: str | None, dtype_name: str, device: str
) -> dict[str, Any]:
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for activation extraction") from error

    resolved_dtype = resolve_torch_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        model.to(torch.device(device))
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    input_device = model_input_device(model)
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
    if input_device.type == "cuda" and torch.cuda.is_available():
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
    }


def extract_one_emotion(
    *,
    emotion: str,
    records: Sequence[SourceRecord],
    model: Any,
    tokenizer: Any,
    model_input_device: torch.device,
    num_layers: int,
    hidden_size: int,
    start_token_position: int,
    batch_size: int,
    max_sequence_length: int | None,
    story_activation_dtype: str,
    extraction_signature: dict[str, Any],
    model_name: str,
    resolved_model_revision: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    activation_batches: list[torch.Tensor] = []
    activation_sum_float64 = torch.zeros(
        (num_layers, hidden_size), dtype=torch.float64, device="cpu"
    )
    metadata: dict[str, list[Any]] = {
        "record_ids": [],
        "emotion_groups": [],
        "topic_ids": [],
        "topics": [],
        "sample_indices": [],
        "prompt_versions": [],
        "generator_models": [],
        "accepted_seeds": [],
        "attempt_counts": [],
        "emotion_word_present": [],
        "non_latin_letter_present": [],
        "accepted_after_max_attempts": [],
        "created_at_values": [],
        "original_token_counts": [],
        "processed_token_counts": [],
        "truncated": [],
        "input_filenames": [],
        "input_line_numbers": [],
        "source_metadata": [],
    }
    skipped_short: list[dict[str, Any]] = []
    storage_dtype = (
        torch.float16 if story_activation_dtype == "float16" else torch.float32
    )

    for batch_start in range(0, len(records), batch_size):
        source_batch = list(records[batch_start : batch_start + batch_size])
        stories = [record.story for record in source_batch]
        encoded, original_counts, processed_counts = tokenize_story_batch(
            tokenizer=tokenizer,
            stories=stories,
            max_sequence_length=max_sequence_length,
        )
        long_indices: list[int] = []
        for index, (record, original_count, processed_count) in enumerate(
            zip(source_batch, original_counts, processed_counts, strict=True)
        ):
            if processed_count < start_token_position:
                skipped_short.append(
                    short_story_entry(
                        record=record,
                        token_count=processed_count,
                        original_token_count=original_count,
                        required_start_position=start_token_position,
                    )
                )
            else:
                long_indices.append(index)
        if not long_indices:
            continue

        selected_indices = torch.tensor(long_indices, dtype=torch.long)
        input_ids_cpu = encoded["input_ids"].index_select(0, selected_indices)
        attention_mask_cpu = encoded["attention_mask"].index_select(
            0, selected_indices
        )
        valid_records = [source_batch[index] for index in long_indices]
        valid_original_counts = [original_counts[index] for index in long_indices]
        valid_processed_counts = [processed_counts[index] for index in long_indices]
        try:
            input_ids = input_ids_cpu.to(model_input_device)
            attention_mask = attention_mask_cpu.to(model_input_device)
            content_positions = attention_mask.long().cumsum(dim=1)
            selected_token_mask = attention_mask.bool() & (
                content_positions >= start_token_position
            )
            selected_counts = selected_token_mask.sum(dim=1, keepdim=True)
            if bool(torch.any(selected_counts <= 0)):
                raise ValueError(
                    f"{emotion}: token-selection mask produced an empty mean"
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
                if hidden_states is None:
                    raise ValueError("Model did not return hidden states")
                if len(hidden_states) != num_layers + 1:
                    raise ValueError(
                        "Hidden-state count mismatch: "
                        f"expected {num_layers + 1}, received {len(hidden_states)}"
                    )

                layer_means_cpu: list[torch.Tensor] = []
                for layer_index in range(num_layers):
                    hidden = hidden_states[layer_index + 1]
                    expected_shape = (
                        len(valid_records),
                        input_ids.shape[1],
                        hidden_size,
                    )
                    if tuple(hidden.shape) != expected_shape:
                        raise ValueError(
                            f"Layer {layer_index} hidden shape {tuple(hidden.shape)} "
                            f"does not match {expected_shape}"
                        )
                    layer_mask = selected_token_mask.to(hidden.device).unsqueeze(-1)
                    layer_denominator = (
                        selected_counts.to(hidden.device, dtype=torch.float32)
                        .clamp_min(1)
                    )
                    hidden_fp32 = hidden.float()
                    summed = (hidden_fp32 * layer_mask).sum(dim=1)
                    layer_mean_cpu = (summed / layer_denominator).to(
                        device="cpu", dtype=torch.float32
                    )
                    layer_means_cpu.append(layer_mean_cpu)
                    del (
                        hidden_fp32,
                        summed,
                        layer_mask,
                        layer_denominator,
                        layer_mean_cpu,
                    )
                story_layer_means_cpu = torch.stack(layer_means_cpu, dim=1)
                expected_story_batch_shape = (
                    len(valid_records),
                    num_layers,
                    hidden_size,
                )
                if tuple(story_layer_means_cpu.shape) != expected_story_batch_shape:
                    raise ValueError(
                        "Story activation batch has shape "
                        f"{tuple(story_layer_means_cpu.shape)}; "
                        f"expected {expected_story_batch_shape}"
                    )
        except RuntimeError as error:
            if is_out_of_memory(error):
                raise ExtractionOutOfMemory(
                    emotion=emotion,
                    record_ids=[str(record.record_id) for record in valid_records],
                    batch_size=len(valid_records),
                    sequence_lengths=valid_processed_counts,
                    original_error=error,
                ) from error
            raise
        finally:
            if "outputs" in locals():
                del outputs
            if "hidden_states" in locals():
                del hidden_states

        if not bool(torch.isfinite(story_layer_means_cpu).all()):
            raise ValueError(f"{emotion}: non-finite story activation detected")
        activation_sum_float64 += story_layer_means_cpu.to(torch.float64).sum(dim=0)
        activation_batches.append(story_layer_means_cpu.to(storage_dtype))

        for record, original_count, processed_count in zip(
            valid_records,
            valid_original_counts,
            valid_processed_counts,
            strict=True,
        ):
            append_source_metadata(
                metadata=metadata,
                record=record,
                original_token_count=original_count,
                processed_token_count=processed_count,
            )
        logger.info(
            "batch_complete emotion=%s processed=%d/%d batch_size=%d "
            "sequence_length=%d",
            emotion,
            len(metadata["record_ids"]),
            len(records),
            len(valid_records),
            int(input_ids.shape[1]),
        )

        del (
            story_layer_means_cpu,
            layer_means_cpu,
            hidden,
            input_ids,
            attention_mask,
            content_positions,
            selected_token_mask,
            selected_counts,
        )

    valid_count = len(metadata["record_ids"])
    if valid_count > 0:
        activations = torch.cat(activation_batches, dim=0)
    else:
        activations = torch.empty(
            (0, num_layers, hidden_size), dtype=storage_dtype, device="cpu"
        )
    expected_shape = (valid_count, num_layers, hidden_size)
    if tuple(activations.shape) != expected_shape:
        raise ValueError(
            f"{emotion}: activation tensor shape {tuple(activations.shape)} "
            f"does not match {expected_shape}"
        )
    if activations.dtype != storage_dtype:
        raise ValueError(
            f"{emotion}: activation dtype {activations.dtype} does not match {storage_dtype}"
        )
    if not bool(torch.isfinite(activations).all()):
        raise ValueError(f"{emotion}: saved activations contain NaN or infinity")
    if not bool(torch.isfinite(activation_sum_float64).all()):
        raise ValueError(f"{emotion}: activation sum contains NaN or infinity")

    return {
        "emotion": emotion,
        "target_model": model_name,
        "model_revision": resolved_model_revision,
        "start_token_position": start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "num_transformer_layers": num_layers,
        "hidden_size": hidden_size,
        "story_activation_dtype": story_activation_dtype,
        "extraction_configuration": extraction_signature,
        "extraction_configuration_sha256": canonical_json_sha256(
            extraction_signature
        ),
        **metadata,
        "skipped_short_records": skipped_short,
        "activation_sum_float64": activation_sum_float64,
        "activations": activations,
    }


def tokenize_story_batch(
    *, tokenizer: Any, stories: Sequence[str], max_sequence_length: int | None
) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    tokenization_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_sequence_length is not None,
        "add_special_tokens": True,
        "return_attention_mask": True,
    }
    if max_sequence_length is not None:
        tokenization_kwargs["max_length"] = max_sequence_length
    encoded_raw = tokenizer(list(stories), **tokenization_kwargs)
    input_ids = encoded_raw["input_ids"]
    attention_mask = encoded_raw.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    processed_counts = [int(value) for value in attention_mask.sum(dim=1).tolist()]

    if max_sequence_length is None:
        original_counts = list(processed_counts)
    else:
        original_encoded = tokenizer(
            list(stories),
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )
        original_ids = original_encoded["input_ids"]
        original_counts = [len(row) for row in original_ids]
    return encoded, original_counts, processed_counts


def append_source_metadata(
    *,
    metadata: dict[str, list[Any]],
    record: SourceRecord,
    original_token_count: int,
    processed_token_count: int,
) -> None:
    payload = record.payload
    field_mapping = {
        "record_ids": "record_id",
        "emotion_groups": "emotion_group",
        "topic_ids": "topic_id",
        "topics": "topic",
        "sample_indices": "sample_index",
        "prompt_versions": "prompt_version",
        "generator_models": "generator_model",
        "accepted_seeds": "accepted_seed",
        "attempt_counts": "attempt_count",
        "emotion_word_present": "emotion_word_present",
        "non_latin_letter_present": "non_latin_letter_present",
        "accepted_after_max_attempts": "accepted_after_max_attempts",
        "created_at_values": "created_at",
    }
    for output_name, source_name in field_mapping.items():
        metadata[output_name].append(payload.get(source_name))
    metadata["original_token_counts"].append(original_token_count)
    metadata["processed_token_counts"].append(processed_token_count)
    metadata["truncated"].append(original_token_count > processed_token_count)
    metadata["input_filenames"].append(record.input_file)
    metadata["input_line_numbers"].append(record.line_number)
    metadata["source_metadata"].append(
        {
            key: value
            for key, value in payload.items()
            if key not in {"story", "prompt", "raw_completion"}
        }
    )


def short_story_entry(
    *,
    record: SourceRecord,
    token_count: int,
    original_token_count: int,
    required_start_position: int,
) -> dict[str, Any]:
    payload = record.payload
    return {
        "input_filename": record.input_file,
        "line_number": record.line_number,
        "record_id": payload.get("record_id"),
        "emotion": payload.get("emotion"),
        "topic_id": payload.get("topic_id"),
        "sample_index": payload.get("sample_index"),
        "token_count": token_count,
        "original_token_count": original_token_count,
        "required_start_position": required_start_position,
        "reason": (
            f"story contains fewer than {required_start_position} valid tokens"
        ),
    }


def compute_and_save_emotion_vectors(
    *,
    output_dir: Path,
    requested_emotions: tuple[str, ...],
    extraction_signature: dict[str, Any],
    num_layers: int,
    hidden_size: int,
    story_activation_dtype: str,
) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    emotion_sums: dict[str, torch.Tensor] = {}
    emotion_counts: dict[str, int] = {}
    for emotion in requested_emotions:
        path = output_dir / "story_activations" / f"{emotion}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing story activation file: {path}")
        payload = load_and_validate_story_activation(
            path=path,
            expected_emotion=emotion,
            expected_signature=extraction_signature,
            num_layers=num_layers,
            hidden_size=hidden_size,
            expected_dtype=story_activation_dtype,
        )
        payloads[emotion] = payload
        count = int(payload["activations"].shape[0])
        if count <= 0:
            raise ValueError(f"No valid stories available for emotion: {emotion}")
        emotion_counts[emotion] = count
        emotion_sums[emotion] = payload["activation_sum_float64"]

    total_sum = sum(emotion_sums.values())
    total_count = sum(emotion_counts.values())
    emotion_means: dict[str, torch.Tensor] = {}
    other_emotions_means: dict[str, torch.Tensor] = {}
    raw_vectors: dict[str, torch.Tensor] = {}
    unit_vectors: dict[str, torch.Tensor] = {}
    vector_norms: dict[str, dict[str, float]] = {}

    expected_layer_shape = (num_layers, hidden_size)
    for emotion in requested_emotions:
        emotion_count = emotion_counts[emotion]
        other_count = total_count - emotion_count
        if other_count <= 0:
            raise ValueError(f"No valid comparison stories for emotion: {emotion}")
        emotion_mean64 = emotion_sums[emotion] / emotion_count
        other_mean64 = (total_sum - emotion_sums[emotion]) / other_count
        emotion_mean = emotion_mean64.to(torch.float32)
        other_mean = other_mean64.to(torch.float32)
        raw_vector = (emotion_mean64 - other_mean64).to(torch.float32)
        for name, tensor in (
            ("emotion mean", emotion_mean),
            ("other-emotions mean", other_mean),
            ("emotion vector", raw_vector),
        ):
            if tuple(tensor.shape) != expected_layer_shape:
                raise ValueError(
                    f"{emotion} {name} shape {tuple(tensor.shape)} "
                    f"does not match {expected_layer_shape}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{emotion} {name} contains NaN or infinity")

        norms = raw_vector.norm(p=2, dim=-1, keepdim=True)
        unit_vector = raw_vector / norms.clamp_min(NORMALIZATION_EPSILON)
        nonzero = norms.squeeze(-1) > NORMALIZATION_EPSILON
        if bool(torch.any(nonzero)):
            observed = unit_vector.norm(p=2, dim=-1)[nonzero]
            if not torch.allclose(
                observed,
                torch.ones_like(observed),
                rtol=1e-4,
                atol=1e-5,
            ):
                raise ValueError(f"{emotion}: unit-vector norms are not close to one")

        emotion_means[emotion] = emotion_mean
        other_emotions_means[emotion] = other_mean
        raw_vectors[emotion] = raw_vector
        unit_vectors[emotion] = unit_vector
        vector_norms[emotion] = {
            str(layer): float(norms[layer, 0].item())
            for layer in range(num_layers)
        }

    stacked = torch.stack(
        [raw_vectors[emotion] for emotion in requested_emotions], dim=0
    )
    expected_stacked_shape = (len(requested_emotions), num_layers, hidden_size)
    if tuple(stacked.shape) != expected_stacked_shape:
        raise ValueError(
            f"Stacked vector shape {tuple(stacked.shape)} "
            f"does not match {expected_stacked_shape}"
        )
    if not bool(torch.isfinite(stacked).all()):
        raise ValueError("Stacked emotion vectors contain NaN or infinity")

    atomic_torch_save(output_dir / "emotion_means.pt", emotion_means)
    atomic_write_json(output_dir / "emotion_counts.json", emotion_counts)
    atomic_torch_save(output_dir / "emotion_vectors_raw.pt", raw_vectors)
    atomic_torch_save(output_dir / "emotion_vectors_unit.pt", unit_vectors)
    atomic_torch_save(
        output_dir / "emotion_vectors_stacked.pt",
        {
            "emotion_order": list(requested_emotions),
            "vectors": stacked,
        },
    )
    atomic_write_json(output_dir / "emotion_vector_norms.json", vector_norms)
    validate_final_artifacts(
        output_dir=output_dir,
        requested_emotions=requested_emotions,
        emotion_counts=emotion_counts,
        emotion_sums=emotion_sums,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )
    return {
        "payloads": payloads,
        "emotion_counts": emotion_counts,
        "emotion_means": emotion_means,
        "other_emotions_means": other_emotions_means,
        "raw_vectors": raw_vectors,
        "unit_vectors": unit_vectors,
        "stacked_vectors": stacked,
    }


def load_and_validate_story_activation(
    *,
    path: Path,
    expected_emotion: str,
    expected_signature: dict[str, Any],
    num_layers: int,
    hidden_size: int,
    expected_dtype: str,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not load story activation file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Story activation file is not a dictionary: {path}")
    if payload.get("emotion") != expected_emotion:
        raise ValueError(
            f"Emotion mismatch in {path}: {payload.get('emotion')!r} != {expected_emotion!r}"
        )
    if payload.get("extraction_configuration") != expected_signature:
        raise ValueError(f"Extraction configuration mismatch in {path}")
    if payload.get("extraction_configuration_sha256") != canonical_json_sha256(
        expected_signature
    ):
        raise ValueError(f"Extraction configuration fingerprint mismatch in {path}")
    activations = payload.get("activations")
    if not isinstance(activations, torch.Tensor) or activations.ndim != 3:
        raise ValueError(f"Invalid activation tensor in {path}")
    expected_tensor_dtype = (
        torch.float16 if expected_dtype == "float16" else torch.float32
    )
    if activations.dtype != expected_tensor_dtype:
        raise ValueError(
            f"Activation dtype mismatch in {path}: {activations.dtype} "
            f"!= {expected_tensor_dtype}"
        )
    if tuple(activations.shape[1:]) != (num_layers, hidden_size):
        raise ValueError(
            f"Activation layer/hidden shape mismatch in {path}: {tuple(activations.shape)}"
        )
    if not bool(torch.isfinite(activations).all()):
        raise ValueError(f"Activation file contains NaN or infinity: {path}")
    count = int(activations.shape[0])
    list_fields = (
        "record_ids",
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
    for field in list_fields:
        value = payload.get(field)
        if not isinstance(value, list) or len(value) != count:
            raise ValueError(
                f"Metadata field {field!r} in {path} does not match activation count {count}"
            )
    activation_sum = payload.get("activation_sum_float64")
    if (
        not isinstance(activation_sum, torch.Tensor)
        or activation_sum.dtype != torch.float64
        or tuple(activation_sum.shape) != (num_layers, hidden_size)
        or not bool(torch.isfinite(activation_sum).all())
    ):
        raise ValueError(f"Invalid float64 activation sum in {path}")
    skipped = payload.get("skipped_short_records")
    if not isinstance(skipped, list):
        raise ValueError(f"Invalid skipped-short metadata in {path}")
    return payload


def validate_final_artifacts(
    *,
    output_dir: Path,
    requested_emotions: tuple[str, ...],
    emotion_counts: dict[str, int],
    emotion_sums: dict[str, torch.Tensor],
    num_layers: int,
    hidden_size: int,
) -> None:
    expected_shape = (num_layers, hidden_size)
    expected_keys = set(requested_emotions)
    means = load_torch_dictionary(output_dir / "emotion_means.pt")
    raw = load_torch_dictionary(output_dir / "emotion_vectors_raw.pt")
    unit = load_torch_dictionary(output_dir / "emotion_vectors_unit.pt")
    for artifact_name, payload in (
        ("emotion_means.pt", means),
        ("emotion_vectors_raw.pt", raw),
        ("emotion_vectors_unit.pt", unit),
    ):
        if set(payload) != expected_keys:
            raise ValueError(f"Unexpected emotion keys in {artifact_name}")
        for emotion in requested_emotions:
            tensor = payload[emotion]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != expected_shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(
                    f"Invalid {artifact_name} tensor for emotion {emotion}"
                )

    for emotion in requested_emotions:
        emotion_count = emotion_counts[emotion]
        total_count = sum(emotion_counts.values())
        other_count = total_count - emotion_count
        total_sum = sum(emotion_sums.values())
        expected_mean = (emotion_sums[emotion] / emotion_count).to(torch.float32)
        expected_raw = (
            emotion_sums[emotion] / emotion_count
            - (total_sum - emotion_sums[emotion]) / other_count
        ).to(torch.float32)
        expected_unit = expected_raw / expected_raw.norm(
            p=2, dim=-1, keepdim=True
        ).clamp_min(NORMALIZATION_EPSILON)
        if not torch.equal(means[emotion], expected_mean):
            raise ValueError(f"Saved emotion mean is incorrect for {emotion}")
        if not torch.equal(raw[emotion], expected_raw):
            raise ValueError(f"Saved raw emotion vector is incorrect for {emotion}")
        if not torch.equal(unit[emotion], expected_unit):
            raise ValueError(f"Saved unit emotion vector is incorrect for {emotion}")
        raw_norms = raw[emotion].norm(p=2, dim=-1)
        unit_norms = unit[emotion].norm(p=2, dim=-1)
        nonzero = raw_norms > NORMALIZATION_EPSILON
        if bool(torch.any(nonzero)) and not torch.allclose(
            unit_norms[nonzero],
            torch.ones_like(unit_norms[nonzero]),
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError(f"Invalid unit-vector norms for emotion {emotion}")

    stacked_path = output_dir / "emotion_vectors_stacked.pt"
    try:
        stacked_payload = torch.load(
            stacked_path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not load {stacked_path}: {error}") from error
    if not isinstance(stacked_payload, dict):
        raise ValueError(f"Invalid stacked-vector payload: {stacked_path}")
    if stacked_payload.get("emotion_order") != list(requested_emotions):
        raise ValueError("Stacked emotion order does not match the requested order")
    stacked = stacked_payload.get("vectors")
    expected_stacked_shape = (len(requested_emotions), num_layers, hidden_size)
    if (
        not isinstance(stacked, torch.Tensor)
        or stacked.dtype != torch.float32
        or tuple(stacked.shape) != expected_stacked_shape
        or not bool(torch.isfinite(stacked).all())
    ):
        raise ValueError("Invalid stacked emotion-vector tensor")
    expected_stacked = torch.stack(
        [raw[emotion] for emotion in requested_emotions], dim=0
    )
    if not torch.equal(stacked, expected_stacked):
        raise ValueError("Stacked vectors do not exactly match raw emotion vectors")

    saved_counts = read_json_object(output_dir / "emotion_counts.json")
    if saved_counts != emotion_counts:
        raise ValueError("Saved emotion counts do not match computed counts")
    norms_payload = read_json_object(output_dir / "emotion_vector_norms.json")
    if set(norms_payload) != expected_keys:
        raise ValueError("Saved emotion-vector norms have unexpected emotion keys")
    for emotion in requested_emotions:
        layer_norms = norms_payload[emotion]
        if not isinstance(layer_norms, dict) or set(layer_norms) != {
            str(layer) for layer in range(num_layers)
        }:
            raise ValueError(f"Invalid layer norm keys for emotion {emotion}")
        for layer in range(num_layers):
            expected_norm = float(raw[emotion][layer].norm(p=2).item())
            observed_norm = layer_norms[str(layer)]
            if not isinstance(observed_norm, (int, float)) or not torch.isclose(
                torch.tensor(float(observed_norm)),
                torch.tensor(expected_norm),
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError(
                    f"Saved norm mismatch for emotion {emotion}, layer {layer}"
                )


def load_torch_dictionary(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value


def build_extraction_signature(
    *,
    args: argparse.Namespace,
    requested_emotions: tuple[str, ...],
    input_dir: Path | None,
    input_file: Path | None,
    source_files: list[dict[str, Any]],
    model_bundle: dict[str, Any],
    num_layers: int,
    hidden_size: int,
) -> dict[str, Any]:
    return {
        "extraction_schema_version": 1,
        "input_mode": "directory" if input_dir is not None else "combined_file",
        "input_directory": str(input_dir) if input_dir is not None else None,
        "input_file": str(input_file) if input_file is not None else None,
        "source_files": source_files,
        "input_text_field": INPUT_TEXT_FIELD,
        "target_model": args.model,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": model_bundle["resolved_revision"],
        "tokenizer_name": args.model,
        "tokenizer_revision_resolved": model_bundle["tokenizer_revision"],
        "tokenizer_class": model_bundle["tokenizer_class"],
        "model_class": model_bundle["model_class"],
        "model_configuration_class": model_bundle["model_config_class"],
        "transformers_version": model_bundle["transformers_version"],
        "pytorch_version": torch.__version__,
        "requested_emotions": list(requested_emotions),
        "start_token_position": args.start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "chat_template_used": False,
        "add_special_tokens": True,
        "padding_side": "right",
        "truncation_side": "right",
        "truncation_enabled": args.max_sequence_length is not None,
        "maximum_sequence_length": args.max_sequence_length,
        "max_records_per_emotion": args.max_records_per_emotion,
        "model_dtype": args.dtype,
        "resolved_model_dtype": model_bundle["resolved_dtype"],
        "story_activation_dtype": args.story_activation_dtype,
        "embedding_hidden_state_included": False,
        "residual_stream_definition": RESIDUAL_STREAM_DEFINITION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "num_transformer_layers": num_layers,
        "hidden_size": hidden_size,
        "token_averaging_dtype": "float32",
        "layers_averaged_together": False,
    }


def prepare_config(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    signature: dict[str, Any],
    created_at: str,
    resume: bool,
) -> dict[str, Any]:
    path = output_dir / "config.json"
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
        if config.get("compatibility_signature") != signature:
            raise ValueError(
                "Requested extraction configuration does not match the existing run"
            )
        if config.get("compatibility_signature_sha256") != canonical_json_sha256(
            signature
        ):
            raise ValueError("Existing extraction configuration fingerprint mismatch")
        invocations = config.setdefault("invocations", [])
        if not isinstance(invocations, list):
            raise ValueError(f"Invalid invocations metadata in {path}")
        invocations.append(invocation)
        config["last_invocation_cli_arguments"] = json_safe_cli_arguments(args)
        atomic_write_json(path, config)
        return config

    config = {
        "input_directory": signature["input_directory"],
        "input_file": signature["input_file"],
        "model_name": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": signature["model_revision_resolved"],
        "tokenizer_name": args.model,
        "resolved_tokenizer_revision": signature["tokenizer_revision_resolved"],
        "selected_emotions": signature["requested_emotions"],
        "start_token_position": args.start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "batch_size": args.batch_size,
        "model_dtype": args.dtype,
        "story_activation_dtype": args.story_activation_dtype,
        "device_configuration": args.device,
        "truncation_enabled": args.max_sequence_length is not None,
        "maximum_sequence_length": args.max_sequence_length,
        "output_directory": str(output_dir),
        "complete_cli_arguments": json_safe_cli_arguments(args),
        "created_at": created_at,
        "compatibility_signature": signature,
        "compatibility_signature_sha256": canonical_json_sha256(signature),
        "invocations": [invocation],
    }
    atomic_write_json(path, config)
    return config


def prepare_metadata(
    *,
    output_dir: Path,
    signature: dict[str, Any],
    model_bundle: dict[str, Any],
    args: argparse.Namespace,
    loaded: LoadedRecords,
    extraction_start: str,
) -> dict[str, Any]:
    path = output_dir / "metadata.json"
    existing = read_json_object(path) if path.exists() else {}
    generator_models = {
        record.payload.get("generator_model")
        for records in loaded.by_emotion.values()
        for record in records
    }
    generator_models.discard(None)
    story_generator_model: str | list[str] | None
    if len(generator_models) == 1:
        story_generator_model = str(next(iter(generator_models)))
    elif generator_models:
        story_generator_model = sorted(str(value) for value in generator_models)
    else:
        story_generator_model = None

    metadata = {
        **existing,
        "status": "running",
        "target_model": args.model,
        "model_revision_requested": args.model_revision,
        "resolved_model_revision": model_bundle["resolved_revision"],
        "resolved_tokenizer_revision": model_bundle["tokenizer_revision"],
        "resolved_model_dtype": model_bundle["resolved_dtype"],
        "story_generator_model": story_generator_model,
        "same_generator_and_target_model": story_generator_model == args.model,
        "input_text_field": INPUT_TEXT_FIELD,
        "chat_template_used": False,
        "start_token_position": args.start_token_position,
        "token_position_indexing": TOKEN_POSITION_INDEXING,
        "anthropic_original_start_position": ANTHROPIC_ORIGINAL_START_POSITION,
        "method_difference": (
            "This experiment averages activations beginning at token position "
            f"{args.start_token_position} instead of position "
            f"{ANTHROPIC_ORIGINAL_START_POSITION}."
        ),
        "embedding_hidden_state_included": False,
        "residual_stream_definition": RESIDUAL_STREAM_DEFINITION,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "token_averaging_dtype": "float32",
        "story_activation_dtype": args.story_activation_dtype,
        "emotion_mean_dtype": "float32",
        "emotion_vector_dtype": "float32",
        "emotion_vector_definition": (
            "target-emotion mean at each layer minus the story-weighted mean of "
            "all other emotion stories at the same layer"
        ),
        "layers_averaged_together": False,
        "tokenizer_class": model_bundle["tokenizer_class"],
        "model_class": model_bundle["model_class"],
        "model_configuration_class": model_bundle["model_config_class"],
        "number_of_transformer_layers": signature["num_transformer_layers"],
        "hidden_size": signature["hidden_size"],
        "pytorch_version": torch.__version__,
        "transformers_version": model_bundle["transformers_version"],
        "cuda_version": torch.version.cuda,
        "gpu_name": model_bundle["gpu_name"],
        "total_input_records": loaded.total_input_records,
        "basic_valid_input_records": loaded.basic_valid_records,
        "valid_input_records": loaded.selected_valid_records,
        "malformed_records": len(loaded.invalid_records),
        "unrequested_valid_records": loaded.unrequested_valid_records,
        "records_excluded_by_limit": loaded.records_excluded_by_limit,
        "stories_shorter_than_start_position": existing.get(
            "stories_shorter_than_start_position", 0
        ),
        "valid_count_per_emotion": existing.get("valid_count_per_emotion", {}),
        "truncation_count": existing.get("truncation_count", 0),
        "extraction_start_timestamp": existing.get(
            "extraction_start_timestamp", extraction_start
        ),
        "extraction_end_timestamp": None,
        "extraction_configuration": signature,
        "extraction_configuration_sha256": canonical_json_sha256(signature),
    }
    return metadata


def build_progress(
    *,
    requested_emotions: tuple[str, ...],
    completed_emotions: Sequence[str],
    completed_payloads: dict[str, dict[str, Any]],
    status: str,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed_set = set(completed_emotions)
    completed = [emotion for emotion in requested_emotions if emotion in completed_set]
    progress: dict[str, Any] = {
        "status": status,
        "completed_emotions": completed,
        "remaining_emotions": [
            emotion for emotion in requested_emotions if emotion not in completed_set
        ],
        "valid_story_counts": {
            emotion: int(completed_payloads[emotion]["activations"].shape[0])
            for emotion in completed
        },
        "skipped_short_story_counts": {
            emotion: len(completed_payloads[emotion]["skipped_short_records"])
            for emotion in completed
        },
        "updated_at": utc_now(),
    }
    if failure is not None:
        progress["failure"] = failure
    return progress


def write_short_story_log(
    path: Path,
    payloads: dict[str, dict[str, Any]],
    requested_emotions: Sequence[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for emotion in requested_emotions:
        payload = payloads.get(emotion)
        if payload is not None:
            rows.extend(payload["skipped_short_records"])
    atomic_write_jsonl(path, rows)


def _prepare_output_directory(output_dir: Path, *, resume: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --resume or use a new path."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("emotion_activation_extraction")
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
    *, model: Any, tokenizer: Any, requested_revision: str | None
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
    *, tokenizer: Any, requested_revision: str | None
) -> str | None:
    candidates = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        requested_revision,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _canonical_requested_emotions(raw: Sequence[str]) -> tuple[str, ...]:
    if len(set(raw)) != len(raw):
        raise ValueError("--emotions cannot contain duplicates")
    requested = set(raw)
    return tuple(emotion for emotion in EMOTIONS if emotion in requested)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
