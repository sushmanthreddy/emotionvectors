#!/usr/bin/env python3
"""Score plain-text probe passages with a selected subset of clean vectors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..extraction.story_raw_vectors import (
    atomic_write_json,
    resolve_hub_cache_dir,
)
from ..generation.emotion_config import load_emotion_config


SCORE_SCHEMA_VERSION = "anthropic_style_subset_token_scores_v1"
HIDDEN_STATE_MAPPING = "saved layer l equals outputs.hidden_states[l + 1]"
LOGGER = logging.getLogger("emotionvectors.subset_score")


def build_parser() -> argparse.ArgumentParser:
    """Build the model-agnostic subset token-scoring CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Score plain-text tokens at one transformer layer against only "
            "the requested clean unit emotion vectors."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--clean-unit-vectors", type=Path, required=True)
    parser.add_argument("--cleaning-metadata", type=Path, required=True)
    parser.add_argument("--activation-metadata", type=Path, required=True)
    parser.add_argument("--emotion-config", type=Path, required=True)
    parser.add_argument("--emotions", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--expected-records", type=_positive_int, required=True)
    parser.add_argument("--expected-layers", type=_positive_int, required=True)
    parser.add_argument(
        "--expected-hidden-size",
        type=_positive_int,
        required=True,
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subset probe-scoring job."""

    args = build_parser().parse_args(argv)
    try:
        score_subset(args)
    except Exception as error:
        print(
            f"Subset token scoring failed: {error.__class__.__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


def score_subset(args: argparse.Namespace) -> None:
    """Load one pinned model and score ``[tokens, selected_probes]``."""

    if not args.local_files_only:
        raise ValueError("--local-files-only is required for this workflow")
    if args.expected_records != len(args.emotions):
        raise ValueError(
            "--expected-records must equal the number of selected emotions "
            "because this smoke test requires one primary passage per probe"
        )
    if len(set(args.emotions)) != len(args.emotions):
        raise ValueError("--emotions contains duplicate labels")
    if not 0 <= args.layer < args.expected_layers:
        raise ValueError(
            f"--layer must be between 0 and {args.expected_layers - 1}"
        )

    paths = {
        name: value.expanduser().resolve()
        for name, value in (
            ("input_jsonl", args.input_jsonl),
            ("clean_unit_vectors", args.clean_unit_vectors),
            ("cleaning_metadata", args.cleaning_metadata),
            ("activation_metadata", args.activation_metadata),
            ("emotion_config", args.emotion_config),
            ("cache_dir", args.cache_dir),
            ("output_dir", args.output_dir),
        )
    }
    validate_new_output_directory(paths["output_dir"])
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    configure_logging(paths["output_dir"] / "subset_scoring.log")
    started_at = utc_now()

    emotion_specs = load_emotion_config(paths["emotion_config"])
    emotion_order = [spec.emotion for spec in emotion_specs]
    emotion_slugs = [spec.slug for spec in emotion_specs]
    if not set(args.emotions).issubset(emotion_order):
        invalid = sorted(set(args.emotions) - set(emotion_order))
        raise ValueError(f"Selected emotions are not configured: {invalid!r}")

    cleaning_metadata = read_json(paths["cleaning_metadata"])
    activation_metadata = read_json(paths["activation_metadata"])
    validate_metadata(
        cleaning_metadata=cleaning_metadata,
        activation_metadata=activation_metadata,
        model=args.model,
        model_revision=args.model_revision,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        emotion_order=emotion_order,
        emotion_slugs=emotion_slugs,
    )
    records = load_probe_records(
        paths["input_jsonl"],
        configured_emotions=set(emotion_order),
        selected_emotions=args.emotions,
        expected_records=args.expected_records,
    )
    probes_cpu = load_selected_probes(
        paths["clean_unit_vectors"],
        configured_emotions=emotion_order,
        selected_emotions=args.emotions,
        layer_index=args.layer,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
    )
    hub_cache = resolve_hub_cache_dir(
        cache_dir=paths["cache_dir"],
        model_name=args.model,
        model_revision=args.model_revision,
    )

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for token scoring") from error

    common_kwargs = {
        "revision": args.model_revision,
        "cache_dir": str(hub_cache),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common_kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for character offsets")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **common_kwargs,
        torch_dtype=model_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    if int(model.config.num_hidden_layers) != args.expected_layers:
        raise ValueError("Loaded model layer count is incompatible")
    if int(model.config.hidden_size) != args.expected_hidden_size:
        raise ValueError("Loaded model hidden size is incompatible")
    resolved_model_revision = (
        getattr(model.config, "_commit_hash", None) or args.model_revision
    )
    resolved_tokenizer_revision = (
        tokenizer.init_kwargs.get("_commit_hash") or args.model_revision
    )
    if resolved_model_revision != args.model_revision:
        raise ValueError("Loaded model revision differs from the pinned revision")
    if resolved_tokenizer_revision != args.model_revision:
        raise ValueError(
            "Loaded tokenizer revision differs from the pinned revision"
        )

    input_device = model_input_device(model)
    score_rows: list[dict[str, Any]] = []
    total_tokens = 0
    LOGGER.info(
        "scoring_started records=%d probes=%s layer=%d batch_size=%d",
        len(records),
        ",".join(args.emotions),
        args.layer,
        args.batch_size,
    )
    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        encoded = tokenizer(
            [record["text"] for record in batch_records],
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        offsets = encoded.pop("offset_mapping")
        special_tokens = encoded.pop("special_tokens_mask")
        model_inputs = {
            key: value.to(input_device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
            if len(outputs.hidden_states) != args.expected_layers + 1:
                raise ValueError(
                    "Hidden-state tuple does not contain embedding plus every "
                    "transformer layer"
                )
            hidden = outputs.hidden_states[args.layer + 1].float()
            if hidden.shape[-1] != args.expected_hidden_size:
                raise ValueError("Selected hidden state has the wrong size")
            probes = probes_cpu.to(hidden.device)
            scores = torch.matmul(hidden, probes.T).cpu()

        attention_mask = encoded["attention_mask"]
        input_ids = encoded["input_ids"]
        for batch_index, record in enumerate(batch_records):
            token_count = int(attention_mask[batch_index].sum().item())
            token_ids = input_ids[batch_index, :token_count].tolist()
            token_offsets = offsets[batch_index, :token_count].tolist()
            token_special = special_tokens[
                batch_index, :token_count
            ].tolist()
            token_scores = scores[batch_index, :token_count].tolist()
            token_strings = tokenizer.convert_ids_to_tokens(token_ids)
            mappings = [
                token_sentence_mapping(start, end, record["sentence_spans"])
                for start, end in token_offsets
            ]
            sentence_indices = [mapping[0] for mapping in mappings]
            overlap_chars = [mapping[1] for mapping in mappings]
            exclusion_reasons: dict[str, str] = {
                "0": "position_zero_artifact"
            }
            for token_index, (
                (start, end),
                sentence_index,
                is_special,
            ) in enumerate(
                zip(token_offsets, sentence_indices, token_special, strict=True)
            ):
                if token_index == 0:
                    continue
                if is_special:
                    exclusion_reasons[str(token_index)] = "special_token"
                elif start >= end:
                    exclusion_reasons[str(token_index)] = (
                        "empty_character_span"
                    )
                elif sentence_index < 0:
                    exclusion_reasons[str(token_index)] = (
                        "unmapped_to_sentence"
                    )
            if len(token_scores) != token_count or any(
                len(values) != len(args.emotions) for values in token_scores
            ):
                raise RuntimeError("Unexpected subset score tensor shape")
            if not all(
                math.isfinite(float(value))
                for values in token_scores
                for value in values
            ):
                raise ValueError(
                    f"Nonfinite score for {record['example_id']}"
                )
            score_rows.append(
                {
                    "schema_version": SCORE_SCHEMA_VERSION,
                    "example_id": record["example_id"],
                    "primary_emotion": record["primary_emotion"],
                    "secondary_emotion": record["secondary_emotion"],
                    "expected_emotions": record["expected_emotions"],
                    "text": record["text"],
                    "sentences": record["sentence_spans"],
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "resolved_model_revision": resolved_model_revision,
                    "resolved_tokenizer_revision": (
                        resolved_tokenizer_revision
                    ),
                    "layer_index_zero_based": args.layer,
                    "layer_number_one_based": args.layer + 1,
                    "hidden_state_mapping": HIDDEN_STATE_MAPPING,
                    "embedding_hidden_state_included": False,
                    "logits_used": False,
                    "chat_template_used": False,
                    "probe_order": list(args.emotions),
                    "token_ids": token_ids,
                    "token_strings": token_strings,
                    "token_offsets": token_offsets,
                    "token_sentence_indices": sentence_indices,
                    "token_sentence_overlap_chars": overlap_chars,
                    "probe_scores": token_scores,
                    "excluded_token_indices": sorted(
                        int(index) for index in exclusion_reasons
                    ),
                    "exclusion_reasons": exclusion_reasons,
                }
            )
            total_tokens += token_count
            LOGGER.info(
                "example_scored example_id=%s tokens=%d completed=%d/%d",
                record["example_id"],
                token_count,
                len(score_rows),
                len(records),
            )
        del outputs, hidden, probes, scores, model_inputs

    output_jsonl = (
        paths["output_dir"]
        / f"subset_token_scores_layer_{args.layer}.jsonl"
    )
    atomic_write_jsonl(output_jsonl, score_rows)
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "created_at": utc_now(),
        "started_at": started_at,
        "input_jsonl": str(paths["input_jsonl"]),
        "input_sha256": sha256_file(paths["input_jsonl"]),
        "clean_unit_vectors": str(paths["clean_unit_vectors"]),
        "clean_unit_vectors_sha256": sha256_file(
            paths["clean_unit_vectors"]
        ),
        "cleaning_metadata": str(paths["cleaning_metadata"]),
        "activation_metadata": str(paths["activation_metadata"]),
        "emotion_config": str(paths["emotion_config"]),
        "model": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": resolved_model_revision,
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "model_dtype": str(model_dtype),
        "input_text_field": "text",
        "plain_text_tokenization": True,
        "chat_template_used": False,
        "truncation_used": False,
        "add_special_tokens": True,
        "layer_index_zero_based": args.layer,
        "layer_number_one_based": args.layer + 1,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "probe_order": list(args.emotions),
        "number_of_probes": len(args.emotions),
        "records": len(records),
        "records_per_primary_emotion": 1,
        "total_scored_tokens": total_tokens,
        "token_score_definition": (
            "post-layer token activation dot clean unit emotion vector"
        ),
        "logits_used": False,
        "position_zero_policy": (
            "Score is preserved, but position zero is excluded from "
            "calibration and highlighting."
        ),
        "output_jsonl": str(output_jsonl),
        "output_sha256": sha256_file(output_jsonl),
        "local_files_only": True,
        "cache_dir": str(paths["cache_dir"]),
        "resolved_hub_cache_dir": str(hub_cache),
        "batch_size": args.batch_size,
        "gpu_name": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cli_arguments": json_safe_cli_arguments(args),
    }
    atomic_write_json(
        paths["output_dir"] / "subset_scoring_metadata.json",
        metadata,
    )
    LOGGER.info(
        "scoring_complete records=%d probes=%d output=%s",
        len(records),
        len(args.emotions),
        output_jsonl,
    )
    del model, tokenizer, probes_cpu
    gc.collect()
    torch.cuda.empty_cache()


def load_probe_records(
    path: Path,
    *,
    configured_emotions: set[str],
    selected_emotions: Sequence[str],
    expected_records: int,
) -> list[dict[str, Any]]:
    """Load one six-sentence primary passage for every selected probe."""

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    primary_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            example_id = record.get("example_id")
            primary = record.get("primary_emotion")
            secondary = record.get("secondary_emotion")
            sentences = record.get("sentences")
            expected = record.get("sentence_expected_emotions")
            text = record.get("text")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"Line {line_number} lacks example_id")
            if example_id in seen_ids:
                raise ValueError(f"Duplicate example_id {example_id!r}")
            if primary not in selected_emotions:
                raise ValueError(
                    f"Primary emotion {primary!r} is not a selected probe"
                )
            if (
                secondary not in configured_emotions
                or primary == secondary
                or record.get("expected_emotions") != [primary, secondary]
            ):
                raise ValueError(f"Invalid emotion pair in {example_id}")
            if (
                not isinstance(sentences, list)
                or len(sentences) != 6
                or not all(
                    isinstance(sentence, str) and sentence
                    for sentence in sentences
                )
                or text != " ".join(sentences)
            ):
                raise ValueError(
                    f"{example_id} must contain six joined sentences"
                )
            pair = {primary, secondary}
            if (
                not isinstance(expected, list)
                or len(expected) != 6
                or not all(
                    isinstance(labels, list)
                    and len(labels) == len(set(labels))
                    and set(labels).issubset(pair)
                    for labels in expected
                )
                or not any(primary in labels for labels in expected)
                or not any(secondary in labels for labels in expected)
            ):
                raise ValueError(
                    f"Invalid sentence annotations in {example_id}"
                )
            normalized = dict(record)
            normalized["sentence_spans"] = sentence_spans(
                text,
                sentences,
                expected,
            )
            records.append(normalized)
            seen_ids.add(example_id)
            primary_counts[primary] += 1
    if len(records) != expected_records:
        raise ValueError(
            f"Expected {expected_records} records, found {len(records)}"
        )
    expected_counts = {emotion: 1 for emotion in selected_emotions}
    if dict(primary_counts) != expected_counts:
        raise ValueError(
            "Expected one record per selected primary emotion: "
            f"{dict(primary_counts)!r}"
        )
    return records


def sentence_spans(
    text: str,
    sentences: Sequence[str],
    expected: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    """Return exact half-open spans for sentences joined by one space."""

    spans: list[dict[str, Any]] = []
    cursor = 0
    for index, (sentence, labels) in enumerate(
        zip(sentences, expected, strict=True)
    ):
        if index:
            cursor += 1
        start = cursor
        end = start + len(sentence)
        if text[start:end] != sentence:
            raise ValueError("Sentence span does not reproduce source text")
        spans.append(
            {
                "sentence_index": index,
                "text": sentence,
                "char_start": start,
                "char_end": end,
                "expected_emotions": list(labels),
            }
        )
        cursor = end
    if cursor != len(text):
        raise ValueError("Sentence spans do not cover the complete text")
    return spans


def load_selected_probes(
    path: Path,
    *,
    configured_emotions: Sequence[str],
    selected_emotions: Sequence[str],
    layer_index: int,
    expected_layers: int,
    expected_hidden_size: int,
) -> torch.Tensor:
    """Load only requested rows into ``[selected_probes, hidden]``."""

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("Clean unit-vector file must contain a dictionary")
    if set(payload) != set(configured_emotions):
        raise ValueError(
            "Clean unit-vector labels differ from the emotion configuration"
        )
    selected: list[torch.Tensor] = []
    for emotion in selected_emotions:
        tensor = payload[emotion]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape)
            != (expected_layers, expected_hidden_size)
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"Clean unit vectors for {emotion!r} are incompatible"
            )
        probe = tensor[layer_index].float().contiguous()
        norm = float(probe.norm().item())
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"Selected vector {emotion!r} has layer norm {norm}"
            )
        selected.append(probe)
    probes = torch.stack(selected, dim=0)
    if tuple(probes.shape) != (
        len(selected_emotions),
        expected_hidden_size,
    ):
        raise RuntimeError("Selected probe stack has the wrong shape")
    return probes


def validate_metadata(
    *,
    cleaning_metadata: Mapping[str, Any],
    activation_metadata: Mapping[str, Any],
    model: str,
    model_revision: str,
    expected_layers: int,
    expected_hidden_size: int,
    emotion_order: Sequence[str],
    emotion_slugs: Sequence[str],
) -> None:
    """Reject cross-model, cross-revision, or cross-layer probe inputs."""

    for name, metadata in (
        ("cleaning", cleaning_metadata),
        ("activation", activation_metadata),
    ):
        if metadata.get("model") != model:
            raise ValueError(f"{name} metadata model is incompatible")
        if metadata.get("model_revision") != model_revision:
            raise ValueError(f"{name} metadata revision is incompatible")
        if metadata.get("number_of_layers") != expected_layers:
            raise ValueError(f"{name} metadata layer count is incompatible")
        if metadata.get("hidden_size") != expected_hidden_size:
            raise ValueError(f"{name} metadata hidden size is incompatible")
    if cleaning_metadata.get("status") != "completed":
        raise ValueError("Cleaning metadata is not completed")
    if cleaning_metadata.get("emotion_order") != list(emotion_order):
        raise ValueError("Cleaning metadata emotion order differs")
    if cleaning_metadata.get("emotion_slugs") != list(emotion_slugs):
        raise ValueError("Cleaning metadata emotion slugs differ")
    if cleaning_metadata.get("clean_vector_shape_per_emotion") != [
        expected_layers,
        expected_hidden_size,
    ]:
        raise ValueError("Cleaning metadata vector shape differs")
    if cleaning_metadata.get("layers_averaged_together") is not False:
        raise ValueError("Clean vectors report averaged layers")
    if activation_metadata.get("hidden_state_mapping") != HIDDEN_STATE_MAPPING:
        raise ValueError("Activation layer definition is incompatible")
    if (
        activation_metadata.get("embedding_hidden_state_included")
        is not False
    ):
        raise ValueError("Activation metadata includes embedding state")
    if activation_metadata.get("layers_averaged_together") is not False:
        raise ValueError("Activation metadata reports averaged layers")


def token_sentence_mapping(
    token_start: int,
    token_end: int,
    spans: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Map one token span to the sentence with maximum character overlap."""

    if token_start >= token_end:
        return -1, 0
    best_index = -1
    best_overlap = 0
    for sentence in spans:
        overlap = max(
            0,
            min(token_end, int(sentence["char_end"]))
            - max(token_start, int(sentence["char_start"])),
        )
        if overlap > best_overlap:
            best_index = int(sentence["sentence_index"])
            best_overlap = overlap
    return best_index, best_overlap


def model_input_device(model: torch.nn.Module) -> torch.device:
    """Return the concrete device holding the input embedding table."""

    device = model.get_input_embeddings().weight.device
    if device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise ValueError("Could not resolve a non-meta model input device")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL through a same-directory temporary file and rename."""

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
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def configure_logging(path: Path) -> None:
    """Write concise progress to stdout and the run log."""

    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def validate_new_output_directory(path: Path) -> None:
    """Require a new or empty output directory."""

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe_cli_arguments(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value.expanduser().resolve())
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


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


if __name__ == "__main__":
    raise SystemExit(main())
