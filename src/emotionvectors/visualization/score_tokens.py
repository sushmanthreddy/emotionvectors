#!/usr/bin/env python3
"""Score six-sentence mixed-emotion passages with 12 clean emotion vectors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from ..constants import EMOTIONS, MODEL_ID, MODEL_REVISION


EXPECTED_MODEL = MODEL_ID
EXPECTED_REVISION = MODEL_REVISION

LOGGER = logging.getLogger("emotionvectors.token_scoring")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one layer's token activations for mixed-emotion "
            "passages and score every token against all clean emotion vectors."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--clean-unit-vectors", type=Path, required=True)
    parser.add_argument("--cleaning-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_REVISION)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-records", type=int, default=24)
    return parser.parse_args(argv)


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        output_dir / "token_scoring.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def sentence_spans(
    text: str,
    sentences: Sequence[str],
    expected: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    """Return exact half-open spans for sentences joined by one ASCII space."""
    if text != " ".join(sentences):
        raise ValueError("text must equal the sentences joined by one space")
    spans: list[dict[str, Any]] = []
    cursor = 0
    for sentence_index, (sentence, labels) in enumerate(zip(sentences, expected)):
        if sentence_index:
            if text[cursor : cursor + 1] != " ":
                raise ValueError("Unexpected sentence separator")
            cursor += 1
        start = cursor
        end = start + len(sentence)
        if text[start:end] != sentence:
            raise ValueError("Sentence span does not reproduce source text")
        spans.append(
            {
                "sentence_index": sentence_index,
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


def load_and_validate_records(
    path: Path,
    expected_records: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    primary_counts = {emotion: 0 for emotion in EMOTIONS}
    valid_emotions = set(EMOTIONS)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            example_id = record.get("example_id")
            primary = record.get("primary_emotion")
            secondary = record.get("secondary_emotion")
            sentences = record.get("sentences")
            expected = record.get("sentence_expected_emotions")
            text = record.get("text")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"Line {line_number} lacks example_id")
            if example_id in seen_ids:
                raise ValueError(f"Duplicate example_id {example_id}")
            if primary not in valid_emotions or secondary not in valid_emotions:
                raise ValueError(f"Invalid emotion label in {example_id}")
            if primary == secondary:
                raise ValueError(f"Primary and secondary match in {example_id}")
            if record.get("expected_emotions") != [primary, secondary]:
                raise ValueError(f"Incorrect expected_emotions in {example_id}")
            if not isinstance(sentences, list) or len(sentences) != 6:
                raise ValueError(f"{example_id} must have six sentences")
            if not all(isinstance(sentence, str) and sentence for sentence in sentences):
                raise ValueError(f"Invalid sentence in {example_id}")
            if not isinstance(expected, list) or len(expected) != len(sentences):
                raise ValueError(f"Invalid sentence annotations in {example_id}")
            pair_labels = {primary, secondary}
            if not all(
                isinstance(labels, list)
                and len(labels) == len(set(labels))
                and set(labels).issubset(pair_labels)
                for labels in expected
            ):
                raise ValueError(f"Invalid sentence emotion label in {example_id}")
            if not any(primary in labels for labels in expected):
                raise ValueError(f"No primary-positive sentence in {example_id}")
            if not any(secondary in labels for labels in expected):
                raise ValueError(f"No secondary-positive sentence in {example_id}")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Missing text in {example_id}")
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
    if any(count != 2 for count in primary_counts.values()):
        raise ValueError(
            f"Expected two examples per primary emotion: {primary_counts}"
        )
    return records


def load_clean_probes(path: Path, layer_index: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != set(EMOTIONS):
        raise ValueError("Clean unit-vector file does not contain all 12 emotions")
    vectors: list[torch.Tensor] = []
    for emotion in EMOTIONS:
        tensor = payload[emotion]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (28, 3584):
            raise ValueError(
                f"Unexpected clean vector shape for {emotion}: "
                f"{getattr(tensor, 'shape', None)}"
            )
        vector = tensor[layer_index].float().contiguous()
        if not torch.isfinite(vector).all():
            raise ValueError(f"Non-finite clean vector for {emotion}")
        norm = float(vector.norm().item())
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"Clean vector for {emotion} has norm {norm}")
        vectors.append(vector)
    return torch.stack(vectors, dim=0)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def validate_cleaning_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "completed",
        "target_model": EXPECTED_MODEL,
        "exact_model_revision": EXPECTED_REVISION,
        "resolved_tokenizer_revision": EXPECTED_REVISION,
        "tokenizer_class": "Qwen2TokenizerFast",
        "model_class": "Qwen2ForCausalLM",
        "number_of_layers": 28,
        "hidden_size": 3584,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": "saved layer l equals outputs.hidden_states[l + 1]",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Cleaning metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    return metadata


def model_input_device(model: torch.nn.Module) -> torch.device:
    device = model.get_input_embeddings().weight.device
    if device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise ValueError("Could not resolve a non-meta model input device")


def token_sentence_mapping(
    token_start: int,
    token_end: int,
    spans: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    """Map a half-open token span by maximum positive character overlap."""
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
            best_overlap = overlap
            best_index = int(sentence["sentence_index"])
    return best_index, best_overlap


def score_records(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required for token scoring; install the project "
            "runtime dependencies"
        ) from error

    configure_logging(args.output_dir)
    started_at = utc_now()
    if args.model != EXPECTED_MODEL:
        raise ValueError(f"Only {EXPECTED_MODEL} is supported")
    if args.model_revision != EXPECTED_REVISION:
        raise ValueError(f"Required model revision is {EXPECTED_REVISION}")
    validate_cleaning_metadata(args.cleaning_metadata)
    records = load_and_validate_records(args.input_jsonl, args.expected_records)
    if not 0 <= args.layer < 28:
        raise ValueError("--layer must be between 0 and 27")
    probes_cpu = load_clean_probes(args.clean_unit_vectors, args.layer)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for character offsets")
    if tokenizer.__class__.__name__ != "Qwen2TokenizerFast":
        raise ValueError(f"Unexpected tokenizer class {tokenizer.__class__.__name__}")
    resolved_tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    if (
        resolved_tokenizer_revision is not None
        and resolved_tokenizer_revision != EXPECTED_REVISION
    ):
        raise ValueError(
            f"Tokenizer revision mismatch: {resolved_tokenizer_revision}"
        )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device,
    )
    model.eval()
    model.config.use_cache = False
    if model.__class__.__name__ != "Qwen2ForCausalLM":
        raise ValueError(f"Unexpected model class {model.__class__.__name__}")
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if (
        resolved_model_revision is not None
        and resolved_model_revision != EXPECTED_REVISION
    ):
        raise ValueError(f"Model revision mismatch: {resolved_model_revision}")
    if model.config.num_hidden_layers != 28 or model.config.hidden_size != 3584:
        raise ValueError("Loaded model dimensions do not match Qwen2.5-7B-Instruct")
    input_device = model_input_device(model)

    output_path = args.output_dir / f"token_scores_layer_{args.layer}.jsonl"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_tokens = 0
    LOGGER.info(
        "scoring_started records=%d layer=%d batch_size=%d",
        len(records),
        args.layer,
        args.batch_size,
    )
    with temporary_path.open("w", encoding="utf-8") as output_handle:
        for batch_start in range(0, len(records), args.batch_size):
            batch_records = records[batch_start : batch_start + args.batch_size]
            texts = [record["text"] for record in batch_records]
            encoded = tokenizer(
                texts,
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
            try:
                with torch.inference_mode():
                    outputs = model(
                        **model_inputs,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    if len(outputs.hidden_states) != 29:
                        raise ValueError("Expected 29 hidden states")
                    hidden = outputs.hidden_states[args.layer + 1].float()
                    probes = probes_cpu.to(hidden.device)
                    score_tensor = torch.matmul(hidden, probes.T).cpu()
            except torch.cuda.OutOfMemoryError as exc:
                lengths = encoded["attention_mask"].sum(dim=1).tolist()
                LOGGER.error(
                    "cuda_oom example_ids=%s sequence_lengths=%s batch_size=%d",
                    [record["example_id"] for record in batch_records],
                    lengths,
                    args.batch_size,
                )
                raise RuntimeError(
                    "CUDA out of memory; rerun with a smaller --batch-size"
                ) from exc

            attention_mask = encoded["attention_mask"]
            input_ids = encoded["input_ids"]
            for batch_index, record in enumerate(batch_records):
                token_count = int(attention_mask[batch_index].sum().item())
                token_ids = input_ids[batch_index, :token_count].tolist()
                token_strings = tokenizer.convert_ids_to_tokens(token_ids)
                token_offsets = offsets[batch_index, :token_count].tolist()
                token_special = special_tokens[batch_index, :token_count].tolist()
                token_scores = score_tensor[batch_index, :token_count].tolist()
                mappings = [
                    token_sentence_mapping(start, end, record["sentence_spans"])
                    for start, end in token_offsets
                ]
                sentence_indices = [mapping[0] for mapping in mappings]
                overlap_chars = [mapping[1] for mapping in mappings]
                exclusion_reasons: dict[str, str] = {"0": "position_zero_artifact"}
                for token_index, ((start, end), sentence_index, is_special) in enumerate(
                    zip(token_offsets, sentence_indices, token_special)
                ):
                    if token_index == 0:
                        continue
                    if is_special:
                        exclusion_reasons[str(token_index)] = "special_token"
                    elif start >= end:
                        exclusion_reasons[str(token_index)] = "empty_character_span"
                    elif sentence_index < 0:
                        exclusion_reasons[str(token_index)] = "unmapped_to_sentence"
                if len(token_scores) != token_count or any(
                    len(scores) != len(EMOTIONS) for scores in token_scores
                ):
                    raise ValueError("Unexpected token-score tensor shape")
                if not all(
                    math.isfinite(float(score))
                    for scores in token_scores
                    for score in scores
                ):
                    raise ValueError(f"Non-finite score for {record['example_id']}")
                row = {
                    "schema_version": "long_mixed_emotion_token_scores_v1",
                    "example_id": record["example_id"],
                    "primary_emotion": record["primary_emotion"],
                    "secondary_emotion": record["secondary_emotion"],
                    "expected_emotions": record["expected_emotions"],
                    "text": record["text"],
                    "sentences": record["sentence_spans"],
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "resolved_model_revision": (
                        resolved_model_revision or args.model_revision
                    ),
                    "resolved_tokenizer_revision": (
                        resolved_tokenizer_revision or args.model_revision
                    ),
                    "layer_index_zero_based": args.layer,
                    "layer_number_one_based": args.layer + 1,
                    "hidden_state_mapping": (
                        "saved layer l equals outputs.hidden_states[l + 1]"
                    ),
                    "probe_order": list(EMOTIONS),
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
                output_handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                output_handle.flush()
                total_tokens += token_count
                LOGGER.info(
                    "example_scored example_id=%s tokens=%d completed=%d/%d",
                    record["example_id"],
                    token_count,
                    batch_start + batch_index + 1,
                    len(records),
                )

            del outputs, hidden, probes, score_tensor, model_inputs

        os.fsync(output_handle.fileno())
    temporary_path.replace(output_path)

    metadata = {
        "status": "completed",
        "created_at": utc_now(),
        "started_at": started_at,
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": sha256_file(args.input_jsonl),
        "clean_unit_vectors": str(args.clean_unit_vectors),
        "clean_unit_vectors_sha256": sha256_file(args.clean_unit_vectors),
        "cleaning_metadata": str(args.cleaning_metadata),
        "cleaning_metadata_sha256": sha256_file(args.cleaning_metadata),
        "model": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": resolved_model_revision or args.model_revision,
        "resolved_tokenizer_revision": (
            resolved_tokenizer_revision or args.model_revision
        ),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "model_dtype": str(resolve_dtype(args.dtype)),
        "chat_template_used": False,
        "input_text_handling": "plain text from record['text']",
        "layer_index_zero_based": args.layer,
        "layer_number_one_based": args.layer + 1,
        "embedding_hidden_state_included": False,
        "hidden_state_mapping": "saved layer l equals outputs.hidden_states[l + 1]",
        "probe_order": list(EMOTIONS),
        "records": len(records),
        "records_per_primary_emotion": 2,
        "sentences_per_record": 6,
        "total_scored_tokens": total_tokens,
        "position_zero_artifact_policy": (
            "Score is preserved, but position zero is excluded from calibration, "
            "highlighting, sentence metrics, and localization metrics."
        ),
        "output_jsonl": str(output_path),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    }
    write_json_atomic(args.output_dir / "token_scoring_metadata.json", metadata)
    LOGGER.info(
        "scoring_complete records=%d total_tokens=%d output=%s",
        len(records),
        total_tokens,
        output_path,
    )

    del model, probes_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    score_records(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
