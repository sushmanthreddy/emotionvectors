"""Command-line orchestration for independent emotional-story generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..constants import MODEL_ID, MODEL_REVISION
from .emotion_config import (
    EmotionSpec,
    build_emotion_specs,
    load_emotion_config,
)
from .model import (
    GenerationParameters,
    LocalHuggingFaceGenerator,
    clean_generation,
    contains_exact_emotion_word,
    contains_non_latin_letter,
)
from .story_prompt import PROMPT_VERSION, STORY_PROMPT_TEMPLATE, render_story_prompt
from .seeds import create_seed
from .storage import AttemptProgress, DatasetStorage, RecordKey

ALL_EMOTIONS = (
    "anger",
    "fear",
    "disgust",
    "sadness",
    "anxiety",
    "desperation",
    "frustration",
    "hostility",
    "calmness",
    "compassion",
    "joy",
    "trust",
)
DEFAULT_EMOTION_SPECS = tuple(
    EmotionSpec(emotion=emotion) for emotion in ALL_EMOTIONS
)
DEFAULT_MODEL = MODEL_ID
DEFAULT_TOPICS_PATH = Path("config/topics.json")
MAX_TOTAL_ATTEMPTS = 5
ATTEMPT_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "emotion",
        "topic_id",
        "topic",
        "sample_index",
        "attempt_number",
        "seed",
        "prompt_version",
        "prompt",
        "generator_model",
        "model_revision",
        "raw_completion",
        "story",
        "emotion_word_present",
        "non_latin_letter_present",
        "generation_error",
        "generation_batch_size",
        "created_at",
    }
)
ACCEPTED_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "emotion",
        "topic_id",
        "topic",
        "sample_index",
        "prompt_version",
        "prompt",
        "raw_completion",
        "story",
        "generator_model",
        "model_revision",
        "generation_parameters",
        "generation_batch_size",
        "accepted_seed",
        "attempt_count",
        "emotion_word_present",
        "non_latin_letter_present",
        "accepted_after_max_attempts",
        "created_at",
    }
)


@dataclass(frozen=True)
class Topic:
    topic_id: int
    topic: str


@dataclass(frozen=True)
class GenerationJob:
    emotion: str
    emotion_slug: str
    topic: Topic
    sample_index: int

    @property
    def key(self) -> RecordKey:
        return self.emotion, self.topic.topic_id, self.sample_index

    @property
    def record_id(self) -> str:
        return (
            f"{self.emotion_slug}_topic_{self.topic.topic_id:03d}_"
            f"sample_{self.sample_index:02d}"
        )


@dataclass(frozen=True)
class AttemptRequest:
    job: GenerationJob
    prompt: str
    state: AttemptProgress
    attempt_number: int
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save independent emotional stories with a local model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or path")
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Pinned Hugging Face commit for the model and tokenizer. "
            "The released 7B model defaults to its pinned revision; other models "
            "should pass an immutable revision explicitly."
        ),
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=DEFAULT_TOPICS_PATH,
        help="JSON file containing stable topic_id/topic objects",
    )
    parser.add_argument(
        "--expected-topics-sha256",
        default=None,
        help="Optional lowercase SHA-256 guard for the exact topics file bytes",
    )
    emotion_source = parser.add_mutually_exclusive_group()
    emotion_source.add_argument(
        "--emotions",
        nargs="+",
        default=None,
        help=(
            "Exact ordered emotion labels for an ad-hoc run; defaults to the "
            "released twelve-emotion set."
        ),
    )
    emotion_source.add_argument(
        "--emotion-config",
        type=Path,
        default=None,
        help="Flat emotion JSON config; preserves emotion order",
    )
    parser.add_argument(
        "--expected-emotion-config-sha256",
        default=None,
        help="Optional lowercase SHA-256 guard for the flat emotion config bytes",
    )
    parser.add_argument(
        "--max-topics",
        type=_positive_int,
        default=None,
        help="Use only the first N topics",
    )
    parser.add_argument("--samples-per-pair", type=_positive_int, default=12)
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Execution-only prompt batch size; does not alter the dataset identity",
    )
    parser.add_argument("--max-attempts", type=_attempt_count, default=5)
    parser.add_argument("--temperature", type=_positive_float, default=0.9)
    parser.add_argument("--top-p", type=_top_p, default=0.95)
    parser.add_argument(
        "--top-k",
        type=_nonnegative_int,
        default=20,
        help="Explicit sampling cutoff; zero disables top-k",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=_positive_float,
        default=1.05,
        help="Explicit Transformers-compatible repetition penalty",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=512)
    parser.add_argument("--base-seed", type=int, default=20260719)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/emotional_stories")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Optional Transformers attention backend",
    )
    parser.add_argument(
        "--prompt-role",
        default="system",
        choices=("system", "user"),
        help="Chat-template role that receives the story prompt",
    )
    parser.add_argument(
        "--generation-request",
        default="Generate the story now.",
        help="Follow-up request used with chat templates",
    )
    parser.add_argument(
        "--eos-token-policy",
        choices=("model", "tokenizer"),
        default="tokenizer",
        help="Use the model generation-config stop IDs or tokenizer EOS only",
    )
    parser.add_argument(
        "--chat-system-instruction",
        default=None,
        help="Optional system instruction when --prompt-role=user",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow model/tokenizer repositories to execute custom loading code",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forbid network access while loading the model and tokenizer",
    )
    parser.add_argument(
        "--require-model-revision",
        action="store_true",
        help="Fail preflight unless an immutable-looking 40-hex revision is supplied",
    )
    parser.add_argument(
        "--expected-emotions",
        type=_positive_int,
        default=None,
        help="Pre-model-load guard for the exact emotion count",
    )
    parser.add_argument(
        "--expected-topics",
        type=_positive_int,
        default=None,
        help="Pre-model-load guard for the exact topic count",
    )
    parser.add_argument(
        "--expected-stories",
        type=_positive_int,
        default=None,
        help="Pre-model-load guard for emotions × topics × samples",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and print the run plan without writing outputs or loading a model",
    )
    run_mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Require exact completed output coverage without loading a model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.generation_request.strip():
        parser.error("--generation-request cannot be empty")
    if (
        args.chat_system_instruction is not None
        and not args.chat_system_instruction.strip()
    ):
        parser.error("--chat-system-instruction cannot be empty when supplied")
    model_revision = args.model_revision
    if model_revision is None and args.model == DEFAULT_MODEL:
        model_revision = MODEL_REVISION
    if args.require_model_revision and not _is_immutable_revision(model_revision):
        parser.error(
            "--require-model-revision needs --model-revision to be a 40-character "
            "hexadecimal commit"
        )

    try:
        if args.emotion_config is not None:
            emotions_path = args.emotion_config.expanduser().resolve()
            emotion_config_sha256 = _sha256_file(emotions_path)
            emotion_specs = load_emotion_config(emotions_path)
        elif args.emotions is not None:
            emotions_path = None
            emotion_config_sha256 = None
            emotion_specs = build_emotion_specs(args.emotions)
        else:
            emotions_path = None
            emotion_config_sha256 = None
            emotion_specs = DEFAULT_EMOTION_SPECS
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    emotions = tuple(spec.emotion for spec in emotion_specs)
    if args.expected_emotion_config_sha256 is not None:
        if emotion_config_sha256 is None:
            parser.error(
                "--expected-emotion-config-sha256 requires --emotion-config"
            )
        _validate_sha256_expectation(
            parser=parser,
            option="--expected-emotion-config-sha256",
            expected=args.expected_emotion_config_sha256,
            actual=emotion_config_sha256,
        )

    try:
        topics_path = args.topics.expanduser().resolve()
        topics_sha256 = _sha256_file(topics_path)
        topics = load_topics(topics_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.max_topics is not None:
        topics = topics[: args.max_topics]
    if not topics:
        parser.error("At least one topic is required")
    if args.expected_topics_sha256 is not None:
        _validate_sha256_expectation(
            parser=parser,
            option="--expected-topics-sha256",
            expected=args.expected_topics_sha256,
            actual=topics_sha256,
        )

    intended_stories = len(emotions) * len(topics) * args.samples_per_pair
    count_expectations = (
        ("--expected-emotions", args.expected_emotions, len(emotions)),
        ("--expected-topics", args.expected_topics, len(topics)),
        ("--expected-stories", args.expected_stories, intended_stories),
    )
    for option, expected, actual in count_expectations:
        if expected is not None and expected != actual:
            parser.error(f"{option} expected {expected}, but the resolved run has {actual}")

    parameters = GenerationParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
    )
    output_dir = args.output_dir.expanduser().resolve()
    created_at = utc_now()
    run_config = _build_run_config(
        emotion_specs=emotion_specs,
        emotion_config_path=emotions_path,
        emotion_config_sha256=emotion_config_sha256,
        topics=topics,
        topics_path=topics_path,
        topics_sha256=topics_sha256,
        model_name=args.model,
        model_revision=model_revision,
        samples_per_pair=args.samples_per_pair,
        max_attempts=args.max_attempts,
        base_seed=args.base_seed,
        parameters=parameters,
        device=args.device,
        dtype=args.dtype,
        output_dir=output_dir,
        created_at=created_at,
        attn_implementation=args.attn_implementation,
        prompt_role=args.prompt_role,
        generation_request=args.generation_request,
        chat_system_instruction=args.chat_system_instruction,
        eos_token_policy=args.eos_token_policy,
        trust_remote_code=args.trust_remote_code,
    )
    jobs = _create_jobs(
        emotion_specs=emotion_specs,
        topics=topics,
        samples_per_pair=args.samples_per_pair,
        base_seed=args.base_seed,
    )
    intended_keys = {job.key for job in jobs}
    if len(intended_keys) != intended_stories:
        parser.error(
            "Internal key collision while constructing the requested story dataset"
        )

    if args.preflight_only:
        try:
            output_state = _preflight_output_state(
                output_dir=output_dir,
                resume=args.resume,
                run_config=run_config,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "generator_model": args.model,
                    "model_revision": model_revision,
                    "number_of_emotions": len(emotions),
                    "number_of_topics": len(topics),
                    "samples_per_pair": args.samples_per_pair,
                    "intended_stories": intended_stories,
                    "emotions": list(emotions),
                    "output_dir": str(output_dir),
                    "output_state": output_state,
                    "model_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    storage: DatasetStorage | None = None
    if args.validate_only:
        try:
            storage = DatasetStorage(output_dir, resume=True)
            persisted_config = storage.validate_run_config(run_config)
            completed_index = storage.load_completed_index(repair=False)
            attempt_progress, total_attempts = storage.load_attempt_progress(
                completed_index.keys,
                repair=False,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        try:
            storage = DatasetStorage(output_dir, resume=args.resume)
            persisted_config = storage.prepare_run_config(run_config)
            storage.initialize_layout(emotions)
            completed_index = storage.load_completed_index()
            attempt_progress, total_attempts = storage.load_attempt_progress(
                completed_index.keys
            )
        except KeyboardInterrupt:
            interrupted_storage = storage or DatasetStorage(output_dir, resume=True)
            _mark_startup_interrupted(
                storage=interrupted_storage,
                model_name=args.model,
                base_seed=args.base_seed,
                samples_per_pair=args.samples_per_pair,
                max_attempts=args.max_attempts,
                emotions=emotions,
                topics=topics,
                parameters=parameters,
                created_at=created_at,
            )
            print("Generation interrupted during resume preparation", file=sys.stderr)
            return 130
        except (OSError, ValueError) as error:
            parser.error(str(error))

    try:
        unexpected_completed = completed_index.keys - intended_keys
        if unexpected_completed:
            parser.error(
                "Existing accepted records do not belong to this run configuration: "
                f"{sorted(unexpected_completed)[:3]!r}"
            )
        unexpected_attempts = set(attempt_progress) - intended_keys
        if unexpected_attempts:
            parser.error(
                "Existing attempt records do not belong to this run configuration: "
                f"{sorted(unexpected_attempts)[:3]!r}"
            )
        accepted_records = _validate_accepted_records(
            storage=storage,
            jobs=jobs,
            model_name=args.model,
            model_revision=model_revision,
            parameters=parameters,
            base_seed=args.base_seed,
            max_attempts=args.max_attempts,
        )
        _validate_attempt_records(
            storage=storage,
            jobs=jobs,
            accepted_records=accepted_records,
            model_name=args.model,
            model_revision=model_revision,
            base_seed=args.base_seed,
            max_attempts=args.max_attempts,
        )

        completed_keys = set(completed_index.keys)
        effective_batch_size = args.batch_size
        if args.resume and storage.manifest_path.exists():
            previous_manifest = storage.read_manifest()
            previous_effective = previous_manifest.get("effective_batch_size")
            if (
                isinstance(previous_effective, int)
                and not isinstance(previous_effective, bool)
                and previous_effective > 0
            ):
                effective_batch_size = min(effective_batch_size, previous_effective)
        failed_keys = {
            key
            for key, state in attempt_progress.items()
            if key in intended_keys
            and state.max_attempt_number >= args.max_attempts
            and state.latest_script_compliant is None
        }
        manifest = _build_manifest(
            model_name=args.model,
            base_seed=args.base_seed,
            samples_per_pair=args.samples_per_pair,
            max_attempts=args.max_attempts,
            emotions=emotions,
            topics=topics,
            intended_stories=len(jobs),
            accepted_stories=len(completed_keys),
            accepted_after_max_attempts=completed_index.accepted_after_max_attempts,
            failed_stories=len(failed_keys),
            total_attempts=total_attempts,
            parameters=parameters,
            created_at=str(persisted_config["created_at"]),
        )
        manifest["execution_batch_size"] = args.batch_size
        manifest["effective_batch_size"] = effective_batch_size
        manifest["model_revision"] = model_revision
    except KeyboardInterrupt:
        _mark_startup_interrupted(
            storage=storage,
            model_name=args.model,
            base_seed=args.base_seed,
            samples_per_pair=args.samples_per_pair,
            max_attempts=args.max_attempts,
            emotions=emotions,
            topics=topics,
            parameters=parameters,
            created_at=str(persisted_config["created_at"]),
        )
        print("Generation interrupted during job preparation", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.validate_only:
        missing_keys = intended_keys - completed_keys
        validation_errors: list[str] = []
        if missing_keys:
            validation_errors.append(f"{len(missing_keys)} accepted records are missing")
        if failed_keys:
            validation_errors.append(f"{len(failed_keys)} records are terminal failures")
        try:
            persisted_manifest = storage.read_manifest()
        except (OSError, ValueError) as error:
            validation_errors.append(str(error))
            persisted_manifest = {}
        if persisted_manifest.get("status") != "completed":
            validation_errors.append(
                "manifest status is not 'completed': "
                f"{persisted_manifest.get('status')!r}"
            )
        expected_manifest_fields: dict[str, Any] = {
            "generator_model": args.model,
            "model_revision": model_revision,
            "prompt_version": PROMPT_VERSION,
            "base_seed": args.base_seed,
            "samples_per_pair": args.samples_per_pair,
            "max_attempts": args.max_attempts,
            "number_of_emotions": len(emotions),
            "number_of_topics": len(topics),
            "intended_stories": intended_stories,
            "accepted_stories": len(completed_keys),
            "accepted_after_max_attempts": (
                completed_index.accepted_after_max_attempts
            ),
            "failed_stories": len(failed_keys),
            "total_generation_attempts": total_attempts,
            "generation_parameters": parameters.as_dict(),
        }
        for field, expected in expected_manifest_fields.items():
            if persisted_manifest.get(field) != expected:
                validation_errors.append(
                    f"manifest {field} is {persisted_manifest.get(field)!r}, "
                    f"expected {expected!r}"
                )
        if validation_errors:
            print(
                json.dumps(
                    {
                        "status": "validation_failed",
                        "errors": validation_errors,
                        "accepted_stories": len(completed_keys),
                        "intended_stories": intended_stories,
                        "model_loaded": False,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
        print(
            json.dumps(
                {
                    "status": "validation_passed",
                    "accepted_stories": len(completed_keys),
                    "intended_stories": intended_stories,
                    "failed_stories": 0,
                    "model_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    try:
        storage.write_manifest(manifest)
        logger = configure_logging(storage.log_path)
    except KeyboardInterrupt:
        _update_manifest(storage, manifest, status="interrupted")
        print("Generation interrupted during startup", file=sys.stderr)
        return 130
    except OSError as error:
        try:
            _update_manifest(storage, manifest, status="failed")
        except OSError:
            pass
        print(f"Could not initialize generation output: {error}", file=sys.stderr)
        return 1

    try:
        generator: LocalHuggingFaceGenerator | None = None
        if _generation_calls_remain(
            jobs=jobs,
            completed_keys=completed_keys,
            failed_keys=failed_keys,
            attempt_progress=attempt_progress,
            max_attempts=args.max_attempts,
        ):
            generator = LocalHuggingFaceGenerator(
                model_name=args.model,
                device=args.device,
                dtype=args.dtype,
                model_revision=model_revision,
                attn_implementation=args.attn_implementation,
                prompt_role=args.prompt_role,
                generation_request=args.generation_request,
                chat_system_instruction=args.chat_system_instruction,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                use_model_eos_tokens=args.eos_token_policy == "model",
            )
        _run_jobs(
            jobs=jobs,
            generator=generator,
            storage=storage,
            manifest=manifest,
            completed_keys=completed_keys,
            failed_keys=failed_keys,
            attempt_progress=attempt_progress,
            base_seed=args.base_seed,
            max_attempts=args.max_attempts,
            parameters=parameters,
            model_name=args.model,
            model_revision=model_revision,
            logger=logger,
            batch_size=effective_batch_size,
        )
        terminal_keys = completed_keys | failed_keys
        if terminal_keys != intended_keys:
            missing_terminal = intended_keys - terminal_keys
            raise RuntimeError(
                "Generation returned with non-terminal jobs: "
                f"count={len(missing_terminal)} sample={sorted(missing_terminal)[:3]!r}"
            )
    except KeyboardInterrupt:
        _reconcile_manifest(storage, manifest, failed_keys, intended_keys)
        _update_manifest(storage, manifest, status="interrupted")
        logger.warning("Generation interrupted; manifest updated for resume")
        return 130
    except Exception:
        _update_manifest(storage, manifest, status="failed")
        logger.exception("Generation stopped because of an unrecoverable error")
        return 1

    final_status = "completed_with_failures" if failed_keys else "completed"
    _update_manifest(storage, manifest, status=final_status)
    logger.info(
        "generation_complete accepted=%d intended=%d failed=%d attempts=%d",
        manifest["accepted_stories"],
        manifest["intended_stories"],
        manifest["failed_stories"],
        manifest["total_generation_attempts"],
    )
    return 1 if failed_keys else 0


def _run_jobs(
    *,
    jobs: list[GenerationJob],
    generator: LocalHuggingFaceGenerator | None,
    storage: DatasetStorage,
    manifest: dict[str, Any],
    completed_keys: set[RecordKey],
    failed_keys: set[RecordKey],
    attempt_progress: dict[RecordKey, AttemptProgress],
    base_seed: int,
    max_attempts: int,
    parameters: GenerationParameters,
    model_name: str,
    model_revision: str | None,
    logger: logging.Logger,
    batch_size: int,
) -> None:
    pending: deque[GenerationJob] = deque()
    reconciled_records = 0

    for job in jobs:
        if job.key in completed_keys or job.key in failed_keys:
            continue

        state = attempt_progress.setdefault(job.key, AttemptProgress())
        if state.max_attempt_number > max_attempts:
            raise ValueError(
                f"Stored attempt count exceeds --max-attempts for {job.record_id}"
            )

        selected = state.latest_nonempty
        if (
            selected is not None
            and not bool(selected.get("emotion_word_present", True))
            and selected.get("non_latin_letter_present") is False
        ):
            _accept_selected_story(
                job=job,
                state=state,
                selected=selected,
                storage=storage,
                manifest=manifest,
                completed_keys=completed_keys,
                failed_keys=failed_keys,
                max_attempts=max_attempts,
                parameters=parameters,
                model_name=model_name,
                model_revision=model_revision,
            )
            reconciled_records += 1
            continue

        if state.max_attempt_number >= max_attempts:
            selected = state.latest_script_compliant
            if selected is None:
                _mark_job_failed(
                    job=job,
                    failed_keys=failed_keys,
                    manifest=manifest,
                    logger=logger,
                    max_attempts=max_attempts,
                )
            else:
                _accept_selected_story(
                    job=job,
                    state=state,
                    selected=selected,
                    storage=storage,
                    manifest=manifest,
                    completed_keys=completed_keys,
                    failed_keys=failed_keys,
                    max_attempts=max_attempts,
                    parameters=parameters,
                    model_name=model_name,
                    model_revision=model_revision,
                )
            reconciled_records += 1
            continue

        pending.append(job)

    if reconciled_records:
        _update_manifest(storage, manifest)
        logger.info("resume_reconciled records=%d", reconciled_records)

    if pending and generator is None:
        raise RuntimeError("Pending generation jobs have no loaded model")

    batch_number = 0
    effective_batch_size = batch_size
    manifest["effective_batch_size"] = effective_batch_size
    while pending:
        requests: list[AttemptRequest] = []
        while pending and len(requests) < effective_batch_size:
            job = pending.popleft()
            state = attempt_progress.setdefault(job.key, AttemptProgress())
            attempt_number = state.max_attempt_number + 1
            if attempt_number > max_attempts:
                raise RuntimeError(
                    f"No attempts remain for pending record {job.record_id}"
                )
            prompt = render_story_prompt(topic=job.topic.topic, emotion=job.emotion)
            seed = create_seed(
                base_seed=base_seed,
                emotion=job.emotion,
                topic_id=job.topic.topic_id,
                sample_index=job.sample_index,
                attempt_number=attempt_number,
            )
            requests.append(
                AttemptRequest(
                    job=job,
                    prompt=prompt,
                    state=state,
                    attempt_number=attempt_number,
                    seed=seed,
                )
            )

        batch_number += 1
        logger.info(
            "batch_start batch=%d size=%d accepted=%d intended=%d pending=%d",
            batch_number,
            len(requests),
            len(completed_keys),
            manifest["intended_stories"],
            len(pending),
        )
        assert generator is not None
        batch_results, safe_batch_size, batch_was_split = _generate_requests_resilient(
            generator=generator,
            requests=requests,
            parameters=parameters,
            logger=logger,
        )
        if batch_was_split and safe_batch_size < effective_batch_size:
            previous_batch_size = effective_batch_size
            effective_batch_size = safe_batch_size
            manifest["effective_batch_size"] = effective_batch_size
            logger.warning(
                "batch_size_downshift previous=%d effective=%d",
                previous_batch_size,
                effective_batch_size,
            )
            _update_manifest(storage, manifest)

        for request, (raw_completion, generation_error, generation_batch_size) in zip(
            requests,
            batch_results,
            strict=True,
        ):
            job = request.job
            if generation_error is not None or raw_completion is None:
                story = None
                word_present = None
                non_latin_letter_present = None
            else:
                try:
                    story = clean_generation(raw_completion)
                    word_present = contains_exact_emotion_word(
                        generated_text=story,
                        emotion=job.emotion,
                    )
                    non_latin_letter_present = contains_non_latin_letter(story)
                except Exception as error:
                    story = None
                    word_present = None
                    non_latin_letter_present = None
                    generation_error = str(error)

            if generation_error is not None:
                logger.warning(
                    "attempt_failed record_id=%s attempt=%d error=%s",
                    job.record_id,
                    request.attempt_number,
                    generation_error,
                )

            attempt_record = {
                "record_id": job.record_id,
                "emotion": job.emotion,
                "topic_id": job.topic.topic_id,
                "topic": job.topic.topic,
                "sample_index": job.sample_index,
                "attempt_number": request.attempt_number,
                "seed": request.seed,
                "prompt_version": PROMPT_VERSION,
                "prompt": request.prompt,
                "generator_model": model_name,
                "model_revision": model_revision,
                "raw_completion": raw_completion,
                "story": story,
                "emotion_word_present": word_present,
                "non_latin_letter_present": non_latin_letter_present,
                "generation_error": generation_error,
                "generation_batch_size": generation_batch_size,
                "created_at": utc_now(),
            }
            storage.append_attempt(attempt_record)
            request.state.max_attempt_number = request.attempt_number
            manifest["total_generation_attempts"] += 1

            if generation_error is None and story:
                request.state.latest_nonempty = attempt_record
                if non_latin_letter_present is False:
                    request.state.latest_script_compliant = attempt_record
                else:
                    logger.warning(
                        "attempt_rejected_non_latin record_id=%s attempt=%d",
                        job.record_id,
                        request.attempt_number,
                    )
                if not word_present and non_latin_letter_present is False:
                    _accept_selected_story(
                        job=job,
                        state=request.state,
                        selected=attempt_record,
                        storage=storage,
                        manifest=manifest,
                        completed_keys=completed_keys,
                        failed_keys=failed_keys,
                        max_attempts=max_attempts,
                        parameters=parameters,
                        model_name=model_name,
                        model_revision=model_revision,
                    )
                    continue

            if request.state.max_attempt_number >= max_attempts:
                selected = request.state.latest_script_compliant
                if selected is None:
                    _mark_job_failed(
                        job=job,
                        failed_keys=failed_keys,
                        manifest=manifest,
                        logger=logger,
                        max_attempts=max_attempts,
                    )
                else:
                    _accept_selected_story(
                        job=job,
                        state=request.state,
                        selected=selected,
                        storage=storage,
                        manifest=manifest,
                        completed_keys=completed_keys,
                        failed_keys=failed_keys,
                        max_attempts=max_attempts,
                        parameters=parameters,
                        model_name=model_name,
                        model_revision=model_revision,
                    )
            else:
                pending.append(job)

        _update_manifest(storage, manifest)
        logger.info(
            "batch_complete batch=%d accepted=%d failed=%d attempts=%d pending=%d",
            batch_number,
            manifest["accepted_stories"],
            manifest["failed_stories"],
            manifest["total_generation_attempts"],
            len(pending),
        )


def _generate_requests_resilient(
    *,
    generator: LocalHuggingFaceGenerator,
    requests: list[AttemptRequest],
    parameters: GenerationParameters,
    logger: logging.Logger,
) -> tuple[list[tuple[str | None, str | None, int]], int, bool]:
    """Generate a batch, splitting only OOMs and reporting a reusable safe size."""

    oom_message: str | None = None
    try:
        completions = generator.generate_batch(
            prompts=[request.prompt for request in requests],
            seeds=[request.seed for request in requests],
            generation_parameters=parameters,
        )
        if len(completions) != len(requests):
            raise RuntimeError(
                "Generator returned an unexpected number of completions: "
                f"{len(completions)} for {len(requests)} requests"
            )
        return (
            [(completion, None, len(requests)) for completion in completions],
            len(requests),
            False,
        )
    except Exception as error:
        if not _is_out_of_memory_error(error):
            logger.error(
                "systemic_generation_failure size=%d error=%s",
                len(requests),
                error,
            )
            raise
        oom_message = str(error)
        # Do not recurse while the exception traceback still owns failed
        # generation-frame tensors; clearing the allocator cannot free live tensors.
        error.__traceback__ = None

    assert oom_message is not None
    generator.clear_device_cache()
    logger.warning("oom_batch_split size=%d error=%s", len(requests), oom_message)
    if len(requests) == 1:
        raise RuntimeError(
            f"Model generation is out of memory at batch size 1: {oom_message}"
        )

    midpoint = len(requests) // 2
    left_results, left_safe_size, _ = _generate_requests_resilient(
        generator=generator,
        requests=requests[:midpoint],
        parameters=parameters,
        logger=logger,
    )
    right_results, right_safe_size, _ = _generate_requests_resilient(
        generator=generator,
        requests=requests[midpoint:],
        parameters=parameters,
        logger=logger,
    )
    return (
        left_results + right_results,
        max(left_safe_size, right_safe_size),
        True,
    )


def _accept_selected_story(
    *,
    job: GenerationJob,
    state: AttemptProgress,
    selected: dict[str, Any],
    storage: DatasetStorage,
    manifest: dict[str, Any],
    completed_keys: set[RecordKey],
    failed_keys: set[RecordKey],
    max_attempts: int,
    parameters: GenerationParameters,
    model_name: str,
    model_revision: str | None,
) -> None:
    emotion_word_present = bool(selected["emotion_word_present"])
    if selected.get("non_latin_letter_present") is not False:
        raise ValueError(
            f"Cannot accept a story without strict script validation: {job.record_id}"
        )
    non_latin_letter_present = False
    accepted_after_max_attempts = (
        state.max_attempt_number >= max_attempts and emotion_word_present
    )
    accepted_record = {
        "record_id": job.record_id,
        "emotion": job.emotion,
        "topic_id": job.topic.topic_id,
        "topic": job.topic.topic,
        "sample_index": job.sample_index,
        "prompt_version": PROMPT_VERSION,
        "prompt": str(selected["prompt"]),
        "raw_completion": selected["raw_completion"],
        "story": str(selected["story"]),
        "generator_model": model_name,
        "model_revision": model_revision,
        "generation_parameters": parameters.as_dict(),
        "generation_batch_size": int(selected["generation_batch_size"]),
        "accepted_seed": int(selected["seed"]),
        "attempt_count": state.max_attempt_number,
        "emotion_word_present": emotion_word_present,
        "non_latin_letter_present": non_latin_letter_present,
        "accepted_after_max_attempts": accepted_after_max_attempts,
        "created_at": utc_now(),
    }
    storage.append_accepted(job.emotion, accepted_record)
    completed_keys.add(job.key)
    failed_keys.discard(job.key)
    manifest["accepted_stories"] = len(completed_keys)
    manifest["failed_stories"] = len(failed_keys)
    if accepted_after_max_attempts:
        manifest["accepted_after_max_attempts"] += 1


def _mark_job_failed(
    *,
    job: GenerationJob,
    failed_keys: set[RecordKey],
    manifest: dict[str, Any],
    logger: logging.Logger,
    max_attempts: int,
) -> None:
    failed_keys.add(job.key)
    manifest["failed_stories"] = len(failed_keys)
    logger.error("story_failed record_id=%s attempts=%d", job.record_id, max_attempts)


def load_topics(path: Path) -> list[Topic]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Topics file must contain a JSON list: {path}")

    topics: list[Topic] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Topic entry {index} in {path} must be an object")
        topic_id = item.get("topic_id")
        topic_text = item.get("topic")
        if isinstance(topic_id, bool) or not isinstance(topic_id, int):
            raise ValueError(f"Topic entry {index} in {path} has an invalid topic_id")
        if not isinstance(topic_text, str) or not topic_text.strip():
            raise ValueError(f"Topic entry {index} in {path} has an invalid topic")
        if topic_id in seen_ids:
            raise ValueError(f"Duplicate topic_id {topic_id} in {path}")
        seen_ids.add(topic_id)
        topics.append(Topic(topic_id=topic_id, topic=topic_text))
    return topics


def _preflight_output_state(
    *,
    output_dir: Path,
    resume: bool,
    run_config: dict[str, Any],
) -> str:
    """Inspect output compatibility without creating or repairing any files."""

    if not output_dir.exists():
        return "new"
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return "new_empty_directory"
    if not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another directory or pass --resume."
        )
    storage = DatasetStorage(output_dir, resume=True)
    storage.validate_run_config(run_config)
    return "compatible_resume"


def _validate_accepted_records(
    *,
    storage: DatasetStorage,
    jobs: list[GenerationJob],
    model_name: str,
    model_revision: str | None,
    parameters: GenerationParameters,
    base_seed: int | None = None,
    max_attempts: int | None = None,
) -> dict[RecordKey, dict[str, Any]]:
    """Stream-check accepted files before trusting their keys for resume."""

    expected_by_key = {job.key: job for job in jobs}
    expected_files = {f"{job.emotion_slug}.jsonl" for job in jobs}
    records_dir = storage.records_dir
    actual_files = (
        {path.name for path in records_dir.glob("*.jsonl") if path.is_file()}
        if records_dir.exists()
        else set()
    )
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            "Accepted record files do not match the emotion configuration: "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )

    seen: set[RecordKey] = set()
    accepted_records: dict[RecordKey, dict[str, Any]] = {}
    expected_parameters = parameters.as_dict()
    for path, line_number, row in storage.iter_accepted_records():
        location = f"{path}:{line_number}"
        _validate_record_fields(
            row=row,
            expected=ACCEPTED_RECORD_FIELDS,
            kind="Accepted record",
            location=location,
        )
        try:
            emotion = row["emotion"]
            topic_id = row["topic_id"]
            sample_index = row["sample_index"]
        except KeyError as error:
            raise ValueError(f"Missing record key field in {location}: {error}") from error
        if not isinstance(emotion, str):
            raise ValueError(f"Invalid emotion in {location}")
        if isinstance(topic_id, bool) or not isinstance(topic_id, int):
            raise ValueError(f"Invalid topic_id in {location}")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError(f"Invalid sample_index in {location}")
        key = (emotion, topic_id, sample_index)
        job = expected_by_key.get(key)
        if job is None:
            raise ValueError(f"Unexpected accepted record key {key!r} in {location}")
        if key in seen:
            raise ValueError(f"Duplicate accepted record key {key!r} in {location}")
        seen.add(key)
        accepted_records[key] = row

        expected_fields: dict[str, Any] = {
            "record_id": job.record_id,
            "topic": job.topic.topic,
            "prompt_version": PROMPT_VERSION,
            "prompt": render_story_prompt(
                topic=job.topic.topic,
                emotion=job.emotion,
            ),
            "generator_model": model_name,
            "model_revision": model_revision,
            "generation_parameters": expected_parameters,
            "non_latin_letter_present": False,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Accepted record field {field!r} in {location} is "
                    f"{row.get(field)!r}, expected {expected!r}"
                )
        if path.stem != job.emotion_slug:
            raise ValueError(
                f"Accepted record {job.record_id} is stored in the wrong file: {path}"
            )
        story = row.get("story")
        if not isinstance(story, str) or not story:
            raise ValueError(f"Accepted record has no story in {location}")
        if contains_non_latin_letter(story):
            raise ValueError(f"Accepted record contains non-Latin letters in {location}")
        word_present = contains_exact_emotion_word(
            generated_text=story,
            emotion=job.emotion,
        )
        if row.get("emotion_word_present") is not word_present:
            raise ValueError(f"Incorrect emotion_word_present flag in {location}")

        attempt_count = row.get("attempt_count")
        accepted_seed = row.get("accepted_seed")
        if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
            raise ValueError(f"Invalid attempt_count in {location}")
        if attempt_count < 1:
            raise ValueError(f"attempt_count must be positive in {location}")
        if max_attempts is not None and attempt_count > max_attempts:
            raise ValueError(f"attempt_count exceeds the configured maximum in {location}")
        expected_after_max = bool(
            max_attempts is not None
            and attempt_count >= max_attempts
            and word_present
        )
        if row.get("accepted_after_max_attempts") is not expected_after_max:
            raise ValueError(f"Incorrect accepted_after_max_attempts flag in {location}")
        if isinstance(accepted_seed, bool) or not isinstance(accepted_seed, int):
            raise ValueError(f"Invalid accepted_seed in {location}")
        generation_batch_size = row.get("generation_batch_size")
        if (
            isinstance(generation_batch_size, bool)
            or not isinstance(generation_batch_size, int)
            or generation_batch_size < 1
        ):
            raise ValueError(f"Invalid generation_batch_size in {location}")
        if base_seed is not None:
            possible_seeds = {
                create_seed(
                    base_seed=base_seed,
                    emotion=job.emotion,
                    topic_id=job.topic.topic_id,
                    sample_index=job.sample_index,
                    attempt_number=attempt_number,
                )
                for attempt_number in range(1, attempt_count + 1)
            }
            if accepted_seed not in possible_seeds:
                raise ValueError(f"accepted_seed is not from this job in {location}")

    if seen != set(expected_by_key) and len(seen) == len(expected_by_key):
        # The size equality makes this unreachable after per-row membership checks,
        # but retains an explicit full-key-set invariant for future changes.
        raise ValueError("Accepted key coverage does not match the requested plan")
    return accepted_records


def _validate_attempt_records(
    *,
    storage: DatasetStorage,
    jobs: list[GenerationJob],
    accepted_records: dict[RecordKey, dict[str, Any]],
    model_name: str,
    model_revision: str | None,
    base_seed: int,
    max_attempts: int,
) -> None:
    """Validate retry history and tie every accepted row to its saved attempt."""

    expected_by_key = {job.key: job for job in jobs}
    attempts_by_key: dict[RecordKey, list[dict[str, Any]]] = {}
    for line_number, row in storage.iter_attempt_records():
        location = f"{storage.attempts_path}:{line_number}"
        _validate_record_fields(
            row=row,
            expected=ATTEMPT_RECORD_FIELDS,
            kind="Attempt record",
            location=location,
        )
        try:
            emotion = row["emotion"]
            topic_id = row["topic_id"]
            sample_index = row["sample_index"]
            attempt_number = row["attempt_number"]
            seed = row["seed"]
        except KeyError as error:
            raise ValueError(f"Missing attempt field in {location}: {error}") from error
        if not isinstance(emotion, str):
            raise ValueError(f"Invalid attempt emotion in {location}")
        if isinstance(topic_id, bool) or not isinstance(topic_id, int):
            raise ValueError(f"Invalid attempt topic_id in {location}")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError(f"Invalid attempt sample_index in {location}")
        if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
            raise ValueError(f"Invalid attempt_number in {location}")
        if not 1 <= attempt_number <= max_attempts:
            raise ValueError(f"attempt_number is outside the configured range in {location}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Invalid attempt seed in {location}")

        key = (emotion, topic_id, sample_index)
        job = expected_by_key.get(key)
        if job is None:
            raise ValueError(f"Unexpected attempt record key {key!r} in {location}")
        expected_seed = create_seed(
            base_seed=base_seed,
            emotion=job.emotion,
            topic_id=job.topic.topic_id,
            sample_index=job.sample_index,
            attempt_number=attempt_number,
        )
        expected_fields: dict[str, Any] = {
            "record_id": job.record_id,
            "topic": job.topic.topic,
            "seed": expected_seed,
            "prompt_version": PROMPT_VERSION,
            "prompt": render_story_prompt(
                topic=job.topic.topic,
                emotion=job.emotion,
            ),
            "generator_model": model_name,
            "model_revision": model_revision,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Attempt field {field!r} in {location} is "
                    f"{row.get(field)!r}, expected {expected!r}"
                )
        generation_batch_size = row.get("generation_batch_size")
        if (
            isinstance(generation_batch_size, bool)
            or not isinstance(generation_batch_size, int)
            or generation_batch_size < 1
        ):
            raise ValueError(f"Invalid attempt generation_batch_size in {location}")

        story = row.get("story")
        generation_error = row.get("generation_error")
        if story is None:
            if row.get("emotion_word_present") is not None:
                raise ValueError(f"Invalid emotion-word flag for failed attempt in {location}")
            if row.get("non_latin_letter_present") is not None:
                raise ValueError(f"Invalid script flag for failed attempt in {location}")
            if not isinstance(generation_error, str) or not generation_error:
                raise ValueError(f"Missing generation_error for null story in {location}")
        elif isinstance(story, str):
            word_present = contains_exact_emotion_word(
                generated_text=story,
                emotion=job.emotion,
            )
            non_latin = contains_non_latin_letter(story)
            if row.get("emotion_word_present") is not word_present:
                raise ValueError(f"Incorrect attempt emotion-word flag in {location}")
            if row.get("non_latin_letter_present") is not non_latin:
                raise ValueError(f"Incorrect attempt script flag in {location}")
            if generation_error is not None:
                raise ValueError(f"Successful attempt has generation_error in {location}")
        else:
            raise ValueError(f"Invalid attempt story in {location}")
        attempts_by_key.setdefault(key, []).append(row)

    for key, attempts in attempts_by_key.items():
        attempt_numbers = [int(row["attempt_number"]) for row in attempts]
        expected_numbers = list(range(1, max(attempt_numbers) + 1))
        if attempt_numbers != expected_numbers:
            raise ValueError(
                f"Attempt sequence for record key {key!r} is {attempt_numbers!r}, "
                f"expected {expected_numbers!r}"
            )

    for key, accepted in accepted_records.items():
        attempts = attempts_by_key.get(key, [])
        accepted_seed = int(accepted["accepted_seed"])
        selected_matches = [
            attempt for attempt in attempts if attempt.get("seed") == accepted_seed
        ]
        if len(selected_matches) != 1:
            raise ValueError(
                f"Accepted record key {key!r} does not select exactly one saved attempt"
            )
        selected = selected_matches[0]
        if accepted.get("attempt_count") != len(attempts):
            raise ValueError(
                f"Accepted record key {key!r} has an inconsistent attempt_count"
            )
        copied_fields = (
            "prompt",
            "raw_completion",
            "story",
            "emotion_word_present",
            "non_latin_letter_present",
            "generation_batch_size",
        )
        for field in copied_fields:
            if accepted.get(field) != selected.get(field):
                raise ValueError(
                    f"Accepted record key {key!r} does not match its attempt field "
                    f"{field!r}"
                )


def _validate_record_fields(
    *,
    row: dict[str, Any],
    expected: frozenset[str],
    kind: str,
    location: str,
) -> None:
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{kind} fields do not match the flat story schema in {location}: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _is_immutable_revision(revision: str | None) -> bool:
    return bool(revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256_expectation(
    *,
    parser: argparse.ArgumentParser,
    option: str,
    expected: str,
    actual: str,
) -> None:
    normalized = expected.casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        parser.error(f"{option} must be exactly 64 hexadecimal characters")
    if normalized != actual:
        parser.error(f"{option} expected {normalized}, but the resolved file is {actual}")


def _is_out_of_memory_error(error: BaseException) -> bool:
    """Recognize resource exhaustion without masking unrelated model failures."""

    current: BaseException | None = error
    while current is not None:
        if current.__class__.__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
            return True
        message = str(current).casefold()
        if "out of memory" in message or "not enough memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _create_jobs(
    *,
    emotion_specs: tuple[EmotionSpec, ...],
    topics: list[Topic],
    samples_per_pair: int,
    base_seed: int,
) -> list[GenerationJob]:
    randomizer = random.Random(base_seed)
    emotions = tuple(spec.emotion for spec in emotion_specs)
    slugs = {spec.emotion: spec.slug for spec in emotion_specs}
    jobs_by_emotion = {
        emotion: [
            GenerationJob(
                emotion=emotion,
                emotion_slug=slugs[emotion],
                topic=topic,
                sample_index=sample_index,
            )
            for topic in topics
            for sample_index in range(samples_per_pair)
        ]
        for emotion in emotions
    }
    for emotion_jobs in jobs_by_emotion.values():
        randomizer.shuffle(emotion_jobs)

    jobs: list[GenerationJob] = []
    emotion_order = list(emotions)
    jobs_per_emotion = len(topics) * samples_per_pair
    for position in range(jobs_per_emotion):
        randomizer.shuffle(emotion_order)
        if jobs and len(emotion_order) > 1 and emotion_order[0] == jobs[-1].emotion:
            swap_index = next(
                index
                for index, emotion in enumerate(emotion_order[1:], start=1)
                if emotion != jobs[-1].emotion
            )
            emotion_order[0], emotion_order[swap_index] = (
                emotion_order[swap_index],
                emotion_order[0],
            )
        jobs.extend(jobs_by_emotion[emotion][position] for emotion in emotion_order)
    return jobs


def _generation_calls_remain(
    *,
    jobs: list[GenerationJob],
    completed_keys: set[RecordKey],
    failed_keys: set[RecordKey],
    attempt_progress: dict[RecordKey, AttemptProgress],
    max_attempts: int,
) -> bool:
    for job in jobs:
        if job.key in completed_keys or job.key in failed_keys:
            continue
        state = attempt_progress.get(job.key, AttemptProgress())
        selected = state.latest_nonempty
        should_retry_selected = selected is None or bool(
            selected.get("emotion_word_present", True)
        ) or selected.get("non_latin_letter_present") is not False
        if should_retry_selected and state.max_attempt_number < max_attempts:
            return True
    return False


def _build_run_config(
    *,
    emotion_specs: tuple[EmotionSpec, ...],
    emotion_config_path: Path | None,
    emotion_config_sha256: str | None,
    topics: list[Topic],
    topics_path: Path,
    topics_sha256: str,
    model_name: str,
    model_revision: str | None,
    samples_per_pair: int,
    max_attempts: int,
    base_seed: int,
    parameters: GenerationParameters,
    device: str,
    dtype: str,
    output_dir: Path,
    created_at: str,
    attn_implementation: str | None,
    prompt_role: str,
    generation_request: str,
    chat_system_instruction: str | None,
    eos_token_policy: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    emotions = tuple(spec.emotion for spec in emotion_specs)
    return {
        "schema_version": 2,
        "emotions": list(emotions),
        "emotion_record_stems": {
            spec.emotion: spec.slug for spec in emotion_specs
        },
        "emotion_config_source": (
            str(emotion_config_path) if emotion_config_path is not None else None
        ),
        "emotion_config_sha256": emotion_config_sha256,
        "generator_model": model_name,
        "model_revision": model_revision,
        "topics_source": str(topics_path),
        "topics_sha256": topics_sha256,
        "topics": [
            {"topic_id": topic.topic_id, "topic": topic.topic} for topic in topics
        ],
        "prompt_version": PROMPT_VERSION,
        "prompt_template": STORY_PROMPT_TEMPLATE,
        "samples_per_pair": samples_per_pair,
        "max_attempts": max_attempts,
        "base_seed": base_seed,
        "generation_parameters": parameters.as_dict(),
        "device": device,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "prompt_role": prompt_role,
        "generation_request": generation_request,
        "chat_system_instruction": chat_system_instruction,
        "eos_token_policy": eos_token_policy,
        "trust_remote_code": trust_remote_code,
        "output_path": str(output_dir),
        "created_at": created_at,
    }


def _build_manifest(
    *,
    model_name: str,
    base_seed: int,
    samples_per_pair: int,
    max_attempts: int,
    emotions: tuple[str, ...],
    topics: list[Topic],
    intended_stories: int,
    accepted_stories: int,
    accepted_after_max_attempts: int,
    failed_stories: int,
    total_attempts: int,
    parameters: GenerationParameters,
    created_at: str,
) -> dict[str, Any]:
    return {
        "generator_model": model_name,
        "prompt_version": PROMPT_VERSION,
        "base_seed": base_seed,
        "samples_per_pair": samples_per_pair,
        "max_attempts": max_attempts,
        "number_of_emotions": len(emotions),
        "number_of_topics": len(topics),
        "intended_stories": intended_stories,
        "accepted_stories": accepted_stories,
        "accepted_after_max_attempts": accepted_after_max_attempts,
        "failed_stories": failed_stories,
        "total_generation_attempts": total_attempts,
        "generation_parameters": parameters.as_dict(),
        "status": "running",
        "created_at": created_at,
        "updated_at": utc_now(),
    }


def _update_manifest(
    storage: DatasetStorage,
    manifest: dict[str, Any],
    *,
    status: str | None = None,
) -> None:
    if status is not None:
        manifest["status"] = status
    manifest["updated_at"] = utc_now()
    storage.write_manifest(manifest)


def _reconcile_manifest(
    storage: DatasetStorage,
    manifest: dict[str, Any],
    failed_keys: set[RecordKey],
    intended_keys: set[RecordKey],
) -> None:
    completed = storage.load_completed_index()
    progress, total_attempts = storage.load_attempt_progress(completed.keys)
    terminal_failures = {
        key
        for key, state in progress.items()
        if key in intended_keys
        and state.max_attempt_number >= int(manifest["max_attempts"])
        and state.latest_script_compliant is None
    }
    failed_keys.clear()
    failed_keys.update(terminal_failures)
    manifest["accepted_stories"] = len(completed.keys)
    manifest["accepted_after_max_attempts"] = completed.accepted_after_max_attempts
    manifest["failed_stories"] = len(failed_keys)
    manifest["total_generation_attempts"] = total_attempts


def _mark_startup_interrupted(
    *,
    storage: DatasetStorage,
    model_name: str,
    base_seed: int,
    samples_per_pair: int,
    max_attempts: int,
    emotions: tuple[str, ...],
    topics: list[Topic],
    parameters: GenerationParameters,
    created_at: str,
) -> None:
    if storage.manifest_path.exists():
        interrupted_manifest = storage.read_manifest()
    else:
        interrupted_manifest = _build_manifest(
            model_name=model_name,
            base_seed=base_seed,
            samples_per_pair=samples_per_pair,
            max_attempts=max_attempts,
            emotions=emotions,
            topics=topics,
            intended_stories=len(emotions) * len(topics) * samples_per_pair,
            accepted_stories=0,
            accepted_after_max_attempts=0,
            failed_stories=0,
            total_attempts=0,
            parameters=parameters,
            created_at=created_at,
        )
    _update_manifest(storage, interrupted_manifest, status="interrupted")


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("emotionvectors.generation.emotional_stories")
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return value


def _attempt_count(raw: str) -> int:
    value = _positive_int(raw)
    if value > MAX_TOTAL_ATTEMPTS:
        raise argparse.ArgumentTypeError(
            f"value cannot exceed {MAX_TOTAL_ATTEMPTS} total attempts"
        )
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return value


def _top_p(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("value must be in the interval (0, 1]")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
