"""Command-line orchestration for independent emotional-story generation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..constants import MODEL_ID, MODEL_REVISION
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

NEGATIVE_EMOTIONS = (
    "anger",
    "fear",
    "disgust",
    "sadness",
    "anxiety",
    "desperation",
    "frustration",
    "hostility",
)
POSITIVE_EMOTIONS = ("calmness", "compassion", "joy", "trust")
ALL_EMOTIONS = NEGATIVE_EMOTIONS + POSITIVE_EMOTIONS
EMOTION_GROUP = {
    **{emotion: "negative" for emotion in NEGATIVE_EMOTIONS},
    **{emotion: "positive" for emotion in POSITIVE_EMOTIONS},
}
DEFAULT_MODEL = MODEL_ID
DEFAULT_TOPICS_PATH = Path("config/topics.json")
MAX_TOTAL_ATTEMPTS = 5


@dataclass(frozen=True)
class Topic:
    topic_id: int
    topic: str


@dataclass(frozen=True)
class GenerationJob:
    emotion: str
    topic: Topic
    sample_index: int

    @property
    def key(self) -> RecordKey:
        return self.emotion, self.topic.topic_id, self.sample_index

    @property
    def record_id(self) -> str:
        return (
            f"{self.emotion}_topic_{self.topic.topic_id:03d}_"
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
        default=MODEL_REVISION,
        help="Pinned Hugging Face commit for the model and tokenizer",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=DEFAULT_TOPICS_PATH,
        help="JSON file containing stable topic_id/topic objects",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=ALL_EMOTIONS,
        default=None,
        help="Subset of emotions to generate; defaults to all twelve",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model != DEFAULT_MODEL:
        parser.error(f"--model must be exactly {DEFAULT_MODEL}")
    if args.model_revision != MODEL_REVISION:
        parser.error(f"--model-revision must be exactly {MODEL_REVISION}")

    emotions = tuple(args.emotions) if args.emotions is not None else ALL_EMOTIONS
    if len(set(emotions)) != len(emotions):
        parser.error("--emotions cannot contain duplicates")

    try:
        topics_path = args.topics.expanduser().resolve()
        topics = load_topics(topics_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.max_topics is not None:
        topics = topics[: args.max_topics]
    if not topics:
        parser.error("At least one topic is required")

    parameters = GenerationParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
    )
    output_dir = args.output_dir.expanduser().resolve()
    created_at = utc_now()
    run_config = _build_run_config(
        emotions=emotions,
        topics=topics,
        topics_path=topics_path,
        model_name=args.model,
        samples_per_pair=args.samples_per_pair,
        max_attempts=args.max_attempts,
        base_seed=args.base_seed,
        parameters=parameters,
        device=args.device,
        dtype=args.dtype,
        output_dir=output_dir,
        created_at=created_at,
    )
    run_config["model_revision"] = args.model_revision

    storage: DatasetStorage | None = None
    try:
        storage = DatasetStorage(output_dir, resume=args.resume)
        persisted_config = storage.prepare_run_config(run_config)
        storage.initialize_layout(ALL_EMOTIONS)
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
        jobs = _create_jobs(
            emotions=emotions,
            topics=topics,
            samples_per_pair=args.samples_per_pair,
            base_seed=args.base_seed,
        )
        intended_keys = {job.key for job in jobs}
        unexpected_completed = completed_index.keys - intended_keys
        if unexpected_completed:
            parser.error(
                "Existing accepted records do not belong to this run configuration: "
                f"{sorted(unexpected_completed)[:3]!r}"
            )

        completed_keys = set(completed_index.keys)
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
        manifest["model_revision"] = args.model_revision
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
                model_revision=args.model_revision,
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
            model_revision=args.model_revision,
            logger=logger,
            batch_size=args.batch_size,
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
    while pending:
        requests: list[AttemptRequest] = []
        while pending and len(requests) < batch_size:
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
        batch_results = _generate_requests_resilient(
            generator=generator,
            requests=requests,
            parameters=parameters,
            logger=logger,
        )

        for request, (raw_completion, generation_error) in zip(
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
                "emotion_group": EMOTION_GROUP[job.emotion],
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
) -> list[tuple[str | None, str | None]]:
    """Generate a batch, splitting failures until a single bad row is isolated."""

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
        return [(completion, None) for completion in completions]
    except Exception as error:
        generator.clear_device_cache()
        logger.warning(
            "technical_batch_failure size=%d error=%s",
            len(requests),
            error,
        )
        if len(requests) == 1:
            return [(None, str(error))]

        midpoint = len(requests) // 2
        return _generate_requests_resilient(
            generator=generator,
            requests=requests[:midpoint],
            parameters=parameters,
            logger=logger,
        ) + _generate_requests_resilient(
            generator=generator,
            requests=requests[midpoint:],
            parameters=parameters,
            logger=logger,
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
        "emotion_group": EMOTION_GROUP[job.emotion],
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


def _create_jobs(
    *,
    emotions: tuple[str, ...],
    topics: list[Topic],
    samples_per_pair: int,
    base_seed: int,
) -> list[GenerationJob]:
    randomizer = random.Random(base_seed)
    jobs_by_emotion = {
        emotion: [
            GenerationJob(emotion=emotion, topic=topic, sample_index=sample_index)
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
    emotions: tuple[str, ...],
    topics: list[Topic],
    topics_path: Path,
    model_name: str,
    samples_per_pair: int,
    max_attempts: int,
    base_seed: int,
    parameters: GenerationParameters,
    device: str,
    dtype: str,
    output_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    return {
        "emotions": list(emotions),
        "emotion_groups": {
            "negative": [emotion for emotion in emotions if emotion in NEGATIVE_EMOTIONS],
            "positive": [emotion for emotion in emotions if emotion in POSITIVE_EMOTIONS],
        },
        "generator_model": model_name,
        "topics_source": str(topics_path),
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
