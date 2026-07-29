#!/usr/bin/env python3
"""Generate independent, structurally validated neutral Person-AI dialogues."""

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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..constants import MODEL_ID, MODEL_REVISION
from .model import (
    GenerationParameters,
    LocalHuggingFaceGenerator,
    contains_non_latin_letter,
)
from .seeds import create_seed
from .storage import (
    append_jsonl,
    atomic_write_json,
    repair_incomplete_jsonl_tail,
)
from .neutral_prompt import (
    CHAT_PROMPT_ROLE,
    CHAT_SYSTEM_INSTRUCTION,
    DIALOGUE_TYPES,
    DIALOGUE_TYPE_INSTRUCTIONS,
    GENERATION_REQUEST,
    NEUTRAL_DIALOGUE_PROMPT_TEMPLATE,
    OPTIONAL_SYSTEM_INSTRUCTION_INCLUDED,
    OPTIONAL_SYSTEM_INSTRUCTION_NOT_INCLUDED,
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION_NOT_REQUIRED,
    SYSTEM_INSTRUCTION_REQUIRED,
    render_neutral_dialogue_prompt,
)

DEFAULT_MODEL = MODEL_ID
DEFAULT_TOPICS_PATH = Path("config/topics.json")
DEFAULT_OUTPUT_DIR = Path("outputs/neutral_dialogues")
LABEL = "neutral"
MAX_TOTAL_ATTEMPTS = 20
ATTEMPT_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "label",
        "topic_id",
        "topic",
        "sample_index",
        "attempt_number",
        "seed",
        "dialogue_type",
        "system_instruction_included",
        "prompt_version",
        "prompt",
        "raw_completion",
        "dialogue_person_ai",
        "structurally_valid",
        "non_latin_letter_present",
        "failure_reasons",
        "generation_error",
        "created_at",
    }
)
ACCEPTED_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "label",
        "topic_id",
        "topic",
        "sample_index",
        "dialogue_type",
        "system_instruction_included",
        "prompt_version",
        "prompt",
        "raw_completion",
        "dialogue_person_ai",
        "transcript",
        "generator_model",
        "accepted_seed",
        "attempt_count",
        "structurally_valid",
        "non_latin_letter_present",
        "accepted_after_max_attempts",
        "created_at",
    }
)

SPEAKER_RE = re.compile(r"^(Person|AI):")
OTHER_SPEAKER_RE = re.compile(
    r"^(?:System|Human|Assistant|User|ChatGPT|Bot|Agent|Customer|Client|Narrator|"
    r"Model|Helper|Computer|Moderator|Friend|Neighbor|Guide|Expert|Researcher|"
    r"Advisor|Consultant|Support|Operator|Teacher|Student|Coach|Parent|Child|"
    r"Employee|Employer|Manager|Coworker|Host|Guest|Caller|Representative|"
    r"Developer|Programmer|Interviewer|Respondent|Virtual Assistant|AI Assistant|"
    r"Speaker(?:\s*[A-Za-z0-9]+)?):",
    flags=re.IGNORECASE,
)
GENERIC_LABEL_RE = re.compile(
    r"^(?P<label>[A-Z][A-Za-z0-9]*(?:[ \t]+[A-Z][A-Za-z0-9]*){0,2}):"
)
CONTENT_HEADING_LABELS = frozenset(
    {
        "answer",
        "background",
        "benefits",
        "caution",
        "cautions",
        "checklist",
        "code",
        "considerations",
        "data",
        "details",
        "example",
        "examples",
        "formula",
        "input",
        "instructions",
        "key points",
        "limitations",
        "method",
        "next steps",
        "note",
        "notes",
        "options",
        "output",
        "procedure",
        "python",
        "recommendations",
        "references",
        "requirements",
        "result",
        "results",
        "risks",
        "sources",
        "steps",
        "summary",
        "timeline",
        "tips",
    }
)
DIALOGUE_MARKER_RE = re.compile(
    r"^[ \t]*(?:"
    r"\[\s*(?:dialogue|conversation)(?:\s+\d+)?\s*\][ \t]*:?"
    r"|(?:dialogue|conversation)(?:\s+(?:#\s*)?\d+)?\s*:"
    r")(?:[ \t]+.*)?[ \t]*$",
    flags=re.IGNORECASE,
)
END_MARKER_RE = re.compile(
    r"^[ \t]*\[?[ \t]*(?:end of (?:the )?"
    r"(?:dialogue|conversation|transcript)|"
    r"(?:dialogue|conversation|transcript) ends)[.!]?[ \t]*\]?[ \t]*$",
    flags=re.IGNORECASE,
)
FENCE_LINE_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
SPEAKER_INLINE_FENCE_RE = re.compile(
    r"^(?:Person|AI):[ \t]*(?P<fence>`{3,}|~{3,})"
)
OUTER_OPEN_FENCE_RE = re.compile(r"(?P<fence>`{3,}|~{3,})[^\r\n]*")
OUTER_CLOSE_FENCE_RE = re.compile(r"(?P<fence>`{3,}|~{3,})")
META_PREFACE_RE = re.compile(
    r"^[ \t]*(?:Here (?:is|are)|Here['’]s|Below (?:is|are)|Sure\b|Certainly\b|"
    r"This (?:dialogue|conversation)\b|(?:Explanation|Analysis|Preface)\s*:)",
    flags=re.IGNORECASE,
)
META_CONCLUSION_RE = re.compile(
    r"^[ \t]*(?:(?:This|The) (?:dialogue|conversation)\b|"
    r"(?:Explanation|Note|Conclusion|Analysis|Preface|Afterword|Commentary)\s*:)",
    flags=re.IGNORECASE,
)

RecordKey = tuple[int, int]


@dataclass(frozen=True)
class Topic:
    topic_id: int
    topic: str


@dataclass(frozen=True)
class GenerationJob:
    topic: Topic
    sample_index: int
    samples_per_topic: int

    @property
    def key(self) -> RecordKey:
        return self.topic.topic_id, self.sample_index

    @property
    def record_id(self) -> str:
        return (
            f"neutral_topic_{self.topic.topic_id:03d}_"
            f"sample_{self.sample_index:02d}"
        )

    @property
    def dialogue_type(self) -> str:
        type_index = (
            self.topic.topic_id * self.samples_per_topic + self.sample_index
        ) % len(DIALOGUE_TYPES)
        return DIALOGUE_TYPES[type_index]

    @property
    def system_instruction_included(self) -> bool:
        return self.sample_index % 3 == 0


@dataclass
class AttemptState:
    max_attempt_number: int = 0
    latest_attempt: dict[str, Any] | None = None


@dataclass(frozen=True)
class AttemptRequest:
    job: GenerationJob
    state: AttemptState
    prompt: str
    attempt_number: int
    seed: int
    created_at: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate independent neutral Person-AI dialogues with a local model."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Pinned Hugging Face commit for the model and tokenizer. "
            "The released 7B model defaults to its pinned revision; other models "
            "should pass an immutable revision explicitly."
        ),
    )
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS_PATH)
    parser.add_argument(
        "--expected-topics-sha256",
        default=None,
        help="Optional lowercase SHA-256 guard for the exact topics file bytes",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-topic", type=_positive_int, default=12)
    parser.add_argument("--max-topics", type=_positive_int, default=None)
    parser.add_argument("--max-attempts", type=_attempt_count, default=20)
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Execution-only prompt batch size; does not alter dataset identity",
    )
    parser.add_argument("--temperature", type=_positive_float, default=0.9)
    parser.add_argument("--top-p", type=_top_p, default=0.95)
    parser.add_argument(
        "--top-k",
        type=_nonnegative_int,
        default=None,
        help="Optional explicit sampling cutoff; zero disables top-k",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=_positive_float,
        default=None,
        help="Optional explicit Transformers-compatible repetition penalty",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=500)
    parser.add_argument("--base-seed", type=int, default=20260721)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument(
        "--eos-token-policy",
        choices=("model", "tokenizer"),
        default="tokenizer",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-model-revision",
        action="store_true",
        help="Require --model-revision to be a 40-character hexadecimal commit",
    )
    parser.add_argument(
        "--expected-topics",
        type=_positive_int,
        default=None,
        help="Pre-model-load guard for the exact topic count",
    )
    parser.add_argument(
        "--expected-dialogues",
        type=_positive_int,
        default=None,
        help="Pre-model-load guard for topics × samples",
    )
    parser.add_argument("--resume", action="store_true")
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
    model_revision = args.model_revision
    if model_revision is None and args.model == DEFAULT_MODEL:
        model_revision = MODEL_REVISION
    if args.model != DEFAULT_MODEL and not _is_immutable_revision(model_revision):
        parser.error(
            "Non-default models require --model-revision as a 40-character "
            "hexadecimal commit"
        )
    if args.require_model_revision and not _is_immutable_revision(model_revision):
        parser.error(
            "--require-model-revision needs --model-revision to be a 40-character "
            "hexadecimal commit"
        )
    args.model_revision = model_revision

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

    intended_dialogues = len(topics) * args.samples_per_topic
    count_expectations = (
        ("--expected-topics", args.expected_topics, len(topics)),
        ("--expected-dialogues", args.expected_dialogues, intended_dialogues),
    )
    for option, expected, actual in count_expectations:
        if expected is not None and expected != actual:
            parser.error(f"{option} expected {expected}, but the resolved run has {actual}")

    output_dir = args.output_dir.expanduser().resolve()
    parameters = GenerationParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
    )
    jobs = create_jobs(
        topics=topics,
        samples_per_topic=args.samples_per_topic,
        base_seed=args.base_seed,
    )
    intended_keys = {job.key for job in jobs}
    if len(intended_keys) != intended_dialogues:
        parser.error("Internal key collision in the neutral-dialogue plan")
    paths = output_paths(output_dir)
    created_at = utc_now()
    run_config = build_run_config(
        args=args,
        topics=topics,
        topics_path=topics_path,
        topics_sha256=topics_sha256,
        output_dir=output_dir,
        parameters=parameters,
        created_at=created_at,
    )

    if args.preflight_only:
        try:
            output_state = preflight_output_state(
                paths=paths,
                run_config=run_config,
                resume=args.resume,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "generator_model": args.model,
                    "model_revision": model_revision,
                    "number_of_topics": len(topics),
                    "samples_per_topic": args.samples_per_topic,
                    "intended_dialogues": intended_dialogues,
                    "output_dir": str(output_dir),
                    "output_state": output_state,
                    "model_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    startup_manifest = build_manifest(
        model_name=args.model,
        model_revision=model_revision,
        base_seed=args.base_seed,
        topics=topics,
        samples_per_topic=args.samples_per_topic,
        max_attempts=args.max_attempts,
        intended_dialogues=len(jobs),
        accepted_dialogues=0,
        accepted_after_max_attempts=0,
        failed_dialogues=0,
        total_generation_attempts=0,
        parameters=parameters,
        created_at=created_at,
    )
    if args.batch_size != 1:
        startup_manifest["execution_batch_size"] = args.batch_size
        startup_manifest["effective_batch_size"] = args.batch_size

    if args.validate_only:
        try:
            persisted_config = validate_existing_output(
                paths=paths,
                run_config=run_config,
            )
            completed_keys, accepted_after_max_attempts = load_completed_keys(
                paths["accepted"],
                repair=False,
            )
            unexpected_completed = completed_keys - intended_keys
            if unexpected_completed:
                raise ValueError(
                    "Accepted output contains keys outside this run: "
                    f"{sorted(unexpected_completed)[:3]!r}"
                )
            attempt_states, total_attempts = load_attempt_states(
                paths["attempts"],
                completed_keys=completed_keys,
                intended_keys=intended_keys,
                max_attempts=args.max_attempts,
                repair=False,
            )
            validate_output_records(
                accepted_path=paths["accepted"],
                attempts_path=paths["attempts"],
                jobs=jobs,
                model_name=args.model,
                base_seed=args.base_seed,
                max_attempts=args.max_attempts,
            )
            failed_keys = terminal_failure_keys(
                jobs=jobs,
                attempt_states=attempt_states,
                completed_keys=completed_keys,
                max_attempts=args.max_attempts,
            )
            persisted_manifest = read_json_object(paths["manifest"])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(
                json.dumps(
                    {
                        "status": "validation_failed",
                        "errors": [str(error)],
                        "model_loaded": False,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1

        validation_errors: list[str] = []
        missing_keys = intended_keys - completed_keys
        if missing_keys:
            validation_errors.append(
                f"{len(missing_keys)} accepted dialogues are missing"
            )
        if failed_keys:
            validation_errors.append(
                f"{len(failed_keys)} dialogues are terminal failures"
            )
        expected_manifest_fields: dict[str, Any] = {
            "generator_model": args.model,
            "model_revision": model_revision,
            "prompt_version": PROMPT_VERSION,
            "label": LABEL,
            "base_seed": args.base_seed,
            "number_of_topics": len(topics),
            "samples_per_topic": args.samples_per_topic,
            "max_attempts": args.max_attempts,
            "intended_dialogues": intended_dialogues,
            "accepted_dialogues": len(completed_keys),
            "accepted_after_max_attempts": accepted_after_max_attempts,
            "failed_dialogues": len(failed_keys),
            "total_generation_attempts": total_attempts,
            "generation_parameters": parameters.as_dict(),
            "status": "completed",
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
                        "accepted_dialogues": len(completed_keys),
                        "intended_dialogues": intended_dialogues,
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
                    "accepted_dialogues": len(completed_keys),
                    "intended_dialogues": intended_dialogues,
                    "failed_dialogues": 0,
                    "configuration_created_at": persisted_config[
                        "creation_timestamp"
                    ],
                    "model_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    try:
        persisted_config = prepare_output(
            paths=paths,
            run_config=run_config,
            resume=args.resume,
        )
        startup_manifest["created_at"] = str(
            persisted_config["creation_timestamp"]
        )
        if not paths["manifest"].exists():
            write_manifest(paths["manifest"], startup_manifest)
        logger = configure_logging(paths["log"])
        completed_keys, accepted_after_max_attempts = load_completed_keys(
            paths["accepted"],
            repair=True,
        )
        unexpected_completed = completed_keys - intended_keys
        if unexpected_completed:
            raise ValueError(
                "Accepted output contains keys outside this run: "
                f"{sorted(unexpected_completed)[:3]!r}"
            )
        attempt_states, total_attempts = load_attempt_states(
            paths["attempts"],
            completed_keys=completed_keys,
            intended_keys=intended_keys,
            max_attempts=args.max_attempts,
            repair=True,
        )
        validate_output_records(
            accepted_path=paths["accepted"],
            attempts_path=paths["attempts"],
            jobs=jobs,
            model_name=args.model,
            base_seed=args.base_seed,
            max_attempts=args.max_attempts,
        )
        completed_keys = set(completed_keys)
        accepted_after_max_attempts = reconcile_flushed_attempts(
            jobs=jobs,
            attempt_states=attempt_states,
            completed_keys=completed_keys,
            accepted_path=paths["accepted"],
            model_name=args.model,
            max_attempts=args.max_attempts,
            accepted_after_max_attempts=accepted_after_max_attempts,
            logger=logger,
        )
        failed_keys = terminal_failure_keys(
            jobs=jobs,
            attempt_states=attempt_states,
            completed_keys=completed_keys,
            max_attempts=args.max_attempts,
        )
        manifest = build_manifest(
            model_name=args.model,
            model_revision=model_revision,
            base_seed=args.base_seed,
            topics=topics,
            samples_per_topic=args.samples_per_topic,
            max_attempts=args.max_attempts,
            intended_dialogues=len(jobs),
            accepted_dialogues=len(completed_keys),
            accepted_after_max_attempts=accepted_after_max_attempts,
            failed_dialogues=len(failed_keys),
            total_generation_attempts=total_attempts,
            parameters=parameters,
            created_at=str(persisted_config["creation_timestamp"]),
        )
        effective_batch_size = args.batch_size
        previous_effective: Any = None
        if args.resume and paths["manifest"].exists():
            previous_manifest = read_json_object(paths["manifest"])
            previous_effective = previous_manifest.get("effective_batch_size")
            if (
                isinstance(previous_effective, int)
                and not isinstance(previous_effective, bool)
                and previous_effective > 0
            ):
                effective_batch_size = min(effective_batch_size, previous_effective)
        if args.batch_size != 1 or previous_effective is not None:
            manifest["execution_batch_size"] = args.batch_size
            manifest["effective_batch_size"] = effective_batch_size
        write_manifest(paths["manifest"], manifest)
    except KeyboardInterrupt:
        if paths["config"].exists():
            interrupted_manifest = (
                read_json_object(paths["manifest"])
                if paths["manifest"].exists()
                else startup_manifest
            )
            interrupted_manifest["status"] = "interrupted"
            write_manifest(paths["manifest"], interrupted_manifest)
        print("Generation interrupted during startup", file=sys.stderr)
        return 130
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            f"Could not initialize neutral-dialogue generation: {error}",
            file=sys.stderr,
        )
        return 1

    try:
        generator: LocalHuggingFaceGenerator | None = None
        if len(completed_keys) + len(failed_keys) < len(jobs):
            generator = LocalHuggingFaceGenerator(
                model_name=args.model,
                model_revision=model_revision,
                generation_request=GENERATION_REQUEST,
                prompt_role=CHAT_PROMPT_ROLE,
                chat_system_instruction=CHAT_SYSTEM_INSTRUCTION,
                device=args.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                use_model_eos_tokens=args.eos_token_policy == "model",
            )
        run_jobs(
            jobs=jobs,
            generator=generator,
            attempts_path=paths["attempts"],
            accepted_path=paths["accepted"],
            manifest_path=paths["manifest"],
            manifest=manifest,
            completed_keys=completed_keys,
            failed_keys=failed_keys,
            attempt_states=attempt_states,
            model_name=args.model,
            base_seed=args.base_seed,
            max_attempts=args.max_attempts,
            parameters=parameters,
            logger=logger,
            batch_size=effective_batch_size,
        )
    except KeyboardInterrupt:
        refresh_manifest_counts(
            manifest=manifest,
            completed_keys=completed_keys,
            failed_keys=failed_keys,
        )
        manifest["status"] = "interrupted"
        write_manifest(paths["manifest"], manifest)
        logger.warning("generation_interrupted manifest_updated_for_resume=true")
        return 130
    except Exception:
        refresh_manifest_counts(
            manifest=manifest,
            completed_keys=completed_keys,
            failed_keys=failed_keys,
        )
        manifest["status"] = "failed"
        write_manifest(paths["manifest"], manifest)
        logger.exception("generation_stopped unrecoverable_error=true")
        return 1

    manifest["status"] = "completed_with_failures" if failed_keys else "completed"
    refresh_manifest_counts(
        manifest=manifest,
        completed_keys=completed_keys,
        failed_keys=failed_keys,
    )
    write_manifest(paths["manifest"], manifest)
    logger.info(
        "generation_complete accepted=%d intended=%d failed=%d attempts=%d",
        manifest["accepted_dialogues"],
        manifest["intended_dialogues"],
        manifest["failed_dialogues"],
        manifest["total_generation_attempts"],
    )
    return 1 if failed_keys else 0


def run_jobs(
    *,
    jobs: list[GenerationJob],
    generator: LocalHuggingFaceGenerator | None,
    attempts_path: Path,
    accepted_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    completed_keys: set[RecordKey],
    failed_keys: set[RecordKey],
    attempt_states: dict[RecordKey, AttemptState],
    model_name: str,
    base_seed: int,
    max_attempts: int,
    parameters: GenerationParameters,
    logger: logging.Logger,
    batch_size: int,
) -> None:
    intended_dialogues = len(jobs)
    pending = deque(
        job
        for job in jobs
        if job.key not in completed_keys and job.key not in failed_keys
    )
    effective_batch_size = batch_size
    batch_number = 0

    while pending:
        requests: list[AttemptRequest] = []
        while pending and len(requests) < effective_batch_size:
            job = pending.popleft()
            if job.key in completed_keys or job.key in failed_keys:
                continue
            state = attempt_states.setdefault(job.key, AttemptState())
            attempt_number = state.max_attempt_number + 1
            if attempt_number > max_attempts:
                failed_keys.add(job.key)
                continue
            prompt = render_neutral_dialogue_prompt(
                topic=job.topic.topic,
                dialogue_type=job.dialogue_type,
                include_system_instruction=job.system_instruction_included,
            )
            requests.append(
                AttemptRequest(
                    job=job,
                    state=state,
                    prompt=prompt,
                    attempt_number=attempt_number,
                    seed=create_seed(
                        base_seed,
                        LABEL,
                        job.topic.topic_id,
                        job.sample_index,
                        attempt_number,
                    ),
                    created_at=utc_now(),
                )
            )

        if not requests:
            break
        if generator is None:
            raise RuntimeError("Model generator was not initialized")

        batch_number += 1
        logger.info(
            "batch_start batch=%d size=%d accepted=%d intended=%d pending=%d",
            batch_number,
            len(requests),
            len(completed_keys),
            intended_dialogues,
            len(pending),
        )
        completions, safe_batch_size, batch_was_split = (
            _generate_requests_resilient(
                generator=generator,
                requests=requests,
                parameters=parameters,
                logger=logger,
            )
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
            write_manifest(manifest_path, manifest)

        for request, completion in zip(requests, completions, strict=True):
            job = request.job
            raw_completion: str | None = completion
            cleaned_dialogue: str | None = None
            structurally_valid = False
            non_latin_letter_present: bool | None = None
            generation_error: str | None = None
            failure_reasons: list[str] = []

            try:
                cleaned_dialogue = clean_dialogue(raw_completion)
                structurally_valid, failure_reasons = validate_dialogue_structure(
                    cleaned_dialogue,
                    system_instruction_included=job.system_instruction_included,
                )
                non_latin_letter_present = contains_non_latin_letter(cleaned_dialogue)
                if non_latin_letter_present:
                    failure_reasons.append("non-Latin letter present")
            except Exception as error:
                generation_error = safe_error_message(error)
                failure_reasons = ["completion processing exception"]

            attempt_record = {
                "record_id": job.record_id,
                "label": LABEL,
                "topic_id": job.topic.topic_id,
                "topic": job.topic.topic,
                "sample_index": job.sample_index,
                "attempt_number": request.attempt_number,
                "seed": request.seed,
                "dialogue_type": job.dialogue_type,
                "system_instruction_included": job.system_instruction_included,
                "prompt_version": PROMPT_VERSION,
                "prompt": request.prompt,
                "raw_completion": raw_completion,
                "dialogue_person_ai": cleaned_dialogue,
                "structurally_valid": structurally_valid,
                "non_latin_letter_present": non_latin_letter_present,
                "failure_reasons": failure_reasons,
                "generation_error": generation_error,
                "created_at": request.created_at,
            }
            append_jsonl(attempts_path, attempt_record)
            request.state.max_attempt_number = request.attempt_number
            request.state.latest_attempt = attempt_record
            manifest["total_generation_attempts"] += 1

            if attempt_is_acceptable(attempt_record, max_attempts=max_attempts):
                accepted_record = accepted_record_from_attempt(
                    attempt_record,
                    model_name=model_name,
                    max_attempts=max_attempts,
                )
                append_jsonl(accepted_path, accepted_record)
                completed_keys.add(job.key)
                if accepted_record["accepted_after_max_attempts"]:
                    manifest["accepted_after_max_attempts"] += 1
            elif request.attempt_number >= max_attempts:
                failed_keys.add(job.key)
                logger.error(
                    "dialogue_failed record_id=%s attempts=%d reasons=%s",
                    job.record_id,
                    max_attempts,
                    ", ".join(failure_reasons),
                )
            else:
                pending.append(job)

        refresh_manifest_counts(
            manifest=manifest,
            completed_keys=completed_keys,
            failed_keys=failed_keys,
        )
        write_manifest(manifest_path, manifest)
        logger.info(
            "batch_complete batch=%d accepted=%d failed=%d attempts=%d pending=%d",
            batch_number,
            manifest["accepted_dialogues"],
            manifest["failed_dialogues"],
            manifest["total_generation_attempts"],
            len(pending),
        )


def _generate_requests_resilient(
    *,
    generator: LocalHuggingFaceGenerator,
    requests: list[AttemptRequest],
    parameters: GenerationParameters,
    logger: logging.Logger,
) -> tuple[list[str], int, bool]:
    """Generate a batch, splitting only genuine OOM failures."""

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
        return completions, len(requests), False
    except Exception as error:
        if not _is_out_of_memory_error(error):
            logger.error(
                "systemic_generation_failure size=%d error=%s",
                len(requests),
                error,
            )
            raise
        oom_message = safe_error_message(error)
        error.__traceback__ = None

    generator.clear_device_cache()
    logger.warning("oom_batch_split size=%d error=%s", len(requests), oom_message)
    if len(requests) == 1:
        raise RuntimeError(
            f"Model generation is out of memory at batch size 1: {oom_message}"
        )

    midpoint = len(requests) // 2
    left, left_safe_size, _ = _generate_requests_resilient(
        generator=generator,
        requests=requests[:midpoint],
        parameters=parameters,
        logger=logger,
    )
    right, right_safe_size, _ = _generate_requests_resilient(
        generator=generator,
        requests=requests[midpoint:],
        parameters=parameters,
        logger=logger,
    )
    return left + right, max(left_safe_size, right_safe_size), True


def clean_dialogue(raw_completion: str) -> str:
    """Perform only the minimal cleanup allowed by the dataset contract."""

    dialogue = raw_completion.strip()
    lines = dialogue.splitlines()
    if has_surrounding_markdown_fence(lines):
        dialogue = "\n".join(lines[1:-1]).strip()

    dialogue = re.sub(
        r"\A[ \t]*Dialogue[ \t]*:[ \t]*(?:\r?\n)?",
        "",
        dialogue,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return re.sub(
        r"\n[ \t]*\n(?:[ \t]*\n)+",
        "\n\n",
        dialogue,
    ).strip()


def validate_dialogue_structure(
    dialogue: str,
    *,
    system_instruction_included: bool,
) -> tuple[bool, list[str]]:
    """Validate speaker and wrapper structure without judging dialogue content."""

    reasons: list[str] = []

    def add_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not dialogue:
        return False, ["empty output"]

    lines = dialogue.splitlines()
    if has_surrounding_markdown_fence(lines):
        add_reason("dialogue remains wrapped in a Markdown code block")

    speakers: list[tuple[int, str]] = []
    non_code_line_indexes: set[int] = set()
    code_fence_character: str | None = None
    for line_index, line in enumerate(lines):
        fence_match = FENCE_LINE_RE.match(line)
        if fence_match:
            fence_character = fence_match.group("fence")[0]
            if code_fence_character is None:
                code_fence_character = fence_character
            elif code_fence_character == fence_character:
                code_fence_character = None
            continue
        if code_fence_character is not None:
            continue
        non_code_line_indexes.add(line_index)
        speaker_match = SPEAKER_RE.match(line)
        if speaker_match:
            speakers.append((line_index, speaker_match.group(1)))
            inline_fence_match = SPEAKER_INLINE_FENCE_RE.match(line)
            if inline_fence_match:
                code_fence_character = inline_fence_match.group("fence")[0]
            continue
        if OTHER_SPEAKER_RE.match(line):
            add_reason("unexpected speaker label")
        generic_label = GENERIC_LABEL_RE.match(line)
        if (
            generic_label
            and speakers
            and line_index > 0
            and not lines[line_index - 1].strip()
            and generic_label.group("label").casefold()
            not in CONTENT_HEADING_LABELS
        ):
            add_reason("unexpected speaker label")
        if DIALOGUE_MARKER_RE.fullmatch(line) or END_MARKER_RE.fullmatch(line):
            add_reason("multiple dialogues or dialogue wrapper marker")

    if code_fence_character is not None:
        add_reason("unclosed Markdown code block")

    labels = [label for _, label in speakers]
    if "Person" not in labels:
        add_reason("missing Person turn")
    if "AI" not in labels:
        add_reason("missing AI turn")
    if labels and labels[0] != "Person":
        add_reason("first speaker is not Person")

    expected_labels = [
        "Person" if index % 2 == 0 else "AI" for index in range(len(labels))
    ]
    if labels != expected_labels:
        add_reason("speaker turns do not alternate")
    if len(labels) < 2 or len(labels) > 12 or len(labels) % 2 != 0:
        add_reason("dialogue must contain 1 to 6 complete exchanges")

    if speakers:
        first_speaker_line = speakers[0][0]
        preface_lines = [
            line.strip() for line in lines[:first_speaker_line] if line.strip()
        ]
        if system_instruction_included:
            if (
                len(preface_lines) != 1
                or not re.search(
                    r"\b(?:you|the assistant)\b",
                    preface_lines[0] if preface_lines else "",
                    flags=re.IGNORECASE,
                )
            ):
                add_reason("missing required system instruction")
            if any(META_PREFACE_RE.match(line) for line in preface_lines):
                add_reason("unexpected text before dialogue")
            if first_speaker_line == 0 or lines[first_speaker_line - 1].strip():
                add_reason("missing blank line before speaker turn")
        elif preface_lines:
            add_reason("unexpected text before dialogue")

        for line_index, _ in speakers[1:]:
            if line_index == 0 or lines[line_index - 1].strip():
                add_reason("missing blank line before speaker turn")

        last_speaker_line = speakers[-1][0]
        if any(
            line_index in non_code_line_indexes
            and META_CONCLUSION_RE.match(lines[line_index])
            for line_index in range(last_speaker_line + 1, len(lines))
        ):
            add_reason("explanation after dialogue")

    return not reasons, reasons


def has_surrounding_markdown_fence(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    opening = OUTER_OPEN_FENCE_RE.fullmatch(lines[0].strip())
    closing = OUTER_CLOSE_FENCE_RE.fullmatch(lines[-1].strip())
    return bool(
        opening
        and closing
        and opening.group("fence")[0] == closing.group("fence")[0]
    )


def accepted_record_from_attempt(
    attempt: dict[str, Any],
    *,
    model_name: str,
    max_attempts: int,
) -> dict[str, Any]:
    dialogue = attempt.get("dialogue_person_ai")
    if not isinstance(dialogue, str) or not dialogue:
        raise ValueError("Cannot accept an attempt without a cleaned dialogue")
    if attempt.get("structurally_valid") is not True:
        raise ValueError("Cannot accept a structurally invalid dialogue")

    transcript = re.sub(r"(?m)^Person:", "Human:", dialogue)
    transcript = re.sub(r"(?m)^AI:", "Assistant:", transcript)
    attempt_number = int(attempt["attempt_number"])
    non_latin = bool(attempt["non_latin_letter_present"])
    accepted_after_max_attempts = (
        non_latin
        and max_attempts == MAX_TOTAL_ATTEMPTS
        and attempt_number == MAX_TOTAL_ATTEMPTS
    )
    return {
        "record_id": str(attempt["record_id"]),
        "label": LABEL,
        "topic_id": int(attempt["topic_id"]),
        "topic": str(attempt["topic"]),
        "sample_index": int(attempt["sample_index"]),
        "dialogue_type": str(attempt["dialogue_type"]),
        "system_instruction_included": bool(
            attempt["system_instruction_included"]
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt": str(attempt["prompt"]),
        "raw_completion": str(attempt["raw_completion"]),
        "dialogue_person_ai": dialogue,
        "transcript": transcript,
        "generator_model": model_name,
        "accepted_seed": int(attempt["seed"]),
        "attempt_count": attempt_number,
        "structurally_valid": True,
        "non_latin_letter_present": non_latin,
        "accepted_after_max_attempts": accepted_after_max_attempts,
        "created_at": utc_now(),
    }


def attempt_is_acceptable(attempt: dict[str, Any], *, max_attempts: int) -> bool:
    if attempt.get("generation_error") is not None:
        return False
    if attempt.get("structurally_valid") is not True:
        return False
    if attempt.get("non_latin_letter_present") is False:
        return True
    return (
        attempt.get("non_latin_letter_present") is True
        and max_attempts == MAX_TOTAL_ATTEMPTS
        and int(attempt["attempt_number"]) == MAX_TOTAL_ATTEMPTS
    )


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


def create_jobs(
    *,
    topics: list[Topic],
    samples_per_topic: int,
    base_seed: int,
) -> list[GenerationJob]:
    jobs = [
        GenerationJob(
            topic=topic,
            sample_index=sample_index,
            samples_per_topic=samples_per_topic,
        )
        for topic in topics
        for sample_index in range(samples_per_topic)
    ]
    random.Random(base_seed).shuffle(jobs)
    return jobs


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "directory": output_dir,
        "config": output_dir / "config.json",
        "manifest": output_dir / "manifest.json",
        "attempts": output_dir / "attempts.jsonl",
        "accepted": output_dir / "neutral_dialogues.jsonl",
        "log": output_dir / "generation.log",
    }


def preflight_output_state(
    *,
    paths: dict[str, Path],
    run_config: dict[str, Any],
    resume: bool,
) -> str:
    """Inspect output compatibility without creating or repairing files."""

    output_dir = paths["directory"]
    if not output_dir.exists():
        return "new"
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return "new_empty_directory"
    if not resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --resume or use "
            "another directory."
        )
    validate_persisted_config(paths["config"], run_config)
    return "compatible_resume"


def validate_existing_output(
    *,
    paths: dict[str, Path],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the required output files without modifying their timestamps."""

    output_dir = paths["directory"]
    if not output_dir.is_dir():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    persisted_config = validate_persisted_config(paths["config"], run_config)
    for name in ("manifest", "attempts", "accepted", "log"):
        path = paths[name]
        if not path.is_file():
            raise ValueError(f"Required neutral output file does not exist: {path}")
    return persisted_config


def validate_persisted_config(
    path: Path,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Neutral run configuration does not exist: {path}")
    existing = read_json_object(path)
    if existing.get("configuration_signature") != run_config.get(
        "configuration_signature"
    ):
        raise ValueError(f"Requested settings do not match existing config: {path}")
    return existing


def prepare_output(
    *,
    paths: dict[str, Path],
    run_config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    output_dir = paths["directory"]
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --resume or use "
            "another directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if paths["config"].exists():
        persisted_config = validate_persisted_config(paths["config"], run_config)
    else:
        existing_output_file = any(
            path.exists() for name, path in paths.items() if name != "directory"
        )
        if resume and existing_output_file:
            raise ValueError("Cannot resume output files without config.json")
        atomic_write_json(paths["config"], run_config)
        persisted_config = run_config

    paths["attempts"].touch(exist_ok=True)
    paths["accepted"].touch(exist_ok=True)
    paths["log"].touch(exist_ok=True)
    return persisted_config


def load_completed_keys(
    path: Path,
    *,
    repair: bool = True,
) -> tuple[set[RecordKey], int]:
    if repair:
        repair_incomplete_jsonl_tail(path)
    completed: set[RecordKey] = set()
    accepted_after_max_attempts = 0
    for line_number, row in iter_jsonl(path):
        if row.get("label") != LABEL:
            raise ValueError(f"Unexpected label in {path}:{line_number}")
        key = record_key(row, path=path, line_number=line_number)
        if key in completed:
            raise ValueError(f"Duplicate accepted key {key!r} in {path}")
        completed.add(key)
        accepted_after_max_attempts += int(
            bool(row.get("accepted_after_max_attempts", False))
        )
    return completed, accepted_after_max_attempts


def load_attempt_states(
    path: Path,
    *,
    completed_keys: set[RecordKey],
    intended_keys: set[RecordKey],
    max_attempts: int,
    repair: bool = True,
) -> tuple[dict[RecordKey, AttemptState], int]:
    if repair:
        repair_incomplete_jsonl_tail(path)
    states: dict[RecordKey, AttemptState] = {}
    seen_attempts: set[tuple[RecordKey, int]] = set()
    total_attempts = 0
    for line_number, row in iter_jsonl(path):
        total_attempts += 1
        if row.get("label") != LABEL:
            raise ValueError(f"Unexpected label in {path}:{line_number}")
        key = record_key(row, path=path, line_number=line_number)
        if key not in intended_keys:
            raise ValueError(f"Attempt key {key!r} is outside the intended run")
        try:
            attempt_number = int(row["attempt_number"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid attempt number in {path}:{line_number}"
            ) from error
        if not 1 <= attempt_number <= max_attempts:
            raise ValueError(f"Attempt number out of range in {path}:{line_number}")
        attempt_key = (key, attempt_number)
        if attempt_key in seen_attempts:
            raise ValueError(f"Duplicate attempt {attempt_key!r} in {path}")
        seen_attempts.add(attempt_key)
        if key in completed_keys:
            continue
        state = states.setdefault(key, AttemptState())
        if attempt_number > state.max_attempt_number:
            state.max_attempt_number = attempt_number
            state.latest_attempt = row
    return states, total_attempts


def validate_output_records(
    *,
    accepted_path: Path,
    attempts_path: Path,
    jobs: list[GenerationJob],
    model_name: str,
    base_seed: int,
    max_attempts: int,
) -> None:
    """Validate exact row schemas and link every accepted row to one attempt."""

    expected_by_key = {job.key: job for job in jobs}
    accepted_by_key: dict[RecordKey, dict[str, Any]] = {}
    for line_number, row in iter_jsonl(accepted_path):
        location = f"{accepted_path}:{line_number}"
        _validate_exact_record_fields(
            row=row,
            expected=ACCEPTED_RECORD_FIELDS,
            kind="Accepted record",
            location=location,
        )
        key = record_key(row, path=accepted_path, line_number=line_number)
        job = expected_by_key.get(key)
        if job is None:
            raise ValueError(f"Unexpected accepted key {key!r} in {location}")
        if key in accepted_by_key:
            raise ValueError(f"Duplicate accepted key {key!r} in {location}")

        prompt = render_neutral_dialogue_prompt(
            topic=job.topic.topic,
            dialogue_type=job.dialogue_type,
            include_system_instruction=job.system_instruction_included,
        )
        expected_fields: dict[str, Any] = {
            "record_id": job.record_id,
            "label": LABEL,
            "topic_id": job.topic.topic_id,
            "topic": job.topic.topic,
            "sample_index": job.sample_index,
            "dialogue_type": job.dialogue_type,
            "system_instruction_included": job.system_instruction_included,
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
            "generator_model": model_name,
            "structurally_valid": True,
        }
        _validate_expected_fields(
            row=row,
            expected=expected_fields,
            kind="Accepted record",
            location=location,
        )

        raw_completion = row["raw_completion"]
        dialogue = row["dialogue_person_ai"]
        transcript = row["transcript"]
        if not isinstance(raw_completion, str) or not isinstance(dialogue, str):
            raise ValueError(f"Accepted dialogue text is invalid in {location}")
        if clean_dialogue(raw_completion) != dialogue:
            raise ValueError(f"Accepted cleaned dialogue is inconsistent in {location}")
        structurally_valid, _ = validate_dialogue_structure(
            dialogue,
            system_instruction_included=job.system_instruction_included,
        )
        if not structurally_valid:
            raise ValueError(f"Accepted dialogue is structurally invalid in {location}")
        expected_transcript = re.sub(r"(?m)^Person:", "Human:", dialogue)
        expected_transcript = re.sub(
            r"(?m)^AI:",
            "Assistant:",
            expected_transcript,
        )
        if transcript != expected_transcript:
            raise ValueError(f"Accepted transcript conversion is invalid in {location}")

        non_latin = contains_non_latin_letter(dialogue)
        if row["non_latin_letter_present"] is not non_latin:
            raise ValueError(f"Accepted script flag is invalid in {location}")
        attempt_count = row["attempt_count"]
        accepted_seed = row["accepted_seed"]
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or not 1 <= attempt_count <= max_attempts
        ):
            raise ValueError(f"Accepted attempt_count is invalid in {location}")
        if isinstance(accepted_seed, bool) or not isinstance(accepted_seed, int):
            raise ValueError(f"Accepted seed is invalid in {location}")
        expected_seed = create_seed(
            base_seed,
            LABEL,
            job.topic.topic_id,
            job.sample_index,
            attempt_count,
        )
        if accepted_seed != expected_seed:
            raise ValueError(f"Accepted seed is invalid in {location}")
        expected_after_max = (
            non_latin
            and max_attempts == MAX_TOTAL_ATTEMPTS
            and attempt_count == MAX_TOTAL_ATTEMPTS
        )
        if row["accepted_after_max_attempts"] is not expected_after_max:
            raise ValueError(f"Accepted fallback flag is invalid in {location}")
        if not isinstance(row["created_at"], str) or not row["created_at"]:
            raise ValueError(f"Accepted timestamp is invalid in {location}")
        accepted_by_key[key] = row

    attempts_by_key: dict[RecordKey, list[dict[str, Any]]] = {}
    seen_attempts: set[tuple[RecordKey, int]] = set()
    for line_number, row in iter_jsonl(attempts_path):
        location = f"{attempts_path}:{line_number}"
        _validate_exact_record_fields(
            row=row,
            expected=ATTEMPT_RECORD_FIELDS,
            kind="Attempt record",
            location=location,
        )
        key = record_key(row, path=attempts_path, line_number=line_number)
        job = expected_by_key.get(key)
        if job is None:
            raise ValueError(f"Unexpected attempt key {key!r} in {location}")
        attempt_number = row["attempt_number"]
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or not 1 <= attempt_number <= max_attempts
        ):
            raise ValueError(f"Attempt number is invalid in {location}")
        attempt_key = (key, attempt_number)
        if attempt_key in seen_attempts:
            raise ValueError(f"Duplicate attempt {attempt_key!r} in {location}")
        seen_attempts.add(attempt_key)

        prompt = render_neutral_dialogue_prompt(
            topic=job.topic.topic,
            dialogue_type=job.dialogue_type,
            include_system_instruction=job.system_instruction_included,
        )
        expected_seed = create_seed(
            base_seed,
            LABEL,
            job.topic.topic_id,
            job.sample_index,
            attempt_number,
        )
        expected_fields = {
            "record_id": job.record_id,
            "label": LABEL,
            "topic_id": job.topic.topic_id,
            "topic": job.topic.topic,
            "sample_index": job.sample_index,
            "attempt_number": attempt_number,
            "seed": expected_seed,
            "dialogue_type": job.dialogue_type,
            "system_instruction_included": job.system_instruction_included,
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
        }
        _validate_expected_fields(
            row=row,
            expected=expected_fields,
            kind="Attempt record",
            location=location,
        )
        if not isinstance(row["created_at"], str) or not row["created_at"]:
            raise ValueError(f"Attempt timestamp is invalid in {location}")
        if not isinstance(row["failure_reasons"], list) or not all(
            isinstance(reason, str) for reason in row["failure_reasons"]
        ):
            raise ValueError(f"Attempt failure reasons are invalid in {location}")

        generation_error = row["generation_error"]
        if generation_error is None:
            raw_completion = row["raw_completion"]
            dialogue = row["dialogue_person_ai"]
            if not isinstance(raw_completion, str) or not isinstance(dialogue, str):
                raise ValueError(f"Attempt dialogue text is invalid in {location}")
            if clean_dialogue(raw_completion) != dialogue:
                raise ValueError(f"Attempt cleaned dialogue is invalid in {location}")
            structurally_valid, failure_reasons = validate_dialogue_structure(
                dialogue,
                system_instruction_included=job.system_instruction_included,
            )
            non_latin = contains_non_latin_letter(dialogue)
            if non_latin:
                failure_reasons.append("non-Latin letter present")
            if row["structurally_valid"] is not structurally_valid:
                raise ValueError(f"Attempt structural flag is invalid in {location}")
            if row["non_latin_letter_present"] is not non_latin:
                raise ValueError(f"Attempt script flag is invalid in {location}")
            if row["failure_reasons"] != failure_reasons:
                raise ValueError(f"Attempt failure reasons are invalid in {location}")
        elif not isinstance(generation_error, str) or not generation_error:
            raise ValueError(f"Attempt generation error is invalid in {location}")

        attempts_by_key.setdefault(key, []).append(row)

    for key, attempts in attempts_by_key.items():
        numbers = [int(attempt["attempt_number"]) for attempt in attempts]
        expected_numbers = list(range(1, max(numbers) + 1))
        if numbers != expected_numbers:
            raise ValueError(
                f"Attempt sequence for key {key!r} is {numbers!r}, "
                f"expected {expected_numbers!r}"
            )

    for key, accepted in accepted_by_key.items():
        attempts = attempts_by_key.get(key, [])
        attempt_count = int(accepted["attempt_count"])
        if len(attempts) != attempt_count:
            raise ValueError(f"Accepted attempt count is inconsistent for key {key!r}")
        selected = [
            attempt
            for attempt in attempts
            if attempt["attempt_number"] == attempt_count
        ]
        if len(selected) != 1:
            raise ValueError(f"Accepted attempt linkage is invalid for key {key!r}")
        selected_attempt = selected[0]
        if not attempt_is_acceptable(
            selected_attempt,
            max_attempts=max_attempts,
        ):
            raise ValueError(
                f"Accepted dialogue links to an unacceptable attempt for key {key!r}"
            )
        copied_fields = (
            "record_id",
            "label",
            "topic_id",
            "topic",
            "sample_index",
            "dialogue_type",
            "system_instruction_included",
            "prompt_version",
            "prompt",
            "raw_completion",
            "dialogue_person_ai",
            "structurally_valid",
            "non_latin_letter_present",
        )
        for field in copied_fields:
            if accepted[field] != selected_attempt[field]:
                raise ValueError(
                    f"Accepted field {field!r} does not match its attempt for "
                    f"key {key!r}"
                )
        if accepted["accepted_seed"] != selected_attempt["seed"]:
            raise ValueError(f"Accepted seed linkage is invalid for key {key!r}")


def _validate_exact_record_fields(
    *,
    row: dict[str, Any],
    expected: frozenset[str],
    kind: str,
    location: str,
) -> None:
    actual = set(row)
    if actual != expected:
        raise ValueError(
            f"{kind} fields do not match the neutral schema in {location}: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}"
        )


def _validate_expected_fields(
    *,
    row: dict[str, Any],
    expected: dict[str, Any],
    kind: str,
    location: str,
) -> None:
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(
                f"{kind} field {field!r} in {location} is "
                f"{row.get(field)!r}, expected {value!r}"
            )


def reconcile_flushed_attempts(
    *,
    jobs: list[GenerationJob],
    attempt_states: dict[RecordKey, AttemptState],
    completed_keys: set[RecordKey],
    accepted_path: Path,
    model_name: str,
    max_attempts: int,
    accepted_after_max_attempts: int,
    logger: logging.Logger,
) -> int:
    for job in jobs:
        if job.key in completed_keys:
            continue
        state = attempt_states.get(job.key)
        if state is None or state.latest_attempt is None:
            continue
        if not attempt_is_acceptable(state.latest_attempt, max_attempts=max_attempts):
            continue
        accepted = accepted_record_from_attempt(
            state.latest_attempt,
            model_name=model_name,
            max_attempts=max_attempts,
        )
        append_jsonl(accepted_path, accepted)
        completed_keys.add(job.key)
        accepted_after_max_attempts += int(accepted["accepted_after_max_attempts"])
        logger.info("resume_reconciled record_id=%s", job.record_id)
    return accepted_after_max_attempts


def terminal_failure_keys(
    *,
    jobs: list[GenerationJob],
    attempt_states: dict[RecordKey, AttemptState],
    completed_keys: set[RecordKey],
    max_attempts: int,
) -> set[RecordKey]:
    failed: set[RecordKey] = set()
    for job in jobs:
        if job.key in completed_keys:
            continue
        state = attempt_states.get(job.key)
        if state is not None and state.max_attempt_number >= max_attempts:
            failed.add(job.key)
    return failed


def build_run_config(
    *,
    args: argparse.Namespace,
    topics: list[Topic],
    topics_path: Path,
    topics_sha256: str,
    output_dir: Path,
    parameters: GenerationParameters,
    created_at: str,
) -> dict[str, Any]:
    topics_payload = [
        {"topic_id": topic.topic_id, "topic": topic.topic} for topic in topics
    ]
    signature = {
        "topic_source_path": str(topics_path),
        "topics": topics_payload,
        "generator_model": args.model,
        "model_revision": args.model_revision,
        "prompt_version": PROMPT_VERSION,
        "generation_request": GENERATION_REQUEST,
        "chat_message_layout": {
            "prompt_role": CHAT_PROMPT_ROLE,
            "system_instruction": CHAT_SYSTEM_INSTRUCTION,
            "generation_request_appended_to_prompt": True,
        },
        "prompt_template": NEUTRAL_DIALOGUE_PROMPT_TEMPLATE,
        "dialogue_types": list(DIALOGUE_TYPES),
        "dialogue_type_instructions": dict(DIALOGUE_TYPE_INSTRUCTIONS),
        "system_instruction_requirements": {
            "included": SYSTEM_INSTRUCTION_REQUIRED,
            "not_included": SYSTEM_INSTRUCTION_NOT_REQUIRED,
        },
        "optional_system_instruction_format": {
            "included": OPTIONAL_SYSTEM_INSTRUCTION_INCLUDED,
            "not_included": OPTIONAL_SYSTEM_INSTRUCTION_NOT_INCLUDED,
        },
        "neutral_label": LABEL,
        "samples_per_topic": args.samples_per_topic,
        "maximum_attempts": args.max_attempts,
        "base_seed": args.base_seed,
        "generation_settings": parameters.as_dict(),
        "device": args.device,
        "dtype": args.dtype,
        "output_directory": str(output_dir),
    }
    if args.attn_implementation is not None:
        signature["attn_implementation"] = args.attn_implementation
    if args.eos_token_policy != "tokenizer":
        signature["eos_token_policy"] = args.eos_token_policy
    if args.trust_remote_code:
        signature["trust_remote_code"] = True
    cli_arguments = {
        "model": args.model,
        "model_revision": args.model_revision,
        "topics": str(topics_path),
        "output_dir": str(output_dir),
        "samples_per_topic": args.samples_per_topic,
        "max_topics": args.max_topics,
        "max_attempts": args.max_attempts,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
        "base_seed": args.base_seed,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "eos_token_policy": args.eos_token_policy,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "require_model_revision": args.require_model_revision,
        "resume": args.resume,
    }
    return {
        "topic_source_path": str(topics_path),
        "topics_sha256": topics_sha256,
        "topics": topics_payload,
        "generator_model": args.model,
        "model_revision": args.model_revision,
        "tokenizer": {"name": args.model, "revision": args.model_revision},
        "complete_prompt_template": NEUTRAL_DIALOGUE_PROMPT_TEMPLATE,
        "generation_request": GENERATION_REQUEST,
        "chat_message_layout": {
            "prompt_role": CHAT_PROMPT_ROLE,
            "system_instruction": CHAT_SYSTEM_INSTRUCTION,
            "generation_request_appended_to_prompt": True,
        },
        "prompt_version": PROMPT_VERSION,
        "neutral_label": LABEL,
        "dialogue_types": list(DIALOGUE_TYPES),
        "dialogue_type_instructions": dict(DIALOGUE_TYPE_INSTRUCTIONS),
        "system_instruction_requirements": {
            "included": SYSTEM_INSTRUCTION_REQUIRED,
            "not_included": SYSTEM_INSTRUCTION_NOT_REQUIRED,
        },
        "optional_system_instruction_format": {
            "included": OPTIONAL_SYSTEM_INSTRUCTION_INCLUDED,
            "not_included": OPTIONAL_SYSTEM_INSTRUCTION_NOT_INCLUDED,
        },
        "dialogue_type_assignment_rule": (
            "(topic_id * samples_per_topic + sample_index) % len(DIALOGUE_TYPES)"
        ),
        "system_instruction_assignment_rule": "sample_index % 3 == 0",
        "samples_per_topic": args.samples_per_topic,
        "maximum_attempts": args.max_attempts,
        "base_seed": args.base_seed,
        "generation_settings": parameters.as_dict(),
        "attn_implementation": args.attn_implementation,
        "eos_token_policy": args.eos_token_policy,
        "trust_remote_code": args.trust_remote_code,
        "output_directory": str(output_dir),
        "cli_arguments": cli_arguments,
        "creation_timestamp": created_at,
        "configuration_signature": signature,
    }


def build_manifest(
    *,
    model_name: str,
    model_revision: str | None,
    base_seed: int,
    topics: list[Topic],
    samples_per_topic: int,
    max_attempts: int,
    intended_dialogues: int,
    accepted_dialogues: int,
    accepted_after_max_attempts: int,
    failed_dialogues: int,
    total_generation_attempts: int,
    parameters: GenerationParameters,
    created_at: str,
) -> dict[str, Any]:
    return {
        "generator_model": model_name,
        "model_revision": model_revision,
        "prompt_version": PROMPT_VERSION,
        "label": LABEL,
        "base_seed": base_seed,
        "number_of_topics": len(topics),
        "samples_per_topic": samples_per_topic,
        "max_attempts": max_attempts,
        "intended_dialogues": intended_dialogues,
        "accepted_dialogues": accepted_dialogues,
        "accepted_after_max_attempts": accepted_after_max_attempts,
        "failed_dialogues": failed_dialogues,
        "total_generation_attempts": total_generation_attempts,
        "generation_parameters": parameters.as_dict(),
        "status": "running",
        "created_at": created_at,
        "updated_at": utc_now(),
    }


def refresh_manifest_counts(
    *,
    manifest: dict[str, Any],
    completed_keys: set[RecordKey],
    failed_keys: set[RecordKey],
) -> None:
    manifest["accepted_dialogues"] = len(completed_keys)
    manifest["failed_dialogues"] = len(failed_keys)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_write_json(path, manifest)


def configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("emotionvectors.generation.neutral_dialogues")
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


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            yield line_number, row


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def record_key(row: dict[str, Any], *, path: Path, line_number: int) -> RecordKey:
    try:
        topic_id = int(row["topic_id"])
        sample_index = int(row["sample_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid record key in {path}:{line_number}") from error
    return topic_id, sample_index


def safe_error_message(error: Exception) -> str:
    message = str(error) or error.__class__.__name__
    message = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_HF_TOKEN]", message)
    message = re.sub(
        r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        message,
    )
    return message[:2000]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    current: BaseException | None = error
    while current is not None:
        if current.__class__.__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
            return True
        message = str(current).casefold()
        if "out of memory" in message or "not enough memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


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
