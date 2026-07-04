"""Stage 1: resumable generation of emotional stories and neutral dialogues.

The public entry point, :func:`generate_raw_dataset`, accepts a generation backend so
the orchestration can use a Hugging Face model while tests can use a tiny deterministic
stub.  Each model response is cached as an input-addressed shard before the canonical
JSONL files are assembled; a crash can therefore never force completed generations to
be repeated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .lexicon import EMOTION_SYNONYMS, forbidden_terms

if TYPE_CHECKING:
    from .config import ExperimentConfig

LOGGER = logging.getLogger(__name__)

STORY_DELIMITER = "<NEW STORY>"
DIALOGUE_DELIMITER = "<NEW DIALOGUE>"
GENERATION_SCHEMA_VERSION = 2
_GENERATION_IMPLEMENTATION_HASH = hashlib.sha256(
    Path(__file__).read_bytes() + Path(__file__).with_name("lexicon.py").read_bytes()
).hexdigest()


class GenerationBackend(Protocol):
    """Minimal interface needed by Stage 1."""

    def generate(self, prompts: Sequence[str], *, seed: int) -> Sequence[str]:
        """Return one generated continuation for every prompt."""


@dataclass(frozen=True)
class GenerationSummary:
    """Counts emitted by a generation run."""

    story_count: int
    neutral_count: int
    generated_shards: int
    cached_shards: int


class HuggingFaceGenerator:
    """Batched deterministic-seed adapter around a causal Hugging Face model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompts: Sequence[str], *, seed: int) -> Sequence[str]:
        if not prompts:
            return []

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        tokenizer = self.tokenizer
        previous_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
        device = _model_input_device(self.model)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])

        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-6),
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            tokenizer.padding_side = previous_padding_side

        continuations = generated[:, prompt_width:]
        return tokenizer.batch_decode(continuations, skip_special_tokens=True)


class LazyHuggingFaceGenerator:
    """Load the large generator only if Stage 1 encounters a cache miss."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._backend: HuggingFaceGenerator | None = None

    def generate(self, prompts: Sequence[str], *, seed: int) -> Sequence[str]:
        if self._backend is None:
            from .model import load_model

            bundle = load_model(self.config, model_name=self.config.generator)
            self._backend = HuggingFaceGenerator(
                bundle.model,
                bundle.tokenizer,
                max_new_tokens=self.config.generation_max_new_tokens,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
            )
        return self._backend.generate(prompts, seed=seed)


def _model_input_device(model: Any) -> Any:
    """Find a safe input device for regular and Accelerate-dispatched models."""

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            if device not in {"cpu", "disk"}:
                return device
    return next(model.parameters()).device


def parse_delimited(text: str, delimiter: str) -> list[str]:
    """Parse non-empty records following an exact model-output delimiter.

    Models occasionally preface an answer with a sentence.  Text before the first
    delimiter is intentionally discarded because it is not part of the requested data.
    """

    if delimiter not in text:
        return []
    return [part.strip() for part in text.split(delimiter)[1:] if part.strip()]


def parse_stories(text: str, expected: int | None = None) -> list[str]:
    stories = parse_delimited(text, STORY_DELIMITER)
    return stories if expected is None else stories[:expected]


_PERSON_TURN = re.compile(r"(?m)^(\s*)Person:")
_AI_TURN = re.compile(r"(?m)^(\s*)AI:")


def normalize_dialogue(text: str) -> str:
    """Map only speaker tags, leaving words such as ``Personhood`` untouched."""

    return _AI_TURN.sub(r"\1Assistant:", _PERSON_TURN.sub(r"\1Human:", text.strip()))


def parse_dialogues(text: str, expected: int | None = None) -> list[str]:
    dialogues = [normalize_dialogue(item) for item in parse_delimited(text, DIALOGUE_DELIMITER)]
    return dialogues if expected is None else dialogues[:expected]


def generate_raw_dataset(
    config: ExperimentConfig,
    backend: GenerationBackend,
    *,
    emotions: Sequence[str] | None = None,
    topics: Sequence[str] | None = None,
    n_stories: int | None = None,
    strict_counts: bool = True,
) -> GenerationSummary:
    """Generate and assemble all Stage-1 artifacts.

    A single backend call is made per ``(emotion, topic)`` and per neutral topic.
    Calls may be batched, but every prompt still has one independently decoded output.
    """

    selected_emotions = tuple(emotions if emotions is not None else config.emotions)
    selected_topics = tuple(topics if topics is not None else config.topics)
    requested = int(n_stories if n_stories is not None else config.n_stories)
    lexical_policy = str(getattr(config, "generation_lexical_policy", "warn"))
    if requested < 1:
        raise ValueError("n_stories must be positive")
    if not selected_emotions or not selected_topics:
        raise ValueError("At least one emotion and one topic are required")
    if lexical_policy not in {"strict", "warn", "off"}:
        raise ValueError("generation_lexical_policy must be strict, warn, or off")
    canonical_topic_ids = {topic: index for index, topic in enumerate(config.topics)}
    unknown_topics = [topic for topic in selected_topics if topic not in canonical_topic_ids]
    if unknown_topics:
        raise ValueError(f"Selected topics are not in config.topics: {unknown_topics!r}")

    raw_dir = config.resolve_path(config.paths.raw)
    stories_dir = raw_dir / "stories"
    # Config hash + job cache key makes every shard content addressed. Changing
    # the generator, seed, sampling parameters, prompt assets, or model therefore
    # cannot silently reuse an incompatible response.
    shard_root = raw_dir / ".shards" / config.config_hash
    stories_dir.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)

    jobs: list[_GenerationJob] = []
    for emotion in selected_emotions:
        for topic in selected_topics:
            topic_id = canonical_topic_ids[topic]
            prompt = config.emotional_stories_prompt.format(
                n_stories=requested, topic=topic, emotion=emotion
            )
            jobs.append(
                _GenerationJob(
                    kind="story",
                    topic_id=topic_id,
                    topic=topic,
                    prompt=prompt,
                    expected=requested,
                    emotion=emotion,
                    lexical_policy=lexical_policy,
                )
            )
    for topic in selected_topics:
        topic_id = canonical_topic_ids[topic]
        prompt = config.neutral_dialogues_prompt.format(n_stories=requested, topic=topic)
        jobs.append(
            _GenerationJob(
                kind="neutral",
                topic_id=topic_id,
                topic=topic,
                prompt=prompt,
                expected=requested,
                lexical_policy=lexical_policy,
            )
        )

    generated, cached = _materialize_jobs(
        jobs,
        backend,
        shard_root=shard_root,
        batch_size=int(config.generation_batch_size),
        seed=int(config.seed),
        resume=bool(config.resume),
        strict_counts=strict_counts,
        max_attempts=int(getattr(config, "generation_max_attempts", 3)),
    )

    story_count = 0
    for emotion in selected_emotions:
        records: list[dict[str, object]] = []
        for job in jobs:
            if job.kind != "story" or job.emotion != emotion:
                continue
            records.extend(_read_shard(job.shard_path(shard_root))["records"])
        records.sort(key=lambda row: (int(row["topic_id"]), int(row["story_idx"])))
        _write_jsonl_atomic(stories_dir / f"{emotion}.jsonl", records)
        story_count += len(records)

    neutral_records: list[dict[str, object]] = []
    for job in jobs:
        if job.kind == "neutral":
            neutral_records.extend(_read_shard(job.shard_path(shard_root))["records"])
    neutral_records.sort(key=lambda row: (int(row["topic_id"]), int(row["dialogue_idx"])))
    _write_jsonl_atomic(raw_dir / "neutral.jsonl", neutral_records)

    return GenerationSummary(
        story_count=story_count,
        neutral_count=len(neutral_records),
        generated_shards=generated,
        cached_shards=cached,
    )


@dataclass(frozen=True)
class _GenerationJob:
    kind: str
    topic_id: int
    topic: str
    prompt: str
    expected: int
    emotion: str | None = None
    lexical_policy: str = "warn"

    @property
    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "schema": GENERATION_SCHEMA_VERSION,
                "implementation_sha256": _GENERATION_IMPLEMENTATION_HASH,
                "kind": self.kind,
                "emotion": self.emotion,
                "topic_id": self.topic_id,
                "topic": self.topic,
                "prompt": self.prompt,
                "expected": self.expected,
                "lexical_policy": self.lexical_policy,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def shard_path(self, root: Path) -> Path:
        parent = root / (self.emotion if self.kind == "story" else "neutral")
        return parent / f"{self.topic_id:03d}-{self.cache_key[:16]}.json"

    def records_from_output(self, output: str) -> list[dict[str, object]]:
        if self.kind == "story":
            parsed = parse_stories(output, self.expected)
            return [
                {
                    "emotion": self.emotion,
                    "topic_id": self.topic_id,
                    "topic": self.topic,
                    "story_idx": index,
                    "text": text,
                }
                for index, text in enumerate(parsed)
            ]
        parsed = parse_dialogues(output, self.expected)
        return [
            {
                "topic_id": self.topic_id,
                "topic": self.topic,
                "dialogue_idx": index,
                "text": text,
            }
            for index, text in enumerate(parsed)
        ]


def _materialize_jobs(
    jobs: Sequence[_GenerationJob],
    backend: GenerationBackend,
    *,
    shard_root: Path,
    batch_size: int,
    seed: int,
    resume: bool,
    strict_counts: bool,
    max_attempts: int,
) -> tuple[int, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    cached = sum(bool(resume and _valid_shard(job.shard_path(shard_root), job)) for job in jobs)
    generated = 0
    for start in range(0, len(jobs), batch_size):
        batch = list(jobs[start : start + batch_size])
        existing = [resume and _valid_shard(job.shard_path(shard_root), job) for job in batch]
        missing = [index for index, is_cached in enumerate(existing) if not is_cached]
        if not missing:
            continue

        payloads = (
            _read_batch_cache(_batch_cache_path(shard_root, batch, seed + start), batch)
            if resume
            else None
        )
        if payloads is None and not any(existing):
            outputs = list(backend.generate([job.prompt for job in batch], seed=seed + start))
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Generator returned {len(outputs)} responses for {len(batch)} prompts"
                )
            parsed_payloads: list[dict[str, Any] | None] = []
            parse_errors: list[ValueError] = []
            for job, output in zip(batch, outputs, strict=True):
                try:
                    parsed_payloads.append(
                        _payload_from_output(job, output, strict_counts=strict_counts)
                    )
                except ValueError as error:
                    parsed_payloads.append(None)
                    parse_errors.append(error)
            if parse_errors:
                # Preserve every valid neighbor before surfacing the malformed
                # response. Invalid jobs are retried individually with distinct,
                # deterministic seeds, so one malformed decode cannot permanently
                # block an otherwise complete batch.
                for index, payload in enumerate(parsed_payloads):
                    if payload is None:
                        continue
                    _write_json_atomic(batch[index].shard_path(shard_root), payload)
                    generated += 1
                for index, payload in enumerate(parsed_payloads):
                    if payload is not None:
                        continue
                    job = batch[index]
                    payload = _generate_job_with_retries(
                        job,
                        backend,
                        base_seed=seed + int(job.cache_key[:8], 16),
                        strict_counts=strict_counts,
                        max_attempts=max_attempts,
                    )
                    _write_json_atomic(job.shard_path(shard_root), payload)
                    generated += 1
                continue
            payloads = [payload for payload in parsed_payloads if payload is not None]
            # Persist the whole decoded batch before individual shards. A crash
            # while materializing shards can then resume without another model call.
            _write_batch_cache(_batch_cache_path(shard_root, batch, seed + start), batch, payloads)

        if payloads is None:
            # A partial legacy/corrupt batch has no atomic batch cache. Never
            # regenerate valid shards; generate each missing job with a stable seed.
            payloads = [None] * len(batch)
            for index in missing:
                job = batch[index]
                stable_seed = seed + int(job.cache_key[:8], 16)
                payloads[index] = _generate_job_with_retries(
                    job,
                    backend,
                    base_seed=stable_seed,
                    strict_counts=strict_counts,
                    max_attempts=max_attempts,
                )

        for index in missing:
            job = batch[index]
            payload = payloads[index]
            if payload is None:
                raise RuntimeError("Generation batch cache omitted a required payload")
            _write_json_atomic(job.shard_path(shard_root), payload)
            generated += 1
            LOGGER.info(
                "generated_shard",
                extra={
                    "kind": job.kind,
                    "emotion": job.emotion,
                    "topic_id": job.topic_id,
                    "records": len(payload["records"]),
                },
            )
    return generated, cached


def _generate_job_with_retries(
    job: _GenerationJob,
    backend: GenerationBackend,
    *,
    base_seed: int,
    strict_counts: bool,
    max_attempts: int,
) -> dict[str, Any]:
    last_error: ValueError | None = None
    for attempt in range(max_attempts):
        attempt_seed = base_seed + attempt * 1_000_003
        outputs = list(backend.generate([job.prompt], seed=attempt_seed))
        if len(outputs) != 1:
            raise RuntimeError("Generator must return one response for one prompt")
        try:
            return _payload_from_output(job, outputs[0], strict_counts=strict_counts)
        except ValueError as error:
            last_error = error
            LOGGER.warning(
                "retrying_invalid_generation",
                extra={
                    "kind": job.kind,
                    "emotion": job.emotion,
                    "topic_id": job.topic_id,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "error": str(error),
                },
            )
    assert last_error is not None
    raise ValueError(
        f"Generation remained invalid after {max_attempts} deterministic attempts: {last_error}"
    ) from last_error


def _payload_from_output(
    job: _GenerationJob, output: object, *, strict_counts: bool
) -> dict[str, Any]:
    records = job.records_from_output(str(output))
    if strict_counts and len(records) != job.expected:
        raise ValueError(
            f"{job.kind} shard {job.topic_id} ({job.emotion or 'neutral'}) "
            f"contained {len(records)} records; expected {job.expected}. "
            "The incomplete response was not cached."
        )
    if job.kind == "story" and job.emotion is not None:
        exact_pattern = re.compile(
            rf"(?<![a-z]){re.escape(job.emotion.lower())}(?![a-z])",
            flags=re.IGNORECASE,
        )
        exact_violations = [
            str(record["text"]) for record in records if exact_pattern.search(str(record["text"]))
        ]
        if exact_violations:
            raise ValueError(
                f"story shard {job.topic_id} explicitly names target emotion "
                f"{job.emotion!r}; the response was not cached"
            )
        synonym_violations = sorted(
            {
                term
                for record in records
                for term in forbidden_terms(str(record["text"]), (job.emotion,))
                if term != job.emotion.lower()
            }
        )
        if synonym_violations and job.lexical_policy == "strict":
            raise ValueError(
                f"story shard {job.topic_id} explicitly names {job.emotion!r} or a direct "
                f"synonym ({', '.join(synonym_violations)}); the response was not cached"
            )
        if synonym_violations and job.lexical_policy == "warn":
            LOGGER.warning(
                "accepted_story_synonym_warning",
                extra={
                    "emotion": job.emotion,
                    "topic_id": job.topic_id,
                    "terms": synonym_violations,
                    "lexical_policy": job.lexical_policy,
                },
            )
    if job.kind == "neutral":
        violations = sorted(
            {
                term
                for record in records
                for term in forbidden_terms(
                    str(record["text"]),
                    EMOTION_SYNONYMS,
                    neutral=True,
                )
            }
        )
        malformed = [
            record
            for record in records
            if "Human:" not in str(record["text"]) or "Assistant:" not in str(record["text"])
        ]
        if malformed:
            raise ValueError(
                f"neutral shard {job.topic_id} is missing Human:/Assistant: speaker turns; "
                "the response was not cached"
            )
        if violations and job.lexical_policy == "strict":
            raise ValueError(
                f"neutral shard {job.topic_id} contains forbidden affective language "
                f"({', '.join(violations)}); the response was not cached"
            )
        if violations and job.lexical_policy == "warn":
            LOGGER.warning(
                "accepted_neutral_lexical_warning",
                extra={
                    "topic_id": job.topic_id,
                    "terms": violations,
                    "lexical_policy": job.lexical_policy,
                },
            )
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "cache_key": job.cache_key,
        "kind": job.kind,
        "emotion": job.emotion,
        "topic_id": job.topic_id,
        "records": records,
    }


def _batch_cache_path(shard_root: Path, jobs: Sequence[_GenerationJob], batch_seed: int) -> Path:
    encoded = json.dumps(
        {"keys": [job.cache_key for job in jobs], "seed": batch_seed},
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return shard_root / ".batches" / f"{digest}.json"


def _write_batch_cache(
    path: Path, jobs: Sequence[_GenerationJob], payloads: Sequence[dict[str, Any]]
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "job_cache_keys": [job.cache_key for job in jobs],
            "payloads": list(payloads),
        },
    )


def _read_batch_cache(path: Path, jobs: Sequence[_GenerationJob]) -> list[dict[str, Any]] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("payloads")
        if (
            payload.get("schema_version") != GENERATION_SCHEMA_VERSION
            or payload.get("job_cache_keys") != [job.cache_key for job in jobs]
            or not isinstance(entries, list)
            or len(entries) != len(jobs)
            or not all(
                isinstance(entry, dict)
                and entry.get("cache_key") == job.cache_key
                and len(entry.get("records", [])) == job.expected
                and all(_valid_record(record, job) for record in entry.get("records", []))
                for entry, job in zip(entries, jobs, strict=True)
            )
        ):
            return None
        return entries
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _valid_shard(path: Path, job: _GenerationJob) -> bool:
    try:
        payload = _read_shard(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    records = payload.get("records", [])
    return (
        payload.get("schema_version") == GENERATION_SCHEMA_VERSION
        and payload.get("cache_key") == job.cache_key
        and len(records) == job.expected
        and all(_valid_record(record, job) for record in records)
    )


def _valid_record(record: Any, job: _GenerationJob) -> bool:
    if not isinstance(record, dict):
        return False
    common = (
        record.get("topic_id") == job.topic_id
        and record.get("topic") == job.topic
        and isinstance(record.get("text"), str)
        and bool(record["text"].strip())
    )
    if not common:
        return False
    if job.kind == "story":
        return (
            record.get("emotion") == job.emotion
            and isinstance(record.get("story_idx"), int)
            and 0 <= record["story_idx"] < job.expected
        )
    return (
        isinstance(record.get("dialogue_idx"), int) and 0 <= record["dialogue_idx"] < job.expected
    )


def _read_shard(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Invalid generation shard: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text_write(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_jsonl_atomic(path: Path, records: Iterable[dict[str, object]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_text_write(path, text)


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def callable_backend(function: Callable[[Sequence[str], int], Sequence[str]]) -> GenerationBackend:
    """Adapt a simple callable, mainly useful for smoke runs and downstream users."""

    class _CallableBackend:
        def generate(self, prompts: Sequence[str], *, seed: int) -> Sequence[str]:
            return function(prompts, seed)

    return _CallableBackend()
