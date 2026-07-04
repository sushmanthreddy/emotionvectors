from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from emotion_vectors.generate import (
    DIALOGUE_DELIMITER,
    STORY_DELIMITER,
    generate_raw_dataset,
    normalize_dialogue,
    parse_dialogues,
    parse_stories,
)


@dataclass(frozen=True)
class _Paths:
    raw: str


@dataclass(frozen=True)
class _Config:
    root: Path
    paths: _Paths
    emotions: tuple[str, ...] = ("calm", "angry")
    topics: tuple[str, ...] = ("one", "two", "three")
    emotional_stories_prompt: str = "EMOTION={emotion};TOPIC={topic};N={n_stories}"
    neutral_dialogues_prompt: str = "NEUTRAL;TOPIC={topic};N={n_stories}"
    n_stories: int = 2
    generation_batch_size: int = 4
    generation_max_attempts: int = 3
    generation_lexical_policy: str = "warn"
    batch_size: int = 4
    seed: int = 0
    resume: bool = True

    @property
    def config_hash(self) -> str:
        return "test-config-hash"

    def resolve_path(self, value: str) -> Path:
        return self.root / value


class _Backend:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompts: list[str], *, seed: int) -> list[str]:
        del seed
        self.calls += len(prompts)
        responses = []
        for prompt in prompts:
            if prompt.startswith("NEUTRAL"):
                responses.append(
                    f"preface\n{DIALOGUE_DELIMITER}\nPerson: q1\n\nAI: a1\n"
                    f"{DIALOGUE_DELIMITER}\nPerson: q2\n\nAI: a2"
                )
            else:
                responses.append(
                    f"preface\n{STORY_DELIMITER}\nfirst indirect story\n"
                    f"{STORY_DELIMITER}\nsecond indirect story"
                )
        return responses


class _FailBackend:
    def generate(self, prompts: list[str], *, seed: int) -> list[str]:
        raise AssertionError(f"cache miss for {len(prompts)} prompts at seed {seed}")


class _QualityRetryBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.bad_story_emitted = False
        self.bad_neutral_emitted = False

    def generate(self, prompts: list[str], *, seed: int) -> list[str]:
        del seed
        self.calls += len(prompts)
        responses: list[str] = []
        for prompt in prompts:
            if prompt.startswith("NEUTRAL"):
                if "TOPIC=one" in prompt and not self.bad_neutral_emitted:
                    self.bad_neutral_emitted = True
                    responses.append(
                        f"{DIALOGUE_DELIMITER}\nPerson: q1\n\nAI: Thank you.\n"
                        f"{DIALOGUE_DELIMITER}\nPerson: q2\n\nAI: a2"
                    )
                else:
                    responses.append(
                        f"{DIALOGUE_DELIMITER}\nPerson: q1\n\nAI: a1\n"
                        f"{DIALOGUE_DELIMITER}\nPerson: q2\n\nAI: a2"
                    )
            elif "EMOTION=calm;TOPIC=one" in prompt and not self.bad_story_emitted:
                self.bad_story_emitted = True
                responses.append(
                    f"{STORY_DELIMITER}\na peaceful scene\n"
                    f"{STORY_DELIMITER}\nan indirect second scene"
                )
            else:
                responses.append(
                    f"{STORY_DELIMITER}\nfirst indirect story\n"
                    f"{STORY_DELIMITER}\nsecond indirect story"
                )
        return responses


def test_parsers_use_exact_delimiters_and_normalize_speaker_tags() -> None:
    assert parse_stories(f"ignored{STORY_DELIMITER} a {STORY_DELIMITER} b") == ["a", "b"]
    assert parse_stories("no delimiter") == []
    dialogues = parse_dialogues(f"ignored{DIALOGUE_DELIMITER}\nPerson: Personhood\n\nAI: response")
    assert dialogues == ["Human: Personhood\n\nAssistant: response"]
    assert normalize_dialogue("A Person: inline\nAI: yes") == "A Person: inline\nAssistant: yes"


def test_tiny_smoke_generation_matches_contract_and_resumes(tmp_path: Path) -> None:
    config = _Config(tmp_path, _Paths("data/raw"))
    backend = _Backend()
    result = generate_raw_dataset(config, backend)
    assert result.story_count == 12
    assert result.neutral_count == 6
    assert result.generated_shards == 9
    assert backend.calls == 9

    story_path = tmp_path / "data/raw/stories/calm.jsonl"
    rows = [json.loads(line) for line in story_path.read_text().splitlines()]
    assert rows[0] == {
        "emotion": "calm",
        "topic_id": 0,
        "topic": "one",
        "story_idx": 0,
        "text": "first indirect story",
    }
    neutral = [
        json.loads(line) for line in (tmp_path / "data/raw/neutral.jsonl").read_text().splitlines()
    ]
    assert neutral[0]["text"] == "Human: q1\n\nAssistant: a1"

    resumed = generate_raw_dataset(config, _FailBackend())
    assert resumed.generated_shards == 0
    assert resumed.cached_shards == 9


def test_partial_resume_uses_atomic_batch_cache_without_model_call(tmp_path: Path) -> None:
    config = _Config(tmp_path, _Paths("data/raw"))
    generate_raw_dataset(config, _Backend())
    shard = next(
        path
        for path in (tmp_path / "data/raw/.shards/test-config-hash/calm").glob("*.json")
        if path.name.startswith("000-")
    )
    original = shard.read_bytes()
    shard.unlink()

    result = generate_raw_dataset(config, _FailBackend())
    assert result.generated_shards == 1
    assert result.cached_shards == 8
    assert shard.read_bytes() == original


def test_nonprefix_topic_subset_preserves_canonical_topic_ids(tmp_path: Path) -> None:
    config = _Config(tmp_path, _Paths("data/raw"))
    result = generate_raw_dataset(
        config,
        _Backend(),
        emotions=("calm",),
        topics=("one", "three"),
    )
    assert result.story_count == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "data/raw/stories/calm.jsonl").read_text().splitlines()
    ]
    assert {row["topic_id"] for row in rows} == {0, 2}


def test_generation_retries_direct_synonyms_and_non_neutral_dialogues(tmp_path: Path) -> None:
    config = _Config(
        tmp_path,
        _Paths("data/raw"),
        generation_lexical_policy="strict",
    )
    backend = _QualityRetryBackend()

    result = generate_raw_dataset(config, backend)

    assert result.story_count == 12
    assert result.neutral_count == 6
    assert backend.bad_story_emitted and backend.bad_neutral_emitted
    calm_text = (tmp_path / "data/raw/stories/calm.jsonl").read_text(encoding="utf-8")
    neutral_text = (tmp_path / "data/raw/neutral.jsonl").read_text(encoding="utf-8")
    assert "peaceful" not in calm_text.lower()
    assert "thank you" not in neutral_text.lower()


def test_generation_stops_after_bounded_invalid_attempts(tmp_path: Path) -> None:
    config = _Config(
        tmp_path,
        _Paths("data/raw"),
        emotions=("calm",),
        topics=("one",),
        generation_batch_size=2,
        generation_max_attempts=2,
        generation_lexical_policy="strict",
    )

    class AlwaysExplicit:
        def generate(self, prompts: list[str], *, seed: int) -> list[str]:
            del seed
            responses = []
            for prompt in prompts:
                delimiter = DIALOGUE_DELIMITER if prompt.startswith("NEUTRAL") else STORY_DELIMITER
                body = (
                    "Person: q\n\nAI: Thank you"
                    if prompt.startswith("NEUTRAL")
                    else "a peaceful character"
                )
                responses.append(f"{delimiter}\n{body}\n{delimiter}\n{body}")
            return responses

    with pytest.raises(ValueError, match="remained invalid after 2"):
        generate_raw_dataset(config, AlwaysExplicit())


def test_warning_policy_accepts_synonyms_but_still_rejects_exact_target(tmp_path: Path) -> None:
    config = _Config(
        tmp_path,
        _Paths("data/raw"),
        emotions=("calm",),
        topics=("one",),
        generation_batch_size=2,
        generation_lexical_policy="warn",
    )

    result = generate_raw_dataset(config, _QualityRetryBackend())

    assert result.story_count == 2
    story_text = (tmp_path / "data/raw/stories/calm.jsonl").read_text(encoding="utf-8")
    assert "peaceful" in story_text.lower()
