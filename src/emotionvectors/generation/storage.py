"""Immediate JSONL persistence and atomic run metadata updates."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RecordKey = tuple[str, int, int]


@dataclass(frozen=True)
class CompletedIndex:
    """Identifiers and aggregate flags loaded from accepted record files."""

    keys: frozenset[RecordKey]
    accepted_after_max_attempts: int


@dataclass
class AttemptProgress:
    """Minimal resumable state retained for an unfinished record."""

    max_attempt_number: int = 0
    latest_nonempty: dict[str, Any] | None = None
    latest_script_compliant: dict[str, Any] | None = None


class DatasetStorage:
    """Own the requested dataset layout and its crash-safe writes."""

    def __init__(self, output_dir: Path, *, resume: bool) -> None:
        self.output_dir = output_dir
        self.resume = resume
        self.config_path = output_dir / "config.json"
        self.manifest_path = output_dir / "manifest.json"
        self.attempts_path = output_dir / "attempts.jsonl"
        self.records_dir = output_dir / "records"
        self.log_path = output_dir / "generation.log"

        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if output_dir.exists() and not resume and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another directory or pass --resume."
            )

    def prepare_run_config(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Create config metadata or verify that a resumed run is compatible."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            existing = _read_json_object(self.config_path)
            expected_comparable = {
                key: value for key, value in run_config.items() if key != "created_at"
            }
            existing_comparable = {
                key: value for key, value in existing.items() if key != "created_at"
            }
            if existing_comparable != expected_comparable:
                raise ValueError(
                    "The requested run configuration does not match the existing "
                    f"configuration in {self.config_path}"
                )
            return existing

        atomic_write_json(self.config_path, run_config)
        return run_config

    def initialize_layout(self, emotions: tuple[str, ...]) -> None:
        """Create every required append target without generating any records."""

        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_path.touch(exist_ok=True)
        self.log_path.touch(exist_ok=True)
        for emotion in emotions:
            (self.records_dir / f"{emotion}.jsonl").touch(exist_ok=True)

    def load_completed_index(self) -> CompletedIndex:
        """Stream accepted files, retaining only stable completed identifiers."""

        completed: set[RecordKey] = set()
        accepted_after_max_attempts = 0
        if not self.records_dir.exists():
            return CompletedIndex(frozenset(), 0)

        for path in sorted(self.records_dir.glob("*.jsonl")):
            repair_incomplete_jsonl_tail(path)
            for line_number, row in _iter_jsonl(path):
                key = _record_key(row, path=path, line_number=line_number)
                if key[0] != path.stem:
                    raise ValueError(
                        f"Emotion {key[0]!r} in {path}:{line_number} does not match "
                        f"the record filename {path.stem!r}"
                    )
                if key in completed:
                    raise ValueError(f"Duplicate accepted record key {key!r} in {path}")
                completed.add(key)
                accepted_after_max_attempts += int(
                    bool(row.get("accepted_after_max_attempts", False))
                )
        return CompletedIndex(frozenset(completed), accepted_after_max_attempts)

    def load_attempt_progress(
        self,
        completed_keys: frozenset[RecordKey],
    ) -> tuple[dict[RecordKey, AttemptProgress], int]:
        """Load only the state needed to continue unfinished attempt sequences."""

        progress: dict[RecordKey, AttemptProgress] = {}
        total_attempts = 0
        if not self.attempts_path.exists():
            return progress, total_attempts

        repair_incomplete_jsonl_tail(self.attempts_path)
        for line_number, row in _iter_jsonl(self.attempts_path):
            total_attempts += 1
            key = _record_key(row, path=self.attempts_path, line_number=line_number)
            if key in completed_keys:
                continue
            try:
                attempt_number = int(row["attempt_number"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid attempt number in {self.attempts_path}:{line_number}"
                ) from error
            if attempt_number < 1:
                raise ValueError(
                    f"Attempt number must be positive in {self.attempts_path}:{line_number}"
                )

            state = progress.setdefault(key, AttemptProgress())
            state.max_attempt_number = max(state.max_attempt_number, attempt_number)
            story = row.get("story")
            previous_number = (
                int(state.latest_nonempty["attempt_number"])
                if state.latest_nonempty is not None
                else 0
            )
            if isinstance(story, str) and story and attempt_number >= previous_number:
                state.latest_nonempty = dict(row)
            previous_compliant_number = (
                int(state.latest_script_compliant["attempt_number"])
                if state.latest_script_compliant is not None
                else 0
            )
            if (
                isinstance(story, str)
                and story
                and row.get("non_latin_letter_present") is False
                and attempt_number >= previous_compliant_number
            ):
                state.latest_script_compliant = dict(row)

        return progress, total_attempts

    def append_attempt(self, record: dict[str, Any]) -> None:
        append_jsonl(self.attempts_path, record)

    def append_accepted(self, emotion: str, record: dict[str, Any]) -> None:
        append_jsonl(self.records_dir / f"{emotion}.jsonl", record)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def read_manifest(self) -> dict[str, Any]:
        return _read_json_object(self.manifest_path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and flush one JSON object before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON object through a same-directory temporary file and rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def repair_incomplete_jsonl_tail(path: Path) -> None:
    """Drop only a partial final JSONL line left by an interrupted append."""

    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return

        end_position = handle.tell()
        search_position = end_position
        line_start = 0
        tail = bytearray()
        while search_position > 0:
            chunk_start = max(0, search_position - 8192)
            handle.seek(chunk_start)
            chunk = handle.read(search_position - chunk_start)
            newline_position = chunk.rfind(b"\n")
            if newline_position >= 0:
                line_start = chunk_start + newline_position + 1
                tail[:0] = chunk[newline_position + 1 :]
                break
            tail[:0] = chunk
            search_position = chunk_start

        try:
            json.loads(tail.decode("utf-8"))
            final_line_is_complete = True
        except (UnicodeDecodeError, json.JSONDecodeError):
            final_line_is_complete = False

        if final_line_is_complete:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        else:
            handle.truncate(line_start)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON object from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            yield line_number, row


def _record_key(row: dict[str, Any], *, path: Path, line_number: int) -> RecordKey:
    try:
        emotion = str(row["emotion"])
        topic_id = int(row["topic_id"])
        sample_index = int(row["sample_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid record identifier in {path}:{line_number}") from error
    return emotion, topic_id, sample_index


__all__ = [
    "AttemptProgress",
    "CompletedIndex",
    "DatasetStorage",
    "RecordKey",
    "append_jsonl",
    "atomic_write_json",
    "repair_incomplete_jsonl_tail",
]
