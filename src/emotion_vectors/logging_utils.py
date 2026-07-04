"""Structured logging, deterministic seeding, and reproducible run manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig

_STANDARD_LOG_ATTRIBUTES = frozenset(logging.makeLogRecord({}).__dict__) | {
    "asctime",
    "message",
}


class StructuredFormatter(logging.Formatter):
    """Format each log record as one machine-readable JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRIBUTES and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(
            _strict_json_value(payload),
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )


def setup_logging(
    name: str = "emotion_vectors",
    *,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """Configure a logger with JSON output to stderr and optionally a JSONL file.

    Calling this function repeatedly for the same logger is idempotent: handlers
    installed by an earlier call are replaced instead of duplicated.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    for handler in tuple(logger.handlers):
        if getattr(handler, "_emotion_vectors_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = StructuredFormatter()
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler._emotion_vectors_handler = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    if log_file is not None:
        destination = Path(log_file).expanduser().resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._emotion_vectors_handler = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)
    return logger


configure_logging = setup_logging


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger without changing global logging state."""

    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit an informational structured event."""

    logger.info(event, extra={"event": event, **fields})


def set_global_seed(
    seed: int,
    *,
    deterministic_algorithms: bool = True,
    warn_only: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch and configure deterministic execution.

    ``PYTHONHASHSEED`` only affects interpreters started after it is set, so the
    manifest records it while Python/NumPy/PyTorch are also seeded immediately.
    PyTorch is imported lazily to keep configuration-only commands lightweight.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    seeded: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "python_hash_seed": str(seed),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "numpy": False,
        "torch": False,
        "cuda": False,
        "deterministic_algorithms": False,
        "warn_only": warn_only,
    }
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded["cuda"] = True
        torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=warn_only)
        seeded["deterministic_algorithms"] = deterministic_algorithms
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = deterministic_algorithms
    except ImportError:
        pass

    if logger is not None:
        log_event(
            logger,
            "random_seeds_configured",
            **seeded,
            nondeterministic_warning=(
                "PyTorch will warn when a nondeterministic operation is encountered"
                if deterministic_algorithms and warn_only
                else None
            ),
        )
    return seeded


seed_everything = set_global_seed
set_seed = set_global_seed


def git_sha(repository: str | Path | None = None) -> str:
    """Return the current Git commit SHA, or ``"unknown"`` outside a worktree."""

    cwd = Path(repository).resolve(strict=False) if repository is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return "unknown"
    value = result.stdout.strip()
    return value if len(value) == 40 else "unknown"


get_git_sha = git_sha


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    config: ExperimentConfig,
    stage: str,
    *,
    counts: Mapping[str, Any] | None = None,
    started_at: datetime | str | None = None,
    completed_at: datetime | str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical manifest payload for a pipeline stage."""

    if not stage.strip():
        raise ValueError("stage must be a non-empty string")
    now = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "stage": stage,
        "config_hash": config.config_hash,
        "git_sha": git_sha(config.project_root),
        "seed": config.seed,
        "counts": dict(counts or {}),
        "started_at": _timestamp(started_at or now),
        "completed_at": _timestamp(completed_at or now),
        "created_at": now.isoformat(),
        "config": config.as_dict(include_assets=True),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
    }
    if extra:
        collisions = manifest.keys() & extra.keys()
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Manifest extra fields collide with reserved fields: {names}")
        manifest.update(extra)
    # Validate strict RFC-8259 serializability before touching disk. Scientific
    # diagnostics sometimes contain undefined correlations; encode those as
    # descriptive strings rather than JavaScript-only NaN literals.
    strict_manifest = _strict_json_value(manifest)
    json.dumps(
        strict_manifest,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
        allow_nan=False,
    )
    return strict_manifest


def write_run_manifest(
    output_dir: str | Path,
    config: ExperimentConfig,
    stage: str,
    *,
    counts: Mapping[str, Any] | None = None,
    started_at: datetime | str | None = None,
    completed_at: datetime | str | None = None,
    extra: Mapping[str, Any] | None = None,
    filename: str = "run_manifest.json",
    logger: logging.Logger | None = None,
) -> Path:
    """Atomically write a stage run manifest into its output directory."""

    directory = config.resolve_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    manifest = build_run_manifest(
        config,
        stage,
        counts=counts,
        started_at=started_at,
        completed_at=completed_at,
        extra=extra,
    )
    _write_json_atomic(destination, manifest)
    if logger is not None:
        log_event(
            logger,
            "run_manifest_written",
            stage=stage,
            path=str(destination),
            config_hash=config.config_hash,
            git_sha=manifest["git_sha"],
            seed=config.seed,
            counts=manifest["counts"],
        )
    return destination


write_manifest = write_run_manifest


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("timestamp strings must not be empty")
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _timestamp(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _timestamp(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if hasattr(value, "item"):
        try:
            return _strict_json_value(value.item())
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "StructuredFormatter",
    "build_run_manifest",
    "configure_logging",
    "get_git_sha",
    "get_logger",
    "git_sha",
    "log_event",
    "seed_everything",
    "set_global_seed",
    "set_seed",
    "setup_logging",
    "sha256_file",
    "write_manifest",
    "write_run_manifest",
]
