"""Command-line orchestration shared by the thin ``scripts/`` entry points."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, load_config
from .logging_utils import set_global_seed, setup_logging, write_run_manifest


def generate_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("Generate emotional stories and neutral dialogues (Stage 1)")
    parser.add_argument("--smoke", action="store_true", help="2 emotions x 3 topics x 2 items")
    args = parser.parse_args(argv)
    config = _load_cli_config(args, smoke=args.smoke)
    logger = _stage_logger(config, "01_generate", config.paths.raw)
    set_global_seed(config.seed, logger=logger)
    started = datetime.now(UTC)

    from .generate import LazyHuggingFaceGenerator, generate_raw_dataset

    backend = LazyHuggingFaceGenerator(config)
    summary = generate_raw_dataset(config, backend)
    counts = dataclasses.asdict(summary)
    write_run_manifest(
        config.paths.raw,
        config,
        "01_generate",
        counts=counts,
        started_at=started,
        completed_at=datetime.now(UTC),
        logger=logger,
    )
    logger.info("stage_complete", extra={"stage": "01_generate", **counts})
    return 0


def extract_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("Extract reduced all-layer residual activations (Stage 2)")
    parser.add_argument("--smoke", action="store_true", help="consume smoke Stage-1 artifacts")
    args = parser.parse_args(argv)
    config = _load_cli_config(args, smoke=args.smoke)
    logger = _stage_logger(config, "02_extract", config.paths.activations)
    set_global_seed(config.seed, logger=logger)
    started = datetime.now(UTC)

    from .activations import extract_activations

    summary = extract_activations(config)
    counts = _summary_counts(summary)
    write_run_manifest(
        config.paths.activations,
        config,
        "02_extract",
        counts=counts,
        started_at=started,
        completed_at=datetime.now(UTC),
        logger=logger,
    )
    logger.info("stage_complete", extra={"stage": "02_extract", **counts})
    return 0


def build_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("Build and neutral-PCA-denoise emotion vectors (Stage 3)")
    parser.add_argument("--smoke", action="store_true", help="consume smoke Stage-2 artifacts")
    args = parser.parse_args(argv)
    config = _load_cli_config(args, smoke=args.smoke)
    logger = _stage_logger(config, "03_build_vectors", config.paths.vectors)
    set_global_seed(config.seed, logger=logger)
    started = datetime.now(UTC)

    from .vectors import build_vectors

    result = build_vectors(config)
    counts = {
        "emotions": len(result.emotions),
        "layers": int(result.raw.shape[1]),
        "d_model": int(result.raw.shape[2]),
        "neutral_pcs_total": int(result.pca.n_components.sum()),
        "primary_layer": int(result.primary_layer),
    }
    write_run_manifest(
        config.paths.vectors,
        config,
        "03_build_vectors",
        counts=counts,
        started_at=started,
        completed_at=datetime.now(UTC),
        logger=logger,
    )
    logger.info("stage_complete", extra={"stage": "03_build_vectors", **counts})
    return 0


def verify_main(argv: Sequence[str] | None = None) -> int:
    parser = _parser("Run computational emotion-vector verification (Stage 4)")
    parser.add_argument("--smoke", action="store_true", help="use smoke artifacts and caps")
    parser.add_argument(
        "--checks",
        default="V1,V2,V3,V4,V5,V6",
        help="comma-separated subset of V1,V2,V3,V4,V5,V6",
    )
    args = parser.parse_args(argv)
    config = _load_cli_config(args, smoke=args.smoke)
    checks = tuple(item.strip().upper() for item in args.checks.split(",") if item.strip())
    if not checks:
        parser.error("--checks must name at least one of V1,V2,V3,V4,V5,V6")
    duplicates = sorted({check for check in checks if checks.count(check) > 1})
    if duplicates:
        parser.error(f"duplicate checks are not allowed: {', '.join(duplicates)}")
    invalid = sorted(set(checks) - {f"V{index}" for index in range(1, 7)})
    if invalid:
        parser.error(f"unknown checks: {', '.join(invalid)}")
    logger = _stage_logger(config, "04_verify", config.paths.outputs)
    set_global_seed(config.seed, logger=logger)
    started = datetime.now(UTC)

    from .verify import run_verifications

    # Verification dispatch loads the model lazily only for a non-cached check
    # that needs it. Geometry-only or fully resumed runs never allocate 64 GB.
    results = run_verifications(config, checks=checks)
    counts = {
        "checks_requested": len(checks),
        "checks_completed": len(results),
        "passed": sum(bool(_result_passed(result)) for result in results.values()),
    }
    write_run_manifest(
        config.paths.outputs,
        config,
        "04_verify",
        counts=counts,
        started_at=started,
        completed_at=datetime.now(UTC),
        extra={"checks": list(checks), "results": _jsonable(results)},
        logger=logger,
    )
    logger.info("stage_complete", extra={"stage": "04_verify", **counts})
    complete = counts["checks_completed"] == counts["checks_requested"]
    return 0 if complete and counts["passed"] == counts["checks_requested"] else 2


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a YAML value; repeatable and supports dotted path keys",
    )
    return parser


def _load_cli_config(args: argparse.Namespace, *, smoke: bool) -> ExperimentConfig:
    overrides = _parse_overrides(args.set)
    if smoke:
        overrides.setdefault("n_stories", 2)
        overrides.setdefault("generation_batch_size", 3)
        overrides.setdefault("generation_max_new_tokens", 1024)
        overrides.setdefault("batch_size", 2)
        overrides.setdefault("heldout_max_docs", 32)
        overrides.setdefault("attn_implementation", "sdpa")
    config = load_config(args.config, overrides=overrides)
    if smoke:
        heldout_index = config.heldout_datasets.index("isotonic_ha")
        config = dataclasses.replace(
            config,
            emotions=config.emotions[:2],
            topics=config.topics[:3],
            heldout_datasets=("isotonic_ha",),
            heldout_revisions=(config.heldout_revisions[heldout_index],),
        )
    return config


def _parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --set {item!r}; expected KEY=VALUE")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("Invalid --set: key cannot be empty")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def _stage_logger(config: ExperimentConfig, stage: str, directory: Path) -> Any:
    output_dir = config.resolve_path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    return setup_logging(log_file=output_dir / f"{stage}.jsonl")


def _summary_counts(summary: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(summary):
        values = dataclasses.asdict(summary)
    elif isinstance(summary, Mapping):
        values = dict(summary)
    else:
        values = {"result": str(summary)}
    return {str(key): _jsonable(value) for key, value in values.items()}


def _result_passed(result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("passed", result.get("pass", False)))
    return bool(getattr(result, "passed", getattr(result, "pass", False)))


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Use one of scripts/01_generate.py through scripts/04_verify.py")
