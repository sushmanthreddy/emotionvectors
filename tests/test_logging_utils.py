from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

from emotion_vectors.config import load_config
from emotion_vectors.logging_utils import (
    StructuredFormatter,
    build_run_manifest,
    set_global_seed,
    write_run_manifest,
)


def test_structured_formatter_emits_json_fields() -> None:
    record = logging.LogRecord(
        name="emotion_vectors.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="finished %s",
        args=("stage",),
        exc_info=None,
    )
    record.event = "stage_finished"
    record.count = 7

    payload = json.loads(StructuredFormatter().format(record))
    assert payload["message"] == "finished stage"
    assert payload["level"] == "INFO"
    assert payload["event"] == "stage_finished"
    assert payload["count"] == 7
    assert payload["timestamp"].endswith("+00:00")


def test_manifest_records_reproducibility_fields(tmp_path: Path) -> None:
    config = load_config(overrides={"paths.outputs": tmp_path})
    payload = build_run_manifest(config, "smoke", counts={"stories": 4})

    assert payload["config_hash"] == config.config_hash
    assert payload["seed"] == 0
    assert payload["counts"] == {"stories": 4}
    assert len(payload["git_sha"]) == 40
    assert payload["config"]["n_stories"] == 12

    path = write_run_manifest(tmp_path, config, "smoke", counts={"stories": 4})
    assert path == tmp_path / "run_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["counts"] == {"stories": 4}


def test_global_seed_reproduces_python_and_numpy() -> None:
    set_global_seed(19, deterministic_algorithms=False)
    first = (random.random(), np.random.random())
    set_global_seed(19, deterministic_algorithms=False)
    second = (random.random(), np.random.random())

    assert first == second


def test_manifest_and_structured_logs_are_strict_json_with_nonfinite_metrics() -> None:
    config = load_config()
    manifest = build_run_manifest(config, "strict-json", counts={"correlation": float("nan")})
    encoded = json.dumps(manifest, allow_nan=False)
    assert json.loads(encoded)["counts"]["correlation"] == "NaN"

    record = logging.LogRecord(
        name="emotion_vectors.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="metric",
        args=(),
        exc_info=None,
    )
    record.correlation = float("inf")
    payload = json.loads(StructuredFormatter().format(record))
    assert payload["correlation"] == "Infinity"
