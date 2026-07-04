from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from numpy.lib import format as npy_format

from em_organism_dir.emotion_analysis.config import (
    DEFAULT_CONFIG_PATH,
    REFERENCE_EMOTIONS,
    RQ1ConfigError,
    load_config,
)
from em_organism_dir.emotion_analysis.data import (
    RQ1DataError,
    build_reuse_dataset,
    load_dataset_index,
    open_aligned_activations,
    sha256_file,
    write_dataset_artifacts,
)


def _npy_header(shape: tuple[int, ...], dtype: np.dtype[object]) -> bytes:
    buffer = io.BytesIO()
    npy_format.write_array_header_1_0(
        buffer,
        {"descr": npy_format.dtype_to_descr(dtype), "fortran_order": False, "shape": shape},
    )
    return buffer.getvalue()


def _small_npy(array: np.ndarray[object, object]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_activation_archive(
    path: Path,
    rows: list[tuple[int, int]] | None,
    *,
    first_dimension: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("vectors.npy", _npy_header((first_dimension, 48, 5120), np.dtype("f4")))
        if rows is not None:
            metadata = np.array(rows, dtype=[("topic_id", "<i8"), ("story_idx", "<i8")])
            archive.writestr("meta.npy", _small_npy(metadata))
    cache = {
        "archive_sha256": _sha256(path),
        "archive_size": path.stat().st_size,
        "cache_key": f"fixture-{path.stem}",
        "has_meta": rows is not None,
        "schema_version": 1,
        "vector_dtype": "float32",
        "vector_shape": [first_dimension, 48, 5120],
    }
    path.with_suffix(".cache.json").write_text(json.dumps(cache), encoding="utf-8")


def _write_vector_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (12, 48, 5120)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("raw.npy", _npy_header(shape, np.dtype("f4")))
        archive.writestr("denoised.npy", _npy_header(shape, np.dtype("f4")))
        archive.writestr("emotions.npy", _small_npy(np.asarray(REFERENCE_EMOTIONS)))
        archive.writestr("primary_layer.npy", _small_npy(np.asarray(32, dtype=np.int64)))
    cache = {"schema_version": 2, "output_sha256": _sha256(path)}
    path.with_suffix(".cache.json").write_text(json.dumps(cache), encoding="utf-8")


def _run_manifest() -> dict[str, object]:
    return {
        "stage": "fixture",
        "config_hash": "fixture-config",
        "config": {
            "emotions": list(REFERENCE_EMOTIONS),
            "model_name": "Qwen/Qwen2.5-14B-Instruct",
            "model_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
            "token_start": 50,
            "topics": [f"shared topic {topic_id}" for topic_id in range(20)],
        },
    }


def _write_fixture(root: Path, *, missing: tuple[str, int, int] = ("furious", 2, 3)) -> Path:
    raw_dir = root / "raw"
    activation_dir = root / "activations"
    vector_path = root / "vectors" / "emotion_vectors.npz"
    (raw_dir / "stories").mkdir(parents=True)
    all_cells = [(topic_id, story_idx) for topic_id in range(20) for story_idx in range(4)]

    for emotion in REFERENCE_EMOTIONS:
        lines = []
        for topic_id, story_idx in all_cells:
            record = {
                "emotion": emotion,
                "story_idx": story_idx,
                "text": f"{emotion} fixture text for topic {topic_id}, story {story_idx}",
                "topic": f"shared topic {topic_id}",
                "topic_id": topic_id,
            }
            lines.append(json.dumps(record))
        (raw_dir / "stories" / f"{emotion}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        metadata_rows = [
            cell
            for cell in all_cells
            if not (emotion == missing[0] and cell == (missing[1], missing[2]))
        ]
        _write_activation_archive(
            activation_dir / "stories" / f"{emotion}.npz",
            metadata_rows,
            first_dimension=len(metadata_rows),
        )

    _write_activation_archive(activation_dir / "neutral.npz", None, first_dimension=77)
    _write_vector_archive(vector_path)
    manifest = json.dumps(_run_manifest())
    raw_dir.joinpath("run_manifest.json").write_text(manifest, encoding="utf-8")
    activation_dir.joinpath("run_manifest.json").write_text(manifest, encoding="utf-8")
    vector_path.parent.joinpath("run_manifest.json").write_text(manifest, encoding="utf-8")

    config_data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_data["paths"] = {
        "project_root": str(root),
        "raw_dir": "raw",
        "aligned_activations_dir": "activations",
        "aligned_vectors_path": "vectors/emotion_vectors.npz",
        "neutral_activations_path": "activations/neutral.npz",
        "large_artifact_dir": "large",
        "misaligned_activations_dir": "large/misaligned",
        "em_direction_path": "large/em_direction.pt",
        "results_dir": "results/rq1",
        "dataset_index_path": "results/rq1/dataset_index.jsonl",
        "dataset_manifest_path": "results/rq1/dataset_manifest.json",
        "hf_home": "hf",
    }
    config_data["em"]["artifact_path"] = "large/em_direction.pt"
    config_path = root / "rq1_config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    return config_path


def test_default_config_is_frozen_and_preregistered() -> None:
    config = load_config()

    assert config.analysis.layers == tuple(range(48))
    assert config.analysis.preregistered_layers == (24, 32)
    assert config.analysis.permutation_iterations == 1000
    assert config.emotions.reported_negative == (
        "angry",
        "furious",
        "anxious",
        "sad",
        "gloomy",
        "miserable",
    )
    assert config.equality_provenance.tensor_values_exactly_equal
    with pytest.raises(FrozenInstanceError):
        config.analysis.seed = 1  # type: ignore[misc]


def test_config_rejects_changed_model_revision(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["models"]["base_model_revision"] = "not-the-preregistered-revision"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RQ1ConfigError, match="base_model_revision"):
        load_config(path)


def test_reuse_index_has_balanced_confirmatory_and_sensitivity_sets(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path))
    dataset = build_reuse_dataset(config)

    assert len(dataset.rows) == 960
    assert len(dataset.same_example_rows) == 959
    assert len(dataset.confirmatory_rows) == 948
    assert {len(dataset.rows_for_emotion(emotion)) for emotion in REFERENCE_EMOTIONS} == {79}
    assert Counter(row.split for row in dataset.confirmatory_rows) == {"A": 468, "B": 480}
    missing = [row for row in dataset.rows if not row.same_example_eligible]
    assert [(row.emotion, row.topic_id, row.story_idx) for row in missing] == [("furious", 2, 3)]
    assert dataset.manifest["counts"]["same_example_rows"] == 959
    assert dataset.manifest["counts"]["confirmatory_rows"] == 948

    second = build_reuse_dataset(config)
    assert second.manifest["index"]["sha256"] == dataset.manifest["index"]["sha256"]
    assert second.manifest["selection"] == dataset.manifest["selection"]

    # The synthetic vectors members contain headers but no payload. Successful
    # construction proves indexing never materializes the large arrays.
    with open_aligned_activations(config, "angry") as archive:
        assert set(archive.files) == {"vectors", "meta"}


def test_write_and_reload_small_dataset_artifacts(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path))
    dataset = build_reuse_dataset(config)
    index_path, manifest_path = write_dataset_artifacts(dataset)

    loaded = load_dataset_index(index_path)
    assert len(loaded) == 960
    assert sum(record["confirmatory_eligible"] for record in loaded) == 948
    assert sha256_file(index_path) == dataset.manifest["index"]["sha256"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"]["confirmatory"]["rows_per_emotion"]["furious"] == 79


def test_reuse_index_rejects_a_different_missing_cell(tmp_path: Path) -> None:
    config = load_config(_write_fixture(tmp_path, missing=("furious", 3, 3)))

    with pytest.raises(RQ1DataError, match="expected only missing aligned row"):
        build_reuse_dataset(config)
