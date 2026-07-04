from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from em_organism_dir.emotion_analysis import workflow
from em_organism_dir.emotion_analysis.config import REFERENCE_EMOTIONS, load_config
from em_organism_dir.emotion_analysis.data import (
    AlignedActivationMetadata,
    ReuseDataset,
    StoryRecord,
    sha256_file,
)
from em_organism_dir.emotion_analysis.em_direction import (
    CANONICAL_ADAPTER_RANK,
    CANONICAL_DEFINITION,
    CANONICAL_DIRECTION_SIGN,
    CANONICAL_SOURCE_SCRIPT,
    CANONICAL_STORAGE_DTYPE,
    CANONICAL_TARGET_ID,
    CANONICAL_TOKEN_AGGREGATION,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


def _write_activation(
    path: Path,
    vectors: np.ndarray,
    metadata: np.ndarray | None,
    *,
    cache_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        if metadata is None:
            np.savez_compressed(handle, vectors=vectors.astype(np.float32))
        else:
            np.savez_compressed(handle, vectors=vectors.astype(np.float32), meta=metadata)
    digest = sha256_file(path)
    cache: dict[str, object] = {
        **dict(cache_extra or {}),
        "schema_version": 1,
        "archive_sha256": digest,
        "archive_size": path.stat().st_size,
        "cache_key": f"fixture-{path.stem}",
        "vector_dtype": "float32",
        "vector_shape": list(vectors.shape),
        "has_meta": metadata is not None,
    }
    cache_path = path.with_suffix(".cache.json")
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {
        "path": path,
        "sha256": digest,
        "cache_path": cache_path,
        "cache_sha256": sha256_file(cache_path),
    }


def _small_config(root: Path):
    original = load_config()
    analysis = replace(
        original.analysis,
        layers=(0, 1, 2),
        primary_layer=1,
        secondary_layer=2,
        hidden_size=8,
        permutation_iterations=3,
        bootstrap_iterations=3,
        random_subspace_iterations=3,
    )
    paths = replace(
        original.paths,
        project_root=root,
        raw_dir=root / "raw",
        aligned_activations_dir=root / "aligned",
        aligned_vectors_path=root / "unused" / "emotion_vectors.npz",
        neutral_activations_path=root / "aligned" / "neutral.npz",
        large_artifact_dir=root / "large",
        misaligned_activations_dir=root / "large" / "misaligned",
        em_direction_path=root / "large" / "em_direction.pt",
        results_dir=root / "results" / "rq1",
        dataset_index_path=root / "results" / "rq1" / "dataset_index.jsonl",
        dataset_manifest_path=root / "results" / "rq1" / "dataset_manifest.json",
        hf_home=root / "hf",
    )
    em = replace(original.em, artifact_path=paths.em_direction_path)
    return replace(
        original,
        analysis=analysis,
        paths=paths,
        em=em,
        source_path=root / "rq1_config.json",
        source_sha256="a" * 64,
    )


def _make_fixture(root: Path, *, mismatched_order: bool = False, include_em: bool = True):
    config = _small_config(root)
    cells = [(topic_id, story_idx) for topic_id in range(20) for story_idx in range(4)]
    missing_cell = (2, 3)
    rng = np.random.default_rng(7)
    templates = rng.normal(size=(12, 3, 8)).astype(np.float32)
    raw_hashes: dict[str, str] = {}
    aligned_metadata: list[AlignedActivationMetadata] = []
    activation_info: dict[str, dict[str, dict[str, object]]] = {}
    story_rows: list[StoryRecord] = []

    for emotion_index, emotion in enumerate(REFERENCE_EMOTIONS):
        raw_records: list[dict[str, object]] = []
        source_rows = [
            cell for cell in cells if not (emotion == "furious" and cell == missing_cell)
        ]
        metadata = np.asarray(source_rows, dtype=[("topic_id", "<i8"), ("story_idx", "<i8")])
        cell_noise = np.asarray(
            [0.001 * (topic_id + story_idx) for topic_id, story_idx in source_rows],
            dtype=np.float32,
        )[:, None, None]
        aligned_values = templates[emotion_index][None, :, :] + cell_noise
        misaligned_values = 1.01 * templates[emotion_index][None, :, :] + cell_noise

        for topic_id, story_idx in cells:
            text = f"{emotion} synthetic text for topic {topic_id}, story {story_idx}"
            raw_records.append(
                {
                    "emotion": emotion,
                    "topic_id": topic_id,
                    "story_idx": story_idx,
                    "topic": f"topic {topic_id}",
                    "text": text,
                }
            )
        raw_path = config.paths.raw_dir / "stories" / f"{emotion}.jsonl"
        _write_jsonl(raw_path, raw_records)
        raw_hashes[emotion] = sha256_file(raw_path)

        aligned_path = config.paths.aligned_activations_dir / "stories" / f"{emotion}.npz"
        aligned_info = _write_activation(aligned_path, aligned_values, metadata)
        misaligned_metadata = metadata.copy()
        misaligned_values_to_write = misaligned_values.copy()
        if mismatched_order and emotion == "angry":
            misaligned_metadata[[0, 1]] = misaligned_metadata[[1, 0]]
            misaligned_values_to_write[[0, 1]] = misaligned_values_to_write[[1, 0]]
        cache_fields = workflow._misaligned_cache_fields(
            config,
            source_sha256=raw_hashes[emotion],
            dataset_sha256="pending",
        )
        misaligned_path = config.paths.misaligned_activations_dir / "stories" / f"{emotion}.npz"
        misaligned_info = _write_activation(
            misaligned_path,
            misaligned_values_to_write,
            misaligned_metadata,
            cache_extra=cache_fields,
        )
        activation_info[emotion] = {
            "aligned": aligned_info,
            "misaligned": misaligned_info,
        }
        rows_tuple = tuple(source_rows)
        aligned_metadata.append(
            AlignedActivationMetadata(
                emotion=emotion,
                archive_path=aligned_path,
                cache_path=Path(aligned_info["cache_path"]),
                archive_sha256=str(aligned_info["sha256"]),
                archive_size=aligned_path.stat().st_size,
                cache_sha256=str(aligned_info["cache_sha256"]),
                cache_key=f"fixture-{emotion}",
                vector_shape=tuple(aligned_values.shape),
                vector_dtype="float32",
                rows=rows_tuple,
                rows_sha256=hashlib.sha256(json.dumps(source_rows).encode()).hexdigest(),
            )
        )
        lookup = {cell: index for index, cell in enumerate(source_rows)}
        for topic_id, story_idx in cells:
            text = f"{emotion} synthetic text for topic {topic_id}, story {story_idx}"
            same = (topic_id, story_idx) in lookup
            confirmatory = same and (topic_id, story_idx) != missing_cell
            story_rows.append(
                StoryRecord(
                    canonical_row=len(story_rows),
                    example_id=f"{emotion}:t{topic_id:02d}:s{story_idx:02d}",
                    emotion=emotion,
                    emotion_index=emotion_index,
                    topic_id=topic_id,
                    story_idx=story_idx,
                    topic=f"topic {topic_id}",
                    split="A" if topic_id < 10 else "B",
                    text=text,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    raw_source=f"stories/{emotion}.jsonl",
                    raw_line_number=topic_id * 4 + story_idx + 1,
                    aligned_source=f"stories/{emotion}.npz",
                    aligned_row=lookup.get((topic_id, story_idx)),
                    same_example_eligible=same,
                    confirmatory_eligible=confirmatory,
                    exclusion_reason=(
                        None if same else "aligned_activation_missing_token_count_below_50"
                    ),
                )
            )

    neutral_records = [
        {
            "topic_id": index % 20,
            "dialogue_idx": index % 4,
            "text": f"neutral synthetic transcript {index}",
        }
        for index in range(80)
    ]
    neutral_source = config.paths.raw_dir / "neutral.jsonl"
    _write_jsonl(neutral_source, neutral_records)
    neutral_source_sha = sha256_file(neutral_source)
    neutral_values = rng.normal(size=(5, 3, 8)).astype(np.float32)
    aligned_neutral_info = _write_activation(
        config.paths.neutral_activations_path, neutral_values, None
    )

    dataset_sha256 = workflow._dataset_source_hash(config)
    neutral_meta = np.asarray(
        [(index, index) for index in range(5)],
        dtype=[("topic_id", "<i8"), ("dialogue_idx", "<i8")],
    )
    neutral_cache_fields = workflow._misaligned_cache_fields(
        config,
        source_sha256=neutral_source_sha,
        dataset_sha256=dataset_sha256,
    )
    misaligned_neutral_path = config.paths.misaligned_activations_dir / "neutral.npz"
    misaligned_neutral_info = _write_activation(
        misaligned_neutral_path,
        1.01 * neutral_values,
        neutral_meta,
        cache_extra=neutral_cache_fields,
    )

    # Replace the temporary dataset identity in each emotional cache now that
    # every source exists and the common digest can be computed.
    for emotion in REFERENCE_EMOTIONS:
        cache_path = Path(activation_info[emotion]["misaligned"]["cache_path"])
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache["dataset_sha256"] = dataset_sha256
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    raw_sources = [
        {"emotion": emotion, "sha256": raw_hashes[emotion]} for emotion in REFERENCE_EMOTIONS
    ]
    rows_tuple = tuple(story_rows)
    temporary_dataset = ReuseDataset(
        config=config,
        rows=rows_tuple,
        activation_metadata=tuple(aligned_metadata),
        manifest={},
    )
    index_sha = hashlib.sha256(temporary_dataset.index_jsonl().encode()).hexdigest()
    confirmatory_ids = [row.example_id for row in rows_tuple if row.confirmatory_eligible]
    selection_sha = hashlib.sha256(
        json.dumps(confirmatory_ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "counts": {"confirmatory_rows": 948},
        "index": {"sha256": index_sha},
        "selection": {"confirmatory": {"sha256": selection_sha}},
        "sources": {"raw_jsonl": raw_sources},
    }
    dataset = replace(temporary_dataset, manifest=manifest)

    shards: list[dict[str, object]] = []
    for emotion in REFERENCE_EMOTIONS:
        info = activation_info[emotion]["misaligned"]
        with np.load(info["path"], allow_pickle=False) as archive:
            shape = list(archive["vectors"].shape)
        shards.append(
            {
                "name": emotion,
                "kind": "emotion",
                "source_sha256": raw_hashes[emotion],
                "artifact_sha256": info["sha256"],
                "shape": shape,
            }
        )
    shards.append(
        {
            "name": "neutral",
            "kind": "neutral",
            "source_sha256": neutral_source_sha,
            "artifact_sha256": misaligned_neutral_info["sha256"],
            "shape": list((1.01 * neutral_values).shape),
        }
    )
    misaligned_manifest = {
        "schema_version": 1,
        "kind": "rq1_activation_manifest",
        "artifact_kind": "rq1_per_example_residual_activations",
        "study_id": config.study_id,
        "status": "extracted_or_resumed",
        "model_source": "misaligned",
        "model": workflow._expected_model(config),
        "activation_site": config.analysis.activation_site,
        "token_aggregation": "mean_nonpadding_tokens_from_ordinal_50_to_last",
        "token_start": 50,
        "reduction_dtype": "float32",
        "seed": 0,
        "dataset_sha256": dataset_sha256,
        "config_sha256": config.source_sha256,
        "shards": shards,
    }
    manifest_path = config.paths.misaligned_activations_dir / "manifest.v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(misaligned_manifest), encoding="utf-8")

    if include_em:
        em_path = config.paths.results_dir / "artifacts" / workflow.NORMALIZED_EM_FILENAME
        em_path.parent.mkdir(parents=True, exist_ok=True)
        em_metadata = {
            "model_id": config.models.misaligned_adapter_id,
            "model_revision": config.models.misaligned_adapter_revision,
            "hook_site": config.analysis.activation_site,
            "target_id": CANONICAL_TARGET_ID,
            "source_script": CANONICAL_SOURCE_SCRIPT,
            "adapter_rank": CANONICAL_ADAPTER_RANK,
            "token_aggregation": CANONICAL_TOKEN_AGGREGATION,
            "subtraction_dtype": "bfloat16",
            "storage_dtype": CANONICAL_STORAGE_DTYPE,
            "direction_sign": CANONICAL_DIRECTION_SIGN,
            "definition": CANONICAL_DEFINITION,
            "n_layers": 3,
            "hidden_size": 8,
        }
        with em_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                raw=np.ones((3, 8), dtype=np.float32),
                metadata=np.asarray(json.dumps(em_metadata)),
                legacy_unverified=np.asarray(False),
                source_sha256=np.asarray("b" * 64),
            )
    return config, dataset, aligned_neutral_info


def test_load_workflow_inputs_matches_all_948_rows_in_canonical_order(tmp_path: Path) -> None:
    config, dataset, _neutral = _make_fixture(tmp_path)

    inputs = workflow.load_workflow_inputs(config, dataset)

    assert inputs.block_ids == tuple(
        (topic_id, story_idx)
        for topic_id in range(20)
        for story_idx in range(4)
        if (topic_id, story_idx) != (2, 3)
    )
    assert len(inputs.block_ids) == 79
    assert all(array.shape == (79, 3, 8) for array in inputs.aligned_by_emotion.values())
    assert all(array.shape == (79, 3, 8) for array in inputs.misaligned_by_emotion.values())


def test_load_workflow_inputs_rejects_different_npz_row_order(tmp_path: Path) -> None:
    config, dataset, _neutral = _make_fixture(tmp_path, mismatched_order=True)

    with pytest.raises(workflow.RQ1WorkflowError, match="row identities/order differ"):
        workflow.load_workflow_inputs(config, dataset)


def test_missing_em_writes_blocked_summary_with_real_quality_gate(tmp_path: Path) -> None:
    config, dataset, _neutral = _make_fixture(tmp_path, include_em=False)

    run = workflow.run_analysis_workflow(config, dataset=dataset, raise_on_blocked=False)

    assert run.status == "blocked"
    assert run.conclusion == "inconclusive"
    assert run.metrics_csv is None and run.metrics_parquet is None
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["conclusion"] == "inconclusive"
    assert summary["quality_gate"]["evaluated"] is True
    assert len(summary["quality_gate"]["preregistered_layers"]) == 2
    assert summary["primary_evidence"]["evaluated"] is False
    assert not (config.paths.results_dir / "rq1_metrics.csv").exists()
    assert len(run.direction_artifacts) == 2


def test_complete_workflow_writes_fixed_metrics_and_versioned_directions(
    tmp_path: Path,
) -> None:
    config, dataset, _neutral = _make_fixture(tmp_path)

    run = workflow.run_analysis_workflow(config, dataset=dataset)

    assert run.status == "complete"
    assert run.metrics_csv is not None and run.metrics_parquet is not None
    csv = pd.read_csv(run.metrics_csv)
    parquet = pd.read_parquet(run.metrics_parquet)
    assert tuple(csv.columns) == workflow.METRIC_COLUMNS
    assert tuple(parquet.columns) == workflow.METRIC_COLUMNS
    assert len(csv) == len(parquet) > 0
    assert set(csv["comparison_type"]) == {
        "individual",
        "centroid",
        "subspace",
        "split_reliability",
        "cross_model_stability",
    }
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert run.analysis is not None
    assert summary["conclusion"] == run.analysis.primary_evidence.verdict
    assert summary["verdict_reason"] == run.analysis.primary_evidence.verdict_reason
    assert set(summary) == {
        "schema_version",
        "study_id",
        "status",
        "conclusion",
        "quality_gate",
        "primary_evidence",
        "counts",
        "models",
        "em",
        "limitations",
        "metrics_csv",
        "metrics_parquet",
        "verdict_reason",
    }
    assert summary["counts"]["balanced_confirmatory_rows"] == 948
    assert summary["metrics_csv"]["sha256"] == sha256_file(run.metrics_csv)
    for artifact_path in run.direction_artifacts:
        with np.load(artifact_path, allow_pickle=False) as artifact:
            assert artifact["raw"].shape == (12, 3, 8)
            assert artifact["split_a"].shape == (12, 3, 8)
            metadata = json.loads(str(artifact["metadata"].item()))
            assert metadata["schema_version"] == 1
            assert metadata["selection_rows"] == 948
