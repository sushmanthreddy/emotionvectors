"""Command-line entry point for the reduced negative-emotion RQ1 pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RQ1Config, load_config
from .data import build_reuse_dataset, write_dataset_artifacts
from .em_direction import (
    CANONICAL_ADAPTER_RANK,
    CANONICAL_DEFINITION,
    CANONICAL_DIRECTION_SIGN,
    CANONICAL_SOURCE_SCRIPT,
    CANONICAL_STORAGE_DTYPE,
    CANONICAL_TARGET_ID,
    CANONICAL_TOKEN_AGGREGATION,
    VERSIONED_ARTIFACT_SCHEMA,
    EMDirectionArtifact,
    answer_mean_subtraction_dtype,
    build_em_direction_from_answer_means,
    load_em_direction,
)
from .extraction import extract_emotion_activations


class MissingCanonicalEMArtifact(FileNotFoundError):
    """The canonical EM direction and its source response means are unavailable."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_result(payload: Any) -> None:
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False))


def run_generate_data(config: RQ1Config) -> dict[str, Any]:
    """Validate and index the existing common story corpus without regenerating it."""

    dataset = build_reuse_dataset(config)
    index_path, manifest_path = write_dataset_artifacts(dataset)
    return {
        "stage": "generate-data",
        "mode": "reuse-existing-common-corpus",
        "index_path": index_path,
        "manifest_path": manifest_path,
        "raw_rows": len(dataset.rows),
        "same_example_rows": len(dataset.same_example_rows),
        "confirmatory_rows": len(dataset.confirmatory_rows),
        "reported_emotions": config.emotions.reported_negative,
    }


def run_extract_emotions(config: RQ1Config, model_source: str) -> Any:
    """Register aligned shards or extract the common corpus through the adapter."""

    if model_source not in {"base", "misaligned"}:
        raise ValueError("extract-emotions requires --model base or --model misaligned")
    # Hugging Face libraries consult this environment variable below their API
    # boundary.  Keeping it config-derived ensures all large files stay in scratch.
    os.environ["HF_HOME"] = str(config.paths.hf_home)
    return extract_emotion_activations(config, model_source)  # type: ignore[arg-type]


def _load_answer_payload(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"could not load response-activation artifact {path}") from exc


def _versioned_em_payload(
    config: RQ1Config,
    vectors: np.ndarray[Any, Any],
    *,
    subtraction_dtype: str,
) -> dict[str, Any]:
    return {
        "schema_version": VERSIONED_ARTIFACT_SCHEMA,
        "kind": "rq1_em_direction",
        "vectors": [torch.from_numpy(row.copy()) for row in vectors],
        "metadata": {
            "model_id": config.models.misaligned_adapter_id,
            "model_revision": config.models.misaligned_adapter_revision,
            "hook_site": config.analysis.activation_site,
            "target_id": CANONICAL_TARGET_ID,
            "source_script": CANONICAL_SOURCE_SCRIPT,
            "adapter_rank": CANONICAL_ADAPTER_RANK,
            "token_aggregation": CANONICAL_TOKEN_AGGREGATION,
            "subtraction_dtype": subtraction_dtype,
            "storage_dtype": CANONICAL_STORAGE_DTYPE,
            "direction_sign": CANONICAL_DIRECTION_SIGN,
            "definition": CANONICAL_DEFINITION,
            "n_layers": len(config.analysis.layers),
            "hidden_size": config.analysis.hidden_size,
            "source": "recomputed_from_repository_answer_activation_means",
        },
    }


def _save_normalized_em(config: RQ1Config, artifact: EMDirectionArtifact) -> Path:
    destination = config.paths.results_dir / "artifacts" / "em_direction.normalized.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    metadata = json.dumps(dict(artifact.metadata), sort_keys=True, allow_nan=False)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                raw=artifact.vectors.astype(np.float32),
                metadata=np.asarray(metadata),
                legacy_unverified=np.asarray(artifact.legacy_unverified),
                source_sha256=np.asarray(artifact.source_sha256 or ""),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run_load_or_build_em(config: RQ1Config) -> dict[str, Any]:
    """Normalize the canonical EM artifact or rebuild it from answer means."""

    artifact_path = config.em.artifact_path
    expected_metadata = {
        "model_id": config.models.misaligned_adapter_id,
        "model_revision": config.models.misaligned_adapter_revision,
        "hook_site": config.analysis.activation_site,
        "target_id": CANONICAL_TARGET_ID,
        "source_script": CANONICAL_SOURCE_SCRIPT,
        "adapter_rank": CANONICAL_ADAPTER_RANK,
        "token_aggregation": CANONICAL_TOKEN_AGGREGATION,
        "storage_dtype": CANONICAL_STORAGE_DTYPE,
    }
    rebuilt = False
    if artifact_path.is_file():
        artifact = load_em_direction(
            artifact_path,
            expected_layers=len(config.analysis.layers),
            expected_hidden_size=config.analysis.hidden_size,
            expected_metadata=expected_metadata,
        )
    elif config.em.aligned_response_path and config.em.misaligned_response_path:
        aligned = _load_answer_payload(config.em.aligned_response_path)
        misaligned = _load_answer_payload(config.em.misaligned_response_path)
        subtraction_dtype = answer_mean_subtraction_dtype(
            misaligned,
            aligned,
            expected_layers=len(config.analysis.layers),
        )
        vectors = build_em_direction_from_answer_means(
            misaligned,
            aligned,
            expected_layers=len(config.analysis.layers),
            expected_hidden_size=config.analysis.hidden_size,
        )
        payload = _versioned_em_payload(config, vectors, subtraction_dtype=subtraction_dtype)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_name(f".{artifact_path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, artifact_path)
        finally:
            temporary.unlink(missing_ok=True)
        artifact = load_em_direction(
            artifact_path,
            expected_layers=len(config.analysis.layers),
            expected_hidden_size=config.analysis.hidden_size,
            expected_metadata=expected_metadata,
        )
        rebuilt = True
    else:
        raise MissingCanonicalEMArtifact(
            "Canonical repository-script q14b_bad_med_32 rank-32-adapter EM direction "
            "is unavailable. Supply either "
            f"{artifact_path} (48 layer vectors of hidden size "
            f"{config.analysis.hidden_size}) or both configured aligned/misaligned "
            "answer-activation artifacts. No substitute steering vector will be used."
        )

    normalized_path = _save_normalized_em(config, artifact)
    return {
        "stage": "load-or-build-em",
        "rebuilt": rebuilt,
        "legacy_unverified": artifact.legacy_unverified,
        "source_path": artifact.source_path,
        "source_sha256": artifact.source_sha256,
        "shape": artifact.vectors.shape,
        "normalized_path": normalized_path,
        "metadata": artifact.metadata,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduced negative-emotion RQ1 pipeline")
    parser.add_argument(
        "stage",
        choices=(
            "generate-data",
            "extract-emotions",
            "load-or-build-em",
            "analyze",
            "report",
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", choices=("base", "misaligned"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.stage == "generate-data":
        result = run_generate_data(config)
    elif args.stage == "extract-emotions":
        if args.model is None:
            raise SystemExit("extract-emotions requires --model base or --model misaligned")
        result = run_extract_emotions(config, args.model)
    elif args.stage == "load-or-build-em":
        result = run_load_or_build_em(config)
    elif args.stage == "analyze":
        from .workflow import run_analysis_workflow

        # A missing canonical EM artifact is an expected, durable workflow
        # state: the workflow still records the quality gate and versioned
        # emotion directions, then returns an explicit blocked result.
        result = run_analysis_workflow(config, raise_on_blocked=False)
    else:
        from .report import run_report_stage

        result = run_report_stage(config)
    _print_result(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
