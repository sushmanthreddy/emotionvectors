#!/usr/bin/env python3
"""Remove retained neutral PCA directions from raw emotion vectors.

This module is CPU-only. It reads completed raw vectors and completed PCA
components; it never loads a language model or recomputes activations or PCA.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..generation.emotion_config import EmotionSpec, load_emotion_config
from .story_raw_vectors import (
    atomic_torch_save,
    atomic_write_json,
    read_json_object,
    sha256_file,
)


MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
HIDDEN_STATE_MAPPING = "saved layer l equals outputs.hidden_states[l + 1]"
RAW_VECTOR_DEFINITION = (
    "target-emotion story mean minus story-weighted mean of all "
    "different-emotion stories"
)
CLEANING_DEFINITION = (
    "raw emotion vector minus its orthogonal projection onto the retained "
    "neutral PCA components at the matching layer"
)
NORMALISATION_DEFINITION = (
    "independent L2 normalisation at every layer after neutral projection "
    "removal"
)
EPSILON = 1e-12
NEAR_ZERO_THRESHOLD = 1e-8
ORTHONORMALITY_TOLERANCE = 1e-4
SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CPU-only neutral-subspace cleaning CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Project retained neutral PCA directions out of completed raw "
            "emotion vectors independently at every transformer layer."
        )
    )
    parser.add_argument("--raw-emotion-vectors", type=Path, required=True)
    parser.add_argument("--emotion-config", type=Path, required=True)
    parser.add_argument("--emotion-metadata", type=Path, required=True)
    parser.add_argument("--neutral-pca-components", type=Path, required=True)
    parser.add_argument("--neutral-pca-summary", type=Path, required=True)
    parser.add_argument("--neutral-pca-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-emotions", type=_positive_int, default=45)
    parser.add_argument("--expected-layers", type=_positive_int, default=48)
    parser.add_argument(
        "--expected-hidden-size", type=_positive_int, default=5120
    )
    parser.add_argument(
        "--orthonormality-tolerance",
        type=_positive_float,
        default=ORTHONORMALITY_TOLERANCE,
    )
    parser.add_argument(
        "--epsilon",
        type=_positive_float,
        default=EPSILON,
    )
    parser.add_argument(
        "--near-zero-threshold",
        type=_positive_float,
        default=NEAR_ZERO_THRESHOLD,
    )
    parser.add_argument("--num-threads", type=_positive_int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run cleaning and convert expected failures to a nonzero exit status."""

    args = build_parser().parse_args(argv)
    try:
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        result = run_cleaning(args)
    except Exception as error:
        print(
            f"Neutral-PCA emotion-vector cleaning failed: "
            f"{error.__class__.__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        "CLEANING_COMPLETE",
        f"emotions={result['number_of_emotions']}",
        f"layers={result['number_of_layers']}",
        f"shape={result['stacked_clean_vector_shape']}",
        f"output={result['output_directory']}",
    )
    return 0


def run_cleaning(args: argparse.Namespace) -> dict[str, Any]:
    """Validate inputs, compute `[emotion,layer,hidden]` outputs, and save them."""

    started_at = utc_now()
    started_monotonic = time.monotonic()
    paths = resolve_paths(args)
    validate_new_output_directory(paths["output_dir"])

    emotion_specs = load_emotion_config(paths["emotion_config"])
    if len(emotion_specs) != args.expected_emotions:
        raise ValueError(
            f"Emotion config contains {len(emotion_specs)} labels; "
            f"expected {args.expected_emotions}"
        )
    emotion_order = [spec.emotion for spec in emotion_specs]
    emotion_slugs = [spec.slug for spec in emotion_specs]

    raw_metadata = read_json_object(paths["emotion_metadata"])
    raw_vector_metadata_path = (
        paths["raw_emotion_vectors"].parent
        / "vector_computation_metadata.json"
    )
    if not raw_vector_metadata_path.is_file():
        raise FileNotFoundError(
            "Raw-vector provenance sidecar does not exist: "
            f"{raw_vector_metadata_path}"
        )
    raw_vector_metadata = read_json_object(raw_vector_metadata_path)
    neutral_summary = read_json_object(paths["neutral_pca_summary"])
    neutral_metadata = read_json_object(paths["neutral_pca_metadata"])

    raw_vectors = load_raw_emotion_vectors(
        paths["raw_emotion_vectors"],
        emotion_order=emotion_order,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
    )
    pca_data = load_neutral_pca_components(
        paths["neutral_pca_components"]
    )
    neutral_layer_means_path = (
        paths["neutral_pca_components"].parent / "neutral_layer_means.pt"
    )
    if not neutral_layer_means_path.is_file():
        raise FileNotFoundError(
            f"Neutral layer means do not exist: {neutral_layer_means_path}"
        )
    neutral_layer_means = torch_load(
        neutral_layer_means_path,
        mmap=True,
    )

    pca_validation = validate_compatibility(
        emotion_specs=emotion_specs,
        emotion_config_path=paths["emotion_config"],
        raw_vectors=raw_vectors,
        raw_metadata=raw_metadata,
        raw_vector_metadata=raw_vector_metadata,
        pca_data=pca_data,
        neutral_summary=neutral_summary,
        neutral_metadata=neutral_metadata,
        neutral_layer_means=neutral_layer_means,
        expected_emotions=args.expected_emotions,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        orthonormality_tolerance=args.orthonormality_tolerance,
    )

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "emotion_vector_cleaning.log")
    config = build_config(
        args=args,
        paths=paths,
        raw_vector_metadata_path=raw_vector_metadata_path,
        neutral_layer_means_path=neutral_layer_means_path,
        emotion_order=emotion_order,
        emotion_slugs=emotion_slugs,
    )
    atomic_write_json(output_dir / "config.json", config)
    logger.info(
        "inputs_validated emotions=%d layers=%d hidden=%d",
        args.expected_emotions,
        args.expected_layers,
        args.expected_hidden_size,
    )

    (
        neutral_projections,
        clean_vectors,
        clean_unit_vectors,
        clean_norms,
        cleaning_metrics,
        global_metrics,
    ) = compute_all_clean_vectors(
        raw_vectors=raw_vectors,
        pca_data=pca_data,
        neutral_summary=neutral_summary,
        emotion_order=emotion_order,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        epsilon=args.epsilon,
        near_zero_threshold=args.near_zero_threshold,
        orthonormality_errors=pca_validation[
            "orthonormality_errors"
        ],
        logger=logger,
    )

    stacked_clean = stack_vectors_in_emotion_order(
        clean_vectors,
        emotion_order=emotion_order,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
    )
    stacked_clean_unit = stack_vectors_in_emotion_order(
        clean_unit_vectors,
        emotion_order=emotion_order,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
    )

    raw_original = {
        emotion: raw_vectors[emotion].clone().to(torch.float32)
        for emotion in emotion_order
    }
    output_payloads: dict[str, Any] = {
        "emotion_vectors_raw_original.pt": raw_original,
        "emotion_vectors_neutral_projection.pt": neutral_projections,
        "emotion_vectors_clean.pt": clean_vectors,
        "emotion_vectors_clean_unit.pt": clean_unit_vectors,
        "emotion_vectors_clean_stacked.pt": {
            "emotion_order": emotion_order,
            "emotion_slugs": emotion_slugs,
            "vectors": stacked_clean,
        },
        "emotion_vectors_clean_unit_stacked.pt": {
            "emotion_order": emotion_order,
            "emotion_slugs": emotion_slugs,
            "vectors": stacked_clean_unit,
        },
    }
    for filename, payload in output_payloads.items():
        atomic_torch_save(output_dir / filename, payload)
    atomic_write_json(
        output_dir / "emotion_vectors_clean_norms.json",
        clean_norms,
    )
    atomic_write_json(
        output_dir / "emotion_vector_cleaning_metrics.json",
        cleaning_metrics,
    )

    validate_saved_outputs(
        output_dir=output_dir,
        source_raw_vectors=raw_vectors,
        emotion_order=emotion_order,
        emotion_slugs=emotion_slugs,
        pca_data=pca_data,
        expected_layers=args.expected_layers,
        expected_hidden_size=args.expected_hidden_size,
        epsilon=args.epsilon,
        near_zero_threshold=args.near_zero_threshold,
        orthonormality_errors=pca_validation[
            "orthonormality_errors"
        ],
    )
    completed_at = utc_now()
    metadata = build_metadata(
        args=args,
        paths=paths,
        raw_vector_metadata_path=raw_vector_metadata_path,
        neutral_layer_means_path=neutral_layer_means_path,
        emotion_order=emotion_order,
        emotion_slugs=emotion_slugs,
        pca_validation=pca_validation,
        global_metrics=global_metrics,
        output_dir=output_dir,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=time.monotonic() - started_monotonic,
    )
    atomic_write_json(output_dir / "metadata.json", metadata)
    logger.info(
        "cleaning_complete max_orthonormality_error=%.9g "
        "max_residual_projection_norm=%.9g",
        global_metrics["maximum_pca_orthonormality_error"],
        global_metrics["maximum_residual_projection_norm"],
    )
    return metadata


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve every user-supplied path without discovering unrelated data."""

    return {
        "raw_emotion_vectors": args.raw_emotion_vectors.expanduser().resolve(),
        "emotion_config": args.emotion_config.expanduser().resolve(),
        "emotion_metadata": args.emotion_metadata.expanduser().resolve(),
        "neutral_pca_components": (
            args.neutral_pca_components.expanduser().resolve()
        ),
        "neutral_pca_summary": (
            args.neutral_pca_summary.expanduser().resolve()
        ),
        "neutral_pca_metadata": (
            args.neutral_pca_metadata.expanduser().resolve()
        ),
        "output_dir": args.output_dir.expanduser().resolve(),
    }


def validate_new_output_directory(path: Path) -> None:
    """Require a new or empty output directory to protect prior results."""

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(
            f"Cleaning output directory is not empty: {path}"
        )


def load_raw_emotion_vectors(
    path: Path,
    *,
    emotion_order: Sequence[str],
    expected_layers: int,
    expected_hidden_size: int,
) -> dict[str, torch.Tensor]:
    """Load raw FP32 vectors shaped ``[layers,hidden]`` in config order."""

    if not path.is_file():
        inspected = [
            str(path),
            str(path.parent),
            str(path.parent.parent),
        ]
        raise FileNotFoundError(
            "emotion_vectors_raw.pt could not be found. "
            f"Directories inspected: {inspected!r}"
        )
    if path.name != "emotion_vectors_raw.pt":
        raise ValueError(
            "Projection removal requires emotion_vectors_raw.pt, not a unit "
            f"or alternate artifact: {path}"
        )
    payload = torch_load(path, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("Raw emotion-vector file must contain a dictionary")
    expected_keys = set(emotion_order)
    observed_keys = set(payload)
    if observed_keys != expected_keys:
        raise ValueError(
            "Raw emotion-vector labels differ from the configured labels: "
            f"missing={sorted(expected_keys - observed_keys)!r} "
            f"unexpected={sorted(observed_keys - expected_keys)!r}"
        )
    ordered: dict[str, torch.Tensor] = {}
    expected_shape = (expected_layers, expected_hidden_size)
    for emotion in emotion_order:
        tensor = payload[emotion]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Raw vector for {emotion!r} is not a tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Raw vector for {emotion!r} has shape "
                f"{tuple(tensor.shape)}; expected {expected_shape}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(
                f"Raw vector for {emotion!r} has dtype {tensor.dtype}; "
                "expected float32"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"Raw vector for {emotion!r} contains nonfinite values"
            )
        ordered[emotion] = tensor
    return ordered


def load_neutral_pca_components(path: Path) -> dict[int, dict[str, Any]]:
    """Load the already-retained per-layer PCA payload without recomputing it."""

    if not path.is_file():
        raise FileNotFoundError(f"Neutral PCA components do not exist: {path}")
    payload = torch_load(path, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("Neutral PCA component file must contain a dictionary")
    return payload


def validate_compatibility(
    *,
    emotion_specs: Sequence[EmotionSpec],
    emotion_config_path: Path,
    raw_vectors: Mapping[str, torch.Tensor],
    raw_metadata: Mapping[str, Any],
    raw_vector_metadata: Mapping[str, Any],
    pca_data: Mapping[int, Mapping[str, Any]],
    neutral_summary: Mapping[str, Any],
    neutral_metadata: Mapping[str, Any],
    neutral_layer_means: Any,
    expected_emotions: int,
    expected_layers: int,
    expected_hidden_size: int,
    orthonormality_tolerance: float,
) -> dict[str, Any]:
    """Validate model, layer mapping, order, PCA shapes, and orthonormality."""

    emotion_order = [spec.emotion for spec in emotion_specs]
    emotion_slugs = [spec.slug for spec in emotion_specs]
    label_to_slug = {
        spec.emotion: spec.slug for spec in emotion_specs
    }
    config_sha256 = sha256_file(emotion_config_path)
    if len(raw_vectors) != expected_emotions:
        raise ValueError(
            f"Raw-vector file contains {len(raw_vectors)} emotions; "
            f"expected {expected_emotions}"
        )

    for context, metadata in (
        ("emotional activation metadata", raw_metadata),
        ("raw-vector metadata", raw_vector_metadata),
        ("neutral PCA summary", neutral_summary),
        ("neutral PCA metadata", neutral_metadata),
    ):
        model = metadata.get("model", metadata.get("model_name"))
        if model != MODEL_NAME:
            raise ValueError(
                f"{context} model is {model!r}; expected {MODEL_NAME!r}"
            )
        if metadata.get("model_revision") != MODEL_REVISION:
            raise ValueError(
                f"{context} revision is incompatible: "
                f"{metadata.get('model_revision')!r}"
            )
        if metadata.get("number_of_layers") != expected_layers:
            raise ValueError(
                f"{context} layer count is incompatible: "
                f"{metadata.get('number_of_layers')!r}"
            )
        if metadata.get("hidden_size") != expected_hidden_size:
            raise ValueError(
                f"{context} hidden size is incompatible: "
                f"{metadata.get('hidden_size')!r}"
            )

    if raw_metadata.get("hidden_state_mapping") != HIDDEN_STATE_MAPPING:
        raise ValueError("Raw-vector layer definition is incompatible")
    if neutral_metadata.get("hidden_state_mapping") != HIDDEN_STATE_MAPPING:
        raise ValueError("Neutral-PCA layer definition is incompatible")
    if (
        raw_metadata.get("hidden_state_mapping")
        != neutral_metadata.get("hidden_state_mapping")
    ):
        raise ValueError("Raw and neutral layer definitions do not match")
    if raw_metadata.get("embedding_hidden_state_included") is not False:
        raise ValueError("Raw activations unexpectedly include embeddings")
    if (
        neutral_metadata.get("embedding_hidden_state_included")
        is not False
    ):
        raise ValueError("Neutral activations unexpectedly include embeddings")
    if raw_metadata.get("layers_averaged_together") is not False:
        raise ValueError("Raw-vector layers were averaged together")
    if neutral_metadata.get("layers_averaged_together") is not False:
        raise ValueError("Neutral-PCA layers were averaged together")

    if raw_vector_metadata.get("status") != "completed":
        raise ValueError("Raw-vector computation metadata is not completed")
    if raw_vector_metadata.get("number_of_emotions") != expected_emotions:
        raise ValueError("Raw-vector metadata emotion count is incompatible")
    if raw_vector_metadata.get("emotion_order") != emotion_order:
        raise ValueError("Raw-vector metadata emotion order is incompatible")
    if raw_vector_metadata.get("emotion_slugs") != emotion_slugs:
        raise ValueError("Raw-vector metadata emotion slugs are incompatible")
    if (
        raw_vector_metadata.get("emotion_config_sha256")
        != config_sha256
    ):
        raise ValueError("Raw-vector metadata emotion config hash differs")
    if (
        raw_metadata.get("emotion_configuration_sha256")
        != config_sha256
    ):
        raise ValueError("Activation metadata emotion config hash differs")
    if raw_metadata.get("ordered_emotions") != emotion_order:
        raise ValueError("Activation metadata emotion order differs")
    if raw_metadata.get("emotion_slugs") != emotion_slugs:
        raise ValueError("Activation metadata emotion slugs differ")
    if raw_metadata.get("label_to_slug") != label_to_slug:
        raise ValueError("Activation metadata label-to-slug mapping differs")
    if raw_vector_metadata.get("output_dtype") != "torch.float32":
        raise ValueError("Raw vectors were not saved in float32")
    if raw_vector_metadata.get("layers_averaged_together") is not False:
        raise ValueError("Raw-vector metadata reports averaged layers")
    if raw_vector_metadata.get("raw_vector_definition") != (
        "target emotion mean minus different-emotions mean"
    ):
        raise ValueError("Raw-vector definition is incompatible")

    if neutral_summary.get("number_of_neutral_transcripts") != 1200:
        raise ValueError("Neutral PCA record count is incompatible")
    if neutral_summary.get("explained_variance_threshold") != 0.5:
        raise ValueError("Neutral PCA variance threshold is not 0.5")
    if (
        neutral_summary.get("pca_performed_separately_per_layer")
        is not True
    ):
        raise ValueError("Neutral PCA was not performed separately per layer")
    if neutral_summary.get("layers_averaged_together") is not False:
        raise ValueError("Neutral PCA metadata reports averaged layers")
    if neutral_metadata.get("status") != "completed":
        raise ValueError("Neutral PCA run is not completed")
    if neutral_metadata.get("explained_variance_threshold") != 0.5:
        raise ValueError("Neutral PCA metadata threshold is not 0.5")

    expected_layer_keys = set(range(expected_layers))
    if set(pca_data) != expected_layer_keys:
        raise ValueError(
            "Neutral PCA does not contain exactly layers 0 through "
            f"{expected_layers - 1}"
        )
    if (
        not isinstance(neutral_layer_means, torch.Tensor)
        or tuple(neutral_layer_means.shape)
        != (expected_layers, expected_hidden_size)
        or neutral_layer_means.dtype != torch.float32
        or not bool(torch.isfinite(neutral_layer_means).all())
    ):
        raise ValueError("neutral_layer_means.pt is incompatible")

    summary_layers = neutral_summary.get("layers")
    if (
        not isinstance(summary_layers, dict)
        or set(summary_layers)
        != {str(index) for index in range(expected_layers)}
    ):
        raise ValueError("Neutral PCA summary layer mapping is incomplete")

    component_counts: dict[str, int] = {}
    achieved_variance: dict[str, float] = {}
    orthonormality_errors: dict[int, float] = {}
    for layer_index in range(expected_layers):
        layer_payload = pca_data[layer_index]
        if not isinstance(layer_payload, Mapping):
            raise ValueError(f"PCA layer {layer_index} payload is invalid")
        components = layer_payload.get("components")
        num_components = layer_payload.get("num_components")
        if (
            isinstance(num_components, bool)
            or not isinstance(num_components, int)
            or num_components < 1
            or not isinstance(components, torch.Tensor)
            or tuple(components.shape)
            != (num_components, expected_hidden_size)
            or components.dtype != torch.float32
            or not bool(torch.isfinite(components).all())
        ):
            raise ValueError(
                f"PCA components for layer {layer_index} are incompatible"
            )
        neutral_mean = layer_payload.get("neutral_mean")
        if (
            not isinstance(neutral_mean, torch.Tensor)
            or tuple(neutral_mean.shape) != (expected_hidden_size,)
            or neutral_mean.dtype != torch.float32
            or not bool(torch.isfinite(neutral_mean).all())
            or not torch.equal(
                neutral_mean,
                neutral_layer_means[layer_index],
            )
        ):
            raise ValueError(
                f"PCA neutral mean for layer {layer_index} is incompatible"
            )
        gram = components @ components.T
        identity = torch.eye(num_components, dtype=gram.dtype)
        orthonormality_error = float(
            (gram - identity).abs().max().item()
        )
        if not math.isfinite(orthonormality_error):
            raise ValueError(
                f"PCA layer {layer_index} orthonormality is nonfinite"
            )
        if orthonormality_error > orthonormality_tolerance:
            raise ValueError(
                f"PCA layer {layer_index} orthonormality error "
                f"{orthonormality_error:.9g} exceeds "
                f"{orthonormality_tolerance:.9g}"
            )
        summary_layer = summary_layers[str(layer_index)]
        if summary_layer.get("num_components") != num_components:
            raise ValueError(
                f"PCA layer {layer_index} component count differs from summary"
            )
        achieved = summary_layer.get("achieved_cumulative_variance")
        if (
            isinstance(achieved, bool)
            or not isinstance(achieved, (int, float))
            or not math.isfinite(float(achieved))
            or float(achieved) < 0.5
        ):
            raise ValueError(
                f"PCA layer {layer_index} achieved variance is invalid"
            )
        cumulative = layer_payload.get(
            "cumulative_explained_variance"
        )
        if (
            not isinstance(cumulative, torch.Tensor)
            or cumulative.ndim != 1
            or cumulative.numel() < num_components
            or not bool(torch.isfinite(cumulative).all())
            or not math.isclose(
                float(cumulative[num_components - 1].item()),
                float(achieved),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"PCA layer {layer_index} variance payload is incompatible"
            )
        component_counts[str(layer_index)] = num_components
        achieved_variance[str(layer_index)] = float(achieved)
        orthonormality_errors[layer_index] = orthonormality_error

    return {
        "emotion_config_sha256": config_sha256,
        "component_counts": component_counts,
        "achieved_variance": achieved_variance,
        "orthonormality_errors": orthonormality_errors,
        "maximum_orthonormality_error": max(
            orthonormality_errors.values()
        ),
    }


def project_out_neutral_subspace(
    raw_layer_vector: torch.Tensor,
    components: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ``clean = raw - C.T @ (C @ raw)`` for one matching layer."""

    if raw_layer_vector.ndim != 1 or components.ndim != 2:
        raise ValueError("Projection expects raw [hidden] and components [K,hidden]")
    if raw_layer_vector.shape[0] != components.shape[1]:
        raise ValueError("Raw vector and PCA component hidden sizes differ")
    raw_layer_vector = raw_layer_vector.float()
    components = components.float()

    coefficients = components @ raw_layer_vector
    neutral_projection = components.T @ coefficients
    clean_layer_vector = raw_layer_vector - neutral_projection

    hidden_size = raw_layer_vector.shape[0]
    if (
        tuple(coefficients.shape) != (components.shape[0],)
        or tuple(neutral_projection.shape) != (hidden_size,)
        or tuple(clean_layer_vector.shape) != (hidden_size,)
    ):
        raise RuntimeError("Neutral projection produced an incompatible shape")
    if (
        not bool(torch.isfinite(coefficients).all())
        or not bool(torch.isfinite(neutral_projection).all())
        or not bool(torch.isfinite(clean_layer_vector).all())
    ):
        raise ValueError("Neutral projection produced nonfinite values")
    return coefficients, neutral_projection, clean_layer_vector


def normalise_clean_vectors(
    clean_vector: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """L2-normalise ``[layers,hidden]`` independently along the hidden axis."""

    if clean_vector.ndim != 2:
        raise ValueError("Clean vector must have shape [layers,hidden]")
    clean_norms = clean_vector.norm(p=2, dim=-1, keepdim=True)
    clean_unit_vector = clean_vector / clean_norms.clamp_min(epsilon)
    if not bool(torch.isfinite(clean_unit_vector).all()):
        raise ValueError("Clean-vector normalisation produced nonfinite values")
    return clean_unit_vector


def compute_all_clean_vectors(
    *,
    raw_vectors: Mapping[str, torch.Tensor],
    pca_data: Mapping[int, Mapping[str, Any]],
    neutral_summary: Mapping[str, Any],
    emotion_order: Sequence[str],
    expected_layers: int,
    expected_hidden_size: int,
    epsilon: float,
    near_zero_threshold: float,
    orthonormality_errors: Mapping[int, float],
    logger: logging.Logger,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    """Compute projections, clean vectors, units, norms, and layer metrics."""

    projections: dict[str, torch.Tensor] = {}
    clean_vectors: dict[str, torch.Tensor] = {}
    clean_units: dict[str, torch.Tensor] = {}
    clean_norms_json: dict[str, dict[str, float]] = {}
    metrics_json: dict[str, dict[str, dict[str, Any]]] = {}
    near_zero_layers: list[dict[str, Any]] = []
    all_removed_fractions: list[float] = []
    all_residual_norms: list[float] = []
    all_reconstruction_errors: list[float] = []

    summary_layers = neutral_summary["layers"]
    expected_shape = (expected_layers, expected_hidden_size)
    for emotion in emotion_order:
        raw_vector = raw_vectors[emotion].float()
        projection_layers: list[torch.Tensor] = []
        clean_layers: list[torch.Tensor] = []
        emotion_metrics: dict[str, dict[str, Any]] = {}
        emotion_norms: dict[str, float] = {}
        for layer_index in range(expected_layers):
            components = pca_data[layer_index]["components"].float()
            raw_layer_vector = raw_vector[layer_index]
            (
                coefficients,
                neutral_projection,
                clean_layer_vector,
            ) = project_out_neutral_subspace(
                raw_layer_vector,
                components,
            )
            metrics = compute_cleaning_metrics(
                raw_layer_vector=raw_layer_vector,
                coefficients=coefficients,
                neutral_projection=neutral_projection,
                clean_layer_vector=clean_layer_vector,
                components=components,
                num_components=int(
                    pca_data[layer_index]["num_components"]
                ),
                neutral_explained_variance=float(
                    summary_layers[str(layer_index)][
                        "achieved_cumulative_variance"
                    ]
                ),
                pca_orthonormality_error=float(
                    orthonormality_errors[layer_index]
                ),
                epsilon=epsilon,
            )
            validate_layer_cleaning(
                emotion=emotion,
                layer_index=layer_index,
                metrics=metrics,
                raw_norm=metrics["raw_norm"],
                num_components=metrics["num_neutral_components"],
                pca_orthonormality_error=metrics[
                    "pca_orthonormality_error"
                ],
            )
            projection_layers.append(neutral_projection)
            clean_layers.append(clean_layer_vector)
            emotion_metrics[str(layer_index)] = metrics
            emotion_norms[str(layer_index)] = metrics["clean_norm"]
            all_removed_fractions.append(
                metrics["fraction_squared_norm_removed"]
            )
            all_residual_norms.append(metrics["residual_projection_norm"])
            all_reconstruction_errors.append(
                metrics["reconstruction_error"]
            )
            if metrics["clean_norm"] <= near_zero_threshold:
                near_zero_layers.append(
                    {
                        "emotion": emotion,
                        "layer": layer_index,
                        "clean_norm": metrics["clean_norm"],
                    }
                )

        projection = torch.stack(projection_layers, dim=0).float()
        clean = torch.stack(clean_layers, dim=0).float()
        if tuple(projection.shape) != expected_shape:
            raise RuntimeError(
                f"Projection for {emotion!r} has shape {tuple(projection.shape)}"
            )
        if tuple(clean.shape) != expected_shape:
            raise RuntimeError(
                f"Clean vector for {emotion!r} has shape {tuple(clean.shape)}"
            )
        unit = normalise_clean_vectors(clean, epsilon=epsilon)
        if tuple(unit.shape) != expected_shape:
            raise RuntimeError(
                f"Clean unit vector for {emotion!r} has shape {tuple(unit.shape)}"
            )
        projections[emotion] = projection
        clean_vectors[emotion] = clean
        clean_units[emotion] = unit
        clean_norms_json[emotion] = emotion_norms
        metrics_json[emotion] = emotion_metrics
        logger.info("emotion_complete emotion=%s", emotion)

    maximum_orthonormality_error = max(orthonormality_errors.values())
    global_metrics = {
        "maximum_pca_orthonormality_error": (
            maximum_orthonormality_error
        ),
        "maximum_residual_projection_norm": max(all_residual_norms),
        "maximum_reconstruction_error": max(all_reconstruction_errors),
        "minimum_fraction_squared_norm_removed": min(
            all_removed_fractions
        ),
        "maximum_fraction_squared_norm_removed": max(
            all_removed_fractions
        ),
        "near_zero_threshold": near_zero_threshold,
        "number_of_near_zero_clean_vectors": len(near_zero_layers),
        "near_zero_clean_vectors": near_zero_layers,
    }
    return (
        projections,
        clean_vectors,
        clean_units,
        clean_norms_json,
        metrics_json,
        global_metrics,
    )


def compute_cleaning_metrics(
    *,
    raw_layer_vector: torch.Tensor,
    coefficients: torch.Tensor,
    neutral_projection: torch.Tensor,
    clean_layer_vector: torch.Tensor,
    components: torch.Tensor,
    num_components: int,
    neutral_explained_variance: float,
    pca_orthonormality_error: float,
    epsilon: float,
) -> dict[str, Any]:
    """Calculate projection-removal diagnostics for one emotion/layer pair."""

    raw_norm = raw_layer_vector.norm(p=2)
    removed_norm = neutral_projection.norm(p=2)
    clean_norm = clean_layer_vector.norm(p=2)
    residual_coefficients = components @ clean_layer_vector
    residual_projection_norm = residual_coefficients.norm(p=2)
    maximum_absolute_residual = residual_coefficients.abs().max()
    reconstructed = neutral_projection + clean_layer_vector
    reconstruction_error = (reconstructed - raw_layer_vector).norm(p=2)
    clean_projection_dot = torch.dot(
        clean_layer_vector,
        neutral_projection,
    )
    raw_clean_dot = torch.dot(raw_layer_vector, clean_layer_vector)

    raw_norm_value = float(raw_norm.item())
    removed_norm_value = float(removed_norm.item())
    clean_norm_value = float(clean_norm.item())
    clean_projection_dot_value = float(clean_projection_dot.item())
    raw_clean_cosine = float(
        (
            raw_clean_dot
            / (raw_norm * clean_norm).clamp_min(epsilon)
        ).item()
    )
    clean_projection_cosine: float | None
    if clean_norm_value > epsilon and removed_norm_value > epsilon:
        clean_projection_cosine = float(
            (
                clean_projection_dot
                / (clean_norm * removed_norm).clamp_min(epsilon)
            ).item()
        )
    else:
        clean_projection_cosine = None

    metrics: dict[str, Any] = {
        "raw_norm": raw_norm_value,
        "removed_projection_norm": removed_norm_value,
        "clean_norm": clean_norm_value,
        "fraction_squared_norm_removed": float(
            removed_norm_value**2 / (raw_norm_value**2 + epsilon)
        ),
        "fraction_norm_retained": float(
            clean_norm_value / (raw_norm_value + epsilon)
        ),
        "raw_clean_cosine": raw_clean_cosine,
        "maximum_absolute_residual_coefficient": float(
            maximum_absolute_residual.item()
        ),
        "residual_projection_norm": float(
            residual_projection_norm.item()
        ),
        "num_neutral_components": num_components,
        "neutral_explained_variance": neutral_explained_variance,
        "pca_orthonormality_error": pca_orthonormality_error,
        "reconstruction_error": float(reconstruction_error.item()),
        "clean_projection_absolute_dot": abs(
            clean_projection_dot_value
        ),
        "clean_projection_cosine": clean_projection_cosine,
    }
    for field, value in metrics.items():
        if value is not None and isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise ValueError(f"Cleaning metric {field!r} is nonfinite")
    return metrics


def validate_layer_cleaning(
    *,
    emotion: str,
    layer_index: int,
    metrics: Mapping[str, Any],
    raw_norm: float,
    num_components: int,
    pca_orthonormality_error: float,
) -> None:
    """Apply scale-aware reconstruction and residual-projection checks."""

    reconstruction_limit = max(1e-6, 5e-6 * max(raw_norm, 1.0))
    if metrics["reconstruction_error"] > reconstruction_limit:
        raise ValueError(
            f"{emotion!r} layer {layer_index} reconstruction error "
            f"{metrics['reconstruction_error']:.9g} exceeds "
            f"{reconstruction_limit:.9g}"
        )
    residual_limit = max(
        1e-6,
        10.0
        * (
            num_components * pca_orthonormality_error
            + torch.finfo(torch.float32).eps
        )
        * max(raw_norm, 1.0),
    )
    if metrics["residual_projection_norm"] > residual_limit:
        raise ValueError(
            f"{emotion!r} layer {layer_index} residual neutral projection "
            f"{metrics['residual_projection_norm']:.9g} exceeds "
            f"{residual_limit:.9g}"
        )


def stack_vectors_in_emotion_order(
    vectors: Mapping[str, torch.Tensor],
    *,
    emotion_order: Sequence[str],
    expected_layers: int,
    expected_hidden_size: int,
) -> torch.Tensor:
    """Stack exact configured order to ``[emotion,layer,hidden]``."""

    stacked = torch.stack(
        [vectors[emotion] for emotion in emotion_order],
        dim=0,
    ).float()
    expected_shape = (
        len(emotion_order),
        expected_layers,
        expected_hidden_size,
    )
    if tuple(stacked.shape) != expected_shape:
        raise RuntimeError(
            f"Stacked vector shape {tuple(stacked.shape)} != {expected_shape}"
        )
    if not bool(torch.isfinite(stacked).all()):
        raise ValueError("Stacked vectors contain nonfinite values")
    return stacked


def validate_saved_outputs(
    *,
    output_dir: Path,
    source_raw_vectors: Mapping[str, torch.Tensor],
    emotion_order: Sequence[str],
    emotion_slugs: Sequence[str],
    pca_data: Mapping[int, Mapping[str, Any]],
    expected_layers: int,
    expected_hidden_size: int,
    epsilon: float,
    near_zero_threshold: float,
    orthonormality_errors: Mapping[int, float],
) -> None:
    """Reload every saved tensor artifact and verify formulas and shapes."""

    expected_shape = (expected_layers, expected_hidden_size)
    expected_stacked_shape = (
        len(emotion_order),
        expected_layers,
        expected_hidden_size,
    )
    raw_original = torch_load(
        output_dir / "emotion_vectors_raw_original.pt",
        mmap=True,
    )
    projections = torch_load(
        output_dir / "emotion_vectors_neutral_projection.pt",
        mmap=True,
    )
    clean_vectors = torch_load(
        output_dir / "emotion_vectors_clean.pt",
        mmap=True,
    )
    clean_units = torch_load(
        output_dir / "emotion_vectors_clean_unit.pt",
        mmap=True,
    )
    for name, payload in (
        ("raw original", raw_original),
        ("neutral projection", projections),
        ("clean vectors", clean_vectors),
        ("clean unit vectors", clean_units),
    ):
        if not isinstance(payload, dict) or set(payload) != set(emotion_order):
            raise ValueError(f"Saved {name} labels are incompatible")

    for emotion in emotion_order:
        raw = raw_original[emotion]
        projection = projections[emotion]
        clean = clean_vectors[emotion]
        unit = clean_units[emotion]
        for name, tensor in (
            ("raw original", raw),
            ("neutral projection", projection),
            ("clean", clean),
            ("clean unit", unit),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != expected_shape
                or tensor.dtype != torch.float32
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(
                    f"Saved {name} tensor for {emotion!r} is incompatible"
                )
        if not torch.equal(raw, source_raw_vectors[emotion]):
            raise ValueError(
                f"Raw original copy differs for emotion {emotion!r}"
            )
        reconstructed = projection + clean
        reconstruction_errors = (reconstructed - raw).norm(
            p=2,
            dim=-1,
        )
        if float(reconstruction_errors.max().item()) > max(
            1e-6,
            5e-6 * max(float(raw.norm(dim=-1).max().item()), 1.0),
        ):
            raise ValueError(
                f"Saved outputs do not reconstruct raw vector for {emotion!r}"
            )
        clean_norms = clean.norm(p=2, dim=-1)
        unit_norms = unit.norm(p=2, dim=-1)
        nonzero = clean_norms > near_zero_threshold
        if bool(nonzero.any()) and not torch.allclose(
            unit_norms[nonzero],
            torch.ones_like(unit_norms[nonzero]),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                f"Saved clean units for {emotion!r} are not layer-normalised"
            )
        if bool((~nonzero).any()) and not bool(
            torch.isfinite(unit[~nonzero]).all()
        ):
            raise ValueError(
                f"Near-zero clean units for {emotion!r} are nonfinite"
            )
        for layer_index in range(expected_layers):
            components = pca_data[layer_index]["components"].float()
            residual = components @ clean[layer_index]
            raw_norm = float(raw[layer_index].norm().item())
            residual_limit = max(
                1e-6,
                10.0
                * (
                    components.shape[0]
                    * orthonormality_errors[layer_index]
                    + torch.finfo(torch.float32).eps
                )
                * max(raw_norm, 1.0),
            )
            if float(residual.norm().item()) > residual_limit:
                raise ValueError(
                    f"Saved clean vector {emotion!r} layer {layer_index} "
                    "retains excessive neutral projection"
                )

    stacked_clean = torch_load(
        output_dir / "emotion_vectors_clean_stacked.pt",
        mmap=True,
    )
    stacked_unit = torch_load(
        output_dir / "emotion_vectors_clean_unit_stacked.pt",
        mmap=True,
    )
    for name, payload, per_emotion in (
        ("clean stacked", stacked_clean, clean_vectors),
        ("clean unit stacked", stacked_unit, clean_units),
    ):
        if (
            not isinstance(payload, dict)
            or payload.get("emotion_order") != list(emotion_order)
            or payload.get("emotion_slugs") != list(emotion_slugs)
            or not isinstance(payload.get("vectors"), torch.Tensor)
            or tuple(payload["vectors"].shape) != expected_stacked_shape
            or payload["vectors"].dtype != torch.float32
            or not bool(torch.isfinite(payload["vectors"]).all())
        ):
            raise ValueError(f"Saved {name} payload is incompatible")
        expected = torch.stack(
            [per_emotion[emotion] for emotion in emotion_order],
            dim=0,
        )
        if not torch.equal(payload["vectors"], expected):
            raise ValueError(
                f"Saved {name} tensor differs from per-emotion tensors"
            )


def build_config(
    *,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    raw_vector_metadata_path: Path,
    neutral_layer_means_path: Path,
    emotion_order: Sequence[str],
    emotion_slugs: Sequence[str],
) -> dict[str, Any]:
    """Build immutable input and computation provenance."""

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "raw_emotion_vectors": str(paths["raw_emotion_vectors"]),
        "raw_emotion_vectors_sha256": sha256_file(
            paths["raw_emotion_vectors"]
        ),
        "emotion_config": str(paths["emotion_config"]),
        "emotion_config_sha256": sha256_file(paths["emotion_config"]),
        "emotion_metadata": str(paths["emotion_metadata"]),
        "raw_vector_metadata": str(raw_vector_metadata_path),
        "neutral_pca_components": str(paths["neutral_pca_components"]),
        "neutral_pca_components_sha256": sha256_file(
            paths["neutral_pca_components"]
        ),
        "neutral_pca_summary": str(paths["neutral_pca_summary"]),
        "neutral_pca_summary_sha256": sha256_file(
            paths["neutral_pca_summary"]
        ),
        "neutral_pca_metadata": str(paths["neutral_pca_metadata"]),
        "neutral_layer_means": str(neutral_layer_means_path),
        "neutral_means_validated_but_subtracted": False,
        "output_directory": str(paths["output_dir"]),
        "emotion_order": list(emotion_order),
        "emotion_slugs": list(emotion_slugs),
        "expected_emotions": args.expected_emotions,
        "expected_layers": args.expected_layers,
        "expected_hidden_size": args.expected_hidden_size,
        "epsilon": args.epsilon,
        "near_zero_threshold": args.near_zero_threshold,
        "orthonormality_tolerance": args.orthonormality_tolerance,
        "compute_device": "cpu",
        "model_loaded": False,
        "inference_performed": False,
        "pca_recomputed": False,
        "raw_vectors_recomputed": False,
        "cli_arguments": json_safe_cli_arguments(args),
    }


def build_metadata(
    *,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    raw_vector_metadata_path: Path,
    neutral_layer_means_path: Path,
    emotion_order: Sequence[str],
    emotion_slugs: Sequence[str],
    pca_validation: Mapping[str, Any],
    global_metrics: Mapping[str, Any],
    output_dir: Path,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the final completion marker and output inventory."""

    label_to_slug = dict(zip(emotion_order, emotion_slugs, strict=True))
    output_names = [
        "config.json",
        "emotion_vectors_raw_original.pt",
        "emotion_vectors_neutral_projection.pt",
        "emotion_vectors_clean.pt",
        "emotion_vectors_clean_unit.pt",
        "emotion_vectors_clean_stacked.pt",
        "emotion_vectors_clean_unit_stacked.pt",
        "emotion_vectors_clean_norms.json",
        "emotion_vector_cleaning_metrics.json",
        "emotion_vector_cleaning.log",
        "metadata.json",
    ]
    output_files: dict[str, Any] = {}
    for name in output_names:
        path = output_dir / name
        if path.is_file() and name not in {
            "metadata.json",
            "emotion_vector_cleaning.log",
        }:
            output_files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            output_files[name] = None
    neutral_pca_run = paths["neutral_pca_components"].parent
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "number_of_emotions": len(emotion_order),
        "number_of_layers": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "raw_emotion_vector_definition": RAW_VECTOR_DEFINITION,
        "raw_emotion_vector_path": str(paths["raw_emotion_vectors"]),
        "raw_emotion_vector_sha256": sha256_file(
            paths["raw_emotion_vectors"]
        ),
        "raw_vector_metadata": str(raw_vector_metadata_path),
        "neutral_pca_run": str(neutral_pca_run),
        "neutral_pca_components_file": str(
            paths["neutral_pca_components"]
        ),
        "neutral_pca_components_sha256": sha256_file(
            paths["neutral_pca_components"]
        ),
        "neutral_pca_summary": str(paths["neutral_pca_summary"]),
        "neutral_pca_metadata": str(paths["neutral_pca_metadata"]),
        "neutral_layer_means": str(neutral_layer_means_path),
        "neutral_means_subtracted_directly": False,
        "neutral_variance_threshold": 0.5,
        "pca_component_count_per_layer": pca_validation[
            "component_counts"
        ],
        "achieved_neutral_variance_per_layer": pca_validation[
            "achieved_variance"
        ],
        "pca_orthonormality_error_per_layer": {
            str(layer): float(error)
            for layer, error in pca_validation[
                "orthonormality_errors"
            ].items()
        },
        "cleaning_definition": CLEANING_DEFINITION,
        "normalisation": NORMALISATION_DEFINITION,
        "epsilon": args.epsilon,
        "layers_averaged_together": False,
        "clean_vector_shape_per_emotion": [
            args.expected_layers,
            args.expected_hidden_size,
        ],
        "stacked_clean_vector_shape": [
            len(emotion_order),
            args.expected_layers,
            args.expected_hidden_size,
        ],
        "emotion_order": list(emotion_order),
        "emotion_slugs": list(emotion_slugs),
        "label_to_slug": label_to_slug,
        "output_directory": str(output_dir),
        "output_files": output_files,
        "global_metrics": dict(global_metrics),
        "compute_device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "model_loaded": False,
        "inference_performed": False,
        "activations_extracted": False,
        "pca_recomputed": False,
        "raw_vectors_recomputed": False,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "cli_arguments": json_safe_cli_arguments(args),
        "number_of_emotions_processed": len(emotion_order),
        "number_of_layers_processed": args.expected_layers,
        "raw_vector_shape": [args.expected_layers, args.expected_hidden_size],
        "clean_vector_shape": [args.expected_layers, args.expected_hidden_size],
        "stacked_clean_vector_shape_verified": [
            len(emotion_order),
            args.expected_layers,
            args.expected_hidden_size,
        ],
        **global_metrics,
    }


def configure_logging(path: Path) -> logging.Logger:
    """Log cleaning progress to the output directory and standard output."""

    logger = logging.getLogger(f"emotionvectors.clean_vectors.{path.parent}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)sZ %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def torch_load(path: Path, *, mmap: bool = False) -> Any:
    """Load trusted local PyTorch artifacts on CPU."""

    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def json_safe_cli_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse values into JSON-safe provenance."""

    return {
        key: (
            str(value.expanduser().resolve())
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
