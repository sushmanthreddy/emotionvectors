"""Canonical emergent-misalignment direction loading and validation.

The repository's historical EM artifacts are either lists of one vector per
layer or dictionaries keyed by ``layer_N``.  Those containers do not carry
provenance, so callers must explicitly opt into loading them as unverified
legacy artifacts.  New artifacts use the small versioned envelope documented
by :func:`load_em_direction_payload`.

The target in this module is specifically the direction constructed by
``em_organism_dir/steering/activation_steering.py`` for
``q14b_bad_med_32`` (the rank-32 bad-medical-advice adapter). It is not the
paper's rank-1-adapter direction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

CANONICAL_DIRECTION_SIGN = "misaligned_minus_aligned"
CANONICAL_DEFINITION = "mean_misaligned_answers_minus_mean_aligned_answers"
CANONICAL_TARGET_ID = "repository_script_q14b_bad_med_32_rank32_adapter"
CANONICAL_SOURCE_SCRIPT = "em_organism_dir/steering/activation_steering.py"
CANONICAL_ADAPTER_RANK = 32
CANONICAL_TOKEN_AGGREGATION = (
    "pooled_mean_over_all_nonpadding_tokens_after_user_only_chat_template_prefix_"
    "in_full_user_assistant_chat_including_assistant_header_answer_and_chat_suffix"
)
CANONICAL_STORAGE_DTYPE = "float32"
SUPPORTED_SUBTRACTION_DTYPES = frozenset({"bfloat16", "float32"})
VERSIONED_ARTIFACT_KIND = "rq1_em_direction"
VERSIONED_ARTIFACT_SCHEMA = 2

REQUIRED_METADATA_FIELDS = frozenset(
    {
        "model_id",
        "model_revision",
        "hook_site",
        "target_id",
        "source_script",
        "adapter_rank",
        "token_aggregation",
        "subtraction_dtype",
        "storage_dtype",
        "direction_sign",
        "definition",
        "n_layers",
        "hidden_size",
    }
)

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EMDirectionArtifact:
    """A validated layer-wise EM direction and its provenance."""

    vectors: FloatArray
    metadata: Mapping[str, Any]
    legacy_unverified: bool
    source_path: Path | None = None
    source_sha256: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_keys(expected_layers: int) -> tuple[str, ...]:
    return tuple(f"layer_{index}" for index in range(expected_layers))


def _as_vector(
    value: Any,
    *,
    layer: int,
    expected_hidden_size: int,
    require_nonzero: bool,
) -> FloatArray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().to(torch.float32).numpy()
    else:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            raise TypeError(f"layer {layer} must contain a numeric tensor")
        array = array.astype(np.float32, copy=False)
    if array.ndim != 1:
        raise ValueError(f"layer {layer} must be one-dimensional, got shape {tuple(array.shape)}")
    if array.shape[0] != expected_hidden_size:
        raise ValueError(
            f"layer {layer} has hidden size {array.shape[0]}, expected {expected_hidden_size}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"layer {layer} contains non-finite values")
    if require_nonzero and not np.any(array):
        raise ValueError(f"layer {layer} is the zero vector")
    return np.ascontiguousarray(array, dtype=np.float32)


def normalize_layer_vectors(
    payload: Any,
    *,
    expected_layers: int = 48,
    expected_hidden_size: int = 5120,
    require_nonzero: bool = True,
) -> FloatArray:
    """Normalize a repository list or ``layer_N`` dictionary to ``[layer, hidden]``.

    Every layer value must itself be one-dimensional.  A pre-stacked matrix is
    intentionally rejected because the canonical legacy contract specifies
    exactly one tensor per layer.
    """

    if expected_layers <= 0:
        raise ValueError("expected_layers must be positive")
    if expected_hidden_size <= 0:
        raise ValueError("expected_hidden_size must be positive")

    values: Sequence[Any]
    if isinstance(payload, Mapping):
        expected_keys = _layer_keys(expected_layers)
        actual_keys = set(payload)
        if actual_keys != set(expected_keys):
            missing = sorted(set(expected_keys) - actual_keys)
            extra = sorted(actual_keys - set(expected_keys))
            raise ValueError(
                "layer dictionary must contain exactly layer_0 through "
                f"layer_{expected_layers - 1}; missing={missing}, extra={extra}"
            )
        values = [payload[key] for key in expected_keys]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if len(payload) != expected_layers:
            raise ValueError(
                f"EM artifact contains {len(payload)} layers, expected {expected_layers}"
            )
        values = payload
    else:
        raise TypeError("EM vectors must be a layer list or a layer_N dictionary")

    vectors = [
        _as_vector(
            value,
            layer=layer,
            expected_hidden_size=expected_hidden_size,
            require_nonzero=require_nonzero,
        )
        for layer, value in enumerate(values)
    ]
    return np.stack(vectors, axis=0).astype(np.float32, copy=False)


def _validate_metadata(
    metadata: Any,
    *,
    expected_layers: int,
    expected_hidden_size: int,
    expected_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("versioned EM artifact metadata must be a mapping")
    missing = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
    if missing:
        raise ValueError(f"versioned EM artifact metadata is missing fields: {missing}")

    normalized = dict(metadata)
    if normalized["direction_sign"] != CANONICAL_DIRECTION_SIGN:
        raise ValueError(
            "EM direction sign must be "
            f"{CANONICAL_DIRECTION_SIGN!r}, got {normalized['direction_sign']!r}"
        )
    if normalized["definition"] != CANONICAL_DEFINITION:
        raise ValueError(
            f"EM definition must be {CANONICAL_DEFINITION!r}, " f"got {normalized['definition']!r}"
        )
    if normalized["target_id"] != CANONICAL_TARGET_ID:
        raise ValueError(
            f"EM target must be {CANONICAL_TARGET_ID!r}, got {normalized['target_id']!r}"
        )
    if normalized["source_script"] != CANONICAL_SOURCE_SCRIPT:
        raise ValueError(
            "EM source script must be "
            f"{CANONICAL_SOURCE_SCRIPT!r}, got {normalized['source_script']!r}"
        )
    if normalized["adapter_rank"] != CANONICAL_ADAPTER_RANK:
        raise ValueError(
            f"EM adapter rank must be {CANONICAL_ADAPTER_RANK}, "
            f"got {normalized['adapter_rank']!r}"
        )
    if normalized["token_aggregation"] != CANONICAL_TOKEN_AGGREGATION:
        raise ValueError(
            "EM token aggregation must exactly describe the repository pooled-token region: "
            f"{CANONICAL_TOKEN_AGGREGATION!r}"
        )
    if normalized["subtraction_dtype"] not in SUPPORTED_SUBTRACTION_DTYPES:
        raise ValueError(
            "EM subtraction_dtype must be one of "
            f"{sorted(SUPPORTED_SUBTRACTION_DTYPES)!r}, "
            f"got {normalized['subtraction_dtype']!r}"
        )
    if normalized["storage_dtype"] != CANONICAL_STORAGE_DTYPE:
        raise ValueError(
            f"EM storage_dtype must be {CANONICAL_STORAGE_DTYPE!r}, "
            f"got {normalized['storage_dtype']!r}"
        )
    if normalized["n_layers"] != expected_layers:
        raise ValueError(
            f"metadata declares {normalized['n_layers']} layers, expected {expected_layers}"
        )
    if normalized["hidden_size"] != expected_hidden_size:
        raise ValueError(
            "metadata declares hidden size "
            f"{normalized['hidden_size']}, expected {expected_hidden_size}"
        )
    for field in (
        "model_id",
        "model_revision",
        "hook_site",
        "target_id",
        "source_script",
        "token_aggregation",
        "subtraction_dtype",
        "storage_dtype",
    ):
        if not isinstance(normalized[field], str) or not normalized[field].strip():
            raise ValueError(f"metadata field {field!r} must be a non-empty string")

    for key, expected_value in dict(expected_metadata or {}).items():
        if key not in normalized:
            raise ValueError(f"EM artifact metadata is missing expected field {key!r}")
        if normalized[key] != expected_value:
            raise ValueError(
                f"EM artifact metadata mismatch for {key!r}: "
                f"got {normalized[key]!r}, expected {expected_value!r}"
            )
    return normalized


def load_em_direction_payload(
    payload: Any,
    *,
    expected_layers: int = 48,
    expected_hidden_size: int = 5120,
    expected_metadata: Mapping[str, Any] | None = None,
    allow_unverified_legacy: bool = False,
) -> EMDirectionArtifact:
    """Validate a loaded EM payload.

    A versioned payload has this envelope::

        {
            "schema_version": 2,
            "kind": "rq1_em_direction",
            "vectors": [tensor_for_layer_0, ...],
            "metadata": {...},
        }

    Bare lists and bare ``layer_N`` dictionaries are historical repository
    formats.  Because they cannot establish model, hook, or sign provenance,
    they require ``allow_unverified_legacy=True``.
    """

    versioned = (
        isinstance(payload, Mapping)
        and any(key in payload for key in ("schema_version", "kind", "vectors", "metadata"))
        and not all(str(key).startswith("layer_") for key in payload)
    )
    if versioned:
        if payload.get("schema_version") != VERSIONED_ARTIFACT_SCHEMA:
            raise ValueError(
                "unsupported EM artifact schema_version: " f"{payload.get('schema_version')!r}"
            )
        if payload.get("kind") != VERSIONED_ARTIFACT_KIND:
            raise ValueError(f"unsupported EM artifact kind: {payload.get('kind')!r}")
        if "vectors" not in payload:
            raise ValueError("versioned EM artifact does not contain vectors")
        metadata = _validate_metadata(
            payload.get("metadata"),
            expected_layers=expected_layers,
            expected_hidden_size=expected_hidden_size,
            expected_metadata=expected_metadata,
        )
        vectors = normalize_layer_vectors(
            payload["vectors"],
            expected_layers=expected_layers,
            expected_hidden_size=expected_hidden_size,
        )
        return EMDirectionArtifact(vectors, metadata, legacy_unverified=False)

    if not allow_unverified_legacy:
        raise ValueError(
            "legacy EM artifacts contain no model/hook/sign metadata; "
            "set allow_unverified_legacy=True only after verifying provenance"
        )
    vectors = normalize_layer_vectors(
        payload,
        expected_layers=expected_layers,
        expected_hidden_size=expected_hidden_size,
    )
    metadata = {
        "legacy_unverified": True,
        "direction_sign_assumed": CANONICAL_DIRECTION_SIGN,
        **dict(expected_metadata or {}),
    }
    return EMDirectionArtifact(vectors, metadata, legacy_unverified=True)


def load_em_direction(
    path: str | Path,
    *,
    expected_layers: int = 48,
    expected_hidden_size: int = 5120,
    expected_metadata: Mapping[str, Any] | None = None,
    allow_unverified_legacy: bool = False,
) -> EMDirectionArtifact:
    """Load a PyTorch EM artifact safely on CPU and validate its contract."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"could not load EM artifact {source}") from exc
    artifact = load_em_direction_payload(
        payload,
        expected_layers=expected_layers,
        expected_hidden_size=expected_hidden_size,
        expected_metadata=expected_metadata,
        allow_unverified_legacy=allow_unverified_legacy,
    )
    return EMDirectionArtifact(
        vectors=artifact.vectors,
        metadata=artifact.metadata,
        legacy_unverified=artifact.legacy_unverified,
        source_path=source.resolve(),
        source_sha256=_sha256_file(source),
    )


def _answer_layers(payload: Any) -> Any:
    if isinstance(payload, Mapping) and "answer" in payload:
        return payload["answer"]
    return payload


def _ordered_answer_values(payload: Any, expected_layers: int) -> list[Any]:
    layers = _answer_layers(payload)
    if isinstance(layers, Mapping):
        keys = _layer_keys(expected_layers)
        if set(layers) != set(keys):
            raise ValueError("answer means must contain exactly the expected layer_N keys")
        return [layers[key] for key in keys]
    if isinstance(layers, Sequence) and not isinstance(layers, (str, bytes, bytearray)):
        if len(layers) != expected_layers:
            raise ValueError(
                f"answer activation artifact contains {len(layers)} layers, expected {expected_layers}"
            )
        return list(layers)
    raise TypeError("answer means must be a layer list or a layer_N dictionary")


def _answer_dtype(value: Any, layer: int) -> str:
    if isinstance(value, torch.Tensor):
        if value.dtype is torch.bfloat16:
            return "bfloat16"
        if value.dtype is torch.float32:
            return "float32"
        raise TypeError(f"answer mean layer {layer} has unsupported dtype {value.dtype}")
    if np.asarray(value).dtype == np.float32:
        return "float32"
    raise TypeError(f"answer mean layer {layer} must use bfloat16 or float32")


def answer_mean_subtraction_dtype(
    misaligned_answer_means: Any,
    aligned_answer_means: Any,
    *,
    expected_layers: int = 48,
) -> str:
    """Identify the uniform dtype used by repository-style layer subtraction."""

    values = (
        *_ordered_answer_values(misaligned_answer_means, expected_layers),
        *_ordered_answer_values(aligned_answer_means, expected_layers),
    )
    dtypes = {_answer_dtype(value, layer) for layer, value in enumerate(values)}
    if len(dtypes) != 1:
        raise TypeError(f"answer means must have one uniform dtype, got {sorted(dtypes)!r}")
    return dtypes.pop()


def build_em_direction_from_answer_means(
    misaligned_answer_means: Any,
    aligned_answer_means: Any,
    *,
    expected_layers: int = 48,
    expected_hidden_size: int = 5120,
) -> FloatArray:
    """Return the repository definition: misaligned-answer mean minus aligned.

    Inputs may be direct layer containers or repository hidden-state mappings
    with an ``answer`` member. They must already contain the repository's
    globally pooled response-region means. BF16 inputs are subtracted in BF16,
    matching ``subtract_layerwise``, and only then converted to float32 for
    normalized storage. Explicit float32 inputs are subtracted in float32 and
    are versioned as such by the pipeline. This function never substitutes
    model differences on neutral prompts.
    """

    dtype = answer_mean_subtraction_dtype(
        misaligned_answer_means, aligned_answer_means, expected_layers=expected_layers
    )
    misaligned = _ordered_answer_values(misaligned_answer_means, expected_layers)
    aligned = _ordered_answer_values(aligned_answer_means, expected_layers)
    differences: list[torch.Tensor] = []
    expected_torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    for left, right in zip(misaligned, aligned, strict=True):
        left_tensor = (
            left.detach().cpu()
            if isinstance(left, torch.Tensor)
            else torch.from_numpy(np.asarray(left))
        )
        right_tensor = (
            right.detach().cpu()
            if isinstance(right, torch.Tensor)
            else torch.from_numpy(np.asarray(right))
        )
        if (
            left_tensor.dtype is not expected_torch_dtype
            or right_tensor.dtype is not expected_torch_dtype
        ):
            raise TypeError("answer means changed dtype during subtraction preparation")
        # Deliberately subtract before conversion: repository BF16 subtraction
        # rounds differently from subtracting float32-converted BF16 operands.
        differences.append(left_tensor - right_tensor)
    difference = normalize_layer_vectors(
        differences,
        expected_layers=expected_layers,
        expected_hidden_size=expected_hidden_size,
        require_nonzero=False,
    )
    zero_layers = np.flatnonzero(np.linalg.norm(difference, axis=1) == 0)
    if zero_layers.size:
        raise ValueError(f"EM answer difference is zero at layers {zero_layers.tolist()}")
    return difference.astype(np.float32, copy=False)


__all__ = [
    "CANONICAL_ADAPTER_RANK",
    "CANONICAL_DEFINITION",
    "CANONICAL_DIRECTION_SIGN",
    "CANONICAL_SOURCE_SCRIPT",
    "CANONICAL_STORAGE_DTYPE",
    "CANONICAL_TARGET_ID",
    "CANONICAL_TOKEN_AGGREGATION",
    "REQUIRED_METADATA_FIELDS",
    "SUPPORTED_SUBTRACTION_DTYPES",
    "VERSIONED_ARTIFACT_KIND",
    "VERSIONED_ARTIFACT_SCHEMA",
    "EMDirectionArtifact",
    "answer_mean_subtraction_dtype",
    "build_em_direction_from_answer_means",
    "load_em_direction",
    "load_em_direction_payload",
    "normalize_layer_vectors",
]
