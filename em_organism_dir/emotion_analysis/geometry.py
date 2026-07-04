"""Pure geometric operations for RQ1 emotion/EM comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SubspaceProjection:
    """Projection of one target onto the retained span of basis vectors."""

    projected: FloatArray
    explained_fraction: float
    cosine: float
    effective_rank: int
    condition_number: float
    singular_values: FloatArray


@dataclass(frozen=True, slots=True)
class LayerwiseSubspaceProjection:
    """Per-layer emotion-subspace reconstruction results."""

    projected: FloatArray
    explained_fraction: FloatArray
    cosine: FloatArray
    effective_rank: npt.NDArray[np.int64]
    condition_number: FloatArray


def _finite_array(value: Any, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def unit_vector(vector: Any, *, name: str = "vector") -> FloatArray:
    """Return a finite one-dimensional unit vector."""

    array = _finite_array(vector, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have a non-zero norm")
    return array / norm


def cosine_similarity(left: Any, right: Any) -> float:
    """Compute signed cosine similarity between two non-zero vectors."""

    left_unit = unit_vector(left, name="left")
    right_unit = unit_vector(right, name="right")
    if left_unit.shape != right_unit.shape:
        raise ValueError(
            f"cosine vectors must have the same shape, got {left_unit.shape} and "
            f"{right_unit.shape}"
        )
    return float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))


def normalized_centroid(vectors: Any) -> FloatArray:
    """Normalize every vector before averaging, then normalize the centroid."""

    array = _finite_array(vectors, name="vectors")
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("vectors must have shape [n_vectors, hidden] with n_vectors > 0")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        bad = np.flatnonzero(norms <= np.finfo(np.float64).eps).tolist()
        raise ValueError(f"centroid inputs contain zero vectors at indices {bad}")
    mean = (array / norms[:, None]).mean(axis=0)
    return unit_vector(mean, name="centroid")


def layerwise_normalized_centroid(vectors: Any) -> FloatArray:
    """Construct normalized centroids for ``[emotion, layer, hidden]`` vectors."""

    array = _finite_array(vectors, name="vectors")
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("vectors must have shape [emotion, layer, hidden]")
    return np.stack(
        [normalized_centroid(array[:, layer, :]) for layer in range(array.shape[1])],
        axis=0,
    )


def project_onto_subspace(
    target: Any,
    basis_vectors: Any,
    *,
    relative_tolerance: float = 1e-6,
    normalize_basis: bool = True,
) -> SubspaceProjection:
    """Project a target onto the SVD-retained span of row-wise basis vectors.

    ``basis_vectors`` has shape ``[n_vectors, hidden]``.  The scientific RQ1
    construction uses unit emotion vectors as columns of its design matrix, so
    row vectors are normalized by default before transposition and SVD.
    """

    target_array = _finite_array(target, name="target")
    if target_array.ndim != 1:
        raise ValueError(f"target must be one-dimensional, got {target_array.shape}")
    target_norm = float(np.linalg.norm(target_array))
    if target_norm <= np.finfo(np.float64).eps:
        raise ValueError("target must have a non-zero norm")
    basis = _finite_array(basis_vectors, name="basis_vectors")
    if basis.ndim != 2 or basis.shape[0] == 0:
        raise ValueError("basis_vectors must have shape [n_vectors, hidden]")
    if basis.shape[1] != target_array.shape[0]:
        raise ValueError(
            f"basis hidden size {basis.shape[1]} does not match target " f"{target_array.shape[0]}"
        )
    if not np.isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be finite and in (0, 1)")

    norms = np.linalg.norm(basis, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        bad = np.flatnonzero(norms <= np.finfo(np.float64).eps).tolist()
        raise ValueError(f"basis contains zero vectors at indices {bad}")
    if normalize_basis:
        basis = basis / norms[:, None]

    # Emotion vectors are rows above, hence columns in the scientific E matrix.
    design = basis.T
    left, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= np.finfo(np.float64).eps:
        raise ValueError("basis span has zero numerical rank")
    retained = singular_values > relative_tolerance * singular_values[0]
    rank = int(retained.sum())
    if rank == 0:
        raise ValueError("basis span has zero rank at the configured tolerance")
    orthonormal_basis = left[:, retained]
    projected = orthonormal_basis @ (orthonormal_basis.T @ target_array)
    explained = float(np.dot(projected, projected) / (target_norm * target_norm))
    explained = float(np.clip(explained, 0.0, 1.0))
    projected_norm = float(np.linalg.norm(projected))
    cosine = (
        float(
            np.clip(
                np.dot(projected, target_array) / (projected_norm * target_norm),
                -1.0,
                1.0,
            )
        )
        if projected_norm > np.finfo(np.float64).eps
        else 0.0
    )
    retained_values = singular_values[retained]
    condition = float(retained_values[0] / retained_values[-1])
    return SubspaceProjection(
        projected=projected,
        explained_fraction=explained,
        cosine=cosine,
        effective_rank=rank,
        condition_number=condition,
        singular_values=singular_values,
    )


def layerwise_subspace_projection(
    targets: Any,
    emotion_vectors: Any,
    *,
    relative_tolerance: float = 1e-6,
    normalize_basis: bool = True,
) -> LayerwiseSubspaceProjection:
    """Run :func:`project_onto_subspace` independently at every layer."""

    target_array = _finite_array(targets, name="targets")
    emotions = _finite_array(emotion_vectors, name="emotion_vectors")
    if target_array.ndim != 2:
        raise ValueError("targets must have shape [layer, hidden]")
    if emotions.ndim != 3:
        raise ValueError("emotion_vectors must have shape [emotion, layer, hidden]")
    if emotions.shape[1:] != target_array.shape:
        raise ValueError(
            f"emotion layer/hidden shape {emotions.shape[1:]} does not match "
            f"targets {target_array.shape}"
        )

    results = [
        project_onto_subspace(
            target_array[layer],
            emotions[:, layer, :],
            relative_tolerance=relative_tolerance,
            normalize_basis=normalize_basis,
        )
        for layer in range(target_array.shape[0])
    ]
    return LayerwiseSubspaceProjection(
        projected=np.stack([result.projected for result in results], axis=0),
        explained_fraction=np.asarray(
            [result.explained_fraction for result in results], dtype=np.float64
        ),
        cosine=np.asarray([result.cosine for result in results], dtype=np.float64),
        effective_rank=np.asarray([result.effective_rank for result in results], dtype=np.int64),
        condition_number=np.asarray(
            [result.condition_number for result in results], dtype=np.float64
        ),
    )


__all__ = [
    "LayerwiseSubspaceProjection",
    "SubspaceProjection",
    "cosine_similarity",
    "layerwise_normalized_centroid",
    "layerwise_subspace_projection",
    "normalized_centroid",
    "project_onto_subspace",
    "unit_vector",
]
