"""Projection and percentile thresholding for held-out activation sweeps.

The exact percentile is a dataset-wide reduction.  Until profiling demonstrates
that a custom implementation wins over cuBLAS plus ``torch.quantile``, this
module intentionally keeps a single, well-tested PyTorch implementation.  Its
API accepts precomputed thresholds so a corpus can be processed in a streaming
second pass without retaining raw activations.
"""

from __future__ import annotations

from typing import TypeAlias, overload

import torch
from torch import Tensor

ProjectionThresholdPair: TypeAlias = tuple[Tensor, Tensor]
ProjectionThresholdTriple: TypeAlias = tuple[Tensor, Tensor, Tensor]


def _validate_inputs(
    activations: Tensor,
    vectors: Tensor,
    attention_mask: Tensor | None,
) -> tuple[Tensor, Tensor, tuple[int, ...], Tensor]:
    if activations.ndim < 2:
        raise ValueError("activations must have shape [..., hidden]")
    if vectors.ndim == 1:
        vectors = vectors.unsqueeze(0)
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [n_vectors, hidden]")
    if activations.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            "activation and vector hidden sizes differ: "
            f"{activations.shape[-1]} != {vectors.shape[-1]}"
        )
    if activations.device != vectors.device:
        raise ValueError("activations and vectors must be on the same device")

    leading_shape = tuple(activations.shape[:-1])
    if attention_mask is None:
        valid = torch.ones(leading_shape, dtype=torch.bool, device=activations.device)
    else:
        if tuple(attention_mask.shape) != leading_shape:
            raise ValueError(
                "attention_mask must match activation leading dimensions; "
                f"expected {leading_shape}, got {tuple(attention_mask.shape)}"
            )
        valid = attention_mask.to(device=activations.device, dtype=torch.bool)

    flat_activations = activations.reshape(-1, activations.shape[-1])
    return flat_activations, vectors, leading_shape, valid.reshape(-1)


@overload
def project_threshold_fallback(
    activations: Tensor,
    vectors: Tensor,
    percentile: float = 90.0,
    *,
    attention_mask: Tensor | None = None,
    thresholds: Tensor | None = None,
    return_thresholds: bool = False,
) -> ProjectionThresholdPair: ...


@overload
def project_threshold_fallback(
    activations: Tensor,
    vectors: Tensor,
    percentile: float = 90.0,
    *,
    attention_mask: Tensor | None = None,
    thresholds: Tensor | None = None,
    return_thresholds: bool,
) -> ProjectionThresholdPair | ProjectionThresholdTriple: ...


def project_threshold_fallback(
    activations: Tensor,
    vectors: Tensor,
    percentile: float = 90.0,
    *,
    attention_mask: Tensor | None = None,
    thresholds: Tensor | None = None,
    return_thresholds: bool = False,
) -> ProjectionThresholdPair | ProjectionThresholdTriple:
    """Project activations and flag values strictly above each cutoff.

    Args:
        activations: ``[..., hidden]`` residual-stream activations.
        vectors: ``[n_vectors, hidden]`` emotion vectors.
        percentile: Percentile in ``[0, 100]`` used when ``thresholds`` is not
            supplied.  It is computed over every valid leading position,
            independently for each vector.
        attention_mask: Optional boolean/integer mask matching ``...``.
        thresholds: Optional precomputed scalar per vector, useful for streamed
            corpus batches.
        return_thresholds: Include the cutoff tensor as a third return value.

    Projections always use float32 accumulation/output.  Padded positions retain
    their numeric projection but are never marked above threshold.
    """

    if not 0.0 <= float(percentile) <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    flat, vector_matrix, leading_shape, valid = _validate_inputs(
        activations, vectors, attention_mask
    )

    projections_flat = torch.matmul(
        flat.to(torch.float32), vector_matrix.to(torch.float32).transpose(0, 1)
    )
    n_vectors = vector_matrix.shape[0]

    if thresholds is None:
        if not bool(torch.any(valid)):
            raise ValueError("cannot compute percentiles without any valid activations")
        cutoffs = torch.quantile(projections_flat[valid], float(percentile) / 100.0, dim=0)
    else:
        cutoffs = torch.as_tensor(thresholds, dtype=torch.float32, device=activations.device)
        if cutoffs.ndim == 0:
            cutoffs = cutoffs.expand(n_vectors)
        if cutoffs.shape != (n_vectors,):
            raise ValueError(
                f"thresholds must have shape {(n_vectors,)}, got {tuple(cutoffs.shape)}"
            )

    above_flat = (projections_flat > cutoffs[None, :]) & valid[:, None]
    output_shape = (*leading_shape, n_vectors)
    projections = projections_flat.reshape(output_shape)
    above = above_flat.reshape(output_shape)
    if return_thresholds:
        return projections, above, cutoffs
    return projections, above


@overload
def project_threshold(
    activations: Tensor,
    vectors: Tensor,
    percentile: float = 90.0,
    *,
    attention_mask: Tensor | None = None,
    thresholds: Tensor | None = None,
    use_kernel: bool = True,
    use_kernels: bool | None = None,
    return_thresholds: bool = False,
) -> ProjectionThresholdPair: ...


@overload
def project_threshold(
    activations: Tensor,
    vectors: Tensor,
    percentile: float = 90.0,
    *,
    attention_mask: Tensor | None = None,
    thresholds: Tensor | None = None,
    use_kernel: bool = True,
    use_kernels: bool | None = None,
    return_thresholds: bool,
) -> ProjectionThresholdPair | ProjectionThresholdTriple: ...


def project_threshold(
    activations: Tensor,
    vectors: Tensor,
    percentile: float = 90.0,
    *,
    attention_mask: Tensor | None = None,
    thresholds: Tensor | None = None,
    use_kernel: bool = True,
    use_kernels: bool | None = None,
    return_thresholds: bool = False,
) -> ProjectionThresholdPair | ProjectionThresholdTriple:
    """Dispatch projection/thresholding.

    ``use_kernel`` is accepted for a stable dispatcher API.  The current exact
    implementation uses optimized PyTorch matmul/quantile operations on both CPU
    and CUDA; no speculative Triton percentile kernel is selected.
    """

    del use_kernel, use_kernels
    return project_threshold_fallback(
        activations,
        vectors,
        percentile,
        attention_mask=attention_mask,
        thresholds=thresholds,
        return_thresholds=return_thresholds,
    )


project_threshold_torch = project_threshold_fallback
torch_project_threshold = project_threshold_fallback


__all__ = [
    "project_threshold",
    "project_threshold_fallback",
    "project_threshold_torch",
    "torch_project_threshold",
]
