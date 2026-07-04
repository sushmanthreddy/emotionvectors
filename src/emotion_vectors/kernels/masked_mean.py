"""Attention-mask aware token reduction.

The CUDA implementation consumes a residual-stream tensor and writes only one
vector per sequence.  This is important for activation extraction: hook code
must not retain a full ``[batch, sequence, hidden]`` tensor after a transformer
block has returned.

The Triton dependency is deliberately optional.  CPU runs, unsupported dtypes,
and environments without Triton use the reference PyTorch implementation.
"""

from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Sequence

import torch
from torch import Tensor

try:  # pragma: no cover - availability is environment dependent.
    import triton
    import triton.language as tl
except (ImportError, OSError):  # pragma: no cover - exercised without Triton.
    triton = None
    tl = None

_CONFIGURED_COMPILER = os.environ.get("CC")
_TRITON_COMPILER_AVAILABLE = bool(
    (_CONFIGURED_COMPILER and shutil.which(_CONFIGURED_COMPILER))
    or any(shutil.which(name) for name in ("cc", "gcc", "clang"))
)


def _normalise_inputs(
    hidden_states: Tensor,
    attention_mask: Tensor | None,
    token_start: int | Tensor | Sequence[int],
) -> tuple[Tensor, Tensor, Tensor, bool]:
    """Validate inputs and return batched tensors plus a squeeze flag."""

    if hidden_states.ndim not in (2, 3):
        raise ValueError(
            "hidden_states must have shape [sequence, hidden] or "
            f"[batch, sequence, hidden], got {tuple(hidden_states.shape)}"
        )

    squeeze_batch = hidden_states.ndim == 2
    states = hidden_states.unsqueeze(0) if squeeze_batch else hidden_states
    batch_size, sequence_length, _ = states.shape

    if attention_mask is None:
        mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=states.device,
        )
    else:
        mask = attention_mask
        if squeeze_batch and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must match the leading hidden-state dimensions; "
                f"expected {(batch_size, sequence_length)}, got {tuple(mask.shape)}"
            )
        mask = mask.to(device=states.device, dtype=torch.bool)

    if isinstance(token_start, int):
        starts = torch.full((batch_size,), token_start, dtype=torch.int64, device=states.device)
    else:
        starts = torch.as_tensor(token_start, dtype=torch.int64, device=states.device)
        if starts.ndim == 0:
            starts = starts.expand(batch_size)
        if starts.shape != (batch_size,):
            raise ValueError(
                "token_start must be a scalar or have one value per sequence; "
                f"expected {(batch_size,)}, got {tuple(starts.shape)}"
            )
    # Avoid a device synchronization in every layer hook.  Extraction validates
    # starts once before the forward; direct CPU calls still fail eagerly.
    if starts.device.type == "cpu" and bool(torch.any(starts < 0)):
        raise ValueError("token_start values must be non-negative")

    return states, mask, starts, squeeze_batch


def eligible_token_mask(
    attention_mask: Tensor,
    token_start: int | Tensor | Sequence[int],
) -> Tensor:
    """Return tokens at or after ``token_start`` among non-padding tokens.

    Counting *valid-token ordinals* instead of padded tensor positions makes the
    operation correct for both left- and right-padded tokenizers.
    """

    if attention_mask.ndim not in (1, 2):
        raise ValueError("attention_mask must have shape [sequence] or [batch, sequence]")
    squeeze_batch = attention_mask.ndim == 1
    mask = attention_mask.unsqueeze(0) if squeeze_batch else attention_mask
    mask = mask.to(dtype=torch.bool)
    batch_size = mask.shape[0]

    if isinstance(token_start, int):
        starts = torch.full((batch_size,), token_start, dtype=torch.int64, device=mask.device)
    else:
        starts = torch.as_tensor(token_start, dtype=torch.int64, device=mask.device)
        if starts.ndim == 0:
            starts = starts.expand(batch_size)
        if starts.shape != (batch_size,):
            raise ValueError("token_start must be a scalar or have one value per sequence")
    if starts.device.type == "cpu" and bool(torch.any(starts < 0)):
        raise ValueError("token_start values must be non-negative")

    ordinal = torch.cumsum(mask.to(torch.int64), dim=1) - 1
    selected = mask & (ordinal >= starts[:, None])
    return selected.squeeze(0) if squeeze_batch else selected


def eligible_token_counts(
    attention_mask: Tensor,
    token_start: int | Tensor | Sequence[int],
) -> Tensor:
    """Count selected, non-padding tokens for every sequence."""

    selected = eligible_token_mask(attention_mask, token_start)
    if selected.ndim == 1:
        return selected.sum().reshape(1)
    return selected.sum(dim=1)


def masked_mean_fallback(
    hidden_states: Tensor,
    attention_mask: Tensor | None = None,
    token_start: int | Tensor | Sequence[int] = 0,
) -> Tensor:
    """Reference reduction over valid tokens at/after ``token_start``.

    Accumulation and output are float32 even when model activations are bf16 or
    fp16.  Rows with no eligible token are represented by zeros; extraction code
    normally prevents such rows via its short-story policy.
    """

    states, mask, starts, squeeze_batch = _normalise_inputs(
        hidden_states, attention_mask, token_start
    )
    selected = eligible_token_mask(mask, starts)
    counts = selected.sum(dim=1)

    # The extra tensor is confined to the fallback.  The Triton path below
    # performs masking and fp32 accumulation without materialising it.
    masked = states * selected.unsqueeze(-1).to(dtype=states.dtype)
    totals = masked.sum(dim=1, dtype=torch.float32)
    means = totals / counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
    means = torch.where(counts[:, None] > 0, means, torch.zeros_like(means))
    return means.squeeze(0) if squeeze_batch else means


if triton is not None:  # pragma: no branch - definition depends on installation.

    @triton.jit
    def _masked_mean_kernel(  # pragma: no cover - compiled only on CUDA.
        states_ptr,
        mask_ptr,
        starts_ptr,
        output_ptr,
        sequence_length,
        hidden_size,
        state_stride_batch,
        state_stride_sequence,
        state_stride_hidden,
        mask_stride_batch,
        mask_stride_sequence,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        hidden_block = tl.program_id(1)
        hidden_offsets = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        hidden_in_bounds = hidden_offsets < hidden_size

        start = tl.load(starts_ptr + batch_index)
        accumulator = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
        selected_count = 0
        valid_ordinal = 0

        # sequence_length remains a runtime value so a new sequence length does
        # not force a fully-unrolled compilation.
        for position in range(0, sequence_length):
            valid = (
                tl.load(
                    mask_ptr + batch_index * mask_stride_batch + position * mask_stride_sequence
                )
                != 0
            )
            take = valid & (valid_ordinal >= start)
            values = tl.load(
                states_ptr
                + batch_index * state_stride_batch
                + position * state_stride_sequence
                + hidden_offsets * state_stride_hidden,
                mask=hidden_in_bounds,
                other=0.0,
            )
            accumulator += tl.where(take, values, 0.0)
            selected_count += take
            valid_ordinal += valid

        denominator = tl.maximum(selected_count, 1).to(tl.float32)
        result = accumulator / denominator
        result = tl.where(selected_count > 0, result, 0.0)
        output_offsets = batch_index * hidden_size + hidden_offsets
        tl.store(output_ptr + output_offsets, result, mask=hidden_in_bounds)


def triton_masked_mean_available(hidden_states: Tensor | None = None) -> bool:
    """Whether the fused implementation can run for ``hidden_states``."""

    if triton is None or not torch.cuda.is_available() or not _TRITON_COMPILER_AVAILABLE:
        return False
    if hidden_states is None:
        return True
    return hidden_states.is_cuda and hidden_states.dtype in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }


def masked_mean_triton(
    hidden_states: Tensor,
    attention_mask: Tensor | None = None,
    token_start: int | Tensor | Sequence[int] = 0,
) -> Tensor:
    """Run the fused CUDA masked mean.

    Raises ``RuntimeError`` when the CUDA/Triton path is unavailable.  Callers
    normally use :func:`masked_mean`, which dispatches safely.
    """

    states, mask, starts, squeeze_batch = _normalise_inputs(
        hidden_states, attention_mask, token_start
    )
    if not triton_masked_mean_available(states):
        raise RuntimeError("the Triton masked-mean kernel is unavailable")

    batch_size, sequence_length, hidden_size = states.shape
    output = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=states.device)
    starts = starts.to(dtype=torch.int32).contiguous()
    mask = mask.contiguous()
    block_hidden = 256
    grid = (batch_size, triton.cdiv(hidden_size, block_hidden))
    _masked_mean_kernel[grid](
        states,
        mask,
        starts,
        output,
        sequence_length,
        hidden_size,
        states.stride(0),
        states.stride(1),
        states.stride(2),
        mask.stride(0),
        mask.stride(1),
        BLOCK_HIDDEN=block_hidden,
    )
    return output.squeeze(0) if squeeze_batch else output


_warned_about_triton_failure = False


def masked_mean(
    hidden_states: Tensor,
    attention_mask: Tensor | None = None,
    token_start: int | Tensor | Sequence[int] = 0,
    *,
    use_kernel: bool = True,
    use_kernels: bool | None = None,
) -> Tensor:
    """Dispatch to the fused CUDA reduction or the PyTorch fallback."""

    global _warned_about_triton_failure
    if use_kernels is not None:
        use_kernel = bool(use_kernels)
    if use_kernel and triton_masked_mean_available(hidden_states):
        try:
            return masked_mean_triton(hidden_states, attention_mask, token_start)
        except Exception as exc:  # Triton exposes backend-specific compile errors.
            # Compilation can fail for a particular driver/architecture.  The
            # correct fallback is preferable to aborting a long extraction run.
            if not _warned_about_triton_failure:
                warnings.warn(
                    f"Triton masked mean failed; using PyTorch fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warned_about_triton_failure = True
    return masked_mean_fallback(hidden_states, attention_mask, token_start)


masked_mean_torch = masked_mean_fallback
torch_masked_mean = masked_mean_fallback


__all__ = [
    "eligible_token_counts",
    "eligible_token_mask",
    "masked_mean",
    "masked_mean_fallback",
    "masked_mean_torch",
    "masked_mean_triton",
    "torch_masked_mean",
    "triton_masked_mean_available",
]
