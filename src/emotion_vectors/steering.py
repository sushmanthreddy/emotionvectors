"""Norm-relative residual-stream steering utilities used by V6."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def scaled_steering_vector(vector: Tensor, layer_norm: float | Tensor, strength: float) -> Tensor:
    """Return ``strength * layer_norm * unit(vector)`` (the A6 convention)."""

    if vector.ndim != 1:
        raise ValueError(f"vector must be one-dimensional, got shape {tuple(vector.shape)}")
    if not math.isfinite(float(strength)):
        raise ValueError("strength must be finite")
    working = vector if vector.is_floating_point() else vector.to(torch.float32)
    magnitude = torch.linalg.vector_norm(working.float())
    if not torch.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Cannot steer with a zero or non-finite vector")
    norm_tensor = torch.as_tensor(layer_norm, dtype=torch.float32, device=vector.device)
    if norm_tensor.numel() != 1 or not torch.isfinite(norm_tensor) or norm_tensor <= 0:
        raise ValueError("layer_norm must be a finite positive scalar")
    return working * (float(strength) * norm_tensor.to(working.dtype) / magnitude.to(working.dtype))


def apply_steering(
    hidden_states: Tensor,
    delta: Tensor,
    token_mask: Tensor | None = None,
) -> Tensor:
    """Add a steering delta to selected ``[batch, sequence]`` positions."""

    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, sequence, d_model], "
            f"got {tuple(hidden_states.shape)}"
        )
    if delta.ndim != 1 or delta.shape[0] != hidden_states.shape[-1]:
        raise ValueError(
            f"delta must have shape [{hidden_states.shape[-1]}], got {tuple(delta.shape)}"
        )
    addition = delta.to(device=hidden_states.device, dtype=hidden_states.dtype)
    if token_mask is None:
        return hidden_states + addition.view(1, 1, -1)
    if token_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"token_mask must have shape {tuple(hidden_states.shape[:2])}, "
            f"got {tuple(token_mask.shape)}"
        )
    mask = token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states + mask.unsqueeze(-1) * addition.view(1, 1, -1)


@dataclass
class SteeringHook:
    """Forward-hook callable that preserves tuple/list model outputs."""

    delta: Tensor
    token_mask: Tensor | None = None

    def __call__(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        if isinstance(output, Tensor):
            return apply_steering(output, self.delta, self.token_mask)
        if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
            return (apply_steering(output[0], self.delta, self.token_mask), *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], Tensor):
            return [apply_steering(output[0], self.delta, self.token_mask), *output[1:]]
        raise TypeError(f"Unsupported transformer-block output type: {type(output).__name__}")


@contextmanager
def steering_hook(
    layer: nn.Module,
    vector: Tensor,
    *,
    layer_norm: float | Tensor,
    strength: float,
    token_mask: Tensor | None = None,
) -> Iterator[Tensor]:
    """Install a norm-relative hook for one forward pass or generation context."""

    delta = scaled_steering_vector(vector, layer_norm, strength)
    handle = layer.register_forward_hook(SteeringHook(delta, token_mask))
    try:
        yield delta
    finally:
        handle.remove()


def assistant_prefix_mask(
    tokenizer: Any,
    prompts: str | Sequence[str],
    *,
    marker: str = "Assistant:",
    device: torch.device | str | None = None,
) -> Tensor:
    """Build a mask from the final Assistant marker through each prompt's last token.

    V6 prompts each contain exactly one Assistant prefix.  Computing the boundary by
    tokenizing the text before the marker avoids assumptions about Qwen's token IDs.
    The returned mask follows the tokenizer's padding side and includes ``Assistant:``
    and the ``He feels``/``I feel`` suffix.
    """

    texts = [prompts] if isinstance(prompts, str) else list(prompts)
    if not texts:
        raise ValueError("At least one prompt is required")
    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    encoded = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=True)
    attention = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
    masks = torch.zeros_like(attention, dtype=torch.bool)
    try:
        for row, text in enumerate(texts):
            boundary = text.rfind(marker)
            if boundary < 0:
                raise ValueError(f"Prompt does not contain Assistant marker: {text!r}")
            prefix_ids = tokenizer(text[:boundary], add_special_tokens=True)["input_ids"]
            actual_length = int(attention[row].sum().item())
            pad_length = (
                attention.shape[1] - actual_length if previous_padding_side == "left" else 0
            )
            start = min(pad_length + len(prefix_ids), attention.shape[1] - 1)
            masks[row, start:] = attention[row, start:].bool()
    finally:
        tokenizer.padding_side = previous_padding_side
    return masks.to(device=device) if device is not None else masks


def single_token_id(tokenizer: Any, word: str) -> int | None:
    """Return the ID for a matching emotion continuation, or ``None`` if multi-token.

    Causal LMs normally represent the continuation with a leading-space token, so that
    form is preferred and the unprefixed form is used only as a fallback.
    """

    for candidate in (f" {word}", word):
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
    return None
