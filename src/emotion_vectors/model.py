"""Hugging Face model loading and residual-stream capture.

The extraction wrapper registers one hook per decoder block, runs the model
exactly once per batch, and reduces each block output while that hook is active.
Only ``[batch, layers, hidden]`` tensors survive the forward pass.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .kernels import (
    eligible_token_counts,
    eligible_token_mask,
    masked_mean,
    triton_masked_mean_available,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureBatch:
    """Reduced activations and optional statistics from one model forward."""

    vectors: Tensor
    token_counts: Tensor
    mean_token_norms: Tensor | None = None

    @property
    def reduced(self) -> Tensor:
        """Alias describing ``vectors`` at the capture boundary."""

        return self.vectors


def resolve_torch_dtype(name: str | torch.dtype) -> torch.dtype:
    """Resolve a config dtype without silently accepting unsupported values."""

    if isinstance(name, torch.dtype):
        return name
    normalised = str(name).strip().lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype: {name!r}") from exc


def resolve_attention_backend(requested: str) -> str:
    """Resolve the attention implementation that this environment will use."""

    backend = str(requested)
    if backend == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        return "sdpa"
    return backend


def resolve_masked_mean_backend(
    use_kernels: bool,
    *,
    device: torch.device | str | None = None,
) -> str:
    """Return the effective masked-mean implementation for cache identity."""

    if not use_kernels:
        return "torch"
    target = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if target.type != "cuda":
        return "torch"
    return "triton" if triton_masked_mean_available() else "torch"


def _resolve_attr_path(obj: Any, path: str) -> Any:
    value = obj
    for component in path.split("."):
        if not hasattr(value, component):
            return None
        value = getattr(value, component)
    return value


def _is_block_sequence(value: Any) -> bool:
    return (
        isinstance(value, (nn.ModuleList, nn.Sequential, list, tuple))
        and bool(value)
        and all(isinstance(module, nn.Module) for module in value)
    )


def find_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Locate decoder blocks for Qwen and common Hugging Face architectures."""

    candidates = (
        "model.layers",  # Qwen, Llama, Mistral, Gemma
        "model.decoder.layers",  # BART-family decoder
        "transformer.h",  # GPT-2, GPT-J
        "transformer.blocks",
        "gpt_neox.layers",
        "decoder.layers",
        "layers",
        "blocks",
    )
    for path in candidates:
        value = _resolve_attr_path(model, path)
        if _is_block_sequence(value):
            return list(value)

    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
        value = getattr(decoder, "layers", None)
        if _is_block_sequence(value):
            return list(value)

    # Wrapped/PEFT models can add another prefix.  Prefer the largest module
    # list whose name denotes decoder layers rather than guessing one element.
    discovered: list[tuple[int, str, list[nn.Module]]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 0:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"layers", "h", "blocks"} and all(
                isinstance(item, nn.Module) for item in module
            ):
                discovered.append((len(module), name, list(module)))
    if discovered:
        _, name, blocks = max(discovered, key=lambda item: item[0])
        LOGGER.debug("resolved transformer blocks from %s", name)
        return blocks

    raise ValueError(
        "could not locate transformer blocks; pass blocks= explicitly for this architecture"
    )


def _first_hidden_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("last_hidden_state", "hidden_states", "hidden_state"):
            value = output.get(key)
            if isinstance(value, Tensor):
                return value
        for value in output.values():
            try:
                return _first_hidden_tensor(value)
            except (TypeError, ValueError):
                pass
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return _first_hidden_tensor(value)
            except (TypeError, ValueError):
                pass
    raise TypeError("transformer block output does not contain a hidden-state tensor")


def _model_hidden_size(model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


class ResidualStreamModel:
    """A generic decoder-only model wrapper for reduced layer capture."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None,
        *,
        blocks: Sequence[nn.Module] | None = None,
        use_kernels: bool = True,
        output_device: torch.device | str = "cpu",
        default_layer: int | None = None,
        effective_attention_backend: str | None = None,
        effective_masked_mean_backend: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.blocks = list(blocks) if blocks is not None else find_transformer_blocks(model)
        if not self.blocks:
            raise ValueError("at least one transformer block is required")
        self.use_kernels = bool(use_kernels)
        self.output_device = torch.device(output_device)
        model_config = getattr(model, "config", None)
        self.effective_attention_backend = str(
            effective_attention_backend
            or getattr(model_config, "_emotion_vectors_attention_backend", None)
            or getattr(model_config, "_attn_implementation", None)
            or "unknown"
        )
        self.effective_masked_mean_backend = str(
            effective_masked_mean_backend
            or resolve_masked_mean_backend(self.use_kernels, device=self.input_device)
        )
        self.default_layer = (
            min(self.n_layers - 1, max(0, int(default_layer)))
            if default_layer is not None
            else min(self.n_layers - 1, round((2.0 / 3.0) * self.n_layers))
        )
        self.model.eval()

        if tokenizer is not None and hasattr(tokenizer, "padding_side"):
            # Ordinal masking is padding-side agnostic, but right padding also
            # avoids surprising positional-id behavior in simple model stubs.
            tokenizer.padding_side = "right"

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def layers(self) -> list[nn.Module]:
        """Transformer blocks under the CLI-facing bundle name."""

        return self.blocks

    @property
    def hidden_size(self) -> int | None:
        return _model_hidden_size(self.model)

    @property
    def input_device(self) -> torch.device:
        get_embeddings = getattr(self.model, "get_input_embeddings", None)
        if callable(get_embeddings):
            embeddings = get_embeddings()
            weight = getattr(embeddings, "weight", None)
            if isinstance(weight, Tensor) and weight.device.type != "meta":
                return weight.device
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cpu")

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Tokenize without padding and return model-input lengths."""

        if self.tokenizer is None:
            raise RuntimeError("token_lengths requires a tokenizer")
        if not texts:
            return []
        encoded = self.tokenizer(
            list(texts),
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        if isinstance(encoded, Mapping):
            masks = encoded.get("attention_mask")
            ids = encoded.get("input_ids")
        else:
            masks = getattr(encoded, "attention_mask", None)
            ids = getattr(encoded, "input_ids", None)
        if masks is not None:
            if isinstance(masks, Tensor):
                if masks.ndim == 1 and len(texts) == 1:
                    return [int(masks.sum().item())]
                return [int(row.sum().item()) for row in masks]
            return [int(sum(row)) for row in masks]
        if ids is None:
            raise TypeError("tokenizer output does not contain input_ids")
        if isinstance(ids, Tensor):
            if ids.ndim == 1 and len(texts) == 1:
                return [int(ids.numel())]
            return [int(row.numel()) for row in ids]
        if len(texts) == 1 and ids and isinstance(ids[0], int):
            return [len(ids)]
        return [len(row) for row in ids]

    def tokenize(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Pad a batch and move only model inputs to the embedding device."""

        if self.tokenizer is None:
            raise RuntimeError("tokenize requires a tokenizer")
        if not texts:
            raise ValueError("cannot tokenize an empty batch")
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded.input_ids
            attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        device = self.input_device
        return input_ids.to(device), attention_mask.to(device)

    def _forward_kwargs(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        forward = getattr(self.model, "forward", None)
        if forward is None:
            return kwargs
        try:
            signature = inspect.signature(forward)
        except (TypeError, ValueError):
            return kwargs
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs or "use_cache" in signature.parameters:
            kwargs["use_cache"] = False
        if accepts_kwargs or "output_hidden_states" in signature.parameters:
            kwargs["output_hidden_states"] = False
        return kwargs

    def capture_batch(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        token_start: int | Tensor | Sequence[int] = 0,
        collect_norms: bool = False,
    ) -> CaptureBatch:
        """Run one forward and capture the reduced output of every block."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        model_device = self.input_device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        batch_size = input_ids.shape[0]
        starts = torch.as_tensor(token_start, dtype=torch.int64, device=model_device)
        if starts.ndim == 0:
            starts = starts.expand(batch_size)
        if starts.shape != (batch_size,):
            raise ValueError("token_start must be scalar or have one value per batch row")
        if bool(torch.any(starts < 0)):
            raise ValueError("token_start values must be non-negative")

        reduced: list[Tensor | None] = [None] * self.n_layers
        norm_means: list[Tensor | None] | None = [None] * self.n_layers if collect_norms else None
        handles: list[Any] = []

        def make_hook(layer_index: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                hidden = _first_hidden_tensor(output)
                if hidden.ndim != 3:
                    raise ValueError(
                        "decoder block hidden state must have shape "
                        f"[batch, sequence, hidden], got {tuple(hidden.shape)}"
                    )
                layer_mask = attention_mask.to(hidden.device)
                layer_starts = starts.to(hidden.device)
                layer_mean = masked_mean(
                    hidden,
                    layer_mask,
                    layer_starts,
                    use_kernel=self.use_kernels,
                )
                reduced[layer_index] = layer_mean.detach().to(
                    self.output_device, dtype=torch.float32
                )

                if norm_means is not None:
                    selected = eligible_token_mask(layer_mask, layer_starts)
                    counts = selected.sum(dim=1)
                    # ``dtype=`` accumulates in fp32 without first allocating a
                    # full fp32 copy of ``[batch, sequence, hidden]``.
                    token_norms = torch.linalg.vector_norm(
                        hidden.detach(), dim=-1, dtype=torch.float32
                    )
                    totals = (token_norms * selected).sum(dim=1)
                    means = totals / counts.clamp_min(1).to(torch.float32)
                    means = torch.where(counts > 0, means, torch.zeros_like(means))
                    norm_means[layer_index] = means.to(self.output_device, dtype=torch.float32)

            return hook

        try:
            handles = [
                block.register_forward_hook(make_hook(index))
                for index, block in enumerate(self.blocks)
            ]
            with torch.inference_mode():
                self.model(**self._forward_kwargs(input_ids, attention_mask))
        finally:
            for handle in handles:
                handle.remove()

        missing = [index for index, value in enumerate(reduced) if value is None]
        if missing:
            raise RuntimeError(f"model forward did not execute transformer blocks {missing}")
        vectors = torch.stack([value for value in reduced if value is not None], dim=1)
        counts = eligible_token_counts(attention_mask, starts).to(self.output_device)

        stacked_norms: Tensor | None = None
        if norm_means is not None:
            missing_norms = [index for index, value in enumerate(norm_means) if value is None]
            if missing_norms:
                raise RuntimeError(f"missing token norms for layers {missing_norms}")
            stacked_norms = torch.stack([value for value in norm_means if value is not None], dim=1)
        return CaptureBatch(
            vectors=vectors,
            token_counts=counts,
            mean_token_norms=stacked_norms,
        )

    def capture_reduced(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        token_start: int | Tensor | Sequence[int] = 0,
    ) -> Tensor:
        """Convenience API returning only ``[batch, layers, hidden]``."""

        return self.capture_batch(input_ids, attention_mask, token_start=token_start).vectors

    def encode_and_capture(
        self,
        texts: Sequence[str],
        *,
        token_start: int | Tensor | Sequence[int] = 0,
        collect_norms: bool = False,
    ) -> CaptureBatch:
        """Tokenize texts and capture all layers in one model call."""

        input_ids, attention_mask = self.tokenize(texts)
        return self.capture_batch(
            input_ids,
            attention_mask,
            token_start=token_start,
            collect_norms=collect_norms,
        )

    def capture_token_activations(
        self,
        text: str,
        *,
        layer: int | None = None,
    ) -> tuple[list[str], Tensor]:
        """Capture unpadded per-token activations for one verification input.

        Unlike Stage 2, V1/V2 need token resolution.  Only the requested layer
        (the configured primary layer by default) is retained, so this helper
        does not materialise every layer's token tensor.
        """

        return self.capture_token_activations_batch([text], layer=layer)[0]

    def capture_token_activations_batch(
        self,
        texts: Sequence[str],
        *,
        layer: int | None = None,
        max_length: int | None = None,
    ) -> list[tuple[list[str], Tensor]]:
        """Capture one selected layer for a padded batch of verification texts."""

        if self.tokenizer is None:
            raise RuntimeError("token activation capture requires a tokenizer")
        if not texts:
            return []
        selected_layer = self.default_layer if layer is None else int(layer)
        if not 0 <= selected_layer < self.n_layers:
            raise IndexError(f"layer {selected_layer} is outside [0, {self.n_layers})")
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(self.input_device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            self.input_device
        )
        captured: Tensor | None = None

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            nonlocal captured
            hidden = _first_hidden_tensor(output)
            captured = hidden.detach().to(self.output_device, dtype=torch.float32)

        handle = self.blocks[selected_layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self.model(**self._forward_kwargs(input_ids, attention_mask))
        finally:
            handle.remove()
        if captured is None:
            raise RuntimeError(f"model forward did not execute transformer block {selected_layer}")

        converter = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        decoder = getattr(self.tokenizer, "decode", None)
        results: list[tuple[list[str], Tensor]] = []
        for row in range(len(texts)):
            valid_device = attention_mask[row].to(dtype=torch.bool)
            valid_output = valid_device.to(self.output_device)
            activations = captured[row, valid_output]
            token_ids = input_ids[row, valid_device].detach().cpu().tolist()
            if callable(converter):
                tokens = [str(token) for token in converter(token_ids)]
            else:
                tokens = (
                    [str(decoder([token_id])) for token_id in token_ids]
                    if callable(decoder)
                    else [str(token_id) for token_id in token_ids]
                )
            results.append((tokens, activations))
        return results

    # Verification modules probe these conventional names in order.
    token_activations = capture_token_activations
    get_token_activations = capture_token_activations

    def capture_token_projections_batch(
        self,
        texts: Sequence[str],
        vectors: Tensor | Any,
        *,
        layer: int | None = None,
        max_length: int | None = None,
    ) -> list[tuple[list[str], Tensor]]:
        """Project a selected layer on-device and retain no raw V2 activations."""

        if self.tokenizer is None:
            raise RuntimeError("token projection capture requires a tokenizer")
        if not texts:
            return []
        selected_layer = self.default_layer if layer is None else int(layer)
        if not 0 <= selected_layer < self.n_layers:
            raise IndexError(f"layer {selected_layer} is outside [0, {self.n_layers})")
        directions = torch.as_tensor(vectors, dtype=torch.float32)
        if directions.ndim != 2 or directions.shape[1] != self.hidden_size:
            raise ValueError(
                f"vectors must have shape [emotion, {self.hidden_size}], got {tuple(directions.shape)}"
            )
        norms = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        if bool(torch.any(~torch.isfinite(norms))) or bool(torch.any(norms <= 0)):
            raise ValueError("projection vectors must have finite non-zero norms")
        directions = directions / norms

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(self.input_device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            self.input_device
        )
        captured: Tensor | None = None

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            nonlocal captured
            hidden = _first_hidden_tensor(output)
            layer_directions = directions.to(hidden.device)
            captured = (
                torch.matmul(hidden.float(), layer_directions.T).detach().to(self.output_device)
            )

        handle = self.blocks[selected_layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self.model(**self._forward_kwargs(input_ids, attention_mask))
        finally:
            handle.remove()
        if captured is None:
            raise RuntimeError(f"model forward did not execute transformer block {selected_layer}")

        converter = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        decoder = getattr(self.tokenizer, "decode", None)
        results: list[tuple[list[str], Tensor]] = []
        for row in range(len(texts)):
            valid_device = attention_mask[row].to(dtype=torch.bool)
            valid_output = valid_device.to(self.output_device)
            projections = captured[row, valid_output]
            token_ids = input_ids[row, valid_device].detach().cpu().tolist()
            if callable(converter):
                tokens = [str(token) for token in converter(token_ids)]
            else:
                tokens = (
                    [str(decoder([token_id])) for token_id in token_ids]
                    if callable(decoder)
                    else [str(token_id) for token_id in token_ids]
                )
            results.append((tokens, projections))
        return results


# Descriptive aliases for the CLI and extraction code.
ResidualStreamExtractor = ResidualStreamModel
ModelBundle = ResidualStreamModel


def load_model_and_tokenizer(
    config: Any,
    model_name: str | None = None,
    revision: str | None = None,
) -> tuple[nn.Module, Any]:
    """Load a configured Hugging Face causal LM and tokenizer lazily."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # keeps CPU-only unit tests importable.
        raise RuntimeError(
            "transformers is required to load the configured Hugging Face model"
        ) from exc

    resolved_model_name = str(model_name or config.model_name)
    resolved_revision = str(
        revision
        or (
            getattr(config, "generator_revision", None)
            if model_name is not None
            and resolved_model_name == str(getattr(config, "generator", ""))
            else getattr(config, "model_revision", None)
        )
        or "main"
    )
    common_kwargs = {
        "revision": resolved_revision,
        "local_files_only": bool(getattr(config, "local_files_only", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, **common_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    requested_attention_backend = str(config.attn_implementation)
    attention_backend = resolve_attention_backend(requested_attention_backend)
    if attention_backend != requested_attention_backend:
        LOGGER.warning("flash-attn is unavailable; falling back to SDPA")

    kwargs: dict[str, Any] = {
        "dtype": resolve_torch_dtype(config.dtype),
        "low_cpu_mem_usage": True,
        "attn_implementation": attention_backend,
    }
    if torch.cuda.is_available() and int(config.num_gpus) > 0:
        requested_gpus = int(config.num_gpus)
        available_gpus = torch.cuda.device_count()
        if requested_gpus > available_gpus:
            raise RuntimeError(
                f"num_gpus={requested_gpus} was requested, but only "
                f"{available_gpus} CUDA device(s) are visible"
            )
        if requested_gpus == 1:
            # Qwen2.5-32B bf16 fits on one H200. An explicit map prevents
            # Accelerate's ``auto`` policy from silently consuming every
            # visible GPU despite num_gpus=1.
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {
                index: int(torch.cuda.mem_get_info(index)[0] * 0.9)
                for index in range(requested_gpus)
            }
    kwargs.update(common_kwargs)

    model = AutoModelForCausalLM.from_pretrained(resolved_model_name, **kwargs)
    model.config._emotion_vectors_attention_backend = attention_backend
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def load_model(
    config: Any,
    model_name: str | None = None,
    revision: str | None = None,
) -> ModelBundle:
    """Load and wrap the configured model for residual-stream extraction."""

    model, tokenizer = load_model_and_tokenizer(config, model_name=model_name, revision=revision)
    blocks = find_transformer_blocks(model)
    effective_attention_backend = str(
        getattr(
            model.config,
            "_emotion_vectors_attention_backend",
            resolve_attention_backend(str(config.attn_implementation)),
        )
    )
    return ResidualStreamModel(
        model,
        tokenizer,
        blocks=blocks,
        use_kernels=bool(config.use_kernels),
        output_device="cpu",
        default_layer=round(float(config.primary_layer_frac) * len(blocks)),
        effective_attention_backend=effective_attention_backend,
    )


def capture_reduced_activations(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    *,
    token_start: int | Tensor | Sequence[int] = 0,
    blocks: Sequence[nn.Module] | None = None,
    use_kernels: bool = True,
) -> Tensor:
    """Functional one-forward capture API, useful for tests and toy models."""

    wrapper = ResidualStreamModel(
        model,
        tokenizer=None,
        blocks=blocks,
        use_kernels=use_kernels,
        output_device="cpu",
    )
    return wrapper.capture_reduced(input_ids, attention_mask, token_start=token_start)


def transformer_layers(model: nn.Module) -> list[nn.Module]:
    """CLI-facing alias for architecture-independent block discovery."""

    return find_transformer_blocks(model)


__all__ = [
    "CaptureBatch",
    "ModelBundle",
    "ResidualStreamExtractor",
    "ResidualStreamModel",
    "capture_reduced_activations",
    "find_transformer_blocks",
    "load_model",
    "load_model_and_tokenizer",
    "resolve_attention_backend",
    "resolve_masked_mean_backend",
    "resolve_torch_dtype",
    "transformer_layers",
]
