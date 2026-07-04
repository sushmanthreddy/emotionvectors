"""Pinned model loading for the reduced RQ1 activation experiment.

The aligned activation artifacts were produced with the official Qwen
checkpoint.  The audited Unsloth parent is tensor-identical to that checkpoint
and is the parent declared by the bad-medical-advice adapter.  Consequently the
misaligned pass loads the pinned Unsloth parent, attaches the pinned PEFT
adapter *without merging it*, and deliberately uses the pinned official Qwen
tokenizer for both passes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from emotion_vectors.model import (
    ResidualStreamModel,
    find_transformer_blocks,
    resolve_attention_backend,
    resolve_torch_dtype,
)

ModelSource = Literal["base", "misaligned"]

EXPECTED_ACTIVATION_SITE = "model.model.layers[l].output[0]"
DEFAULT_ATTENTION_BACKEND = "sdpa"
DEFAULT_EXPECTED_LAYERS = 48
DEFAULT_EXPECTED_HIDDEN_SIZE = 5120
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Exact model/tokenizer identity attached to a loaded extractor."""

    model_source: ModelSource
    model_id: str
    model_revision: str
    base_model_id: str
    base_model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    adapter_id: str | None
    adapter_revision: str | None
    adapter_merged: bool
    extraction_dtype: str
    attention_backend: str
    activation_site: str
    n_layers: int
    hidden_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for artifact metadata."""

        return asdict(self)


def _lookup_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                return _MISSING
            current = current[component]
        else:
            if not hasattr(current, component):
                return _MISSING
            current = getattr(current, component)
    return current


def config_value(config: Any, *paths: str, default: Any = _MISSING) -> Any:
    """Read the first present dotted field from dataclasses or mappings.

    The helper keeps this module usable by tiny smoke configurations while the
    strict :class:`RQ1Config` remains the production interface.
    """

    for path in paths:
        value = _lookup_path(config, path)
        if value is not _MISSING:
            return value
    if default is not _MISSING:
        return default
    joined = ", ".join(paths)
    raise ValueError(f"configuration is missing required field (tried: {joined})")


def _hub_cache_dir(config: Any) -> str | None:
    """Translate an HF_HOME root into the cache_dir expected by Hub APIs."""

    configured = config_value(config, "paths.hf_home", "hf_home", default=None)
    if configured is None:
        return None
    root = Path(configured).expanduser()
    return str(root if root.name == "hub" else root / "hub")


def require_immutable_revision(revision: Any, *, field: str) -> str:
    """Require a full immutable Hub commit rather than a moving branch."""

    normalized = str(revision).strip()
    if not _IMMUTABLE_REVISION.fullmatch(normalized):
        raise ValueError(f"{field} must be a full immutable 40-64 character hexadecimal revision")
    return normalized


def _model_contract(config: Any) -> tuple[int, int, str, tuple[int, ...]]:
    layers = tuple(
        int(layer)
        for layer in config_value(
            config,
            "analysis.layers",
            "layers",
            default=tuple(range(DEFAULT_EXPECTED_LAYERS)),
        )
    )
    expected_layers = len(layers)
    if layers != tuple(range(expected_layers)):
        raise ValueError("analysis.layers must contain every layer exactly once in order")
    expected_hidden = int(
        config_value(
            config,
            "analysis.hidden_size",
            "hidden_size",
            default=DEFAULT_EXPECTED_HIDDEN_SIZE,
        )
    )
    activation_site = str(
        config_value(
            config,
            "analysis.activation_site",
            "activation_site",
            default=EXPECTED_ACTIVATION_SITE,
        )
    )
    if activation_site != EXPECTED_ACTIVATION_SITE:
        raise ValueError(
            f"activation site must be {EXPECTED_ACTIVATION_SITE!r}, got {activation_site!r}"
        )
    return expected_layers, expected_hidden, activation_site, layers


def validate_model_contract(
    model: nn.Module,
    config: Any,
    *,
    require_unmerged_adapter: bool = False,
) -> list[nn.Module]:
    """Validate the Qwen residual-stream contract and return decoder blocks."""

    expected_layers, expected_hidden, _site, _indices = _model_contract(config)
    blocks = find_transformer_blocks(model)
    if len(blocks) != expected_layers:
        raise ValueError(f"model has {len(blocks)} residual blocks, expected {expected_layers}")

    model_config = getattr(model, "config", None)
    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model_config, "d_model", None)
    if hidden_size is None or int(hidden_size) != expected_hidden:
        raise ValueError(f"model hidden size is {hidden_size!r}, expected {expected_hidden}")

    declared_layers = getattr(model_config, "num_hidden_layers", None)
    if declared_layers is not None and int(declared_layers) != expected_layers:
        raise ValueError(
            f"model config declares {declared_layers} layers, expected {expected_layers}"
        )
    model_type = getattr(model_config, "model_type", None)
    if model_type is not None and str(model_type) not in {"qwen2", "qwen2_moe"}:
        raise ValueError(f"RQ1 requires a Qwen2-family model, got model_type={model_type!r}")

    if require_unmerged_adapter and not getattr(model, "peft_config", None):
        raise ValueError("misaligned extraction requires an attached, unmerged PEFT adapter")
    return blocks


def _tokenizer(config: Any, auto_tokenizer_cls: Any) -> tuple[Any, str, str, bool]:
    tokenizer_id = str(
        config_value(config, "models.tokenizer_id", "tokenizer_id", "models.base_model_id")
    )
    tokenizer_revision = require_immutable_revision(
        config_value(
            config,
            "models.tokenizer_revision",
            "tokenizer_revision",
            "models.base_model_revision",
        ),
        field="tokenizer_revision",
    )
    local_only = bool(
        config_value(
            config,
            "analysis.local_files_only",
            "local_files_only",
            default=False,
        )
    )
    tokenizer_kwargs: dict[str, Any] = {
        "revision": tokenizer_revision,
        "local_files_only": local_only,
        "use_fast": True,
    }
    cache_dir = _hub_cache_dir(config)
    if cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = str(cache_dir)
    tokenizer = auto_tokenizer_cls.from_pretrained(tokenizer_id, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer, tokenizer_id, tokenizer_revision, local_only


def _load_classes(
    auto_model_cls: Any | None,
    auto_tokenizer_cls: Any | None,
    peft_model_cls: Any | None,
) -> tuple[Any, Any, Any | None]:
    if auto_model_cls is None or auto_tokenizer_cls is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency is in the project env.
            raise RuntimeError("transformers is required for RQ1 model loading") from exc
        auto_model_cls = auto_model_cls or AutoModelForCausalLM
        auto_tokenizer_cls = auto_tokenizer_cls or AutoTokenizer
    if peft_model_cls is None:
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None
        peft_model_cls = PeftModel
    return auto_model_cls, auto_tokenizer_cls, peft_model_cls


def load_rq1_model(
    config: Any,
    model_source: ModelSource,
    *,
    auto_model_cls: Any | None = None,
    auto_tokenizer_cls: Any | None = None,
    peft_model_cls: Any | None = None,
) -> ResidualStreamModel:
    """Load a pinned aligned or misaligned Qwen model for layer-wise capture.

    ``model_source="misaligned"`` always loads the audited adapter parent and
    then calls ``PeftModel.from_pretrained``.  It never calls
    ``merge_and_unload``; the returned model therefore preserves the exact PEFT
    forward path used by the published adapter.
    """

    if model_source not in {"base", "misaligned"}:
        raise ValueError("model_source must be 'base' or 'misaligned'")
    auto_model_cls, auto_tokenizer_cls, peft_model_cls = _load_classes(
        auto_model_cls, auto_tokenizer_cls, peft_model_cls
    )
    tokenizer, tokenizer_id, tokenizer_revision, local_only = _tokenizer(config, auto_tokenizer_cls)

    if model_source == "base":
        base_model_id = str(config_value(config, "models.base_model_id", "base_model_id"))
        base_revision = require_immutable_revision(
            config_value(config, "models.base_model_revision", "base_model_revision"),
            field="base_model_revision",
        )
        model_id = base_model_id
        model_revision = base_revision
        adapter_id = None
        adapter_revision = None
    else:
        base_model_id = str(
            config_value(
                config,
                "models.audited_parent_model_id",
                "audited_parent_model_id",
            )
        )
        base_revision = require_immutable_revision(
            config_value(
                config,
                "models.audited_parent_revision",
                "audited_parent_revision",
            ),
            field="audited_parent_revision",
        )
        adapter_id = str(
            config_value(
                config,
                "models.misaligned_adapter_id",
                "misaligned_adapter_id",
            )
        )
        adapter_revision = require_immutable_revision(
            config_value(
                config,
                "models.misaligned_adapter_revision",
                "misaligned_adapter_revision",
            ),
            field="misaligned_adapter_revision",
        )
        model_id = adapter_id
        model_revision = adapter_revision
        if peft_model_cls is None:
            raise RuntimeError("peft is required to attach the misaligned adapter")

    dtype_name = str(
        config_value(
            config,
            "analysis.extraction_dtype",
            "extraction_dtype",
            "dtype",
            default="bfloat16",
        )
    )
    dtype = resolve_torch_dtype(dtype_name)
    if dtype is not torch.bfloat16:
        raise ValueError("RQ1 extraction requires the model in bfloat16")
    requested_attention = str(
        config_value(
            config,
            "analysis.attention_implementation",
            "attn_implementation",
            default=DEFAULT_ATTENTION_BACKEND,
        )
    )
    attention_backend = resolve_attention_backend(requested_attention)
    if attention_backend != DEFAULT_ATTENTION_BACKEND:
        raise ValueError("RQ1 requires the SDPA attention implementation")

    model_kwargs: dict[str, Any] = {
        "revision": base_revision,
        "local_files_only": local_only,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention_backend,
    }
    cache_dir = _hub_cache_dir(config)
    if cache_dir is not None:
        model_kwargs["cache_dir"] = str(cache_dir)
    configured_device_map = config_value(config, "device_map", default=None)
    if configured_device_map is not None:
        model_kwargs["device_map"] = configured_device_map
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    model = auto_model_cls.from_pretrained(base_model_id, **model_kwargs)
    if model_source == "misaligned":
        assert adapter_id is not None and adapter_revision is not None
        model = peft_model_cls.from_pretrained(
            model,
            adapter_id,
            revision=adapter_revision,
            local_files_only=local_only,
            is_trainable=False,
            adapter_name="default",
            autocast_adapter_dtype=False,
            low_cpu_mem_usage=True,
            **({"cache_dir": str(cache_dir)} if cache_dir is not None else {}),
        )
    # PEFT checkpoints may store adapter tensors in a different floating
    # dtype, and test doubles need not honor ``dtype=`` during construction.
    # The preregistered forward contract is uniformly bfloat16.
    model.to(dtype=dtype)

    model.eval()
    model.requires_grad_(False)
    blocks = validate_model_contract(
        model,
        config,
        require_unmerged_adapter=model_source == "misaligned",
    )
    expected_layers, expected_hidden, activation_site, _indices = _model_contract(config)
    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_config._emotion_vectors_attention_backend = attention_backend

    provenance = ModelProvenance(
        model_source=model_source,
        model_id=model_id,
        model_revision=model_revision,
        base_model_id=base_model_id,
        base_model_revision=base_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        adapter_id=adapter_id,
        adapter_revision=adapter_revision,
        adapter_merged=False,
        extraction_dtype="bfloat16",
        attention_backend=attention_backend,
        activation_site=activation_site,
        n_layers=expected_layers,
        hidden_size=expected_hidden,
    )
    bundle = ResidualStreamModel(
        model,
        tokenizer,
        blocks=blocks,
        use_kernels=bool(config_value(config, "analysis.use_kernels", "use_kernels", default=True)),
        output_device="cpu",
        default_layer=int(
            config_value(config, "analysis.primary_layer", "primary_layer", default=24)
        ),
        effective_attention_backend=attention_backend,
    )
    bundle.rq1_provenance = provenance
    return bundle


def model_provenance(bundle: ResidualStreamModel) -> ModelProvenance:
    """Return validated RQ1 provenance from a loaded extraction bundle."""

    provenance = getattr(bundle, "rq1_provenance", None)
    if not isinstance(provenance, ModelProvenance):
        raise ValueError("model bundle does not carry RQ1 provenance")
    return provenance


__all__ = [
    "DEFAULT_EXPECTED_HIDDEN_SIZE",
    "DEFAULT_EXPECTED_LAYERS",
    "EXPECTED_ACTIVATION_SITE",
    "ModelProvenance",
    "ModelSource",
    "config_value",
    "load_rq1_model",
    "model_provenance",
    "require_immutable_revision",
    "validate_model_contract",
]
