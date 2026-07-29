"""Batched local Hugging Face story generation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationParameters:
    """Sampling settings recorded with every accepted story."""

    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float | None = None
    do_sample: bool = True
    max_new_tokens: int = 512

    def as_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.top_k is not None:
            values["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            values["repetition_penalty"] = self.repetition_penalty
        return values


def resolve_torch_dtype(name: str) -> torch.dtype | str:
    """Resolve a CLI dtype name accepted by Hugging Face model loading."""

    normalized = name.strip().lower().replace("torch.", "")
    aliases: dict[str, torch.dtype | str] = {
        "auto": "auto",
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
        return aliases[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported dtype {name!r}; choose one of: {choices}") from error


class LocalHuggingFaceGenerator:
    """Generate fresh continuations with independent per-story RNG streams."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dtype: str,
        model_revision: str | None = None,
        generation_request: str = "Generate the story now.",
        prompt_role: str = "system",
        chat_system_instruction: str | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        use_model_eos_tokens: bool = False,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.generation.logits_process import (
                LogitsProcessorList,
                TemperatureLogitsWarper,
                TopKLogitsWarper,
                TopPLogitsWarper,
            )
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for generation; install the runtime requirements"
            ) from error

        self.model_name = model_name
        self.device = device
        self.model_revision = model_revision
        self.generation_request = generation_request
        if prompt_role not in {"system", "user"}:
            raise ValueError("prompt_role must be either 'system' or 'user'")
        self.prompt_role = prompt_role
        self.chat_system_instruction = chat_system_instruction
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": resolve_torch_dtype(dtype),
            "low_cpu_mem_usage": True,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs["trust_remote_code"] = trust_remote_code
        model_kwargs["local_files_only"] = local_files_only

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            **model_kwargs,
        )
        if device != "auto":
            model.to(torch.device(device))
        model.eval()
        model.requires_grad_(False)

        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        if use_model_eos_tokens:
            model_generation_config = getattr(model, "generation_config", None)
            configured_eos = getattr(
                model_generation_config,
                "eos_token_id",
                None,
            )
            if configured_eos is not None:
                self.eos_token_id = configured_eos
        self._logits_processor_list = LogitsProcessorList
        self._temperature_warper = TemperatureLogitsWarper
        self._top_k_warper = TopKLogitsWarper
        self._top_p_warper = TopPLogitsWarper

    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        generation_parameters: GenerationParameters,
    ) -> str:
        """Return one decoded continuation using the batched generation path."""

        return self.generate_batch(
            prompts=[prompt],
            seeds=[seed],
            generation_parameters=generation_parameters,
        )[0]

    def generate_batch(
        self,
        *,
        prompts: Sequence[str],
        seeds: Sequence[int],
        generation_parameters: GenerationParameters,
    ) -> list[str]:
        """Return one independently sampled continuation for each prompt and seed."""

        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have the same length")
        if not prompts:
            return []

        formatted_inputs = [self._format_model_input(prompt) for prompt in prompts]
        template_flags = {used_chat_template for _, used_chat_template in formatted_inputs}
        if len(template_flags) != 1:
            raise ValueError("All prompts in a batch must use the same tokenization mode")
        used_chat_template = template_flags.pop()
        encoded = self.tokenizer(
            [model_input for model_input, _ in formatted_inputs],
            return_tensors="pt",
            padding=True,
            add_special_tokens=not used_chat_template,
        )
        prompt_width = int(encoded["input_ids"].shape[-1])
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            left_padding = [0] * len(prompts)
        else:
            left_padding = [
                prompt_width - int(valid_tokens)
                for valid_tokens in attention_mask.sum(dim=1).tolist()
            ]

        input_device = _model_input_device(self.model)
        encoded = {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

        model_generation_config = getattr(self.model, "generation_config", None)
        top_k = generation_parameters.top_k
        if top_k is None:
            top_k = int(getattr(model_generation_config, "top_k", 0) or 0)
        repetition_penalty = generation_parameters.repetition_penalty
        if repetition_penalty is None:
            repetition_penalty = float(
                getattr(model_generation_config, "repetition_penalty", 1.0) or 1.0
            )

        warpers: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = []
        if generation_parameters.do_sample:
            if generation_parameters.temperature != 1.0:
                warpers.append(
                    self._temperature_warper(generation_parameters.temperature)
                )
            if top_k > 0:
                warpers.append(
                    self._top_k_warper(
                        top_k=top_k,
                        min_tokens_to_keep=1,
                    )
                )
            if generation_parameters.top_p < 1.0:
                warpers.append(
                    self._top_p_warper(
                        top_p=generation_parameters.top_p,
                        min_tokens_to_keep=1,
                    )
                )

        generators = [
            torch.Generator(device=input_device).manual_seed(int(seed)) for seed in seeds
        ]
        seeded_processor = _PerRowSeededLogitsProcessor(
            generators=generators,
            left_padding=left_padding,
            repetition_penalty=repetition_penalty,
            warpers=warpers,
            do_sample=generation_parameters.do_sample,
        )

        generation_kwargs: dict[str, Any] = {
            # The custom processor performs sampling and forces its chosen token;
            # greedy selection prevents a second, process-global RNG draw.
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "return_dict_in_generate": False,
            "output_scores": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "max_new_tokens": generation_parameters.max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.eos_token_id,
            # These are reproduced by the padding-aware processor above.
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "logits_processor": self._logits_processor_list([seeded_processor]),
        }

        with torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_kwargs)
        continuations = generated[:, prompt_width:]
        return [
            str(completion)
            for completion in self.tokenizer.batch_decode(
                continuations,
                skip_special_tokens=True,
            )
        ]

    def _format_model_input(self, prompt: str) -> tuple[str, bool]:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if chat_template and callable(apply_chat_template):
            if self.prompt_role == "system":
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": self.generation_request},
                ]
            else:
                messages = []
                if self.chat_system_instruction:
                    messages.append(
                        {
                            "role": "system",
                            "content": self.chat_system_instruction,
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n{self.generation_request}",
                    }
                )
            formatted = str(
                apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            return formatted, True
        if self.prompt_role == "user":
            return f"{prompt}\n\n{self.generation_request}", False
        return prompt, False

    def clear_device_cache(self) -> None:
        """Release unused CUDA allocator blocks after a recoverable batch failure."""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _PerRowSeededLogitsProcessor:
    """Apply sampling controls with one independent generator per batch row."""

    def __init__(
        self,
        *,
        generators: Sequence[torch.Generator],
        left_padding: Sequence[int],
        repetition_penalty: float,
        warpers: Sequence[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        do_sample: bool,
    ) -> None:
        if len(generators) != len(left_padding):
            raise ValueError("generators and left_padding must have the same length")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero")
        self.generators = tuple(generators)
        self.left_padding = tuple(int(padding) for padding in left_padding)
        self.repetition_penalty = repetition_penalty
        self.warpers = tuple(warpers)
        self.do_sample = do_sample

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[0] != len(self.generators):
            raise ValueError("Logit batch size does not match the number of generators")

        processed_scores = scores
        if self.repetition_penalty != 1.0:
            processed_scores = scores.clone()
            for row_index, padding in enumerate(self.left_padding):
                history = input_ids[row_index, padding:]
                row_scores = processed_scores[row_index]
                history_scores = row_scores.gather(0, history)
                penalized_scores = torch.where(
                    history_scores < 0,
                    history_scores * self.repetition_penalty,
                    history_scores / self.repetition_penalty,
                )
                row_scores.scatter_(0, history, penalized_scores)

        if not self.do_sample:
            return processed_scores

        for warper in self.warpers:
            processed_scores = warper(input_ids, processed_scores)
        probabilities = torch.nn.functional.softmax(processed_scores, dim=-1)
        selected_tokens = torch.cat(
            [
                torch.multinomial(
                    probabilities[row_index : row_index + 1],
                    num_samples=1,
                    generator=generator,
                )
                for row_index, generator in enumerate(self.generators)
            ],
            dim=0,
        )
        forced_scores = torch.full_like(processed_scores, -torch.inf)
        return forced_scores.scatter_(1, selected_tokens, 0.0)


def _model_input_device(model: Any) -> torch.device:
    """Find the input device for regular and Accelerate-dispatched models."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        weight = getattr(get_embeddings(), "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for mapped_device in device_map.values():
            if mapped_device not in {"cpu", "disk"}:
                if isinstance(mapped_device, int):
                    return torch.device("cuda", mapped_device)
                return torch.device(mapped_device)

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def contains_exact_emotion_word(generated_text: str, emotion: str) -> bool:
    """Check only the case-insensitive whole target word, without expansion."""

    pattern = rf"\b{re.escape(emotion)}\b"
    return bool(re.search(pattern, generated_text, flags=re.IGNORECASE))


def contains_non_latin_letter(generated_text: str) -> bool:
    """Detect letters from non-Latin scripts while allowing normal punctuation."""

    for character in generated_text:
        if not unicodedata.category(character).startswith("L"):
            continue
        codepoint = ord(character)
        if 0x2100 <= codepoint <= 0x214F:
            # Letterlike Symbols includes mathematical forms such as ℏ and ℝ.
            continue
        if 0x1D400 <= codepoint <= 0x1D7FF:
            # Mathematical Alphanumeric Symbols includes forms such as 𝑥.
            continue
        if "LATIN" not in unicodedata.name(character, ""):
            return True
    return False


def clean_generation(raw_completion: str) -> str:
    """Apply only the minimal output cleanup allowed by the data contract."""

    story = raw_completion.strip()
    lines = story.splitlines()
    has_surrounding_fence = (
        len(lines) >= 2
        and lines[0].strip().startswith("```")
        and lines[-1].strip() == "```"
    )
    if has_surrounding_fence:
        story = "\n".join(lines[1:-1]).strip()

    prefix = re.compile(r"^(?:Story\s*:|\[Story(?:\s+\d+)?\]\s*:?)\s*", re.IGNORECASE)
    story = prefix.sub("", story, count=1).strip()
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", story).strip()


__all__ = [
    "GenerationParameters",
    "LocalHuggingFaceGenerator",
    "clean_generation",
    "contains_exact_emotion_word",
    "contains_non_latin_letter",
    "resolve_torch_dtype",
]
