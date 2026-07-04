"""V6: test whether norm-relative steering raises emotion-word probability."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..steering import apply_steering, assistant_prefix_mask, scaled_steering_vector
from . import (
    EmotionVectorArtifact,
    VerificationResult,
    _as_numpy,
    _coerce_vector_mapping,
    _config_value,
    _plot_options,
    _resolve_output_dir,
    _save_figure,
    _write_csv,
    _write_manifest_if_supported,
    _write_report,
)

PROBE_HE = "Human: How does he feel?\n\nAssistant: He feels"
PROBE_I = "Human: How do you feel?\n\nAssistant: I feel"
PROBES: Mapping[str, str] = {"he_feels": PROBE_HE, "i_feel": PROBE_I}

DEFAULT_RELATED_WORDS: Mapping[str, tuple[str, ...]] = {
    "excited": ("thrilled",),
    "elated": ("delighted",),
    "ecstatic": ("euphoric",),
    "enthusiastic": ("eager",),
    "joyful": ("happy",),
    "content": ("satisfied",),
    "calm": ("peaceful",),
    "serene": ("tranquil",),
    "grateful": ("thankful",),
    "relaxed": ("at ease",),
    "angry": ("mad",),
    "furious": ("enraged",),
    "terrified": ("afraid",),
    "anxious": ("worried",),
    "panicked": ("frantic",),
    "outraged": ("indignant",),
    "sad": ("unhappy",),
    "depressed": ("hopeless",),
    "gloomy": ("bleak",),
    "lonely": ("alone",),
    "miserable": ("wretched",),
    "bored": ("uninterested",),
    "surprised": ("astonished",),
    "proud": ("accomplished",),
    "hopeful": ("optimistic",),
    "nostalgic": ("wistful",),
    "guilty": ("remorseful",),
    "ashamed": ("embarrassed",),
    "jealous": ("envious",),
    "disgusted": ("repulsed",),
}


@dataclass(frozen=True, slots=True)
class ProbeEvaluation:
    probabilities: Mapping[str, float]
    generated_text: str | None = None


@dataclass(frozen=True, slots=True)
class SteeringProbeResult:
    emotions: tuple[str, ...]
    prompt_names: tuple[str, ...]
    baseline_probabilities: np.ndarray
    steered_probabilities: np.ndarray
    probability_deltas: np.ndarray


ProbabilityEvaluator = Callable[..., Mapping[str, float] | ProbeEvaluation]
GenerationEvaluator = Callable[..., str]


def norm_relative_direction(vector: Any, layer_norm: float, strength: float) -> np.ndarray:
    """Compute ``strength * layer_norm * unit(vector)`` exactly as Stage 3 specifies."""

    import torch

    direction = torch.as_tensor(_as_numpy(vector, dtype=np.float32).reshape(-1))
    scaled = scaled_steering_vector(direction, layer_norm, strength)
    return scaled.detach().cpu().numpy().astype(np.float64)


def compute_probability_deltas(
    baseline_probabilities: Any,
    steered_probabilities: Any,
) -> np.ndarray:
    """Subtract ``[prompt, target]`` baselines from ``[prompt, steer, target]``."""

    baseline = _as_numpy(baseline_probabilities, dtype=np.float64)
    steered = _as_numpy(steered_probabilities, dtype=np.float64)
    if baseline.ndim != 2 or steered.ndim != 3:
        raise ValueError("baseline must be [prompt,target] and steered [prompt,steer,target]")
    if steered.shape[0] != baseline.shape[0] or steered.shape[2] != baseline.shape[1]:
        raise ValueError("baseline and steered probability dimensions do not agree")
    if np.any((baseline < 0.0) | (baseline > 1.0)) or np.any((steered < 0.0) | (steered > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isfinite(baseline).all() or not np.isfinite(steered).all():
        raise ValueError("probabilities contain a non-finite value")
    return steered - baseline[:, None, :]


def _resolve_blocks(model: object) -> Sequence[object]:
    paths = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
    )
    for path in paths:
        value: Any = model
        for part in path:
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__"):
            return value
    raise TypeError("could not locate transformer blocks on model_bundle.model")


def _model_and_tokenizer(model_bundle: object) -> tuple[object, object]:
    model = getattr(model_bundle, "model", model_bundle)
    tokenizer = getattr(model_bundle, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("model_bundle must expose a tokenizer")
    return model, tokenizer


def _causal_logits(output: Any) -> Any:
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    if isinstance(output, Mapping) and output.get("logits") is not None:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("causal language model output does not contain logits")


def evaluate_huggingface_probabilities(
    model_bundle: object,
    prompt: str,
    target_words: Sequence[str],
    *,
    steering_direction: Any | None = None,
    layer: int | None = None,
) -> ProbeEvaluation:
    """Evaluate exact continuation-word probabilities with optional steering.

    Multi-token targets use the product of their teacher-forced conditional token
    probabilities. Steering is applied only to Assistant-side prompt positions,
    never to the appended target tokens used to score the continuation.
    """

    import torch

    model, tokenizer = _model_and_tokenizer(model_bundle)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    if not prompt_ids:
        raise ValueError("probe prompt tokenized to an empty sequence")
    target_ids: list[list[int]] = []
    for word in target_words:
        ids = list(tokenizer.encode(" " + word, add_special_tokens=False))
        if not ids:
            ids = list(tokenizer.encode(word, add_special_tokens=False))
        if not ids:
            raise ValueError(f"tokenizer produced no ids for target word {word!r}")
        target_ids.append(ids)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer has no padding or EOS token")
    lengths = [len(prompt_ids) + len(ids) for ids in target_ids]
    width = max(lengths)
    input_ids = torch.full((len(target_ids), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(target_ids):
        sequence = torch.tensor([*prompt_ids, *ids], dtype=torch.long)
        input_ids[row, : sequence.numel()] = sequence
        attention_mask[row, : sequence.numel()] = 1

    device = getattr(model_bundle, "input_device", None)
    if device is None:
        try:
            device = model.get_input_embeddings().weight.device
        except (AttributeError, TypeError):
            device = next(model.parameters()).device
    encoded = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }

    handle = None
    if steering_direction is not None:
        if layer is None:
            raise ValueError("layer is required when applying a steering direction")
        blocks = _resolve_blocks(model)
        if not 0 <= layer < len(blocks):
            raise IndexError(f"layer {layer} is outside [0, {len(blocks)})")
        direction = torch.as_tensor(_as_numpy(steering_direction, dtype=np.float32))
        prompt_mask = assistant_prefix_mask(tokenizer, prompt)
        if prompt_mask.shape[1] != len(prompt_ids):
            raise ValueError("Assistant mask and prompt tokenization lengths differ")
        steering_mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        steering_mask[:, : len(prompt_ids)] = prompt_mask.to(steering_mask.device).expand(
            len(target_ids), -1
        )

        def steering_hook(_module: object, _inputs: object, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            steered = apply_steering(hidden, direction, steering_mask)
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        handle = blocks[layer].register_forward_hook(steering_hook)

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
            log_probabilities = torch.log_softmax(_causal_logits(output).float(), dim=-1)
        result: dict[str, float] = {}
        prompt_length = len(prompt_ids)
        for row, (word, ids) in enumerate(zip(target_words, target_ids, strict=True)):
            token_log_probability = torch.stack(
                [
                    log_probabilities[row, prompt_length - 1 + offset, int(token_id)]
                    for offset, token_id in enumerate(ids)
                ]
            ).sum()
            result[word] = float(torch.exp(token_log_probability).item())
        return ProbeEvaluation(result)
    finally:
        if handle is not None:
            handle.remove()
        if was_training and hasattr(model, "train"):
            model.train()


def generate_huggingface_steered(
    model_bundle: object,
    prompt: str,
    *,
    steering_direction: Any,
    layer: int,
    max_new_tokens: int = 24,
) -> str:
    """Greedily generate a short sanity continuation with prompt-only steering."""

    import torch

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model, tokenizer = _model_and_tokenizer(model_bundle)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = getattr(model_bundle, "input_device", None)
    if device is None:
        try:
            device = model.get_input_embeddings().weight.device
        except (AttributeError, TypeError):
            device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    blocks = _resolve_blocks(model)
    if not 0 <= layer < len(blocks):
        raise IndexError(f"layer {layer} is outside [0, {len(blocks)})")
    direction = torch.as_tensor(_as_numpy(steering_direction, dtype=np.float32))
    prompt_length = int(encoded["input_ids"].shape[1])
    prompt_mask = assistant_prefix_mask(tokenizer, prompt)
    if prompt_mask.shape != encoded["input_ids"].shape:
        raise ValueError("Assistant mask and prompt tokenization shapes differ")

    def steering_hook(_module: object, _inputs: object, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        # With KV caching, subsequent calls have sequence length one; with a
        # cacheless generator they may contain the full growing sequence. In
        # either case, alter only original Assistant prompt positions.
        if int(hidden.shape[1]) != prompt_length:
            return output
        steered = apply_steering(hidden, direction, prompt_mask)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = blocks[layer].register_forward_hook(steering_hook)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
            )
        continuation = output_ids[0, prompt_length:].detach().cpu().tolist()
        return str(tokenizer.decode(continuation, skip_special_tokens=True)).strip()
    finally:
        handle.remove()
        if was_training and hasattr(model, "train"):
            model.train()


def _prompt_value(
    container: Any,
    prompt_name: str,
    prompt_text: str,
    prompt_index: int,
) -> Any:
    if isinstance(container, Mapping):
        for key in (prompt_name, prompt_text, prompt_index):
            if key in container:
                return container[key]
        raise KeyError(f"no values found for prompt {prompt_name!r}")
    return _as_numpy(container)[prompt_index]


def _probability_arrays(
    baseline: Any,
    steered: Any,
    prompt_names: Sequence[str],
    prompt_texts: Mapping[str, str],
    emotions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(baseline, Mapping) and not isinstance(steered, Mapping):
        base_array = _as_numpy(baseline, dtype=np.float64)
        steered_array = _as_numpy(steered, dtype=np.float64)
        return base_array, steered_array
    base_array = np.empty((len(prompt_names), len(emotions)), dtype=np.float64)
    steered_array = np.empty((len(prompt_names), len(emotions), len(emotions)), dtype=np.float64)
    for prompt_index, prompt_name in enumerate(prompt_names):
        prompt_baseline = _prompt_value(
            baseline,
            prompt_name,
            prompt_texts[prompt_name],
            prompt_index,
        )
        prompt_steered = _prompt_value(
            steered,
            prompt_name,
            prompt_texts[prompt_name],
            prompt_index,
        )
        for target_index, target in enumerate(emotions):
            base_array[prompt_index, target_index] = float(prompt_baseline[target])
        for steer_index, steer in enumerate(emotions):
            steer_values = prompt_steered[steer]
            for target_index, target in enumerate(emotions):
                steered_array[prompt_index, steer_index, target_index] = float(steer_values[target])
    return base_array, steered_array


def _coerce_evaluation(
    output: Mapping[str, float] | ProbeEvaluation,
) -> ProbeEvaluation:
    if isinstance(output, ProbeEvaluation):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("probability evaluator must return a mapping or ProbeEvaluation")
    return ProbeEvaluation({str(key): float(value) for key, value in output.items()})


def _call_evaluator(
    evaluator: ProbabilityEvaluator,
    *,
    prompt: str,
    target_words: Sequence[str],
    steering_emotion: str | None,
    steering_direction: np.ndarray | None,
    layer: int,
    strength: float,
    layer_norm: float,
) -> ProbeEvaluation:
    return _coerce_evaluation(
        evaluator(
            prompt=prompt,
            target_words=target_words,
            steering_emotion=steering_emotion,
            steering_direction=steering_direction,
            layer=layer,
            strength=strength,
            layer_norm=layer_norm,
        )
    )


def _resolve_layer_norm(layer_norms: Any, layer: int) -> float:
    values = _as_numpy(layer_norms, dtype=np.float64)
    if values.ndim == 0:
        return float(values)
    values = values.reshape(-1)
    if not 0 <= layer < values.size:
        raise IndexError(f"layer {layer} is outside layer_norms with length {values.size}")
    return float(values[layer])


def run_steering_probe(
    emotion_vectors: Mapping[str, Any] | np.ndarray | EmotionVectorArtifact | None = None,
    *,
    emotions: Sequence[str] | None = None,
    primary_layer: int | None = None,
    layer_norms: Any | None = None,
    probability_evaluator: ProbabilityEvaluator | None = None,
    generation_evaluator: GenerationEvaluator | None = None,
    model_bundle: object | None = None,
    baseline_probabilities: Any | None = None,
    steered_probabilities: Any | None = None,
    generated_texts: Mapping[str, Mapping[str, str]] | None = None,
    related_words: Mapping[str, Sequence[str]] | None = None,
    story_content_terms: Sequence[str] = (),
    require_generation_sanity: bool | None = None,
    max_generated_words: int | None = None,
    sanity_max_new_tokens: int | None = None,
    strength: float | None = None,
    minimum_emotion_pass_rate: float | None = None,
    minimum_nonmatching_decrease_rate: float | None = None,
    output_dir: str | Path | None = None,
    config: object | None = None,
    plot_formats: Sequence[str] | None = None,
    dpi: int | None = None,
) -> VerificationResult:
    """Run V6 from cached probabilities or evaluate a caller-owned model bundle."""

    vector_map: dict[str, np.ndarray] | None = None
    artifact_layer: int | None = None
    if emotion_vectors is not None:
        if isinstance(emotion_vectors, EmotionVectorArtifact):
            artifact_layer = emotion_vectors.primary_layer
        vector_map = _coerce_vector_mapping(emotion_vectors, emotions)
        labels = tuple(vector_map)
    elif emotions is not None:
        labels = tuple(map(str, emotions))
        if not labels:
            raise ValueError("emotions cannot be empty")
    else:
        raise ValueError("emotion_vectors or emotions is required")
    prompt_texts = {
        "he_feels": str(_config_value(config, "probe_he", PROBE_HE)),
        "i_feel": str(_config_value(config, "probe_i", PROBE_I)),
    }
    prompt_names = tuple(prompt_texts)
    resolved_strength = float(
        strength if strength is not None else _config_value(config, "steering_strength", 0.5)
    )
    resolved_layer = int(
        primary_layer
        if primary_layer is not None
        else artifact_layer if artifact_layer is not None else 0
    )
    destination = _resolve_output_dir(output_dir, config, "V6_steering_probe")
    formats, resolved_dpi = _plot_options(config, plot_formats, dpi)
    resolved_minimum_pass_rate = float(
        minimum_emotion_pass_rate
        if minimum_emotion_pass_rate is not None
        else _config_value(config, "v6_min_emotion_pass_rate", 1.0)
    )
    resolved_nonmatching_rate = float(
        minimum_nonmatching_decrease_rate
        if minimum_nonmatching_decrease_rate is not None
        else _config_value(config, "v6_min_nonmatching_decrease_rate", 1.0)
    )
    resolved_require_sanity = bool(
        require_generation_sanity
        if require_generation_sanity is not None
        else _config_value(config, "v6_require_generation_sanity", True)
    )
    resolved_max_generated_words = int(
        max_generated_words
        if max_generated_words is not None
        else _config_value(config, "v6_max_generated_words", 80)
    )
    resolved_sanity_tokens = int(
        sanity_max_new_tokens
        if sanity_max_new_tokens is not None
        else _config_value(config, "v6_sanity_max_new_tokens", 24)
    )
    if not 0.0 <= resolved_minimum_pass_rate <= 1.0:
        raise ValueError("minimum_emotion_pass_rate must be in [0, 1]")
    if not 0.0 <= resolved_nonmatching_rate <= 1.0:
        raise ValueError("minimum_nonmatching_decrease_rate must be in [0, 1]")
    if resolved_max_generated_words <= 0:
        raise ValueError("max_generated_words must be positive")
    if resolved_sanity_tokens <= 0:
        raise ValueError("sanity_max_new_tokens must be positive")

    baseline_maps: dict[str, Mapping[str, float]] = {}
    steered_maps: dict[str, dict[str, Mapping[str, float]]] = {}
    collected_texts: dict[str, dict[str, str]] = {
        key: dict(value) for key, value in (generated_texts or {}).items()
    }
    if baseline_probabilities is None or steered_probabilities is None:
        if vector_map is None or layer_norms is None:
            raise ValueError("active evaluation requires emotion_vectors and cached layer_norms")
        layer_norm = _resolve_layer_norm(layer_norms, resolved_layer)
        if probability_evaluator is None:
            if model_bundle is None:
                raise ValueError(
                    "provide cached probabilities, probability_evaluator, or model_bundle"
                )
            custom = getattr(model_bundle, "probability_evaluator", None)
            if callable(custom):
                probability_evaluator = custom
            else:

                def hf_evaluator(**kwargs: Any) -> ProbeEvaluation:
                    return evaluate_huggingface_probabilities(
                        model_bundle,
                        kwargs["prompt"],
                        kwargs["target_words"],
                        steering_direction=kwargs["steering_direction"],
                        layer=kwargs["layer"],
                    )

                probability_evaluator = hf_evaluator
        if generation_evaluator is None and model_bundle is not None:

            def hf_generation_evaluator(**kwargs: Any) -> str:
                return generate_huggingface_steered(
                    model_bundle,
                    kwargs["prompt"],
                    steering_direction=kwargs["steering_direction"],
                    layer=kwargs["layer"],
                    max_new_tokens=resolved_sanity_tokens,
                )

            generation_evaluator = hf_generation_evaluator
        related = related_words if related_words is not None else DEFAULT_RELATED_WORDS
        target_words = list(labels)
        for emotion in labels:
            for word in related.get(emotion, ()):
                if word not in target_words:
                    target_words.append(str(word))
        baseline_array = np.empty((len(prompt_names), len(labels)), dtype=np.float64)
        steered_array = np.empty((len(prompt_names), len(labels), len(labels)), dtype=np.float64)
        for prompt_index, prompt_name in enumerate(prompt_names):
            prompt = prompt_texts[prompt_name]
            baseline_evaluation = _call_evaluator(
                probability_evaluator,
                prompt=prompt,
                target_words=target_words,
                steering_emotion=None,
                steering_direction=None,
                layer=resolved_layer,
                strength=resolved_strength,
                layer_norm=layer_norm,
            )
            baseline_maps[prompt_name] = baseline_evaluation.probabilities
            for target_index, target in enumerate(labels):
                baseline_array[prompt_index, target_index] = float(
                    baseline_evaluation.probabilities[target]
                )
            steered_maps[prompt_name] = {}
            for steer_index, emotion in enumerate(labels):
                scaled = norm_relative_direction(vector_map[emotion], layer_norm, resolved_strength)
                evaluation = _call_evaluator(
                    probability_evaluator,
                    prompt=prompt,
                    target_words=target_words,
                    steering_emotion=emotion,
                    steering_direction=scaled,
                    layer=resolved_layer,
                    strength=resolved_strength,
                    layer_norm=layer_norm,
                )
                steered_maps[prompt_name][emotion] = evaluation.probabilities
                if evaluation.generated_text is not None:
                    collected_texts.setdefault(prompt_name, {})[emotion] = evaluation.generated_text
                for target_index, target in enumerate(labels):
                    steered_array[prompt_index, steer_index, target_index] = float(
                        evaluation.probabilities[target]
                    )
                if generation_evaluator is not None:
                    collected_texts.setdefault(prompt_name, {})[emotion] = generation_evaluator(
                        prompt=prompt,
                        steering_emotion=emotion,
                        steering_direction=scaled,
                        layer=resolved_layer,
                    )
    else:
        baseline_array, steered_array = _probability_arrays(
            baseline_probabilities,
            steered_probabilities,
            prompt_names,
            prompt_texts,
            labels,
        )
        for prompt_index, prompt_name in enumerate(prompt_names):
            baseline_maps[prompt_name] = {
                emotion: float(baseline_array[prompt_index, target])
                for target, emotion in enumerate(labels)
            }
            steered_maps[prompt_name] = {
                steer: {
                    target: float(steered_array[prompt_index, steer_index, target_index])
                    for target_index, target in enumerate(labels)
                }
                for steer_index, steer in enumerate(labels)
            }

    deltas = compute_probability_deltas(baseline_array, steered_array)
    probability_rows: list[dict[str, object]] = []
    for prompt_index, prompt_name in enumerate(prompt_names):
        for steer_index, steer in enumerate(labels):
            for target_index, target in enumerate(labels):
                probability_rows.append(
                    {
                        "prompt": prompt_name,
                        "prompt_text": prompt_texts[prompt_name],
                        "steering_emotion": steer,
                        "target_emotion": target,
                        "matching": steer == target,
                        "baseline_probability": float(baseline_array[prompt_index, target_index]),
                        "steered_probability": float(
                            steered_array[prompt_index, steer_index, target_index]
                        ),
                        "delta_probability": float(deltas[prompt_index, steer_index, target_index]),
                    }
                )

    related_rows: list[dict[str, object]] = []
    active_related = related_words if related_words is not None else DEFAULT_RELATED_WORDS
    for prompt_name in prompt_names:
        for emotion in labels:
            for word in active_related.get(emotion, ()):
                if (
                    word in baseline_maps[prompt_name]
                    and word in steered_maps[prompt_name][emotion]
                ):
                    baseline_value = float(baseline_maps[prompt_name][word])
                    steered_value = float(steered_maps[prompt_name][emotion][word])
                    related_rows.append(
                        {
                            "prompt": prompt_name,
                            "steering_emotion": emotion,
                            "related_word": word,
                            "baseline_probability": baseline_value,
                            "steered_probability": steered_value,
                            "delta_probability": steered_value - baseline_value,
                        }
                    )

    emotion_rows: list[dict[str, object]] = []
    for emotion_index, emotion in enumerate(labels):
        he_matching = float(deltas[0, emotion_index, emotion_index])
        i_matching = float(deltas[1, emotion_index, emotion_index])
        if len(labels) > 1:
            nonmatching_mask = np.arange(len(labels)) != emotion_index
            he_nonmatching = float(deltas[0, emotion_index, nonmatching_mask].mean())
            he_nonmatching_decrease_rate = float(
                np.mean(deltas[0, emotion_index, nonmatching_mask] < 0.0)
            )
        else:
            he_nonmatching = float("nan")
            he_nonmatching_decrease_rate = 1.0
        he_match_passed = he_matching > 0.0
        he_nonmatch_passed = bool(
            len(labels) == 1
            or (he_nonmatching < 0.0 and he_nonmatching_decrease_rate >= resolved_nonmatching_rate)
        )
        i_match_passed = i_matching > 0.0
        emotion_rows.append(
            {
                "emotion": emotion,
                "he_matching_delta": he_matching,
                "he_mean_nonmatching_delta": he_nonmatching,
                "he_nonmatching_decrease_rate": he_nonmatching_decrease_rate,
                "i_matching_delta": i_matching,
                "he_matching_increased": he_match_passed,
                "he_nonmatching_decreased": he_nonmatch_passed,
                "i_matching_increased": i_match_passed,
                "passed": he_match_passed and he_nonmatch_passed and i_match_passed,
            }
        )

    all_generated = [
        text for prompt_texts in collected_texts.values() for text in prompt_texts.values()
    ]
    lower_terms = tuple(term.lower() for term in story_content_terms if term.strip())
    hallucination_hits = [
        text
        for text in all_generated
        if any(term in text.lower() for term in lower_terms)
        or len(text.split()) > resolved_max_generated_words
    ]
    sanity_assessed = bool(all_generated)
    sanity_passed = not hallucination_hits and (sanity_assessed or not resolved_require_sanity)
    emotion_pass_rate = float(np.mean([bool(row["passed"]) for row in emotion_rows]))
    behavioral_passed = emotion_pass_rate >= resolved_minimum_pass_rate
    passed = behavioral_passed and sanity_passed

    probability_table = _write_csv(
        destination / "probability_deltas.csv",
        tuple(probability_rows[0]),
        probability_rows,
    )
    emotion_table = _write_csv(
        destination / "emotion_summary.csv", tuple(emotion_rows[0]), emotion_rows
    )
    related_fields = (
        "prompt",
        "steering_emotion",
        "related_word",
        "baseline_probability",
        "steered_probability",
        "delta_probability",
    )
    related_table = _write_csv(
        destination / "related_word_deltas.csv", related_fields, related_rows
    )
    generation_rows = [
        {
            "prompt": prompt,
            "steering_emotion": emotion,
            "generated_text": text,
            "word_count": len(text.split()),
            "story_content_hit": any(term in text.lower() for term in lower_terms),
            "length_violation": len(text.split()) > resolved_max_generated_words,
            "passed": not any(term in text.lower() for term in lower_terms)
            and len(text.split()) <= resolved_max_generated_words,
        }
        for prompt, values in collected_texts.items()
        for emotion, text in values.items()
    ]
    generation_table = _write_csv(
        destination / "generation_sanity.csv",
        (
            "prompt",
            "steering_emotion",
            "generated_text",
            "word_count",
            "story_content_hit",
            "length_violation",
            "passed",
        ),
        generation_rows,
    )
    matching_source = _write_csv(
        destination / "matching_probability_delta.csv",
        tuple(emotion_rows[0]),
        emotion_rows,
    )
    matrix_source = _write_csv(
        destination / "probability_matrix.csv",
        tuple(probability_rows[0]),
        probability_rows,
    )
    tables = (
        probability_table,
        emotion_table,
        related_table,
        generation_table,
        matching_source,
        matrix_source,
    )

    figures: list[Path] = []
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(8.0, 0.45 * len(labels)), 5.0))
    axis.bar(
        positions - width / 2,
        np.diagonal(deltas[0]),
        width,
        label="He feels",
        color="#4c78a8",
    )
    axis.bar(
        positions + width / 2,
        np.diagonal(deltas[1]),
        width,
        label="I feel",
        color="#f28e2b",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=50, ha="right")
    axis.set_ylabel("matching-word Δ probability")
    axis.set_title(f"Norm-relative steering strength s={resolved_strength:g}")
    axis.legend(frameon=False)
    figures.extend(
        _save_figure(
            figure,
            destination / "matching_probability_delta",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(max(12.0, 0.72 * len(labels)), 6.0), squeeze=False)
    maximum = max(float(np.abs(deltas).max()), np.finfo(np.float64).eps)
    for prompt_index, prompt_name in enumerate(prompt_names):
        axis = axes[0, prompt_index]
        image = axis.imshow(
            deltas[prompt_index],
            vmin=-maximum,
            vmax=maximum,
            cmap="coolwarm",
            aspect="auto",
        )
        axis.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xlabel("target emotion word")
        axis.set_ylabel("steering vector")
        axis.set_title(prompt_name)
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Δ probability")
    figure.suptitle("Steering-vector x emotion-word probability")
    figures.extend(
        _save_figure(
            figure,
            destination / "probability_matrix",
            formats=formats,
            dpi=resolved_dpi,
        )
    )
    plt.close(figure)

    sanity_text = (
        f"Generation sanity assessed on {len(all_generated)} samples; "
        f"story-content/length violations: {len(hallucination_hits)} "
        f"(maximum {resolved_max_generated_words} words)."
        if sanity_assessed
        else "Generation sanity was not assessed (no generated samples supplied)."
    )
    report = _write_report(
        destination / "report.md",
        title="V6 — Steering emotion-word probability",
        passed=passed,
        summary=(
            f"Used the two verbatim probes at norm-relative strength s={resolved_strength:g}.",
            f"Emotion behavioral pass rate: {emotion_pass_rate:.3f}; required: "
            f"{resolved_minimum_pass_rate:.3f}.",
            "An emotion passes when its matching word increases on both probes and its mean "
            "non-matching-word probability decreases on the third-person probe.",
            f"{sanity_text} Sanity required: {resolved_require_sanity}; "
            f"passed: {sanity_passed}.",
            f"Related-word deltas reported: {len(related_rows)}.",
        ),
        figures=figures,
        tables=tables,
    )
    _write_manifest_if_supported(
        destination,
        config,
        "V6_steering_probe",
        {
            "emotions": len(labels),
            "generated_samples": len(all_generated),
            "passed": passed,
        },
    )
    return VerificationResult(
        name="V6_steering_probe",
        passed=passed,
        output_dir=destination,
        report=report,
        tables=tables,
        figures=tuple(figures),
        metrics={
            "n_emotions": len(labels),
            "strength": resolved_strength,
            "emotion_pass_rate": emotion_pass_rate,
            "generation_sanity_assessed": sanity_assessed,
            "generation_sanity_passed": sanity_passed,
        },
    )


__all__ = [
    "DEFAULT_RELATED_WORDS",
    "PROBES",
    "PROBE_HE",
    "PROBE_I",
    "ProbeEvaluation",
    "SteeringProbeResult",
    "compute_probability_deltas",
    "evaluate_huggingface_probabilities",
    "generate_huggingface_steered",
    "norm_relative_direction",
    "run_steering_probe",
]
