"""Typed, immutable configuration loading for the emotion-vector pipeline.

This is the only module that reads files from the repository's ``config/``
directory.  Pipeline stages receive an :class:`ExperimentConfig` instance and
never need to know where the YAML, topic, emotion, or prompt assets live.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

_SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_CONFIG_DIR = _SOURCE_PROJECT_ROOT / "config"
_BUNDLED_CONFIG_DIR = Path(__file__).resolve().parent / "assets"
_IN_SOURCE_CHECKOUT = (_SOURCE_CONFIG_DIR / "default.yaml").is_file()
_PROJECT_ROOT = _SOURCE_PROJECT_ROOT if _IN_SOURCE_CHECKOUT else Path.cwd().resolve()
_CONFIG_DIR = _SOURCE_CONFIG_DIR if _IN_SOURCE_CHECKOUT else _BUNDLED_CONFIG_DIR
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "default.yaml"

_TOP_LEVEL_KEYS = frozenset(
    {
        "model_name",
        "model_revision",
        "generator",
        "generator_revision",
        "n_stories",
        "generation_batch_size",
        "generation_max_new_tokens",
        "generation_max_attempts",
        "generation_lexical_policy",
        "generation_temperature",
        "generation_top_p",
        "token_start",
        "layer_norm_sample_size",
        "short_story_policy",
        "pca_variance",
        "pca_device",
        "primary_layer_frac",
        "steering_strength",
        "kmeans_k",
        "heldout_datasets",
        "heldout_revisions",
        "activation_percentile",
        "seed",
        "batch_size",
        "dtype",
        "attn_implementation",
        "num_gpus",
        "local_files_only",
        "length_bucketing",
        "resume",
        "heldout_max_docs",
        "heldout_max_tokens",
        "use_kernels",
        "plot_format",
        "dpi",
        "v1_top_fraction",
        "v1_concentration_factor",
        "v1_min_emotion_pass_rate",
        "v1_min_control_percentile",
        "v1_sample_stories_per_emotion",
        "v2_top_documents",
        "v2_min_emotion_pass_rate",
        "v3_top_k",
        "v3_min_related_hit_rate",
        "v5_min_absolute_correlation",
        "v6_min_emotion_pass_rate",
        "v6_min_nonmatching_decrease_rate",
        "v6_require_generation_sanity",
        "v6_max_generated_words",
        "v6_sanity_max_new_tokens",
        "paths",
    }
)
_PATH_KEYS = frozenset({"raw", "activations", "vectors", "outputs"})


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Absolute paths for all pipeline artifacts."""

    raw: Path
    activations: Path
    vectors: Path
    outputs: Path

    def as_dict(self, *, relative_to: Path | None = None) -> dict[str, str]:
        """Return a JSON-serializable path mapping.

        Paths below ``relative_to`` are represented relative to it.  This keeps
        configuration hashes stable when the checkout is moved.
        """

        return {
            name: _display_path(getattr(self, name), relative_to)
            for name in ("raw", "activations", "vectors", "outputs")
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete immutable configuration for one pipeline run."""

    model_name: str
    model_revision: str
    generator: str
    generator_revision: str
    n_stories: int
    generation_batch_size: int
    generation_max_new_tokens: int
    generation_max_attempts: int
    generation_lexical_policy: str
    generation_temperature: float
    generation_top_p: float
    token_start: int
    layer_norm_sample_size: int
    short_story_policy: str
    pca_variance: float
    pca_device: str
    primary_layer_frac: float
    steering_strength: float
    kmeans_k: int
    heldout_datasets: tuple[str, ...]
    heldout_revisions: tuple[str, ...]
    activation_percentile: float
    seed: int
    batch_size: int
    dtype: str
    attn_implementation: str
    num_gpus: int
    local_files_only: bool
    length_bucketing: bool
    resume: bool
    heldout_max_docs: int
    heldout_max_tokens: int
    use_kernels: bool
    plot_format: tuple[str, ...]
    dpi: int
    v1_top_fraction: float
    v1_concentration_factor: float
    v1_min_emotion_pass_rate: float
    v1_min_control_percentile: float
    v1_sample_stories_per_emotion: int
    v2_top_documents: int
    v2_min_emotion_pass_rate: float
    v3_top_k: int
    v3_min_related_hit_rate: float
    v5_min_absolute_correlation: float
    v6_min_emotion_pass_rate: float
    v6_min_nonmatching_decrease_rate: float
    v6_require_generation_sanity: bool
    v6_max_generated_words: int
    v6_sanity_max_new_tokens: int
    paths: PathsConfig
    topics: tuple[str, ...]
    emotions: tuple[str, ...]
    emotional_stories_prompt: str
    neutral_dialogues_prompt: str
    probe_he: str
    probe_i: str
    config_path: Path
    project_root: Path

    @property
    def config_hash(self) -> str:
        """SHA-256 of configuration values and centrally loaded assets."""

        return hash_config(self)

    @property
    def story_prompt(self) -> str:
        """Compatibility name for the emotional-story prompt template."""

        return self.emotional_stories_prompt

    @property
    def neutral_prompt(self) -> str:
        """Compatibility name for the neutral-dialogue prompt template."""

        return self.neutral_dialogues_prompt

    @property
    def probe_prompts(self) -> tuple[str, str]:
        """The two V6 steering probe prompts, in specification order."""

        return (self.probe_he, self.probe_i)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to this checkout's root."""

        return resolve_path(path, base_dir=self.project_root)

    def as_dict(self, *, include_assets: bool = True) -> dict[str, Any]:
        """Return a canonical, JSON-serializable configuration snapshot."""

        data: dict[str, Any] = {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "generator": self.generator,
            "generator_revision": self.generator_revision,
            "n_stories": self.n_stories,
            "generation_batch_size": self.generation_batch_size,
            "generation_max_new_tokens": self.generation_max_new_tokens,
            "generation_max_attempts": self.generation_max_attempts,
            "generation_lexical_policy": self.generation_lexical_policy,
            "generation_temperature": self.generation_temperature,
            "generation_top_p": self.generation_top_p,
            "token_start": self.token_start,
            "layer_norm_sample_size": self.layer_norm_sample_size,
            "short_story_policy": self.short_story_policy,
            "pca_variance": self.pca_variance,
            "pca_device": self.pca_device,
            "primary_layer_frac": self.primary_layer_frac,
            "steering_strength": self.steering_strength,
            "kmeans_k": self.kmeans_k,
            "heldout_datasets": list(self.heldout_datasets),
            "heldout_revisions": dict(
                zip(self.heldout_datasets, self.heldout_revisions, strict=True)
            ),
            "activation_percentile": self.activation_percentile,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "num_gpus": self.num_gpus,
            "local_files_only": self.local_files_only,
            "length_bucketing": self.length_bucketing,
            "resume": self.resume,
            "heldout_max_docs": self.heldout_max_docs,
            "heldout_max_tokens": self.heldout_max_tokens,
            "use_kernels": self.use_kernels,
            "plot_format": list(self.plot_format),
            "dpi": self.dpi,
            "v1_top_fraction": self.v1_top_fraction,
            "v1_concentration_factor": self.v1_concentration_factor,
            "v1_min_emotion_pass_rate": self.v1_min_emotion_pass_rate,
            "v1_min_control_percentile": self.v1_min_control_percentile,
            "v1_sample_stories_per_emotion": self.v1_sample_stories_per_emotion,
            "v2_top_documents": self.v2_top_documents,
            "v2_min_emotion_pass_rate": self.v2_min_emotion_pass_rate,
            "v3_top_k": self.v3_top_k,
            "v3_min_related_hit_rate": self.v3_min_related_hit_rate,
            "v5_min_absolute_correlation": self.v5_min_absolute_correlation,
            "v6_min_emotion_pass_rate": self.v6_min_emotion_pass_rate,
            "v6_min_nonmatching_decrease_rate": self.v6_min_nonmatching_decrease_rate,
            "v6_require_generation_sanity": self.v6_require_generation_sanity,
            "v6_max_generated_words": self.v6_max_generated_words,
            "v6_sanity_max_new_tokens": self.v6_sanity_max_new_tokens,
            "paths": self.paths.as_dict(relative_to=self.project_root),
        }
        if include_assets:
            data.update(
                {
                    "topics": list(self.topics),
                    "emotions": list(self.emotions),
                    "prompts": {
                        "emotional_stories": self.emotional_stories_prompt,
                        "neutral_dialogues": self.neutral_dialogues_prompt,
                        "probe_he": self.probe_he,
                        "probe_i": self.probe_i,
                    },
                }
            )
        return data


Config = ExperimentConfig


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Expand and resolve ``path``, relative to the repository by default."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        root = Path(base_dir) if base_dir is not None else _PROJECT_ROOT
        candidate = root / candidate
    return candidate.resolve(strict=False)


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load YAML plus topic, emotion, and prompt assets into a frozen config.

    ``overrides`` accepts either a nested mapping (for example
    ``{"paths": {"outputs": "/tmp/out"}}``) or dotted keys (for example
    ``{"paths.outputs": "/tmp/out"}``). Unknown keys fail early so misspelled
    hyperparameters cannot silently alter an experiment.
    """

    config_path = resolve_path(path or _DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    values = _deep_copy_mapping(loaded)
    if overrides:
        _apply_overrides(values, overrides)
    _validate_keys(values)

    asset_dir = _select_asset_dir(config_path.parent)
    topics = _load_line_asset(asset_dir / "topics.txt", comments=False)
    emotions = _load_line_asset(asset_dir / "emotions.txt", comments=True)
    prompts = _load_prompt_module(asset_dir / "prompts.py")
    path_values = _mapping(values, "paths")

    config = ExperimentConfig(
        model_name=_string(values, "model_name"),
        model_revision=_string(values, "model_revision"),
        generator=_string(values, "generator"),
        generator_revision=_string(values, "generator_revision"),
        n_stories=_positive_int(values, "n_stories"),
        generation_batch_size=_positive_int(values, "generation_batch_size"),
        generation_max_new_tokens=_positive_int(values, "generation_max_new_tokens"),
        generation_max_attempts=_positive_int(values, "generation_max_attempts"),
        generation_lexical_policy=_choice(
            values,
            "generation_lexical_policy",
            {"strict", "warn", "off"},
        ),
        generation_temperature=_bounded_float(
            values, "generation_temperature", lower=0.0, upper=10.0
        ),
        generation_top_p=_bounded_float(
            values, "generation_top_p", lower=0.0, upper=1.0, lower_open=True
        ),
        token_start=_nonnegative_int(values, "token_start"),
        layer_norm_sample_size=_positive_int(values, "layer_norm_sample_size"),
        short_story_policy=_choice(values, "short_story_policy", {"skip", "second_half"}),
        pca_variance=_bounded_float(values, "pca_variance", lower=0.0, upper=1.0, lower_open=True),
        pca_device=_choice(values, "pca_device", {"auto", "cpu", "cuda"}),
        primary_layer_frac=_bounded_float(
            values, "primary_layer_frac", lower=0.0, upper=1.0, lower_open=True
        ),
        steering_strength=_finite_float(values, "steering_strength"),
        kmeans_k=_positive_int(values, "kmeans_k"),
        heldout_datasets=_string_tuple(values, "heldout_datasets"),
        heldout_revisions=_revision_tuple(values),
        activation_percentile=_bounded_float(
            values, "activation_percentile", lower=0.0, upper=100.0
        ),
        seed=_nonnegative_int(values, "seed"),
        batch_size=_positive_int(values, "batch_size"),
        dtype=_string(values, "dtype"),
        attn_implementation=_string(values, "attn_implementation"),
        num_gpus=_positive_int(values, "num_gpus"),
        local_files_only=_boolean(values, "local_files_only"),
        length_bucketing=_boolean(values, "length_bucketing"),
        resume=_boolean(values, "resume"),
        heldout_max_docs=_positive_int(values, "heldout_max_docs"),
        heldout_max_tokens=_positive_int(values, "heldout_max_tokens"),
        use_kernels=_boolean(values, "use_kernels"),
        plot_format=_plot_formats(values, "plot_format"),
        dpi=_positive_int(values, "dpi"),
        v1_top_fraction=_bounded_float(
            values, "v1_top_fraction", lower=0.0, upper=1.0, lower_open=True
        ),
        v1_concentration_factor=_finite_float(values, "v1_concentration_factor"),
        v1_min_emotion_pass_rate=_bounded_float(
            values, "v1_min_emotion_pass_rate", lower=0.0, upper=1.0
        ),
        v1_min_control_percentile=_bounded_float(
            values, "v1_min_control_percentile", lower=0.0, upper=1.0
        ),
        v1_sample_stories_per_emotion=_nonnegative_int(values, "v1_sample_stories_per_emotion"),
        v2_top_documents=_positive_int(values, "v2_top_documents"),
        v2_min_emotion_pass_rate=_bounded_float(
            values, "v2_min_emotion_pass_rate", lower=0.0, upper=1.0
        ),
        v3_top_k=_positive_int(values, "v3_top_k"),
        v3_min_related_hit_rate=_bounded_float(
            values, "v3_min_related_hit_rate", lower=0.0, upper=1.0
        ),
        v5_min_absolute_correlation=_bounded_float(
            values, "v5_min_absolute_correlation", lower=0.0, upper=1.0
        ),
        v6_min_emotion_pass_rate=_bounded_float(
            values, "v6_min_emotion_pass_rate", lower=0.0, upper=1.0
        ),
        v6_min_nonmatching_decrease_rate=_bounded_float(
            values, "v6_min_nonmatching_decrease_rate", lower=0.0, upper=1.0
        ),
        v6_require_generation_sanity=_boolean(values, "v6_require_generation_sanity"),
        v6_max_generated_words=_positive_int(values, "v6_max_generated_words"),
        v6_sanity_max_new_tokens=_positive_int(values, "v6_sanity_max_new_tokens"),
        paths=PathsConfig(
            raw=resolve_path(_path_string(path_values, "raw")),
            activations=resolve_path(_path_string(path_values, "activations")),
            vectors=resolve_path(_path_string(path_values, "vectors")),
            outputs=resolve_path(_path_string(path_values, "outputs")),
        ),
        topics=topics,
        emotions=emotions,
        emotional_stories_prompt=_module_string(prompts, "EMOTIONAL_STORIES_PROMPT"),
        neutral_dialogues_prompt=_module_string(prompts, "NEUTRAL_DIALOGUES_PROMPT"),
        probe_he=_module_string(prompts, "PROBE_HE"),
        probe_i=_module_string(prompts, "PROBE_I"),
        config_path=config_path,
        project_root=_PROJECT_ROOT,
    )
    if config.v1_concentration_factor <= 1.0:
        raise ValueError("Configuration key 'v1_concentration_factor' must be greater than 1")
    _validate_assets(config)
    return config


def hash_config(config: ExperimentConfig) -> str:
    """Return a stable SHA-256 for all experiment inputs."""

    encoded = json.dumps(
        config.as_dict(include_assets=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_asset_dir(candidate: Path) -> Path:
    required = ("topics.txt", "emotions.txt", "prompts.py")
    if all((candidate / name).is_file() for name in required):
        return candidate
    return _CONFIG_DIR


def _load_line_asset(path: Path, *, comments: bool) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        message = f"Required configuration asset does not exist: {path}"
        raise FileNotFoundError(message) from exc
    entries = tuple(
        stripped
        for line in lines
        if (stripped := line.strip()) and not (comments and stripped.startswith("#"))
    )
    if not entries:
        raise ValueError(f"Configuration asset is empty: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"Configuration asset contains duplicate entries: {path}")
    return entries


def _load_prompt_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration asset does not exist: {path}")
    path_hash = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    module_name = f"_emotion_vectors_prompts_{path_hash}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prompt configuration from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_assets(config: ExperimentConfig) -> None:
    if "{n_stories}" not in config.emotional_stories_prompt:
        raise ValueError("Emotional-story prompt is missing the {n_stories} placeholder")
    for placeholder in ("{topic}", "{emotion}"):
        if placeholder not in config.emotional_stories_prompt:
            raise ValueError(f"Emotional-story prompt is missing the {placeholder} placeholder")
    for placeholder in ("{n_stories}", "{topic}"):
        if placeholder not in config.neutral_dialogues_prompt:
            raise ValueError(f"Neutral-dialogue prompt is missing the {placeholder} placeholder")
    if not config.probe_he.endswith("Assistant: He feels"):
        raise ValueError("PROBE_HE is not the specified V6 prompt")
    if not config.probe_i.endswith("Assistant: I feel"):
        raise ValueError("PROBE_I is not the specified V6 prompt")


def _display_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is not None:
        try:
            return path.relative_to(relative_to).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _deep_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"Configuration keys must be strings, got {key!r}")
        result[key] = _deep_copy_mapping(item) if isinstance(item, Mapping) else item
    return result


def _apply_overrides(values: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for raw_key, value in overrides.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"Override keys must be non-empty strings, got {raw_key!r}")
        parts = raw_key.split(".")
        target = values
        for part in parts[:-1]:
            existing = target.get(part)
            if existing is None:
                existing = {}
                target[part] = existing
            if not isinstance(existing, dict):
                raise ValueError(f"Cannot apply nested override beneath non-mapping key: {raw_key}")
            target = existing
        leaf = parts[-1]
        if raw_key == "heldout_revisions" and isinstance(value, Mapping):
            target[leaf] = _deep_copy_mapping(value)
        elif isinstance(value, Mapping) and isinstance(target.get(leaf), dict):
            _merge_mapping(target[leaf], value)
        else:
            target[leaf] = _deep_copy_mapping(value) if isinstance(value, Mapping) else value


def _merge_mapping(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if not isinstance(key, str):
            raise ValueError(f"Override keys must be strings, got {key!r}")
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge_mapping(target[key], value)
        else:
            target[key] = _deep_copy_mapping(value) if isinstance(value, Mapping) else value


def _validate_keys(values: Mapping[str, Any]) -> None:
    missing = sorted(_TOP_LEVEL_KEYS - values.keys())
    unknown = sorted(values.keys() - _TOP_LEVEL_KEYS)
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
    paths = _mapping(values, "paths")
    missing_paths = sorted(_PATH_KEYS - paths.keys())
    unknown_paths = sorted(paths.keys() - _PATH_KEYS)
    if missing_paths:
        raise ValueError(f"Missing required path keys: {', '.join(missing_paths)}")
    if unknown_paths:
        raise ValueError(f"Unknown path keys: {', '.join(unknown_paths)}")


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration key '{key}' must be a mapping")
    return value


def _string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration key '{key}' must be a non-empty string")
    return value


def _path_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Path key '{key}' must be a non-empty path string")
    return str(value)


def _boolean(values: Mapping[str, Any], key: str) -> bool:
    value = values.get(key)
    if type(value) is not bool:
        raise ValueError(f"Configuration key '{key}' must be a boolean")
    return value


def _integer(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if type(value) is not int:
        raise ValueError(f"Configuration key '{key}' must be an integer")
    return value


def _positive_int(values: Mapping[str, Any], key: str) -> int:
    value = _integer(values, key)
    if value <= 0:
        raise ValueError(f"Configuration key '{key}' must be positive")
    return value


def _nonnegative_int(values: Mapping[str, Any], key: str) -> int:
    value = _integer(values, key)
    if value < 0:
        raise ValueError(f"Configuration key '{key}' must be non-negative")
    return value


def _finite_float(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration key '{key}' must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Configuration key '{key}' must be finite")
    return result


def _bounded_float(
    values: Mapping[str, Any],
    key: str,
    *,
    lower: float,
    upper: float,
    lower_open: bool = False,
) -> float:
    value = _finite_float(values, key)
    lower_valid = value > lower if lower_open else value >= lower
    if not lower_valid or value > upper:
        left = "(" if lower_open else "["
        raise ValueError(f"Configuration key '{key}' must be in {left}{lower}, {upper}]")
    return value


def _choice(values: Mapping[str, Any], key: str, choices: set[str]) -> str:
    value = _string(values, key)
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"Configuration key '{key}' must be one of: {expected}")
    return value


def _string_tuple(values: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Configuration key '{key}' must be a non-empty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"Configuration key '{key}' must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"Configuration key '{key}' must not contain duplicates")
    return result


def _revision_tuple(values: Mapping[str, Any]) -> tuple[str, ...]:
    datasets = _string_tuple(values, "heldout_datasets")
    revisions = _mapping(values, "heldout_revisions")
    if set(revisions) != set(datasets):
        raise ValueError("heldout_revisions keys must exactly match heldout_datasets")
    result = tuple(revisions[name] for name in datasets)
    if any(not isinstance(item, str) or len(item.strip()) < 7 for item in result):
        raise ValueError("heldout_revisions values must be revision strings")
    return result


def _plot_formats(values: Mapping[str, Any], key: str) -> tuple[str, ...]:
    formats = tuple(item.lower().lstrip(".") for item in _string_tuple(values, key))
    if any(not item.isalnum() for item in formats):
        raise ValueError(f"Configuration key '{key}' contains an invalid file format")
    return formats


def _module_string(module: ModuleType, name: str) -> str:
    value = getattr(module, name, None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Prompt asset must define a non-empty string named {name}")
    return value


__all__ = [
    "Config",
    "ExperimentConfig",
    "PathsConfig",
    "hash_config",
    "load_config",
    "resolve_path",
]
