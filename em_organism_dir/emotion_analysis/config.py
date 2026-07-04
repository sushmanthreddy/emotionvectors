"""Frozen, strictly validated configuration for the reduced RQ1 study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("rq1_config.json")

REFERENCE_EMOTIONS = (
    "excited",
    "enthusiastic",
    "joyful",
    "content",
    "calm",
    "serene",
    "angry",
    "furious",
    "anxious",
    "sad",
    "gloomy",
    "miserable",
)
REPORTED_NEGATIVE_EMOTIONS = ("angry", "furious", "anxious", "sad", "gloomy", "miserable")
PREREGISTERED_GROUPS = {
    "anger_fury": ("angry", "furious"),
    "high_arousal_negative": ("angry", "furious", "anxious"),
    "low_arousal_negative": ("sad", "gloomy", "miserable"),
    "all_six_negative": REPORTED_NEGATIVE_EMOTIONS,
}

BASE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
BASE_MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
AUDITED_PARENT_ID = "unsloth/Qwen2.5-14B-Instruct"
AUDITED_PARENT_REVISION = "facfb1bad6443964128be460ff6c98928a4ad4ab"
MISALIGNED_ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice"
MISALIGNED_ADAPTER_REVISION = "25ed05c042afdee9412e9132560cd49f0377ffad"


class RQ1ConfigError(ValueError):
    """Raised when an RQ1 configuration is incomplete or scientifically inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_model_id: str
    base_model_revision: str
    audited_parent_model_id: str
    audited_parent_revision: str
    misaligned_adapter_id: str
    misaligned_adapter_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    adapter_declared_base_revision: str | None


@dataclass(frozen=True, slots=True)
class EmotionGroup:
    name: str
    emotions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmotionConfig:
    reference: tuple[str, ...]
    reported_negative: tuple[str, ...]
    preregistered_groups: tuple[EmotionGroup, ...]

    def group(self, name: str) -> tuple[str, ...]:
        """Return a preregistered group's emotions, failing on an unknown name."""

        for group in self.preregistered_groups:
            if group.name == name:
                return group.emotions
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    layers: tuple[int, ...]
    primary_layer: int
    secondary_layer: int
    activation_site: str
    token_start: int
    hidden_size: int
    extraction_dtype: str
    reduction_dtype: str
    seed: int
    batch_size: int
    attention_implementation: str
    length_bucketing: bool
    resume: bool
    use_kernels: bool
    local_files_only: bool
    short_story_policy: str
    topic_split_a: tuple[int, ...]
    topic_split_b: tuple[int, ...]
    stories_per_topic: int
    expected_stories_per_emotion: int
    permutation_iterations: int
    bootstrap_iterations: int
    random_subspace_iterations: int
    pca_variance_threshold: float
    svd_relative_tolerance: float
    reliability_threshold: float
    expected_missing_emotion: str
    expected_missing_topic_id: int
    expected_missing_story_idx: int

    @property
    def preregistered_layers(self) -> tuple[int, int]:
        return (self.primary_layer, self.secondary_layer)

    @property
    def expected_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (topic_id, story_idx)
            for topic_id in (*self.topic_split_a, *self.topic_split_b)
            for story_idx in range(self.stories_per_topic)
        )

    @property
    def balanced_excluded_cell(self) -> tuple[int, int]:
        return (self.expected_missing_topic_id, self.expected_missing_story_idx)


@dataclass(frozen=True, slots=True)
class PathConfig:
    project_root: Path
    raw_dir: Path
    aligned_activations_dir: Path
    aligned_vectors_path: Path
    neutral_activations_path: Path
    large_artifact_dir: Path
    misaligned_activations_dir: Path
    em_direction_path: Path
    results_dir: Path
    dataset_index_path: Path
    dataset_manifest_path: Path
    hf_home: Path


@dataclass(frozen=True, slots=True)
class EMConfig:
    artifact_path: Path
    aligned_response_path: Path | None
    misaligned_response_path: Path | None
    allow_unverified_legacy: bool
    direction_sign: str


@dataclass(frozen=True, slots=True)
class EqualityProvenance:
    comparison_date_utc: str
    comparison_method: str
    tensors_compared: int
    scalar_values_compared: int
    tensor_names_shapes_dtypes_equal: bool
    tensor_values_exactly_equal: bool
    qwen_shards: int
    unsloth_shards: int
    tokenizer_texts_compared: int
    unpadded_input_ids_exactly_equal: bool
    tokenizer_policy: str
    adapter_tensor_count: int
    adapter_layer_count: int
    adapter_rank: int
    adapter_alpha: int
    adapter_uses_rslora: bool


@dataclass(frozen=True, slots=True)
class RQ1Config:
    schema_version: int
    study_id: str
    models: ModelConfig
    emotions: EmotionConfig
    analysis: AnalysisConfig
    paths: PathConfig
    em: EMConfig
    equality_provenance: EqualityProvenance
    source_path: Path
    source_sha256: str


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RQ1ConfigError(f"{context} must be a JSON object")
    return value


def _require_exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise RQ1ConfigError(f"{context} keys invalid: missing={missing}, unknown={unknown}")


def _tuple_of_strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RQ1ConfigError(f"{context} must be a list of strings")
    return tuple(value)


def _tuple_of_ints(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise RQ1ConfigError(f"{context} must be a list of integers")
    return tuple(value)


def _resolve_path(project_root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RQ1ConfigError(f"{context} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _parse_models(raw: Any) -> ModelConfig:
    data = _require_mapping(raw, "models")
    _require_exact_keys(
        data,
        {
            "base_model_id",
            "base_model_revision",
            "audited_parent_model_id",
            "audited_parent_revision",
            "misaligned_adapter_id",
            "misaligned_adapter_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "adapter_declared_base_revision",
        },
        "models",
    )
    config = ModelConfig(**data)
    expected = {
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "audited_parent_model_id": AUDITED_PARENT_ID,
        "audited_parent_revision": AUDITED_PARENT_REVISION,
        "misaligned_adapter_id": MISALIGNED_ADAPTER_ID,
        "misaligned_adapter_revision": MISALIGNED_ADAPTER_REVISION,
        "tokenizer_id": BASE_MODEL_ID,
        "tokenizer_revision": BASE_MODEL_REVISION,
        "adapter_declared_base_revision": None,
    }
    for field, expected_value in expected.items():
        if getattr(config, field) != expected_value:
            raise RQ1ConfigError(
                f"models.{field} must be the preregistered value {expected_value!r}"
            )
    return config


def _parse_emotions(raw: Any) -> EmotionConfig:
    data = _require_mapping(raw, "emotions")
    _require_exact_keys(
        data, {"reference", "reported_negative", "preregistered_groups"}, "emotions"
    )
    reference = _tuple_of_strings(data["reference"], "emotions.reference")
    reported = _tuple_of_strings(data["reported_negative"], "emotions.reported_negative")
    if reference != REFERENCE_EMOTIONS:
        raise RQ1ConfigError(f"emotions.reference must equal {REFERENCE_EMOTIONS!r}")
    if reported != REPORTED_NEGATIVE_EMOTIONS:
        raise RQ1ConfigError(
            f"emotions.reported_negative must equal {REPORTED_NEGATIVE_EMOTIONS!r}"
        )

    groups_raw = _require_mapping(data["preregistered_groups"], "emotions.preregistered_groups")
    if set(groups_raw) != set(PREREGISTERED_GROUPS):
        raise RQ1ConfigError(
            "emotions.preregistered_groups must contain exactly " f"{tuple(PREREGISTERED_GROUPS)!r}"
        )
    groups = tuple(
        EmotionGroup(name=name, emotions=_tuple_of_strings(groups_raw[name], f"group {name}"))
        for name in PREREGISTERED_GROUPS
    )
    for group in groups:
        if group.emotions != PREREGISTERED_GROUPS[group.name]:
            raise RQ1ConfigError(
                f"preregistered group {group.name!r} must equal "
                f"{PREREGISTERED_GROUPS[group.name]!r}"
            )
    return EmotionConfig(
        reference=reference,
        reported_negative=reported,
        preregistered_groups=groups,
    )


def _parse_analysis(raw: Any) -> AnalysisConfig:
    data = _require_mapping(raw, "analysis")
    expected_keys = {
        "layers",
        "primary_layer",
        "secondary_layer",
        "activation_site",
        "token_start",
        "hidden_size",
        "extraction_dtype",
        "reduction_dtype",
        "seed",
        "batch_size",
        "attention_implementation",
        "length_bucketing",
        "resume",
        "use_kernels",
        "local_files_only",
        "short_story_policy",
        "topic_split_a",
        "topic_split_b",
        "stories_per_topic",
        "expected_stories_per_emotion",
        "permutation_iterations",
        "bootstrap_iterations",
        "random_subspace_iterations",
        "pca_variance_threshold",
        "svd_relative_tolerance",
        "reliability_threshold",
        "expected_missing_emotion",
        "expected_missing_topic_id",
        "expected_missing_story_idx",
    }
    _require_exact_keys(data, expected_keys, "analysis")
    converted = dict(data)
    converted["layers"] = _tuple_of_ints(data["layers"], "analysis.layers")
    converted["topic_split_a"] = _tuple_of_ints(data["topic_split_a"], "analysis.topic_split_a")
    converted["topic_split_b"] = _tuple_of_ints(data["topic_split_b"], "analysis.topic_split_b")
    config = AnalysisConfig(**converted)

    exact_values: dict[str, Any] = {
        "layers": tuple(range(48)),
        "primary_layer": 24,
        "secondary_layer": 32,
        "activation_site": "model.model.layers[l].output[0]",
        "token_start": 50,
        "hidden_size": 5120,
        "extraction_dtype": "bfloat16",
        "reduction_dtype": "float32",
        "seed": 0,
        "batch_size": 64,
        "attention_implementation": "sdpa",
        "length_bucketing": True,
        "resume": True,
        "use_kernels": True,
        "local_files_only": True,
        "short_story_policy": "skip",
        "topic_split_a": tuple(range(10)),
        "topic_split_b": tuple(range(10, 20)),
        "stories_per_topic": 4,
        "expected_stories_per_emotion": 80,
        "permutation_iterations": 1000,
        "bootstrap_iterations": 1000,
        "random_subspace_iterations": 1000,
        "pca_variance_threshold": 0.5,
        "svd_relative_tolerance": 1e-6,
        "reliability_threshold": 0.7,
        "expected_missing_emotion": "furious",
        "expected_missing_topic_id": 2,
        "expected_missing_story_idx": 3,
    }
    for field, expected_value in exact_values.items():
        if getattr(config, field) != expected_value:
            raise RQ1ConfigError(
                f"analysis.{field} must be the preregistered value {expected_value!r}"
            )
    return config


def _parse_paths(raw: Any, config_path: Path) -> PathConfig:
    data = _require_mapping(raw, "paths")
    expected_keys = {
        "project_root",
        "raw_dir",
        "aligned_activations_dir",
        "aligned_vectors_path",
        "neutral_activations_path",
        "large_artifact_dir",
        "misaligned_activations_dir",
        "em_direction_path",
        "results_dir",
        "dataset_index_path",
        "dataset_manifest_path",
        "hf_home",
    }
    _require_exact_keys(data, expected_keys, "paths")
    root_value = data["project_root"]
    if not isinstance(root_value, str) or not root_value:
        raise RQ1ConfigError("paths.project_root must be a non-empty path string")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    resolved = {
        key: _resolve_path(root, data[key], f"paths.{key}")
        for key in expected_keys - {"project_root"}
    }
    return PathConfig(project_root=root, **resolved)


def _parse_optional_path(project_root: Path, value: Any, context: str) -> Path | None:
    if value is None:
        return None
    return _resolve_path(project_root, value, context)


def _parse_em(raw: Any, project_root: Path) -> EMConfig:
    data = _require_mapping(raw, "em")
    _require_exact_keys(
        data,
        {
            "artifact_path",
            "aligned_response_path",
            "misaligned_response_path",
            "allow_unverified_legacy",
            "direction_sign",
        },
        "em",
    )
    config = EMConfig(
        artifact_path=_resolve_path(project_root, data["artifact_path"], "em.artifact_path"),
        aligned_response_path=_parse_optional_path(
            project_root, data["aligned_response_path"], "em.aligned_response_path"
        ),
        misaligned_response_path=_parse_optional_path(
            project_root, data["misaligned_response_path"], "em.misaligned_response_path"
        ),
        allow_unverified_legacy=data["allow_unverified_legacy"],
        direction_sign=data["direction_sign"],
    )
    if type(config.allow_unverified_legacy) is not bool or config.allow_unverified_legacy:
        raise RQ1ConfigError("em.allow_unverified_legacy must be false")
    if config.direction_sign != "misaligned_minus_aligned":
        raise RQ1ConfigError("em.direction_sign must be misaligned_minus_aligned")
    return config


def _parse_provenance(raw: Any) -> EqualityProvenance:
    data = _require_mapping(raw, "equality_provenance")
    expected_keys = {
        "comparison_date_utc",
        "comparison_method",
        "tensors_compared",
        "scalar_values_compared",
        "tensor_names_shapes_dtypes_equal",
        "tensor_values_exactly_equal",
        "qwen_shards",
        "unsloth_shards",
        "tokenizer_texts_compared",
        "unpadded_input_ids_exactly_equal",
        "tokenizer_policy",
        "adapter_tensor_count",
        "adapter_layer_count",
        "adapter_rank",
        "adapter_alpha",
        "adapter_uses_rslora",
    }
    _require_exact_keys(data, expected_keys, "equality_provenance")
    config = EqualityProvenance(**data)
    exact_values: dict[str, Any] = {
        "comparison_method": "tensor-by-tensor torch.equal",
        "tensors_compared": 579,
        "scalar_values_compared": 14_770_033_664,
        "tensor_names_shapes_dtypes_equal": True,
        "tensor_values_exactly_equal": True,
        "qwen_shards": 8,
        "unsloth_shards": 6,
        "tokenizer_texts_compared": 1040,
        "unpadded_input_ids_exactly_equal": True,
        "tokenizer_policy": "use Qwen cf98 tokenizer for both model passes",
        "adapter_tensor_count": 672,
        "adapter_layer_count": 48,
        "adapter_rank": 32,
        "adapter_alpha": 64,
        "adapter_uses_rslora": True,
    }
    for field, expected_value in exact_values.items():
        if getattr(config, field) != expected_value:
            raise RQ1ConfigError(
                f"equality_provenance.{field} must equal audited value {expected_value!r}"
            )
    return config


def load_config(path: str | Path | None = None) -> RQ1Config:
    """Load and strictly validate an immutable reduced-RQ1 configuration."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.expanduser().resolve()
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as exc:
        raise RQ1ConfigError(f"cannot read RQ1 config {config_path}: {exc}") from exc
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RQ1ConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    data = _require_mapping(raw, "root")
    _require_exact_keys(
        data,
        {
            "schema_version",
            "study_id",
            "models",
            "emotions",
            "analysis",
            "paths",
            "em",
            "equality_provenance",
        },
        "root",
    )
    if data["schema_version"] != 1:
        raise RQ1ConfigError("schema_version must be 1")
    if data["study_id"] != "rq1_negative_emotion_overlap_reuse_v1":
        raise RQ1ConfigError("study_id must be rq1_negative_emotion_overlap_reuse_v1")

    paths = _parse_paths(data["paths"], config_path)
    em = _parse_em(data["em"], paths.project_root)
    if em.artifact_path != paths.em_direction_path:
        raise RQ1ConfigError("em.artifact_path and paths.em_direction_path must be identical")

    return RQ1Config(
        schema_version=1,
        study_id=data["study_id"],
        models=_parse_models(data["models"]),
        emotions=_parse_emotions(data["emotions"]),
        analysis=_parse_analysis(data["analysis"]),
        paths=paths,
        em=em,
        equality_provenance=_parse_provenance(data["equality_provenance"]),
        source_path=config_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
