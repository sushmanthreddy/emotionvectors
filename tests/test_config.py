from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from emotion_vectors.config import ExperimentConfig, PathsConfig, load_config


def test_bundled_assets_match_repository_config() -> None:
    root = Path(__file__).resolve().parents[1]
    bundled = root / "src/emotion_vectors/assets"
    for name in ("default.yaml", "topics.txt", "emotions.txt", "prompts.py"):
        assert (bundled / name).read_bytes() == (root / "config" / name).read_bytes()


def test_heldout_dataset_and_revision_can_be_replaced_together() -> None:
    config = load_config(
        overrides={
            "heldout_datasets": ["isotonic_ha"],
            "heldout_revisions": {"isotonic_ha": "abcdef1234567"},
        }
    )
    assert config.heldout_datasets == ("isotonic_ha",)
    assert config.heldout_revisions == ("abcdef1234567",)


def test_default_config_matches_build_spec() -> None:
    config = load_config()

    assert isinstance(config, ExperimentConfig)
    assert is_dataclass(config)
    assert config.model_name == "Qwen/Qwen2.5-32B-Instruct"
    assert config.model_revision == "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    assert config.generator == "Qwen/Qwen2.5-32B-Instruct"
    assert config.generator_revision == config.model_revision
    assert config.n_stories == 12
    assert config.generation_batch_size == 16
    assert config.generation_max_attempts == 3
    assert config.generation_lexical_policy == "warn"
    assert config.token_start == 50
    assert config.layer_norm_sample_size == 1024
    assert config.short_story_policy == "skip"
    assert config.pca_variance == pytest.approx(0.5)
    assert config.pca_device == "auto"
    assert config.primary_layer_frac == pytest.approx(0.6667)
    assert config.steering_strength == pytest.approx(0.5)
    assert config.kmeans_k == 6
    assert config.heldout_datasets == (
        "common_corpus",
        "pile_subset",
        "lmsys_chat_1m",
        "isotonic_ha",
    )
    assert config.activation_percentile == 90
    assert config.seed == 0
    assert config.batch_size == 64
    assert config.dtype == "bfloat16"
    assert config.attn_implementation == "flash_attention_2"
    assert config.num_gpus == 1
    assert config.local_files_only is False
    assert config.length_bucketing is True
    assert config.resume is True
    assert config.heldout_max_docs == 50_000
    assert config.heldout_max_tokens == 512
    assert len(config.heldout_revisions) == len(config.heldout_datasets)
    assert config.use_kernels is True
    assert config.plot_format == ("png", "svg")
    assert config.dpi == 150
    assert config.v1_top_fraction == pytest.approx(0.1)
    assert config.v1_min_control_percentile == pytest.approx(0.75)
    assert config.v2_top_documents == 3
    assert config.v3_top_k == 20
    assert config.v5_min_absolute_correlation == pytest.approx(0.5)
    assert config.v6_require_generation_sanity is True


def test_assets_are_loaded_centrally_and_verbatim() -> None:
    config = load_config()

    assert len(config.topics) == 100
    assert config.topics[0] == "An artist discovers someone has tattooed their work"
    assert config.topics[-1] == (
        "Someone finds their college application essay published as an example"
    )
    assert len(config.emotions) == 30
    assert config.emotions[:5] == (
        "excited",
        "elated",
        "ecstatic",
        "enthusiastic",
        "joyful",
    )
    assert config.emotions[-5:] == (
        "nostalgic",
        "guilty",
        "ashamed",
        "jealous",
        "disgusted",
    )
    assert all(not emotion.startswith("#") for emotion in config.emotions)

    assert config.emotional_stories_prompt.startswith(
        "Write {n_stories} different stories based on the following premise."
    )
    assert "Topic: {topic}" in config.emotional_stories_prompt
    assert "feeling {emotion}" in config.emotional_stories_prompt
    assert config.neutral_dialogues_prompt.startswith(
        "Write {n_stories} different dialogues based on the following topic."
    )
    assert config.probe_he == "Human: How does he feel?\n\nAssistant: He feels"
    assert config.probe_i == "Human: How do you feel?\n\nAssistant: I feel"
    assert config.probe_prompts == (config.probe_he, config.probe_i)


def test_fast_14b_preset_is_balanced_reduced_and_isolated() -> None:
    root = Path(__file__).resolve().parents[1]
    full = load_config()
    fast = load_config(root / "config/fast_14b/default.yaml")
    revision = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    quadrants = (
        ("excited", "enthusiastic", "joyful"),
        ("content", "calm", "serene"),
        ("angry", "furious", "anxious"),
        ("sad", "gloomy", "miserable"),
    )

    assert fast.model_name == "Qwen/Qwen2.5-14B-Instruct"
    assert fast.generator == fast.model_name
    assert fast.model_revision == revision
    assert fast.generator_revision == revision
    assert fast.emotions == tuple(emotion for quadrant in quadrants for emotion in quadrant)
    assert set(fast.emotions) < set(full.emotions)
    assert len(fast.emotions) == 12
    assert len(fast.topics) == 20
    assert fast.n_stories == 4
    assert fast.kmeans_k == 4
    assert fast.heldout_max_docs == 3_000
    assert fast.local_files_only is True

    assert fast.paths.raw == root / "data/fast_14b/raw"
    assert fast.paths.activations == Path("/tmp/emotion-vectors-fast14b/activations")
    assert fast.paths.vectors == root / "data/fast_14b/vectors"
    assert fast.paths.outputs == root / "outputs/fast_14b"
    assert fast.paths.raw != full.paths.raw
    assert fast.paths.activations != full.paths.activations
    assert fast.paths.vectors != full.paths.vectors
    assert fast.paths.outputs != full.paths.outputs

    assert len(full.emotions) == 30
    assert len(full.topics) == 100


def test_config_and_nested_paths_are_frozen_and_resolved() -> None:
    config = load_config()

    assert isinstance(config.paths, PathsConfig)
    assert config.paths.raw == config.project_root / "data/raw"
    assert config.paths.outputs == config.project_root / "outputs"
    assert config.resolve_path("data/vectors") == config.paths.vectors
    assert config.resolve_path(config.paths.raw) == config.paths.raw

    with pytest.raises(FrozenInstanceError):
        config.seed = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.paths.raw = Path("elsewhere")  # type: ignore[misc]


def test_overrides_support_nested_and_dotted_keys(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    config = load_config(
        overrides={
            "batch_size": 2,
            "paths": {"raw": tmp_path / "raw"},
            "paths.outputs": output_dir,
            "plot_format": ["PNG"],
        }
    )

    assert config.batch_size == 2
    assert config.paths.raw == (tmp_path / "raw").resolve()
    assert config.paths.outputs == output_dir.resolve()
    assert config.plot_format == ("png",)


def test_config_hash_is_stable_and_sensitive_to_inputs() -> None:
    first = load_config()
    second = load_config()
    changed = load_config(overrides={"seed": 1})

    assert len(first.config_hash) == 64
    assert first.config_hash == second.config_hash
    assert changed.config_hash != first.config_hash


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"not_a_key": 1}, "Unknown configuration keys"),
        ({"short_story_policy": "truncate"}, "must be one of"),
        ({"pca_variance": 0.0}, "must be in"),
        ({"batch_size": True}, "must be an integer"),
        ({"plot_format": []}, "must be a non-empty list"),
        ({"v1_concentration_factor": 1.0}, "must be greater than 1"),
        ({"paths.unknown": "somewhere"}, "Unknown path keys"),
    ],
)
def test_invalid_config_fails_early(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(overrides=overrides)
