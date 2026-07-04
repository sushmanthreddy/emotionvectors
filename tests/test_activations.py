from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor, nn

from emotion_vectors.activations import (
    _artifact_cache_key,
    _cached_identity_bundle,
    extract_activations,
    length_bucket_indices,
    plan_short_stories,
)
from emotion_vectors.model import (
    CaptureBatch,
    ModelBundle,
    resolve_attention_backend,
    resolve_masked_mean_backend,
    transformer_layers,
)


class _AddBlock(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, hidden: Tensor) -> tuple[Tensor]:
        return (hidden + self.amount,)


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AddBlock(1.0), _AddBlock(2.0), _AddBlock(3.0)])


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()
        self.embedding = nn.Embedding(32, 4)
        self.config = SimpleNamespace(hidden_size=4, num_hidden_layers=3)
        self.forward_calls = 0
        with torch.no_grad():
            values = torch.arange(32, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
            self.embedding.weight.copy_(values)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Tensor:
        del attention_mask, use_cache, output_hidden_states
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


class _SpaceTokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, **_kwargs):
        rows = [[int(word) for word in text.split()] for text in texts]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, values in enumerate(rows):
            ids[row, : len(values)] = torch.tensor(values)
            mask[row, : len(values)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def convert_ids_to_tokens(self, ids):
        return [str(value) for value in ids]


def test_model_captures_every_layer_in_one_forward_and_reduces_in_hook() -> None:
    model = _ToyModel()
    bundle = ModelBundle(model, tokenizer=None, use_kernels=False)
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    captured = bundle.capture_batch(
        input_ids,
        attention_mask,
        token_start=torch.tensor([1, 3]),
        collect_norms=True,
    )

    assert model.forward_calls == 1
    assert len(transformer_layers(model)) == 3
    assert captured.vectors.shape == (2, 3, 4)
    assert captured.mean_token_norms is not None
    assert captured.mean_token_norms.shape == (2, 3)
    assert captured.token_counts.tolist() == [2, 2]

    base = model.embedding(input_ids)
    cumulative_adds = (1.0, 3.0, 6.0)
    expected_layers = []
    for amount in cumulative_adds:
        hidden = base + amount
        expected_layers.append(
            torch.stack([hidden[0, 1:3].mean(dim=0), hidden[1, 3:5].mean(dim=0)])
        )
    expected = torch.stack(expected_layers, dim=1)
    torch.testing.assert_close(captured.vectors, expected)


def test_v2_capture_projects_in_hook_without_retaining_raw_activations() -> None:
    model = _ToyModel()
    bundle = ModelBundle(model, tokenizer=_SpaceTokenizer(), use_kernels=False)
    texts = ["1 2 3", "4 5"]
    directions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    full = bundle.capture_token_activations_batch(texts, layer=1)
    projected = bundle.capture_token_projections_batch(texts, directions, layer=1)
    unit = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    for (tokens, activations), (projected_tokens, values) in zip(full, projected, strict=True):
        assert projected_tokens == tokens
        torch.testing.assert_close(values, activations @ unit.T)


def test_short_story_policies_use_slice_safe_boundary() -> None:
    skipped = plan_short_stories([4, 5, 6, 10], token_start=5, policy="skip")
    assert skipped.keep_indices == (2, 3)
    assert skipped.token_starts == (5, 5)
    assert skipped.skipped_indices == (0, 1)

    second_half = plan_short_stories([4, 5, 6, 10], token_start=5, policy="second_half")
    assert second_half.keep_indices == (0, 1, 2, 3)
    assert second_half.token_starts == (2, 2, 5, 5)
    assert second_half.skipped_indices == ()


def test_length_bucketing_is_deterministic_and_can_be_disabled() -> None:
    lengths = [10, 2, 8, 2, 7]
    assert length_bucket_indices(lengths, batch_size=2, enabled=True) == [
        [1, 3],
        [4, 2],
        [0],
    ]
    assert length_bucket_indices(lengths, batch_size=2, enabled=False) == [
        [0, 1],
        [2, 3],
        [4],
    ]


class _StubBundle:
    n_layers = 2
    hidden_size = 3
    effective_attention_backend = "sdpa"
    effective_masked_mean_backend = "torch"
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="stub/model"))

    def __init__(self) -> None:
        self.seen_batch_lengths: list[list[int]] = []

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def encode_and_capture(
        self,
        texts: list[str],
        *,
        token_start: Tensor,
        collect_norms: bool,
    ) -> CaptureBatch:
        lengths = [len(text.split()) for text in texts]
        self.seen_batch_lengths.append(lengths)
        starts = token_start.to(torch.int64)
        counts = torch.tensor(lengths, dtype=torch.int64) - starts
        vectors = torch.empty(len(texts), self.n_layers, self.hidden_size)
        for row, length in enumerate(lengths):
            for layer in range(self.n_layers):
                vectors[row, layer].fill_(length * 10 + layer)
        norm_means = torch.tensor([[2.0, 4.0]] * len(texts)) if collect_norms else None
        return CaptureBatch(vectors, counts, norm_means)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_stage_two_writes_schema_restores_order_and_caches_layer_norms(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    activations = tmp_path / "activations"
    _write_jsonl(
        raw / "stories" / "sad.jsonl",
        [
            {"emotion": "sad", "topic_id": 1, "story_idx": 0, "text": "a b c d e"},
            {"emotion": "sad", "topic_id": 2, "story_idx": 1, "text": "a b"},
            {"emotion": "sad", "topic_id": 3, "story_idx": 2, "text": "a b c"},
        ],
    )
    _write_jsonl(
        raw / "neutral.jsonl",
        [{"topic_id": 1, "dialogue_idx": 0, "text": "a b c d"}],
    )
    config = SimpleNamespace(
        paths=SimpleNamespace(raw=raw, activations=activations),
        model_name="stub/model",
        model_revision="abc",
        dtype="float32",
        attn_implementation="sdpa",
        use_kernels=False,
        num_gpus=0,
        local_files_only=True,
        emotions=("sad",),
        token_start=2,
        short_story_policy="skip",
        batch_size=2,
        length_bucketing=True,
        resume=False,
    )
    bundle = _StubBundle()

    summary = extract_activations(config, bundle)  # type: ignore[arg-type]

    assert summary.input_count == 4
    assert summary.output_count == 3
    assert summary.skipped_count == 1
    assert bundle.seen_batch_lengths[0] == [3, 5]
    # Norms are collected by those clean-run forwards, not a duplicate pass.
    assert bundle.seen_batch_lengths == [[3, 5], [4]]
    with np.load(activations / "stories" / "sad.npz", allow_pickle=False) as artifact:
        assert set(artifact.files) == {"vectors", "meta"}
        assert artifact["vectors"].shape == (2, 2, 3)
        assert artifact["vectors"].dtype == np.float32
        # Length-bucketed inference is restored to Stage-1 record order on disk.
        np.testing.assert_array_equal(artifact["vectors"][:, 0, 0], [50.0, 30.0])
        assert artifact["meta"].dtype.names == ("topic_id", "story_idx")
        np.testing.assert_array_equal(artifact["meta"]["topic_id"], [1, 3])
    with np.load(activations / "neutral.npz", allow_pickle=False) as artifact:
        assert set(artifact.files) == {"vectors"}
        assert artifact["vectors"].shape == (1, 2, 3)
    np.testing.assert_allclose(
        np.load(activations / "layer_norms.npy", allow_pickle=False),
        np.array([2.0, 4.0], dtype=np.float32),
    )
    story_path = activations / "stories" / "sad.npz"
    story_cache = json.loads((activations / "stories" / "sad.cache.json").read_text())
    assert story_cache["archive_sha256"] == hashlib.sha256(story_path.read_bytes()).hexdigest()
    assert story_cache["effective_attention_backend"] == "sdpa"
    assert story_cache["effective_masked_mean_backend"] == "torch"

    # When activation shards resume but the norm cache is missing, only the
    # canonical norm sample is forwarded again.
    (activations / "layer_norms.npy").unlink()
    (activations / "layer_norms.cache.json").unlink()
    config.resume = True
    bundle.seen_batch_lengths.clear()
    resumed_without_norms = extract_activations(config, bundle)  # type: ignore[arg-type]
    assert resumed_without_norms.resumed_files == 2
    assert bundle.seen_batch_lengths == [[3, 5]]

    # A fully valid cache performs no model forwards.
    bundle.seen_batch_lengths.clear()
    fully_resumed = extract_activations(config, bundle)  # type: ignore[arg-type]
    assert fully_resumed.resumed_files == 2
    assert bundle.seen_batch_lengths == []

    # A same-size bit flip must fail digest validation and rebuild that shard.
    with story_path.open("r+b") as handle:
        handle.seek(story_path.stat().st_size // 2)
        original = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original[0] ^ 0xFF]))
    bundle.seen_batch_lengths.clear()
    repaired = extract_activations(config, bundle)  # type: ignore[arg-type]
    assert repaired.resumed_files == 1
    assert bundle.seen_batch_lengths == [[3, 5]]
    repaired_cache = json.loads((activations / "stories" / "sad.cache.json").read_text())
    assert repaired_cache["archive_sha256"] == hashlib.sha256(story_path.read_bytes()).hexdigest()

    # The no-weight resume path resolves the same effective backends and reuses
    # digest validation rather than requiring a ModelBundle.
    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=3,
            _name_or_path="stub/model",
        ),
    )
    identity = _cached_identity_bundle(config)
    assert identity is not None
    assert identity.effective_attention_backend == "sdpa"
    assert identity.effective_masked_mean_backend == "torch"
    assert len(identity._validated_artifact_keys) == 2


def test_second_half_policy_reduces_short_text_instead_of_skipping(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    activations = tmp_path / "activations"
    _write_jsonl(
        raw / "stories" / "calm.jsonl",
        [{"emotion": "calm", "topic_id": 1, "story_idx": 0, "text": "a b c"}],
    )
    _write_jsonl(
        raw / "neutral.jsonl",
        [{"topic_id": 1, "dialogue_idx": 0, "text": "a b c"}],
    )
    config = SimpleNamespace(
        paths=SimpleNamespace(raw=raw, activations=activations),
        token_start=5,
        short_story_policy="second_half",
        batch_size=1,
        length_bucketing=True,
        resume=False,
    )

    summary = extract_activations(config, _StubBundle())  # type: ignore[arg-type]
    assert summary.skipped_count == 0
    assert summary.output_count == 2
    with np.load(activations / "stories" / "calm.npz", allow_pickle=False) as artifact:
        assert artifact["vectors"].shape[0] == 1


def test_effective_backends_are_part_of_activation_cache_identity(tmp_path: Path) -> None:
    source = tmp_path / "stories.jsonl"
    _write_jsonl(
        source,
        [{"emotion": "calm", "topic_id": 1, "story_idx": 0, "text": "a b c"}],
    )
    config = SimpleNamespace(
        model_name="stub/model",
        model_revision="abc",
        dtype="float32",
        attn_implementation="flash_attention_2",
        token_start=2,
        short_story_policy="skip",
        batch_size=1,
        length_bucketing=True,
        use_kernels=True,
        num_gpus=1,
    )

    def bundle(attention: str, reduction: str) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(_name_or_path="stub/model")),
            n_layers=2,
            hidden_size=3,
            effective_attention_backend=attention,
            effective_masked_mean_backend=reduction,
        )

    sdpa_torch = _artifact_cache_key(
        source, config, bundle("sdpa", "torch"), metadata_fields=("topic_id", "story_idx")
    )
    flash_torch = _artifact_cache_key(
        source,
        config,
        bundle("flash_attention_2", "torch"),
        metadata_fields=("topic_id", "story_idx"),
    )
    sdpa_triton = _artifact_cache_key(
        source, config, bundle("sdpa", "triton"), metadata_fields=("topic_id", "story_idx")
    )
    assert len({sdpa_torch, flash_torch, sdpa_triton}) == 3
    assert resolve_masked_mean_backend(True, device="cpu") == "torch"
    assert resolve_attention_backend("sdpa") == "sdpa"
