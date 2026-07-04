from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from em_organism_dir.emotion_analysis.extraction import (
    extract_misaligned_shards,
    register_existing_aligned_shards,
)
from em_organism_dir.emotion_analysis.modeling import (
    EXPECTED_ACTIVATION_SITE,
    ModelProvenance,
    load_rq1_model,
    require_immutable_revision,
)
from emotion_vectors.model import ResidualStreamModel

EMOTIONS = (
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


class _AddBlock(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, hidden: Tensor) -> tuple[Tensor]:
        return (hidden + self.amount,)


class _Core(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_AddBlock(index + 1.0) for index in range(n_layers))


class _TinyQwen(nn.Module):
    def __init__(self, *, n_layers: int = 3, hidden_size: int = 4) -> None:
        super().__init__()
        self.model = _Core(n_layers)
        self.embedding = nn.Embedding(64, hidden_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            model_type="qwen2",
        )
        self.forward_calls = 0
        self.last_attention_mask: Tensor | None = None
        with torch.no_grad():
            values = torch.arange(64, dtype=torch.float32).unsqueeze(1)
            self.embedding.weight.copy_(values.repeat(1, hidden_size))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Tensor:
        del use_cache, output_hidden_states
        self.forward_calls += 1
        self.last_attention_mask = None if attention_mask is None else attention_mask.detach().cpu()
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


class _NumericTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    padding_side = "left"

    def __init__(self, *, missing_pad: bool = False) -> None:
        self._pad_token = None if missing_pad else "<pad>"
        self.pad_token_id = None if missing_pad else 0

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        self._pad_token = value
        self.pad_token_id = self.eos_token_id

    def __call__(self, texts, *, padding=False, return_tensors=None, **_kwargs):
        rows = [[int(word) for word in text.split()] for text in texts]
        if not padding:
            return {
                "input_ids": rows,
                "attention_mask": [[1] * len(row) for row in rows],
            }
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row)
            attention_mask[row_index, : len(row)] = 1
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}


class _FakeAutoTokenizer:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append((model_id, kwargs))
        return _NumericTokenizer(missing_pad=True)


class _FakeAutoModel:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []
    latest: ClassVar[_TinyQwen | None] = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append((model_id, kwargs))
        cls.latest = _TinyQwen()
        return cls.latest


class _FakePeftModel:
    calls: ClassVar[list[tuple[_TinyQwen, str, dict[str, object]]]] = []

    @classmethod
    def from_pretrained(cls, model: _TinyQwen, adapter_id: str, **kwargs):
        cls.calls.append((model, adapter_id, kwargs))
        model.peft_config = {"default": SimpleNamespace(inference_mode=True)}
        return model


def _config(tmp_path: Path) -> SimpleNamespace:
    raw = tmp_path / "raw"
    aligned = tmp_path / "aligned"
    large = tmp_path / "large"
    return SimpleNamespace(
        schema_version=1,
        study_id="rq1-test-v1",
        source_sha256="config-sha",
        models=SimpleNamespace(
            base_model_id="Qwen/test-base",
            base_model_revision="a" * 40,
            audited_parent_model_id="unsloth/test-base",
            audited_parent_revision="b" * 40,
            misaligned_adapter_id="org/test-adapter",
            misaligned_adapter_revision="c" * 40,
            tokenizer_id="Qwen/test-base",
            tokenizer_revision="a" * 40,
        ),
        emotions=SimpleNamespace(reference=EMOTIONS),
        analysis=SimpleNamespace(
            layers=tuple(range(3)),
            primary_layer=1,
            activation_site=EXPECTED_ACTIVATION_SITE,
            token_start=2,
            hidden_size=4,
            extraction_dtype="bfloat16",
            reduction_dtype="float32",
            seed=0,
            batch_size=2,
            attention_implementation="sdpa",
            length_bucketing=True,
            resume=True,
            use_kernels=False,
            local_files_only=True,
            short_story_policy="skip",
        ),
        paths=SimpleNamespace(
            raw_dir=raw,
            aligned_activations_dir=aligned,
            large_artifact_dir=large,
            misaligned_activations_dir=large / "misaligned",
            hf_home=tmp_path / "hf",
        ),
    )


def _provenance(config: SimpleNamespace, source: str) -> ModelProvenance:
    if source == "base":
        base_id = config.models.base_model_id
        base_revision = config.models.base_model_revision
        model_id = base_id
        model_revision = base_revision
        adapter_id = None
        adapter_revision = None
    else:
        base_id = config.models.audited_parent_model_id
        base_revision = config.models.audited_parent_revision
        model_id = config.models.misaligned_adapter_id
        model_revision = config.models.misaligned_adapter_revision
        adapter_id = model_id
        adapter_revision = model_revision
    return ModelProvenance(
        model_source=source,  # type: ignore[arg-type]
        model_id=model_id,
        model_revision=model_revision,
        base_model_id=base_id,
        base_model_revision=base_revision,
        tokenizer_id=config.models.tokenizer_id,
        tokenizer_revision=config.models.tokenizer_revision,
        adapter_id=adapter_id,
        adapter_revision=adapter_revision,
        adapter_merged=False,
        extraction_dtype="bfloat16",
        attention_backend="sdpa",
        activation_site=EXPECTED_ACTIVATION_SITE,
        n_layers=3,
        hidden_size=4,
    )


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _write_corpus(config: SimpleNamespace) -> None:
    for index, emotion in enumerate(EMOTIONS):
        _write_jsonl(
            config.paths.raw_dir / "stories" / f"{emotion}.jsonl",
            {
                "emotion": emotion,
                "topic_id": index,
                "story_idx": 0,
                "text": "1 2 3 4",
            },
        )
    _write_jsonl(
        config.paths.raw_dir / "neutral.jsonl",
        {"topic_id": 0, "dialogue_idx": 0, "text": "5 6 7 8"},
    )


def _write_npz_with_cache(
    path: Path,
    vectors: np.ndarray,
    *,
    metadata: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        np.savez_compressed(path, vectors=vectors)
    else:
        np.savez_compressed(path, vectors=vectors, meta=metadata)
    payload = {
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_size": path.stat().st_size,
        "vector_shape": list(vectors.shape),
        "vector_dtype": "float32",
    }
    path.with_suffix(".cache.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_aligned_artifacts(config: SimpleNamespace) -> None:
    vectors = np.ones((1, 3, 4), dtype=np.float32)
    story_dtype = np.dtype([("topic_id", np.int64), ("story_idx", np.int64)])
    story_meta = np.array([(0, 0)], dtype=story_dtype)
    for emotion in EMOTIONS:
        _write_npz_with_cache(
            config.paths.aligned_activations_dir / "stories" / f"{emotion}.npz",
            vectors,
            metadata=story_meta,
        )
    _write_npz_with_cache(
        config.paths.aligned_activations_dir / "neutral.npz",
        vectors,
        metadata=None,
    )
    manifest = {
        "config": {
            "model_name": config.models.base_model_id,
            "model_revision": config.models.base_model_revision,
            "token_start": config.analysis.token_start,
            "dtype": config.analysis.extraction_dtype,
            "emotions": list(EMOTIONS),
        }
    }
    (config.paths.aligned_activations_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _bundle(config: SimpleNamespace) -> ResidualStreamModel:
    model = _TinyQwen().to(dtype=torch.bfloat16)
    model.peft_config = {"default": SimpleNamespace(inference_mode=True)}
    bundle = ResidualStreamModel(
        model,
        _NumericTokenizer(),
        use_kernels=False,
        output_device="cpu",
        effective_attention_backend="sdpa",
    )
    bundle.rq1_provenance = _provenance(config, "misaligned")
    return bundle


def test_loads_pinned_unmerged_adapter_and_uses_qwen_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _FakeAutoTokenizer.calls.clear()
    _FakeAutoModel.calls.clear()
    _FakePeftModel.calls.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    bundle = load_rq1_model(
        config,
        "misaligned",
        auto_model_cls=_FakeAutoModel,
        auto_tokenizer_cls=_FakeAutoTokenizer,
        peft_model_cls=_FakePeftModel,
    )

    tokenizer_call = _FakeAutoTokenizer.calls[0]
    assert tokenizer_call[0] == config.models.tokenizer_id
    assert tokenizer_call[1]["revision"] == config.models.tokenizer_revision
    assert tokenizer_call[1]["local_files_only"] is True
    assert tokenizer_call[1]["cache_dir"] == str(config.paths.hf_home / "hub")
    assert bundle.tokenizer.padding_side == "right"
    assert bundle.tokenizer.pad_token_id == bundle.tokenizer.eos_token_id

    model_call = _FakeAutoModel.calls[0]
    assert model_call[0] == config.models.audited_parent_model_id
    assert model_call[1]["revision"] == config.models.audited_parent_revision
    assert model_call[1]["cache_dir"] == str(config.paths.hf_home / "hub")
    assert model_call[1]["dtype"] is torch.bfloat16
    assert model_call[1]["attn_implementation"] == "sdpa"
    assert _FakePeftModel.calls[0][1] == config.models.misaligned_adapter_id
    assert _FakePeftModel.calls[0][2]["revision"] == config.models.misaligned_adapter_revision
    assert _FakePeftModel.calls[0][2]["is_trainable"] is False
    assert _FakePeftModel.calls[0][2]["autocast_adapter_dtype"] is False
    assert _FakePeftModel.calls[0][2]["cache_dir"] == str(config.paths.hf_home / "hub")
    assert bundle.model.training is False
    assert all(not parameter.requires_grad for parameter in bundle.model.parameters())
    assert {parameter.dtype for parameter in bundle.model.parameters()} == {torch.bfloat16}
    assert bundle.rq1_provenance.adapter_merged is False

    captured = bundle.encode_and_capture(["1 2 3 4", "5 6"], token_start=1)
    assert captured.token_counts.tolist() == [3, 1]
    assert _FakeAutoModel.latest is not None
    torch.testing.assert_close(
        _FakeAutoModel.latest.last_attention_mask,
        torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    )


def test_base_loader_does_not_attach_adapter(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    _FakeAutoModel.calls.clear()
    _FakePeftModel.calls.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    bundle = load_rq1_model(
        config,
        "base",
        auto_model_cls=_FakeAutoModel,
        auto_tokenizer_cls=_FakeAutoTokenizer,
        peft_model_cls=_FakePeftModel,
    )

    assert _FakeAutoModel.calls[0][0] == config.models.base_model_id
    assert not _FakePeftModel.calls
    assert bundle.rq1_provenance.adapter_id is None


def test_revisions_must_be_immutable() -> None:
    assert require_immutable_revision("a" * 40, field="revision") == "a" * 40
    with pytest.raises(ValueError, match="immutable"):
        require_immutable_revision("main", field="revision")


def test_misaligned_extraction_writes_all_shards_with_metadata_and_resumes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_corpus(config)
    bundle = _bundle(config)

    first = extract_misaligned_shards(config, bundle)

    assert first.model_source == "misaligned"
    assert first.model_loaded is False
    assert len(first.shards) == 13
    assert first.input_count == first.output_count == 13
    assert first.skipped_count == 0
    assert first.resumed_shards == 0
    assert bundle.model.forward_calls == 13
    assert first.manifest_path.name == "manifest.v1.json"

    with np.load(
        config.paths.misaligned_activations_dir / "stories" / "angry.npz",
        allow_pickle=False,
    ) as artifact:
        assert artifact["vectors"].shape == (1, 3, 4)
        assert artifact["vectors"].dtype == np.float32
        assert artifact["meta"].dtype.names == ("topic_id", "story_idx")
    with np.load(
        config.paths.misaligned_activations_dir / "neutral.npz", allow_pickle=False
    ) as artifact:
        assert artifact["meta"].dtype.names == ("topic_id", "dialogue_idx")

    angry_cache = json.loads(
        (config.paths.misaligned_activations_dir / "stories" / "angry.cache.json").read_text(
            encoding="utf-8"
        )
    )
    assert angry_cache["artifact_schema_version"] == 1
    assert angry_cache["model"]["adapter_id"] == config.models.misaligned_adapter_id
    assert angry_cache["activation_site"] == EXPECTED_ACTIVATION_SITE
    assert angry_cache["token_start"] == 2
    assert (
        angry_cache["source_sha256"]
        == hashlib.sha256(
            (config.paths.raw_dir / "stories" / "angry.jsonl").read_bytes()
        ).hexdigest()
    )

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model_source"] == "misaligned"
    assert manifest["model"]["adapter_merged"] is False
    assert len(manifest["shards"]) == 13

    second = extract_misaligned_shards(config, bundle)
    assert bundle.model.forward_calls == 13
    assert second.resumed_shards == 13
    assert all(shard.resumed for shard in second.shards)
    assert second.dataset_sha256 == first.dataset_sha256


def test_base_stage_only_validates_and_registers_existing_shards(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_corpus(config)
    _write_aligned_artifacts(config)

    summary = register_existing_aligned_shards(config)

    assert summary.model_source == "base"
    assert summary.model_loaded is False
    assert summary.artifact_root == config.paths.aligned_activations_dir.resolve()
    assert len(summary.shards) == 13
    assert summary.resumed_shards == 13
    assert summary.manifest_path == config.paths.large_artifact_dir / "aligned_registration.v1.json"
    payload = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "registered_existing_aligned_shards"
    assert payload["model"]["model_id"] == config.models.base_model_id
    assert (
        payload["source_run_manifest"]["sha256"]
        == hashlib.sha256(
            (config.paths.aligned_activations_dir / "run_manifest.json").read_bytes()
        ).hexdigest()
    )


def test_registration_rejects_wrong_activation_shape(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_corpus(config)
    _write_aligned_artifacts(config)
    bad_path = config.paths.aligned_activations_dir / "stories" / "angry.npz"
    meta = np.array([(0, 0)], dtype=np.dtype([("topic_id", np.int64), ("story_idx", np.int64)]))
    _write_npz_with_cache(
        bad_path,
        np.ones((1, 2, 4), dtype=np.float32),
        metadata=meta,
    )

    with pytest.raises(ValueError, match="activation shape"):
        register_existing_aligned_shards(config)
