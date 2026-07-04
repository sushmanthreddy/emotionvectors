from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from emotion_vectors.cli import verify_main
from emotion_vectors.verify import run_verifications
from emotion_vectors.verify.localization import run_localization
from emotion_vectors.verify.logit_lens import token_is_emotion_related
from emotion_vectors.verify.steering_probe import evaluate_huggingface_probabilities


def test_all_six_verifiers_emit_contract_artifacts(tmp_path: Path) -> None:
    labels = ("excited", "joyful", "calm", "sad", "gloomy", "angry")
    vectors = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.8, 0.05],
            [1.0, -1.0, 0.0],
            [-1.0, -0.8, 0.0],
            [-1.0, -1.0, -0.05],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    vector_dir = tmp_path / "vectors"
    activation_dir = tmp_path / "activations"
    output_dir = tmp_path / "outputs"
    vector_dir.mkdir()
    activation_dir.mkdir()
    all_layers = np.stack((vectors * 0.8, vectors, vectors * 1.1), axis=1)
    np.savez(
        vector_dir / "emotion_vectors.npz",
        raw=all_layers,
        denoised=all_layers,
        emotions=np.asarray(labels),
        primary_layer=np.asarray(1),
    )
    np.save(activation_dir / "layer_norms.npy", np.asarray([2.0, 3.0, 4.0]))
    config = SimpleNamespace(
        paths=SimpleNamespace(
            vectors=vector_dir,
            activations=activation_dir,
            outputs=output_dir,
            raw=tmp_path / "raw",
        ),
        plot_format=("png", "svg"),
        dpi=45,
        activation_percentile=80,
        heldout_max_docs=100,
        use_kernels=False,
        kmeans_k=2,
        seed=0,
        steering_strength=0.5,
        emotions=labels,
        topics=(),
    )

    stories = []
    documents = []
    for index, (label, vector) in enumerate(zip(labels, vectors, strict=True)):
        activations = np.zeros((10, vectors.shape[1]), dtype=np.float32)
        activations[7] = vector * 8.0
        tokens = [f"t{position}" for position in range(10)]
        stories.append(
            {
                "emotion": label,
                "story_id": str(index),
                "text": f"indirect {label} context",
                "tokens": tokens,
                "activations": activations,
            }
        )
        documents.append(
            {
                "dataset": "synthetic",
                "document_id": str(index),
                "text": f"This held-out context contains {label}.",
                "tokens": [label, "context", "peak"],
                "activations": np.stack((np.zeros_like(vector), vector * 0.1, vector * 5.0)),
            }
        )

    baseline = np.full((2, len(labels)), 0.08, dtype=np.float64)
    steered = np.full((2, len(labels), len(labels)), 0.07, dtype=np.float64)
    for prompt in range(2):
        for emotion in range(len(labels)):
            steered[prompt, emotion, emotion] = 0.18
    generated = {
        prompt: {label: f" {label}." for label in labels} for prompt in ("he_feels", "i_feel")
    }
    ratings = {
        label: (float(vector[0]), float(vector[1]))
        for label, vector in zip(labels, vectors, strict=True)
    }
    results = run_verifications(
        config,
        checks=("V1", "V2", "V3", "V4", "V5", "V6"),
        inputs={
            "V1": {
                "stories": stories,
                "minimum_emotion_pass_rate": 0.0,
            },
            "V2": {
                "documents": documents,
                "top_documents": 1,
                "minimum_emotion_pass_rate": 0.0,
            },
            "V3": {
                "unembedding": vectors,
                "vocabulary_tokens": labels,
                "top_k": 2,
                "minimum_related_hit_rate": 0.0,
            },
            "V4": {
                "k": 2,
                "synonym_pairs": (("excited", "joyful"), ("sad", "gloomy")),
                "opposite_pairs": (("joyful", "sad"),),
            },
            "V5": {
                "affect_ratings": ratings,
                "minimum_absolute_correlation": 0.0,
            },
            "V6": {
                "baseline_probabilities": baseline,
                "steered_probabilities": steered,
                "generated_texts": generated,
                "related_words": {},
                "minimum_emotion_pass_rate": 1.0,
            },
        },
    )

    assert set(results) == {"V1", "V2", "V3", "V4", "V5", "V6"}
    assert all(result.passed for result in results.values())
    for result in results.values():
        assert result.report.is_file()
        assert "VERDICT: **PASS**" in result.report.read_text(encoding="utf-8")
        assert result.tables and all(path.is_file() for path in result.tables)
        assert result.figures and all(path.is_file() for path in result.figures)
        for figure in result.figures:
            assert figure.with_suffix(".csv").is_file()


class _ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text.startswith(" "):
            return {" one": [5], " multi": [6, 7]}[text]
        if text == "one":
            return [5]
        if text == "multi":
            return [6, 7]
        if "Assistant:" in text:
            return [1, 2, 3, 4]
        return [1, 2]

    def __call__(
        self,
        texts,
        *,
        padding: bool = False,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ):
        values = [texts] if isinstance(texts, str) else list(texts)
        rows = [self.encode(value, add_special_tokens=add_special_tokens) for value in values]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, values_at_row in enumerate(rows):
            ids[row, : len(values_at_row)] = torch.tensor(values_at_row)
            mask[row, : len(values_at_row)] = 1
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": mask}
        single = rows[0] if isinstance(texts, str) else rows
        return {"input_ids": single}


class _ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, 2)
        self.model = SimpleNamespace(layers=nn.ModuleList([nn.Identity()]))
        self.layers = self.model.layers
        self.register_module("layer_0", self.layers[0])
        self.lm_head = nn.Linear(2, 10, bias=False)
        nn.init.zeros_(self.embedding.weight)
        nn.init.zeros_(self.lm_head.weight)
        with torch.no_grad():
            self.lm_head.weight[5, 0] = 2.0
            self.lm_head.weight[6, 0] = 2.0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.layers[0](self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_v6_scores_full_multitoken_probability_and_uses_shared_hook() -> None:
    model = _ToyCausalLM()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=_ToyTokenizer(),
        input_device=torch.device("cpu"),
    )
    prompt = "Human: Q\n\nAssistant: answer"
    baseline = evaluate_huggingface_probabilities(bundle, prompt, ("one", "multi"))
    np.testing.assert_allclose(baseline.probabilities["one"], 0.1, rtol=1e-5)
    np.testing.assert_allclose(baseline.probabilities["multi"], 0.01, rtol=1e-5)

    steered = evaluate_huggingface_probabilities(
        bundle,
        prompt,
        ("one", "multi"),
        steering_direction=np.asarray([1.0, 0.0]),
        layer=0,
    )
    assert steered.probabilities["one"] > baseline.probabilities["one"]
    # The two-token word includes the unchanged second-token factor of 0.1.
    np.testing.assert_allclose(
        steered.probabilities["multi"],
        steered.probabilities["one"] * 0.1,
        rtol=1e-5,
    )


def test_v3_does_not_treat_empty_or_punctuation_tokens_as_related() -> None:
    assert not token_is_emotion_related("excited", "")
    assert not token_is_emotion_related("excited", "!!!")
    assert not token_is_emotion_related("excited", "▁")
    assert token_is_emotion_related("excited", "▁thrilled")


def test_v1_requires_matching_vector_to_beat_nonmatching_control(tmp_path: Path) -> None:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    activations = np.zeros((20, 2), dtype=np.float32)
    activations[15] = vector * 10.0
    stories = [
        {"emotion": label, "story_id": label, "activations": activations}
        for label in ("excited", "elated")
    ]

    result = run_localization(
        stories,
        {"excited": vector, "elated": vector.copy()},
        output_dir=tmp_path,
        minimum_emotion_pass_rate=0.5,
        minimum_control_percentile=0.75,
        plot_formats=("png",),
        dpi=30,
    )

    assert not result.passed
    rows = (tmp_path / "story_concentration.csv").read_text(encoding="utf-8")
    assert "matching_control_percentile" in rows


def test_stage_four_resumes_validated_verifier_outputs(tmp_path: Path, monkeypatch) -> None:
    labels = np.asarray(("excited", "joyful", "sad", "gloomy"))
    vectors = np.asarray(
        [[1.0, 0.8], [1.0, 1.0], [-1.0, -0.8], [-1.0, -1.0]],
        dtype=np.float32,
    )
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    np.savez(
        vector_dir / "emotion_vectors.npz",
        raw=vectors[:, None, :],
        denoised=vectors[:, None, :],
        emotions=labels,
        primary_layer=np.asarray(0),
    )
    config = SimpleNamespace(
        paths=SimpleNamespace(
            vectors=vector_dir,
            outputs=tmp_path / "outputs",
            activations=tmp_path / "activations",
            raw=tmp_path / "raw",
        ),
        plot_format=("png", "svg"),
        dpi=45,
        kmeans_k=2,
        seed=0,
        resume=True,
        emotions=tuple(labels),
    )
    first = run_verifications(config, checks=("V4",))
    assert first["V4"].passed

    monkeypatch.setattr(
        "emotion_vectors.verify.clustering.run_clustering",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = run_verifications(config, checks=("V4",))
    assert second["V4"].passed
    assert second["V4"].metrics["resumed"] is True


def test_stage_four_recomputes_when_cached_evidence_is_modified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    labels = np.asarray(("excited", "joyful", "sad", "gloomy"))
    vectors = np.asarray(
        [[1.0, 0.8], [1.0, 1.0], [-1.0, -0.8], [-1.0, -1.0]],
        dtype=np.float32,
    )
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    np.savez(
        vector_dir / "emotion_vectors.npz",
        raw=vectors[:, None, :],
        denoised=vectors[:, None, :],
        emotions=labels,
        primary_layer=np.asarray(0),
    )
    config = SimpleNamespace(
        paths=SimpleNamespace(
            vectors=vector_dir,
            outputs=tmp_path / "outputs",
            activations=tmp_path / "activations",
            raw=tmp_path / "raw",
        ),
        plot_format=("png", "svg"),
        dpi=45,
        kmeans_k=2,
        seed=0,
        resume=True,
        emotions=tuple(labels),
    )
    first = run_verifications(config, checks=("V4",))
    report = first["V4"].report
    report.write_text(
        report.read_text(encoding="utf-8").replace("**PASS**", "**FAIL**"),
        encoding="utf-8",
    )

    from emotion_vectors.verify import clustering

    original = clustering.run_clustering
    calls = 0

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(clustering, "run_clustering", counted)
    second = run_verifications(config, checks=("V4",))
    assert calls == 1
    assert second["V4"].passed
    assert "VERDICT: **PASS**" in second["V4"].report.read_text(encoding="utf-8")


def test_verify_cli_rejects_duplicate_checks() -> None:
    with pytest.raises(SystemExit, match="2"):
        verify_main(["--checks", "V4,V4"])
