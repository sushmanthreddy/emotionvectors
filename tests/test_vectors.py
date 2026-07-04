from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emotion_vectors.vectors import (
    build_vectors,
    choose_primary_layer,
    fit_and_project_neutral_pca,
    fit_neutral_pca,
    mean_center_emotion_means,
    project_out_components,
)


def test_emotion_means_are_centered_across_emotions() -> None:
    values = np.arange(3 * 4 * 2 * 5, dtype=np.float32).reshape(3, 4, 2, 5)
    raw = mean_center_emotion_means(values)
    np.testing.assert_allclose(raw.mean(axis=0), 0.0, atol=1e-6)
    expected_difference = values[0].mean(axis=0) - values[1].mean(axis=0)
    np.testing.assert_allclose(raw[0] - raw[1], expected_difference, atol=1e-6)


def test_pca_project_out_is_orthogonal_to_retained_components() -> None:
    rng = np.random.default_rng(2)
    neutral = rng.normal(size=(20, 3, 8)).astype(np.float32)
    pca = fit_neutral_pca(neutral, 0.5)
    vectors = rng.normal(size=(5, 3, 8)).astype(np.float32)
    cleaned = project_out_components(vectors, pca.components)
    for layer, components in enumerate(pca.components):
        assert components.shape[0] == pca.n_components[layer]
        np.testing.assert_allclose(cleaned[:, layer] @ components.T, 0.0, atol=2e-6)


def test_layerwise_pca_path_matches_reference_projection() -> None:
    rng = np.random.default_rng(17)
    neutral = rng.normal(size=(12, 2, 6)).astype(np.float32)
    vectors = rng.normal(size=(4, 2, 6)).astype(np.float32)
    reference_pca = fit_neutral_pca(neutral, 0.5)
    reference = project_out_components(vectors, reference_pca.components)
    actual, summary = fit_and_project_neutral_pca(vectors, neutral, 0.5, device="cpu")
    np.testing.assert_allclose(actual, reference, atol=2e-5)
    np.testing.assert_array_equal(summary.n_components, reference_pca.n_components)


def test_primary_layer_rounds_two_thirds_and_stays_in_bounds() -> None:
    assert choose_primary_layer(64, 2 / 3) == 43
    assert choose_primary_layer(1, 2 / 3) == 0
    assert choose_primary_layer(4, 1.0) == 3


def test_stage_three_contract_diagnostics_and_resume(tmp_path: Path, monkeypatch) -> None:
    activations = tmp_path / "activations"
    stories = activations / "stories"
    stories.mkdir(parents=True)
    rng = np.random.default_rng(9)
    sad = rng.normal(size=(5, 3, 4)).astype(np.float32)
    calm = rng.normal(size=(4, 3, 4)).astype(np.float32)
    neutral = rng.normal(size=(8, 3, 4)).astype(np.float32)
    np.savez(stories / "sad.npz", vectors=sad)
    np.savez(stories / "calm.npz", vectors=calm)
    np.savez(activations / "neutral.npz", vectors=neutral)
    config = SimpleNamespace(
        paths=SimpleNamespace(
            activations=activations,
            vectors=tmp_path / "vectors",
            outputs=tmp_path / "outputs",
        ),
        emotions=("sad", "calm"),
        pca_variance=0.5,
        pca_device="cpu",
        primary_layer_frac=2 / 3,
        plot_format=("png", "svg"),
        dpi=60,
        resume=True,
        resolve_path=lambda path: Path(path),
    )

    first = build_vectors(config)  # type: ignore[arg-type]
    with np.load(first.output_path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"raw", "denoised", "emotions", "primary_layer"}
        assert artifact["raw"].shape == (2, 3, 4)
        np.testing.assert_allclose(artifact["raw"].mean(axis=0), 0.0, atol=1e-6)
        assert artifact["emotions"].tolist() == ["sad", "calm"]
    diagnostics = tmp_path / "outputs/diagnostics"
    for stem in ("neutral_pca_scree", "raw_vs_denoised_cosine"):
        assert (diagnostics / f"{stem}.csv").is_file()
        assert (diagnostics / f"{stem}.png").is_file()
        assert (diagnostics / f"{stem}.svg").is_file()

    monkeypatch.setattr(
        "emotion_vectors.vectors.fit_and_project_neutral_pca",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    resumed = build_vectors(config)  # type: ignore[arg-type]
    np.testing.assert_array_equal(resumed.denoised, first.denoised)
    np.testing.assert_array_equal(resumed.pca.n_components, first.pca.n_components)
