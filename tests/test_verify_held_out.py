from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emotion_vectors.verify.held_out import run_held_out


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v2_persists_document_maxima_and_sparse_token_hits(tmp_path: Path) -> None:
    labels = ("calm", "sad")
    projection_blocks = [
        np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32),
        np.asarray([[1.1, -1.2], [1.3, 1.4], [1.5, -1.6]], dtype=np.float32),
        np.asarray([[2.1, -2.2]], dtype=np.float32),
        np.asarray(
            [[3.1, -3.2], [3.3, 3.4], [3.5, -3.6], [3.7, 3.8]],
            dtype=np.float32,
        ),
    ]
    documents = (
        {
            "dataset": "synthetic",
            "document_id": str(index),
            "text": "calm and sad context",
            "tokens": [f"token_{token}" for token in range(block.shape[0])],
            "projections": block,
        }
        for index, block in enumerate(projection_blocks)
    )
    percentile = 37.5
    config = SimpleNamespace(
        activation_percentile=percentile,
        heldout_max_docs=100,
        use_kernels=False,
        plot_format=("png",),
        dpi=40,
        v2_top_documents=1,
        v2_min_emotion_pass_rate=0.0,
    )

    result = run_held_out(
        documents,
        {"calm": np.asarray([1.0, 0.0]), "sad": np.asarray([0.0, 1.0])},
        config=config,
        output_dir=tmp_path,
        expected_terms={"calm": ("not-present",), "sad": ("not-present",)},
    )

    joined = np.concatenate(projection_blocks, axis=0)
    expected_thresholds = np.percentile(joined, percentile, axis=0)
    expected_offsets = np.asarray([0, 2, 5, 6, 10], dtype=np.int64)
    expected_maxima = np.stack([block.max(axis=0) for block in projection_blocks])
    with np.load(tmp_path / "document_projections.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "max_projections",
            "document_offsets",
            "thresholds",
            "emotions",
        }
        np.testing.assert_array_equal(archive["document_offsets"], expected_offsets)
        np.testing.assert_array_equal(archive["emotions"], np.asarray(labels))
        np.testing.assert_allclose(archive["max_projections"], expected_maxima)
        np.testing.assert_allclose(archive["thresholds"], expected_thresholds, rtol=0.0, atol=1e-12)

    expected_token, expected_emotion = np.nonzero(joined > expected_thresholds[None, :])
    with np.load(tmp_path / "above_threshold_indices.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"global_token_index", "emotion_index"}
        np.testing.assert_array_equal(archive["global_token_index"], expected_token)
        np.testing.assert_array_equal(archive["emotion_index"], expected_emotion)

    document_rows = _csv_rows(tmp_path / "document_projections.csv")
    assert len(document_rows) == len(projection_blocks) * len(labels)
    assert {row["document_id"] for row in document_rows} == {"0", "1", "2", "3"}
    assert {row["emotion"] for row in document_rows} == set(labels)
    for row in document_rows:
        document_index = int(row["document_index"])
        emotion_index = labels.index(row["emotion"])
        assert float(row["max_projection"]) == expected_maxima[document_index, emotion_index]

    top_rows = _csv_rows(tmp_path / "top_snippets.csv")
    assert len(top_rows) == len(labels)  # v2_top_documents came from config.
    assert document_rows != top_rows
    histogram_rows = _csv_rows(tmp_path / "projection_histogram.csv")
    for emotion in labels:
        assert (
            sum(int(row["count"]) for row in histogram_rows if row["emotion"] == emotion)
            == joined.shape[0]
        )

    assert not (tmp_path / "token_projections.npz").exists()
    assert not list(tmp_path.glob(".v2-projections-*"))
    assert result.metrics["n_documents"] == len(projection_blocks)
    assert result.metrics["n_tokens"] == joined.shape[0]
    assert result.metrics["emotion_pass_rate"] == 0.0
    assert result.passed  # v2_min_emotion_pass_rate=0 came from config.
    assert "VERDICT: **" in result.report.read_text(encoding="utf-8")
    assert all(figure.with_suffix(".csv").is_file() for figure in result.figures)
