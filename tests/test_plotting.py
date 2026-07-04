from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from emotion_vectors.config import load_config
from emotion_vectors.plotting import (
    plot_style,
    save_csv,
    save_figure,
    save_figure_with_csv,
)

plt.switch_backend("Agg")


def test_save_figure_accepts_config_and_explicit_settings(tmp_path: Path) -> None:
    config = load_config(
        overrides={"paths.outputs": tmp_path, "plot_format": ["png", "svg"], "dpi": 72}
    )
    with plot_style():
        figure, axis = plt.subplots()
        axis.plot([0, 1], [1, 0])

    written = save_figure(figure, tmp_path / "configured", config)
    assert written == (tmp_path / "configured.png", tmp_path / "configured.svg")
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)

    explicit = save_figure(figure, tmp_path / "explicit", ("png",), 72, close=True)
    assert explicit == (tmp_path / "explicit.png",)


def test_save_csv_supports_rows_and_column_mappings(tmp_path: Path) -> None:
    row_path = save_csv(tmp_path / "rows", [{"emotion": "calm", "score": 0.5}])
    with row_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"emotion": "calm", "score": "0.5"}]

    column_path = save_csv(tmp_path / "columns.csv", {"layer": [0, 1], "norm": [1.2, 1.5]})
    with column_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"layer": "0", "norm": "1.2"},
            {"layer": "1", "norm": "1.5"},
        ]


def test_save_figure_with_csv_writes_sibling_artifacts(tmp_path: Path) -> None:
    config = load_config(overrides={"paths.outputs": tmp_path, "plot_format": ["png"]})
    figure, axis = plt.subplots()
    axis.bar(["calm"], [0.7])

    figures, source = save_figure_with_csv(
        figure,
        tmp_path / "scores",
        [{"emotion": "calm", "score": 0.7}],
        config,
        close=True,
    )
    assert figures == (tmp_path / "scores.png",)
    assert source == tmp_path / "scores.csv"
