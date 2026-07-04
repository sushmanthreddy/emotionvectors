"""Shared plotting style and reproducible figure/CSV output helpers."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .config import ExperimentConfig


DEFAULT_FORMATS = ("png", "svg")
DEFAULT_DPI = 150

PLOT_STYLE: dict[str, Any] = {
    "axes.axisbelow": True,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "axes.labelcolor": "#222222",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 10,
    "grid.alpha": 0.25,
    "grid.color": "#8a8a8a",
    "grid.linestyle": "--",
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
}


def apply_plot_style() -> None:
    """Apply the project-wide Matplotlib style to subsequent figures."""

    mpl.rcParams.update(PLOT_STYLE)


@contextmanager
def plot_style() -> Iterator[None]:
    """Temporarily apply the project-wide Matplotlib style."""

    with mpl.rc_context(PLOT_STYLE):
        yield


def save_figure(
    figure: Figure,
    output_base: str | Path,
    config_or_formats: ExperimentConfig | Sequence[str] | str | None = None,
    dpi: int | None = None,
    *,
    formats: Sequence[str] | str | None = None,
    close: bool = False,
    transparent: bool = False,
) -> tuple[Path, ...]:
    """Save a figure in every configured format and return written paths.

    Two call forms are supported so stages can pass their immutable config while
    focused plotting code can pass explicit settings::

        save_figure(fig, output_base, config)
        save_figure(fig, output_base, ("png", "svg"), 150)
    """

    config = config_or_formats if _looks_like_config(config_or_formats) else None
    if config is not None:
        if formats is not None:
            raise TypeError("formats cannot be supplied together with a config object")
        selected_formats = _normalize_formats(config.plot_format)
        selected_dpi = int(config.dpi if dpi is None else dpi)
        base = config.resolve_path(output_base)
    else:
        if formats is not None and config_or_formats is not None:
            raise TypeError("formats were provided both positionally and by keyword")
        explicit_formats = formats if formats is not None else config_or_formats
        selected_formats = _normalize_formats(explicit_formats or DEFAULT_FORMATS)
        selected_dpi = DEFAULT_DPI if dpi is None else int(dpi)
        base = Path(output_base).expanduser().resolve(strict=False)

    if selected_dpi <= 0:
        raise ValueError("dpi must be positive")
    if base.suffix.lower().lstrip(".") in selected_formats:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for file_format in selected_formats:
        destination = base.with_suffix(f".{file_format}")
        figure.savefig(
            destination,
            format=file_format,
            dpi=selected_dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
        written.append(destination)
    if close:
        plt.close(figure)
    return tuple(written)


def save_csv(
    path: str | Path,
    rows: Any,
    fieldnames: Sequence[str] | None = None,
    *,
    index: bool = False,
) -> Path:
    """Atomically write the numeric/source data underlying a figure.

    ``rows`` may be a pandas-like object with ``to_csv``, a sequence of mappings,
    a mapping of columns to sequences, or a sequence of row sequences.
    """

    destination = Path(path).expanduser().resolve(strict=False)
    if destination.suffix.lower() != ".csv":
        destination = destination.with_suffix(".csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if hasattr(rows, "to_csv") and callable(rows.to_csv):
            rows.to_csv(temporary, index=index)
        else:
            _write_csv_rows(temporary, rows, fieldnames)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


write_csv = save_csv
save_figure_data = save_csv


def save_figure_with_csv(
    figure: Figure,
    output_base: str | Path,
    rows: Any,
    config: ExperimentConfig,
    *,
    fieldnames: Sequence[str] | None = None,
    close: bool = False,
) -> tuple[tuple[Path, ...], Path]:
    """Write a configured figure and its sibling source-data CSV."""

    figures = save_figure(figure, output_base, config, close=close)
    base = config.resolve_path(output_base)
    csv_path = save_csv(base.with_suffix(".csv"), rows, fieldnames)
    return figures, csv_path


def _write_csv_rows(
    path: Path,
    rows: Any,
    fieldnames: Sequence[str] | None,
) -> None:
    normalized = _normalize_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if normalized and isinstance(normalized[0], Mapping):
            names = list(fieldnames) if fieldnames is not None else list(normalized[0].keys())
            if not names:
                raise ValueError("fieldnames cannot be empty")
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
            writer.writeheader()
            writer.writerows(normalized)
            return

        if fieldnames is not None:
            csv.writer(handle).writerow(fieldnames)
        csv.writer(handle).writerows(normalized)


def _normalize_rows(rows: Any) -> list[Any]:
    if isinstance(rows, Mapping):
        keys = list(rows)
        values = list(rows.values())
        if not values:
            return []
        sequence_columns = [
            _is_nonstring_iterable(value) and not isinstance(value, Mapping) for value in values
        ]
        if all(sequence_columns):
            columns = [list(value) for value in values]
            lengths = {len(column) for column in columns}
            if len(lengths) > 1:
                raise ValueError("CSV column mappings must have equal-length columns")
            return [
                dict(zip(keys, values_at_row, strict=True))
                for values_at_row in zip(*columns, strict=True)
            ]
        if any(sequence_columns):
            raise ValueError("CSV mappings cannot mix scalar and sequence columns")
        return [dict(rows)]

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise TypeError("rows must be an iterable or a mapping")
    materialized = list(rows)
    if not materialized:
        return []
    first_is_mapping = isinstance(materialized[0], Mapping)
    if any(isinstance(row, Mapping) != first_is_mapping for row in materialized):
        raise ValueError("CSV rows must all use the same row representation")
    return materialized


def _is_nonstring_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _normalize_formats(formats: Sequence[str] | str) -> tuple[str, ...]:
    raw_formats = (formats,) if isinstance(formats, str) else tuple(formats)
    normalized = tuple(file_format.lower().lstrip(".") for file_format in raw_formats)
    if not normalized:
        raise ValueError("at least one plot format is required")
    if any(not file_format or not file_format.isalnum() for file_format in normalized):
        raise ValueError("plot formats must be non-empty alphanumeric extensions")
    if len(normalized) != len(set(normalized)):
        raise ValueError("plot formats must not contain duplicates")
    return normalized


def _looks_like_config(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("plot_format", "dpi", "resolve_path"))


__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_FORMATS",
    "PLOT_STYLE",
    "apply_plot_style",
    "plot_style",
    "save_csv",
    "save_figure",
    "save_figure_data",
    "save_figure_with_csv",
    "write_csv",
]
