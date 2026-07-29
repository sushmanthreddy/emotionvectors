#!/usr/bin/env python3
"""Render a dynamic subset of emotion probes in an Anthropic-style grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .subset_score import HIDDEN_STATE_MAPPING, SCORE_SCHEMA_VERSION


PANEL_WIDTH = 1480
PANEL_HEIGHT = 590
OUTER_MARGIN = 44
COLUMN_GAP = 30
ROW_GAP = 28
TOP_MARGIN = 70
BOTTOM_MARGIN = 46


def build_parser() -> argparse.ArgumentParser:
    """Build the CPU-only dynamic subset renderer CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Render every selected emotion probe against every supplied "
            "passage using per-probe percentile calibration."
        )
    )
    parser.add_argument("--scores-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--highlight-percentile", type=float, default=90.0)
    parser.add_argument("--saturation-percentile", type=float, default=99.5)
    parser.add_argument("--sentences-per-paragraph", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render one validated score file without loading a model."""

    args = build_parser().parse_args(argv)
    try:
        render_subset(
            scores_jsonl=args.scores_jsonl.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
            metadata_path=args.metadata_output.expanduser().resolve(),
            highlight_percentile=args.highlight_percentile,
            saturation_percentile=args.saturation_percentile,
            sentences_per_paragraph=args.sentences_per_paragraph,
        )
    except Exception as error:
        print(
            f"Subset probe rendering failed: "
            f"{error.__class__.__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


def render_subset(
    *,
    scores_jsonl: Path,
    output_path: Path,
    metadata_path: Path,
    highlight_percentile: float,
    saturation_percentile: float,
    sentences_per_paragraph: int,
) -> dict[str, Any]:
    """Render ``records × selected probes`` panels from saved token scores."""

    if not (
        0.0 <= highlight_percentile < saturation_percentile <= 100.0
    ):
        raise ValueError(
            "Percentiles must satisfy 0 <= highlight < saturation <= 100"
        )
    if (
        isinstance(sentences_per_paragraph, bool)
        or not isinstance(sentences_per_paragraph, int)
        or sentences_per_paragraph < 1
        or 6 % sentences_per_paragraph
    ):
        raise ValueError(
            "--sentences-per-paragraph must be a positive divisor of six"
        )
    if output_path == metadata_path:
        raise ValueError("Image and metadata paths must differ")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Renderer outputs already exist")

    rows = load_and_validate_scores(scores_jsonl)
    probe_order = list(rows[0]["probe_order"])
    calibration = build_calibration(
        rows,
        probe_order=probe_order,
        highlight_percentile=highlight_percentile,
        saturation_percentile=saturation_percentile,
    )
    canvas_width = (
        2 * OUTER_MARGIN
        + len(probe_order) * PANEL_WIDTH
        + (len(probe_order) - 1) * COLUMN_GAP
    )
    canvas_height = (
        TOP_MARGIN
        + len(rows) * PANEL_HEIGHT
        + (len(rows) - 1) * ROW_GAP
        + BOTTOM_MARGIN
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    heading_font = load_font(32)
    title_font = load_font(27)
    token_font = load_font(23, monospaced=True)
    draw = ImageDraw.Draw(canvas)
    model = rows[0]["model"]
    layer_number = rows[0]["layer_number_one_based"]
    draw.text(
        (OUTER_MARGIN, 18),
        (
            f"Anthropic-style subset probes · {model} · "
            f"transformer layer {layer_number}"
        ),
        fill=(25, 25, 25),
        font=heading_font,
    )
    try:
        for row_index, row in enumerate(rows):
            for probe_index, emotion in enumerate(probe_order):
                panel = render_panel(
                    row=row,
                    emotion=emotion,
                    probe_index=probe_index,
                    calibration=calibration[emotion],
                    title_font=title_font,
                    token_font=token_font,
                    sentences_per_paragraph=sentences_per_paragraph,
                )
                try:
                    x = OUTER_MARGIN + probe_index * (
                        PANEL_WIDTH + COLUMN_GAP
                    )
                    y = TOP_MARGIN + row_index * (
                        PANEL_HEIGHT + ROW_GAP
                    )
                    canvas.paste(panel, (x, y))
                finally:
                    panel.close()
        save_png_atomic(canvas, output_path)
    finally:
        canvas.close()

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "scores_jsonl": str(scores_jsonl),
        "scores_sha256": sha256_file(scores_jsonl),
        "output_image": str(output_path),
        "output_sha256": sha256_file(output_path),
        "model": rows[0]["model"],
        "model_revision": rows[0]["model_revision"],
        "layer_index_zero_based": rows[0]["layer_index_zero_based"],
        "layer_number_one_based": rows[0]["layer_number_one_based"],
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
        "logits_used": False,
        "probe_order": probe_order,
        "number_of_probes": len(probe_order),
        "record_order": [row["example_id"] for row in rows],
        "number_of_records": len(rows),
        "panel_count": len(rows) * len(probe_order),
        "grid_definition": (
            "rows are input passages; columns are selected clean unit "
            "emotion-vector probes"
        ),
        "highlight_percentile": highlight_percentile,
        "saturation_percentile": saturation_percentile,
        "sentences_per_paragraph": sentences_per_paragraph,
        "calibration_scope": (
            "Each probe is calibrated independently using every eligible "
            "token across all supplied records."
        ),
        "position_zero_highlighted": False,
        "calibration": calibration,
        "image_dimensions": [canvas_width, canvas_height],
    }
    atomic_write_json(metadata_path, metadata)
    print(
        "SUBSET_PLOT_COMPLETE",
        f"records={len(rows)}",
        f"probes={len(probe_order)}",
        f"layer={rows[0]['layer_index_zero_based']}",
        f"output={output_path}",
    )
    return metadata


def load_and_validate_scores(path: Path) -> list[dict[str, Any]]:
    """Load dynamic score rows and validate common provenance and arrays."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid score JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"Score line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("Score JSONL contains no records")

    first = rows[0]
    probe_order = first.get("probe_order")
    if (
        not isinstance(probe_order, list)
        or len(probe_order) < 1
        or len(probe_order) != len(set(probe_order))
        or not all(isinstance(item, str) and item for item in probe_order)
    ):
        raise ValueError("Invalid dynamic probe order")
    common_fields = (
        "schema_version",
        "model",
        "model_revision",
        "resolved_model_revision",
        "resolved_tokenizer_revision",
        "layer_index_zero_based",
        "layer_number_one_based",
        "hidden_state_mapping",
        "probe_order",
    )
    expected_common = {field: first.get(field) for field in common_fields}
    if expected_common["schema_version"] != SCORE_SCHEMA_VERSION:
        raise ValueError("Unsupported score schema")
    if expected_common["hidden_state_mapping"] != HIDDEN_STATE_MAPPING:
        raise ValueError("Hidden-state mapping is incompatible")
    if expected_common["layer_number_one_based"] != (
        expected_common["layer_index_zero_based"] + 1
    ):
        raise ValueError("Layer indexing metadata is inconsistent")

    seen_ids: set[str] = set()
    for row in rows:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("Score row lacks example_id")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate score example {example_id!r}")
        for field, expected in expected_common.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Common score field {field!r} differs in {example_id}"
                )
        if row.get("logits_used") is not False:
            raise ValueError(f"Logits were used for {example_id}")
        if row.get("chat_template_used") is not False:
            raise ValueError(f"Chat template was used for {example_id}")
        primary = row.get("primary_emotion")
        secondary = row.get("secondary_emotion")
        if primary not in probe_order or primary == secondary:
            raise ValueError(f"Invalid emotion pair for {example_id}")
        if row.get("expected_emotions") != [primary, secondary]:
            raise ValueError(f"Invalid expected labels for {example_id}")
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Missing text for {example_id}")
        arrays = (
            row.get("token_ids"),
            row.get("token_strings"),
            row.get("token_offsets"),
            row.get("token_sentence_indices"),
            row.get("probe_scores"),
        )
        if (
            any(not isinstance(array, list) for array in arrays)
            or not arrays[0]
            or len({len(array) for array in arrays}) != 1
        ):
            raise ValueError(f"Mismatched token arrays for {example_id}")
        if not all(
            isinstance(values, list)
            and len(values) == len(probe_order)
            and all(math.isfinite(float(value)) for value in values)
            for values in row["probe_scores"]
        ):
            raise ValueError(f"Invalid probe scores for {example_id}")
        if not all(
            isinstance(offset, list)
            and len(offset) == 2
            and all(isinstance(value, int) for value in offset)
            and 0 <= offset[0] <= offset[1] <= len(text)
            for offset in row["token_offsets"]
        ):
            raise ValueError(f"Invalid token offsets for {example_id}")
        excluded = row.get("excluded_token_indices")
        if (
            not isinstance(excluded, list)
            or 0 not in excluded
            or len(excluded) != len(set(excluded))
            or not all(
                isinstance(index, int)
                and 0 <= index < len(row["token_ids"])
                for index in excluded
            )
        ):
            raise ValueError(f"Invalid exclusions for {example_id}")
        validate_sentences(row)
        seen_ids.add(example_id)
    if {row["primary_emotion"] for row in rows} != set(probe_order):
        raise ValueError("Every selected probe needs one primary passage")
    return rows


def validate_sentences(row: Mapping[str, Any]) -> None:
    """Validate exact sentence spans and token-to-sentence indices."""

    sentences = row.get("sentences")
    example_id = row["example_id"]
    if (
        not isinstance(sentences, list)
        or len(sentences) != 6
        or [sentence.get("sentence_index") for sentence in sentences]
        != list(range(6))
    ):
        raise ValueError(f"Expected six ordered sentences in {example_id}")
    pair = {row["primary_emotion"], row["secondary_emotion"]}
    for sentence in sentences:
        start = sentence.get("char_start")
        end = sentence.get("char_end")
        labels = sentence.get("expected_emotions")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(row["text"])
            or row["text"][start:end] != sentence.get("text")
            or not isinstance(labels, list)
            or not set(labels).issubset(pair)
        ):
            raise ValueError(f"Invalid sentence span in {example_id}")
    mappings = row["token_sentence_indices"]
    if not all(
        isinstance(index, int) and -1 <= index < 6 for index in mappings
    ):
        raise ValueError(f"Invalid token sentence mapping in {example_id}")


def build_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    probe_order: Sequence[str],
    highlight_percentile: float,
    saturation_percentile: float,
) -> dict[str, dict[str, float | int]]:
    """Calculate q90/q99.5-style thresholds independently per probe."""

    calibration: dict[str, dict[str, float | int]] = {}
    for probe_index, emotion in enumerate(probe_order):
        values = [
            float(row["probe_scores"][token_index][probe_index])
            for row in rows
            for token_index in eligible_token_indices(row)
        ]
        lower = percentile(values, highlight_percentile)
        upper = percentile(values, saturation_percentile)
        if not upper > lower:
            raise ValueError(f"Degenerate calibration for {emotion!r}")
        calibration[emotion] = {
            "eligible_token_count": len(values),
            "lower_percentile": highlight_percentile,
            "upper_percentile": saturation_percentile,
            "lower_score": lower,
            "upper_score": upper,
        }
    return calibration


def eligible_token_indices(row: Mapping[str, Any]) -> Iterable[int]:
    excluded = set(row["excluded_token_indices"])
    for index, (start, end) in enumerate(row["token_offsets"]):
        if index not in excluded and end > start:
            yield index


def render_panel(
    *,
    row: Mapping[str, Any],
    emotion: str,
    probe_index: int,
    calibration: Mapping[str, float | int],
    title_font: ImageFont.FreeTypeFont,
    token_font: ImageFont.FreeTypeFont,
    sentences_per_paragraph: int,
) -> Image.Image:
    """Render one selected probe over one passage."""

    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "white")
    draw = ImageDraw.Draw(panel)
    title = (
        f"{display_name(emotion)} probe · "
        f"{display_name(row['primary_emotion'])} passage"
    )
    draw.text((5, 2), title, fill=(30, 30, 30), font=title_font)
    box = (4, 48, PANEL_WIDTH - 5, PANEL_HEIGHT - 5)
    draw.rectangle(box, outline=(215, 215, 215), width=2)
    padding_x = 12
    padding_y = 12
    line_height = 38
    placements, content_height = layout_passage(
        row,
        font=token_font,
        content_width=box[2] - box[0] - 2 * padding_x,
        line_height=line_height,
        sentences_per_paragraph=sentences_per_paragraph,
        paragraph_gap=12,
    )
    if content_height > box[3] - box[1] - 2 * padding_y:
        panel.close()
        raise ValueError(f"Passage {row['example_id']} does not fit")
    excluded = set(row["excluded_token_indices"])
    low_color = (255, 246, 236)
    high_color = (249, 115, 22)
    for token_x, token_y, fragment, token_index in placements:
        score = float(row["probe_scores"][token_index][probe_index])
        intensity = (
            0.0
            if token_index in excluded
            else score_intensity(
                score,
                float(calibration["lower_score"]),
                float(calibration["upper_score"]),
            )
        )
        x = box[0] + padding_x + token_x
        y = box[1] + padding_y + token_y
        width = max(1, round(draw.textlength(fragment, font=token_font)))
        if intensity > 0:
            draw.rounded_rectangle(
                (x - 1, y - 2, x + width + 1, y + 30),
                radius=4,
                fill=blend_color(
                    low_color,
                    high_color,
                    0.16 + 0.84 * intensity,
                ),
            )
        draw.text((x, y), fragment, fill=(25, 25, 25), font=token_font)
    return panel


def layout_passage(
    row: Mapping[str, Any],
    *,
    font: ImageFont.FreeTypeFont,
    content_width: int,
    line_height: int,
    sentences_per_paragraph: int,
    paragraph_gap: int,
) -> tuple[list[tuple[int, int, str, int]], int]:
    """Lay out exact decoded token fragments with display-only paragraphs."""

    measure_image = Image.new("RGB", (content_width, 100), "white")
    measure = ImageDraw.Draw(measure_image)
    placements: list[tuple[int, int, str, int]] = []
    x = 0
    y = 0
    previous_sentence = -1
    for token_index, ((start, end), mapped_sentence) in enumerate(
        zip(
            row["token_offsets"],
            row["token_sentence_indices"],
            strict=True,
        )
    ):
        sentence_index = int(mapped_sentence)
        fragment = row["text"][start:end]
        if (
            sentence_index >= 0
            and sentence_index != previous_sentence
            and sentence_index > 0
            and sentence_index % sentences_per_paragraph == 0
        ):
            x = 0
            y += line_height + paragraph_gap
            fragment = fragment.lstrip()
        if sentence_index >= 0:
            previous_sentence = sentence_index
        if not fragment:
            continue
        width = max(1, round(measure.textlength(fragment, font=font)))
        if x + width > content_width and x > 0:
            fragment = fragment.lstrip()
            if not fragment:
                x = 0
                y += line_height
                continue
            width = max(1, round(measure.textlength(fragment, font=font)))
            x = 0
            y += line_height
        placements.append((x, y, fragment, token_index))
        x += width
    measure_image.close()
    return placements, y + line_height


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def score_intensity(score: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return min(max((score - lower) / (upper - lower), 0.0), 1.0)


def blend_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    amount = min(max(amount, 0.0), 1.0)
    return tuple(
        round(left + (right - left) * amount)
        for left, right in zip(start, end, strict=True)
    )


def display_name(emotion: str) -> str:
    return emotion.replace("_", " ").replace("-", " ").title()


def load_font(
    size: int,
    *,
    monospaced: bool = False,
) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSansMono.ttf" if monospaced else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/local/share/fonts") / filename,
        Path(
            "/opt/conda/lib/python3.11/site-packages/matplotlib/"
            f"mpl-data/fonts/ttf/{filename}"
        ),
        Path("/opt/conda/fonts") / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype(filename, size=size)


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
