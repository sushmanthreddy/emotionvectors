"""Render the released Anthropic-style emotion-vector paragraph figure.

The renderer consumes decoded token-score records.  It does not load a model or
compute activations.  Each panel uses the clean vector matching the passage's
primary emotion, while color thresholds are calibrated independently for each
probe across every eligible token in the supplied records.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from ..constants import (
    EMOTIONS,
    HIDDEN_STATE_MAPPING,
    MODEL_ID,
    MODEL_REVISION,
    VISUALIZATION_LAYER_INDEX,
    VISUALIZATION_LAYER_NUMBER,
)

DISPLAY_NAMES = {
    "anger": "Angry",
    "fear": "Afraid",
    "disgust": "Disgusted",
    "sadness": "Sad",
    "anxiety": "Anxious",
    "desperation": "Desperate",
    "frustration": "Frustrated",
    "hostility": "Hostile",
    "calmness": "Calm",
    "compassion": "Compassionate",
    "joy": "Joyful",
    "trust": "Trusting",
}

MODEL_NAME = MODEL_ID
LAYER_INDEX_ZERO_BASED = VISUALIZATION_LAYER_INDEX
LAYER_NUMBER_ONE_BASED = VISUALIZATION_LAYER_NUMBER
SCORE_SCHEMA_VERSION = "long_mixed_emotion_token_scores_v1"

_CANVAS_WIDTH = 3350
_OUTER_MARGIN = 58
_COLUMN_GAP = 42
_ROW_GAP = 24
_PANEL_HEIGHT = 540
_TOP_MARGIN = 18
_BOTTOM_MARGIN = 82
_IMAGE_METADATA_KEY = "emotionvectors"


def load_score_records(path: str | Path) -> list[dict[str, Any]]:
    """Load decoded token-score mappings from a JSONL file."""
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source} on line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {source} on line {line_number}"
                )
            records.append(value)
    if not records:
        raise ValueError(f"No score records found in {source}")
    return records


def get_plot_metadata(image: Image.Image) -> dict[str, Any]:
    """Return a detached copy of metadata created by this renderer."""
    metadata = image.info.get(_IMAGE_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("Image was not created by plot_anthropic_style_paragraphs")
    return json.loads(json.dumps(metadata, ensure_ascii=False))


def save_plot_metadata(metadata: Mapping[str, Any], path: str | Path) -> None:
    """Atomically save JSON-serializable plot metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _percentile(values: Sequence[float], percentage: float) -> float:
    """Calculate a linearly interpolated percentile without NumPy."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _blend_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    amount = min(max(amount, 0.0), 1.0)
    return tuple(
        round(start_channel + (end_channel - start_channel) * amount)
        for start_channel, end_channel in zip(start, end)
    )


def _score_intensity(score: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return min(max((score - lower) / (upper - lower), 0.0), 1.0)


def _load_font(
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


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _validate_sentence_schema(row: Mapping[str, Any]) -> None:
    example_id = row["example_id"]
    primary = row["primary_emotion"]
    secondary = row["secondary_emotion"]
    text = row["text"]
    sentences = row.get("sentences")
    mappings = row.get("token_sentence_indices")

    if not isinstance(sentences, list) or len(sentences) != 6:
        raise ValueError(f"Expected six sentences in {example_id}")
    if [sentence.get("sentence_index") for sentence in sentences] != list(
        range(6)
    ):
        raise ValueError(f"Nonsequential sentence indices in {example_id}")
    if not (
        isinstance(mappings, list)
        and len(mappings) == len(row["token_ids"])
        and all(isinstance(index, int) and -1 <= index < 6 for index in mappings)
    ):
        raise ValueError(f"Invalid token-to-sentence map in {example_id}")

    for sentence in sentences:
        start = sentence.get("char_start")
        end = sentence.get("char_end")
        labels = sentence.get("expected_emotions")
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
            and text[start:end] == sentence.get("text")
        ):
            raise ValueError(f"Invalid sentence span in {example_id}")
        if not (
            isinstance(labels, list)
            and len(labels) == len(set(labels))
            and set(labels).issubset({primary, secondary})
        ):
            raise ValueError(f"Invalid sentence labels in {example_id}")


def _validate_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if not records:
        raise ValueError("At least one score record is required")

    validated: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    primary_counts: Counter[str] = Counter()
    expected_provenance = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": MODEL_REVISION,
        "resolved_tokenizer_revision": MODEL_REVISION,
        "layer_index_zero_based": LAYER_INDEX_ZERO_BASED,
        "layer_number_one_based": LAYER_NUMBER_ONE_BASED,
        "hidden_state_mapping": HIDDEN_STATE_MAPPING,
    }

    for record_index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise TypeError(f"Record {record_index} is not a mapping")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"Missing example_id in record {record_index}")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate example_id {example_id}")
        for key, expected in expected_provenance.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Unexpected {key} for {example_id}: "
                    f"{row.get(key)!r} != {expected!r}"
                )
        if tuple(row.get("probe_order", ())) != EMOTIONS:
            raise ValueError(f"Unexpected probe order for {example_id}")

        primary = row.get("primary_emotion")
        secondary = row.get("secondary_emotion")
        if primary not in EMOTIONS or secondary not in EMOTIONS:
            raise ValueError(f"Invalid emotion pair for {example_id}")
        if primary == secondary:
            raise ValueError(f"Matching primary and secondary in {example_id}")
        if row.get("expected_emotions") != [primary, secondary]:
            raise ValueError(f"Invalid expected emotion order in {example_id}")

        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Missing text in {example_id}")
        token_ids = row.get("token_ids")
        token_strings = row.get("token_strings")
        token_offsets = row.get("token_offsets")
        probe_scores = row.get("probe_scores")
        arrays = (token_ids, token_strings, token_offsets, probe_scores)
        if any(not isinstance(array, list) for array in arrays):
            raise ValueError(f"Missing token array in {example_id}")
        if not token_ids or len({len(array) for array in arrays}) != 1:
            raise ValueError(f"Mismatched token arrays in {example_id}")
        if not all(
            _is_sequence(scores)
            and len(scores) == len(EMOTIONS)
            and all(math.isfinite(float(score)) for score in scores)
            for scores in probe_scores
        ):
            raise ValueError(f"Invalid probe scores in {example_id}")

        token_count = len(token_ids)
        excluded = row.get("excluded_token_indices")
        if not (
            isinstance(excluded, list)
            and len(excluded) == len(set(excluded))
            and all(
                isinstance(index, int) and 0 <= index < token_count
                for index in excluded
            )
            and 0 in excluded
        ):
            raise ValueError(f"Invalid excluded indices in {example_id}")
        if not all(
            _is_sequence(offset)
            and len(offset) == 2
            and all(isinstance(value, int) for value in offset)
            and 0 <= offset[0] <= offset[1] <= len(text)
            for offset in token_offsets
        ):
            raise ValueError(f"Invalid token offsets in {example_id}")
        if (
            token_offsets[0][0] != 0
            or token_offsets[-1][1] != len(text)
            or any(
                current[1] != following[0]
                for current, following in zip(token_offsets, token_offsets[1:])
            )
        ):
            raise ValueError(
                f"Token offsets do not exactly partition text in {example_id}"
            )

        _validate_sentence_schema(row)
        validated.append(row)
        seen_ids.add(example_id)
        primary_counts[primary] += 1

    missing = [emotion for emotion in EMOTIONS if not primary_counts[emotion]]
    if missing:
        raise ValueError(
            "Missing primary-emotion records for: " + ", ".join(missing)
        )
    return validated


def _eligible_token_indices(row: Mapping[str, Any]) -> Iterable[int]:
    excluded = set(int(index) for index in row["excluded_token_indices"])
    for token_index, offset in enumerate(row["token_offsets"]):
        if token_index not in excluded and int(offset[1]) > int(offset[0]):
            yield token_index


def _build_calibration(
    rows: Sequence[Mapping[str, Any]],
    highlight_percentile: float,
    saturation_percentile: float,
) -> dict[str, dict[str, float | int]]:
    calibration: dict[str, dict[str, float | int]] = {}
    for probe_index, emotion in enumerate(EMOTIONS):
        values = [
            float(row["probe_scores"][token_index][probe_index])
            for row in rows
            for token_index in _eligible_token_indices(row)
        ]
        lower = _percentile(values, highlight_percentile)
        upper = _percentile(values, saturation_percentile)
        if not upper > lower:
            raise ValueError(f"Degenerate calibration for {emotion}")
        calibration[emotion] = {
            "eligible_token_count": len(values),
            "highlight_percentile": highlight_percentile,
            "saturation_percentile": saturation_percentile,
            "lower_score": lower,
            "upper_score": upper,
        }
    return calibration


def _layout_passage(
    row: Mapping[str, Any],
    font: ImageFont.FreeTypeFont,
    content_width: int,
    line_height: int,
    sentences_per_paragraph: int,
    paragraph_gap: int,
) -> tuple[list[tuple[int, int, str, int]], int]:
    """Lay out exact token fragments with display-only paragraph breaks."""
    measure_image = Image.new("RGB", (content_width, 100), "white")
    measure = ImageDraw.Draw(measure_image)
    placements: list[tuple[int, int, str, int]] = []
    x = 0
    y = 0
    previous_sentence = -1
    text = row["text"]
    for token_index, ((start, end), mapped_sentence) in enumerate(
        zip(row["token_offsets"], row["token_sentence_indices"])
    ):
        sentence_index = int(mapped_sentence)
        fragment = text[int(start) : int(end)]
        new_paragraph = (
            sentence_index >= 0
            and sentence_index != previous_sentence
            and sentence_index > 0
            and sentence_index % sentences_per_paragraph == 0
        )
        if new_paragraph:
            x = 0
            y += line_height + paragraph_gap
            fragment = fragment.lstrip()
        if sentence_index >= 0:
            previous_sentence = sentence_index
        if not fragment:
            continue
        token_width = max(1, round(measure.textlength(fragment, font=font)))
        if x + token_width > content_width and x > 0:
            fragment = fragment.lstrip()
            if not fragment:
                x = 0
                y += line_height
                continue
            token_width = max(1, round(measure.textlength(fragment, font=font)))
            x = 0
            y += line_height
        placements.append((x, y, fragment, token_index))
        x += token_width
    measure_image.close()
    return placements, y + line_height


def _render_panel(
    row: Mapping[str, Any],
    emotion: str,
    calibration: Mapping[str, float | int],
    panel_width: int,
    title_font: ImageFont.FreeTypeFont,
    token_font: ImageFont.FreeTypeFont,
    sentences_per_paragraph: int,
) -> Image.Image:
    panel = Image.new("RGB", (panel_width, _PANEL_HEIGHT), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((0, 0), DISPLAY_NAMES[emotion], fill=(30, 30, 30), font=title_font)

    box_left = 0
    box_top = 58
    box_right = panel_width - 1
    box_bottom = _PANEL_HEIGHT - 1
    draw.rectangle(
        (box_left, box_top, box_right, box_bottom),
        outline=(218, 218, 218),
        width=2,
        fill=(255, 255, 255),
    )
    padding_x = 7
    padding_y = 9
    line_height = 41
    placements, final_y = _layout_passage(
        row,
        token_font,
        box_right - box_left - 2 * padding_x,
        line_height,
        sentences_per_paragraph,
        paragraph_gap=12,
    )
    available_height = box_bottom - box_top - 2 * padding_y
    if final_y > available_height:
        panel.close()
        raise ValueError(
            f"Passage {row['example_id']} does not fit: "
            f"required={final_y}, available={available_height}"
        )

    excluded = set(int(index) for index in row["excluded_token_indices"])
    probe_index = EMOTIONS.index(emotion)
    low_color = (255, 246, 236)
    high_color = (249, 115, 22)
    origin_x = box_left + padding_x
    origin_y = box_top + padding_y
    for token_x, token_y, fragment, token_index in placements:
        score = float(row["probe_scores"][token_index][probe_index])
        intensity = (
            0.0
            if token_index in excluded
            else _score_intensity(
                score,
                float(calibration["lower_score"]),
                float(calibration["upper_score"]),
            )
        )
        absolute_x = origin_x + token_x
        absolute_y = origin_y + token_y
        token_width = max(1, round(draw.textlength(fragment, font=token_font)))
        if intensity > 0.0:
            draw.rounded_rectangle(
                (
                    absolute_x - 1,
                    absolute_y - 2,
                    absolute_x + token_width + 1,
                    absolute_y + 32,
                ),
                radius=4,
                fill=_blend_color(
                    low_color,
                    high_color,
                    0.16 + 0.84 * intensity,
                ),
            )
        draw.text(
            (absolute_x, absolute_y),
            fragment,
            fill=(25, 25, 25),
            font=token_font,
        )
    return panel


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        image.save(temporary, format="PNG", optimize=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plot_anthropic_style_paragraphs(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_index: int = 1,
    output_path: str | Path | None = None,
    highlight_percentile: float = 90.0,
    saturation_percentile: float = 99.5,
    sentences_per_paragraph: int = 2,
) -> Image.Image:
    """Render one 12-panel paragraph figure from decoded probe-score records.

    ``sample_index`` is zero-based within records sharing a primary emotion, so
    the default value ``1`` selects the released ``sample_02`` page.  Record
    selection uses stable ``example_id`` order and never uses probe scores.

    The returned RGB image remains open and belongs to the caller.  Plot
    provenance, layer details, selected examples, and color calibration are
    available through :func:`get_plot_metadata`.  When ``output_path`` is set,
    the same image is also saved atomically as a PNG.
    """
    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
        raise TypeError("sample_index must be an integer")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    if not (
        0.0 <= highlight_percentile < saturation_percentile <= 100.0
    ):
        raise ValueError(
            "Percentiles must satisfy 0 <= highlight < saturation <= 100"
        )
    if (
        not isinstance(sentences_per_paragraph, int)
        or isinstance(sentences_per_paragraph, bool)
        or sentences_per_paragraph < 1
        or 6 % sentences_per_paragraph
    ):
        raise ValueError(
            "sentences_per_paragraph must be a positive divisor of six"
        )

    rows = _validate_records(records)
    by_primary = {
        emotion: sorted(
            (row for row in rows if row["primary_emotion"] == emotion),
            key=lambda row: row["example_id"],
        )
        for emotion in EMOTIONS
    }
    unavailable = [
        emotion
        for emotion, emotion_rows in by_primary.items()
        if sample_index >= len(emotion_rows)
    ]
    if unavailable:
        raise ValueError(
            f"sample_index {sample_index} is unavailable for: "
            + ", ".join(unavailable)
        )
    selected_rows = [by_primary[emotion][sample_index] for emotion in EMOTIONS]
    if len({row["text"] for row in selected_rows}) != len(EMOTIONS):
        raise ValueError("Every emotion panel must use a different passage")

    calibration = _build_calibration(
        rows,
        highlight_percentile,
        saturation_percentile,
    )
    columns = 2
    row_count = 6
    panel_width = (
        _CANVAS_WIDTH - 2 * _OUTER_MARGIN - (columns - 1) * _COLUMN_GAP
    ) // columns
    canvas_height = (
        _TOP_MARGIN
        + row_count * _PANEL_HEIGHT
        + (row_count - 1) * _ROW_GAP
        + _BOTTOM_MARGIN
    )
    canvas = Image.new("RGB", (_CANVAS_WIDTH, canvas_height), "white")
    title_font = _load_font(36)
    token_font = _load_font(25, monospaced=True)
    try:
        for index, (row, emotion) in enumerate(zip(selected_rows, EMOTIONS)):
            panel = _render_panel(
                row,
                emotion,
                calibration[emotion],
                panel_width,
                title_font,
                token_font,
                sentences_per_paragraph,
            )
            try:
                grid_row = index // columns
                column = index % columns
                x = _OUTER_MARGIN + column * (panel_width + _COLUMN_GAP)
                y = _TOP_MARGIN + grid_row * (_PANEL_HEIGHT + _ROW_GAP)
                canvas.paste(panel, (x, y))
            finally:
                panel.close()

        metadata: dict[str, Any] = {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "layer_index_zero_based": LAYER_INDEX_ZERO_BASED,
            "layer_number_one_based": LAYER_NUMBER_ONE_BASED,
            "hidden_state_mapping": HIDDEN_STATE_MAPPING,
            "logits_used": False,
            "probe_order": list(EMOTIONS),
            "score_schema_version": SCORE_SCHEMA_VERSION,
            "record_count": len(rows),
            "sample_index_zero_based": sample_index,
            "sample_number_one_based": sample_index + 1,
            "highlight_percentile": highlight_percentile,
            "saturation_percentile": saturation_percentile,
            "sentences_per_paragraph": sentences_per_paragraph,
            "visual_paragraphs_per_panel": 6 // sentences_per_paragraph,
            "calibration_scope": (
                "For each emotion probe independently, percentile thresholds "
                "are calculated from every eligible token in all supplied records."
            ),
            "position_zero_policy": (
                "Scorer-excluded tokens are omitted from calibration and shown "
                "without a highlight."
            ),
            "selection_rule": (
                "Select one record per primary emotion by stable example_id order; "
                "selection does not use probe scores."
            ),
            "visual_paragraph_rule": (
                "The source text and token scores are unchanged; paragraph breaks "
                "are display-only and inserted after each configured sentence group."
            ),
            "calibration": calibration,
            "panels": [
                {
                    "emotion": emotion,
                    "display_name": DISPLAY_NAMES[emotion],
                    "example_id": row["example_id"],
                    "primary_emotion": row["primary_emotion"],
                    "secondary_emotion": row["secondary_emotion"],
                    "sentence_count": len(row["sentences"]),
                    "word_count": len(row["text"].split()),
                }
                for emotion, row in zip(EMOTIONS, selected_rows)
            ],
            "image_width": _CANVAS_WIDTH,
            "image_height": canvas_height,
        }
        canvas.info[_IMAGE_METADATA_KEY] = metadata
        if output_path is not None:
            _save_png_atomic(canvas, Path(output_path))
        return canvas
    except Exception:
        canvas.close()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Render one figure from precomputed JSONL scores without loading a model."""

    parser = argparse.ArgumentParser(
        description="Render the 12-emotion Anthropic-style paragraph grid."
    )
    parser.add_argument("--scores-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--highlight-percentile", type=float, default=90.0)
    parser.add_argument("--saturation-percentile", type=float, default=99.5)
    parser.add_argument("--sentences-per-paragraph", type=int, default=2)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args(argv)

    image = plot_anthropic_style_paragraphs(
        load_score_records(args.scores_jsonl),
        sample_index=args.sample_index,
        output_path=args.output,
        highlight_percentile=args.highlight_percentile,
        saturation_percentile=args.saturation_percentile,
        sentences_per_paragraph=args.sentences_per_paragraph,
    )
    try:
        metadata = get_plot_metadata(image)
        if args.metadata_output is not None:
            save_plot_metadata(metadata, args.metadata_output)
    finally:
        image.close()

    print(
        f"saved {args.output} using transformer layer "
        f"{metadata['layer_number_one_based']} "
        f"(zero-based index {metadata['layer_index_zero_based']})"
    )
    return 0


__all__ = [
    "DISPLAY_NAMES",
    "EMOTIONS",
    "HIDDEN_STATE_MAPPING",
    "LAYER_INDEX_ZERO_BASED",
    "LAYER_NUMBER_ONE_BASED",
    "MODEL_NAME",
    "MODEL_REVISION",
    "get_plot_metadata",
    "load_score_records",
    "main",
    "plot_anthropic_style_paragraphs",
    "save_plot_metadata",
]


if __name__ == "__main__":
    raise SystemExit(main())
