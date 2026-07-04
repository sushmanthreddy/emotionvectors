"""Final report generation for the reduced RQ1 experiment."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .em_direction import (
    CANONICAL_ADAPTER_RANK,
    CANONICAL_SOURCE_SCRIPT,
    CANONICAL_TARGET_ID,
    CANONICAL_TOKEN_AGGREGATION,
)
from .figures import FigureArtifact, create_all_figures, load_metrics

Conclusion = Literal["positive", "null", "inconclusive"]
RunStatus = Literal["complete", "inconclusive", "blocked"]

SUMMARY_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "study_id",
        "status",
        "conclusion",
        "quality_gate",
        "primary_evidence",
        "counts",
        "models",
        "em",
        "limitations",
        "metrics_csv",
        "metrics_parquet",
        "verdict_reason",
    }
)


@dataclass(frozen=True, slots=True)
class ReportStageResult:
    """Files and verdict emitted by :func:`run_report_stage`."""

    status: RunStatus
    conclusion: Conclusion
    report_path: Path
    metrics_path: Path | None
    figures: tuple[FigureArtifact, ...]
    scientific_conclusion_available: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["report_path"] = str(self.report_path)
        payload["metrics_path"] = str(self.metrics_path) if self.metrics_path is not None else None
        payload["figures"] = [
            {
                "name": figure.name,
                "png_path": str(figure.png_path),
                "svg_path": str(figure.svg_path),
                "csv_path": str(figure.csv_path),
            }
            for figure in self.figures
        ]
        return payload


def _config_value(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for component in path.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                return default
            value = value[component]
        elif hasattr(value, component):
            value = getattr(value, component)
        else:
            return default
    return value


def _results_dir(config: Any) -> Path:
    value = _config_value(config, "paths.results_dir")
    if value is None:
        value = _config_value(config, "results_dir")
    if value is None:
        raise ValueError("configuration does not define paths.results_dir")
    return Path(value).resolve()


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read analysis summary {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("analysis_summary.json must contain one JSON object")
    missing = sorted(SUMMARY_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(f"analysis_summary.json is missing required fields: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError(
            f"unsupported analysis summary schema_version: {payload['schema_version']!r}"
        )
    status = payload["status"]
    conclusion = payload["conclusion"]
    if status not in {"complete", "inconclusive", "blocked"}:
        raise ValueError(f"unsupported analysis status: {status!r}")
    if conclusion not in {"positive", "null", "inconclusive"}:
        raise ValueError(f"unsupported conclusion: {conclusion!r}")
    if status in {"blocked", "inconclusive"} and conclusion != "inconclusive":
        raise ValueError(f"analysis status {status!r} requires conclusion='inconclusive'")
    for key in ("quality_gate", "primary_evidence", "counts", "models", "em"):
        if not isinstance(payload[key], Mapping):
            raise ValueError(f"analysis summary field {key!r} must be an object")
    if not isinstance(payload["verdict_reason"], str) or not payload["verdict_reason"].strip():
        raise ValueError("analysis summary verdict_reason must be a non-empty string")
    limitations = payload["limitations"]
    if not isinstance(limitations, (str, list, tuple, Mapping)):
        raise ValueError("analysis summary limitations must be text, a list, or an object")
    return payload


def _validate_study_identity(config: Any, summary: Mapping[str, Any]) -> None:
    configured = _config_value(config, "study_id")
    if configured is not None and summary["study_id"] != configured:
        raise ValueError(
            f"analysis summary study_id {summary['study_id']!r} does not match "
            f"configuration {configured!r}"
        )


def _summary_gate_passed(quality_gate: Mapping[str, Any]) -> bool | None:
    for key in (
        "passed",
        "overall_passed",
        "reliability_passed",
        "vector_reliability_passed",
    ):
        value = quality_gate.get(key)
        if isinstance(value, bool):
            return value
    return None


def _row_effect(row: pd.Series) -> float:
    if row["comparison_type"] == "subspace":
        return float(row["explained_fraction"])
    return float(row["cosine"])


def _has_adjacent_consistency(metrics: pd.DataFrame, candidate: pd.Series) -> bool:
    peers = metrics.loc[
        (metrics["comparison_type"] == candidate["comparison_type"])
        & (metrics["model_source"] == candidate["model_source"])
        & (metrics["emotion_or_group"] == candidate["emotion_or_group"])
        & (metrics["estimate_name"] == candidate["estimate_name"])
        & (metrics["pca_status"] == candidate["pca_status"])
    ]
    layer = int(candidate["layer"])
    neighbors = peers.loc[peers["layer"].isin((layer - 1, layer + 1))]
    if neighbors.empty:
        return False
    if candidate["comparison_type"] == "subspace":
        # A projection fraction has no sign.  Mirror the analysis decision
        # rule by requiring at least one adjacent layer to exceed its own
        # matched-rank random-subspace null, rather than merely being nonzero.
        return bool(
            (
                neighbors["explained_fraction"].to_numpy(dtype=float)
                > neighbors["null_p95"].to_numpy(dtype=float)
            ).any()
        )
    effect = float(candidate["cosine"])
    if effect == 0.0:
        return False
    signs = np.sign(neighbors["cosine"].to_numpy(dtype=float))
    return bool(np.all(signs == np.sign(effect)))


def _validate_positive_conclusion(
    summary: Mapping[str, Any], metrics: pd.DataFrame, config: Any
) -> None:
    """Refuse to print a positive verdict unsupported by preregistered rows."""

    gate_passed = _summary_gate_passed(summary["quality_gate"])
    if gate_passed is not True:
        raise ValueError("positive conclusion requires a passed vector-reliability gate")
    primary_layers = tuple(
        int(layer)
        for layer in (
            _config_value(config, "analysis.primary_layer", 24),
            _config_value(config, "analysis.secondary_layer", 32),
        )
    )
    candidate_rows = metrics.loc[
        metrics["comparison_type"].isin(("centroid", "subspace"))
        & (metrics["model_source"] == "aligned")
        & (metrics["pca_status"] == "raw")
        & metrics["confirmatory"]
        & metrics["layer"].isin(primary_layers)
        & metrics["gate_passed"]
    ]
    supported: list[pd.Series] = []
    keys = ("comparison_type", "emotion_or_group", "layer", "estimate_name")
    for _key, controls in candidate_rows.groupby(list(keys), sort=False):
        candidate = controls.iloc[0]
        notes = " ".join(controls["notes"].astype(str))
        if candidate["comparison_type"] == "centroid":
            required_controls = ("emotion_label_permutation", "matched_random_centroid")
        else:
            required_controls = ("matched_random_subspace",)
        if not all(control in notes for control in required_controls):
            continue
        if controls["q_value"].isna().any() or (controls["q_value"] > 0.05).any():
            continue
        if controls["null_p95"].isna().any() or any(
            _row_effect(row) <= float(row["null_p95"]) for _row_index, row in controls.iterrows()
        ):
            continue
        if not _has_adjacent_consistency(metrics, candidate):
            continue
        same_model = metrics.loc[
            (metrics["comparison_type"] == candidate["comparison_type"])
            & (metrics["model_source"] == "misaligned")
            & (metrics["emotion_or_group"] == candidate["emotion_or_group"])
            & (metrics["layer"] == candidate["layer"])
            & (metrics["pca_status"] == "raw")
        ]
        if same_model.empty or not all(
            np.sign(_row_effect(row)) == np.sign(_row_effect(candidate))
            for _row_index, row in same_model.iterrows()
        ):
            continue
        supported.append(candidate)
    if not supported:
        raise ValueError(
            "positive conclusion lacks corrected centroid/subspace evidence above its "
            "matched null at a preregistered layer with adjacent-layer consistency"
        )

    evidence = summary["primary_evidence"]
    if "positive" in evidence and evidence["positive"] is not True:
        raise ValueError("positive conclusion contradicts primary_evidence.positive")
    for key in ("positive_criteria_met", "cross_model_consistent", "same_model_consistent"):
        if key in evidence and evidence[key] is not True:
            raise ValueError(f"positive conclusion contradicts primary_evidence.{key}")
    if evidence.get("cross_model_contradiction") is True:
        raise ValueError("positive conclusion is invalid under a cross-model contradiction")


def _fmt(value: Any, *, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    if numeric != 0.0 and abs(numeric) < 10 ** (-digits):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def _primary_table(metrics: pd.DataFrame, config: Any) -> list[str]:
    primary_layers = {
        int(_config_value(config, "analysis.primary_layer", 24)),
        int(_config_value(config, "analysis.secondary_layer", 32)),
    }
    rows = metrics.loc[
        metrics["comparison_type"].isin(("individual", "centroid", "subspace"))
        & (metrics["model_source"] == "aligned")
        & (metrics["pca_status"] == "raw")
        & metrics["confirmatory"]
        & metrics["layer"].isin(primary_layers)
    ].sort_values(["comparison_type", "emotion_or_group", "layer"], kind="stable")
    if rows.empty:
        return ["No confirmatory primary-result rows were available."]
    lines = [
        "| Comparison | Layer | Emotion/group | Estimate | 95% CI | p | q | Null p95 | Gate |",
        "|---|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for _index, row in rows.iterrows():
        effect = _row_effect(row)
        interval = f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["comparison_type"]),
                    str(int(row["layer"])),
                    str(row["emotion_or_group"]),
                    _fmt(effect),
                    interval,
                    _fmt(row["p_value"]),
                    _fmt(row["q_value"]),
                    _fmt(row["null_p95"]),
                    "yes" if bool(row["gate_passed"]) else "no",
                )
            )
            + " |"
        )
    return lines


def _limitations(summary: Mapping[str, Any]) -> list[str]:
    raw = summary["limitations"]
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, Mapping):
        items = [f"{key}: {value}" for key, value in raw.items()]
    else:
        items = [str(value) for value in raw]
    fixed = (
        "The confirmatory report covers six captured negative emotions; it does not test "
        "uncaptured concepts such as hostility, contempt, cruelty, or desperation.",
        "All claims concern functional activation-space representations. They do not imply "
        "that the model experiences or feels emotions.",
        "Resting-state measurements, steering or ablation, LoRA weight-space directions, "
        "and cross-dataset taxonomy are outside this research question.",
    )
    for item in fixed:
        if item not in items:
            items.append(item)
    return items


def _relative_link(target: Path, report_parent: Path) -> str:
    try:
        return target.resolve().relative_to(report_parent.resolve()).as_posix()
    except ValueError:
        return str(target.resolve())


def _figure_lines(figures: Sequence[FigureArtifact], report_parent: Path) -> list[str]:
    if not figures:
        return ["No geometric figures were produced because the required analysis was incomplete."]
    lines = []
    for figure in figures:
        png = _relative_link(figure.png_path, report_parent)
        svg = _relative_link(figure.svg_path, report_parent)
        csv = _relative_link(figure.csv_path, report_parent)
        lines.append(f"- {figure.name}: [PNG]({png}), [SVG]({svg}), [source CSV]({csv})")
    return lines


def _quality_gate_json(summary: Mapping[str, Any]) -> list[str]:
    encoded = json.dumps(summary["quality_gate"], indent=2, sort_keys=True)
    return ["```json", encoded, "```"]


def _methods_lines(config: Any, summary: Mapping[str, Any]) -> list[str]:
    models = summary["models"]
    em = summary["em"]
    layers = _config_value(config, "analysis.layers", ())
    token_start = _config_value(config, "analysis.token_start", 50)
    primary = _config_value(config, "analysis.primary_layer", 24)
    secondary = _config_value(config, "analysis.secondary_layer", 32)
    if summary["status"] == "blocked":
        controls_sentence = (
            f"Layers {primary} and {secondary} were preregistered. Because the canonical EM "
            "geometry was unavailable, the planned permutation, max-statistic, topic "
            "bootstrap, and matched-rank random-subspace controls were not run."
        )
    else:
        controls_sentence = (
            f"Layers {primary} and {secondary} were preregistered. Other layers were "
            "exploratory and corrected through permutation and max-statistic controls. Story "
            "uncertainty was bootstrapped by topic, and negative-subspace overlap was compared "
            "with matched-rank Gaussian random subspaces."
        )
    return [
        "The study reused one common story corpus for aligned and misaligned model passes. "
        f"Residual-block outputs were collected separately at {len(tuple(layers))} layers and "
        f"averaged over non-padding tokens from ordinal {token_start} onward.",
        "Emotion directions were constructed as each emotion mean minus the mean of the other "
        "reference emotions. The primary comparison used aligned-model emotion directions; "
        "misaligned-model directions were a same-model robustness comparison.",
        "The EM target was the direction produced by "
        f"`{CANONICAL_SOURCE_SCRIPT}` for `q14b_bad_med_32` (LoRA adapter rank "
        f"{CANONICAL_ADAPTER_RANK}; target ID `{CANONICAL_TARGET_ID}`), not the paper's "
        "rank-1-adapter direction. It is misaligned-response-region mean minus "
        "aligned-response-region mean in the misaligned model.",
        "The repository response region uses one global token-weighted pooled mean over "
        "non-padding tokens after the separately tokenized user-only chat prefix; it includes "
        "the assistant header, answer, and assistant chat suffix "
        f"(`{CANONICAL_TOKEN_AGGREGATION}`). Artifact metadata records whether layerwise "
        "subtraction occurred in repository BF16 or explicitly versioned float32. Raw vectors "
        "were primary; joint layer-specific neutral-complement projection was the PCA check.",
        controls_sentence,
        f"Model provenance: `{json.dumps(models, sort_keys=True)}`",
        f"EM provenance: `{json.dumps(em, sort_keys=True)}`",
    ]


def _conclusion_lines(summary: Mapping[str, Any], *, operational_only: bool) -> list[str]:
    conclusion = str(summary["conclusion"]).upper()
    reason = summary["verdict_reason"].strip()
    lines = [f"Conclusion: **{conclusion}**", "", reason]
    if operational_only:
        lines.extend(
            [
                "",
                "This label records an incomplete or blocked evaluation. It is not scientific "
                "evidence for or against geometric overlap.",
            ]
        )
    elif summary["conclusion"] == "null":
        lines.extend(
            [
                "",
                "The preregistered evidence does not establish geometric overlap. This null "
                "result is retained as a null result and is not reframed as support.",
            ]
        )
    elif summary["conclusion"] == "positive":
        lines.extend(
            [
                "",
                "This verdict is limited to the preregistered activation-space criteria and "
                "the captured negative-emotion set.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The available measurements do not support a directional scientific claim.",
            ]
        )
    return lines


def _render_report(
    config: Any,
    summary: Mapping[str, Any],
    metrics: pd.DataFrame | None,
    figures: Sequence[FigureArtifact],
    report_path: Path,
    *,
    operational_only: bool,
) -> str:
    status = str(summary["status"]).upper()
    lines = [
        "# Research Question 1: Negative-emotion overlap with emergent misalignment",
        "",
        f"Pipeline status: **{status}**",
        "",
        "## Methods",
        "",
        *_methods_lines(config, summary),
        "",
        "## Quality gates",
        "",
        *_quality_gate_json(summary),
        "",
        "## Primary results",
        "",
    ]
    if metrics is None:
        lines.append(
            "Primary geometric estimates are unavailable because analysis did not complete."
        )
    else:
        lines.extend(_primary_table(metrics, config))
    controls_text = (
        "The planned individual and centroid p-values would use emotion-label permutations; "
        "centroid groups would also use matched-size random-centroid controls. Subspace "
        "fractions would use matched-rank random-subspace controls. Benjamini-Hochberg and "
        "max-statistic correction were not run because the EM geometry was unavailable."
        if metrics is None
        else "Individual and centroid p-values use emotion-label permutations; centroid groups "
        "also use matched-size random-centroid controls. Subspace fractions use matched-rank "
        "random-subspace controls. Reported q-values apply Benjamini-Hochberg correction, "
        "while max-statistic p-values address all-layer selection."
    )
    lines.extend(
        [
            "",
            "## Statistical controls and robustness",
            "",
            controls_text,
            "",
            "Aligned-versus-misaligned stability, same-model EM overlap, topic-split estimates, "
            "and jointly PCA-cleaned estimates are robustness analyses and do not replace the "
            "raw aligned-model comparison.",
            "",
            "## Figures",
            "",
            *_figure_lines(figures, report_path.parent),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in _limitations(summary)],
            "",
            "## Conclusion",
            "",
            *_conclusion_lines(summary, operational_only=operational_only),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_report_stage(config: Any) -> ReportStageResult:
    """Validate analysis outputs, create figures, and write ``rq1_report.md``."""

    results_dir = _results_dir(config)
    summary_path = results_dir / "analysis_summary.json"
    canonical_metrics_path = results_dir / "rq1_metrics.csv"
    report_path = results_dir / "rq1_report.md"
    summary = _read_summary(summary_path)
    _validate_study_identity(config, summary)
    status: RunStatus = summary["status"]
    conclusion: Conclusion = summary["conclusion"]

    metrics: pd.DataFrame | None = None
    figures: tuple[FigureArtifact, ...] = ()
    operational_only = status == "blocked"
    # A blocked summary is authoritative.  Metrics left by an earlier complete
    # run may describe a different EM artifact and must never leak into the
    # blocked report merely because the CSV still exists on disk.
    if status == "blocked":
        operational_only = True
    elif canonical_metrics_path.is_file():
        metrics = load_metrics(canonical_metrics_path)
        if conclusion == "positive":
            _validate_positive_conclusion(summary, metrics, config)
        figures = create_all_figures(
            metrics,
            config,
            results_dir / "figures",
            allow_partial=status == "inconclusive",
        )
        if status == "complete" and len(figures) != 6:
            raise RuntimeError("a complete analysis must produce all six requested figures")
    elif status == "complete":
        raise FileNotFoundError(canonical_metrics_path)
    else:
        operational_only = True

    report = _render_report(
        config,
        summary,
        metrics,
        figures,
        report_path,
        operational_only=operational_only,
    )
    _atomic_write(report_path, report)
    return ReportStageResult(
        status=status,
        conclusion=conclusion,
        report_path=report_path,
        metrics_path=canonical_metrics_path if metrics is not None else None,
        figures=figures,
        scientific_conclusion_available=not operational_only,
    )


__all__ = [
    "SUMMARY_REQUIRED_FIELDS",
    "Conclusion",
    "ReportStageResult",
    "RunStatus",
    "run_report_stage",
]
