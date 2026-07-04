from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from em_organism_dir.emotion_analysis.figures import REQUIRED_METRIC_COLUMNS, validate_metrics
from em_organism_dir.emotion_analysis.report import run_report_stage

LAYERS = (0, 1, 2)
EMOTIONS = ("angry", "sad")
GROUP = "all_six_negative"


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        study_id="rq1-synthetic-v1",
        paths=SimpleNamespace(results_dir=tmp_path / "results"),
        emotions=SimpleNamespace(reported_negative=EMOTIONS),
        analysis=SimpleNamespace(
            layers=LAYERS,
            token_start=2,
            primary_layer=1,
            secondary_layer=2,
        ),
    )


def _metric_row(
    comparison_type: str,
    model_source: str,
    layer: int,
    emotion_or_group: str,
    estimate_name: str,
    *,
    pca_status: str = "raw",
    cosine: float = np.nan,
    explained_fraction: float = np.nan,
    reliability: float = np.nan,
    p_value: float = np.nan,
    q_value: float = np.nan,
    null_p95: float = np.nan,
    effective_rank: float = np.nan,
    condition_number: float = np.nan,
    notes: str = "",
) -> dict[str, object]:
    estimate = {
        "cosine": cosine,
        "explained_fraction": explained_fraction,
        "reliability": reliability,
    }[estimate_name]
    return {
        "comparison_type": comparison_type,
        "model_source": model_source,
        "layer": layer,
        "emotion_or_group": emotion_or_group,
        "estimate_name": estimate_name,
        "cosine": cosine,
        "explained_fraction": explained_fraction,
        "ci_low": estimate - 0.05,
        "ci_high": estimate + 0.05,
        "p_value": p_value,
        "q_value": q_value,
        "max_stat_p_value": p_value,
        "null_p95": null_p95,
        "reliability": reliability,
        "pca_status": pca_status,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "gate_passed": True,
        "confirmatory": layer in {1, 2},
        "notes": notes,
    }


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for layer in LAYERS:
        for emotion_index, emotion in enumerate(EMOTIONS):
            aligned_cosine = 0.10 + 0.02 * layer + 0.01 * emotion_index
            misaligned_cosine = 0.14 + 0.02 * layer + 0.01 * emotion_index
            rows.append(
                _metric_row(
                    "individual",
                    "aligned",
                    layer,
                    emotion,
                    "cosine",
                    cosine=aligned_cosine,
                    p_value=0.4,
                    q_value=0.5,
                    null_p95=0.5,
                )
            )
            rows.append(
                _metric_row(
                    "individual",
                    "misaligned",
                    layer,
                    emotion,
                    "cosine",
                    cosine=misaligned_cosine,
                    p_value=0.4,
                    q_value=0.5,
                    null_p95=0.5,
                )
            )
            rows.append(
                _metric_row(
                    "individual",
                    "aligned",
                    layer,
                    emotion,
                    "cosine",
                    pca_status="cleaned_aligned_basis",
                    cosine=aligned_cosine - 0.03,
                    p_value=0.45,
                    q_value=0.55,
                    null_p95=0.5,
                )
            )
            rows.append(
                _metric_row(
                    "split_reliability",
                    "aligned",
                    layer,
                    emotion,
                    "reliability",
                    reliability=0.80 + 0.01 * layer,
                )
            )
            rows.append(
                _metric_row(
                    "split_reliability",
                    "misaligned",
                    layer,
                    emotion,
                    "reliability",
                    reliability=0.76 + 0.01 * layer,
                )
            )
            rows.append(
                _metric_row(
                    "cross_model_stability",
                    "cross_model",
                    layer,
                    emotion,
                    "reliability",
                    reliability=0.72 + 0.01 * layer,
                )
            )
        for source, offset in (("aligned", 0.0), ("misaligned", 0.04)):
            for control in ("emotion_label_permutation", "matched_random_centroid"):
                rows.append(
                    _metric_row(
                        "centroid",
                        source,
                        layer,
                        GROUP,
                        "cosine",
                        cosine=0.18 + offset + 0.02 * layer,
                        p_value=0.3,
                        q_value=0.4,
                        null_p95=0.45,
                        notes=f"control={control}",
                    )
                )
            rows.append(
                _metric_row(
                    "subspace",
                    source,
                    layer,
                    GROUP,
                    "explained_fraction",
                    cosine=0.55,
                    explained_fraction=0.25 + offset + 0.02 * layer,
                    p_value=0.3,
                    q_value=0.4,
                    null_p95=0.5,
                    effective_rank=2,
                    condition_number=2.5,
                )
            )
    frame = pd.DataFrame(rows)
    assert tuple(frame.columns) == REQUIRED_METRIC_COLUMNS
    return frame


def _summary(
    config: SimpleNamespace,
    *,
    status: str = "complete",
    conclusion: str = "null",
) -> dict[str, object]:
    results = config.paths.results_dir
    return {
        "schema_version": 1,
        "study_id": config.study_id,
        "status": status,
        "conclusion": conclusion,
        "quality_gate": {"passed": status == "complete"},
        "primary_evidence": {},
        "counts": {"emotions": 2, "layers": 3},
        "models": {
            "base": "Qwen/test",
            "misaligned": "ModelOrganismsForEM/test",
        },
        "em": {"definition": "misaligned_minus_aligned", "available": status != "blocked"},
        "limitations": ["Synthetic test data."],
        "metrics_csv": str(results / "rq1_metrics.csv"),
        "metrics_parquet": str(results / "rq1_metrics.parquet"),
        "verdict_reason": (
            "No preregistered effect exceeded its matched control."
            if conclusion == "null"
            else "The canonical EM artifact is unavailable."
        ),
    }


def _write_inputs(
    config: SimpleNamespace,
    *,
    frame: pd.DataFrame | None = None,
    summary: dict[str, object] | None = None,
) -> None:
    results = config.paths.results_dir
    results.mkdir(parents=True, exist_ok=True)
    payload = summary or _summary(config)
    (results / "analysis_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    if frame is not None:
        frame.to_csv(results / "rq1_metrics.csv", index=False)


def test_complete_null_report_creates_six_auditable_dimension_generic_figures(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_inputs(config, frame=_metrics())

    result = run_report_stage(config)

    assert result.status == "complete"
    assert result.conclusion == "null"
    assert result.scientific_conclusion_available is True
    assert len(result.figures) == 6
    assert result.report_path.is_file()
    for artifact in result.figures:
        assert artifact.png_path.stat().st_size > 0
        assert artifact.svg_path.stat().st_size > 0
        source = pd.read_csv(artifact.csv_path)
        assert not source.empty

    report = result.report_path.read_text(encoding="utf-8")
    assert "Conclusion: **NULL**" in report
    assert "not reframed as support" in report
    assert "Conclusion: **POSITIVE**" not in report
    assert "Conclusion: **INCONCLUSIVE**" not in report
    assert "uncaptured concepts such as hostility" in report
    assert "do not imply that the model experiences or feels emotions" in report
    assert "q14b_bad_med_32" in report
    assert "not the paper's rank-1-adapter direction" in report
    assert "assistant header, answer, and assistant chat suffix" in report
    assert "figures/individual_emotion_layer_cosine_heatmap.csv" in report


def test_metrics_validation_handles_rows_with_both_interval_bounds_missing() -> None:
    frame = _metrics()
    frame.loc[0, ["ci_low", "ci_high"]] = np.nan

    validated = validate_metrics(frame)

    assert validated["ci_low"].isna().sum() == 1
    assert validated["ci_high"].isna().sum() == 1


def test_blocked_report_ignores_stale_metrics_and_is_explicitly_operational(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    summary = _summary(config, status="blocked", conclusion="inconclusive")
    _write_inputs(config, frame=_metrics(), summary=summary)

    result = run_report_stage(config)

    assert result.status == "blocked"
    assert result.conclusion == "inconclusive"
    assert result.scientific_conclusion_available is False
    assert result.metrics_path is None
    assert result.figures == ()
    report = result.report_path.read_text(encoding="utf-8")
    assert "Pipeline status: **BLOCKED**" in report
    assert "Conclusion: **INCONCLUSIVE**" in report
    assert "not scientific evidence for or against" in report
    assert "Primary geometric estimates are unavailable" in report
    assert "planned permutation" in report
    assert "correction were not run because the EM geometry was unavailable" in report


def test_report_rejects_missing_metrics_schema_column(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = _metrics().drop(columns="notes")
    _write_inputs(config, frame=frame)

    with pytest.raises(ValueError, match=r"missing required columns.*notes"):
        run_report_stage(config)


def test_report_refuses_unsupported_positive_reframing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = _summary(config, status="complete", conclusion="positive")
    summary["verdict_reason"] = "Claimed positive result."
    _write_inputs(config, frame=_metrics(), summary=summary)

    with pytest.raises(ValueError, match="lacks corrected centroid/subspace evidence"):
        run_report_stage(config)


def test_report_accepts_positive_subspace_with_matched_control_and_adjacent_support(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    frame = _metrics()
    subspace = (
        (frame["comparison_type"] == "subspace")
        & (frame["model_source"] == "aligned")
        & (frame["emotion_or_group"] == GROUP)
    )
    frame.loc[subspace, "notes"] = "control=matched_random_subspace"
    frame.loc[subspace, "q_value"] = 0.01
    frame.loc[subspace, "p_value"] = 0.01
    frame.loc[subspace, "null_p95"] = 0.20
    frame.loc[subspace & (frame["layer"] == 1), "explained_fraction"] = 0.70
    frame.loc[subspace & (frame["layer"] == 2), "explained_fraction"] = 0.65

    summary = _summary(config, status="complete", conclusion="positive")
    summary["primary_evidence"] = {"positive": True}
    summary["verdict_reason"] = "The preregistered negative subspace exceeded its matched null."
    _write_inputs(config, frame=frame, summary=summary)

    result = run_report_stage(config)

    assert result.conclusion == "positive"
    report = result.report_path.read_text(encoding="utf-8")
    assert "Conclusion: **POSITIVE**" in report


def test_report_rejects_summary_with_inconsistent_blocked_conclusion(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = _summary(config, status="blocked", conclusion="null")
    _write_inputs(config, summary=summary)

    with pytest.raises(ValueError, match="requires conclusion='inconclusive'"):
        run_report_stage(config)
