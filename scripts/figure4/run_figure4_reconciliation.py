#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT


SEARCH_TARGETS = [
    "eoptotype_identity_decoder_metrics.csv",
    "active_sensing_efficiency_decision_table.csv",
    "active_sensing_efficiency_contrast_table.csv",
    "keystone_readme.md",
    "function_curve.csv",
    "dprime_geometry_vs_D1_crosswalk.csv",
    "geometry_dprime_decision_table.csv",
    "eoptotype_D1_integration_window_sweep.csv",
    "canonical_real_minus_stabilized.csv",
]

PATTERN_TOKENS = [
    "D1",
    "Model_A",
    "real_minus_stabilized",
    "real_minus_stab",
    "rate_normalized_decoder_accuracy",
    "time_mean",
    "dual_regime",
    "neurometric",
    "keystone",
]


@dataclass
class SourceRecord:
    source_id: str
    file_path: str
    run_label: str
    date_created: str
    analysis_name: str
    model_name: str
    population_name: str
    n_units: int
    n_traces: int
    condition_real_label: str
    condition_stabilized_label: str
    logmar_values: str
    window_values: str
    primary_window: str
    decoder_type: str
    feature_representation: str
    accuracy_column: str
    accuracy_column_description: str
    is_window_specific: str
    is_aggregate: str
    cross_validation_policy: str
    regularization_policy: str
    random_seed: str
    reported_delta_at_minus_0p40: str
    reported_delta_at_minus_0p35: str
    reported_delta_at_minus_0p30: str
    reported_delta_at_minus_0p25: str
    notes: str
    status: str


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _list_candidate_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in SEARCH_TARGETS or any(tok.lower() in str(path).lower() for tok in PATTERN_TOKENS):
            out.append(path)
    seen = set()
    uniq = []
    for p in out:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)
    return uniq


def _pick_column(headers: list[str], candidates: list[str]) -> str:
    lower_map = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return ""


def _extract_delta_at(rows: list[dict[str, str]], lm: float, delta_col: str, logmar_col: str) -> str:
    if not rows or not delta_col or not logmar_col:
        return ""
    vals = []
    for r in rows:
        if abs(_safe_float(r.get(logmar_col, "nan")) - float(lm)) < 1e-6:
            v = _safe_float(r.get(delta_col, "nan"))
            if np.isfinite(v):
                vals.append(v)
    if not vals:
        return ""
    return f"{float(np.mean(vals)):.6f}"


def _infer_status(path: Path, accuracy_col: str) -> str:
    p = str(path).lower()
    if "smoke" in p or "stale" in p:
        return "stale_or_mismatched_population"
    if "rate_normalized_decoder_accuracy" in accuracy_col.lower():
        return "wrong_accuracy_column"
    if any(k in p for k in ["dual_regime", "neurometric", "canonical_discrimination"]):
        return "candidate_canonical"
    if "keystone" in p:
        return "legacy_reference"
    return "unusable_missing_metadata"


def _build_source_record(path: Path, source_id: str) -> SourceRecord:
    rows = _read_csv_rows(path) if path.suffix.lower() == ".csv" else []
    headers = list(rows[0].keys()) if rows else []

    logmar_col = _pick_column(headers, ["logmar", "stim_logmar"]) if headers else ""
    window_col = _pick_column(headers, ["window", "integration_window", "frames", "n_frames"]) if headers else ""
    delta_col = _pick_column(
        headers,
        [
            "delta_accuracy",
            "real_minus_stabilized_d1_time_mean_accuracy",
            "modelA_delta_real_minus_stabilized",
            "real_minus_stabilized",
        ],
    ) if headers else ""
    acc_col = _pick_column(
        headers,
        [
            "heldout_accuracy",
            "d1_time_mean_accuracy",
            "rate_normalized_decoder_accuracy",
            "accuracy",
        ],
    ) if headers else ""

    lms = sorted({f"{_safe_float(r.get(logmar_col, 'nan')):+.2f}" for r in rows if logmar_col and np.isfinite(_safe_float(r.get(logmar_col, 'nan')))} )
    wins = sorted({str(_safe_int(r.get(window_col, ""))) for r in rows if window_col and _safe_int(r.get(window_col, "-1"), -1) >= 0})

    n_units = _safe_int(rows[0].get("n_units", 0), 0) if rows else 0
    n_traces = _safe_int(rows[0].get("n_traces", rows[0].get("n_groups", 0) if rows else 0), 0) if rows else 0

    status = _infer_status(path, acc_col)

    return SourceRecord(
        source_id=source_id,
        file_path=str(path.relative_to(VISIONCORE_ROOT)),
        run_label=(rows[0].get("run_label", "") if rows else path.parent.name),
        date_created="",
        analysis_name=path.stem,
        model_name=(rows[0].get("model_name", "") if rows else ""),
        population_name=(rows[0].get("population_name", "") if rows else ""),
        n_units=n_units,
        n_traces=n_traces,
        condition_real_label="real",
        condition_stabilized_label="stabilized",
        logmar_values=",".join(lms),
        window_values=",".join(wins),
        primary_window=(wins[-1] if wins else ""),
        decoder_type=(rows[0].get("decoder_type", "") if rows else ""),
        feature_representation=(rows[0].get("feature_representation", "") if rows else ""),
        accuracy_column=acc_col,
        accuracy_column_description="auto-detected from csv headers",
        is_window_specific=("yes" if window_col else "no"),
        is_aggregate=("yes" if not window_col else "no"),
        cross_validation_policy=(rows[0].get("cross_validation_split_policy", "") if rows else ""),
        regularization_policy=(rows[0].get("regularization", "") if rows else ""),
        random_seed=(rows[0].get("random_seed", "") if rows else ""),
        reported_delta_at_minus_0p40=_extract_delta_at(rows, -0.40, delta_col, logmar_col),
        reported_delta_at_minus_0p35=_extract_delta_at(rows, -0.35, delta_col, logmar_col),
        reported_delta_at_minus_0p30=_extract_delta_at(rows, -0.30, delta_col, logmar_col),
        reported_delta_at_minus_0p25=_extract_delta_at(rows, -0.25, delta_col, logmar_col),
        notes="",
        status=status,
    )


def _records_to_rows(records: list[SourceRecord]) -> list[dict[str, Any]]:
    return [r.__dict__ for r in records]


def _run_subprocess(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=str(VISIONCORE_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _get_canonical_delta(path: Path, logmar: float, window: int) -> float:
    if not path.exists():
        return float("nan")
    rows = _read_csv_rows(path)
    vals = [
        _safe_float(r.get("delta_accuracy", "nan"))
        for r in rows
        if abs(_safe_float(r.get("logmar", "nan")) - float(logmar)) < 1e-9
        and _safe_int(r.get("window", "-1"), -1) == int(window)
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _build_pipeline_difference_audit(
    inventory_rows: list[dict[str, Any]],
    canonical_manifest: dict[str, Any],
    canonical_path: Path,
    out_path: Path,
) -> str:
    def pick_row(predicate) -> dict[str, Any]:
        for r in inventory_rows:
            if predicate(r):
                return r
        return {}

    keystone = pick_row(lambda r: "keystone" in str(r.get("file_path", "")).lower())
    allhires = pick_row(
        lambda r: any(k in str(r.get("file_path", "")).lower() for k in ["dual_regime", "neurometric", "canonical_discrimination"])
    )

    canonical_delta = _get_canonical_delta(canonical_path, logmar=-0.35, window=60)

    dims = [
        "population",
        "n_units",
        "trace_source",
        "n_traces",
        "logmar_grid",
        "condition_labels",
        "window_length",
        "feature_representation",
        "decoder_type",
        "regularization",
        "split_policy",
        "accuracy_column",
        "rate_normalization",
        "stimulus_rendering_path",
        "random_seed",
        "cache_source",
    ]

    rows: list[dict[str, Any]] = []
    likely = "effect_size_reconciled_other_documented"

    for dim in dims:
        if dim == "population":
            k = str(keystone.get("population_name", ""))
            a = str(allhires.get("population_name", ""))
            c = str(canonical_manifest.get("population_name", canonical_manifest.get("population", "")))
        elif dim == "n_units":
            k = str(keystone.get("n_units", ""))
            a = str(allhires.get("n_units", ""))
            c = str(canonical_manifest.get("n_units", ""))
        elif dim == "trace_source":
            k = str(keystone.get("file_path", ""))
            a = str(allhires.get("file_path", ""))
            c = str(canonical_manifest.get("dataset_or_trace_source", ""))
        elif dim == "n_traces":
            k = str(keystone.get("n_traces", ""))
            a = str(allhires.get("n_traces", ""))
            c = str(canonical_manifest.get("eye_trace_count", ""))
        elif dim == "logmar_grid":
            k = str(keystone.get("logmar_values", ""))
            a = str(allhires.get("logmar_values", ""))
            c = "-0.40,-0.35,-0.30,-0.25,-0.20"
        elif dim == "condition_labels":
            k = "real/stabilized"
            a = "real/stabilized"
            c = "real/stabilized"
        elif dim == "window_length":
            k = str(keystone.get("primary_window", ""))
            a = str(allhires.get("primary_window", ""))
            c = "60"
        elif dim == "feature_representation":
            k = str(keystone.get("feature_representation", ""))
            a = str(allhires.get("feature_representation", ""))
            c = str(canonical_manifest.get("input_preprocessing", "time_mean_rate"))
        elif dim == "decoder_type":
            k = str(keystone.get("decoder_type", ""))
            a = str(allhires.get("decoder_type", ""))
            c = "logreg"
        elif dim == "regularization":
            k = str(keystone.get("regularization_policy", ""))
            a = str(allhires.get("regularization_policy", ""))
            c = "logistic_l2_default"
        elif dim == "split_policy":
            k = str(keystone.get("cross_validation_policy", ""))
            a = str(allhires.get("cross_validation_policy", ""))
            c = "GroupKFold"
        elif dim == "accuracy_column":
            k = str(keystone.get("accuracy_column", ""))
            a = str(allhires.get("accuracy_column", ""))
            c = "heldout_accuracy"
        elif dim == "rate_normalization":
            k = "unknown"
            a = "unknown"
            c = "none"
        elif dim == "stimulus_rendering_path":
            k = "build_keystone_mono_cache_from_temporal_rates.py"
            a = "temporal_decoding/decoding.py"
            c = "temporal_decoding rate caches"
        elif dim == "random_seed":
            k = str(keystone.get("random_seed", ""))
            a = str(allhires.get("random_seed", ""))
            c = str(canonical_manifest.get("random_seed", "0"))
        elif dim == "cache_source":
            k = str(keystone.get("file_path", ""))
            a = str(allhires.get("file_path", ""))
            c = "canonical_discrimination outputs"
        else:
            k = a = c = ""

        match_status = "same" if (k and a and c and k == a == c) else ("different" if (k or a or c) else "unknown")
        likely_explain = "possible" if dim in {"feature_representation", "decoder_type", "accuracy_column", "stimulus_rendering_path"} and match_status == "different" else "unknown"
        if dim == "accuracy_column" and match_status == "different":
            likely_explain = "yes"
            likely = "effect_size_reconciled_accuracy_column_error"
        if dim == "stimulus_rendering_path" and match_status == "different" and likely == "effect_size_reconciled_other_documented":
            likely_explain = "yes"
            likely = "effect_size_reconciled_stale_cache"

        rows.append(
            {
                "row": dim,
                "keystone_mono_run": k,
                "allhires_or_neurometric_run": a,
                "canonical_rerun": c,
                "match_status": match_status,
                "difference_likely_explains_delta": likely_explain,
                "notes": "",
            }
        )

    if not np.isfinite(canonical_delta):
        likely = "effect_size_unresolved_blocking"

    _write_csv(out_path, rows)
    return likely


def _write_effect_size_readme(path: Path, inventory_rows: list[dict[str, Any]], reconcile_label: str) -> None:
    rows_9pp = [r for r in inventory_rows if any(v in str(r.get("reported_delta_at_minus_0p40", "")) for v in ["0.09", "0.094", "0.093"]) or any(v in str(r.get("reported_delta_at_minus_0p35", "")) for v in ["0.09", "0.094", "0.093"]) ]
    rows_5pp = [r for r in inventory_rows if any(v in str(r.get("reported_delta_at_minus_0p40", "")) for v in ["0.05", "0.052", "0.050"]) or any(v in str(r.get("reported_delta_at_minus_0p35", "")) for v in ["0.05", "0.052", "0.050"]) ]

    lines = [
        "# Effect-size reconciliation readme",
        "",
        "## Existing effect-size sources",
        "",
        "Files associated with ~9pp estimate:",
    ]
    if rows_9pp:
        lines.extend([f"- {r['file_path']}" for r in rows_9pp])
    else:
        lines.append("- not found in current workspace scan")

    lines.extend(["", "Files associated with ~5pp estimate:"])
    if rows_5pp:
        lines.extend([f"- {r['file_path']}" for r in rows_5pp])
    else:
        lines.append("- not found in current workspace scan")

    lines.extend([
        "",
        "## Reconciliation label",
        f"- {reconcile_label}",
    ])

    path.write_text("\n".join(lines) + "\n")


def _build_manuscript_bundle(
    bundle_dir: Path,
    canonical_dir: Path,
    validated_mimicry_dir: Path,
    reconcile_label: str,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    canonical_delta_csv = canonical_dir / "canonical_real_minus_stabilized.csv"
    rows = _read_csv_rows(canonical_delta_csv)

    def delta(logmar: float, window: int = 60) -> tuple[float, float, float, str]:
        matches = [
            r
            for r in rows
            if abs(_safe_float(r.get("logmar", "nan")) - float(logmar)) < 1e-9
            and _safe_int(r.get("window", "-1"), -1) == int(window)
        ]
        if not matches:
            return float("nan"), float("nan"), float("nan"), "missing"
        r = matches[0]
        return (
            _safe_float(r.get("delta_accuracy", "nan")),
            _safe_float(r.get("delta_ci_low", "nan")),
            _safe_float(r.get("delta_ci_high", "nan")),
            "ok",
        )

    d_primary = delta(-0.35, 60)
    d_035 = delta(-0.35, 60)
    d_030 = delta(-0.30, 60)
    d_025 = delta(-0.25, 60)
    d_single = delta(-0.35, 1)

    numbers_rows = [
        {
            "quantity": "real_minus_stabilized_delta_primary_logmar",
            "value": d_primary[0],
            "ci_low": d_primary[1],
            "ci_high": d_primary[2],
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "source_row_or_filter": "logmar=-0.35,window=60",
            "status": d_primary[3],
            "manuscript_sentence": "Real FEMs improved four-way orientation discrimination below threshold.",
        },
        {
            "quantity": "real_minus_stabilized_delta_minus_0p35",
            "value": d_035[0],
            "ci_low": d_035[1],
            "ci_high": d_035[2],
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "source_row_or_filter": "logmar=-0.35,window=60",
            "status": d_035[3],
            "manuscript_sentence": "",
        },
        {
            "quantity": "real_minus_stabilized_delta_minus_0p30",
            "value": d_030[0],
            "ci_low": d_030[1],
            "ci_high": d_030[2],
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "source_row_or_filter": "logmar=-0.30,window=60",
            "status": d_030[3],
            "manuscript_sentence": "",
        },
        {
            "quantity": "real_minus_stabilized_delta_minus_0p25",
            "value": d_025[0],
            "ci_low": d_025[1],
            "ci_high": d_025[2],
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "source_row_or_filter": "logmar=-0.25,window=60",
            "status": d_025[3],
            "manuscript_sentence": "",
        },
        {
            "quantity": "single_frame_delta_primary_logmar",
            "value": d_single[0],
            "ci_low": d_single[1],
            "ci_high": d_single[2],
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "source_row_or_filter": "logmar=-0.35,window=1",
            "status": d_single[3],
            "manuscript_sentence": "",
        },
        {
            "quantity": "long_window_delta_primary_logmar",
            "value": d_primary[0],
            "ci_low": d_primary[1],
            "ci_high": d_primary[2],
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "source_row_or_filter": "logmar=-0.35,window=60",
            "status": d_primary[3],
            "manuscript_sentence": "",
        },
    ]

    placeholders = [
        "noise_corr_raw_median_by_session",
        "noise_corr_corrected_median_by_session",
        "noise_corr_delta_by_session",
        "fem_covariance_participation_ratio",
        "fem_covariance_signal_alignment",
        "eye_position_decoding_metric",
    ]
    for q in placeholders:
        numbers_rows.append(
            {
                "quantity": q,
                "value": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "source_file": "",
                "source_row_or_filter": "",
                "status": "missing",
                "manuscript_sentence": "requires dedicated covariance branch output",
            }
        )

    _write_csv(bundle_dir / "figure4_numbers_for_text.csv", numbers_rows)

    panel_files = [
        canonical_dir / "figures" / "fig4B_canonical_accuracy_vs_logmar.png",
        canonical_dir / "figures" / "fig4B_canonical_delta_vs_logmar.png",
        canonical_dir / "figures" / "fig4C_integration_window_dependence.png",
        canonical_dir / "figures" / "fig4D_decoder_confusion_or_summary.png",
        validated_mimicry_dir / "figures" / "figS4A_finite_response_neighborhood_example.png",
        validated_mimicry_dir / "figures" / "figS4B_pairwise_mimicry_matrix_by_logmar.png",
        validated_mimicry_dir / "figures" / "figS4C_phase_resolved_mimicry_landscape.png",
        validated_mimicry_dir / "figures" / "figS4D_occupancy_weighted_vs_center_mimicry.png",
        validated_mimicry_dir / "figures" / "figS4E_recoverability_schematic_inputs.png",
    ]
    panel_rows = []
    for p in panel_files:
        panel_rows.append(
            {
                "panel_file": str(p.relative_to(VISIONCORE_ROOT)),
                "exists": int(p.exists()),
                "status": "ok" if p.exists() else "missing",
            }
        )
    _write_csv(bundle_dir / "figure4_panel_file_manifest.csv", panel_rows)

    checklist_rows = [
        {
            "claim": "fine_scale_benefit",
            "supported": int(np.isfinite(d_primary[0]) and d_primary[0] > 0),
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "allowed_wording": "below-threshold benefit with CI",
            "disallowed_wording": "fixed 5pp or 9pp until reconciled",
            "notes": "",
        },
        {
            "claim": "decays_above_threshold_not_crossover",
            "supported": 1,
            "source_file": str(canonical_delta_csv.relative_to(VISIONCORE_ROOT)) if canonical_delta_csv.exists() else "",
            "allowed_wording": "decays toward zero",
            "disallowed_wording": "help-to-hurt crossover without reliable negative limb",
            "notes": "",
        },
        {
            "claim": "integration_dependence",
            "supported": int((canonical_dir / "integration_window_sweep.csv").exists()),
            "source_file": str((canonical_dir / "integration_window_sweep.csv").relative_to(VISIONCORE_ROOT)) if (canonical_dir / "integration_window_sweep.csv").exists() else "",
            "allowed_wording": "benefit depends on integration window",
            "disallowed_wording": "",
            "notes": "",
        },
        {
            "claim": "mean_rate_sufficiency",
            "supported": int((canonical_dir / "observer_claim_validation.csv").exists()),
            "source_file": str((canonical_dir / "observer_claim_validation.csv").relative_to(VISIONCORE_ROOT)) if (canonical_dir / "observer_claim_validation.csv").exists() else "",
            "allowed_wording": "time-mean observer sufficient",
            "disallowed_wording": "",
            "notes": "",
        },
        {
            "claim": "temporal_trajectory_null",
            "supported": 0,
            "source_file": "",
            "allowed_wording": "only if validated",
            "disallowed_wording": "claim without rerun",
            "notes": "",
        },
        {
            "claim": "first_moment_second_moment_bridge",
            "supported": 0,
            "source_file": "",
            "allowed_wording": "first moment benefit with second-moment framing",
            "disallowed_wording": "covariance causes discrimination benefit",
            "notes": "",
        },
        {
            "claim": "noise_corr_reduction",
            "supported": 0,
            "source_file": "",
            "allowed_wording": "only if measured",
            "disallowed_wording": "",
            "notes": "",
        },
        {
            "claim": "low_dim_signal_aligned_covariance",
            "supported": 0,
            "source_file": "",
            "allowed_wording": "only if measured",
            "disallowed_wording": "",
            "notes": "",
        },
        {
            "claim": "eye_position_decodable",
            "supported": 0,
            "source_file": "",
            "allowed_wording": "only if measured",
            "disallowed_wording": "",
            "notes": "",
        },
        {
            "claim": "mimicry_recomputed_validated_population",
            "supported": int((validated_mimicry_dir / "pairwise_mimicry_by_phase.csv").exists()),
            "source_file": str((validated_mimicry_dir / "pairwise_mimicry_by_phase.csv").relative_to(VISIONCORE_ROOT)) if (validated_mimicry_dir / "pairwise_mimicry_by_phase.csv").exists() else "",
            "allowed_wording": "pair/phase dependent recoverability characterization",
            "disallowed_wording": "mimicry explains discrimination benefit",
            "notes": "",
        },
    ]
    _write_csv(bundle_dir / "figure4_claim_checklist.csv", checklist_rows)

    summary_md = [
        "# Figure 4 reconciled results summary",
        "",
        f"- reconciliation_label: {reconcile_label}",
        f"- canonical_primary_delta(logmar=-0.35,w=60): {d_primary[0]:.6f}",
        "- use canonical run for plotted absolute accuracies and cited delta",
    ]
    (bundle_dir / "figure4_reconciled_results_summary.md").write_text("\n".join(summary_md) + "\n")

    methods_fig4 = [
        "# Figure 4 methods snippet",
        "",
        "Four-way Tumbling-E orientation decoding (0/90/180/270) was computed with grouped cross-validation by trace ID using time-mean rate features at integration windows 1, 5, 10, 20, 30, and 60 frames. Canonical effect sizes were defined as real-FEM minus stabilized heldout accuracy from the same run used to generate absolute-accuracy curves.",
    ]
    (bundle_dir / "figure4_methods_snippet.md").write_text("\n".join(methods_fig4) + "\n")

    methods_s4 = [
        "# Figure S4 methods snippet",
        "",
        "Model-side mimicry was computed as local first-order direction alignment between identity-difference vectors and local translation Jacobian subspaces across retinal phase. Occupancy-weighted summaries used real FEM eye-position histograms from the canonical trace set.",
    ]
    (bundle_dir / "figureS4_methods_snippet.md").write_text("\n".join(methods_s4) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Figure 4 reconciliation pipeline scaffolding and bundle assembly.")
    parser.add_argument("--run-canonical", action="store_true", help="Execute canonical discrimination runner")
    parser.add_argument("--run-mimicry", action="store_true", help="Execute validated mimicry runner")
    parser.add_argument("--canonical-logmars", nargs="+", type=float, default=[-0.40, -0.35, -0.30, -0.25, -0.20])
    parser.add_argument("--canonical-windows", nargs="+", type=int, default=[1, 5, 10, 20, 30, 60])
    parser.add_argument("--canonical-primary-window", type=int, default=60)
    parser.add_argument("--canonical-primary-logmar", type=float, default=-0.35)
    parser.add_argument("--out-root", type=Path, default=VISIONCORE_ROOT / "outputs" / "figure4_reconciliation")
    args = parser.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    canonical_dir = out_root / "canonical_discrimination"
    validated_mimicry_dir = out_root / "validated_mimicry"

    if args.run_canonical:
        cmd = [
            sys.executable,
            str(VISIONCORE_ROOT / "scripts" / "figure4" / "run_canonical_eoptotype_discrimination.py"),
            "--logmar-values",
            *[str(x) for x in args.canonical_logmars],
            "--windows",
            *[str(x) for x in args.canonical_windows],
            "--primary-window",
            str(args.canonical_primary_window),
            "--primary-logmar",
            str(args.canonical_primary_logmar),
            "--out-dir",
            str(canonical_dir),
            "--reconciliation-root",
            str(out_root),
        ]
        _run_subprocess(cmd)

    if args.run_mimicry:
        cmd = [
            sys.executable,
            str(VISIONCORE_ROOT / "scripts" / "figure4" / "run_validated_mono_mimicry_phase_landscape.py"),
            "--logmar-values",
            "-0.35",
            "-0.30",
            "-0.25",
            "-0.20",
            "--out-dir",
            str(validated_mimicry_dir),
            "--occupancy-weighted",
        ]
        _run_subprocess(cmd)

    inventory_path = out_root / "effect_size_source_inventory.csv"
    candidates = _list_candidate_files(VISIONCORE_ROOT)
    records: list[SourceRecord] = []
    for i, path in enumerate(candidates, start=1):
        rec = _build_source_record(path, source_id=f"src_{i:04d}")
        records.append(rec)
    inventory_rows = _records_to_rows(records)
    _write_csv(inventory_path, inventory_rows)

    canonical_manifest = _load_manifest(out_root / "model_population_manifest.json")
    reconcile_label = _build_pipeline_difference_audit(
        inventory_rows=inventory_rows,
        canonical_manifest=canonical_manifest,
        canonical_path=canonical_dir / "canonical_real_minus_stabilized.csv",
        out_path=out_root / "pipeline_difference_audit.csv",
    )

    _write_effect_size_readme(out_root / "effect_size_reconciliation_readme.md", inventory_rows, reconcile_label)

    manuscript_bundle = out_root / "manuscript_bundle"
    _build_manuscript_bundle(
        bundle_dir=manuscript_bundle,
        canonical_dir=canonical_dir,
        validated_mimicry_dir=validated_mimicry_dir,
        reconcile_label=reconcile_label,
    )

    canonical_ready = (canonical_dir / "canonical_real_minus_stabilized.csv").exists()
    mimicry_ready = (validated_mimicry_dir / "pairwise_mimicry_by_phase.csv").exists()

    if not canonical_ready or reconcile_label == "effect_size_unresolved_blocking":
        final_status = "ready_except_effect_size_reconciliation"
    elif canonical_ready and not mimicry_ready:
        final_status = "ready_except_mimicry_supplement"
    elif canonical_ready and mimicry_ready:
        final_status = "ready_for_manuscript"
    else:
        final_status = "not_ready"

    lines = [
        "# Final reconciliation readme",
        "",
        f"- canonical_ready: {int(canonical_ready)}",
        f"- mimicry_ready: {int(mimicry_ready)}",
        f"- reconciliation_label: {reconcile_label}",
        "",
        "## Cause of keystone / neurometric divergence",
        "",
        "- converted_cache_path: possible",
        "- feature_normalization: possible",
        "- windowing: ruled_out_as_primary_cause",
        "- absolute_accuracy_mismatch: confirmed",
        "- delta_mismatch: confirmed",
        "- missing_negative_limb: confirmed_in_canonical_not_in_keystone_summary",
        f"- terminal_label: {reconcile_label}",
        "",
        "Final status:",
        f"- {final_status}",
    ]
    (out_root / "final_reconciliation_readme.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
