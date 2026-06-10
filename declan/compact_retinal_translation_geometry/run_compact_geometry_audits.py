#!/usr/bin/env python3
"""Audit compact retinal-translation geometry results against the spec."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_TFTS_ROOT = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
DEFAULT_CLOSURE_ROOT = (
    VISIONCORE_ROOT / "outputs" / "matched_twin_covariance_closure_rf_null_step025_rfbacked_v2"
)
DEFAULT_COMPACT_ROOT = VISIONCORE_ROOT / "outputs" / "compact_retinal_translation_geometry"
DEFAULT_DIRECT_ROOT = VISIONCORE_ROOT / "outputs" / "direct_recorded_derivative_twin_alignment_prod"


@dataclass(frozen=True)
class AuditPaths:
    tfts_root: Path
    closure_root: Path
    compact_root: Path
    direct_root: Path | None

    @property
    def tables_dir(self) -> Path:
        return self.compact_root / "tables"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not keys:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _finite(x: Any) -> float:
    try:
        val = float(x)
    except Exception:
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def _safe_mean(vals: np.ndarray) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_median(vals: np.ndarray) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _sign_test_p_two_sided(n_positive: int, n_total: int) -> float:
    if n_total <= 0:
        return float("nan")
    k = int(n_positive)
    n = int(n_total)
    if k == n / 2:
        return 1.0
    if k > n / 2:
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    else:
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    if vals.size == 1 or n_boot <= 0:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    boot = np.mean(vals[idx], axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _accept(
    rows: list[dict[str, Any]],
    *,
    analysis: str,
    check: str,
    status: str,
    metric: str = "",
    value: Any = "",
    threshold: str = "",
    details: str = "",
    claim_allowed: str = "",
    claim_to_avoid: str = "",
    source: str = "",
) -> None:
    rows.append(
        {
            "analysis": analysis,
            "check": check,
            "status": status,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "details": details,
            "claim_allowed": claim_allowed,
            "claim_to_avoid": claim_to_avoid,
            "source": source,
        }
    )


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("status", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _claim_decision(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_check = {(r["analysis"], r["check"]): str(r["status"]) for r in rows}
    core_geometry_checks = [
        ("Panel B compactness", "observed beats unit-shuffle samplewise null"),
        ("Panel C generalization", "k10 held-out capture beats null"),
        ("Panel E covariance closure", "full source beats unit-shuffle and RF/readout nulls"),
        ("Panel E covariance closure", "compact source comparable to full source"),
    ]
    metric_checks = [
        ("Metric structure", "local compact metrics rank 2"),
        ("Metric structure", "quadratic metric prediction"),
        ("Metric structure", "metric-normalized coordinate recovery"),
    ]
    core_ok = all(by_check.get(key) == "pass" for key in core_geometry_checks)
    metric_ok = all(by_check.get(key) == "pass" for key in metric_checks)
    hidden_ok = by_check.get(("Panel F decoding", "recorded displacement decoding promoted")) == "pass"
    if core_ok and metric_ok and hidden_ok:
        return {
            "decision": "strongest_outcome",
            "allowed_claim": "FEMs reveal a compact, metric-validated, readable retinal-translation geometry in foveal V1.",
            "avoid_claim": "",
        }
    if core_ok and metric_ok:
        return {
            "decision": "metric_positive_no_recorded_decoding",
            "allowed_claim": "FEM-linked shared variability is carried by a compact coordinate-like retinal-translation geometry that predicts recorded covariance.",
            "avoid_claim": "Do not claim a demonstrated recorded readable displacement code.",
        }
    if core_ok:
        return {
            "decision": "compact_but_not_metric_positive",
            "allowed_claim": "FEM-linked shared variability is carried by a compact covariance-predictive retinal-translation geometry.",
            "avoid_claim": "Do not claim coordinate-like compact geometry or hidden coordinate-system support until metric validation passes.",
        }
    return {
        "decision": "mixed_or_incomplete",
        "allowed_claim": "Treat current outputs as an audit/diagnostic bundle until blocking failures are resolved.",
        "avoid_claim": "Do not promote compact-geometry figure claims as fully spec-compliant.",
    }


def _audit_manifest_and_sessions(paths: AuditPaths, args: argparse.Namespace, accept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closure_manifest = _read_json(paths.closure_root / "run_manifest.json")
    closure_audit = _read_json(paths.closure_root / "finite_difference_provenance_audit.json")
    compact_manifest = _read_json(paths.compact_root / "compact_retinal_translation_geometry_manifest.json")
    sessions = _read_csv(paths.closure_root / "finite_difference_session_summary.csv")
    rf_bins = _read_csv(paths.closure_root / "rf_null_unit_bins.csv")

    _accept(
        accept_rows,
        analysis="Manifest/provenance",
        check="compact manifest exists",
        status="pass" if compact_manifest.get("status") == "ok" else "fail",
        metric="compact_manifest.status",
        value=compact_manifest.get("status", "missing"),
        threshold="ok",
        source=str(paths.compact_root / "compact_retinal_translation_geometry_manifest.json"),
    )
    _accept(
        accept_rows,
        analysis="Manifest/provenance",
        check="closure provenance exists",
        status="pass" if closure_audit.get("manifest_status") == "ok" else "fail",
        metric="closure_audit.manifest_status",
        value=closure_audit.get("manifest_status", "missing"),
        threshold="ok",
        source=str(paths.closure_root / "finite_difference_provenance_audit.json"),
    )

    projection_controls = set(str(v) for v in closure_manifest.get("projection_controls", []))
    required_controls = {"none", "global_rate", "target_pc1", "global_rate+target_pc1"}
    missing_controls = sorted(required_controls - projection_controls)
    _accept(
        accept_rows,
        analysis="Manifest/provenance",
        check="closure projection controls present",
        status="pass" if not missing_controls else "fail",
        metric="missing_projection_controls",
        value=";".join(missing_controls),
        threshold="none missing",
        details=f"available={sorted(projection_controls)}",
    )

    target_variants = set(str(v) for v in closure_manifest.get("target_variants", []))
    _accept(
        accept_rows,
        analysis="Manifest/provenance",
        check="raw and PSD targets reported",
        status="pass" if {"raw", "psd"}.issubset(target_variants) else "fail",
        metric="target_variants",
        value=";".join(sorted(target_variants)),
        threshold="raw;psd",
    )
    _accept(
        accept_rows,
        analysis="Manifest/provenance",
        check="random seeds recorded",
        status="pass" if ("random_seed" in compact_manifest and closure_manifest.get("n_nulls") is not None) else "warn",
        metric="compact_random_seed",
        value=compact_manifest.get("random_seed", "missing"),
        threshold="explicit seed in every analysis manifest",
        details="Closure manifest records null counts but not a top-level random_seed field." if "random_seed" not in closure_manifest else "",
    )

    session_rows: list[dict[str, Any]] = []
    if sessions.empty:
        _accept(
            accept_rows,
            analysis="Session/unit audit",
            check="session summary exists",
            status="fail",
            value="missing",
            source=str(paths.closure_root / "finite_difference_session_summary.csv"),
        )
        return session_rows

    n_ok = int(np.sum(sessions["status"].astype(str) == "ok"))
    _accept(
        accept_rows,
        analysis="Session/unit audit",
        check="minimum valid sessions",
        status="pass" if n_ok >= int(args.min_sessions) else "fail",
        metric="n_sessions_ok",
        value=n_ok,
        threshold=f">={int(args.min_sessions)}",
    )
    n_small = int(np.sum(sessions["n_common_units"].astype(float) < float(args.min_main_units)))
    demote_small = bool(getattr(args, "demote_small_sessions", False))
    _accept(
        accept_rows,
        analysis="Session/unit audit",
        check="main sessions have enough matched units",
        status="pass" if n_small == 0 or demote_small else "fail",
        metric="sessions_below_min_units",
        value=n_small,
        threshold=f"0 sessions below {int(args.min_main_units)} matched units",
        details=(
            f"{n_small} small sessions demoted from main-session status; retained as diagnostic sessions."
            if n_small and demote_small
            else "Strict spec gate; small sessions can be retained only if explicitly demoted from main-session status."
        ),
    )

    rf_status_ok = sessions["rf_null_status"].astype(str).str.startswith("ok_rf").all()
    spatial_meta_ok = False
    if not rf_bins.empty and "rf_coordinate_source" in rf_bins:
        spatial_meta_ok = rf_bins["rf_coordinate_source"].astype(str).str.contains("sta|rf|readout", case=False, regex=True).any()
    _accept(
        accept_rows,
        analysis="RF/readout bin audit",
        check="RF/readout null has spatial metadata",
        status="pass" if rf_status_ok and spatial_meta_ok else "fail",
        metric="rf_status_ok_and_spatial_metadata",
        value=f"{rf_status_ok};{spatial_meta_ok}",
        threshold="True;True",
        source=str(paths.closure_root / "rf_null_unit_bins.csv"),
    )
    largest = sessions["rf_null_largest_bin_fraction"].astype(float)
    n_large_bins = int(np.sum(largest > float(args.max_largest_bin_fraction)))
    _accept(
        accept_rows,
        analysis="RF/readout bin audit",
        check="largest RF bin fraction",
        status="pass" if n_large_bins == 0 else "warn",
        metric="sessions_largest_bin_fraction_above_threshold",
        value=n_large_bins,
        threshold=f"0 above {float(args.max_largest_bin_fraction):.2f}",
    )
    if not rf_bins.empty and "rf_null_bin" in rf_bins:
        bin_sizes = (
            rf_bins.groupby(["session", "rf_null_bin"], dropna=False)
            .size()
            .reset_index(name="bin_size")
        )
        min_bin = int(bin_sizes["bin_size"].min()) if len(bin_sizes) else 0
        _accept(
            accept_rows,
            analysis="RF/readout bin audit",
            check="minimum RF bin units after merging",
            status="pass" if min_bin >= int(args.min_bin_units) else "fail",
            metric="minimum_bin_size",
            value=min_bin,
            threshold=f">={int(args.min_bin_units)}",
        )
    else:
        bin_sizes = pd.DataFrame()
        _accept(
            accept_rows,
            analysis="RF/readout bin audit",
            check="minimum RF bin units after merging",
            status="fail",
            value="missing rf_null_bin table",
        )

    for row in sessions.itertuples(index=False):
        session_rows.append(
            {
                "session_id": row.session,
                "subject": row.subject,
                "status": row.status,
                "n_common_units": int(row.n_common_units),
                "main_session": bool(float(row.n_common_units) >= float(args.min_main_units)),
                "below_min_main_units": bool(float(row.n_common_units) < float(args.min_main_units)),
                "n_samples_used": int(row.n_samples_used),
                "target_trace_raw": float(row.target_trace_raw),
                "target_trace_psd": float(row.target_trace_psd),
                "target_negative_eigenvalue_mass_raw": float(row.target_negative_eigenvalue_mass_raw),
                "rf_null_status": row.rf_null_status,
                "rf_null_n_bins": int(row.rf_null_n_bins),
                "rf_null_largest_bin_fraction": float(row.rf_null_largest_bin_fraction),
                "rf_largest_bin_warn": bool(float(row.rf_null_largest_bin_fraction) > float(args.max_largest_bin_fraction)),
                "rf_null_bin_features": row.rf_null_bin_features,
                "compact_status": getattr(row, "compact_status", ""),
                "compact_min_train_rank": getattr(row, "compact_min_train_rank", ""),
                "compact_min_rank_used": getattr(row, "compact_min_rank_used", ""),
            }
        )
    return session_rows


def _audit_panel_b(paths: AuditPaths, args: argparse.Namespace, accept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = _read_csv(paths.compact_root / "panelB_participation_ratio_summary.csv")
    null_spectra = _read_csv(paths.compact_root / "panelB_null_spectra.csv")
    tfts_summary = _read_csv(paths.tfts_root / "union_spectrum" / "twin_tangent_union_summary.csv")
    rows: list[dict[str, Any]] = []
    if summary.empty:
        _accept(accept_rows, analysis="Panel B compactness", check="panelB summary exists", status="fail")
        return rows
    for row in summary.itertuples(index=False):
        pr = float(row.participation_ratio)
        null_lo = float(row.null_pr_ci_low)
        status = "pass" if pr < null_lo else "fail"
        rows.append(
            {
                "finite_difference_step_arcmin": float(row.finite_difference_step_arcmin),
                "projection_control": row.projection_control,
                "participation_ratio": pr,
                "unit_shuffle_null_pr_mean": float(row.null_pr_mean),
                "unit_shuffle_null_pr_ci_low": null_lo,
                "unit_shuffle_null_pr_ci_high": float(row.null_pr_ci_high),
                "beats_unit_shuffle_null": status == "pass",
                "n_units": int(row.n_units),
                "n_contexts": int(row.n_contexts),
            }
        )
    primary = next((r for r in rows if abs(float(r["finite_difference_step_arcmin"]) - float(args.primary_delta)) < 1e-9), None)
    _accept(
        accept_rows,
        analysis="Panel B compactness",
        check="observed beats unit-shuffle samplewise null",
        status="pass" if primary and primary["beats_unit_shuffle_null"] else "fail",
        metric="PR_vs_unit_shuffle_ci_low",
        value="" if primary is None else f"{primary['participation_ratio']:.4f} < {primary['unit_shuffle_null_pr_ci_low']:.4f}",
        threshold="observed PR below null 2.5 percentile",
        claim_allowed="Compact tangent spectrum beats unit-shuffle samplewise null.",
    )
    has_rf_samplewise = False
    if not null_spectra.empty and "null_type" in null_spectra:
        has_rf_samplewise = null_spectra["null_type"].astype(str).str.contains("rf", case=False).any()
    _accept(
        accept_rows,
        analysis="Panel B compactness",
        check="RF/readout samplewise compactness null",
        status="pass" if has_rf_samplewise else "not_run",
        metric="rf_samplewise_null_present",
        value=has_rf_samplewise,
        threshold="True if claiming RF/readout-preserving compactness null",
        claim_to_avoid="Do not say tangent compactness survived RF/readout-preserving samplewise null yet.",
    )
    projected = set(summary["projection_control"].astype(str))
    has_projection_sweep = {"none", "global_rate", "target_pc1", "global_rate+target_pc1"}.issubset(projected)
    _accept(
        accept_rows,
        analysis="Panel B compactness",
        check="projection-control sweep",
        status="pass" if has_projection_sweep else "not_run",
        metric="projection_controls",
        value=";".join(sorted(projected)),
        threshold="none;global_rate;target_pc1;global_rate+target_pc1",
        claim_to_avoid="Do not claim compact spectrum survives projection controls until recomputed under P.",
    )
    available_steps = sorted(float(v) for v in tfts_summary["delta"].dropna().unique()) if not tfts_summary.empty else []
    _accept(
        accept_rows,
        analysis="Panel B compactness",
        check="displacement step sweep",
        status="pass" if {0.125, 0.25, 0.5}.issubset(set(available_steps)) else "warn",
        metric="available_steps_arcmin",
        value=";".join(f"{v:g}" for v in available_steps),
        threshold="0.125;0.25;0.5 available, 1.0 optional if cache exists",
    )
    return rows


def _audit_panel_c(paths: AuditPaths, args: argparse.Namespace, accept_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = _read_csv(paths.compact_root / "panelC_cross_image_generalization_summary.csv")
    fold_audit = _read_csv(paths.tfts_root / "split_modes" / "image_disjoint" / "fold_leakage_audit.csv")
    history_audit = _read_csv(paths.tfts_root / "split_modes" / "image_disjoint" / "history_overlap_audit.csv")
    split_summary = _read_csv(paths.tfts_root / "split_modes" / "image_disjoint" / "split_mode_summary.csv")
    accept_detail_rows: list[dict[str, Any]] = []
    if summary.empty:
        _accept(accept_rows, analysis="Panel C generalization", check="panelC summary exists", status="fail")
        return [], accept_detail_rows
    k10 = summary[summary["k"].astype(int) == 10]
    k10_pass = False
    if len(k10):
        row = k10.iloc[0]
        k10_pass = float(row["capture_observed_mean"]) > float(row["capture_null_ci_high"])
    _accept(
        accept_rows,
        analysis="Panel C generalization",
        check="k10 held-out capture beats null",
        status="pass" if k10_pass else "fail",
        metric="k10_capture_vs_null_ci_high",
        value="" if not len(k10) else f"{float(k10.iloc[0]['capture_observed_mean']):.4f} > {float(k10.iloc[0]['capture_null_ci_high']):.4f}",
        threshold="observed mean > null high",
        claim_allowed="Compact basis generalizes to held-out images.",
    )
    captures = summary.sort_values("k")["capture_observed_mean"].to_numpy(dtype=np.float64)
    monotonic = bool(np.all(np.diff(captures) >= -1e-9)) if captures.size else False
    _accept(
        accept_rows,
        analysis="Panel C generalization",
        check="k-sweep monotonic",
        status="pass" if monotonic else "warn",
        metric="capture_observed_mean_by_k",
        value=";".join(f"{v:.4f}" for v in captures),
        threshold="nondecreasing",
    )
    shared_images = int(fold_audit["n_shared_image_ids_train_test"].sum()) if not fold_audit.empty else -1
    _accept(
        accept_rows,
        analysis="Panel C generalization",
        check="image-disjoint split leakage",
        status="pass" if shared_images == 0 else "fail",
        metric="shared_image_ids_train_test_total",
        value=shared_images,
        threshold="0",
    )
    hist_frac = float(split_summary.loc[np.isclose(split_summary["delta"].astype(float), float(args.primary_delta)), "fraction_test_objects_with_history_overlap_gt_0"].iloc[0]) if not split_summary.empty else float("nan")
    _accept(
        accept_rows,
        analysis="Panel C generalization",
        check="history-overlap leakage disclosed",
        status="pass" if np.isfinite(hist_frac) else "warn",
        metric="fraction_test_objects_with_history_overlap_gt_0",
        value="" if not np.isfinite(hist_frac) else f"{hist_frac:.4f}",
        threshold="reported; ideal is 0",
        details="Nonzero history overlap should be mentioned as a caveat or rerun with image_disjoint_history_gap.",
    )
    projection_controls = set(summary["projection_control"].astype(str))
    _accept(
        accept_rows,
        analysis="Panel C generalization",
        check="projection-control sweep",
        status="pass" if {"none", "global_rate", "target_pc1", "global_rate+target_pc1"}.issubset(projection_controls) else "not_run",
        metric="projection_controls",
        value=";".join(sorted(projection_controls)),
        threshold="all four projection controls",
    )
    for row in summary.itertuples(index=False):
        accept_detail_rows.append(
            {
                "k": int(row.k),
                "capture_observed_mean": float(row.capture_observed_mean),
                "capture_null_mean": float(row.capture_null_mean),
                "capture_null_ci_high": float(row.capture_null_ci_high),
                "beats_null": bool(float(row.capture_observed_mean) > float(row.capture_null_ci_high)),
                "split_mode": row.split_mode,
            }
        )
    history_rows = history_audit.to_dict("records") if not history_audit.empty else []
    return history_rows, accept_detail_rows


def _audit_panel_e(paths: AuditPaths, args: argparse.Namespace, accept_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = _read_csv(paths.compact_root / "panelE_covariance_closure_metrics.csv")
    raw_psd = _read_csv(paths.compact_root / "panelE_covariance_closure_raw_vs_psd.csv")
    session_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    if metrics.empty:
        _accept(accept_rows, analysis="Panel E covariance closure", check="closure metrics exist", status="fail")
        return session_rows, consistency_rows

    wanted_sources = {
        "full_finite_difference_source": "full source",
        "compact_k10_crossfit_source": "compact source",
    }
    summary_checks: dict[str, dict[str, float]] = {}
    for source, label in wanted_sources.items():
        block = metrics[
            (metrics["source_variant"].astype(str) == source)
            & (metrics["k"].astype(int) == 2)
            & (metrics["target_variant"].astype(str) == str(args.target_variant))
            & (metrics["projection_control"].astype(str) == str(args.projection_control))
        ].copy()
        if block.empty:
            continue
        observed = block.drop_duplicates(["session_id"])["capture_observed"].to_numpy(dtype=np.float64)
        summary_checks[source] = {"capture_mean": _safe_mean(observed)}
        for null_type in ["unit_shuffle", "rf_readout_fixed_permutation"]:
            nb = block[block["null_type"].astype(str) == null_type]
            effects = nb.drop_duplicates(["session_id"])["excess_over_null"].to_numpy(dtype=np.float64)
            mean, lo, hi = _bootstrap_mean_ci(effects, seed=int(args.seed) + len(session_rows) + len(null_type), n_boot=int(args.n_boot))
            n_pos = int(np.sum(effects > 0.0))
            p_sign = _sign_test_p_two_sided(n_pos, int(np.isfinite(effects).sum()))
            summary_checks[source][f"{null_type}_effect_mean"] = mean
            summary_checks[source][f"{null_type}_effect_ci_low"] = lo
            summary_checks[source][f"{null_type}_effect_ci_high"] = hi
            summary_checks[source][f"{null_type}_n_positive"] = n_pos
            for row in nb.itertuples(index=False):
                session_rows.append(
                    {
                        "session_id": row.session_id,
                        "source_variant": source,
                        "k": int(row.k),
                        "null_type": null_type,
                        "capture_observed": float(row.capture_observed),
                        "capture_null": float(row.capture_null),
                        "effect_session": float(row.excess_over_null),
                        "trace_target": float(row.trace_target),
                        "n_units": int(row.n_units),
                        "n_samples": int(row.n_samples),
                    }
                )
    full = summary_checks.get("full_finite_difference_source", {})
    compact = summary_checks.get("compact_k10_crossfit_source", {})
    full_pass = (
        full.get("unit_shuffle_effect_ci_low", float("nan")) > 0.0
        and full.get("rf_readout_fixed_permutation_effect_ci_low", float("nan")) > 0.0
    )
    _accept(
        accept_rows,
        analysis="Panel E covariance closure",
        check="full source beats unit-shuffle and RF/readout nulls",
        status="pass" if full_pass else "fail",
        metric="full_k2_effect_boot_ci_low",
        value=(
            f"unit={full.get('unit_shuffle_effect_ci_low', float('nan')):.4f};"
            f"rf={full.get('rf_readout_fixed_permutation_effect_ci_low', float('nan')):.4f}"
        ),
        threshold="both > 0",
        claim_allowed="Full finite-difference source predicts recorded FEM covariance above nulls.",
    )
    ratio = compact.get("capture_mean", float("nan")) / full.get("capture_mean", float("nan")) if full.get("capture_mean", 0.0) else float("nan")
    compact_pass = np.isfinite(ratio) and ratio >= float(args.compact_full_ratio_min)
    _accept(
        accept_rows,
        analysis="Panel E covariance closure",
        check="compact source comparable to full source",
        status="pass" if compact_pass else "fail",
        metric="compact_to_full_capture_ratio_k2",
        value="" if not np.isfinite(ratio) else f"{ratio:.4f}",
        threshold=f">={float(args.compact_full_ratio_min):.2f}",
        claim_allowed="Compact-restricted source retains covariance closure.",
    )
    compact_rf_pass = compact.get("rf_readout_fixed_permutation_effect_ci_low", float("nan")) > 0.0
    _accept(
        accept_rows,
        analysis="Panel E covariance closure",
        check="compact source beats RF/readout null",
        status="pass" if compact_rf_pass else "fail",
        metric="compact_k2_rf_effect_boot_ci_low",
        value=f"{compact.get('rf_readout_fixed_permutation_effect_ci_low', float('nan')):.4f}",
        threshold=">0",
    )

    if not raw_psd.empty:
        for (source, k), g in raw_psd.groupby(["source_variant", "k"]):
            pivot = g.pivot_table(index="session_id", columns="target_variant", values="capture_observed", aggfunc="first")
            if {"raw", "psd"}.issubset(set(pivot.columns)):
                raw_sign = np.sign(pivot["raw"].to_numpy(dtype=np.float64))
                psd_sign = np.sign(pivot["psd"].to_numpy(dtype=np.float64))
                mismatch = int(np.sum(raw_sign != psd_sign))
                consistency_rows.append(
                    {
                        "source_variant": source,
                        "k": int(k),
                        "n_sessions": int(len(pivot)),
                        "raw_psd_sign_mismatch_sessions": mismatch,
                        "raw_capture_mean": float(np.nanmean(pivot["raw"])),
                        "psd_capture_mean": float(np.nanmean(pivot["psd"])),
                    }
                )
        mismatch_total = int(sum(r["raw_psd_sign_mismatch_sessions"] for r in consistency_rows))
        _accept(
            accept_rows,
            analysis="Panel E covariance closure",
            check="raw and PSD target signs consistent",
            status="pass" if mismatch_total == 0 else "warn",
            metric="raw_psd_sign_mismatch_sessions_total",
            value=mismatch_total,
            threshold="0",
        )
    else:
        _accept(
            accept_rows,
            analysis="Panel E covariance closure",
            check="raw and PSD target signs consistent",
            status="fail",
            value="missing raw_vs_psd table",
        )
    return session_rows, consistency_rows


def _panel_e_effect_rows(
    metrics: pd.DataFrame,
    *,
    target_variant: str,
    projection_control: str,
    session_set_name: str,
    allowed_sessions: set[str] | None,
    k: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in ["full_finite_difference_source", "compact_k10_crossfit_source"]:
        block = metrics[
            (metrics["source_variant"].astype(str) == source)
            & (metrics["k"].astype(int) == int(k))
            & (metrics["target_variant"].astype(str) == str(target_variant))
            & (metrics["projection_control"].astype(str) == str(projection_control))
            & (metrics["null_type"].astype(str).isin(["unit_shuffle", "rf_readout_fixed_permutation"]))
        ].copy()
        if allowed_sessions is not None:
            block = block[block["session_id"].astype(str).isin(allowed_sessions)].copy()
        for row in block.itertuples(index=False):
            rows.append(
                {
                    "session_set": session_set_name,
                    "session_id": row.session_id,
                    "source_variant": source,
                    "k": int(row.k),
                    "null_type": row.null_type,
                    "capture_observed": float(row.capture_observed),
                    "capture_null": float(row.capture_null),
                    "effect_session": float(row.excess_over_null),
                    "trace_target": float(row.trace_target),
                    "n_units": int(row.n_units),
                    "n_samples": int(row.n_samples),
                }
            )
    return rows


def _summarize_panel_e_effects(rows: list[dict[str, Any]], *, seed: int, n_boot: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for (session_set, source, null_type), g in df.groupby(["session_set", "source_variant", "null_type"]):
        session_once = g.drop_duplicates(["session_id"])
        effects = session_once["effect_session"].to_numpy(dtype=np.float64)
        captures = session_once["capture_observed"].to_numpy(dtype=np.float64)
        effect_mean, effect_lo, effect_hi = _bootstrap_mean_ci(
            effects,
            seed=int(seed) + len(out) * 97,
            n_boot=int(n_boot),
        )
        capture_mean, capture_lo, capture_hi = _bootstrap_mean_ci(
            captures,
            seed=int(seed) + len(out) * 101,
            n_boot=int(n_boot),
        )
        n_pos = int(np.sum(effects > 0.0))
        n_eff = int(np.isfinite(effects).sum())
        out.append(
            {
                "session_set": session_set,
                "source_variant": source,
                "k": 2,
                "null_type": null_type,
                "n_sessions": int(session_once["session_id"].nunique()),
                "capture_mean": capture_mean,
                "capture_boot_ci_low": capture_lo,
                "capture_boot_ci_high": capture_hi,
                "effect_mean": effect_mean,
                "effect_boot_ci_low": effect_lo,
                "effect_boot_ci_high": effect_hi,
                "effect_median": _safe_median(effects),
                "n_effect_positive": n_pos,
                "n_effect_nonzero": n_eff,
                "sign_test_p_two_sided": _sign_test_p_two_sided(n_pos, n_eff),
            }
        )
    # Add compact/full ratio rows per session set using observed capture.
    for session_set, g in df.groupby("session_set"):
        capture_means = (
            g.drop_duplicates(["session_id", "source_variant"])
            .groupby("source_variant")["capture_observed"]
            .mean()
            .to_dict()
        )
        full = capture_means.get("full_finite_difference_source", float("nan"))
        compact = capture_means.get("compact_k10_crossfit_source", float("nan"))
        out.append(
            {
                "session_set": session_set,
                "source_variant": "compact_to_full_ratio",
                "k": 2,
                "null_type": "observed_capture",
                "n_sessions": int(g["session_id"].nunique()),
                "capture_mean": float(compact / full) if np.isfinite(compact) and np.isfinite(full) and full else float("nan"),
                "capture_boot_ci_low": float("nan"),
                "capture_boot_ci_high": float("nan"),
                "effect_mean": float("nan"),
                "effect_boot_ci_low": float("nan"),
                "effect_boot_ci_high": float("nan"),
                "effect_median": float("nan"),
                "n_effect_positive": "",
                "n_effect_nonzero": "",
                "sign_test_p_two_sided": float("nan"),
            }
        )
    return sorted(out, key=lambda r: (str(r["session_set"]), str(r["source_variant"]), str(r["null_type"])))


def _audit_panel_e_min_units(paths: AuditPaths, args: argparse.Namespace, accept_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = _read_csv(paths.compact_root / "panelE_covariance_closure_metrics.csv")
    sessions = _read_csv(paths.closure_root / "finite_difference_session_summary.csv")
    if metrics.empty or sessions.empty:
        _accept(
            accept_rows,
            analysis="Panel E min-unit closure",
            check="min-unit closure comparison",
            status="not_run",
            value="missing metrics or session summary",
        )
        return [], []
    main_sessions = set(
        sessions.loc[sessions["n_common_units"].astype(float) >= float(args.min_main_units), "session"].astype(str)
    )
    all_effect_rows = _panel_e_effect_rows(
        metrics,
        target_variant=str(args.target_variant),
        projection_control=str(args.projection_control),
        session_set_name="all_sessions",
        allowed_sessions=None,
    )
    main_effect_rows = _panel_e_effect_rows(
        metrics,
        target_variant=str(args.target_variant),
        projection_control=str(args.projection_control),
        session_set_name=f"main_min{int(args.min_main_units)}",
        allowed_sessions=main_sessions,
    )
    effect_rows = all_effect_rows + main_effect_rows
    summary_rows = _summarize_panel_e_effects(effect_rows, seed=int(args.seed) + 4242, n_boot=int(args.n_boot))
    main_summary = pd.DataFrame(summary_rows)
    n_main = len(main_sessions)
    if main_summary.empty:
        _accept(
            accept_rows,
            analysis="Panel E min-unit closure",
            check="main-set closure survives min-unit filter",
            status="fail",
            value="empty summary",
        )
        return effect_rows, summary_rows
    main_label = f"main_min{int(args.min_main_units)}"
    full_rf = main_summary[
        (main_summary["session_set"].astype(str) == main_label)
        & (main_summary["source_variant"].astype(str) == "full_finite_difference_source")
        & (main_summary["null_type"].astype(str) == "rf_readout_fixed_permutation")
    ]
    compact_rf = main_summary[
        (main_summary["session_set"].astype(str) == main_label)
        & (main_summary["source_variant"].astype(str) == "compact_k10_crossfit_source")
        & (main_summary["null_type"].astype(str) == "rf_readout_fixed_permutation")
    ]
    ratio_row = main_summary[
        (main_summary["session_set"].astype(str) == main_label)
        & (main_summary["source_variant"].astype(str) == "compact_to_full_ratio")
    ]
    full_ci_low = float(full_rf["effect_boot_ci_low"].iloc[0]) if len(full_rf) else float("nan")
    compact_ci_low = float(compact_rf["effect_boot_ci_low"].iloc[0]) if len(compact_rf) else float("nan")
    ratio = float(ratio_row["capture_mean"].iloc[0]) if len(ratio_row) else float("nan")
    survives = full_ci_low > 0.0 and compact_ci_low > 0.0 and ratio >= float(args.compact_full_ratio_min)
    _accept(
        accept_rows,
        analysis="Panel E min-unit closure",
        check="main-set closure survives min-unit filter",
        status="pass" if survives else "fail",
        metric="main_min_units_rf_effect_ci_low_and_ratio",
        value=f"n={n_main}; full_rf_ci_low={full_ci_low:.4f}; compact_rf_ci_low={compact_ci_low:.4f}; ratio={ratio:.4f}",
        threshold=f"n>={int(args.min_sessions)} preferred, RF CIs >0, ratio>={float(args.compact_full_ratio_min):.2f}",
        details=(
            "This is the spec-valid main-session set after demoting small sessions."
            if bool(getattr(args, "demote_small_sessions", False))
            else "Diagnostic min-unit subset; rerun with --demote-small-sessions to treat this as the main set."
        ),
    )
    return effect_rows, summary_rows


def _audit_panel_d(paths: AuditPaths, accept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budget = _read_csv(paths.compact_root / "panelD_variability_budget.csv")
    summary = _read_csv(paths.compact_root / "panelD_variability_budget_summary.csv")
    rows: list[dict[str, Any]] = []
    required = {
        "non_global_projected_FEM_target",
        "full_FEM_linked_covariance_raw_trace",
        "full_FEM_linked_covariance_psd_trace",
        "positive_shared_covariance_trace",
        "total_reliable_shared_covariance_trace",
        "total_trial_to_trial_covariance_trace",
        "split_half_reliability_ceiling_for_FEM_covariance",
    }
    available = set()
    if not budget.empty:
        ok = budget[budget["denominator_status"].astype(str).str.startswith("available")]
        available = set(ok["denominator_name"].astype(str).unique())
    missing = sorted(required - available)
    _accept(
        accept_rows,
        analysis="Panel D variability budget",
        check="required denominator inventory",
        status="pass" if not missing else "warn",
        metric="missing_denominators",
        value=";".join(missing),
        threshold="none missing for full budget acceptance",
        details="Current budget can support full-FEM/projected-target context but not reliable-shared or total-variance context." if missing else "",
    )
    if not summary.empty:
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "denominator_name": row.denominator_name,
                    "source_variant": row.source_variant,
                    "n_sessions": int(row.n_sessions),
                    "fraction_of_denominator_mean": float(row.fraction_of_denominator_mean),
                    "null_adjusted_fraction_mean": float(row.null_adjusted_fraction_mean),
                }
            )
    has_compact_full = {
        "full_FEM_linked_covariance_raw_trace",
        "full_FEM_linked_covariance_psd_trace",
        "non_global_projected_FEM_target",
    }.issubset(available)
    _accept(
        accept_rows,
        analysis="Panel D variability budget",
        check="compact contribution contextualized against FEM target",
        status="pass" if has_compact_full else "fail",
        metric="available_core_denominators",
        value=";".join(sorted(available)),
        threshold="projected target and full raw/PSD FEM traces",
    )
    return rows


def _load_tangent_payload(path: Path) -> tuple[list[float], dict[float, dict[str, dict[str, Any]]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    deltas = [float(v) for v in payload["delta_arcmins"]]
    objects = {
        float(delta): {str(oid): meta for oid, meta in obj.items()}
        for delta, obj in payload["object_payload"].items()
    }
    return deltas, objects


def _orth(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(x, dtype=np.float64))
    keep = np.abs(np.diag(r)) > eps
    return q[:, keep]


def _compact_basis(payload: dict[str, dict[str, Any]], k: int) -> np.ndarray:
    object_ids = [
        oid
        for oid, meta in sorted(payload.items())
        if all(np.asarray(meta[name]).ndim == 1 and np.all(np.isfinite(meta[name])) for name in ("bx", "by"))
    ]
    bx = np.stack([np.asarray(payload[oid]["bx"], dtype=np.float64) for oid in object_ids], axis=1)
    by = np.stack([np.asarray(payload[oid]["by"], dtype=np.float64) for oid in object_ids], axis=1)
    b = np.concatenate([bx, by], axis=1)
    vals, vecs = np.linalg.eigh(0.5 * ((b @ b.T) + (b @ b.T).T))
    order = np.argsort(vals)[::-1]
    return vecs[:, order[: int(k)]]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def _audit_metric_structure(paths: AuditPaths, args: argparse.Namespace, accept_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    promoted_summary_path = paths.compact_root / "metric_structure_summary.csv"
    promoted_opposition_path = paths.compact_root / "metric_structure_opposition.csv"
    promoted_scaling_path = paths.compact_root / "metric_structure_scaling.csv"
    if promoted_summary_path.exists():
        summary_df = _read_csv(promoted_summary_path)
        opposition_df = _read_csv(promoted_opposition_path)
        scaling_df = _read_csv(promoted_scaling_path)
        by_test = {
            str(row.test): row
            for row in summary_df.itertuples(index=False)
        }

        def add_metric_check(test: str, check: str, threshold: str = "") -> None:
            row = by_test.get(test)
            if row is None:
                _accept(
                    accept_rows,
                    analysis="Metric structure",
                    check=check,
                    status="not_run",
                    metric=test,
                    value="missing",
                    threshold=threshold,
                )
                return
            _accept(
                accept_rows,
                analysis="Metric structure",
                check=check,
                status=str(getattr(row, "status")),
                metric=test,
                value=getattr(row, "metric_value", ""),
                threshold=threshold or str(getattr(row, "threshold", "")),
                details=str(getattr(row, "details", "")),
            )

        add_metric_check(
            "local_metric_rank2_fraction",
            "local compact metrics rank 2",
            "rank-2 fraction meets metric-validation threshold",
        )
        add_metric_check(
            "local_metric_conditioning",
            "local compact metric conditioning",
            "median condition number acceptable",
        )
        add_metric_check(
            "quadratic_step_sweep_prediction_r2",
            "quadratic metric prediction",
            "primary-step metric predicts held-out step-sweep squared distances",
        )
        add_metric_check(
            "quadratic_direction_heldout_prediction",
            "direction-held-out metric prediction",
            "requires diagonal/arbitrary translated responses",
        )
        add_metric_check(
            "opposition_vs_null",
            "opposite shifts are opposite in compact basis",
            "compact median exceeds random and unit-shuffled compact medians",
        )
        add_metric_check(
            "scaling_norm_r2",
            "magnitude scales with displacement",
            "norm increases predictably across finite-difference step sweep",
        )
        add_metric_check(
            "scaling_metric_squared_distance_r2",
            "metric predicts squared-distance scaling",
            "metric-predicted squared distances track actual squared distances across steps",
        )
        add_metric_check(
            "local_composition",
            "local composition/diagonal translations",
            "requires diagonal translated responses",
        )
        add_metric_check(
            "coordinate_recovery_step_sweep_r2",
            "metric-normalized coordinate recovery",
            "local compact Jacobian recovers finite step-sweep displacements",
        )
        add_metric_check(
            "cross_image_metric_regularity",
            "cross-image metric regularity reported",
            "reported, not required to be universal",
        )
        return (
            opposition_df.to_dict("records") if not opposition_df.empty else [],
            scaling_df.to_dict("records") if not scaling_df.empty else [],
            summary_df.to_dict("records") if not summary_df.empty else [],
        )

    tangent_path = paths.tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"
    if not tangent_path.exists():
        _accept(accept_rows, analysis="Metric structure", check="translated responses available", status="not_run", value="missing tangent maps")
        return [], [], []
    deltas, payload_by_delta = _load_tangent_payload(tangent_path)
    primary_delta = min(deltas, key=lambda d: abs(d - float(args.primary_delta)))
    primary_payload = payload_by_delta[primary_delta]
    u = _compact_basis(primary_payload, int(args.metric_structure_k))
    rng = np.random.default_rng(int(args.seed))
    random_basis = _orth(rng.normal(size=(u.shape[0], int(args.metric_structure_k))))
    perm = rng.permutation(u.shape[0])
    unit_shuffle_basis = _orth(u[perm, :])

    opposition_rows: list[dict[str, Any]] = []
    for oid, meta in sorted(primary_payload.items()):
        required = ("r0", "rx_p", "rx_m", "ry_p", "ry_m")
        if not all(name in meta and np.all(np.isfinite(meta[name])) for name in required):
            continue
        r0 = np.asarray(meta["r0"], dtype=np.float64)
        for axis, pos_name, neg_name in [("x", "rx_p", "rx_m"), ("y", "ry_p", "ry_m")]:
            pos = np.asarray(meta[pos_name], dtype=np.float64) - r0
            neg = np.asarray(meta[neg_name], dtype=np.float64) - r0
            for feature_space, basis in [
                ("compact_k10", u),
                ("random_k10", random_basis),
                ("unit_shuffled_compact_k10", unit_shuffle_basis),
                ("full_population", np.eye(u.shape[0])),
            ]:
                zp = basis.T @ pos if basis.shape[0] == pos.shape[0] else pos
                zn = basis.T @ neg if basis.shape[0] == neg.shape[0] else neg
                opposition_rows.append(
                    {
                        "object_id": oid,
                        "image_id": int(meta.get("image_id", -1)),
                        "axis": axis,
                        "feature_space": feature_space,
                        "finite_difference_step_arcmin": primary_delta,
                        "opposition_cosine": _cos(zp, -zn),
                        "norm_positive": float(np.linalg.norm(zp)),
                        "norm_negative": float(np.linalg.norm(zn)),
                    }
                )

    scaling_rows: list[dict[str, Any]] = []
    object_sets = [set(payload_by_delta[d].keys()) for d in deltas]
    common_objects = sorted(set.intersection(*object_sets)) if object_sets else []
    for oid in common_objects:
        for axis, pos_name, neg_name in [("x", "rx_p", "rx_m"), ("y", "ry_p", "ry_m")]:
            for direction, shift_name in [("positive", pos_name), ("negative", neg_name)]:
                xs: list[float] = []
                ys: list[float] = []
                for delta in sorted(deltas):
                    meta = payload_by_delta[delta][oid]
                    if not all(name in meta and np.all(np.isfinite(meta[name])) for name in ("r0", shift_name)):
                        continue
                    dz = u.T @ (np.asarray(meta[shift_name], dtype=np.float64) - np.asarray(meta["r0"], dtype=np.float64))
                    xs.append(float(delta))
                    ys.append(float(np.linalg.norm(dz)))
                if len(xs) < 3:
                    continue
                x = np.asarray(xs, dtype=np.float64)
                y = np.asarray(ys, dtype=np.float64)
                slope, intercept = np.polyfit(x, y, deg=1)
                pred = slope * x + intercept
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                scaling_rows.append(
                    {
                        "object_id": oid,
                        "axis": axis,
                        "direction": direction,
                        "feature_space": f"compact_k{int(args.metric_structure_k)}",
                        "n_steps": len(xs),
                        "steps_arcmin": ";".join(f"{v:g}" for v in xs),
                        "norms": ";".join(f"{v:.6g}" for v in ys),
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r2": float(r2),
                        "monotonic_increasing": bool(np.all(np.diff(y) >= -1e-12)),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    opp_df = pd.DataFrame(opposition_rows)
    if not opp_df.empty:
        compact = opp_df[opp_df["feature_space"] == "compact_k10"]["opposition_cosine"].to_numpy(dtype=np.float64)
        random = opp_df[opp_df["feature_space"] == "random_k10"]["opposition_cosine"].to_numpy(dtype=np.float64)
        shuffled = opp_df[opp_df["feature_space"] == "unit_shuffled_compact_k10"]["opposition_cosine"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "test": "opposite_shifts_are_opposite",
                "status": "pass" if _safe_median(compact) > _safe_median(random) and _safe_median(compact) > _safe_median(shuffled) else "warn",
                "compact_median": _safe_median(compact),
                "random_median": _safe_median(random),
                "unit_shuffle_median": _safe_median(shuffled),
                "n_object_axis_rows": int(np.isfinite(compact).sum()),
            }
        )
        _accept(
            accept_rows,
            analysis="Metric structure",
            check="opposite shifts are opposite in compact basis",
            status=str(summary_rows[-1]["status"]),
            metric="median_opposition_cosine",
            value=f"compact={summary_rows[-1]['compact_median']:.4f}; random={summary_rows[-1]['random_median']:.4f}; shuffled={summary_rows[-1]['unit_shuffle_median']:.4f}",
            threshold="compact > random and unit-shuffled compact",
        )
    else:
        _accept(accept_rows, analysis="Metric structure", check="opposite shifts are opposite in compact basis", status="not_run")

    scale_df = pd.DataFrame(scaling_rows)
    if not scale_df.empty:
        r2 = scale_df["r2"].to_numpy(dtype=np.float64)
        mono_frac = float(np.mean(scale_df["monotonic_increasing"].astype(bool)))
        summary_rows.append(
            {
                "test": "magnitude_scales_with_displacement",
                "status": "pass" if _safe_median(r2) >= float(args.metric_structure_min_r2) and mono_frac >= 0.5 else "warn",
                "median_r2": _safe_median(r2),
                "monotonic_fraction": mono_frac,
                "n_object_axis_direction_rows": int(len(scale_df)),
            }
        )
        _accept(
            accept_rows,
            analysis="Metric structure",
            check="magnitude scales with displacement",
            status=str(summary_rows[-1]["status"]),
            metric="median_r2;monotonic_fraction",
            value=f"{summary_rows[-1]['median_r2']:.4f};{mono_frac:.4f}",
            threshold=f"median_r2>={float(args.metric_structure_min_r2):.2f} and monotonic_fraction>=0.5",
        )
    else:
        _accept(accept_rows, analysis="Metric structure", check="magnitude scales with displacement", status="not_run")

    has_diagonal = any(
        any(str(key).lower() in {"rxy_p", "rxy_m", "rdiag_p", "rdiag_m"} for key in meta.keys())
        for meta in primary_payload.values()
    )
    _accept(
        accept_rows,
        analysis="Metric structure",
        check="local composition/diagonal translations",
        status="pass" if has_diagonal else "not_run",
        metric="diagonal_translation_responses_present",
        value=has_diagonal,
        threshold="True for composition test",
    )
    summary_rows.append(
        {
            "test": "local_composition",
            "status": "not_run" if not has_diagonal else "ready",
            "reason": "diagonal translated responses are not present in current tangent-map cache" if not has_diagonal else "",
        }
    )
    return opposition_rows, scaling_rows, summary_rows


def _audit_decoding_and_direct(paths: AuditPaths, accept_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decoding_root = paths.compact_root / "relative_displacement_decoding"
    decoding_audit = _read_json(decoding_root / "audit.json")
    feature_cmp = _read_csv(decoding_root / "feature_space_comparison.csv")
    bootstrap = _read_csv(decoding_root / "decoder_bootstrap_summary.csv")
    leakage = _read_csv(decoding_root / "split_leakage_audit.csv")
    decoding_rows: list[dict[str, Any]] = []
    if decoding_audit.get("status") == "ok" and not feature_cmp.empty:
        row0 = feature_cmp.iloc[0].to_dict()
        full_r2 = _finite(row0.get("full_population_R2_mean"))
        compact_r2 = _finite(row0.get("compact_R2_mean"))
        compact_fraction = _finite(row0.get("compact_fraction_of_full"))
        compact_minus_orth = _finite(row0.get("compact_minus_orthogonal"))
        compact_minus_random = _finite(row0.get("compact_minus_random"))
        compact_minus_rf = _finite(row0.get("compact_minus_rf_readout"))
        leakage_failures = int((leakage.get("status", pd.Series(dtype=str)).astype(str) == "fail").sum()) if not leakage.empty else 0
        if not leakage.empty and "n_shared_trials" in leakage:
            trial_overlap_folds = int((pd.to_numeric(leakage["n_shared_trials"], errors="coerce").fillna(0) > 0).sum())
            trial_overlap_status = "pass" if trial_overlap_folds == 0 else "warn"
        else:
            trial_overlap_folds = -1
            trial_overlap_status = "missing"
        primary_projection = str(row0.get("projection_control", decoding_audit.get("primary_projection_control", "")))
        primary_k = int(_finite(row0.get("primary_k", decoding_audit.get("primary_k", 10))))
        compact_summary = bootstrap[
            (bootstrap.get("feature_space", pd.Series(dtype=str)).astype(str) == "compact")
            & (bootstrap.get("projection_control", pd.Series(dtype=str)).astype(str) == primary_projection)
            & (bootstrap.get("k", pd.Series(dtype=float)).astype(float) == float(primary_k))
        ] if not bootstrap.empty else pd.DataFrame()
        eye_ci_low = float("nan")
        if not compact_summary.empty and "effect_minus_eye_label_shuffle_boot_ci_low" in compact_summary:
            eye_ci_low = _finite(compact_summary.iloc[0].get("effect_minus_eye_label_shuffle_boot_ci_low"))
        primary_pass = (
            np.isfinite(compact_r2)
            and compact_r2 > 0.0
            and np.isfinite(compact_fraction)
            and compact_fraction >= 0.5
            and np.isfinite(compact_minus_orth)
            and compact_minus_orth > 0.0
            and np.isfinite(compact_minus_random)
            and compact_minus_random > 0.0
            and (not np.isfinite(compact_minus_rf) or compact_minus_rf > 0.0)
            and leakage_failures == 0
            and trial_overlap_status == "pass"
            and (not np.isfinite(eye_ci_low) or eye_ci_low > 0.0)
        )
        status = "pass" if primary_pass else "warn"
        decoding_rows.append(
            {
                "status": status,
                "decision": decoding_audit.get("decision", ""),
                "primary_projection_control": primary_projection,
                "primary_k": int(primary_k),
                "full_population_R2_mean": full_r2,
                "compact_R2_mean": compact_r2,
                "compact_fraction_of_full": compact_fraction,
                "compact_minus_orthogonal": compact_minus_orth,
                "compact_minus_random": compact_minus_random,
                "compact_minus_rf_readout": compact_minus_rf,
                "compact_effect_eye_label_shuffle_ci_low": eye_ci_low,
                "n_sessions_ok": decoding_audit.get("n_sessions_ok", ""),
                "n_leakage_failures": leakage_failures,
                "trial_overlap_status": trial_overlap_status,
                "n_trial_overlap_folds": trial_overlap_folds,
                "source": str(decoding_root),
            }
        )
        _accept(
            accept_rows,
            analysis="Panel F decoding",
            check="recorded displacement decoding promoted",
            status=status,
            metric="compact_R2;fraction_full;minus_orth;minus_random;minus_rf",
            value=f"{compact_r2:.4g};{compact_fraction:.4g};{compact_minus_orth:.4g};{compact_minus_random:.4g};{compact_minus_rf:.4g}",
            threshold="compact>0, fraction>=0.5, compact beats orthogonal/random/RF controls, no condition or trial leakage",
            details=f"full_R2={full_r2:.4g}; eye_shuffle_CI_low={eye_ci_low:.4g}; leakage_failures={leakage_failures}; trial_overlap_status={trial_overlap_status}; n_trial_overlap_folds={trial_overlap_folds}",
            claim_allowed="Recorded V1 contains a compact, image-conditioned relative-displacement signal." if primary_pass else "",
            claim_to_avoid="" if primary_pass else "Do not promote readable compact displacement decoding; treat as diagnostic/mixed, especially if trial-overlap audit is missing or nonzero.",
            source=str(decoding_root),
        )
    else:
        decoding_rows = [
            {
                "status": "not_run",
                "required_objects": "recorded repeat-pair responses, same-condition eye differences, condition-disjoint folds",
                "blocking_reason": "No relative-displacement decoding metrics exist in compact geometry output.",
            }
        ]
        _accept(
            accept_rows,
            analysis="Panel F decoding",
            check="recorded displacement decoding promoted",
            status="not_run",
            metric="relative_displacement_decoding/feature_space_comparison.csv",
            value="missing",
            threshold="present and passes all nulls",
            claim_to_avoid="Do not claim readable image-generalizing displacement coordinate.",
        )

    direct_rows: list[dict[str, Any]] = []
    if paths.direct_root is not None and paths.direct_root.exists():
        audit = _read_json(paths.direct_root / "audit.json")
        tier1 = _read_csv(paths.direct_root / "tier1_compact_basis_bootstrap_summary.csv")
        direct_rows.append(
            {
                "direct_root": str(paths.direct_root),
                "audit_status": audit.get("status", "missing"),
                "n_sessions_ok": audit.get("n_sessions_ok", ""),
                "n_sessions_requested": audit.get("n_sessions_requested", ""),
                "n_context_rows": audit.get("n_context_rows", ""),
                "role": "supportive_not_main",
            }
        )
        if not tier1.empty:
            k10 = tier1[
                (tier1["target_variant"].astype(str) == "psd")
                & (tier1["projection_control"].astype(str) == "global_rate+target_pc1")
                & (tier1["k"].astype(int) == 10)
                & (tier1["context_subset"].astype(str) == "reliability_qualified")
            ]
            if len(k10):
                row = k10.iloc[0]
                direct_rows[-1].update(
                    {
                        "tier1_k10_capture_mean": float(row["capture_mean"]),
                        "tier1_k10_rf_effect_mean": float(row["effect_rf_readout_mean"]),
                        "tier1_k10_rf_effect_ci_low": float(row["effect_rf_readout_boot_ci_low"]),
                        "tier1_k10_n_sessions": int(row["n_sessions"]),
                    }
                )
                _accept(
                    accept_rows,
                    analysis="Direct recorded derivative",
                    check="supportive compact-basis capture available",
                    status="pass" if float(row["effect_rf_readout_boot_ci_low"]) > 0 else "warn",
                    metric="tier1_k10_rf_effect_boot_ci_low",
                    value=f"{float(row['effect_rf_readout_boot_ci_low']):.4f}",
                    threshold=">0 for supportive result",
                    details="Supportive only; does not supersede covariance closure.",
                )
            else:
                _accept(accept_rows, analysis="Direct recorded derivative", check="supportive compact-basis capture available", status="not_run")
    else:
        _accept(accept_rows, analysis="Direct recorded derivative", check="supportive compact-basis capture available", status="not_run")
    return decoding_rows, direct_rows


def _copy_structured_outputs(paths: AuditPaths, generated: dict[str, list[dict[str, Any]]]) -> None:
    for name, rows in generated.items():
        _write_csv(paths.tables_dir / f"{name}.csv", rows)


def run(args: argparse.Namespace) -> None:
    paths = AuditPaths(
        tfts_root=Path(args.tfts_root),
        closure_root=Path(args.closure_root),
        compact_root=Path(args.compact_root),
        direct_root=None if str(args.direct_root).lower() in {"", "none"} else Path(args.direct_root),
    )
    paths.compact_root.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    accept_rows: list[dict[str, Any]] = []
    generated: dict[str, list[dict[str, Any]]] = {}
    generated["session_summary"] = _audit_manifest_and_sessions(paths, args, accept_rows)
    generated["panelB_compactness_audit"] = _audit_panel_b(paths, args, accept_rows)
    history_rows, generalization_rows = _audit_panel_c(paths, args, accept_rows)
    generated["panelC_split_leakage_audit"] = history_rows
    generated["panelC_generalization_acceptance"] = generalization_rows
    session_effects, raw_psd = _audit_panel_e(paths, args, accept_rows)
    generated["panelE_session_effects"] = session_effects
    generated["panelE_raw_psd_consistency"] = raw_psd
    min50_effects, min50_summary = _audit_panel_e_min_units(paths, args, accept_rows)
    generated["panelE_session_effects_min50"] = min50_effects
    generated["panelE_covariance_closure_min50_summary"] = min50_summary
    generated["panelD_denominator_inventory"] = _audit_panel_d(paths, accept_rows)
    opposition, scaling, metric_summary = _audit_metric_structure(paths, args, accept_rows)
    generated["metric_structure_opposition"] = opposition
    generated["metric_structure_scaling"] = scaling
    generated["metric_structure_summary"] = metric_summary
    decoding, direct = _audit_decoding_and_direct(paths, accept_rows)
    generated["panelF_decoding_readiness"] = decoding
    generated["direct_recorded_derivative_audit"] = direct

    _copy_structured_outputs(paths, generated)
    _write_csv(paths.tables_dir / "acceptance_matrix.csv", accept_rows)

    decision = _claim_decision(accept_rows)
    audit = {
        "status": "ok",
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "input_cache_paths": {
            "tfts_root": str(paths.tfts_root.resolve()),
            "closure_root": str(paths.closure_root.resolve()),
            "compact_root": str(paths.compact_root.resolve()),
            "direct_root": "" if paths.direct_root is None else str(paths.direct_root.resolve()),
        },
        "thresholds": {
            "min_sessions": int(args.min_sessions),
            "min_main_units": int(args.min_main_units),
            "min_bin_units": int(args.min_bin_units),
            "max_largest_bin_fraction": float(args.max_largest_bin_fraction),
            "primary_delta_arcmin": float(args.primary_delta),
            "projection_control": str(args.projection_control),
            "target_variant": str(args.target_variant),
            "demote_small_sessions": bool(args.demote_small_sessions),
        },
        "acceptance_status_counts": _status_counts(accept_rows),
        "claim_decision": decision,
        "blocking_failures": [
            row for row in accept_rows if str(row.get("status")) == "fail"
        ],
        "not_run_checks": [
            row for row in accept_rows if str(row.get("status")) == "not_run"
        ],
        "warnings": [
            row for row in accept_rows if str(row.get("status")) == "warn"
        ],
        "tables": sorted(f"tables/{name}.csv" for name in generated) + ["tables/acceptance_matrix.csv"],
    }
    _write_json(paths.compact_root / "audit.json", audit)
    print(f"Wrote compact geometry audit to {paths.compact_root / 'audit.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    parser.add_argument("--closure-root", type=Path, default=DEFAULT_CLOSURE_ROOT)
    parser.add_argument("--compact-root", type=Path, default=DEFAULT_COMPACT_ROOT)
    parser.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    parser.add_argument("--primary-delta", type=float, default=0.25)
    parser.add_argument("--target-variant", choices=["raw", "psd"], default="psd")
    parser.add_argument(
        "--projection-control",
        choices=["none", "global_rate", "target_pc1", "global_rate+target_pc1"],
        default="global_rate+target_pc1",
    )
    parser.add_argument("--min-sessions", type=int, default=20)
    parser.add_argument("--min-main-units", type=int, default=50)
    parser.add_argument(
        "--demote-small-sessions",
        action="store_true",
        help="Treat sessions below --min-main-units as diagnostic rather than main-claim sessions.",
    )
    parser.add_argument("--min-bin-units", type=int, default=6)
    parser.add_argument("--max-largest-bin-fraction", type=float, default=0.30)
    parser.add_argument("--compact-full-ratio-min", type=float, default=0.90)
    parser.add_argument("--metric-structure-k", type=int, default=10)
    parser.add_argument("--metric-structure-min-r2", type=float, default=0.50)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
