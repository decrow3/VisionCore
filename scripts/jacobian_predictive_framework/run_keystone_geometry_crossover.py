#!/usr/bin/env python3
"""Keystone geometry crossover analysis — patched v2.

Tests whether cloud-separability gain G_sep predicts the FEM acuity crossover.

Tiers:
  cloud_separability_crosswalk  (Tier 1): G_sep from mean responses.
  jacobian_mechanism            (Tier 2): ΔM from raw J matrices — deferred until computed.

Steps:
  0  - Cache audit + grid reconciliation  (also: --cache-audit-only)
  1  - Geometry curves (G_sep, S_center, S_stab, S_cloud_mean)
  2  - Function curves (D1 accuracy, explicit source required)
  3  - Test 1: coincidence (dL_median < grid_step AND dL_ci_high < grid_step)
  4  - Test 2: continuous regression
  5  - Test 3: specificity (separate for each difficulty control)
  6  - Test 4: Tier-2 mechanism — deferred if J not available
  7  - Pairwise (coincidence + specificity per pair, separately reported)
  8  - Nulls (phase-shuffle fixed alignment, true pair-label-shuffle, no isotropic)
  9  - Window robustness (disabled if only aggregate accuracy available)
  10 - Tables, figures, decision label, readme
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from VisionCore.paths import VISIONCORE_ROOT

ROOT = VISIONCORE_ROOT
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EPS = 1e-12
ORIENTATIONS = (0, 90, 180, 270)
PAIRS = [(a, b) for i, a in enumerate(ORIENTATIONS) for b in ORIENTATIONS[i + 1:]]

DEFAULT_LOGMAR_GRID = (-0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10)
DEFAULT_RENDER_LIMIT_CONTROL = -0.40
DEFAULT_CORE_RANGE = (-0.35, -0.20)
DEFAULT_PRIMARY_WINDOW = 60
DEFAULT_WINDOWS = (12, 30, 60)
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_RANDOM_SEED = 0
DEFAULT_COINCIDENCE_GRID_STEP = 0.05
DEFAULT_SPECIFICITY_DELTA_R2 = 0.15

# The exact column in eoptotype_identity_decoder_metrics.csv to use for D1 accuracy.
# Must match the mean-rate (D1) readout.  No silent fallback.
DEFAULT_D1_ACC_COL = "rate_normalized_decoder_accuracy"
DEFAULT_FEATURE_REPR = "spatial_avg_time_mean"
DEFAULT_PRIMARY_DIFFICULTY = "S_stab"   # one of: S_center, S_stab, S_cloud_mean

DEFAULT_ACCURACY_SCAN_DIRS = (
    "active_sensing_efficiency_e1_fullscope_20260531",
    "active_sensing_efficiency_e1_d1_reconciliation_20260601",
)
# Explicit D1 sweep source — preferred over decoder_metrics when provided.
DEFAULT_D1_SWEEP_PATH = (
    ROOT / "outputs/jacobian_predictive_framework"
    / "active_sensing_efficiency_e1_d1_reconciliation_20260601"
    / "eoptotype_D1_integration_window_sweep.csv"
)
DEFAULT_D1_SWEEP_COL = "real_minus_stabilized_d1_time_mean_accuracy"
DEFAULT_D1_SWEEP_READOUT = "linear"
DEFAULT_JACOBIAN_SMOOTHNESS_DIR = (
    ROOT / "outputs/stats/eoptotype_jacobian_field_smoothness_pilot3"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_lm(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text((",".join(fieldnames or [])) + "\n")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _pair_label(a: int, b: int) -> str:
    return f"{a}_vs_{b}"


def _interp_zero(xs: np.ndarray, ys: np.ndarray) -> tuple[float, str]:
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if xs.size < 2:
        return float("nan"), "insufficient_points"
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    for i in range(len(xs) - 1):
        if ys[i] == 0.0:
            return float(xs[i]), "exact"
        if ys[i] * ys[i + 1] < 0.0:
            a = -ys[i] / (ys[i + 1] - ys[i])
            return float(xs[i] + a * (xs[i + 1] - xs[i])), "linear"
    return float("nan"), "no_crossing"


def _zero_crossing_core(Ls: np.ndarray, ys: np.ndarray, in_core: np.ndarray) -> tuple[float, str]:
    mask = in_core & np.isfinite(Ls) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan"), "insufficient_core_points"
    return _interp_zero(Ls[mask], ys[mask])


def _pairwise_mean_sep(class_means: dict[int, np.ndarray]) -> float:
    seps = [float(np.linalg.norm(class_means[a] - class_means[b]))
            for a, b in PAIRS if a in class_means and b in class_means]
    return float(np.nanmean(seps)) if seps else float("nan")


def _pairwise_seps(class_means: dict[int, np.ndarray]) -> dict[tuple[int, int], float]:
    return {
        (a, b): float(np.linalg.norm(class_means[a] - class_means[b]))
        if a in class_means and b in class_means else float("nan")
        for a, b in PAIRS
    }


# ---------------------------------------------------------------------------
# Step 0 — Cache audit
# ---------------------------------------------------------------------------

def _check_d1_col_exists(path: Path, col: str) -> bool:
    """Return True iff col is a column header in the CSV."""
    try:
        with path.open("r", newline="") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])
        return col in headers
    except Exception:
        return False


def _scan_response_vectors(
    jpf_base: Path,
    scan_dirs: tuple[str, ...],
    logmar_grid: tuple[float, ...],
    feature_repr: str,
) -> tuple[list[dict], dict, dict]:
    report: list[dict] = []
    merged: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    merged_tidx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    vec_key = f"vectors__{feature_repr}"

    for sd in scan_dirs:
        p = jpf_base / sd / "eoptotype_identity" / "eoptotype_response_vectors.npz"
        if not p.exists():
            for lm in logmar_grid:
                report.append({"quantity": "response_vectors", "logmar": _fmt_lm(lm),
                                "source_path": str(p), "on_target_grid": "no_file",
                                "n_trials": 0, "needs_recompute": "yes",
                                "recompute_cost_note": f"missing: {sd}"})
            continue
        d = np.load(str(p), allow_pickle=True)
        if vec_key not in d:
            for lm in logmar_grid:
                report.append({"quantity": "response_vectors", "logmar": _fmt_lm(lm),
                                "source_path": str(p), "on_target_grid": "wrong_feature",
                                "n_trials": 0, "needs_recompute": "yes",
                                "recompute_cost_note": f"key '{vec_key}' missing"})
            continue
        vecs = d[vec_key]
        d_cond = d["condition"]
        d_lm = d["logmar"].astype(float)
        d_ori = d["orientation"].astype(int)
        d_tidx = d["trial_index"].astype(int)

        for lm in logmar_grid:
            for cond in ("real", "stabilized", "fixed_center"):
                for ori in ORIENTATIONS:
                    mask = (d_cond == cond) & (np.abs(d_lm - lm) < 0.005) & (d_ori == ori)
                    n = int(mask.sum())
                    report.append({
                        "quantity": "response_vectors",
                        "condition": cond, "logmar": _fmt_lm(lm), "orientation": ori,
                        "source_path": str(p),
                        "on_target_grid": "yes" if n > 0 else "no",
                        "n_trials": n,
                        "needs_recompute": "no" if n > 0 else "yes",
                        "recompute_cost_note": "" if n > 0 else f"no {cond} at lm={_fmt_lm(lm)},ori={ori}",
                    })
                    if n > 0:
                        merged[cond][round(lm, 3)][ori].extend(vecs[mask])
                        merged_tidx[cond][round(lm, 3)][ori].extend(d_tidx[mask])
    return report, merged, merged_tidx


def _scan_d1_accuracy(
    jpf_base: Path,
    scan_dirs: tuple[str, ...],
    logmar_grid: tuple[float, ...],
    d1_acc_col: str,
    feature_repr: str,
) -> tuple[list[dict], dict, str | None, bool]:
    """Scan decoder_metrics CSVs for D1 accuracy.

    Returns (report_rows, acc_data, confirmed_source_path, has_window_specific).
    acc_data[lm_key][pair_label][condition] = accuracy_float
    confirmed_source_path: path of the file actually providing the data, or None.
    has_window_specific: True if separate per-window rows exist.
    """
    report: list[dict] = []
    acc_data: dict = defaultdict(lambda: defaultdict(dict))
    confirmed_source: str | None = None
    has_window_specific = False

    for sd in scan_dirs:
        dm = jpf_base / sd / "eoptotype_identity" / "eoptotype_identity_decoder_metrics.csv"
        if not dm.exists():
            for lm in logmar_grid:
                report.append({
                    "quantity": "d1_accuracy",
                    "logmar": _fmt_lm(lm),
                    "d1_acc_col": d1_acc_col,
                    "source_path": str(dm),
                    "on_target_grid": "no_file",
                    "col_found": "no",
                    "has_window_specific": "no",
                    "needs_recompute": "yes",
                    "recompute_cost_note": f"missing: {sd}",
                })
            continue

        col_ok = _check_d1_col_exists(dm, d1_acc_col)
        if not col_ok:
            for lm in logmar_grid:
                report.append({
                    "quantity": "d1_accuracy",
                    "logmar": _fmt_lm(lm),
                    "d1_acc_col": d1_acc_col,
                    "source_path": str(dm),
                    "on_target_grid": "wrong_col",
                    "col_found": "no",
                    "has_window_specific": "no",
                    "needs_recompute": "yes",
                    "recompute_cost_note": f"column '{d1_acc_col}' not in {dm.name}",
                })
            continue

        # Check for a 'window' column (window-specific accuracy)
        rows_all = _read_csv(dm)
        col_names = list(rows_all[0].keys()) if rows_all else []
        has_win_col = "window" in col_names or "n_frames" in col_names
        if has_win_col:
            has_window_specific = True

        found_lms: set[float] = set()
        for row in rows_all:
            if str(row.get("feature_representation", "")) != feature_repr:
                continue
            lm = _f(row.get("logmar", "nan"))
            if not any(abs(lm - glm) < 0.005 for glm in logmar_grid):
                continue
            cond = str(row.get("condition", ""))
            pair_lbl = str(row.get("orientation_pair", ""))
            acc = _f(row.get(d1_acc_col, "nan"))
            lm_key = round(lm, 3)
            found_lms.add(lm_key)
            if pair_lbl and np.isfinite(acc):
                # First-writer wins per (lm_key, pair_lbl, cond) to avoid duplicates
                if cond not in acc_data[lm_key][pair_lbl]:
                    acc_data[lm_key][pair_lbl][cond] = acc
            if confirmed_source is None:
                confirmed_source = str(dm)

        for lm in logmar_grid:
            lm_key = round(lm, 3)
            on = "yes" if lm_key in found_lms else "no"
            report.append({
                "quantity": "d1_accuracy",
                "logmar": _fmt_lm(lm),
                "d1_acc_col": d1_acc_col,
                "source_path": str(dm),
                "on_target_grid": on,
                "col_found": "yes",
                "has_window_specific": "yes" if has_win_col else "no",
                "needs_recompute": "no" if lm_key in found_lms else "yes",
                "recompute_cost_note": "" if lm_key in found_lms else f"lm={_fmt_lm(lm)} not in {sd}",
            })

    return report, acc_data, confirmed_source, has_window_specific


def _scan_d1_sweep_csv(
    sweep_path: Path,
    logmar_grid: tuple[float, ...],
    sweep_col: str,
    windows: tuple[int, ...],
    feature_repr: str,
    readout_type: str,
) -> tuple[list[dict], dict, str | None, bool]:
    """Load D1 accuracy from the integration-window sweep CSV.

    The sweep file has one row per (condition, logmar, orientation_pair, readout_type,
    feature_representation, integration_window).  The delta column ``sweep_col``
    (e.g. real_minus_stabilized_d1_time_mean_accuracy) is pre-computed
    (real − stabilized), so no subtraction is needed here.

    acc_data[lm_key][pair_label][window_int] = delta_float

    Returns (report_rows, acc_data, confirmed_source, has_window_specific=True).
    """
    report: list[dict] = []

    if not sweep_path.exists():
        for lm in logmar_grid:
            report.append({
                "quantity": "d1_sweep",
                "logmar": _fmt_lm(lm),
                "d1_acc_col": sweep_col,
                "source_path": str(sweep_path),
                "on_target_grid": "no_file",
                "col_found": "no",
                "has_window_specific": "no",
                "needs_recompute": "yes",
                "recompute_cost_note": f"sweep file not found: {sweep_path}",
            })
        return report, {}, None, False

    col_ok = _check_d1_col_exists(sweep_path, sweep_col)
    if not col_ok:
        for lm in logmar_grid:
            report.append({
                "quantity": "d1_sweep",
                "logmar": _fmt_lm(lm),
                "d1_acc_col": sweep_col,
                "source_path": str(sweep_path),
                "on_target_grid": "wrong_col",
                "col_found": "no",
                "has_window_specific": "no",
                "needs_recompute": "yes",
                "recompute_cost_note": f"column '{sweep_col}' not in sweep file",
            })
        return report, {}, None, False

    rows_all = _read_csv(sweep_path)
    # acc_data[lm_key][pair_label][window_int] = delta
    acc_data: dict = {}
    found_lms: set[float] = set()

    for row in rows_all:
        # Filter to the primary condition (real) and the specified feature + readout
        if str(row.get("condition", "")) != "real":
            continue
        if str(row.get("feature_representation", "")) != feature_repr:
            continue
        if str(row.get("readout_type", "")) != readout_type:
            continue
        lm = _f(row.get("logmar", "nan"))
        if not any(abs(lm - glm) < 0.005 for glm in logmar_grid):
            continue
        win_raw = row.get("integration_window", "")
        try:
            win = int(float(win_raw))
        except (ValueError, TypeError):
            continue
        pair_lbl = str(row.get("orientation_pair", ""))
        delta = _f(row.get(sweep_col, "nan"))
        lm_key = round(lm, 3)
        found_lms.add(lm_key)
        acc_data.setdefault(lm_key, {}).setdefault(pair_lbl, {})[win] = delta

    for lm in logmar_grid:
        lm_key = round(lm, 3)
        on = "yes" if lm_key in found_lms else "no"
        # Check which windows are available at this logmar
        wins_available = sorted({
            w for plbl_dict in acc_data.get(lm_key, {}).values()
            for w in plbl_dict
        })
        report.append({
            "quantity": "d1_sweep",
            "logmar": _fmt_lm(lm),
            "d1_acc_col": sweep_col,
            "source_path": str(sweep_path),
            "on_target_grid": on,
            "col_found": "yes",
            "has_window_specific": "yes",
            "windows_available": str(wins_available),
            "needs_recompute": "no" if lm_key in found_lms else "yes",
            "recompute_cost_note": "" if lm_key in found_lms else f"lm={_fmt_lm(lm)} not in sweep",
        })

    confirmed = str(sweep_path) if found_lms else None
    return report, acc_data, confirmed, True  # always has_window_specific=True


def _scan_jacobian(jac_dir: Path, logmar_grid: tuple[float, ...]) -> list[dict]:
    rows: list[dict] = []
    diag = jac_dir / "jacobian_grid_diagnostics.csv"
    if not diag.exists():
        for lm in logmar_grid:
            rows.append({"quantity": "jacobian_J", "logmar": _fmt_lm(lm),
                         "source_path": str(diag), "on_target_grid": "no_file",
                         "has_raw_J_matrices": "no", "needs_recompute": "yes",
                         "recompute_cost_note": "jacobian_grid_diagnostics.csv not found"})
        return rows
    jrows = _read_csv(diag)
    found_lms = {round(_f(r.get("logmar", "nan")), 3) for r in jrows}
    # Current cache: norms only, not raw J matrices
    col_names = list(jrows[0].keys()) if jrows else []
    has_raw = any("J_col" in c and "component" in c for c in col_names)
    for lm in logmar_grid:
        lm_key = round(lm, 3)
        if lm_key in found_lms:
            rows.append({
                "quantity": "jacobian_J", "logmar": _fmt_lm(lm),
                "source_path": str(diag),
                "on_target_grid": "yes" if has_raw else "norms_only",
                "has_raw_J_matrices": "yes" if has_raw else "no",
                "needs_recompute": "no" if has_raw else "yes",
                "recompute_cost_note": "" if has_raw
                    else "norms only; raw J matrices needed for mimicry ΔM (Tier 2)",
            })
        else:
            rows.append({
                "quantity": "jacobian_J", "logmar": _fmt_lm(lm),
                "source_path": str(diag),
                "on_target_grid": "no",
                "has_raw_J_matrices": "no", "needs_recompute": "yes",
                "recompute_cost_note": f"lm={_fmt_lm(lm)} not in jacobian cache",
            })
    return rows


def _print_cache_summary(
    logmar_grid: tuple[float, ...],
    core_range: tuple[float, float],
    render_ctrl: float,
    rvec_rows: list[dict],
    acc_rows: list[dict],
    jac_rows: list[dict],
    confirmed_d1_source: str | None,
    has_window_specific: bool,
    d1_acc_col: str,
) -> None:
    """Print a human-readable cache audit summary to stdout."""
    print("\n=== Cache Audit Summary ===")

    # D1 source
    print(f"\n  D1 accuracy column : '{d1_acc_col}'")
    if confirmed_d1_source:
        print(f"  D1 source file     : {confirmed_d1_source}")
    else:
        print("  D1 source file     : NONE FOUND — run will abort without --cache-audit-only")
    print(f"  Window-specific acc: {'YES' if has_window_specific else 'NO (aggregate only)'}")

    # LogMAR availability table
    print(f"\n  {'LogMAR':>8}  {'core':>5}  {'rvec':>5}  {'d1_acc':>6}  {'raw_J':>6}  {'tier1_ok':>9}")
    for lm in logmar_grid:
        lm_key = round(lm, 3)
        in_core = (core_range[0] - 1e-4) <= lm <= (core_range[1] + 1e-4)
        is_ctrl = abs(lm - render_ctrl) < 1e-4

        rvec_ok = any(
            abs(_f(r.get("logmar", "nan")) - lm) < 0.005 and
            r.get("on_target_grid") == "yes"
            for r in rvec_rows if r.get("quantity") == "response_vectors"
        )
        acc_ok = any(
            abs(_f(r.get("logmar", "nan")) - lm) < 0.005 and
            r.get("on_target_grid") == "yes"
            for r in acc_rows
        )
        raw_j = any(
            abs(_f(r.get("logmar", "nan")) - lm) < 0.005 and
            r.get("has_raw_J_matrices") == "yes"
            for r in jac_rows
        )
        tier1_ok = rvec_ok and acc_ok
        ctrl_tag = " [ctrl]" if is_ctrl else ""
        core_tag = "YES" if in_core else "no"
        print(f"  {_fmt_lm(lm):>8}{ctrl_tag}  {core_tag:>5}  {'Y' if rvec_ok else '-':>5}"
              f"  {'Y' if acc_ok else '-':>6}  {'Y' if raw_j else '-':>6}"
              f"  {'READY' if tier1_ok else 'MISSING':>9}")

    # Raw J status
    has_any_j = any(r.get("has_raw_J_matrices") == "yes" for r in jac_rows)
    norms_only = any(r.get("on_target_grid") == "norms_only" for r in jac_rows)
    print(f"\n  Jacobian status    : "
          + ("raw matrices available" if has_any_j
             else "norms only (Tier-2 deferred)" if norms_only
             else "not cached — Tier-2 deferred"))
    print("===========================\n")


def step0_cache_audit(
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict, dict, dict, str | None, bool]:
    """Returns (merged_rvec, merged_tidx, acc_data, confirmed_d1_source, has_window_specific)."""
    print("[Step 0] Cache audit...")
    jpf_base = ROOT / "outputs" / "jacobian_predictive_framework"

    rvec_rows, merged_rvec, merged_tidx = _scan_response_vectors(
        jpf_base, args.accuracy_scan_dirs, args.logmar_grid, args.feature_repr
    )

    # Prefer the explicit D1 sweep file when provided; fall back to decoder_metrics scan.
    sweep_path = Path(args.d1_sweep_path) if args.d1_sweep_path else None
    if sweep_path is not None and sweep_path.exists():
        acc_rows, acc_data, confirmed_d1_source, has_window_specific = _scan_d1_sweep_csv(
            sweep_path, args.logmar_grid,
            args.d1_sweep_col, args.windows,
            args.feature_repr, args.d1_sweep_readout,
        )
        print(f"  Using explicit D1 sweep: {sweep_path.name}")
    else:
        if sweep_path is not None:
            print(f"  WARNING: --d1-sweep-path not found ({sweep_path}); "
                  "falling back to decoder_metrics scan.")
        acc_rows, acc_data, confirmed_d1_source, has_window_specific = _scan_d1_accuracy(
            jpf_base, args.accuracy_scan_dirs, args.logmar_grid,
            args.d1_acc_col, args.feature_repr
        )

    jac_rows = _scan_jacobian(args.jacobian_smoothness_dir, args.logmar_grid)

    # Write cache_availability_report.csv
    all_rows = rvec_rows + acc_rows + jac_rows
    fnames = ["quantity", "condition", "logmar", "orientation", "d1_acc_col",
              "source_path", "on_target_grid", "col_found", "has_raw_J_matrices",
              "has_window_specific", "n_trials", "needs_recompute", "recompute_cost_note"]
    for r in all_rows:
        for f in fnames:
            r.setdefault(f, "")
    _write_csv(out_dir / "cache_availability_report.csv", all_rows, fnames)

    # Write grid_reconciliation.csv
    grid_rows = []
    for lm in args.logmar_grid:
        lm_key = round(lm, 3)
        in_core = (args.core_range[0] - 1e-4) <= lm <= (args.core_range[1] + 1e-4)
        is_ctrl = abs(lm - args.render_limit_control) < 1e-4
        rvec_ok = any(
            abs(_f(r.get("logmar", "nan")) - lm) < 0.005 and r.get("on_target_grid") == "yes"
            for r in rvec_rows
        )
        acc_ok = any(
            abs(_f(r.get("logmar", "nan")) - lm) < 0.005 and r.get("on_target_grid") == "yes"
            for r in acc_rows
        )
        raw_j = any(
            abs(_f(r.get("logmar", "nan")) - lm) < 0.005 and r.get("has_raw_J_matrices") == "yes"
            for r in jac_rows
        )
        n_real = sum(
            len(v) for v in merged_rvec.get("real", {}).get(lm_key, {}).values()
        )
        grid_rows.append({
            "logmar": _fmt_lm(lm),
            "in_core_range": "yes" if in_core else "no",
            "render_limit_control": "yes" if is_ctrl else "no",
            "rvec_available": "yes" if rvec_ok else "no",
            "acc_available": "yes" if acc_ok else "no",
            "raw_J_available": "yes" if raw_j else "no",
            "n_trials_real": n_real,
            "tier1_ready": "yes" if (rvec_ok and acc_ok) else "no",
            "tier2_ready": "yes" if raw_j else "no",
        })
    _write_csv(out_dir / "grid_reconciliation.csv", grid_rows)

    # Print summary
    _print_cache_summary(
        args.logmar_grid, args.core_range, args.render_limit_control,
        rvec_rows, acc_rows, jac_rows,
        confirmed_d1_source, has_window_specific, args.d1_acc_col
    )

    n_core_t1 = sum(1 for r in grid_rows if r["in_core_range"] == "yes" and r["tier1_ready"] == "yes")
    n_core = sum(1 for r in grid_rows if r["in_core_range"] == "yes")
    missing_core = [r["logmar"] for r in grid_rows if r["in_core_range"] == "yes" and r["tier1_ready"] == "no"]
    print(f"  Tier-1 core-range ready: {n_core_t1}/{n_core}")
    if missing_core:
        print(f"  WARN: missing core-range logmars: {missing_core}")

    return merged_rvec, merged_tidx, acc_data, confirmed_d1_source, has_window_specific


# ---------------------------------------------------------------------------
# Step 1 — Geometry curves (cloud_separability_crosswalk)
# ---------------------------------------------------------------------------

def _class_means(merged_rvec: dict, merged_tidx: dict, cond: str, lm: float
                 ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, list[int]]]:
    lm_key = round(lm, 3)
    means: dict[int, np.ndarray] = {}
    trials: dict[int, np.ndarray] = {}
    tidxs: dict[int, list[int]] = {}
    for ori in ORIENTATIONS:
        ts = merged_rvec.get(cond, {}).get(lm_key, {}).get(ori, [])
        if ts:
            arr = np.stack(ts, axis=0).astype(np.float64)
            means[ori] = arr.mean(axis=0)
            trials[ori] = arr
            tidxs[ori] = list(merged_tidx.get(cond, {}).get(lm_key, {}).get(ori, []))
    return means, trials, tidxs


def _cloud_mean_sep_and_pairs(
    merged_rvec: dict, merged_tidx: dict, lm: float
) -> tuple[float, dict[tuple[int, int], float]]:
    """S_cloud_mean: E_{p~real_cloud}[mean_{a<b} ||μ_a(p)-μ_b(p)||]. Matched by trial_index."""
    lm_key = round(lm, 3)
    real_rvec = merged_rvec.get("real", {}).get(lm_key, {})
    real_tidx = merged_tidx.get("real", {}).get(lm_key, {})

    ti_resp: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for ori in ORIENTATIONS:
        for tidx, resp in zip(real_tidx.get(ori, []), real_rvec.get(ori, [])):
            ti_resp[int(tidx)][ori] = np.asarray(resp, dtype=np.float64)

    per_pos_mean: list[float] = []
    per_pair: dict[tuple[int, int], list[float]] = {(a, b): [] for a, b in PAIRS}
    for resp_by_ori in ti_resp.values():
        for a, b in PAIRS:
            if a in resp_by_ori and b in resp_by_ori:
                d = float(np.linalg.norm(resp_by_ori[a] - resp_by_ori[b]))
                per_pair[(a, b)].append(d)
        seps = [float(np.linalg.norm(resp_by_ori[a] - resp_by_ori[b]))
                for a, b in PAIRS if a in resp_by_ori and b in resp_by_ori]
        if seps:
            per_pos_mean.append(float(np.mean(seps)))

    s_cloud_mean = float(np.nanmean(per_pos_mean)) if per_pos_mean else float("nan")
    s_cloud_pairs = {(a, b): float(np.nanmean(v)) if v else float("nan")
                     for (a, b), v in per_pair.items()}
    return s_cloud_mean, s_cloud_pairs


def step1_geometry_curves(
    args: argparse.Namespace,
    merged_rvec: dict,
    merged_tidx: dict,
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    print("[Step 1] Computing geometry curves (cloud_separability_crosswalk)...")
    curve_rows: list[dict] = []
    pair_rows: list[dict] = []

    for lm in args.logmar_grid:
        is_ctrl = abs(lm - args.render_limit_control) < 1e-4

        real_means, _, _ = _class_means(merged_rvec, merged_tidx, "real", lm)
        stab_means, _, _ = _class_means(merged_rvec, merged_tidx, "stabilized", lm)
        cent_means, _, _ = _class_means(merged_rvec, merged_tidx, "fixed_center", lm)

        sep_real = _pairwise_mean_sep(real_means)
        sep_stab = _pairwise_mean_sep(stab_means)
        g_sep = (sep_real - sep_stab
                 if np.isfinite(sep_real) and np.isfinite(sep_stab)
                 else float("nan"))

        s_center = _pairwise_mean_sep(cent_means)
        s_stab = _pairwise_mean_sep(stab_means)
        s_cloud_mean, s_cloud_pairs = _cloud_mean_sep_and_pairs(merged_rvec, merged_tidx, lm)

        lm_key = round(lm, 3)
        n_real = sum(
            len(merged_rvec.get("real", {}).get(lm_key, {}).get(ori, []))
            for ori in ORIENTATIONS
        )
        n_phase = n_real // len(ORIENTATIONS) if n_real > 0 else 0

        curve_rows.append({
            "L": _fmt_lm(lm),
            "render_limit_control": "yes" if is_ctrl else "no",
            "analysis_tier": "cloud_separability_crosswalk",
            "Sep_avg_real": sep_real,
            "Sep_avg_stab": sep_stab,
            "G_sep_mean": g_sep,
            "S_center": s_center,
            "S_stab": s_stab,
            "S_cloud_mean": s_cloud_mean,
            "M_mean": "nan",
            "dM_mean": "nan",
            "n_phase": n_phase,
            "n_pairs": len(PAIRS),
            "metric": "euclidean",
            "n_trials_real": n_real,
        })

        sep_real_p = _pairwise_seps(real_means)
        sep_stab_p = _pairwise_seps(stab_means)
        sep_cent_p = _pairwise_seps(cent_means)
        for a, b in PAIRS:
            sr = sep_real_p.get((a, b), float("nan"))
            ss = sep_stab_p.get((a, b), float("nan"))
            g_pair = sr - ss if np.isfinite(sr) and np.isfinite(ss) else float("nan")
            pair_rows.append({
                "L": _fmt_lm(lm),
                "render_limit_control": "yes" if is_ctrl else "no",
                "pair": _pair_label(a, b),
                "G_sep_pair": g_pair,
                "S_center_pair": sep_cent_p.get((a, b), float("nan")),
                "S_stab_pair": sep_stab_p.get((a, b), float("nan")),
                "S_cloud_mean_pair": s_cloud_pairs.get((a, b), float("nan")),
                "M_pair": "nan",
                "dM_pair": "nan",
            })

    _write_csv(out_dir / "geometry_curves" / "geometry_curve.csv", curve_rows)
    _write_csv(out_dir / "geometry_curves" / "geometry_pairwise.csv", pair_rows)
    print(f"  Saved {len(curve_rows)} logmar rows, {len(pair_rows)} pairwise rows.")
    return curve_rows, pair_rows


# ---------------------------------------------------------------------------
# Step 2 — Function curves (D1 accuracy, explicit source required)
# ---------------------------------------------------------------------------

def step2_function_curves(
    args: argparse.Namespace,
    acc_data: dict,
    confirmed_d1_source: str | None,
    has_window_specific: bool,
    merged_rvec: dict,
    out_dir: Path,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    print("[Step 2] Computing function curves (D1)...")

    if confirmed_d1_source is None:
        raise RuntimeError(
            f"No D1 accuracy source found for column '{args.d1_acc_col}'.\n"
            "Check --d1-acc-col and --accuracy-scan-dirs. "
            "Do not proceed without a confirmed D1 source.\n"
            "Use --cache-audit-only to inspect what is available."
        )

    # Determine which windows to output. If accuracy is aggregate (no per-window rows),
    # only emit primary_window and skip the robustness step.
    emit_windows = (args.windows if has_window_specific else (args.primary_window,))
    if not has_window_specific and len(args.windows) > 1:
        print(f"  NOTE: D1 source is aggregate (no window column). "
              f"Emitting primary window only (W={args.primary_window}). "
              "Window robustness DISABLED.")

    func_rows: list[dict] = []
    func_pair_rows: list[dict] = []

    # Detect acc_data format:
    #   sweep format  → acc_data[lm_key][pair_lbl][window_int] = delta_float
    #   legacy format → acc_data[lm_key][pair_lbl][condition_str] = acc_float
    first_lm_data = next(iter(acc_data.values()), {}) if acc_data else {}
    first_pair_data = next(iter(first_lm_data.values()), {}) if first_lm_data else {}
    is_sweep_format = bool(first_pair_data) and all(isinstance(k, int) for k in first_pair_data)
    source_label = args.d1_sweep_col if is_sweep_format else args.d1_acc_col

    for lm in args.logmar_grid:
        lm_key = round(lm, 3)
        is_ctrl = abs(lm - args.render_limit_control) < 1e-4
        lm_acc = acc_data.get(lm_key, {})

        n_trials = sum(
            len(merged_rvec.get("real", {}).get(lm_key, {}).get(ori, []))
            for ori in ORIENTATIONS
        )

        for w in emit_windows:
            # Compute per-pair deltas for this window
            delta_by_pair: dict[tuple[int, int], float] = {}
            for a, b in PAIRS:
                lbl = _pair_label(a, b)
                pair_acc = lm_acc.get(lbl, {})
                if is_sweep_format:
                    # Delta pre-computed, look up by window
                    delta_by_pair[(a, b)] = _f(pair_acc.get(w, float("nan")))
                else:
                    # Legacy: subtract real - stabilized
                    r_acc = _f(pair_acc.get("real", float("nan")))
                    s_acc = _f(pair_acc.get("stabilized", float("nan")))
                    delta_by_pair[(a, b)] = (
                        r_acc - s_acc
                        if np.isfinite(r_acc) and np.isfinite(s_acc)
                        else float("nan")
                    )

            finite_deltas = [d for d in delta_by_pair.values() if np.isfinite(d)]
            delta_mean = float(np.nanmean(finite_deltas)) if finite_deltas else float("nan")

            # Bootstrap CI — pair-resampling across the 6 pairs
            if len(finite_deltas) >= 2:
                bs = [float(np.mean(rng.choice(finite_deltas, size=len(finite_deltas), replace=True)))
                      for _ in range(args.n_bootstrap)]
                ci_low, ci_high = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                ci_type = (f"d1_w{w}_sweep_pair_resampling"
                           if is_sweep_format else "pair_resampling_exploratory")
            else:
                ci_low = ci_high = float("nan")
                ci_type = "insufficient_pairs"

            func_rows.append({
                "L": _fmt_lm(lm),
                "render_limit_control": "yes" if is_ctrl else "no",
                "window": w,
                "function_metric": f"delta_d1_time_mean_w{w}" if is_sweep_format else "delta_acc_mean",
                "readout_type": args.d1_sweep_readout if is_sweep_format else "linear",
                "d1_acc_col": source_label,
                "source_path": confirmed_d1_source,
                "accuracy_real": float("nan"),   # not separately available from sweep
                "accuracy_stab": float("nan"),
                "delta_acc_mean": delta_mean,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "ci_type": ci_type,
                "n_pairs_contributing": len(finite_deltas),
                "n_trials_real": n_trials,
                "window_specific_acc_available": "yes" if has_window_specific else "no",
            })
            for a, b in PAIRS:
                d = delta_by_pair.get((a, b), float("nan"))
                func_pair_rows.append({
                    "L": _fmt_lm(lm),
                    "render_limit_control": "yes" if is_ctrl else "no",
                    "pair": _pair_label(a, b),
                    "window": w,
                    "function_metric": f"delta_d1_time_mean_w{w}" if is_sweep_format else "delta_acc_pair",
                    "readout_type": args.d1_sweep_readout if is_sweep_format else "linear",
                    "d1_acc_col": source_label,
                    "source_path": confirmed_d1_source,
                    "delta_acc_pair": d,
                    "ci_low": "nan",
                    "ci_high": "nan",
                    "ci_type": "none_single_value",
                    "n_trials": 0,
                })

    _write_csv(out_dir / "function_curves" / "function_curve.csv", func_rows)
    _write_csv(out_dir / "function_curves" / "function_pairwise.csv", func_pair_rows)
    print(f"  D1 source: {confirmed_d1_source}")
    print(f"  Saved {len(func_rows)} rows, {len(func_pair_rows)} pairwise rows.")
    return func_rows, func_pair_rows


# ---------------------------------------------------------------------------
# Array extraction helpers
# ---------------------------------------------------------------------------

def _extract_arrays(
    geom_rows: list[dict],
    func_rows: list[dict],
    window: int,
    render_limit_control: float,
    core_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ls, G_sep, Δacc, S_center, S_stab, S_cloud_mean, in_core).
    All arrays aligned; render_limit_control excluded.
    """
    geom_by_lm = {float(r["L"]): r for r in geom_rows}
    func_by_lm_w = {(float(r["L"]), int(r["window"])): r for r in func_rows}

    Ls, g_sep, d_acc, s_ctr, s_stab, s_cloud, in_core = [], [], [], [], [], [], []
    for lm in sorted(geom_by_lm):
        if abs(lm - render_limit_control) < 1e-4:
            continue
        gr = geom_by_lm[lm]
        fr = func_by_lm_w.get((lm, window))
        Ls.append(lm)
        g_sep.append(_f(gr.get("G_sep_mean")))
        d_acc.append(_f(fr.get("delta_acc_mean")) if fr else float("nan"))
        s_ctr.append(_f(gr.get("S_center")))
        s_stab.append(_f(gr.get("S_stab")))
        s_cloud.append(_f(gr.get("S_cloud_mean")))
        in_core.append((core_range[0] - 1e-4) <= lm <= (core_range[1] + 1e-4))
    return (np.array(Ls), np.array(g_sep), np.array(d_acc),
            np.array(s_ctr), np.array(s_stab), np.array(s_cloud),
            np.array(in_core, dtype=bool))


# ---------------------------------------------------------------------------
# Step 3 — Test 1: Coincidence
# ---------------------------------------------------------------------------

def step3_test1_coincidence(
    args: argparse.Namespace,
    geom_rows: list[dict],
    func_rows: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    print("[Step 3] Test 1: coincidence...")
    test_rows: list[dict] = []
    window = args.primary_window
    grid_step = DEFAULT_COINCIDENCE_GRID_STEP

    Ls, g_sep, d_acc, _, _, _, in_core = _extract_arrays(
        geom_rows, func_rows, window, args.render_limit_control, args.core_range
    )

    L_geom, geom_status = _zero_crossing_core(Ls, g_sep, in_core)
    L_func, func_status = _zero_crossing_core(Ls, d_acc, in_core)
    dL = abs(L_geom - L_func) if np.isfinite(L_geom) and np.isfinite(L_func) else float("nan")

    # Check render-limit confound: does crossing disappear if -0.40 excluded?
    # (Already excluded via render_limit_control=args.render_limit_control above.)
    # Also try full grid (with -0.40 included):
    Ls_all, g_all, d_all, _, _, _, _ = _extract_arrays(
        geom_rows, func_rows, window, -999.0, args.core_range
    )
    L_geom_all, _ = _interp_zero(Ls_all, g_all)
    render_confounded = (
        not np.isfinite(L_geom) and np.isfinite(L_geom_all) and
        abs(L_geom_all - args.render_limit_control) < 0.005
    )

    # Bootstrap dL using pair-resampled Δacc
    valid_core = in_core & np.isfinite(g_sep) & np.isfinite(d_acc)
    dL_bs: list[float] = []
    if valid_core.sum() >= 2:
        Lc = Ls[valid_core]
        gc = g_sep[valid_core]
        dc = d_acc[valid_core]
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, len(dc), size=len(dc))
            # Aggregate duplicates
            unique_L = sorted(set(Lc[idx].tolist()))
            g_agg = np.array([gc[idx][np.abs(Lc[idx] - l) < 1e-5].mean() for l in unique_L])
            d_agg = np.array([dc[idx][np.abs(Lc[idx] - l) < 1e-5].mean() for l in unique_L])
            Lg_b, _ = _interp_zero(np.array(unique_L), g_agg)
            Ld_b, _ = _interp_zero(np.array(unique_L), d_agg)
            if np.isfinite(Lg_b) and np.isfinite(Ld_b):
                dL_bs.append(abs(Lg_b - Ld_b))

    dL_ci_low = float(np.percentile(dL_bs, 2.5)) if len(dL_bs) > 10 else float("nan")
    dL_ci_high = float(np.percentile(dL_bs, 97.5)) if len(dL_bs) > 10 else float("nan")
    dL_median = float(np.median(dL_bs)) if len(dL_bs) > 10 else dL

    # Criterion (patch 5): dL_median < grid_step AND dL_ci_high < grid_step
    coincidence = (
        np.isfinite(dL) and
        np.isfinite(dL_median) and
        dL_median < grid_step and
        np.isfinite(dL_ci_high) and
        dL_ci_high < grid_step and
        not render_confounded
    )

    def _add(name, val, ci_l=float("nan"), ci_h=float("nan"), note=""):
        test_rows.append({
            "test": "test1_coincidence", "level": "mean", "pair": "all",
            "observable": name, "statistic_name": name,
            "value": val, "ci_low": ci_l, "ci_high": ci_h,
            "null_p": "nan", "window": window, "verdict_component": note,
        })

    _add("L_geom_crossing", L_geom, note=geom_status)
    _add("L_func_crossing", L_func, note=func_status)
    _add("dL", dL, dL_ci_low, dL_ci_high, note="coincidence_dL")
    _add("dL_median_bootstrap", dL_median)
    _add("render_limit_confounded", int(render_confounded))
    _add("coincidence_verdict", int(coincidence), note="TRUE" if coincidence else "FALSE")

    L_geom_s = f"{L_geom:.3f}" if np.isfinite(L_geom) else "nan"
    L_func_s = f"{L_func:.3f}" if np.isfinite(L_func) else "nan"
    dL_s = f"{dL:.3f}" if np.isfinite(dL) else "nan"
    print(f"  L_geom={L_geom_s} ({geom_status}), L_func={L_func_s} ({func_status})")
    print(f"  dL={dL_s}, dL_median={f'{dL_median:.3f}' if np.isfinite(dL_median) else 'nan'}, "
          f"dL_ci_high={f'{dL_ci_high:.3f}' if np.isfinite(dL_ci_high) else 'nan'}")
    print(f"  coincidence={'TRUE' if coincidence else 'FALSE'} "
          f"(criterion: dL_median<{grid_step} AND dL_ci_high<{grid_step})")
    return test_rows


# ---------------------------------------------------------------------------
# Step 4 — Test 2: Continuous regression
# ---------------------------------------------------------------------------

def step4_test2_continuous(
    args: argparse.Namespace,
    geom_rows: list[dict],
    func_rows: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    print("[Step 4] Test 2: continuous regression...")
    test_rows: list[dict] = []
    window = args.primary_window

    Ls, g_sep, d_acc, _, _, _, _ = _extract_arrays(
        geom_rows, func_rows, window, args.render_limit_control, args.core_range
    )
    valid = np.isfinite(g_sep) & np.isfinite(d_acc)
    if valid.sum() < 3:
        print("  WARNING: Insufficient data for continuous regression.")
        return test_rows

    gv, dv = g_sep[valid], d_acc[valid]
    slope, _, r, p_val, _ = scipy_stats.linregress(gv, dv)
    r2 = float(r ** 2)
    spearman_rho, _ = scipy_stats.spearmanr(gv, dv)

    slopes_b, r2s_b, rhos_b = [], [], []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, len(gv), size=len(gv))
        if len(set(idx.tolist())) < 2:
            continue
        sl, _, rb, _, _ = scipy_stats.linregress(gv[idx], dv[idx])
        slopes_b.append(sl)
        r2s_b.append(rb ** 2)
        rho_b, _ = scipy_stats.spearmanr(gv[idx], dv[idx])
        rhos_b.append(float(rho_b))

    def _ci(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
                ) if len(arr) >= 10 else (float("nan"), float("nan"))

    slope_ci = _ci(slopes_b)
    r2_ci = _ci(r2s_b)
    rho_ci = _ci(rhos_b)

    continuous_sig = (
        np.isfinite(slope) and slope > 0 and
        np.isfinite(slope_ci[0]) and slope_ci[0] > 0
    )

    def _add(name, val, ci_l=float("nan"), ci_h=float("nan"), note=""):
        test_rows.append({
            "test": "test2_continuous", "level": "mean", "pair": "all",
            "observable": "G_sep_mean vs delta_acc_mean",
            "statistic_name": name, "value": val, "ci_low": ci_l, "ci_high": ci_h,
            "null_p": "nan", "window": window, "verdict_component": note,
        })

    _add("slope", slope, slope_ci[0], slope_ci[1], note="continuous_slope")
    _add("R2", r2, r2_ci[0], r2_ci[1], note="continuous_R2")
    _add("spearman_rho", float(spearman_rho), rho_ci[0], rho_ci[1], note="continuous_spearman")
    _add("p_value_ols", float(p_val))
    _add("continuous_significant", int(continuous_sig), note="TRUE" if continuous_sig else "FALSE")

    print(f"  slope={slope:.4f} CI=[{slope_ci[0]:.4f},{slope_ci[1]:.4f}], "
          f"R²={r2:.3f}, ρ={float(spearman_rho):.3f}")
    print(f"  continuous_significant={'TRUE' if continuous_sig else 'FALSE'}")
    return test_rows


# ---------------------------------------------------------------------------
# Step 5 — Test 3: Specificity (three separate controls, one declared primary)
# ---------------------------------------------------------------------------

def _specificity_for_control(
    g_sep: np.ndarray,
    d_acc: np.ndarray,
    s_ctrl: np.ndarray,
    ctrl_name: str,
    window: int,
    n_bootstrap: int,
    delta_r2_threshold: float,
    rng: np.random.Generator,
) -> tuple[list[dict], bool]:
    """Run one specificity test using ctrl_name as the difficulty covariate.
    Returns (test_rows, passed)."""
    rows: list[dict] = []
    valid = np.isfinite(g_sep) & np.isfinite(d_acc) & np.isfinite(s_ctrl)
    if valid.sum() < 4:
        rows.append({
            "test": "test3_specificity", "level": "mean", "pair": "all",
            "observable": f"delta_acc vs G_sep | {ctrl_name}",
            "statistic_name": "specificity_passed",
            "value": 0, "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
            "window": window, "verdict_component": f"FALSE_insufficient_data ({ctrl_name})",
            "specificity_control": ctrl_name,
        })
        return rows, False

    gv, dv, sv = g_sep[valid], d_acc[valid], s_ctrl[valid]
    n = len(gv)

    def _resid(y, x):
        sl, ic, _, _, _ = scipy_stats.linregress(x, y)
        return y - (sl * x + ic)

    d_resid = _resid(dv, sv)
    g_resid = _resid(gv, sv)
    partial_rho, _ = scipy_stats.spearmanr(d_resid, g_resid)

    def _ols_r2_aic(y, X_cols):
        X = np.column_stack([np.ones(n)] + list(X_cols))
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        ss_res = float(np.dot(res, res))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, EPS)
        k = X.shape[1]
        aic = n * np.log(max(ss_res / n, EPS)) + 2 * k
        return r2, float(aic)

    r2_null, aic_null = _ols_r2_aic(dv, [sv])
    r2_full, aic_full = _ols_r2_aic(dv, [sv, gv])
    delta_r2 = r2_full - r2_null if np.isfinite(r2_full) and np.isfinite(r2_null) else float("nan")
    delta_aic = aic_full - aic_null if np.isfinite(aic_full) and np.isfinite(aic_null) else float("nan")

    rhos_b = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 3:
            continue
        dr = _resid(dv[idx], sv[idx])
        gr = _resid(gv[idx], sv[idx])
        rho_b, _ = scipy_stats.spearmanr(dr, gr)
        rhos_b.append(float(rho_b))
    rho_ci_low = float(np.percentile(rhos_b, 2.5)) if len(rhos_b) >= 10 else float("nan")
    rho_ci_high = float(np.percentile(rhos_b, 97.5)) if len(rhos_b) >= 10 else float("nan")

    passed = (
        np.isfinite(partial_rho) and np.isfinite(rho_ci_low) and
        rho_ci_low > 0.0 and
        np.isfinite(delta_r2) and delta_r2 >= delta_r2_threshold and
        np.isfinite(delta_aic) and delta_aic < 0.0
    )

    def _row(name, val, ci_l=float("nan"), ci_h=float("nan"), note=""):
        rows.append({
            "test": "test3_specificity", "level": "mean", "pair": "all",
            "observable": f"delta_acc vs G_sep | {ctrl_name}",
            "statistic_name": f"{name}__{ctrl_name}",
            "value": val, "ci_low": ci_l, "ci_high": ci_h, "null_p": "nan",
            "window": window, "verdict_component": note,
            "specificity_control": ctrl_name,
        })

    _row("partial_rho", float(partial_rho), rho_ci_low, rho_ci_high, note="partial_rho_given_S")
    _row("nested_delta_R2", delta_r2, note="nested_delta_R2")
    _row("nested_AIC_delta", delta_aic, note="nested_AIC_delta")
    _row("R2_null", r2_null)
    _row("R2_full", r2_full)
    _row("specificity_passed", int(passed), note=f"{'TRUE' if passed else 'FALSE'} ({ctrl_name})")
    return rows, passed


def step5_test3_specificity(
    args: argparse.Namespace,
    geom_rows: list[dict],
    func_rows: list[dict],
    rng: np.random.Generator,
) -> tuple[list[dict], bool]:
    """Run specificity test for each of the 3 difficulty controls.
    Returns (test_rows, primary_specificity_passed).
    """
    print("[Step 5] Test 3: specificity (three controls)...")
    test_rows: list[dict] = []
    window = args.primary_window

    Ls, g_sep, d_acc, s_ctr, s_stab, s_cloud, _ = _extract_arrays(
        geom_rows, func_rows, window, args.render_limit_control, args.core_range
    )

    controls = {"S_center": s_ctr, "S_stab": s_stab, "S_cloud_mean": s_cloud}
    passed_by: dict[str, bool] = {}
    for ctrl_name, s_arr in controls.items():
        rows, passed = _specificity_for_control(
            g_sep, d_acc, s_arr, ctrl_name, window,
            args.n_bootstrap, args.specificity_delta_r2,
            rng
        )
        test_rows += rows
        passed_by[ctrl_name] = passed
        print(f"  {ctrl_name}: {'PASS' if passed else 'fail'}")

    primary_passed = passed_by.get(args.primary_difficulty_control, False)
    print(f"  Primary control ({args.primary_difficulty_control}): "
          f"{'PASS' if primary_passed else 'FAIL'}")
    return test_rows, primary_passed


# ---------------------------------------------------------------------------
# Step 6 — Test 4: Tier-2 mechanism (ΔM) — deferred
# ---------------------------------------------------------------------------

def step6_test4_mechanism(
    args: argparse.Namespace,
    jac_rows_cache: list[dict],
) -> tuple[list[dict], bool]:
    print("[Step 6] Test 4: Tier-2 mechanism (ΔM)...")
    test_rows: list[dict] = []

    raw_j_lms = [_f(r.get("logmar")) for r in jac_rows_cache
                 if r.get("has_raw_J_matrices") == "yes" and r.get("quantity") == "jacobian_J"]
    norms_lms = [_f(r.get("logmar")) for r in jac_rows_cache
                 if r.get("on_target_grid") == "norms_only" and r.get("quantity") == "jacobian_J"]

    if raw_j_lms:
        note = f"Raw J available at {raw_j_lms} — ΔM computation not yet implemented in this script."
        verdict = "deferred_J_available_but_not_computed"
    elif norms_lms:
        note = (f"Jacobian norms at {norms_lms} but raw J matrices not cached. "
                "Recompute storing full J to enable mimicry ΔM.")
        verdict = "deferred_norms_only"
    else:
        note = ("No Jacobian cache found. Run eoptotype_jacobian_field_smoothness.py "
                "with --logmars covering the keystone grid.")
        verdict = "deferred_no_cache"

    test_rows.append({
        "test": "test4_mechanism", "level": "mean", "pair": "all",
        "observable": "dM_mean", "statistic_name": "mechanism_tangent_tracks",
        "value": 0, "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
        "window": args.primary_window,
        "verdict_component": f"FALSE_{verdict}: {note}",
    })
    print(f"  Tier-2 status: {verdict}")
    return test_rows, False


# ---------------------------------------------------------------------------
# Step 7 — Pairwise (coincidence + specificity per pair, separately)
# ---------------------------------------------------------------------------

def step7_pairwise(
    args: argparse.Namespace,
    geom_pair_rows: list[dict],
    func_pair_rows: list[dict],
    rng: np.random.Generator,
) -> tuple[list[dict], int]:
    """Coincidence and specificity per pair. Returns (test_rows, n_pairs_passing_both)."""
    print("[Step 7] Pairwise analysis...")
    test_rows: list[dict] = []
    window = args.primary_window
    grid_step = DEFAULT_COINCIDENCE_GRID_STEP

    geom_p_by = {(float(r["L"]), str(r["pair"])): r for r in geom_pair_rows}
    func_p_by = {(float(r["L"]), str(r["pair"]), int(r["window"])): r for r in func_pair_rows}

    n_passing_both = 0

    for a, b in PAIRS:
        lbl = _pair_label(a, b)
        Ls_p, g_p, d_p, s_stab_p = [], [], [], []
        for lm in args.logmar_grid:
            if abs(lm - args.render_limit_control) < 1e-4:
                continue
            gr = geom_p_by.get((lm, lbl), {})
            fr = func_p_by.get((lm, lbl, window), {})
            Ls_p.append(lm)
            g_p.append(_f(gr.get("G_sep_pair")))
            d_p.append(_f(fr.get("delta_acc_pair")))
            s_stab_p.append(_f(gr.get("S_stab_pair")))

        La = np.array(Ls_p)
        ga = np.array(g_p)
        da = np.array(d_p)
        sa_stab = np.array(s_stab_p)
        in_core = np.array(
            [(args.core_range[0] - 1e-4) <= l <= (args.core_range[1] + 1e-4) for l in Ls_p]
        )

        # --- Coincidence per pair ---
        L_g, gs = _zero_crossing_core(La, ga, in_core)
        L_d, ds = _zero_crossing_core(La, da, in_core)
        dL_pair = abs(L_g - L_d) if np.isfinite(L_g) and np.isfinite(L_d) else float("nan")
        coin_pair = np.isfinite(dL_pair) and dL_pair < grid_step

        test_rows.append({
            "test": "test1_coincidence", "level": "pair", "pair": lbl,
            "observable": "G_sep_pair vs delta_acc_pair",
            "statistic_name": "dL_pair", "value": dL_pair,
            "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
            "window": window,
            "verdict_component": f"pairwise_coincidence={'TRUE' if coin_pair else 'FALSE'} "
                                  f"(geom={gs}, func={ds})",
        })

        # --- Continuous per pair ---
        valid = np.isfinite(ga) & np.isfinite(da)
        slope_pair = float("nan")
        r2_pair = float("nan")
        cont_pair = False
        if valid.sum() >= 3:
            sl, _, rr, _, _ = scipy_stats.linregress(ga[valid], da[valid])
            slope_pair = sl
            r2_pair = rr ** 2
            cont_pair = sl > 0
            test_rows.append({
                "test": "test2_continuous", "level": "pair", "pair": lbl,
                "observable": "G_sep_pair vs delta_acc_pair",
                "statistic_name": "slope_pair", "value": sl,
                "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
                "window": window,
                "verdict_component": f"slope={sl:.4f} R2={rr**2:.3f}",
            })

        # --- Specificity per pair (S_stab as primary, no bootstrap for individual pairs) ---
        spec_pair = False
        valid3 = np.isfinite(ga) & np.isfinite(da) & np.isfinite(sa_stab)
        if valid3.sum() >= 4:
            rows_sp, spec_pair = _specificity_for_control(
                ga, da, sa_stab, "S_stab_pair", window,
                min(args.n_bootstrap, 200),
                args.specificity_delta_r2, rng
            )
            for r in rows_sp:
                r["pair"] = lbl
                r["level"] = "pair"
                r["test"] = "test3_specificity_pair"
            test_rows += rows_sp

        # Pair "passes" = coincidence AND primary specificity
        pair_passes = coin_pair and spec_pair
        if pair_passes:
            n_passing_both += 1
        test_rows.append({
            "test": "pairwise_verdict", "level": "pair", "pair": lbl,
            "observable": "coincidence+specificity",
            "statistic_name": "pair_passes_both",
            "value": int(pair_passes), "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
            "window": window,
            "verdict_component": f"coincidence={coin_pair} spec_S_stab={spec_pair}",
        })

    print(f"  Pairs passing both coincidence + specificity: {n_passing_both}/{len(PAIRS)}")
    return test_rows, n_passing_both


# ---------------------------------------------------------------------------
# Step 8 — Nulls (fixed alignment, true pair-label shuffle, no isotropic)
# ---------------------------------------------------------------------------

def step8_nulls(
    args: argparse.Namespace,
    merged_rvec: dict,
    merged_tidx: dict,
    geom_rows: list[dict],
    geom_pair_rows: list[dict],
    func_rows: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Phase-shuffle and pair-label-shuffle nulls. Cloud-isotropic removed."""
    print("[Step 8] Nulls (phase-shuffle, pair-label-shuffle)...")
    test_rows: list[dict] = []
    window = args.primary_window
    n_nulls = min(args.n_bootstrap, 500)

    Ls, g_sep, d_acc, _, _, _, in_core = _extract_arrays(
        geom_rows, func_rows, window, args.render_limit_control, args.core_range
    )
    valid = np.isfinite(g_sep) & np.isfinite(d_acc)

    # ---- Phase-shuffle null ----
    # For each logmar, permute trial responses across orientations (destroys which
    # orientation has which response pattern), recompute G_sep at that logmar.
    ph_slopes: list[float] = []
    for _ in range(n_nulls):
        g_shuf = np.full(len(Ls), float("nan"))
        for i, lm in enumerate(Ls):
            lm_key = round(lm, 3)
            real_rvec = merged_rvec.get("real", {}).get(lm_key, {})
            stab_rvec = merged_rvec.get("stabilized", {}).get(lm_key, {})
            all_real = [resp for ori in ORIENTATIONS for resp in real_rvec.get(ori, [])]
            if len(all_real) < len(ORIENTATIONS):
                continue
            perm = rng.permutation(len(all_real))
            n_per = len(all_real) // len(ORIENTATIONS)
            shuf_means: dict[int, np.ndarray] = {}
            for k, ori in enumerate(ORIENTATIONS):
                block = perm[k * n_per: (k + 1) * n_per]
                shuf_means[ori] = np.stack([all_real[j] for j in block]).mean(axis=0)
            stab_means: dict[int, np.ndarray] = {}
            for ori in ORIENTATIONS:
                t = stab_rvec.get(ori, [])
                if t:
                    stab_means[ori] = np.stack(t).mean(axis=0)
            sep_r = _pairwise_mean_sep(shuf_means)
            sep_s = _pairwise_mean_sep(stab_means)
            if np.isfinite(sep_r) and np.isfinite(sep_s):
                g_shuf[i] = sep_r - sep_s
        v = np.isfinite(g_shuf) & np.isfinite(d_acc)
        if v.sum() >= 3:
            sl, *_ = scipy_stats.linregress(g_shuf[v], d_acc[v])
            ph_slopes.append(sl)

    # ---- Pair-label-shuffle null ----
    # For each logmar, randomly permute which pair label (0_vs_90 etc.) maps to which
    # G_sep_pair value before averaging → changes which pairs contribute to G_sep_mean.
    geom_pair_by_lm: dict[float, dict[str, dict]] = defaultdict(dict)
    for r in geom_pair_rows:
        geom_pair_by_lm[float(r["L"])][str(r["pair"])] = r

    pair_lbls = [_pair_label(a, b) for a, b in PAIRS]
    ps_slopes: list[float] = []
    for _ in range(n_nulls):
        g_shuf2 = np.full(len(Ls), float("nan"))
        for i, lm in enumerate(Ls):
            lm_key = round(lm, 3)
            stab_rvec = merged_rvec.get("stabilized", {}).get(lm_key, {})
            stab_means: dict[int, np.ndarray] = {}
            for ori in ORIENTATIONS:
                t = stab_rvec.get(ori, [])
                if t:
                    stab_means[ori] = np.stack(t).mean(axis=0)
            sep_stab = _pairwise_mean_sep(stab_means)

            # Get G_sep_pair values for this logmar, permute pair labels
            pair_seps = {
                lbl: _f(geom_pair_by_lm[lm].get(lbl, {}).get("G_sep_pair"))
                for lbl in pair_lbls
            }
            vals = list(pair_seps.values())
            finite_vals = [v for v in vals if np.isfinite(v)]
            if not finite_vals:
                continue
            # Permute: reassign G_sep_pair values to shuffled pair labels
            shuf_vals = rng.permutation(finite_vals)
            # Recompute Sep_avg from shuffled G_sep_pairs:
            # G_sep_pair = sep_real_pair - sep_stab_pair
            # shuffled Sep_avg(real) = mean(shuf_val + sep_stab_pair) ≈ mean(shuf_val) + sep_stab_mean
            # We approximate Sep_avg(real_shuffled) - Sep_avg(stab) = mean(shuf_vals)
            # since G_sep_pair = sep_real_pair - sep_stab_pair already incorporates stab.
            g_shuf2[i] = float(np.mean(shuf_vals))
        v = np.isfinite(g_shuf2) & np.isfinite(d_acc)
        if v.sum() >= 3:
            sl, *_ = scipy_stats.linregress(g_shuf2[v], d_acc[v])
            ps_slopes.append(sl)

    # Real slope for comparison
    real_slope = float("nan")
    if valid.sum() >= 3:
        real_slope, *_ = scipy_stats.linregress(g_sep[valid], d_acc[valid])

    # p-value convention: p = fraction of null slopes < real slope
    # High p (close to 1) means real exceeds most of the null → significant
    def _emp_p(real_val: float, null_vals: list[float]) -> float:
        if not null_vals or not np.isfinite(real_val):
            return float("nan")
        return float(np.mean([nv < real_val for nv in null_vals if np.isfinite(nv)]))

    p_phase = _emp_p(real_slope, ph_slopes)
    p_pair = _emp_p(real_slope, ps_slopes)

    # nulls_passed: real slope exceeds 95th percentile of both nulls
    nulls_passed = (
        np.isfinite(p_phase) and p_phase >= 0.95 and
        np.isfinite(p_pair) and p_pair >= 0.95
    )

    def _add(name, val, null_p=float("nan"), note=""):
        test_rows.append({
            "test": "test_nulls", "level": "mean", "pair": "all",
            "observable": name, "statistic_name": name,
            "value": val, "ci_low": "nan", "ci_high": "nan",
            "null_p": null_p, "window": window, "verdict_component": note,
        })

    _add("real_slope", real_slope)
    _add("slope_phase_shuffle_median",
         float(np.nanmedian(ph_slopes)) if ph_slopes else float("nan"),
         p_phase, "p=frac_null_less_than_real; pass>=0.95")
    _add("slope_pair_label_shuffle_median",
         float(np.nanmedian(ps_slopes)) if ps_slopes else float("nan"),
         p_pair, "p=frac_null_less_than_real; pass>=0.95")
    _add("nulls_passed", int(nulls_passed), note="TRUE" if nulls_passed else "FALSE")

    p_phase_s = f"{p_phase:.3f}" if np.isfinite(p_phase) else "nan"
    p_pair_s = f"{p_pair:.3f}" if np.isfinite(p_pair) else "nan"
    print(f"  p(phase_shuffle < real)={p_phase_s}, p(pair_shuffle < real)={p_pair_s}")
    print(f"  nulls_passed={'TRUE' if nulls_passed else 'FALSE'}")
    return test_rows


# ---------------------------------------------------------------------------
# Step 9 — Window robustness (disabled if aggregate accuracy only)
# ---------------------------------------------------------------------------

def step9_window_robustness(
    args: argparse.Namespace,
    geom_rows: list[dict],
    func_rows: list[dict],
    has_window_specific: bool,
) -> tuple[list[dict], bool]:
    print("[Step 9] Window robustness...")
    test_rows: list[dict] = []

    if not has_window_specific:
        note = ("Window robustness DISABLED: accuracy source is aggregate only "
                "(no per-window column). Re-run with window-specific D1 accuracy "
                "to enable this test.")
        test_rows.append({
            "test": "test_window_robustness", "level": "mean", "pair": "all",
            "observable": "window_robustness",
            "statistic_name": "window_stable",
            "value": float("nan"), "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
            "window": args.primary_window, "verdict_component": note,
        })
        print(f"  DISABLED: {note}")
        return test_rows, False

    slopes: list[float] = []
    for w in args.windows:
        Ls, g_sep, d_acc, _, _, _, _ = _extract_arrays(
            geom_rows, func_rows, w, args.render_limit_control, args.core_range
        )
        valid = np.isfinite(g_sep) & np.isfinite(d_acc)
        if valid.sum() < 3:
            continue
        sl, _, r, _, _ = scipy_stats.linregress(g_sep[valid], d_acc[valid])
        slopes.append(sl)
        L_g, gs = _zero_crossing_core(Ls, g_sep, Ls == Ls)
        L_d, ds = _zero_crossing_core(Ls, d_acc, Ls == Ls)
        test_rows.append({
            "test": "test_window_robustness", "level": "mean", "pair": "all",
            "observable": "G_sep_mean vs delta_acc_mean",
            "statistic_name": f"slope_W{w}", "value": sl,
            "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
            "window": w,
            "verdict_component": f"R2={r**2:.3f} L_geom={f'{L_g:.3f}' if np.isfinite(L_g) else 'nan'} L_func={f'{L_d:.3f}' if np.isfinite(L_d) else 'nan'}",
        })

    finite_sl = [s for s in slopes if np.isfinite(s)]
    window_stable = (len(finite_sl) >= 2 and all(s > 0 for s in finite_sl)
                     and max(finite_sl) / max(min(finite_sl), EPS) < 2.0)
    test_rows.append({
        "test": "test_window_robustness", "level": "mean", "pair": "all",
        "observable": "slope_across_windows",
        "statistic_name": "window_stable",
        "value": int(window_stable), "ci_low": "nan", "ci_high": "nan", "null_p": "nan",
        "window": args.primary_window,
        "verdict_component": f"{'TRUE' if window_stable else 'FALSE'} slopes={finite_sl}",
    })
    print(f"  window_stable={'TRUE' if window_stable else 'FALSE'}, slopes={finite_sl}")
    return test_rows, window_stable


# ---------------------------------------------------------------------------
# Step 10 — Decision, tables, readme
# ---------------------------------------------------------------------------

def _extract_verdicts(test_rows: list[dict]) -> dict[str, Any]:
    vd: dict[str, Any] = {}
    for r in test_rows:
        name = str(r.get("statistic_name", ""))
        val = _f(r.get("value", "nan"))
        if name == "coincidence_verdict":
            vd["coincidence"] = bool(int(val)) if np.isfinite(val) else False
        if name == "continuous_significant":
            vd["continuous_significant"] = bool(int(val)) if np.isfinite(val) else False
        if "specificity_passed" in name and "__S_stab" in name:
            vd["specificity_S_stab"] = bool(int(val)) if np.isfinite(val) else False
        if "specificity_passed" in name and "__S_center" in name:
            vd["specificity_S_center"] = bool(int(val)) if np.isfinite(val) else False
        if "specificity_passed" in name and "__S_cloud" in name:
            vd["specificity_S_cloud"] = bool(int(val)) if np.isfinite(val) else False
        if name == "mechanism_tangent_tracks":
            vd["mechanism_tangent_tracks"] = bool(int(val)) if np.isfinite(val) else False
        if name == "nulls_passed":
            vd["nulls_passed"] = bool(int(val)) if np.isfinite(val) else False
        if name == "window_stable":
            vd["window_stable"] = bool(int(val)) if np.isfinite(val) else False
        if name == "render_limit_confounded":
            vd["render_limit_confounded"] = bool(int(val)) if np.isfinite(val) else False
    return vd


def _apply_decision_logic(
    vd: dict,
    test_rows: list[dict],
    args: argparse.Namespace,
    confirmed_d1_source: str | None,
    has_window_specific: bool,
    n_pairs_passing: int,
    primary_difficulty: str,
) -> tuple[str, str, str]:
    if confirmed_d1_source is None:
        label = "exploratory_only_missing_D1_source"
        impl = (f"D1 accuracy source not confirmed for column '{args.d1_acc_col}'. "
                "All function-curve values are NaN. No verdict possible.")
        action = "Confirm D1 source path and column. Check --d1-acc-col and --accuracy-scan-dirs."
        return label, impl, action

    coincidence = vd.get("coincidence", False)
    continuous = vd.get("continuous_significant", False)
    specificity_primary = vd.get(f"specificity_{primary_difficulty}", False)
    specificity_any = any(vd.get(f"specificity_{s}", False)
                          for s in ["S_stab", "S_center", "S_cloud_mean"])
    mechanism = vd.get("mechanism_tangent_tracks", False)
    nulls = vd.get("nulls_passed", False)
    render_confounded = vd.get("render_limit_confounded", False)

    # Retrieve dL values
    dL = next((_f(r.get("value")) for r in test_rows if r.get("statistic_name") == "dL"),
               float("nan"))
    dL_ci_high = next((_f(r.get("ci_high")) for r in test_rows if r.get("statistic_name") == "dL"),
                       float("nan"))
    L_func = next((_f(r.get("value")) for r in test_rows if r.get("statistic_name") == "L_func_crossing"),
                   float("nan"))
    L_geom = next((_f(r.get("value")) for r in test_rows if r.get("statistic_name") == "L_geom_crossing"),
                   float("nan"))

    if render_confounded:
        label = "render_limit_confounded"
        impl = "Crossing/effect requires -0.40 control point. Inconclusive."
        action = "Add finer logmar grid within core range; check renderer at -0.40."

    elif coincidence and continuous and specificity_primary and mechanism and nulls:
        label = "geometry_predicts_global_crossover"
        impl = ("Tier-1 and Tier-2 both pass. Fig 4 leads with geometry: "
                "translation-tangent manifold predicts AND explains the crossover.")
        action = "Finalize Fig 4. Write Tier-1 and Tier-2 subsections."

    elif coincidence and continuous and specificity_primary and not mechanism:
        label = "tier2_jacobian_missing"
        impl = ("Tier-1 (cloud_separability) passes all criteria. "
                "Tier-2 (Jacobian/ΔM) cannot be assessed — raw J matrices not computed. "
                "Cannot attribute crossing to translation-tangent mechanism yet.")
        action = ("Compute Jacobians via eoptotype_jacobian_field_smoothness.py "
                  "storing raw J arrays. Re-run to unlock Tier-2.")

    elif coincidence and continuous and not specificity_primary and specificity_any:
        label = "cloud_separability_predicts_crossover"
        impl = (f"Cloud-separability gain G_sep coincides with and continuously predicts Δacc. "
                f"Specificity beyond primary control ({primary_difficulty}) not confirmed, "
                f"but at least one other control passes.")
        action = "Report cloud-separability result. Investigate which difficulty control explains most variance."

    elif coincidence and continuous and not specificity_any:
        label = "cloud_separability_tracks_difficulty"
        impl = ("G_sep continuous with Δacc but not beyond any difficulty control. "
                "Geometry tracks task difficulty (S); G_sep adds nothing once S is partialled. "
                "Lead with acuity result; geometry is descriptive of difficulty.")
        action = "Decompose G_sep vs S_stab correlation. Consider whether cloud improves S, not G_sep per se."

    elif not coincidence and n_pairs_passing > 0:
        label = "geometry_partially_predicts_pairwise_benefit"
        impl = (f"Global mean fails but {n_pairs_passing}/{len(PAIRS)} pairs pass "
                "coincidence + specificity. Biologically meaningful but not a global claim.")
        action = "Report which pairs and biological interpretation."

    elif not coincidence and not continuous:
        label = "geometry_descriptive_not_predictive"
        impl = ("G_sep well-behaved but does not predict Δacc. "
                "Drop predicts-function claim; geometry is a separate descriptive result.")
        action = "Separate geometry and function sections."

    else:
        label = "underpowered_or_missing_inputs"
        impl = ("Tests inconclusive — likely insufficient logmar coverage in core range. "
                "Await running job completion or add missing logmars.")
        action = "Check grid_reconciliation.csv for missing core-range points."

    return label, impl, action


def step10_decision_and_output(
    args: argparse.Namespace,
    geom_rows: list[dict],
    geom_pair_rows: list[dict],
    func_rows: list[dict],
    func_pair_rows: list[dict],
    all_test_rows: list[dict],
    mechanism_tangent_tracks: bool,
    window_stable: bool,
    n_pairs_passing: int,
    confirmed_d1_source: str | None,
    has_window_specific: bool,
    out_dir: Path,
) -> None:
    print("[Step 10] Writing outputs...")
    window = args.primary_window

    # keystone_contrasts.csv
    geom_by_lm = {float(r["L"]): r for r in geom_rows}
    func_by_lm = {float(r["L"]): r for r in func_rows if int(r.get("window", 0)) == window}
    contrast_rows = []
    for lm in args.logmar_grid:
        gr = geom_by_lm.get(lm, {})
        fr = func_by_lm.get(lm, {})
        contrast_rows.append({
            "L": _fmt_lm(lm),
            "window": window,
            "G_sep_mean": gr.get("G_sep_mean", "nan"),
            "dM_mean": gr.get("dM_mean", "nan"),
            "delta_acc_mean": fr.get("delta_acc_mean", "nan"),
            "S_center": gr.get("S_center", "nan"),
            "S_stab": gr.get("S_stab", "nan"),
            "S_cloud_mean": gr.get("S_cloud_mean", "nan"),
        })
    _write_csv(out_dir / "tests" / "keystone_contrasts.csv", contrast_rows)

    # keystone_tests.csv
    fnames_t = ["test", "level", "pair", "observable", "statistic_name",
                "value", "ci_low", "ci_high", "null_p", "window",
                "verdict_component", "specificity_control"]
    for r in all_test_rows:
        for f in fnames_t:
            r.setdefault(f, "")
    _write_csv(out_dir / "tests" / "keystone_tests.csv", all_test_rows, fnames_t)

    # Decision table
    vd = _extract_verdicts(all_test_rows)
    vd["mechanism_tangent_tracks"] = mechanism_tangent_tracks
    vd["window_stable"] = window_stable

    label, impl, action = _apply_decision_logic(
        vd, all_test_rows, args, confirmed_d1_source,
        has_window_specific, n_pairs_passing, args.primary_difficulty_control
    )

    # Crossings for the table
    L_func = next((_f(r.get("value")) for r in all_test_rows
                   if r.get("statistic_name") == "L_func_crossing"), float("nan"))
    L_geom = next((_f(r.get("value")) for r in all_test_rows
                   if r.get("statistic_name") == "L_geom_crossing"), float("nan"))
    dL = next((_f(r.get("value")) for r in all_test_rows
               if r.get("statistic_name") == "dL"), float("nan"))
    dL_ci_low = next((_f(r.get("ci_low")) for r in all_test_rows
                      if r.get("statistic_name") == "dL"), float("nan"))
    dL_ci_high = next((_f(r.get("ci_high")) for r in all_test_rows
                       if r.get("statistic_name") == "dL"), float("nan"))

    decision_row = {
        "run_label": args.run_label,
        "primary_window": window,
        "d1_acc_col": args.d1_acc_col,
        "d1_source": confirmed_d1_source or "NONE",
        "window_specific_acc": "yes" if has_window_specific else "no",
        "L_func_core": f"{L_func:.3f}" if np.isfinite(L_func) else "nan",
        "L_geom_core": f"{L_geom:.3f}" if np.isfinite(L_geom) else "nan",
        "dL": f"{dL:.3f}" if np.isfinite(dL) else "nan",
        "dL_ci": f"[{dL_ci_low:.3f},{dL_ci_high:.3f}]"
                  if np.isfinite(dL_ci_low) and np.isfinite(dL_ci_high) else "nan",
        "coincidence": "TRUE" if vd.get("coincidence") else "FALSE",
        "continuous_significant": "TRUE" if vd.get("continuous_significant") else "FALSE",
        "specificity_S_stab": "TRUE" if vd.get("specificity_S_stab") else "FALSE",
        "specificity_S_center": "TRUE" if vd.get("specificity_S_center") else "FALSE",
        "specificity_S_cloud_mean": "TRUE" if vd.get("specificity_S_cloud_mean") else "FALSE",
        "primary_specificity_control": args.primary_difficulty_control,
        "primary_specificity_passed": "TRUE" if vd.get(
            f"specificity_{args.primary_difficulty_control}", False) else "FALSE",
        "mechanism_tangent_tracks": "TRUE" if mechanism_tangent_tracks else "FALSE",
        "nulls_passed": "TRUE" if vd.get("nulls_passed") else "FALSE",
        "window_stable": "TRUE" if window_stable else "FALSE",
        "pairs_passing_both": f"{n_pairs_passing}/{len(PAIRS)}",
        "render_limit_confounded": "TRUE" if vd.get("render_limit_confounded") else "FALSE",
        "analysis_tier": "cloud_separability_crosswalk",
        "decision_label": label,
        "manuscript_implication": impl,
        "next_action": action,
    }
    _write_csv(out_dir / "tests" / "keystone_decision_table.csv", [decision_row])

    # README
    _write_readme(args, vd, label, impl, action, n_pairs_passing,
                  L_func, L_geom, dL, dL_ci_low, dL_ci_high,
                  confirmed_d1_source, has_window_specific, out_dir)

    print(f"\n  === DECISION: {label} ===")
    print(f"  {impl}")
    print(f"  Next: {action}")


def _write_readme(
    args, vd, label, impl, action, n_pairs_passing,
    L_func, L_geom, dL, dL_ci_low, dL_ci_high,
    confirmed_d1_source, has_window_specific, out_dir: Path
) -> None:
    def _yn(k: str) -> str:
        return "YES" if vd.get(k) else "NO"
    def _fstr(v: float) -> str:
        return f"{v:.3f}" if np.isfinite(v) else "nan"

    lines = [
        "# Keystone Geometry Crossover — Results README",
        "",
        f"**Run label:** {args.run_label}  ",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Primary window:** {args.primary_window} frames  ",
        f"**LogMAR grid:** {list(args.logmar_grid)}  ",
        f"**Core range:** {list(args.core_range)}  ",
        f"**Analysis tier (Tier 1):** `cloud_separability_crosswalk`  ",
        f"**D1 accuracy column:** `{args.d1_acc_col}`  ",
        f"**D1 source:** `{confirmed_d1_source or 'NONE'}`  ",
        f"**Window-specific accuracy:** {'yes' if has_window_specific else 'no (aggregate only)'}  ",
        "",
        "## Summary Answers",
        "",
        f"1. **Coincidence within core range?** {_yn('coincidence')}  ",
        f"   G_sep zero at L_geom={_fstr(L_geom)}; Δacc zero at L_func={_fstr(L_func)}.  ",
        f"   dL={_fstr(dL)}, 95% CI [{_fstr(dL_ci_low)}, {_fstr(dL_ci_high)}].  ",
        f"   Criterion: dL_median < {DEFAULT_COINCIDENCE_GRID_STEP} AND dL_ci_high < {DEFAULT_COINCIDENCE_GRID_STEP}.  ",
        "",
        f"2. **Continuous prediction (G_sep → Δacc)?** {_yn('continuous_significant')}  ",
        "   Linear regression and Spearman ρ across logmar grid.  ",
        "",
        "3. **Beyond difficulty controls?**  ",
        f"   S_stab: {_yn('specificity_S_stab')}  ",
        f"   S_center: {_yn('specificity_S_center')}  ",
        f"   S_cloud_mean: {_yn('specificity_S_cloud_mean')}  ",
        f"   Primary ({args.primary_difficulty_control}): "
        f"{_yn(f'specificity_{args.primary_difficulty_control}')}  ",
        "",
        f"4. **ΔM tracks transition (Tier-2)?** {_yn('mechanism_tangent_tracks')}  ",
        "   Tier-2 deferred: raw Jacobian matrices not cached.  ",
        "   Run `eoptotype_jacobian_field_smoothness.py` storing raw J arrays.  ",
        "",
        f"5. **Pairs passing coincidence + primary specificity?** {n_pairs_passing}/{len(PAIRS)}  ",
        "   See keystone_tests.csv for per-pair breakdown.  ",
        "",
        f"6. **Exceed nulls?** {_yn('nulls_passed')}  ",
        "   Phase-shuffle and pair-label-shuffle tested (cloud-isotropic removed).  ",
        "   p = fraction of null slopes < real slope; threshold ≥ 0.95.  ",
        "",
        f"7. **Window stable?** {'N/A (aggregate accuracy)' if not has_window_specific else _yn('window_stable')}  ",
        "",
        f"8. **-0.40 confound?** {_yn('render_limit_confounded')}  ",
        "",
        f"9. **Decision label:** `{label}`  ",
        f"   {impl}  ",
        "",
        "## Manuscript Implication",
        "",
        f"{impl}",
        "",
        f"**Next action:** {action}",
        "",
        "## Notes",
        "",
        "- `G_sep` is a **decoder-free** mean-response separability predictor "
        "(the functional mirror of D1). This is `cloud_separability_crosswalk` (Tier 1).  ",
        "- The Jacobian-specific mechanism (Tier 2) is **not** assessed here.  "
        "  `geometry_predicts_global_crossover` is reserved for when Tier-2 ΔM passes.  ",
        "- Accuracy CIs are pair-resampling exploratory (aggregate accuracy only).  ",
        "- Window robustness is disabled until window-specific D1 accuracy is available.  ",
    ]
    (out_dir / "keystone_readme.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(
    args: argparse.Namespace,
    geom_rows: list[dict],
    geom_pair_rows: list[dict],
    func_rows: list[dict],
    func_pair_rows: list[dict],
    all_test_rows: list[dict],
    fig_dir: Path,
) -> None:
    print("[Figures] Generating figures A-F...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    window = args.primary_window

    Ls_plot_list = sorted(
        float(r["L"]) for r in geom_rows
        if abs(float(r["L"]) - args.render_limit_control) > 1e-4
    )
    Ls, g_sep, d_acc, s_ctr, s_stab, s_cloud, in_core = _extract_arrays(
        geom_rows, func_rows, window, args.render_limit_control, args.core_range
    )
    func_by_lm = {float(r["L"]): r for r in func_rows if int(r.get("window", 0)) == window}
    d_ci_low = np.array([_f(func_by_lm.get(lm, {}).get("delta_ci_low")) for lm in Ls])
    d_ci_high = np.array([_f(func_by_lm.get(lm, {}).get("delta_ci_high")) for lm in Ls])

    C1, C2 = "#2166AC", "#D6604D"
    grid_step = DEFAULT_COINCIDENCE_GRID_STEP

    # --- Fig A: Keystone overlay ---
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    for lm_ctrl in [args.render_limit_control]:
        ax1.axvspan(lm_ctrl - 0.025, lm_ctrl + 0.025, alpha=0.12, color="gray")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.3)
    ax1.plot(Ls, g_sep, "o-", color=C1, lw=2, label="G_sep_mean")
    ax2.plot(Ls, d_acc, "s--", color=C2, lw=2, label="Δacc_mean (D1)")
    if np.any(np.isfinite(d_ci_low)):
        ax2.fill_between(Ls, d_ci_low, d_ci_high, alpha=0.2, color=C2)
    ax1.set_ylabel("G_sep_mean", color=C1, fontsize=10)
    ax2.set_ylabel("Δacc_mean (real − stab)", color=C2, fontsize=10)
    ax1.set_xlabel("LogMAR", fontsize=10)
    ax1.set_title("Fig A: cloud_separability_crosswalk overlay", fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA_keystone_overlay.pdf", dpi=200)
    fig.savefig(fig_dir / "figA_keystone_overlay.png", dpi=150)
    plt.close(fig)

    # --- Fig B: Continuous scatter ---
    valid = np.isfinite(g_sep) & np.isfinite(d_acc)
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(g_sep[valid], d_acc[valid], c=Ls[valid], cmap="plasma",
                    s=60, edgecolors="k", lw=0.5, zorder=3)
    if valid.sum() >= 3:
        sl, ic, r, _, _ = scipy_stats.linregress(g_sep[valid], d_acc[valid])
        xf = np.linspace(g_sep[valid].min(), g_sep[valid].max(), 50)
        ax.plot(xf, sl * xf + ic, "k-", lw=1.5, label=f"slope={sl:.3f}, R²={r**2:.2f}")
        ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="LogMAR")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("G_sep_mean"); ax.set_ylabel("Δacc_mean")
    ax.set_title("Fig B: Continuous G_sep → Δacc")
    fig.tight_layout()
    fig.savefig(fig_dir / "figB_continuous.pdf", dpi=200)
    fig.savefig(fig_dir / "figB_continuous.png", dpi=150)
    plt.close(fig)

    # --- Fig C: Difficulty controls (S_center, S_stab, S_cloud_mean) ---
    s_arrays = [("S_center", s_ctr), ("S_stab", s_stab), ("S_cloud_mean", s_cloud)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (sname, s_arr) in zip(axes, s_arrays):
        v3 = valid & np.isfinite(s_arr)
        if v3.sum() >= 3:
            def _resid(y, x):
                sl, ic, _, _, _ = scipy_stats.linregress(x, y)
                return y - (sl * x + ic)
            dr = _resid(d_acc[v3], s_arr[v3])
            gr = _resid(g_sep[v3], s_arr[v3])
            ax.scatter(gr, dr, c=Ls[v3], cmap="plasma", s=50, edgecolors="k", lw=0.4)
            if len(gr) >= 3:
                sl, ic, r, _, _ = scipy_stats.linregress(gr, dr)
                xf = np.linspace(gr.min(), gr.max(), 50)
                ax.plot(xf, sl * xf + ic, "k-", lw=1.5, label=f"R²={r**2:.2f}")
                ax.legend(fontsize=7)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(f"G_sep | {sname}", fontsize=8)
        ax.set_ylabel("Δacc | " + sname if sname == "S_stab" else "", fontsize=8)
        prim = " [PRIMARY]" if sname == args.primary_difficulty_control else ""
        ax.set_title(f"vs {sname}{prim}", fontsize=9)
    fig.suptitle("Fig C: Specificity beyond difficulty controls", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "figC_difficulty_controls.pdf", dpi=200)
    fig.savefig(fig_dir / "figC_difficulty_controls.png", dpi=150)
    plt.close(fig)

    # --- Fig D: Tier-2 placeholder ---
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5,
            "Tier-2 (ΔM) deferred.\nRaw J matrices not cached.\n"
            "Run eoptotype_jacobian_field_smoothness.py\nstoring raw J arrays.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Fig D: Tier-2 mechanism (ΔM vs G_sep) — DEFERRED", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "figD_tier2.pdf", dpi=200)
    fig.savefig(fig_dir / "figD_tier2.png", dpi=150)
    plt.close(fig)

    # --- Fig E: Pairwise small multiples ---
    gp_by = {(float(r["L"]), str(r["pair"])): r for r in geom_pair_rows}
    fp_by = {(float(r["L"]), str(r["pair"])): r for r in func_pair_rows
             if int(r.get("window", 0)) == window}
    ncols, nrows = 3, math.ceil(len(PAIRS) / 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)
    for idx, (a, b) in enumerate(PAIRS):
        ax = axes[idx // ncols][idx % ncols]
        lbl = _pair_label(a, b)
        gvals = np.array([_f(gp_by.get((lm, lbl), {}).get("G_sep_pair")) for lm in Ls])
        dvals = np.array([_f(fp_by.get((lm, lbl), {}).get("delta_acc_pair")) for lm in Ls])
        ax2 = ax.twinx()
        ax.plot(Ls, gvals, "o-", color=C1, lw=1.5, ms=4)
        ax2.plot(Ls, dvals, "s--", color=C2, lw=1.5, ms=4)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax2.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(lbl, fontsize=8)
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)
    for idx in range(len(PAIRS), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Fig E: Pairwise G_sep vs Δacc", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "figE_pairwise.pdf", dpi=200)
    fig.savefig(fig_dir / "figE_pairwise.png", dpi=150)
    plt.close(fig)

    # --- Fig F: Null comparison ---
    real_slope_row = next((r for r in all_test_rows if r.get("statistic_name") == "real_slope"), None)
    real_slope = _f(real_slope_row.get("value")) if real_slope_row else float("nan")
    null_rows = [r for r in all_test_rows if "shuffle" in str(r.get("statistic_name", ""))]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if np.isfinite(real_slope):
        ax.axvline(real_slope, color="red", lw=2, label=f"Real slope={real_slope:.4f}")
    for nr in null_rows:
        nv = _f(nr.get("value"))
        if np.isfinite(nv):
            nm = str(nr.get("statistic_name", "")).replace("slope_", "").replace("_median", "")
            p = _f(nr.get("null_p"))
            ax.axvline(nv, lw=1.5, ls="--", label=f"{nm} median={nv:.4f} (p={f'{p:.3f}' if np.isfinite(p) else 'nan'})")
    ax.set_xlabel("Slope (G_sep → Δacc)")
    ax.set_title("Fig F: Null comparison (slope medians)", fontsize=11)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.97, "Full null distributions in keystone_tests.csv",
            transform=ax.transAxes, fontsize=7, va="top")
    fig.tight_layout()
    fig.savefig(fig_dir / "figF_nulls.pdf", dpi=200)
    fig.savefig(fig_dir / "figF_nulls.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keystone geometry crossover analysis (patched v2)")
    p.add_argument("--logmar-grid", nargs="+", type=float, default=list(DEFAULT_LOGMAR_GRID))
    p.add_argument("--render-limit-control", type=float, default=DEFAULT_RENDER_LIMIT_CONTROL)
    p.add_argument("--core-range", nargs=2, type=float, default=list(DEFAULT_CORE_RANGE))
    p.add_argument("--primary-window", type=int, default=DEFAULT_PRIMARY_WINDOW)
    p.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--d1-acc-col", type=str, default=DEFAULT_D1_ACC_COL,
                   help="Column name for D1 accuracy in decoder_metrics CSV (legacy fallback).")
    p.add_argument("--d1-sweep-path", type=str, default=str(DEFAULT_D1_SWEEP_PATH),
                   help="Path to eoptotype_D1_integration_window_sweep.csv. "
                        "When present and valid, takes precedence over --d1-acc-col.")
    p.add_argument("--d1-sweep-col", type=str, default=DEFAULT_D1_SWEEP_COL,
                   help="Column in the sweep file providing the pre-computed Δacc "
                        "(real − stabilized). E.g. real_minus_stabilized_d1_time_mean_accuracy.")
    p.add_argument("--d1-sweep-readout", type=str, default=DEFAULT_D1_SWEEP_READOUT,
                   help="readout_type filter when loading the sweep file (default: linear).")
    p.add_argument("--feature-repr", type=str, default=DEFAULT_FEATURE_REPR)
    p.add_argument("--primary-difficulty-control", type=str,
                   default=DEFAULT_PRIMARY_DIFFICULTY,
                   choices=["S_center", "S_stab", "S_cloud_mean"])
    p.add_argument("--specificity-delta-r2", type=float, default=DEFAULT_SPECIFICITY_DELTA_R2)
    p.add_argument("--accuracy-scan-dirs", nargs="+", default=list(DEFAULT_ACCURACY_SCAN_DIRS))
    p.add_argument("--accuracy-sweep", type=str, default="AUTO")
    p.add_argument("--jacobian-smoothness-dir", type=Path,
                   default=DEFAULT_JACOBIAN_SMOOTHNESS_DIR)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--run-label", type=str, default="keystone")
    p.add_argument("--cache-audit-only", action="store_true",
                   help="Run Step 0 only and print cache audit summary. Do not proceed to analysis.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--phase-grid", type=int, default=33)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--model-type", type=str, default=None)
    p.add_argument("--model-index", type=int, default=0)
    p.add_argument("--mcfarland-outputs", type=Path, default=None)
    p.add_argument("--dataset-idx", type=int, default=10)
    p.add_argument("--recompute-accuracy", action="store_true")
    args = p.parse_args()
    args.logmar_grid = tuple(float(x) for x in args.logmar_grid)
    args.core_range = tuple(float(x) for x in args.core_range)
    args.windows = tuple(int(x) for x in args.windows)
    args.accuracy_scan_dirs = tuple(str(s) for s in args.accuracy_scan_dirs)
    return args


def main() -> None:
    t0 = time.time()
    args = _parse_args()
    rng = np.random.default_rng(args.random_seed)

    out_dir = Path(args.out_dir)
    for sub in ("geometry_curves", "function_curves", "tests", "figures", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"Keystone geometry crossover v2 | run_label={args.run_label}")
    print(f"LogMAR grid: {args.logmar_grid}")
    print(f"Core range: {args.core_range} | Primary window: {args.primary_window}")
    print(f"D1 accuracy column: '{args.d1_acc_col}'")
    print(f"Output: {out_dir}")

    # Step 0
    merged_rvec, merged_tidx, acc_data, confirmed_d1_source, has_window_specific = (
        step0_cache_audit(args, out_dir)
    )

    if args.cache_audit_only:
        print("\n[--cache-audit-only] Stopping after Step 0.")
        print(f"  cache_availability_report.csv → {out_dir / 'cache_availability_report.csv'}")
        print(f"  grid_reconciliation.csv        → {out_dir / 'grid_reconciliation.csv'}")
        return

    # Step 1
    geom_rows, geom_pair_rows = step1_geometry_curves(
        args, merged_rvec, merged_tidx, out_dir
    )

    # Step 2 — raises if D1 source missing
    func_rows, func_pair_rows = step2_function_curves(
        args, acc_data, confirmed_d1_source, has_window_specific,
        merged_rvec, out_dir, rng
    )

    # Steps 3–6
    all_test_rows: list[dict] = []
    all_test_rows += step3_test1_coincidence(args, geom_rows, func_rows, rng)
    all_test_rows += step4_test2_continuous(args, geom_rows, func_rows, rng)
    sp_rows, primary_spec_passed = step5_test3_specificity(args, geom_rows, func_rows, rng)
    all_test_rows += sp_rows

    # Jacobian cache info for Step 6
    jac_rows_cache = _scan_jacobian(args.jacobian_smoothness_dir, args.logmar_grid)
    mech_rows, mechanism_tangent_tracks = step6_test4_mechanism(args, jac_rows_cache)
    all_test_rows += mech_rows

    # Step 7
    pw_rows, n_pairs_passing = step7_pairwise(args, geom_pair_rows, func_pair_rows, rng)
    all_test_rows += pw_rows

    # Step 8
    all_test_rows += step8_nulls(
        args, merged_rvec, merged_tidx, geom_rows, geom_pair_rows, func_rows, rng
    )

    # Step 9
    win_rows, window_stable = step9_window_robustness(
        args, geom_rows, func_rows, has_window_specific
    )
    all_test_rows += win_rows

    # Step 10
    step10_decision_and_output(
        args, geom_rows, geom_pair_rows, func_rows, func_pair_rows,
        all_test_rows, mechanism_tangent_tracks, window_stable,
        n_pairs_passing, confirmed_d1_source, has_window_specific, out_dir
    )

    # Figures
    _make_figures(
        args, geom_rows, geom_pair_rows, func_rows, func_pair_rows,
        all_test_rows, out_dir / "figures"
    )

    print(f"\nDone in {time.time() - t0:.1f}s. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
