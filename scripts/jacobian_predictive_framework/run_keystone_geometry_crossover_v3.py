#!/usr/bin/env python3
"""Keystone geometry crossover v3 — finite-cloud d′_geom.

Observable: d′_geom with separation in numerator and (Σ_int/W + Σ_pos) in denominator.
Arm A: Σ_pos empirical from response-vector cloud (no Jacobian needed).
Arm B: Σ_pos = J Σ_eye Jᵀ (deferred until raw J cached).
Function: 4-way D1 time-mean delta from integration-window sweep (W=60 primary).
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EPS = 1e-12
ORIENTATIONS = (0, 90, 180, 270)
PAIRS = [(a, b) for i, a in enumerate(ORIENTATIONS) for b in ORIENTATIONS[i + 1:]]

# Dense spec grid; reconciliation covers -0.40..-0.20 (step 0.05) only → 4 core pts
SPEC_LOGMAR_GRID = (-0.40, -0.375, -0.35, -0.325, -0.30, -0.275,
                    -0.25, -0.225, -0.20, -0.175, -0.15, -0.125, -0.10)
DEFAULT_LOGMAR_GRID = (-0.40, -0.35, -0.30, -0.25, -0.20, -0.10)   # what's cached
DEFAULT_RENDER_LIMIT = -0.40
DEFAULT_CORE_RANGE = (-0.35, -0.20)
DEFAULT_MIN_CORE_POINTS = 8   # required for continuous/specificity verdict
DEFAULT_GRID_STEP = 0.025     # dense grid step for coincidence criterion
DEFAULT_PRIMARY_WINDOW = 60
DEFAULT_WINDOWS = (12, 24, 60)
DEFAULT_N_MC = 2000           # ideal-observer Monte Carlo samples per class
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_RANDOM_SEED = 0
DEFAULT_SPECIFICITY_DELTA_R2 = 0.15
DEFAULT_RIDGE = 1e-6

DEFAULT_FEATURE_REPR = "spatial_avg_time_mean"
DEFAULT_D1_SWEEP_COL = "real_minus_stabilized_d1_time_mean_accuracy"
DEFAULT_D1_SWEEP_READOUT = "linear"
DEFAULT_MCFARLAND_OUTPUTS = ROOT / "scripts" / "mcfarland_outputs.pkl"
DEFAULT_EXPECTED_SESSION = "Allen_2022-04-13"
DEFAULT_EXPECTED_READOUT_UNITS = 10
DEFAULT_CALIBRATION_MODE = "loo"
DEFAULT_CALIBRATION_ANCHOR_LOGMAR = -0.30
DEFAULT_CALIBRATION_MAX_MAE = 0.05

JPF_BASE = ROOT / "outputs" / "jacobian_predictive_framework"
DEFAULT_RVEC_DIRS = (
    "active_sensing_efficiency_e1_d1_reconciliation_20260601",
    "active_sensing_efficiency_e1_fullscope_20260531",
)
DEFAULT_D1_SWEEP_PATH = (
    JPF_BASE / "active_sensing_efficiency_e1_d1_reconciliation_20260601"
    / "eoptotype_D1_integration_window_sweep.csv"
)
DEFAULT_JAC_DIR = ROOT / "outputs/stats/eoptotype_jacobian_field_smoothness_pilot3"
DEFAULT_EYE_TRACES = ROOT / "scripts/temporal_decoding/data/eye_traces.npz"

# Per-position model inference parameters
DEFAULT_PPD = 37.5           # pixels per degree (HiRes renderer)
DEFAULT_N_LAGS = 32
DEFAULT_CLOUD_TRIALS = 20    # FEM traces to sample cloud positions from
DEFAULT_CLOUD_STEP = 3       # subsample every N frames from each trace
DEFAULT_JAC_STEP_PX = 0.125  # for FD Jacobians (Arm B, if ever computed here)


# ---------------------------------------------------------------------------
# Per-position model inference (Arm A proper: μ_θ(p, L) at cloud positions)
# ---------------------------------------------------------------------------

def _load_model_runner(device: str = "cuda") -> Any:
    """Load CurvatureScaleMatchRunner (model + renderer + retina)."""
    import torch
    import dill
    from scripts.jacobian_predictive_framework.run_eoptotype_curvature_scale_match import (
        CurvatureScaleMatchRunner,
    )
    runner = CurvatureScaleMatchRunner(
        device=device,
        pixels_per_degree=DEFAULT_PPD,
        n_lags=DEFAULT_N_LAGS,
        jacobian_step_px=DEFAULT_JAC_STEP_PX,
        model_batch_size=64,
        load_model=True,
    )
    return runner


def _sample_cloud_positions_px(
    eye_traces_path: Path,
    n_trials: int,
    step: int,
    ppd: float,
    rng: np.random.Generator,
    condition: str = "real",
) -> np.ndarray:
    """Sample 2-D eye positions (in pixels) from FEM trace archive.

    For real FEM: subsample frames from the first n_trials traces.
    For stabilized: use trial-mean position for each trace.
    For fixed_center: return a single (0, 0) position (grand mean gaze).
    Returns shape (N_pos, 2) in pixels.
    """
    d = np.load(str(eye_traces_path), allow_pickle=True)
    traces_deg = d["traces"].astype(np.float64)   # (1059, 540, 2) in degrees
    durations = d["durations"].astype(int)          # actual length per trace

    n_use = min(n_trials, traces_deg.shape[0])
    chosen = rng.choice(traces_deg.shape[0], size=n_use, replace=False)

    positions: list[np.ndarray] = []
    if condition == "fixed_center":
        # Grand mean gaze across all traces
        all_pos = np.concatenate([
            traces_deg[i, :durations[i], :] for i in chosen
        ], axis=0)
        grand_mean = all_pos.mean(axis=0)
        positions.append(grand_mean * ppd)
    elif condition == "stabilized":
        for i in chosen:
            trace = traces_deg[i, :durations[i], :]
            mean_pos = trace.mean(axis=0) * ppd
            positions.append(mean_pos)
    else:  # real FEM
        for i in chosen:
            trace = traces_deg[i, :durations[i], :]
            sub = trace[::step] * ppd  # subsample
            positions.extend(sub.tolist())

    return np.array(positions, dtype=np.float64)


def compute_arm_a_per_position(
    runner: Any,
    lm: float,
    orientations: tuple[int, ...],
    eye_traces_path: Path,
    n_cloud_trials: int,
    cloud_step: int,
    n_frames_W: int,
    sigma_model: str,
    n_mc: int,
    rng: np.random.Generator,
) -> dict:
    """Arm A using per-position model inference.

    For each orientation θ and condition (real, stab, center):
      1. Sample cloud positions from FEM traces.
      2. Call runner.evaluate_condition(lm, θ, positions_px) → {pos: rate_vec}.
         rate_vec shape = (N_units,) (time-mean at static position).
      3. ḡ_θ(cond) = mean over positions.
      4. Σ_pos^A(cond) = sample covariance of {rate_vec} over positions.
      5. Σ_tot = Σ_int/W + Σ_pos^A.
      6. d'_pair and Acc_geom via ideal observer Monte Carlo.

    Returns same dict structure as compute_arm_a().
    """
    N = None  # will be set from first response

    # --- sample positions ---
    real_pos_px = _sample_cloud_positions_px(
        eye_traces_path, n_cloud_trials, cloud_step, DEFAULT_PPD, rng, "real")
    stab_pos_px = _sample_cloud_positions_px(
        eye_traces_path, n_cloud_trials, cloud_step, DEFAULT_PPD, rng, "stabilized")
    center_pos_px = _sample_cloud_positions_px(
        eye_traces_path, n_cloud_trials, cloud_step, DEFAULT_PPD, rng, "fixed_center")

    results: dict[str, Any] = {"noise_model": sigma_model, "source": "per_position_model"}

    for cond, pos_px in [("real", real_pos_px),
                          ("stabilized", stab_pos_px),
                          ("fixed_center", center_pos_px)]:
        # Evaluate model at each unique position for each orientation
        ori_responses: dict[int, np.ndarray] = {}  # ori -> (n_pos, N)
        for ori in orientations:
            resp_dict, _ = runner.evaluate_condition(lm, ori, [p for p in pos_px])
            rates = np.array(list(resp_dict.values()), dtype=np.float64)  # (n_pos, N)
            if rates.size == 0:
                continue
            if N is None:
                N = rates.shape[1]
            ori_responses[ori] = rates

        if not ori_responses or N is None:
            results[f"g_bar_{cond}"] = {}
            results[f"sigma_pos_{cond}"] = np.zeros((N or 10, N or 10))
            results[f"sigma_tot_{cond}"] = np.zeros((N or 10, N or 10))
            results[f"dprime_pair_{cond}"] = {}
            results[f"acc_geom_{cond}"] = float("nan")
            continue

        # Class conditional means and positional covariance
        g_bars: dict[int, np.ndarray] = {}
        covs: list[np.ndarray] = []
        for ori, rates in ori_responses.items():
            g_bars[ori] = rates.mean(axis=0)
            if rates.shape[0] >= 2:
                c = np.cov(rates.T, ddof=1)
                if c.ndim == 0:
                    c = float(c) * np.eye(N)
                covs.append(c)

        # Pool within-orientation covariance
        sigma_pos = np.mean(covs, axis=0) if covs else DEFAULT_RIDGE * np.eye(N)
        if cond == "stabilized":
            # Stabilized = single point per trial → set to near-zero
            sigma_pos = sigma_pos * 0.0

        # r_bar for noise model
        all_rates = np.concatenate(list(ori_responses.values()), axis=0)
        r_bar = all_rates.mean(axis=0)

        if sigma_model == "poisson":
            sigma_int = np.diag(r_bar / max(n_frames_W, 1))
        else:
            sigma_int = np.eye(N) * (float(r_bar.mean()) / max(n_frames_W, 1))

        sigma_tot = sigma_int + sigma_pos

        # d' per pair
        dp: dict[tuple[int, int], float] = {}
        for a, b in PAIRS:
            if a in g_bars and b in g_bars:
                dp[(a, b)] = _dprime_pair(g_bars[a], g_bars[b], sigma_tot)

        # 4-way ideal-observer accuracy
        if len(g_bars) == 4:
            acc = _ncm_accuracy(g_bars, sigma_tot, n_mc, rng)
        else:
            acc = float("nan")

        results[f"g_bar_{cond}"] = g_bars
        results[f"sigma_pos_{cond}"] = sigma_pos
        results[f"sigma_tot_{cond}"] = sigma_tot
        results[f"sigma_int"] = sigma_int
        results[f"dprime_pair_{cond}"] = dp
        results[f"acc_geom_{cond}"] = acc
        results[f"n_cloud_positions_{cond}"] = len(pos_px)

    # Contrasts
    dp_real = results.get("dprime_pair_real", {})
    dp_stab = results.get("dprime_pair_stabilized", {})
    delta_dp: dict[tuple[int, int], float] = {}
    for p in PAIRS:
        r_v = dp_real.get(p, float("nan"))
        s_v = dp_stab.get(p, float("nan"))
        delta_dp[p] = r_v - s_v if np.isfinite(r_v) and np.isfinite(s_v) else float("nan")
    results["delta_dprime_pair"] = delta_dp

    acc_real = results.get("acc_geom_real", float("nan"))
    acc_stab = results.get("acc_geom_stabilized", float("nan"))
    results["delta_acc_geom"] = (
        acc_real - acc_stab
        if np.isfinite(acc_real) and np.isfinite(acc_stab) else float("nan"))

    # S controls (single fixed positions — center & stab single points)
    si = results.get("sigma_int", np.zeros((N or 10, N or 10)))
    for ctrl_cond, ctrl_key in [("stabilized", "S_stab"), ("fixed_center", "S_center")]:
        g = results.get(f"g_bar_{ctrl_cond}", {})
        dp_ctrl: dict[tuple[int, int], float] = {}
        for a, b in PAIRS:
            if a in g and b in g:
                dp_ctrl[(a, b)] = _dprime_pair(g[a], g[b], si)
        results[f"dprime_pair_{ctrl_key}"] = dp_ctrl
        if len(g) == 4:
            results[f"acc_geom_{ctrl_key}"] = _ncm_accuracy(g, si, n_mc, rng)
        else:
            results[f"acc_geom_{ctrl_key}"] = float("nan")

    # S_cloud_mean: per-position d', averaged (done differently here: already have per-pos rates)
    real_per_pos: dict[int, np.ndarray] = {}
    for ori, rates in results.get("_real_rates_by_ori", {}).items():
        real_per_pos[ori] = rates

    # Simplified: use per-orientation rate arrays if stored
    per_pos_dp: dict[tuple[int, int], list[float]] = {p: [] for p in PAIRS}
    if hasattr(runner, "_last_real_rates"):
        pass  # not stored; skip
    # Fall back: S_cloud_mean not separately computable without storing all per-pos rates
    results["dprime_pair_S_cloud_mean"] = {
        p: float("nan") for p in PAIRS
    }
    results["dprime_pair_S_best"] = {
        p: float(np.nanmax([
            results.get("dprime_pair_S_stab", {}).get(p, float("nan")),
            results.get("dprime_pair_S_center", {}).get(p, float("nan")),
        ]))
        for p in PAIRS
    }
    s_best_vals = [v for v in results["dprime_pair_S_best"].values() if np.isfinite(v)]
    results["S_best_mean"] = float(np.mean(s_best_vals)) if s_best_vals else float("nan")

    # Sigma_pos diagnostic
    sp_real = results.get("sigma_pos_real", np.zeros((N or 10, N or 10)))
    si_mat = results.get("sigma_int", np.eye(N or 10))
    results["sigma_pos_trace_real"] = float(np.trace(sp_real))
    results["sigma_int_trace"] = float(np.trace(si_mat))
    snr = float(np.trace(sp_real)) / max(float(np.trace(si_mat)), EPS)
    results["sigma_pos_snr"] = snr

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _fmt_lm(v: float) -> str:
    return f"{float(v):.3f}"


def _resolve_d1_sweep_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    raw = getattr(args, "d1_sweep_paths", None)
    if raw:
        return tuple(Path(p) for p in raw)
    return (Path(args.d1_sweep_path),)


def _d1_source_label(paths: tuple[Path, ...]) -> str:
    if len(paths) == 1:
        return paths[0].name
    return ";".join(p.name for p in paths)


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text((",".join(fieldnames or [])) + "\n")
        return
    fnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _pair_label(a: int, b: int) -> str:
    return f"{a}_vs_{b}"


def _interp_zero(xs: np.ndarray, ys: np.ndarray) -> tuple[float, str]:
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
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


def _zero_crossing_core(Ls: np.ndarray, ys: np.ndarray,
                        in_core: np.ndarray) -> tuple[float, str]:
    m = in_core & np.isfinite(Ls) & np.isfinite(ys)
    if m.sum() < 2:
        return float("nan"), "insufficient_core_points"
    return _interp_zero(Ls[m], ys[m])


# ---------------------------------------------------------------------------
# Step 0 — Cache audit
# ---------------------------------------------------------------------------

def _load_rvec_merged(rvec_dirs: tuple[str, ...], feature_repr: str,
                      logmar_grid: tuple[float, ...]
                      ) -> tuple[dict, dict, list[dict]]:
    """Merge response vectors across source dirs.

    merged[cond][lm_key][ori] = list of np.ndarray(N)
    merged_tidx[cond][lm_key][ori] = list of int
    """
    vec_key = f"vectors__{feature_repr}"
    merged: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    merged_tidx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    report: list[dict] = []

    for sd in rvec_dirs:
        p = JPF_BASE / sd / "eoptotype_identity" / "eoptotype_response_vectors.npz"
        if not p.exists():
            for lm in logmar_grid:
                report.append({"source_dir": sd, "logmar": _fmt_lm(lm),
                                "arm_A_ready": "no", "note": "no_rvec_file"})
            continue
        d = np.load(str(p), allow_pickle=True)
        if vec_key not in d:
            for lm in logmar_grid:
                report.append({"source_dir": sd, "logmar": _fmt_lm(lm),
                                "arm_A_ready": "no", "note": f"key_{vec_key}_missing"})
            continue
        vecs = d[vec_key]
        d_cond = d["condition"]
        d_lm = d["logmar"].astype(float)
        d_ori = d["orientation"].astype(int)
        d_tidx = d["trial_index"].astype(int)

        for lm in logmar_grid:
            for cond in ("real", "stabilized", "fixed_center"):
                for ori in ORIENTATIONS:
                    mask = ((d_cond == cond) &
                            (np.abs(d_lm - lm) < 0.005) &
                            (d_ori == ori))
                    n = int(mask.sum())
                    if n > 0:
                        lk = round(lm, 4)
                        merged[cond][lk][ori].extend(vecs[mask])
                        merged_tidx[cond][lk][ori].extend(d_tidx[mask].tolist())
                    report.append({
                        "source_dir": sd, "quantity": "response_vectors",
                        "condition": cond, "logmar": _fmt_lm(lm), "orientation": ori,
                        "n_trials": n, "arm_A_ready": "yes" if n > 0 else "no",
                        "note": "" if n > 0 else "missing",
                    })

    return merged, merged_tidx, report


def _load_d1_sweep(sweep_paths: tuple[Path, ...], sweep_col: str,
                   logmar_grid: tuple[float, ...],
                   feature_repr: str, readout_type: str
                   ) -> tuple[dict, str | None, bool, list[dict]]:
    """Load D1 time-mean sweep.

    Returns (acc_data, confirmed_source, has_window_specific, report_rows).
    acc_data[lm_key][pair_label][window_int] = delta_float
    """
    report: list[dict] = []
    all_rows: list[dict] = []
    valid_sources: list[str] = []
    for sweep_path in sweep_paths:
        if not sweep_path.exists():
            continue
        rows = _read_csv(sweep_path)
        if not rows or sweep_col not in rows[0]:
            continue
        all_rows.extend(rows)
        valid_sources.append(str(sweep_path))

    if not all_rows:
        for lm in logmar_grid:
            report.append({"quantity": "d1_sweep", "logmar": _fmt_lm(lm),
                            "on_grid": "no_file",
                            "note": ";".join(str(p) for p in sweep_paths)})
        return {}, None, False, report

    acc_data: dict = {}
    found_lms: set[float] = set()
    for row in all_rows:
        if str(row.get("condition", "")) != "real":
            continue
        if str(row.get("feature_representation", "")) != feature_repr:
            continue
        if str(row.get("readout_type", "")) != readout_type:
            continue
        lm = _f(row.get("logmar", "nan"))
        if not any(abs(lm - glm) < 0.005 for glm in logmar_grid):
            continue
        try:
            win = int(float(row.get("integration_window", "nan")))
        except (ValueError, TypeError):
            continue
        pair = str(row.get("orientation_pair", ""))
        delta = _f(row.get(sweep_col, "nan"))
        lk = round(lm, 4)
        found_lms.add(lk)
        acc_data.setdefault(lk, {}).setdefault(pair, {})[win] = delta

    for lm in logmar_grid:
        lk = round(lm, 4)
        on = "yes" if lk in found_lms else "no"
        report.append({"quantity": "d1_sweep", "logmar": _fmt_lm(lm),
                        "d1_col": sweep_col, "on_grid": on,
                        "source": ";".join(valid_sources)})

    confirmed = ";".join(valid_sources) if found_lms else None
    return acc_data, confirmed, True, report


def _check_raw_j(jac_dir: Path, logmar_grid: tuple[float, ...]) -> list[dict]:
    rows: list[dict] = []
    diag = jac_dir / "jacobian_grid_diagnostics.csv"
    if not diag.exists():
        for lm in logmar_grid:
            rows.append({"quantity": "raw_J", "logmar": _fmt_lm(lm),
                         "arm_B_ready": "no", "note": "no_diag_file"})
        return rows
    jrows = _read_csv(diag)
    found = {round(_f(r.get("logmar", "nan")), 4) for r in jrows}
    col_names = list(jrows[0].keys()) if jrows else []
    has_raw = any("component" in c for c in col_names)
    for lm in logmar_grid:
        lk = round(lm, 4)
        if lk in found:
            rows.append({"quantity": "raw_J", "logmar": _fmt_lm(lm),
                         "arm_B_ready": "yes" if has_raw else "no",
                         "note": "" if has_raw else "norms_only_need_full_J"})
        else:
            rows.append({"quantity": "raw_J", "logmar": _fmt_lm(lm),
                         "arm_B_ready": "no", "note": "not_in_cache"})
    return rows


def step0_cache_audit(args: argparse.Namespace, out_dir: Path
                      ) -> tuple[dict, dict, dict, str | None, bool]:
    print("[Step 0] Cache audit...")
    d1_paths = _resolve_d1_sweep_paths(args)
    merged, merged_tidx, rvec_report = _load_rvec_merged(
        args.rvec_dirs, args.feature_repr, args.logmar_grid)
    acc_data, confirmed_d1, has_win, d1_report = _load_d1_sweep(
        d1_paths, args.d1_sweep_col,
        args.logmar_grid, args.feature_repr, args.d1_sweep_readout)
    jac_report = _check_raw_j(Path(args.jac_dir), args.logmar_grid)

    # Grid reconciliation
    core_ready = []
    grid_rows: list[dict] = []
    for lm in args.logmar_grid:
        lk = round(lm, 4)
        in_core = (args.core_range[0] - 1e-4) <= lm <= (args.core_range[1] + 1e-4)
        is_ctrl = abs(lm - args.render_limit) < 1e-4
        n_real = sum(len(merged.get("real", {}).get(lk, {}).get(ori, []))
                     for ori in ORIENTATIONS)
        d1_ok = any(abs(_f(r.get("logmar")) - lm) < 0.005 and r.get("on_grid") == "yes"
                    for r in d1_report)
        rvec_ok = n_real > 0
        arm_a = rvec_ok and d1_ok
        if in_core and arm_a:
            core_ready.append(lm)
        grid_rows.append({
            "logmar": _fmt_lm(lm), "in_core": "yes" if in_core else "no",
            "render_ctrl": "yes" if is_ctrl else "no",
            "n_trials_real": n_real, "d1_ok": "yes" if d1_ok else "no",
            "arm_A_ready": "yes" if arm_a else "no",
            "arm_B_ready": "no",   # norms only for now
        })

    n_core = len([lm for lm in args.logmar_grid
                  if (args.core_range[0]-1e-4) <= lm <= (args.core_range[1]+1e-4)
                  and abs(lm - args.render_limit) > 1e-4])
    underpowered = len(core_ready) < args.min_core_points

    _write_csv(out_dir / "grid_reconciliation.csv", grid_rows)
    all_cache = rvec_report + d1_report + jac_report
    fnames = ["quantity", "source_dir", "condition", "logmar", "orientation",
              "n_trials", "d1_col", "source", "on_grid", "arm_A_ready",
              "arm_B_ready", "note"]
    for r in all_cache:
        for f in fnames:
            r.setdefault(f, "")
    _write_csv(out_dir / "cache_availability_report.csv", all_cache, fnames)

    # Summary print
    print(f"\n=== Cache Audit Summary (v3) ===")
    print(f"  D1 source  : {_d1_source_label(d1_paths)}")
    print(f"  D1 column  : '{args.d1_sweep_col}'")
    print(f"  D1 confirmed: {confirmed_d1 is not None}")
    print(f"  4-way D1   : NOT available (pairwise mean will be used as proxy; flagged)")
    print(f"  Win-specific: {'YES' if has_win else 'NO'}")
    print(f"\n  {'LogMAR':>8}  {'core':>5}  {'rvec':>5}  {'d1':>5}  {'armA':>6}")
    for r in grid_rows:
        ctrl = " [ctrl]" if r["render_ctrl"] == "yes" else ""
        print(f"  {r['logmar']:>8}{ctrl}  {r['in_core']:>5}  "
              f"{r['n_trials_real']:>5}  {r['d1_ok']:>5}  {r['arm_A_ready']:>6}")
    print(f"\n  Core points ready: {len(core_ready)}/{n_core} "
          f"(need {args.min_core_points} for continuous/specificity)")
    print(f"  Underpowered: {underpowered}")
    print(f"  Arm B (raw J): deferred (norms only)\n")
    print("=================================\n")

    return merged, merged_tidx, acc_data, confirmed_d1, has_win


# ---------------------------------------------------------------------------
# Arm A — finite-cloud d′_geom
# ---------------------------------------------------------------------------

def _compute_r_bar(merged: dict, feature_repr: str) -> np.ndarray:
    """Global mean rate per neuron, averaged over all trials."""
    all_vecs: list[np.ndarray] = []
    for cond_dict in merged.values():
        for lm_dict in cond_dict.values():
            for ori_list in lm_dict.values():
                all_vecs.extend(ori_list)
    if not all_vecs:
        raise RuntimeError("No response vectors found — cannot compute r_bar.")
    arr = np.stack(all_vecs, axis=0).astype(np.float64)
    return arr.mean(axis=0)  # shape (N,)


def _sigma_int_poisson(r_bar: np.ndarray, n_frames: int) -> np.ndarray:
    """Σ_int_Poisson = diag(r_bar) / n_frames  (N×N)."""
    return np.diag(r_bar / max(n_frames, 1))


def _sigma_int_isotropic(r_bar: np.ndarray, n_frames: int) -> np.ndarray:
    """Σ_int_iso = mean(r_bar)/n_frames * I  (N×N)."""
    N = len(r_bar)
    return np.eye(N) * (float(r_bar.mean()) / max(n_frames, 1))


def _pooled_within_class_cov(trial_arrays: dict[int, np.ndarray],
                              ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Pooled within-orientation covariance from trial response arrays.

    trial_arrays: {ori: (n_trials, N)}
    Returns (N, N).
    """
    covs: list[np.ndarray] = []
    ns: list[int] = []
    for arr in trial_arrays.values():
        if arr.shape[0] < 2:
            continue
        c = np.cov(arr.T, ddof=1)   # (N, N)
        if c.ndim == 0:
            N = arr.shape[1]
            c = float(c) * np.eye(N)
        covs.append(c)
        ns.append(arr.shape[0] - 1)

    if not covs:
        N = next(iter(trial_arrays.values())).shape[1] if trial_arrays else 10
        return ridge * np.eye(N)

    total_n = sum(ns)
    pooled = sum(n * c for n, c in zip(ns, covs)) / total_n
    N = pooled.shape[0]
    return pooled + ridge * np.eye(N)


def _ncm_accuracy(g_bars: dict[int, np.ndarray], sigma_tot: np.ndarray,
                  n_mc: int, rng: np.random.Generator,
                  ridge: float = DEFAULT_RIDGE) -> float:
    """4-way nearest-class-mean ideal observer accuracy (Monte Carlo).

    Draws n_mc samples per class from N(g_bar_θ, sigma_tot), classifies by
    minimum Mahalanobis distance, returns fraction correct.
    """
    oris = sorted(g_bars.keys())
    means = np.stack([g_bars[o] for o in oris])  # (C, N)
    N = means.shape[1]
    S = sigma_tot + ridge * np.eye(N)
    # Cholesky factor for sampling and distance
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Fallback: EVD clamp
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, ridge)
        S = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(S)

    # Precompute Σ^{-1} via Cholesky solve for efficiency
    # Mahalanobis²(x, μ_c) = ||L^{-1}(x - μ_c)||²
    L_inv = np.linalg.inv(L)          # (N, N)
    means_w = (L_inv @ means.T).T     # (C, N) whitened means

    correct = 0
    for true_idx in range(len(oris)):
        eps = rng.standard_normal((n_mc, N))
        samples = means[true_idx] + eps @ L.T  # (n_mc, N)
        samples_w = (L_inv @ samples.T).T       # (n_mc, N)
        # Distance to each whitened mean
        dists = np.sum((samples_w[:, None, :] - means_w[None, :, :]) ** 2,
                       axis=-1)  # (n_mc, C)
        pred = np.argmin(dists, axis=1)
        correct += int(np.sum(pred == true_idx))

    return float(correct) / (n_mc * len(oris))


def _load_d1_absolute(
    sweep_paths: tuple[Path, ...],
    logmar_grid: tuple[float, ...],
    feature_repr: str,
    readout_type: str,
    window: int,
) -> dict:
    """Load absolute D1 accuracy per (lm_key, pair_label, condition) from sweep file.

    Returns d1_abs[lm_key][pair_label][condition] = accuracy_float.
    Both 'real' and 'stabilized' (and 'fixed_center') are loaded.
    """
    d1_abs: dict = {}
    rows: list[dict] = []
    for sweep_path in sweep_paths:
        if not sweep_path.exists():
            continue
        rows.extend(_read_csv(sweep_path))
    if not rows:
        return d1_abs
    for row in rows:
        if str(row.get("feature_representation", "")) != feature_repr:
            continue
        if str(row.get("readout_type", "")) != readout_type:
            continue
        try:
            win = int(float(row.get("integration_window", "nan")))
        except (ValueError, TypeError):
            continue
        if win != window:
            continue
        lm = _f(row.get("logmar", "nan"))
        if not any(abs(lm - glm) < 0.005 for glm in logmar_grid):
            continue
        cond = str(row.get("condition", ""))
        pair = str(row.get("orientation_pair", ""))
        acc = _f(row.get("d1_time_mean_accuracy", "nan"))
        lk = round(lm, 4)
        if np.isfinite(acc) and pair:
            d1_abs.setdefault(lk, {}).setdefault(pair, {})[cond] = acc
    return d1_abs


def _load_mcfarland_readout_audit(
    pkl_path: Path,
    expected_session: str,
    expected_units: int,
) -> dict[str, Any]:
    """Audit mcfarland readout source against the requested D1 contract."""
    import dill

    audit = {
        "mcfarland_path": str(pkl_path),
        "session_expected": expected_session,
        "readout_units_expected": int(expected_units),
        "session_found": "",
        "ccnorm_vector_count": float("nan"),
        "readout_units_found": float("nan"),
        "session_match": 0,
        "readout_units_match": 0,
        "audit_pass": 0,
    }
    if not pkl_path.exists():
        return audit

    with pkl_path.open("rb") as f:
        payload = dill.load(f)
    if not isinstance(payload, list) or not payload:
        return audit

    entry = payload[0]
    if not isinstance(entry, dict):
        return audit

    sess = str(entry.get("sess", ""))
    cc = np.asarray(entry.get("ccnorm", {}).get("ccnorm", []), dtype=float)
    units = int(np.sum(np.isfinite(cc) & (cc > 0.5)))

    session_match = int(sess == expected_session)
    unit_match = int(units == int(expected_units))
    audit.update({
        "session_found": sess,
        "ccnorm_vector_count": int(cc.size),
        "readout_units_found": units,
        "session_match": session_match,
        "readout_units_match": unit_match,
        "audit_pass": int(session_match and unit_match),
    })
    return audit


def _pooled_cov_from_trials(merged: dict, cond: str, lm: float,
                             ridge: float = DEFAULT_RIDGE) -> np.ndarray | None:
    """Return pooled within-orientation sample covariance from cached trial vectors.
    Returns None if insufficient data (< 2 trials in any orientation).
    """
    lk = round(lm, 4)
    trial_arrs: dict[int, np.ndarray] = {}
    for ori in ORIENTATIONS:
        ts = merged.get(cond, {}).get(lk, {}).get(ori, [])
        if ts:
            trial_arrs[ori] = np.stack(ts, axis=0).astype(np.float64)
    if not trial_arrs:
        return None
    return _pooled_within_class_cov(trial_arrs)


def _class_means_from_trials(merged: dict, cond: str, lm: float
                              ) -> dict[int, np.ndarray]:
    lk = round(lm, 4)
    g: dict[int, np.ndarray] = {}
    for ori in ORIENTATIONS:
        ts = merged.get(cond, {}).get(lk, {}).get(ori, [])
        if ts:
            g[ori] = np.stack(ts, axis=0).astype(np.float64).mean(axis=0)
    return g


def _predicted_acc_pairwise(
    g_a: np.ndarray, g_b: np.ndarray, sigma_tot: np.ndarray
) -> float:
    """2AFC predicted accuracy for one pair under Gaussian noise: Φ(d′/2)."""
    return _dprime_to_acc(_dprime_pair(g_a, g_b, sigma_tot))


def step_observer_calibration(
    args: argparse.Namespace,
    merged: dict,
    r_bar: np.ndarray,
    out_dir: Path,
) -> tuple[str, float, list[dict]]:
    """Calibration gate for absolute D1 scale using observers A/B/C."""
    print("[Step 0.5] Observer calibration gate ...")
    W = args.primary_window
    sigma_int = _sigma_int_poisson(r_bar, W)
    d1_paths = _resolve_d1_sweep_paths(args)

    audit = _load_mcfarland_readout_audit(
        Path(args.mcfarland_outputs),
        args.expected_session,
        args.expected_readout_units,
    )
    print(
        "  Source audit: "
        f"session={audit['session_found']} (expected={audit['session_expected']})  "
        f"readout_units={audit['readout_units_found']} (expected={audit['readout_units_expected']})"
    )

    d1_abs = _load_d1_absolute(
        d1_paths, args.logmar_grid,
        args.feature_repr, args.d1_sweep_readout, W,
    )

    # Collect per-(logmar, pair, cond) tuples for fitting and evaluation.
    records: list[dict] = []
    pairwise_rows: list[dict] = []

    for lm in args.logmar_grid:
        if abs(lm - args.render_limit) < 1e-4:
            continue
        lk = round(lm, 4)
        lm_d1 = d1_abs.get(lk, {})
        if not lm_d1:
            continue

        g_real = _class_means_from_trials(merged, "real", lm)
        g_stab = _class_means_from_trials(merged, "stabilized", lm)
        sp_real = _pooled_cov_from_trials(merged, "real", lm)
        sp_stab = _pooled_cov_from_trials(merged, "stabilized", lm)
        if sp_real is None or sp_stab is None:
            continue

        for a, b in PAIRS:
            pair_lbl = _pair_label(a, b)
            if pair_lbl not in lm_d1:
                continue
            for cond, g_bars, sp in [
                ("real", g_real, sp_real),
                ("stabilized", g_stab, sp_stab),
            ]:
                if a not in g_bars or b not in g_bars or sp is None:
                    continue
                d1_acc = lm_d1[pair_lbl].get(cond, float("nan"))
                if not np.isfinite(d1_acc):
                    continue
                diff = g_bars[a] - g_bars[b]
                rec = {
                    "logmar": lm, "pair": pair_lbl, "condition": cond,
                    "diff": diff, "sp": sp,
                    "d1_acc": d1_acc,
                }
                records.append(rec)
                pairwise_rows.append(rec)

    if not records:
        print("  WARNING: no records for calibration — skipping gate")
        _write_csv(out_dir / "observer_calibration_input_audit.csv", [audit])
        return "dprime_observer_miscalibrated", 1.0, [
            {
                "test": "observer_calibration",
                "statistic": "calibration_records",
                "value": 0,
                "note": "no_records",
            }
        ]

    n_rec = len(records)
    print(f"  Calibration records: {n_rec}")

    def obs_a_acc(rec: dict) -> float:
        st = sigma_int + rec["sp"] + 1e-9 * np.eye(rec["sp"].shape[0])
        return _dprime_to_acc(_dprime_pair(np.zeros(len(rec["diff"])), rec["diff"], st))

    def obs_b_acc(rec: dict) -> float:
        sp = rec["sp"] + 1e-9 * np.eye(rec["sp"].shape[0])
        return _dprime_to_acc(_dprime_pair(np.zeros(len(rec["diff"])), rec["diff"], sp))

    def obs_c_acc(rec: dict, lam: float) -> float:
        st = lam * sigma_int + rec["sp"] + 1e-9 * np.eye(rec["sp"].shape[0])
        return _dprime_to_acc(_dprime_pair(np.zeros(len(rec["diff"])), rec["diff"], st))

    def mse_lambda(lam: float, fit_recs: list[dict]) -> float:
        errs = [(obs_c_acc(r, lam) - r["d1_acc"]) ** 2 for r in fit_recs]
        return float(np.mean(errs)) if errs else float("inf")

    lambdas = np.logspace(-4, 6, 200)

    def fit_lambda(recs: list[dict]) -> float:
        if not recs:
            return 1.0
        losses = [mse_lambda(lam, recs) for lam in lambdas]
        return float(lambdas[int(np.argmin(losses))])

    lm_values = sorted(set(float(r["logmar"]) for r in records))
    global_lambda = fit_lambda(records)

    # Either fit at one anchor LogMAR or by leave-one-LogMAR-out.
    if str(args.calibration_mode).lower() == "anchor":
        anchor = float(args.calibration_anchor_logmar)
        fit_recs = [r for r in records if abs(float(r["logmar"]) - anchor) < 0.005]
        best_lambda = fit_lambda(fit_recs if fit_recs else records)
        lambda_by_lm = {lm: best_lambda for lm in lm_values}
        lambda_strategy = "anchor"
    else:
        lambda_by_lm = {}
        for lm in lm_values:
            others = [r for r in records if abs(float(r["logmar"]) - lm) > 1e-4]
            lambda_by_lm[lm] = fit_lambda(others if others else records)
        best_lambda = global_lambda
        lambda_strategy = "loo"

    gate_rows: list[dict] = []
    mae_acc = {"A": [], "B": [], "C": []}
    pair_keys = sorted(set((float(r["logmar"]), str(r["pair"])) for r in pairwise_rows))
    for lm, pair in pair_keys:
        rec_real = next((r for r in pairwise_rows
                         if abs(float(r["logmar"]) - lm) < 1e-4
                         and str(r["pair"]) == pair
                         and str(r["condition"]) == "real"), None)
        rec_stab = next((r for r in pairwise_rows
                         if abs(float(r["logmar"]) - lm) < 1e-4
                         and str(r["pair"]) == pair
                         and str(r["condition"]) == "stabilized"), None)
        if rec_real is None or rec_stab is None:
            continue

        lam_eval = float(lambda_by_lm.get(lm, best_lambda))
        d1_real = float(rec_real["d1_acc"])
        d1_stab = float(rec_stab["d1_acc"])
        a_real = float(obs_a_acc(rec_real))
        a_stab = float(obs_a_acc(rec_stab))
        b_real = float(obs_b_acc(rec_real))
        b_stab = float(obs_b_acc(rec_stab))
        c_real = float(obs_c_acc(rec_real, lam_eval))
        c_stab = float(obs_c_acc(rec_stab, lam_eval))

        mae_acc["A"].extend([abs(a_real - d1_real), abs(a_stab - d1_stab)])
        mae_acc["B"].extend([abs(b_real - d1_real), abs(b_stab - d1_stab)])
        mae_acc["C"].extend([abs(c_real - d1_real), abs(c_stab - d1_stab)])

        gate_rows.append({
            "logmar": _fmt_lm(lm),
            "orientation_pair": pair,
            "d1_accuracy_real": d1_real,
            "d1_accuracy_stabilized": d1_stab,
            "acc_geom_real_A": a_real,
            "acc_geom_stabilized_A": a_stab,
            "acc_geom_real_B": b_real,
            "acc_geom_stabilized_B": b_stab,
            "acc_geom_real_C": c_real,
            "acc_geom_stabilized_C": c_stab,
            "delta_d1_real_minus_stabilized": d1_real - d1_stab,
            "delta_A_real_minus_stabilized": a_real - a_stab,
            "delta_B_real_minus_stabilized": b_real - b_stab,
            "delta_C_real_minus_stabilized": c_real - c_stab,
            "calibration_error_A": 0.5 * (abs(a_real - d1_real) + abs(a_stab - d1_stab)),
            "calibration_error_B": 0.5 * (abs(b_real - d1_real) + abs(b_stab - d1_stab)),
            "calibration_error_C": 0.5 * (abs(c_real - d1_real) + abs(c_stab - d1_stab)),
            "calibration_error_delta_A": abs((a_real - a_stab) - (d1_real - d1_stab)),
            "calibration_error_delta_B": abs((b_real - b_stab) - (d1_real - d1_stab)),
            "calibration_error_delta_C": abs((c_real - c_stab) - (d1_real - d1_stab)),
            "lambda_strategy": lambda_strategy,
            "lambda_value": lam_eval,
            "feature_representation": str(args.feature_repr),
            "integration_window": int(W),
        })

    mae_A = float(np.mean(mae_acc["A"])) if mae_acc["A"] else float("nan")
    mae_B = float(np.mean(mae_acc["B"])) if mae_acc["B"] else float("nan")
    mae_C = float(np.mean(mae_acc["C"])) if mae_acc["C"] else float("nan")
    calib_threshold = float(args.calibration_max_mae)

    if np.isfinite(mae_A) and mae_A <= calib_threshold:
        calib_label = "observer_A_calibrated"
        best_obs = "A"
    elif np.isfinite(mae_C) and mae_C <= calib_threshold:
        calib_label = "observer_C_calibrated"
        best_obs = "C"
    elif np.isfinite(mae_B) and mae_B <= calib_threshold:
        calib_label = "observer_B_calibrated_position_only"
        best_obs = "B"
    else:
        calib_label = "dprime_observer_miscalibrated"
        best_obs = "none"

    print(f"  MAE observer A: {mae_A:.4f}")
    print(f"  MAE observer B: {mae_B:.4f}")
    print(f"  MAE observer C: {mae_C:.4f}  (global λ={global_lambda:.4g})")
    print(f"  Threshold: {calib_threshold:.4f}  -> label: {calib_label}")

    _write_csv(out_dir / "observer_calibration_input_audit.csv", [audit])
    _write_csv(out_dir / "observer_calibration_gate.csv", gate_rows)

    summary_row = {
        "logmar": "---SUMMARY---",
        "orientation_pair": "all",
        "d1_accuracy_real": float("nan"),
        "d1_accuracy_stabilized": float("nan"),
        "acc_geom_real_A": float("nan"),
        "acc_geom_stabilized_A": float("nan"),
        "acc_geom_real_B": float("nan"),
        "acc_geom_stabilized_B": float("nan"),
        "acc_geom_real_C": float("nan"),
        "acc_geom_stabilized_C": float("nan"),
        "delta_d1_real_minus_stabilized": float("nan"),
        "delta_A_real_minus_stabilized": float("nan"),
        "delta_B_real_minus_stabilized": float("nan"),
        "delta_C_real_minus_stabilized": float("nan"),
        "calibration_error_A": mae_A,
        "calibration_error_B": mae_B,
        "calibration_error_C": mae_C,
        "calibration_error_delta_A": float("nan"),
        "calibration_error_delta_B": float("nan"),
        "calibration_error_delta_C": float("nan"),
        "lambda_strategy": lambda_strategy,
        "lambda_value": best_lambda,
        "feature_representation": str(args.feature_repr),
        "integration_window": int(W),
    }
    _write_csv(out_dir / "observer_calibration_gate_summary.csv", [summary_row])

    if bool(args.enforce_calibration_gate) and calib_label == "dprime_observer_miscalibrated":
        raise RuntimeError(
            "Observer calibration gate failed: no observer meets calibration threshold. "
            "See observer_calibration_gate.csv for details."
        )

    test_rows: list[dict] = [
        {
            "test": "observer_calibration",
            "statistic": "MAE_obs_A",
            "value": mae_A,
            "note": "Sigma_int_plus_Sigma_pos",
        },
        {
            "test": "observer_calibration",
            "statistic": "MAE_obs_B",
            "value": mae_B,
            "note": "Sigma_pos_only",
        },
        {
            "test": "observer_calibration",
            "statistic": "MAE_obs_C",
            "value": mae_C,
            "note": f"lambda={best_lambda:.4g}",
        },
        {
            "test": "observer_calibration",
            "statistic": "calib_threshold",
            "value": calib_threshold,
            "note": "max_allowed_mae",
        },
        {
            "test": "observer_calibration",
            "statistic": "calibration_records",
            "value": len(gate_rows),
            "note": "logmar_pair_rows",
        },
        {
            "test": "observer_calibration",
            "statistic": "audit_pass",
            "value": float(audit.get("audit_pass", 0)),
            "note": "session_and_readout_match",
        },
        {
            "test": "observer_calibration",
            "statistic": "calib_label",
            "value": float("nan"),
            "note": calib_label,
        },
        {
            "test": "observer_calibration",
            "statistic": "best_observer",
            "value": float("nan"),
            "note": best_obs,
        },
    ]

    return calib_label, best_lambda, test_rows


def _dprime_pair(g_a: np.ndarray, g_b: np.ndarray,
                 sigma_tot: np.ndarray) -> float:
    """d′ = ||ḡ_a - ḡ_b|| / sqrt(û^T Σ_tot û)

    where û = (ḡ_a - ḡ_b) / ||ḡ_a - ḡ_b||.

    Equivalently: ||diff|| / sqrt(diff^T Σ_tot diff / ||diff||²)
                = ||diff||² / sqrt(diff^T Σ_tot diff)

    Code is correct; this docstring shows the full derivation.
    """
    diff = g_a - g_b
    norm_sq = float(np.dot(diff, diff))
    if norm_sq < EPS:
        return 0.0
    var_proj = float(diff @ sigma_tot @ diff)
    if var_proj <= 0:
        return float("nan")
    return float(norm_sq / math.sqrt(var_proj))


def _dprime_to_acc(d_prime: float) -> float:
    """Convert pairwise d′ to 2AFC accuracy under Gaussian noise: Φ(d′/2)."""
    if not np.isfinite(d_prime):
        return float("nan")
    return float(scipy_stats.norm.cdf(d_prime / 2.0))


def compute_arm_a(
    merged: dict, merged_tidx: dict,
    lm: float, noise_model: str,
    r_bar: np.ndarray, n_frames: int,
    n_mc: int, rng: np.random.Generator,
) -> dict:
    """Compute all Arm-A geometry quantities for one logmar.

    Returns dict with keys:
      g_bar_{real,stab,center}   : dict[int, np.ndarray]
      sigma_pos_{real,stab,center}: np.ndarray
      sigma_int, sigma_tot_{real,stab,center}: np.ndarray
      dprime_pair_{real,stab,center}: dict[tuple, float]
      acc_geom_{real,stab,center}: float
      delta_dprime_pair, delta_acc_geom: per-pair and mean contrasts
    """
    lk = round(lm, 4)
    N = len(r_bar)

    if noise_model == "poisson":
        sigma_int = _sigma_int_poisson(r_bar, n_frames)
    else:
        sigma_int = _sigma_int_isotropic(r_bar, n_frames)

    results: dict[str, Any] = {"noise_model": noise_model, "sigma_int": sigma_int}

    for cond in ("real", "stabilized", "fixed_center"):
        cond_rvec = merged.get(cond, {}).get(lk, {})
        # Class conditional means
        g_bars: dict[int, np.ndarray] = {}
        trial_arrs: dict[int, np.ndarray] = {}
        for ori in ORIENTATIONS:
            ts = cond_rvec.get(ori, [])
            if ts:
                arr = np.stack(ts, axis=0).astype(np.float64)
                g_bars[ori] = arr.mean(axis=0)
                trial_arrs[ori] = arr

        if not g_bars:
            results[f"g_bar_{cond}"] = {}
            results[f"sigma_pos_{cond}"] = np.zeros((N, N))
            results[f"sigma_tot_{cond}"] = sigma_int.copy()
            results[f"dprime_pair_{cond}"] = {}
            results[f"acc_geom_{cond}"] = float("nan")
            results[f"acc_geom_pair_{cond}"] = {}
            continue

        # Positional covariance — pooled within-orientation empirical covariance.
        # Computed for ALL conditions including stabilized: stabilized trials use
        # each trial's own mean-gaze position (not a single fixed point), so
        # across-trial variance is real and should NOT be zeroed.
        sigma_pos = _pooled_within_class_cov(trial_arrs)

        sigma_tot = sigma_int + sigma_pos

        # Pairwise d′ and predicted pairwise accuracy Φ(d′/2)
        dp_pairs: dict[tuple[int, int], float] = {}
        acc_pairs: dict[tuple[int, int], float] = {}
        for a, b in PAIRS:
            if a in g_bars and b in g_bars:
                dp = _dprime_pair(g_bars[a], g_bars[b], sigma_tot)
                dp_pairs[(a, b)] = dp
                acc_pairs[(a, b)] = _dprime_to_acc(dp)

        # 4-way ideal-observer accuracy (secondary; mean-pairwise is primary for matching D1)
        if len(g_bars) == 4:
            acc_4way = _ncm_accuracy(g_bars, sigma_tot, n_mc, rng)
        else:
            acc_4way = float("nan")

        results[f"g_bar_{cond}"] = g_bars
        results[f"sigma_pos_{cond}"] = sigma_pos
        results[f"sigma_tot_{cond}"] = sigma_tot
        results[f"dprime_pair_{cond}"] = dp_pairs
        results[f"acc_geom_pair_{cond}"] = acc_pairs
        results[f"acc_geom_{cond}"] = acc_4way

    # Σ_pos diagnostic: flag if positional variance is negligible vs intrinsic noise
    sp_real = results.get("sigma_pos_real", np.zeros((N, N)))
    sp_stab = results.get("sigma_pos_stabilized", np.zeros((N, N)))
    snr_real = float(np.trace(sp_real)) / max(float(np.trace(sigma_int)), EPS)
    snr_stab = float(np.trace(sp_stab)) / max(float(np.trace(sigma_int)), EPS)
    results["sigma_pos_snr_real"] = snr_real
    results["sigma_pos_snr_stab"] = snr_stab
    results["sigma_pos_trace_real"] = float(np.trace(sp_real))
    results["sigma_pos_trace_stab"] = float(np.trace(sp_stab))
    results["sigma_int_trace"] = float(np.trace(sigma_int))

    # Contrasts: real - stabilized
    # PRIMARY observable: mean-pairwise predicted accuracy delta (unit-matches D1)
    acc_pair_real = results.get("acc_geom_pair_real", {})
    acc_pair_stab = results.get("acc_geom_pair_stabilized", {})
    delta_acc_pair: dict[tuple[int, int], float] = {}
    for p in PAIRS:
        ar = acc_pair_real.get(p, float("nan"))
        as_ = acc_pair_stab.get(p, float("nan"))
        delta_acc_pair[p] = ar - as_ if np.isfinite(ar) and np.isfinite(as_) else float("nan")
    results["delta_acc_geom_pair"] = delta_acc_pair

    finite_da = [v for v in delta_acc_pair.values() if np.isfinite(v)]
    results["delta_acc_geom_mean_pairwise"] = (
        float(np.mean(finite_da)) if finite_da else float("nan"))

    # Also keep raw d' contrast and 4-way acc contrast
    dp_real = results.get("dprime_pair_real", {})
    dp_stab = results.get("dprime_pair_stabilized", {})
    delta_dp: dict[tuple[int, int], float] = {}
    for p in PAIRS:
        r, s = dp_real.get(p, float("nan")), dp_stab.get(p, float("nan"))
        delta_dp[p] = r - s if np.isfinite(r) and np.isfinite(s) else float("nan")
    results["delta_dprime_pair"] = delta_dp

    acc_real_4way = results.get("acc_geom_real", float("nan"))
    acc_stab_4way = results.get("acc_geom_stabilized", float("nan"))
    results["delta_acc_geom_4way"] = (
        acc_real_4way - acc_stab_4way
        if np.isfinite(acc_real_4way) and np.isfinite(acc_stab_4way)
        else float("nan"))

    # S controls: difficulty at single fixed positions (Σ_pos = 0)
    for ctrl_cond, ctrl_key in [("stabilized", "S_stab"),
                                  ("fixed_center", "S_center")]:
        g = results.get(f"g_bar_{ctrl_cond}", {})
        si = sigma_int   # Σ_tot = Σ_int only (single position)
        dp_ctrl: dict[tuple[int, int], float] = {}
        for a, b in PAIRS:
            if a in g and b in g:
                dp_ctrl[(a, b)] = _dprime_pair(g[a], g[b], si)
        results[f"dprime_pair_{ctrl_key}"] = dp_ctrl
        if len(g) == 4:
            results[f"acc_geom_{ctrl_key}"] = _ncm_accuracy(g, si, n_mc, rng)
        else:
            results[f"acc_geom_{ctrl_key}"] = float("nan")

    # S_cloud_mean: per-position d′, averaged over cloud (matched by trial_index)
    lk2 = round(lm, 4)
    real_rvec = merged.get("real", {}).get(lk2, {})
    real_tidx = merged_tidx.get("real", {}).get(lk2, {})
    ti_resp: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for ori in ORIENTATIONS:
        for tidx, resp in zip(real_tidx.get(ori, []), real_rvec.get(ori, [])):
            ti_resp[int(tidx)][ori] = np.asarray(resp, dtype=np.float64)
    per_pos_dp: dict[tuple[int, int], list[float]] = {p: [] for p in PAIRS}
    per_pos_acc: list[float] = []
    for resp_by_ori in ti_resp.values():
        if len(resp_by_ori) < 4:
            continue
        # Each position: Σ_tot = Σ_int (no additional positional spread at single point)
        for a, b in PAIRS:
            if a in resp_by_ori and b in resp_by_ori:
                per_pos_dp[(a, b)].append(
                    _dprime_pair(resp_by_ori[a], resp_by_ori[b], sigma_int))
    results["dprime_pair_S_cloud_mean"] = {
        p: float(np.nanmean(v)) if v else float("nan")
        for p, v in per_pos_dp.items()
    }
    # S_best per pair: max of the three single-position controls
    results["dprime_pair_S_best"] = {
        p: float(np.nanmax([
            results["dprime_pair_S_stab"].get(p, float("nan")),
            results["dprime_pair_S_center"].get(p, float("nan")),
            results["dprime_pair_S_cloud_mean"].get(p, float("nan")),
        ]))
        for p in PAIRS
    }
    s_best_vals = [v for v in results["dprime_pair_S_best"].values() if np.isfinite(v)]
    results["S_best_mean"] = float(np.mean(s_best_vals)) if s_best_vals else float("nan")

    return results


def step1_arm_a(
    args: argparse.Namespace,
    merged: dict, merged_tidx: dict,
    r_bar: np.ndarray,
    out_dir: Path, rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict]:
    """Compute Arm A for all logmars, both noise models."""
    print("[Step 1] Arm A: finite-cloud d′_geom ...")
    curve_rows: list[dict] = []
    pair_rows: list[dict] = []
    arm_a_by_lm: dict[float, dict[str, dict]] = {}  # lm → {noise_model: results}

    for lm in args.logmar_grid:
        is_ctrl = abs(lm - args.render_limit) < 1e-4
        arm_a_by_lm[lm] = {}
        for nm in ("poisson", "isotropic"):
            res = compute_arm_a(merged, merged_tidx, lm, nm,
                                r_bar, args.primary_window, args.n_mc, rng)
            arm_a_by_lm[lm][nm] = res

            # Primary row (poisson)
            if nm == "poisson":
                lk = round(lm, 4)
                n_real = sum(len(merged.get("real", {}).get(lk, {}).get(o, []))
                             for o in ORIENTATIONS)

                curve_rows.append({
                    "L": _fmt_lm(lm),
                    "render_ctrl": "yes" if is_ctrl else "no",
                    "noise_model": nm,
                    # PRIMARY geometry observable — mean pairwise Φ(d'/2) delta (unit-matches D1)
                    "delta_acc_geom_mean_pairwise": _f(res.get("delta_acc_geom_mean_pairwise")),
                    # Secondary: true 4-way ideal-observer accuracy delta
                    "delta_acc_geom_4way": _f(res.get("delta_acc_geom_4way")),
                    "acc_geom_real_4way": _f(res.get("acc_geom_real")),
                    "acc_geom_stab_4way": _f(res.get("acc_geom_stabilized")),
                    # Difficulty controls at single positions (Σ_pos=0)
                    "acc_geom_S_center": _f(res.get("acc_geom_S_center")),
                    "acc_geom_S_stab": _f(res.get("acc_geom_S_stab")),
                    "S_best_mean_dprime": _f(res.get("S_best_mean")),
                    # Σ_pos diagnostics
                    "sigma_pos_trace_real": _f(res.get("sigma_pos_trace_real")),
                    "sigma_pos_trace_stab": _f(res.get("sigma_pos_trace_stab")),
                    "sigma_int_trace": _f(res.get("sigma_int_trace")),
                    "sigma_pos_snr_real": _f(res.get("sigma_pos_snr_real")),
                    "sigma_pos_snr_stab": _f(res.get("sigma_pos_snr_stab")),
                    "n_trials_real": n_real,
                })
                for a, b in PAIRS:
                    p = (a, b)
                    pair_rows.append({
                        "L": _fmt_lm(lm),
                        "render_ctrl": "yes" if is_ctrl else "no",
                        "noise_model": nm,
                        "pair": _pair_label(a, b),
                        # Pairwise geometry in accuracy units (Φ(d'/2)) — unit-matches D1 pair delta
                        "acc_geom_pair_real": _f(res.get("acc_geom_pair_real", {}).get(p)),
                        "acc_geom_pair_stab": _f(res.get("acc_geom_pair_stabilized", {}).get(p)),
                        "delta_acc_geom_pair": _f(res.get("delta_acc_geom_pair", {}).get(p)),
                        # Raw d' for reference
                        "dprime_real": _f(res.get("dprime_pair_real", {}).get(p)),
                        "dprime_stab": _f(res.get("dprime_pair_stabilized", {}).get(p)),
                        "delta_dprime": _f(res.get("delta_dprime_pair", {}).get(p)),
                        # S controls (d' units)
                        "dprime_S_center": _f(res.get("dprime_pair_S_center", {}).get(p)),
                        "dprime_S_stab": _f(res.get("dprime_pair_S_stab", {}).get(p)),
                        "dprime_S_cloud_mean": _f(res.get("dprime_pair_S_cloud_mean", {}).get(p)),
                        "dprime_S_best": _f(res.get("dprime_pair_S_best", {}).get(p)),
                    })

    _write_csv(out_dir / "finite_cloud_dprime_geometry.csv", curve_rows)
    _write_csv(out_dir / "finite_cloud_dprime_geometry_pairwise.csv", pair_rows)
    print(f"  Saved {len(curve_rows)} curve rows, {len(pair_rows)} pairwise rows.")
    return curve_rows, pair_rows, arm_a_by_lm


# ---------------------------------------------------------------------------
# Step 2 — Function curves
# ---------------------------------------------------------------------------

def step2_function_curves(
    args: argparse.Namespace,
    acc_data: dict, confirmed_d1: str | None,
    merged: dict, out_dir: Path, rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    print("[Step 2] Function curves (D1 time-mean sweep)...")
    d1_paths = _resolve_d1_sweep_paths(args)
    if confirmed_d1 is None:
        raise RuntimeError(
            f"D1 sweep not found at '{_d1_source_label(d1_paths)}'. "
            "Run --cache-audit-only to inspect."
        )

    func_rows: list[dict] = []
    func_pair_rows: list[dict] = []

    for lm in args.logmar_grid:
        lk = round(lm, 4)
        is_ctrl = abs(lm - args.render_limit) < 1e-4
        lm_acc = acc_data.get(lk, {})

        for w in args.windows:
            # Per-pair delta at this window
            delta_by_pair: dict[tuple[int, int], float] = {}
            for a, b in PAIRS:
                lbl = _pair_label(a, b)
                delta_by_pair[(a, b)] = _f(lm_acc.get(lbl, {}).get(w, float("nan")))

            finite_d = [d for d in delta_by_pair.values() if np.isfinite(d)]
            delta_mean = float(np.nanmean(finite_d)) if finite_d else float("nan")

            # Bootstrap CI over the 6 pairwise deltas
            if len(finite_d) >= 2:
                bs = [float(np.mean(rng.choice(finite_d, size=len(finite_d), replace=True)))
                      for _ in range(args.n_bootstrap)]
                ci_lo, ci_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                ci_type = f"pair_resampling_w{w}"
            else:
                ci_lo = ci_hi = float("nan")
                ci_type = "insufficient"

            n_trials = sum(len(merged.get("real", {}).get(lk, {}).get(o, []))
                           for o in ORIENTATIONS)

            func_rows.append({
                "L": _fmt_lm(lm),
                "render_ctrl": "yes" if is_ctrl else "no",
                "window": w,
                "d1_col": args.d1_sweep_col,
                "source": _d1_source_label(d1_paths),
                "function_metric": "mean_pairwise_proxy_4way",
                "readout_type": args.d1_sweep_readout,
                "delta_acc_mean": delta_mean,
                "delta_ci_low": ci_lo,
                "delta_ci_high": ci_hi,
                "ci_type": ci_type,
                "n_pairs": len(finite_d),
                "n_trials_real": n_trials,
            })
            for a, b in PAIRS:
                func_pair_rows.append({
                    "L": _fmt_lm(lm), "render_ctrl": "yes" if is_ctrl else "no",
                    "window": w, "pair": _pair_label(a, b),
                    "d1_col": args.d1_sweep_col,
                    "function_metric": "pairwise_d1_delta",
                    "delta_acc_pair": delta_by_pair.get((a, b), float("nan")),
                })

    _write_csv(out_dir / "function_curve.csv", func_rows)
    _write_csv(out_dir / "function_pairwise.csv", func_pair_rows)
    print(f"  D1 source: {_d1_source_label(d1_paths)} (col: {args.d1_sweep_col})")
    print(f"  NOTE: '4-way' D1 = mean of 6 pairwise deltas (proxy). True 4-class accuracy not in cache.")
    print(f"  Saved {len(func_rows)} rows, {len(func_pair_rows)} pairwise rows.")
    return func_rows, func_pair_rows


def _build_cached_vector_dataset(
    merged: dict,
    merged_tidx: dict,
    cond: str,
    lm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lk = round(lm, 4)
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    g_rows: list[int] = []
    for class_idx, ori in enumerate(ORIENTATIONS):
        vecs = merged.get(cond, {}).get(lk, {}).get(ori, [])
        tids = merged_tidx.get(cond, {}).get(lk, {}).get(ori, [])
        for vec, tid in zip(vecs, tids):
            X_rows.append(np.asarray(vec, dtype=np.float64))
            y_rows.append(class_idx)
            g_rows.append(int(tid))
    if not X_rows:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=int), np.empty(0, dtype=int)
    return (
        np.stack(X_rows, axis=0).astype(np.float64),
        np.asarray(y_rows, dtype=int),
        np.asarray(g_rows, dtype=int),
    )


def _grouped_multiclass_accuracy(
    X_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    model_name: str,
    rng_seed: int,
) -> tuple[float, float, int, str]:
    if X_all.size == 0 or y_all.size == 0:
        return float("nan"), float("nan"), 0, "no_samples"

    unique_groups = np.unique(groups_all)
    unique_classes = np.unique(y_all)
    n_splits = min(5, unique_groups.size)
    if n_splits < 2 or unique_classes.size < len(ORIENTATIONS):
        return float("nan"), float("nan"), int(n_splits), "insufficient_groups_or_classes"

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    splitter = GroupKFold(n_splits=n_splits)
    fold_accs: list[float] = []
    note = "ok"
    for train_idx, test_idx in splitter.split(X_all, y_all, groups=groups_all):
        X_tr = X_all[train_idx]
        X_te = X_all[test_idx]
        y_tr = y_all[train_idx]
        y_te = y_all[test_idx]

        if model_name == "logistic":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                random_state=rng_seed,
            )
        elif model_name == "lda_empirical":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None)
        elif model_name == "lda_shrinkage":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        else:
            return float("nan"), float("nan"), int(n_splits), f"unknown_model_{model_name}"

        try:
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            fold_accs.append(float(np.mean(pred == y_te)))
        except Exception as exc:
            note = f"fit_failed:{type(exc).__name__}"
            return float("nan"), float("nan"), int(n_splits), note

    if not fold_accs:
        return float("nan"), float("nan"), int(n_splits), "no_folds"
    acc_arr = np.asarray(fold_accs, dtype=float)
    return float(acc_arr.mean()), float(acc_arr.std(ddof=0)), int(n_splits), note


def step2b_readout_validation(
    args: argparse.Namespace,
    merged: dict,
    merged_tidx: dict,
    geom_rows: list[dict],
    out_dir: Path,
) -> tuple[list[dict], dict[str, Any]]:
    print("[Step 2b] Cached-vector readout validation...")
    import pandas as pd

    d1_abs = _load_d1_absolute(
        _resolve_d1_sweep_paths(args),
        args.logmar_grid,
        args.feature_repr,
        args.d1_sweep_readout,
        args.primary_window,
    )
    gby = {float(r["L"]): r for r in geom_rows if r.get("noise_model") == "poisson"}

    rows: list[dict] = []
    summary_rows: list[dict] = []
    core_logmars = [
        float(lm) for lm in args.logmar_grid
        if (args.core_range[0] - 1e-4) <= lm <= (args.core_range[1] + 1e-4)
        and abs(lm - args.render_limit) > 1e-4
    ]
    model_names = ("logistic", "lda_empirical", "lda_shrinkage")

    for lm in core_logmars:
        lk = round(lm, 4)
        pair_map = d1_abs.get(lk, {})
        d1_cond = {
            cond: float(np.mean([
                _f(pair_map.get(_pair_label(a, b), {}).get(cond, float("nan")))
                for a, b in PAIRS
                if np.isfinite(_f(pair_map.get(_pair_label(a, b), {}).get(cond, float("nan"))))
            ])) if any(
                np.isfinite(_f(pair_map.get(_pair_label(a, b), {}).get(cond, float("nan"))))
                for a, b in PAIRS
            ) else float("nan")
            for cond in ("real", "stabilized")
        }
        geom_row = gby.get(lm, {})

        model_accs: dict[str, dict[str, float | str | int]] = {}
        for cond in ("real", "stabilized"):
            X_all, y_all, groups_all = _build_cached_vector_dataset(merged, merged_tidx, cond, lm)
            model_accs[cond] = {}
            for model_name in model_names:
                acc, acc_std, n_splits, note = _grouped_multiclass_accuracy(
                    X_all, y_all, groups_all, model_name, args.random_seed
                )
                model_accs[cond][f"{model_name}_acc"] = acc
                model_accs[cond][f"{model_name}_std"] = acc_std
                model_accs[cond][f"{model_name}_splits"] = n_splits
                model_accs[cond][f"{model_name}_note"] = note

            rows.append({
                "logmar": _fmt_lm(lm),
                "row_type": "condition",
                "condition": cond,
                "d1_accuracy": d1_cond.get(cond, float("nan")),
                "logistic_accuracy": model_accs[cond]["logistic_acc"],
                "lda_accuracy": model_accs[cond]["lda_empirical_acc"],
                "lda_shrinkage_accuracy": model_accs[cond]["lda_shrinkage_acc"],
                "armA_acc_geom": _f(geom_row.get("acc_geom_real_4way" if cond == "real" else "acc_geom_stab_4way")),
                "logistic_std": model_accs[cond]["logistic_std"],
                "lda_std": model_accs[cond]["lda_empirical_std"],
                "lda_shrinkage_std": model_accs[cond]["lda_shrinkage_std"],
                "logistic_note": model_accs[cond]["logistic_note"],
                "lda_note": model_accs[cond]["lda_empirical_note"],
                "lda_shrinkage_note": model_accs[cond]["lda_shrinkage_note"],
            })

        delta_row = {
            "logmar": _fmt_lm(lm),
            "row_type": "delta",
            "condition": "real_minus_stabilized",
            "d1_accuracy": d1_cond.get("real", float("nan")) - d1_cond.get("stabilized", float("nan")) if np.isfinite(d1_cond.get("real", float("nan"))) and np.isfinite(d1_cond.get("stabilized", float("nan"))) else float("nan"),
            "logistic_accuracy": _f(model_accs["real"].get("logistic_acc")) - _f(model_accs["stabilized"].get("logistic_acc")) if np.isfinite(_f(model_accs["real"].get("logistic_acc"))) and np.isfinite(_f(model_accs["stabilized"].get("logistic_acc"))) else float("nan"),
            "lda_accuracy": _f(model_accs["real"].get("lda_empirical_acc")) - _f(model_accs["stabilized"].get("lda_empirical_acc")) if np.isfinite(_f(model_accs["real"].get("lda_empirical_acc"))) and np.isfinite(_f(model_accs["stabilized"].get("lda_empirical_acc"))) else float("nan"),
            "lda_shrinkage_accuracy": _f(model_accs["real"].get("lda_shrinkage_acc")) - _f(model_accs["stabilized"].get("lda_shrinkage_acc")) if np.isfinite(_f(model_accs["real"].get("lda_shrinkage_acc"))) and np.isfinite(_f(model_accs["stabilized"].get("lda_shrinkage_acc"))) else float("nan"),
            "armA_acc_geom": _f(geom_row.get("delta_acc_geom_4way")),
            "logistic_std": float("nan"),
            "lda_std": float("nan"),
            "lda_shrinkage_std": float("nan"),
            "logistic_note": "delta",
            "lda_note": "delta",
            "lda_shrinkage_note": "delta",
        }
        rows.append(delta_row)

    df = pd.DataFrame(rows)
    _write_csv(out_dir / "cached_vector_readout_validation.csv", rows)

    cond_df = df[df["row_type"] == "condition"].copy()
    delta_df = df[df["row_type"] == "delta"].copy()
    model_to_col = {
        "logistic": "logistic_accuracy",
        "lda_empirical": "lda_accuracy",
        "lda_shrinkage": "lda_shrinkage_accuracy",
    }
    chance = 1.0 / len(ORIENTATIONS)
    near_chance_cutoff = chance + 0.10
    abs_mae: dict[str, float] = {}
    delta_mae: dict[str, float] = {}
    mean_acc: dict[str, float] = {}
    for model_name, col in model_to_col.items():
        abs_mae[model_name] = float(np.nanmean(np.abs(cond_df[col] - cond_df["d1_accuracy"])))
        delta_mae[model_name] = float(np.nanmean(np.abs(delta_df[col] - delta_df["d1_accuracy"])))
        mean_acc[model_name] = float(np.nanmean(cond_df[col]))
        summary_rows.append({
            "model": model_name,
            "absolute_accuracy_mae": abs_mae[model_name],
            "delta_accuracy_mae": delta_mae[model_name],
            "mean_accuracy": mean_acc[model_name],
        })

    armA_abs_mae = float(np.nanmean(np.abs(cond_df["armA_acc_geom"] - cond_df["d1_accuracy"])))
    armA_delta_mae = float(np.nanmean(np.abs(delta_df["armA_acc_geom"] - delta_df["d1_accuracy"])))
    summary_rows.append({
        "model": "armA",
        "absolute_accuracy_mae": armA_abs_mae,
        "delta_accuracy_mae": armA_delta_mae,
        "mean_accuracy": float(np.nanmean(cond_df["armA_acc_geom"])),
    })

    best_model = min(model_to_col, key=lambda name: (abs_mae[name], delta_mae[name]))
    best_abs_mae = abs_mae[best_model]
    best_delta_mae = delta_mae[best_model]
    best_mean_acc = mean_acc[best_model]
    if np.isfinite(best_mean_acc) and best_mean_acc <= near_chance_cutoff:
        validation_label = "cached_vector_D1_mismatch"
        note = "Grouped-CV logistic/LDA remain near chance on cached vectors."
    elif np.isfinite(best_abs_mae) and best_abs_mae <= 0.05:
        validation_label = "observer_noise_model_miscalibrated"
        note = "Cached-vector classifiers reproduce D1 substantially better than Arm A observer."
    else:
        validation_label = "readout_validation_indeterminate"
        note = "Cached-vector classifiers are above chance but do not yet match D1 tightly."

    summary_rows.append({
        "model": "decision",
        "absolute_accuracy_mae": best_abs_mae,
        "delta_accuracy_mae": best_delta_mae,
        "mean_accuracy": best_mean_acc,
        "validation_label": validation_label,
        "best_model": best_model,
        "note": note,
    })
    summary_fields = [
        "model",
        "absolute_accuracy_mae",
        "delta_accuracy_mae",
        "mean_accuracy",
        "validation_label",
        "best_model",
        "note",
    ]
    for row in summary_rows:
        for field in summary_fields:
            row.setdefault(field, "")
    _write_csv(out_dir / "cached_vector_readout_validation_summary.csv", summary_rows, summary_fields)

    test_rows = [
        {
            "test": "readout_validation",
            "statistic": f"absolute_mae_{model}",
            "value": abs_mae[model],
            "window": args.primary_window,
            "noise_model": "poisson",
            "note": "condition_accuracy_mae",
        }
        for model in model_to_col
    ]
    test_rows += [
        {
            "test": "readout_validation",
            "statistic": f"delta_mae_{model}",
            "value": delta_mae[model],
            "window": args.primary_window,
            "noise_model": "poisson",
            "note": "real_minus_stabilized_mae",
        }
        for model in model_to_col
    ]
    test_rows += [
        {
            "test": "readout_validation",
            "statistic": "armA_absolute_mae",
            "value": armA_abs_mae,
            "window": args.primary_window,
            "noise_model": "poisson",
            "note": "condition_accuracy_mae",
        },
        {
            "test": "readout_validation",
            "statistic": "armA_delta_mae",
            "value": armA_delta_mae,
            "window": args.primary_window,
            "noise_model": "poisson",
            "note": "real_minus_stabilized_mae",
        },
        {
            "test": "readout_validation",
            "statistic": "best_model",
            "value": float("nan"),
            "window": args.primary_window,
            "noise_model": "poisson",
            "note": best_model,
        },
        {
            "test": "readout_validation",
            "statistic": "validation_label",
            "value": float("nan"),
            "window": args.primary_window,
            "noise_model": "poisson",
            "note": validation_label,
        },
    ]
    print(f"  best_model={best_model} abs_mae={best_abs_mae:.4f} delta_mae={best_delta_mae:.4f} -> {validation_label}")
    return test_rows, {
        "validation_label": validation_label,
        "best_model": best_model,
        "best_abs_mae": best_abs_mae,
        "best_delta_mae": best_delta_mae,
        "best_mean_acc": best_mean_acc,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Array extraction
# ---------------------------------------------------------------------------

def _extract_arrays(
    geom_rows: list[dict],
    func_rows: list[dict],
    window: int,
    render_limit: float,
    core_range: tuple[float, float],
    noise_model: str = "poisson",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ls, dag_primary, dad, S_center, S_stab, S_best, in_core).

    dag_primary: delta_acc_geom_mean_pairwise = Φ(d'/2) contrast, same units as D1.
    dad:         mean-pairwise D1 accuracy delta (function side).
    Both are mean pairwise accuracy deltas → unit-matched.
    """
    gby = {float(r["L"]): r for r in geom_rows if r.get("noise_model") == noise_model}
    fby = {(float(r["L"]), int(r["window"])): r for r in func_rows}

    Ls, dag, dad, s_ctr, s_stab, s_best, ic = [], [], [], [], [], [], []
    for lm in sorted(gby):
        if abs(lm - render_limit) < 1e-4:
            continue
        gr = gby[lm]
        fr = fby.get((lm, window))
        Ls.append(lm)
        # PRIMARY geometry: mean-pairwise Φ(d'/2) contrast
        dag.append(_f(gr.get("delta_acc_geom_mean_pairwise")))
        dad.append(_f(fr.get("delta_acc_mean")) if fr else float("nan"))
        # S controls in d' space (converted via Φ(d'/2) - 0.5 would give comparable units,
        # but for the specificity regression we use raw d' values as the covariate — OK
        # since specificity is about sign/significance, not unit-matching within controls)
        s_ctr.append(_f(gr.get("acc_geom_S_center")))
        s_stab.append(_f(gr.get("acc_geom_S_stab")))
        s_best.append(_f(gr.get("S_best_mean_dprime")))
        ic.append((core_range[0] - 1e-4) <= lm <= (core_range[1] + 1e-4))
    return (np.array(Ls), np.array(dag), np.array(dad),
            np.array(s_ctr), np.array(s_stab), np.array(s_best),
            np.array(ic, dtype=bool))


# ---------------------------------------------------------------------------
# Steps 3–5: Tests
# ---------------------------------------------------------------------------

def step3_coincidence(
    args: argparse.Namespace,
    geom_rows: list[dict], func_rows: list[dict],
    n_core_ready: int,
    rng: np.random.Generator,
) -> tuple[list[dict], bool, bool]:
    print("[Step 3] Coincidence test...")
    rows: list[dict] = []
    w = args.primary_window
    gs = DEFAULT_GRID_STEP

    Ls, dag, dad, _, _, _, ic = _extract_arrays(
        geom_rows, func_rows, w, args.render_limit, args.core_range)

    # Gate: report but stamp underpowered if below min_core_points
    underpowered = n_core_ready < args.min_core_points
    underpowered_note = (f"underpowered_{n_core_ready}_of_{args.min_core_points}_core_pts"
                         if underpowered else "")

    L_geom, gs_status = _zero_crossing_core(Ls, dag, ic)
    L_func, fd_status = _zero_crossing_core(Ls, dad, ic)
    dL = abs(L_geom - L_func) if (np.isfinite(L_geom) and np.isfinite(L_func)) else float("nan")

    # Check render-limit confound
    Ls_all, dag_all, dad_all, _, _, _, _ = _extract_arrays(
        geom_rows, func_rows, w, -999.0, args.core_range)
    L_geom_all, _ = _interp_zero(Ls_all, dag_all)
    render_confounded = (
        not np.isfinite(L_geom) and np.isfinite(L_geom_all) and
        abs(L_geom_all - args.render_limit) < gs
    )

    # Bootstrap dL
    valid_core = ic & np.isfinite(dag) & np.isfinite(dad)
    dL_bs: list[float] = []
    if valid_core.sum() >= 2:
        Lc, gc, dc = Ls[valid_core], dag[valid_core], dad[valid_core]
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, len(gc), size=len(gc))
            uL = sorted(set(Lc[idx].tolist()))
            g_agg = np.array([gc[idx][np.abs(Lc[idx] - l) < 1e-5].mean() for l in uL])
            d_agg = np.array([dc[idx][np.abs(Lc[idx] - l) < 1e-5].mean() for l in uL])
            Lg_b, _ = _interp_zero(np.array(uL), g_agg)
            Ld_b, _ = _interp_zero(np.array(uL), d_agg)
            if np.isfinite(Lg_b) and np.isfinite(Ld_b):
                dL_bs.append(abs(Lg_b - Ld_b))

    dL_ci_lo = float(np.percentile(dL_bs, 2.5)) if len(dL_bs) > 10 else float("nan")
    dL_ci_hi = float(np.percentile(dL_bs, 97.5)) if len(dL_bs) > 10 else float("nan")
    dL_med = float(np.median(dL_bs)) if len(dL_bs) > 10 else dL

    # Criterion: |dL_median| < grid_step AND dL CI contains 0
    coincidence = (
        np.isfinite(dL) and np.isfinite(dL_med) and
        dL_med < gs and
        np.isfinite(dL_ci_lo) and dL_ci_lo <= 0.0 <= dL_ci_hi and
        not render_confounded
    )

    # Flag if geom has no crossing (unexpected for correct d′)
    no_geom_crossing = not np.isfinite(L_geom)

    def _add(name, val, ci_l=float("nan"), ci_h=float("nan"), note=""):
        rows.append({"test": "coincidence", "statistic": name, "value": val,
                     "ci_low": ci_l, "ci_high": ci_h, "window": w,
                     "noise_model": "poisson", "note": note})

    _add("L_geom", L_geom, note=gs_status)
    _add("L_func", L_func, note=fd_status)
    _add("dL", dL, dL_ci_lo, dL_ci_hi)
    _add("dL_median", dL_med)
    _add("render_confounded", int(render_confounded))
    _add("no_geom_crossing", int(no_geom_crossing),
         note="observable_cannot_test_coincidence" if no_geom_crossing else "")
    _add("coincidence_verdict", int(coincidence),
         note=("TRUE" if coincidence else "FALSE") +
              (f" [{underpowered_note}]" if underpowered else ""))

    Lg_s = f"{L_geom:.3f}" if np.isfinite(L_geom) else "nan"
    Ld_s = f"{L_func:.3f}" if np.isfinite(L_func) else "nan"
    dL_s = f"{dL:.3f}" if np.isfinite(dL) else "nan"
    print(f"  L_geom={Lg_s} ({gs_status}), L_func={Ld_s} ({fd_status})")
    print(f"  dL={dL_s}, dL_median={f'{dL_med:.3f}' if np.isfinite(dL_med) else 'nan'}, "
          f"CI=[{f'{dL_ci_lo:.3f}' if np.isfinite(dL_ci_lo) else 'nan'}, "
          f"{f'{dL_ci_hi:.3f}' if np.isfinite(dL_ci_hi) else 'nan'}]")
    if underpowered:
        print(f"  NOTE: underpowered ({underpowered_note}) — coincidence verdict is exploratory")
    print(f"  coincidence={'TRUE' if coincidence else 'FALSE'}")
    if no_geom_crossing:
        print("  WARNING: ΔAcc_geom has no zero-crossing — inspect Σ_pos construction.")
    return rows, coincidence, render_confounded


def step4_continuous(
    args: argparse.Namespace,
    geom_rows: list[dict], func_rows: list[dict],
    n_core_ready: int, rng: np.random.Generator,
) -> tuple[list[dict], bool]:
    print("[Step 4] Continuous regression...")
    rows: list[dict] = []

    if n_core_ready < args.min_core_points:
        note = (f"GATED: only {n_core_ready} core points, need {args.min_core_points}. "
                "inconclusive_underpowered_grid")
        rows.append({"test": "continuous", "statistic": "gated", "value": float("nan"),
                     "ci_low": "nan", "ci_high": "nan", "window": args.primary_window,
                     "noise_model": "poisson", "note": note})
        print(f"  {note}")
        return rows, False

    w = args.primary_window
    Ls, dag, dad, _, _, _, _ = _extract_arrays(
        geom_rows, func_rows, w, args.render_limit, args.core_range)
    valid = np.isfinite(dag) & np.isfinite(dad)
    if valid.sum() < 3:
        print("  WARNING: insufficient valid points.")
        return rows, False

    gv, dv = dag[valid], dad[valid]
    slope, _, r, p_val, _ = scipy_stats.linregress(gv, dv)
    r2 = float(r ** 2)
    rho, _ = scipy_stats.spearmanr(gv, dv)

    slopes_b, r2s_b, rhos_b = [], [], []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, len(gv), size=len(gv))
        if len(set(idx.tolist())) < 2:
            continue
        sl, _, rb, _, _ = scipy_stats.linregress(gv[idx], dv[idx])
        slopes_b.append(sl)
        r2s_b.append(rb ** 2)
        rh, _ = scipy_stats.spearmanr(gv[idx], dv[idx])
        rhos_b.append(float(rh))

    def _ci(arr):
        return ((float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
                if len(arr) >= 10 else (float("nan"), float("nan")))

    sl_ci, r2_ci, rh_ci = _ci(slopes_b), _ci(r2s_b), _ci(rhos_b)
    cont_sig = np.isfinite(slope) and slope > 0 and np.isfinite(sl_ci[0]) and sl_ci[0] > 0

    def _add(name, val, ci_l=float("nan"), ci_h=float("nan"), note=""):
        rows.append({"test": "continuous", "statistic": name, "value": val,
                     "ci_low": ci_l, "ci_high": ci_h, "window": w,
                     "noise_model": "poisson", "note": note})

    _add("slope", slope, sl_ci[0], sl_ci[1])
    _add("R2", r2, r2_ci[0], r2_ci[1])
    _add("spearman_rho", float(rho), rh_ci[0], rh_ci[1])
    _add("p_ols", float(p_val))
    _add("continuous_significant", int(cont_sig), note="TRUE" if cont_sig else "FALSE")

    print(f"  slope={slope:.4f} CI=[{sl_ci[0]:.4f},{sl_ci[1]:.4f}], "
          f"R²={r2:.3f}, ρ={float(rho):.3f}")
    print(f"  continuous_significant={'TRUE' if cont_sig else 'FALSE'}")
    return rows, cont_sig


def _specificity_one_ctrl(gv: np.ndarray, dv: np.ndarray, sv: np.ndarray,
                           ctrl_name: str, window: int, n_bootstrap: int,
                           delta_r2_thresh: float, rng: np.random.Generator,
                           ) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    valid = np.isfinite(gv) & np.isfinite(dv) & np.isfinite(sv)
    if valid.sum() < 4:
        rows.append({"test": "specificity", "statistic": f"passed__{ctrl_name}",
                     "value": 0, "ci_low": "nan", "ci_high": "nan",
                     "window": window, "ctrl": ctrl_name, "noise_model": "poisson",
                     "note": f"insufficient_data ({valid.sum()} pts)"})
        return rows, False

    gv, dv, sv = gv[valid], dv[valid], sv[valid]
    n = len(gv)

    def _resid(y, x):
        sl, ic, _, _, _ = scipy_stats.linregress(x, y)
        return y - (sl * x + ic)

    d_r, g_r = _resid(dv, sv), _resid(gv, sv)
    prho, _ = scipy_stats.spearmanr(d_r, g_r)

    def _r2_aic(y, *Xcols):
        X = np.column_stack([np.ones(n)] + list(Xcols))
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        ss_res = float(np.dot(res, res))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, EPS)
        aic = n * np.log(max(ss_res / n, EPS)) + 2 * X.shape[1]
        return r2, float(aic)

    r2_null, aic_null = _r2_aic(dv, sv)
    r2_full, aic_full = _r2_aic(dv, sv, gv)
    dR2 = r2_full - r2_null if np.isfinite(r2_full) and np.isfinite(r2_null) else float("nan")
    dAIC = aic_full - aic_null if np.isfinite(aic_full) and np.isfinite(aic_null) else float("nan")

    rhos_b = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 3:
            continue
        dr = _resid(dv[idx], sv[idx])
        gr = _resid(gv[idx], sv[idx])
        rh, _ = scipy_stats.spearmanr(dr, gr)
        rhos_b.append(float(rh))
    rh_lo = float(np.percentile(rhos_b, 2.5)) if len(rhos_b) >= 10 else float("nan")
    rh_hi = float(np.percentile(rhos_b, 97.5)) if len(rhos_b) >= 10 else float("nan")

    passed = (np.isfinite(prho) and np.isfinite(rh_lo) and rh_lo > 0.0
              and np.isfinite(dR2) and dR2 >= delta_r2_thresh
              and np.isfinite(dAIC) and dAIC < 0.0)

    for name, val, cl, ch in [
        (f"partial_rho__{ctrl_name}", float(prho), rh_lo, rh_hi),
        (f"delta_R2__{ctrl_name}", dR2, float("nan"), float("nan")),
        (f"delta_AIC__{ctrl_name}", dAIC, float("nan"), float("nan")),
        (f"passed__{ctrl_name}", int(passed), float("nan"), float("nan")),
    ]:
        rows.append({"test": "specificity", "statistic": name, "value": val,
                     "ci_low": cl, "ci_high": ch, "window": window,
                     "ctrl": ctrl_name, "noise_model": "poisson",
                     "note": ("TRUE" if passed else "FALSE") if "passed" in name else ""})
    return rows, passed


def step5_specificity(
    args: argparse.Namespace,
    geom_rows: list[dict], func_rows: list[dict],
    n_core_ready: int, rng: np.random.Generator,
) -> tuple[list[dict], dict[str, bool]]:
    print("[Step 5] Specificity (three controls)...")
    all_rows: list[dict] = []
    passed_by: dict[str, bool] = {}

    if n_core_ready < args.min_core_points:
        note = f"GATED: {n_core_ready} < {args.min_core_points} core points"
        all_rows.append({"test": "specificity", "statistic": "gated", "value": float("nan"),
                         "ci_low": "nan", "ci_high": "nan", "window": args.primary_window,
                         "ctrl": "all", "noise_model": "poisson", "note": note})
        print(f"  {note}")
        return all_rows, {}

    w = args.primary_window
    Ls, dag, dad, s_ctr_arr, s_stab_arr, s_best, _ = _extract_arrays(
        geom_rows, func_rows, w, args.render_limit, args.core_range)

    # Primary difficulty control: S_center (cleanest — no structural overlap with ΔAcc_geom).
    # S_stab is listed but flagged: since acc_geom_S_stab = acc_geom(stab) which is a
    # component of delta_acc_geom_mean_pairwise, controlling for S_stab partials out part
    # of the predictor itself — interpret with caution.
    for ctrl_name, s_arr, note in [
        ("S_center", s_ctr_arr, "primary_clean_control"),
        ("S_stab",   s_stab_arr, "caution_stab_is_component_of_delta_geom"),
        ("S_best",   s_best,     "max_of_controls"),
    ]:
        rs, passed = _specificity_one_ctrl(
            dag, dad, s_arr, ctrl_name, w,
            args.n_bootstrap, args.specificity_delta_r2, rng)
        # Annotate S_stab rows with the contamination warning
        for r in rs:
            if ctrl_name == "S_stab":
                r["note"] = (str(r.get("note", "")) + "|" + note).strip("|")
        all_rows += rs
        passed_by[ctrl_name] = passed
        flag = " [caution: contaminated]" if ctrl_name == "S_stab" else ""
        print(f"  {ctrl_name}: {'PASS' if passed else 'fail'}{flag}")

    return all_rows, passed_by


# ---------------------------------------------------------------------------
# Step 6 — Arm B (deferred)
# ---------------------------------------------------------------------------

def step6_arm_b(args: argparse.Namespace, out_dir: Path) -> list[dict]:
    print("[Step 6] Arm B: Jacobian d′_J — deferred (raw J not cached).")
    rows = [{"test": "arm_b", "statistic": "arm_b_ready", "value": 0,
             "note": "deferred_norms_only; recompute eoptotype_jacobian_field_smoothness with full J"}]
    _write_csv(out_dir / "jacobian_dprime_geometry.csv", rows)
    _write_csv(out_dir / "jacobian_approximation_error.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Step 7 — Tier-2 ΔM (deferred)
# ---------------------------------------------------------------------------

def step7_tier2(args: argparse.Namespace) -> list[dict]:
    print("[Step 7] Tier-2 ΔM: deferred.")
    return [{"test": "tier2_dM", "statistic": "mechanism_tangent_tracks", "value": 0,
             "note": "deferred_raw_J_not_cached"}]


# ---------------------------------------------------------------------------
# Step 8 — Intrinsic-noise sensitivity
# ---------------------------------------------------------------------------

def step8_noise_sensitivity(
    args: argparse.Namespace,
    arm_a_by_lm: dict, func_rows: list[dict],
    out_dir: Path, rng: np.random.Generator,
) -> tuple[list[dict], bool]:
    print("[Step 8] Intrinsic-noise sensitivity (Poisson vs isotropic)...")
    rows: list[dict] = []
    w = args.primary_window

    # Build geom curves for isotropic model from arm_a_by_lm
    def _arm_row(lm: float, nm: str, res: dict) -> dict:
        return {
            "L": _fmt_lm(lm),
            "render_ctrl": "yes" if abs(lm - args.render_limit) < 1e-4 else "no",
            "noise_model": nm,
            "delta_acc_geom_mean_pairwise": _f(res.get("delta_acc_geom_mean_pairwise")),
            "S_best_mean_dprime": _f(res.get("S_best_mean")),
            "acc_geom_S_stab": _f(res.get("acc_geom_S_stab")),
            "acc_geom_S_center": _f(res.get("acc_geom_S_center")),
        }

    iso_curve = [_arm_row(lm, "isotropic", nm_dict.get("isotropic", {}))
                 for lm, nm_dict in arm_a_by_lm.items()]
    poi_curve = [_arm_row(lm, "poisson", nm_dict.get("poisson", {}))
                 for lm, nm_dict in arm_a_by_lm.items()]

    # Check whether crossing sign is consistent between models
    Ls_p, dag_p, dad_p, _, _, _, ic_p = _extract_arrays(
        poi_curve, func_rows, w, args.render_limit, args.core_range, "poisson")

    Ls_i, dag_i, _, _, _, _, ic_i = _extract_arrays(
        iso_curve, func_rows, w, args.render_limit, args.core_range, "isotropic")

    L_geom_p, _ = _zero_crossing_core(Ls_p, dag_p, ic_p)
    L_geom_i, _ = _zero_crossing_core(Ls_i, dag_i, ic_i)

    sign_stable = True
    if np.isfinite(L_geom_p) and np.isfinite(L_geom_i):
        # Both cross — check they're within 2 grid steps
        sign_stable = abs(L_geom_p - L_geom_i) < 2 * DEFAULT_GRID_STEP
    elif np.isfinite(L_geom_p) != np.isfinite(L_geom_i):
        sign_stable = False

    rows.append({
        "test": "noise_sensitivity", "statistic": "L_geom_poisson",
        "value": L_geom_p, "note": ""})
    rows.append({
        "test": "noise_sensitivity", "statistic": "L_geom_isotropic",
        "value": L_geom_i, "note": ""})
    rows.append({
        "test": "noise_sensitivity", "statistic": "sign_stable_across_models",
        "value": int(sign_stable),
        "note": "" if sign_stable else "sign_depends_on_noise_model"})

    print(f"  L_geom (Poisson)={f'{L_geom_p:.3f}' if np.isfinite(L_geom_p) else 'nan'}, "
          f"L_geom (iso)={f'{L_geom_i:.3f}' if np.isfinite(L_geom_i) else 'nan'}")
    print(f"  sign_stable={'YES' if sign_stable else 'NO (caveat: sign_depends_on_noise_model)'}")
    return rows, sign_stable


def _build_pairwise_cond_dataset(
    merged: dict,
    merged_tidx: dict,
    cond: str,
    lm: float,
    a: int,
    b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    lk = round(lm, 4)
    vecs_a = merged.get(cond, {}).get(lk, {}).get(a, [])
    tids_a = merged_tidx.get(cond, {}).get(lk, {}).get(a, [])
    vecs_b = merged.get(cond, {}).get(lk, {}).get(b, [])
    tids_b = merged_tidx.get(cond, {}).get(lk, {}).get(b, [])

    by_a = {int(t): np.asarray(v, dtype=np.float64) for v, t in zip(vecs_a, tids_a)}
    by_b = {int(t): np.asarray(v, dtype=np.float64) for v, t in zip(vecs_b, tids_b)}
    common_ids = sorted(set(by_a.keys()) & set(by_b.keys()))
    if len(common_ids) < 2:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=int), np.empty(0, dtype=int), common_ids

    Xa = np.stack([by_a[t] for t in common_ids], axis=0)
    Xb = np.stack([by_b[t] for t in common_ids], axis=0)
    X = np.concatenate([Xa, Xb], axis=0)
    y = np.concatenate([
        np.zeros(len(common_ids), dtype=np.int64),
        np.ones(len(common_ids), dtype=np.int64),
    ], axis=0)
    groups = np.concatenate([
        np.asarray(common_ids, dtype=np.int64),
        np.asarray(common_ids, dtype=np.int64),
    ], axis=0)
    return X, y, groups, common_ids


def _grouped_binary_logistic_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    rng_seed: int,
) -> tuple[float, int, str]:
    if X.size == 0 or y.size == 0:
        return float("nan"), 0, "no_samples"

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    ug = np.unique(groups)
    n_splits = min(5, ug.size)
    if n_splits < 2:
        return float("nan"), int(n_splits), "insufficient_groups"

    splitter = GroupKFold(n_splits=n_splits)
    preds = np.full_like(y, fill_value=-1)
    for tr, te in splitter.split(X, y, groups=groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=rng_seed)
        clf.fit(Xtr, y[tr])
        preds[te] = clf.predict(Xte)

    return float(np.mean(preds == y)), int(n_splits), "ok"


# ---------------------------------------------------------------------------
# Step 9 — Pairwise
# ---------------------------------------------------------------------------

def step9_pairwise(
    args: argparse.Namespace,
    merged: dict,
    merged_tidx: dict,
    arm_a_by_lm: dict, func_rows: list[dict],
    out_dir: Path,
    rng: np.random.Generator,
) -> tuple[list[dict], int]:
    print("[Step 9] Pairwise primary target (D1-reproducing design)...")
    rows: list[dict] = []
    w = args.primary_window
    d1_abs = _load_d1_absolute(
        _resolve_d1_sweep_paths(args),
        args.logmar_grid,
        args.feature_repr,
        args.d1_sweep_readout,
        w,
    )

    cell_rows: list[dict] = []
    n_passing = 0

    def _sgn(v: float) -> int:
        if not np.isfinite(v) or abs(v) < 1e-12:
            return 0
        return 1 if v > 0 else -1

    for lm in sorted(args.logmar_grid):
        if abs(lm - args.render_limit) < 1e-4:
            continue
        lk = round(lm, 4)
        pair_map = d1_abs.get(lk, {})
        arm = arm_a_by_lm.get(lm, {}).get("poisson", {})

        for a, b in PAIRS:
            lbl = _pair_label(a, b)
            d1_real = _f(pair_map.get(lbl, {}).get("real", float("nan")))
            d1_stab = _f(pair_map.get(lbl, {}).get("stabilized", float("nan")))
            d1_delta = d1_real - d1_stab if np.isfinite(d1_real) and np.isfinite(d1_stab) else float("nan")

            Xr, yr, gr, ids_r = _build_pairwise_cond_dataset(merged, merged_tidx, "real", lm, a, b)
            Xs, ys, gs, ids_s = _build_pairwise_cond_dataset(merged, merged_tidx, "stabilized", lm, a, b)
            acc_real, nsplit_real, note_real = _grouped_binary_logistic_accuracy(Xr, yr, gr, args.random_seed)
            acc_stab, nsplit_stab, note_stab = _grouped_binary_logistic_accuracy(Xs, ys, gs, args.random_seed)
            acc_delta = acc_real - acc_stab if np.isfinite(acc_real) and np.isfinite(acc_stab) else float("nan")

            arm_real = _f(arm.get("acc_geom_pair_real", {}).get((a, b)))
            arm_stab = _f(arm.get("acc_geom_pair_stabilized", {}).get((a, b)))
            arm_delta = _f(arm.get("delta_acc_geom_pair", {}).get((a, b)))

            sign_agree = int(_sgn(d1_delta) != 0 and _sgn(d1_delta) == _sgn(arm_delta))
            meaningful = int(
                sign_agree == 1
                and np.isfinite(d1_delta)
                and np.isfinite(arm_delta)
                and abs(d1_delta) >= float(args.pairwise_meaningful_delta)
                and abs(arm_delta) >= float(args.pairwise_meaningful_delta)
            )

            cell_rows.append({
                "logmar": _fmt_lm(lm),
                "orientation_pair": lbl,
                "d1_pair_acc_real": d1_real,
                "d1_pair_acc_stabilized": d1_stab,
                "d1_pair_delta": d1_delta,
                "cached_pair_logistic_acc_real": acc_real,
                "cached_pair_logistic_acc_stabilized": acc_stab,
                "cached_pair_logistic_delta": acc_delta,
                "armA_pair_acc_real": arm_real,
                "armA_pair_acc_stabilized": arm_stab,
                "armA_pair_delta": arm_delta,
                "sign_agreement_d1_vs_armA": sign_agree,
                "meaningful_crossover_match": meaningful,
                "n_common_trial_ids_real": len(ids_r),
                "n_common_trial_ids_stabilized": len(ids_s),
                "common_trial_ids_real": ",".join(str(v) for v in ids_r),
                "common_trial_ids_stabilized": ",".join(str(v) for v in ids_s),
                "n_splits_real": nsplit_real,
                "n_splits_stabilized": nsplit_stab,
                "cached_logistic_note_real": note_real,
                "cached_logistic_note_stabilized": note_stab,
                "d1_feature_name": f"d1_time_mean_w{int(w)}",
                "cached_feature_name": f"vectors__{args.feature_repr}",
            })

    _write_csv(out_dir / "pairwise_primary_target_cells.csv", cell_rows)

    valid_cells = [
        r for r in cell_rows
        if np.isfinite(_f(r.get("d1_pair_delta"))) and np.isfinite(_f(r.get("armA_pair_delta")))
    ]
    sign_vals = [int(r.get("sign_agreement_d1_vs_armA", 0)) for r in valid_cells]
    overall_sign = float(np.mean(sign_vals)) if sign_vals else float("nan")

    by_pair_summary: list[dict] = []
    for a, b in PAIRS:
        lbl = _pair_label(a, b)
        cells = [r for r in valid_cells if str(r.get("orientation_pair")) == lbl]
        if not cells:
            continue
        sign_rate = float(np.mean([int(r.get("sign_agreement_d1_vs_armA", 0)) for r in cells]))
        meaningful_count = int(sum(int(r.get("meaningful_crossover_match", 0)) for r in cells))
        by_pair_summary.append({
            "summary_type": "by_orientation_pair",
            "orientation_pair": lbl,
            "sign_agreement_rate": sign_rate,
            "n_cells": len(cells),
            "n_meaningful_matches": meaningful_count,
        })

    by_lm_summary: list[dict] = []
    for lm in sorted({float(_f(r.get("logmar"))) for r in valid_cells}):
        cells = [r for r in valid_cells if abs(_f(r.get("logmar")) - lm) < 1e-4]
        if not cells:
            continue
        sign_rate = float(np.mean([int(r.get("sign_agreement_d1_vs_armA", 0)) for r in cells]))
        meaningful_count = int(sum(int(r.get("meaningful_crossover_match", 0)) for r in cells))
        by_lm_summary.append({
            "summary_type": "by_logmar",
            "logmar": _fmt_lm(lm),
            "sign_agreement_rate": sign_rate,
            "n_cells": len(cells),
            "n_meaningful_matches": meaningful_count,
        })

    d1_d = np.asarray([_f(r.get("d1_pair_delta")) for r in valid_cells], dtype=float)
    arm_d = np.asarray([_f(r.get("armA_pair_delta")) for r in valid_cells], dtype=float)
    corr = float("nan")
    if d1_d.size >= 3 and np.nanstd(d1_d) > 0 and np.nanstd(arm_d) > 0:
        corr = float(np.corrcoef(d1_d, arm_d)[0, 1])

    any_meaningful = int(any(int(r.get("meaningful_crossover_match", 0)) == 1 for r in valid_cells))
    if np.isfinite(overall_sign) and overall_sign >= 0.60 and np.isfinite(corr) and corr >= 0.20 and any_meaningful:
        pairwise_label = "pairwise_crossover_supported"
        pairwise_note = "Pairwise D1 deltas and Arm A deltas show consistent sign agreement with meaningful matches."
    elif np.isfinite(overall_sign) and overall_sign >= 0.50 and np.isfinite(corr) and corr > 0:
        pairwise_label = "pairwise_crossover_partial"
        pairwise_note = "Pairwise sign agreement is above chance but crossover support is partial/weak."
    else:
        pairwise_label = "pairwise_no_crossover_match"
        pairwise_note = "Pairwise D1-vs-Arm A deltas do not show robust crossover alignment."

    n_passing = int(sum(1 for r in by_pair_summary if _f(r.get("sign_agreement_rate")) >= 0.5 and int(r.get("n_meaningful_matches", 0)) > 0))
    summary_rows = [
        {
            "summary_type": "overall",
            "orientation_pair": "all",
            "logmar": "all",
            "sign_agreement_rate": overall_sign,
            "n_cells": len(valid_cells),
            "n_meaningful_matches": int(sum(int(r.get("meaningful_crossover_match", 0)) for r in valid_cells)),
            "pairwise_delta_corr": corr,
            "any_meaningful_crossover": any_meaningful,
            "pairwise_primary_label": pairwise_label,
            "note": pairwise_note,
        }
    ] + by_pair_summary + by_lm_summary
    _write_csv(out_dir / "pairwise_primary_target_summary.csv", summary_rows)

    rows.append({
        "test": "pairwise_primary_target",
        "statistic": "overall_sign_agreement",
        "value": overall_sign,
        "window": w,
        "noise_model": "poisson",
        "note": "d1_pair_delta_vs_armA_pair_delta",
    })
    rows.append({
        "test": "pairwise_primary_target",
        "statistic": "pairwise_delta_correlation",
        "value": corr,
        "window": w,
        "noise_model": "poisson",
        "note": "corr(d1_pair_delta, armA_pair_delta)",
    })
    rows.append({
        "test": "pairwise_primary_target",
        "statistic": "any_meaningful_crossover",
        "value": any_meaningful,
        "window": w,
        "noise_model": "poisson",
        "note": f"|delta|>={args.pairwise_meaningful_delta:g}",
    })
    rows.append({
        "test": "pairwise_primary_target",
        "statistic": "pairwise_primary_label",
        "value": float("nan"),
        "window": w,
        "noise_model": "poisson",
        "note": pairwise_label,
    })

    print(f"  Pairwise sign agreement (overall): {overall_sign:.3f}" if np.isfinite(overall_sign) else "  Pairwise sign agreement (overall): nan")
    print(f"  Pairwise delta correlation: {corr:.3f}" if np.isfinite(corr) else "  Pairwise delta correlation: nan")
    print(f"  Pairwise primary label: {pairwise_label}")

    print(f"  Pairs with >=50% sign agreement and >=1 meaningful cell: {n_passing}/{len(PAIRS)}")
    return rows, n_passing


# ---------------------------------------------------------------------------
# Step 10 — Nulls
# ---------------------------------------------------------------------------

def step10_nulls(
    args: argparse.Namespace,
    merged: dict, merged_tidx: dict,
    r_bar: np.ndarray,
    geom_rows: list[dict], func_rows: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    print("[Step 10] Nulls (orientation-label shuffle; logmar-permutation)...")
    print("  NOTE: These are NOT the spec's position-phase shuffle. See null_type column.")
    rows: list[dict] = []
    w = args.primary_window
    n_null = min(args.n_bootstrap, 500)

    Ls, dag, dad, _, _, _, ic = _extract_arrays(
        geom_rows, func_rows, w, args.render_limit, args.core_range)
    valid = np.isfinite(dag) & np.isfinite(dad)

    # Real slope
    real_slope = float("nan")
    if valid.sum() >= 3:
        real_slope, *_ = scipy_stats.linregress(dag[valid], dad[valid])

    # Orientation-label shuffle: permute which class label maps to which trial responses.
    # This tests whether the sign/magnitude of ΔAcc_geom depends on correct class labeling.
    # (Not a position-preserving phase shuffle — overstates the null if responses
    # carry orientation-specific energy. Rename and down-weight accordingly.)
    ori_label_slopes: list[float] = []
    sigma_int = _sigma_int_poisson(r_bar, args.primary_window)
    for _ in range(n_null):
        dag_shuf = np.full(len(Ls), float("nan"))
        for i, lm in enumerate(Ls):
            lk = round(lm, 4)
            real_rvec = merged.get("real", {}).get(lk, {})
            stab_rvec = merged.get("stabilized", {}).get(lk, {})
            all_r = [resp for ori in ORIENTATIONS for resp in real_rvec.get(ori, [])]
            if len(all_r) < 4:
                continue
            perm = rng.permutation(len(all_r))
            n_per = len(all_r) // 4
            shuf_means: dict[int, np.ndarray] = {}
            for k, ori in enumerate(ORIENTATIONS):
                blk = perm[k*n_per:(k+1)*n_per]
                shuf_means[ori] = np.stack([all_r[j] for j in blk]).mean(axis=0)
            stab_means: dict[int, np.ndarray] = {}
            for ori in ORIENTATIONS:
                ts = stab_rvec.get(ori, [])
                if ts:
                    stab_means[ori] = np.stack(ts).mean(axis=0)
            if len(shuf_means) == 4 and len(stab_means) == 4:
                shuf_cov = _pooled_within_class_cov(
                    {ori: np.stack([all_r[perm[k*n_per + j]] for j in range(n_per)])
                     for k, ori in enumerate(ORIENTATIONS)})
                st_real = sigma_int + shuf_cov
                st_stab = sigma_int + _pooled_within_class_cov(
                    {ori: np.stack(stab_rvec.get(ori, [])) for ori in ORIENTATIONS
                     if stab_rvec.get(ori)})
                acc_r = _ncm_accuracy(shuf_means, st_real, args.n_mc // 5, rng)
                acc_s = _ncm_accuracy(stab_means, st_stab, args.n_mc // 5, rng)
                dag_shuf[i] = acc_r - acc_s
        v = np.isfinite(dag_shuf) & np.isfinite(dad)
        if v.sum() >= 3:
            sl, *_ = scipy_stats.linregress(dag_shuf[v], dad[v])
            ori_label_slopes.append(sl)

    # LogMAR-permutation null: permute dag values across LogMAR levels.
    # Tests whether the across-LogMAR structure of dag matters for the slope.
    # (Weak with few points — downweight; renamed from "pair-label shuffle".)
    logmar_perm_slopes: list[float] = []
    for _ in range(n_null):
        if valid.sum() >= 3:
            dag_perm = dag.copy()
            dag_perm[valid] = dag[valid][rng.permutation(valid.sum())]
            v2 = np.isfinite(dag_perm) & np.isfinite(dad)
            if v2.sum() >= 3:
                sl, *_ = scipy_stats.linregress(dag_perm[v2], dad[v2])
                logmar_perm_slopes.append(sl)

    # p = frac(null < real) — high means real exceeds null
    def _ep(real, null):
        if not null or not np.isfinite(real):
            return float("nan")
        return float(np.mean([nv < real for nv in null if np.isfinite(nv)]))

    p_ori = _ep(real_slope, ori_label_slopes)
    p_lmp = _ep(real_slope, logmar_perm_slopes)
    # Do not gate nulls_passed on 4-core-point results; always label clearly
    nulls_passed = (np.isfinite(p_ori) and p_ori >= 0.95 and
                    np.isfinite(p_lmp) and p_lmp >= 0.95)

    rows += [
        {"test": "nulls", "statistic": "real_slope", "value": real_slope, "note": "",
         "null_type": "observed"},
        {"test": "nulls", "statistic": "ori_label_shuffle_slope_median",
         "value": float(np.nanmedian(ori_label_slopes)) if ori_label_slopes else float("nan"),
         "null_p": p_ori, "note": "p=frac_null_lt_real",
         "null_type": "orientation_label_shuffle_NOT_position_phase_shuffle"},
        {"test": "nulls", "statistic": "logmar_permutation_slope_median",
         "value": float(np.nanmedian(logmar_perm_slopes)) if logmar_perm_slopes else float("nan"),
         "null_p": p_lmp, "note": "p=frac_null_lt_real; WEAK on 4 pts",
         "null_type": "logmar_permutation_NOT_pair_label_shuffle"},
        {"test": "nulls", "statistic": "nulls_passed", "value": int(nulls_passed),
         "note": "TRUE" if nulls_passed else "FALSE"},
    ]
    print(f"  p(ori_label<real)={f'{p_ori:.3f}' if np.isfinite(p_ori) else 'nan'}, "
          f"p(logmar_perm<real)={f'{p_lmp:.3f}' if np.isfinite(p_lmp) else 'nan'}")
    print(f"  nulls_passed={'TRUE' if nulls_passed else 'FALSE'}")
    return rows


# ---------------------------------------------------------------------------
# Step 11 — Window robustness
# ---------------------------------------------------------------------------

def step11_window_robustness(
    args: argparse.Namespace,
    geom_rows: list[dict], func_rows: list[dict],
    n_core_ready: int,
) -> tuple[list[dict], bool]:
    print("[Step 11] Window robustness...")
    rows: list[dict] = []
    slopes: list[float] = []

    for w in args.windows:
        Ls, dag, dad, _, _, _, _ = _extract_arrays(
            geom_rows, func_rows, w, args.render_limit, args.core_range)
        valid = np.isfinite(dag) & np.isfinite(dad)
        if valid.sum() < 3:
            continue
        sl, _, rr, _, _ = scipy_stats.linregress(dag[valid], dad[valid])
        slopes.append(sl)
        L_g, _ = _zero_crossing_core(Ls, dag, np.ones(len(Ls), dtype=bool))
        L_d, _ = _zero_crossing_core(Ls, dad, np.ones(len(Ls), dtype=bool))
        rows.append({"test": "window_robustness", "window": w,
                     "statistic": f"slope_W{w}", "value": sl,
                     "note": f"R2={rr**2:.3f}"})

    finite_sl = [s for s in slopes if np.isfinite(s)]
    stable = (len(finite_sl) >= 2 and all(s > 0 for s in finite_sl)
              and max(finite_sl) / max(min(finite_sl), EPS) < 2.0)
    rows.append({"test": "window_robustness", "window": args.primary_window,
                 "statistic": "window_stable", "value": int(stable),
                 "note": f"slopes={[round(s,4) for s in finite_sl]}"})
    print(f"  window_stable={'TRUE' if stable else 'FALSE'}, slopes={[round(s,4) for s in finite_sl]}")
    return rows, stable


# ---------------------------------------------------------------------------
# Step 12 — Decision, tables, readme
# ---------------------------------------------------------------------------

def _apply_decision(
    coincidence: bool, cont_sig: bool, spec_passed: dict[str, bool],
    mechanism: bool, nulls_passed: bool, window_stable: bool,
    sign_stable: bool, n_core_ready: int, min_core: int,
    no_geom_crossing: bool, render_confounded: bool,
    n_pairs_passing: int, confirmed_d1: str | None,
) -> tuple[str, str, str, list[str]]:
    """Returns (label, implication, action, caveat_tags)."""
    caveats: list[str] = []
    if not sign_stable:
        caveats.append("sign_depends_on_noise_model")
    if n_pairs_passing == 1 or n_pairs_passing == 2:
        caveats.append("pairwise_driven")

    if confirmed_d1 is None:
        return ("exploratory_only_missing_D1_source",
                "D1 source unconfirmed — no verdict possible.",
                "Confirm D1 sweep path and column.", caveats)

    if render_confounded:
        return ("render_limit_confounded",
                "Effect requires -0.40 control point. Inconclusive.",
                "Add finer logmar grid within core range.", caveats)

    if n_core_ready < min_core:
        return ("inconclusive_underpowered_grid",
                f"Only {n_core_ready}/{min_core} core points. Densify the logmar grid.",
                "Run eoptotype on dense grid (-0.40 to -0.10, step 0.025).", caveats)

    if no_geom_crossing:
        return ("observable_cannot_test_coincidence",
                "ΔAcc_geom has no zero-crossing in core. Inspect Σ_pos construction.",
                "Check per-orientation covariance computation; verify Σ_pos is non-zero for real.", caveats)

    spec_any = any(spec_passed.values())
    spec_sbest = spec_passed.get("S_best", False)

    if coincidence and cont_sig and spec_sbest and mechanism:
        return ("jacobian_geometry_predicts_D1_crossover",
                "Tier-1 (finite cloud) AND Tier-2 (Jacobian) both pass. "
                "Fig 4 leads with the Jacobian/equivariant-manifold story.",
                "Finalize Fig 4 with both arms.", caveats)

    if coincidence and cont_sig and spec_sbest and not mechanism:
        return ("tier2_jacobian_missing",
                "Arm A (finite-cloud d′) passes all criteria. "
                "Tier-2 deferred: raw J not cached. "
                "Cannot yet attribute to translation-tangent mechanism.",
                "Recompute Jacobians storing full J matrices.", caveats)

    if coincidence and cont_sig and spec_any and not spec_sbest:
        return ("finite_cloud_geometry_predicts_D1_crossover",
                "Finite-cloud d′ predicts the D1 crossover. "
                "Passes coincidence, continuous, specificity beyond at least one difficulty control.",
                "Compute raw J for Arm B. Report which S controls are exceeded.", caveats)

    if cont_sig and not spec_any:
        return ("geometry_tracks_difficulty_only",
                "d′_geom continuous with Δacc but not beyond any difficulty control. "
                "Geometry tracks task difficulty; lead with acuity.",
                "Decompose d′_geom vs S_best correlation.", caveats)

    if coincidence and not cont_sig:
        return ("coincidence_without_continuous",
                "Zero-crossings coincide but no significant continuous relationship.",
                "Investigate if coincidence is driven by single logmar. Check n_core.", caveats)

    return ("geometry_descriptive_not_predictive",
            "Finite-cloud d′ does not predict D1 crossover beyond difficulty.",
            "Separate geometry and function results. Check Σ_pos magnitude.", caveats)


def step12_output(
    args: argparse.Namespace,
    geom_rows: list[dict], func_rows: list[dict],
    all_test_rows: list[dict],
    coincidence: bool, cont_sig: bool,
    spec_passed: dict[str, bool],
    mechanism: bool, nulls_passed_arg: bool,
    window_stable: bool, sign_stable: bool,
    n_core_ready: int, no_geom_crossing: bool,
    render_confounded: bool, n_pairs_passing: int,
    confirmed_d1: str | None,
    out_dir: Path,
) -> None:
    print("[Step 12] Writing outputs...")
    w = args.primary_window

    # Extract nulls_passed from test rows (not the arg, which may be stale)
    nulls_passed = any(
        r.get("statistic") == "nulls_passed" and np.isfinite(_f(r.get("value", 0)))
        and int(_f(r.get("value", 0))) == 1
        for r in all_test_rows
    )

    label, impl, action, caveats = _apply_decision(
        coincidence, cont_sig, spec_passed, mechanism,
        nulls_passed, window_stable, sign_stable,
        n_core_ready, args.min_core_points, no_geom_crossing,
        render_confounded,   # now wired through from step3
        n_pairs_passing, confirmed_d1
    )

    readout_validation_label = next(
        (str(r.get("note", "")) for r in all_test_rows
         if r.get("test") == "readout_validation" and r.get("statistic") == "validation_label"),
        "",
    )
    pairwise_primary_label = next(
        (str(r.get("note", "")) for r in all_test_rows
         if r.get("test") == "pairwise_primary_target" and r.get("statistic") == "pairwise_primary_label"),
        "",
    )
    pairwise_sign_agree = next(
        (_f(r.get("value")) for r in all_test_rows
         if r.get("test") == "pairwise_primary_target" and r.get("statistic") == "overall_sign_agreement"),
        float("nan"),
    )
    pairwise_corr = next(
        (_f(r.get("value")) for r in all_test_rows
         if r.get("test") == "pairwise_primary_target" and r.get("statistic") == "pairwise_delta_correlation"),
        float("nan"),
    )
    pairwise_meaningful = next(
        (_f(r.get("value")) for r in all_test_rows
         if r.get("test") == "pairwise_primary_target" and r.get("statistic") == "any_meaningful_crossover"),
        float("nan"),
    )
    readout_best_model = next(
        (str(r.get("note", "")) for r in all_test_rows
         if r.get("test") == "readout_validation" and r.get("statistic") == "best_model"),
        "",
    )
    if pairwise_primary_label:
        sign_txt = f"{pairwise_sign_agree:.3f}" if np.isfinite(pairwise_sign_agree) else "nan"
        corr_txt = f"{pairwise_corr:.3f}" if np.isfinite(pairwise_corr) else "nan"
        meaningful_txt = str(int(pairwise_meaningful)) if np.isfinite(pairwise_meaningful) else "0"
        label = pairwise_primary_label
        impl = (
            "Primary functional target set to pairwise D1-reproducing grouped-CV design "
            "(same common trial IDs and pairwise task)."
        )
        action = (
            "Use pairwise_primary_target_cells.csv and pairwise_primary_target_summary.csv "
            f"as the main crossover evidence (sign_agreement={sign_txt}, "
            f"corr={corr_txt}, any_meaningful={meaningful_txt})."
        )
    elif readout_validation_label == "observer_noise_model_miscalibrated":
        label = readout_validation_label
        impl = (
            "Cached-vector logistic/LDA readouts reproduce D1 more faithfully than the Arm A "
            "Gaussian observer. Pause Arm A coincidence interpretation until the observer is recalibrated."
        )
        action = (
            f"Refit the Arm A noise scale against absolute real/stabilized D1 accuracies with leave-one-LogMAR-out calibration; best classifier={readout_best_model or 'unknown'}."
        )
    elif readout_validation_label == "cached_vector_D1_mismatch":
        label = readout_validation_label
        impl = (
            "Grouped-CV classifiers are also near chance on the cached vectors. The current Arm A cache does not support the D1 task as expected."
        )
        action = "Audit the cached-vector source, feature extraction, and trial alignment before using Arm A coincidence."

    # Extract key numbers
    L_geom = next((_f(r.get("value")) for r in all_test_rows
                   if r.get("statistic") == "L_geom"), float("nan"))
    L_func = next((_f(r.get("value")) for r in all_test_rows
                   if r.get("statistic") == "L_func"), float("nan"))
    dL = next((_f(r.get("value")) for r in all_test_rows
               if r.get("statistic") == "dL"), float("nan"))
    dL_ci_lo = next((_f(r.get("ci_low")) for r in all_test_rows
                     if r.get("statistic") == "dL"), float("nan"))
    dL_ci_hi = next((_f(r.get("ci_high")) for r in all_test_rows
                     if r.get("statistic") == "dL"), float("nan"))
    slope = next((_f(r.get("value")) for r in all_test_rows
                  if r.get("statistic") == "slope"), float("nan"))
    r2 = next((_f(r.get("value")) for r in all_test_rows
               if r.get("statistic") == "R2"), float("nan"))
    rho = next((_f(r.get("value")) for r in all_test_rows
                if r.get("statistic") == "spearman_rho"), float("nan"))

    # dprime_geometry_vs_D1_crosswalk.csv
    gby = {float(r["L"]): r for r in geom_rows if r.get("noise_model") == "poisson"}
    fby = {float(r["L"]): r for r in func_rows if int(r.get("window", 0)) == w}
    cw_rows = []
    for lm in args.logmar_grid:
        gr = gby.get(lm, {})
        fr = fby.get(lm, {})
        cw_rows.append({
            "L": _fmt_lm(lm),
            "delta_acc_geom_A": gr.get("delta_acc_geom", "nan"),
            "delta_acc_geom_B": "nan",
            "delta_acc_D1_mean_pairwise": fr.get("delta_acc_mean", "nan"),
            "S_best_mean_dprime": gr.get("S_best_mean_dprime", "nan"),
        })
    _write_csv(out_dir / "dprime_geometry_vs_D1_crosswalk.csv", cw_rows)

    # keystone_tests.csv
    fnames_t = ["test", "statistic", "value", "ci_low", "ci_high",
                "window", "noise_model", "ctrl", "pair", "null_p", "note", "null_type"]
    for r in all_test_rows:
        for f in fnames_t:
            r.setdefault(f, "")
    _write_csv(out_dir / "keystone_tests.csv", all_test_rows, fnames_t)

    def _yn(b: bool) -> str:
        return "TRUE" if b else "FALSE"
    def _fs(v: float) -> str:
        return f"{v:.3f}" if np.isfinite(v) else "nan"

    # Decision table
    decision_row = {
        "run_label": args.run_label,
        "primary_window": w,
        "d1_col": args.d1_sweep_col,
        "d1_source": _d1_source_label(_resolve_d1_sweep_paths(args)),
        "n_core_ready": n_core_ready,
        "min_core_required": args.min_core_points,
        "L_geom_A": _fs(L_geom), "L_func": _fs(L_func),
        "dL": _fs(dL), "dL_ci": f"[{_fs(dL_ci_lo)},{_fs(dL_ci_hi)}]",
        "coincidence": _yn(coincidence),
        "continuous_significant": _yn(cont_sig),
        "slope": _fs(slope), "R2": _fs(r2), "spearman_rho": _fs(rho),
        "specificity_S_best": _yn(spec_passed.get("S_best", False)),
        "specificity_S_stab": _yn(spec_passed.get("S_stab", False)),
        "specificity_S_center": _yn(spec_passed.get("S_center", False)),
        "mechanism_tangent_tracks": _yn(mechanism),
        "nulls_passed": _yn(nulls_passed),
        "window_stable": _yn(window_stable),
        "sign_stable_noise_model": _yn(sign_stable),
        "no_geom_crossing": _yn(no_geom_crossing),
        "pairs_passing_both": f"{n_pairs_passing}/{len(PAIRS)}",
        "arm_B_ready": "no",
        "decision_label": label,
        "caveat_tags": "|".join(caveats) if caveats else "",
        "manuscript_implication": impl,
        "next_action": action,
    }
    _write_csv(out_dir / "geometry_dprime_decision_table.csv", [decision_row])

    # README
    lines = [
        "# Keystone Geometry Crossover v3 — Results README",
        "",
        f"**Run label:** {args.run_label}  ",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Observable:** finite-cloud d′_geom (Arm A); Jacobian d′_J (Arm B — deferred)  ",
        f"**D1 column:** `{args.d1_sweep_col}`  ",
        f"**D1 source:** `{_d1_source_label(_resolve_d1_sweep_paths(args))}`  ",
        f"**NOTE:** 4-way D1 not in cache; function obs = mean of 6 pairwise deltas (proxy)  ",
        f"**Primary window:** W={w}  ",
        f"**Grid:** {list(args.logmar_grid)}  ",
        f"**Core range:** {list(args.core_range)}  ",
        f"**Core points ready:** {n_core_ready}/{args.min_core_points} required  ",
        "",
        "## Summary Answers",
        "",
        f"1. **Arm-A coincidence within core?** {'YES' if coincidence else 'NO'}  ",
        f"   L_geom={_fs(L_geom)}, L_func={_fs(L_func)}, dL={_fs(dL)} "
        f"CI=[{_fs(dL_ci_lo)}, {_fs(dL_ci_hi)}]  ",
        "" if np.isfinite(L_geom) else
        "   WARNING: ΔAcc_geom has no zero-crossing (label: observable_cannot_test_coincidence)  ",
        "",
        f"2. **Arm-A continuous (beyond {args.min_core_points} core pts)?** {'YES' if cont_sig else 'NO'}  ",
        f"   slope={_fs(slope)}, R²={_fs(r2)}, ρ={_fs(rho)}  ",
        f"   Core points available: {n_core_ready} (need {args.min_core_points})  ",
        "",
        "3. **Beyond difficulty controls?**  ",
        f"   S_best: {'YES' if spec_passed.get('S_best') else 'NO'}  ",
        f"   S_stab: {'YES' if spec_passed.get('S_stab') else 'NO'}  ",
        f"   S_center: {'YES' if spec_passed.get('S_center') else 'NO'}  ",
        "",
        f"4. **Does Arm B reproduce Arm A?** DEFERRED (raw J not cached)  ",
        "",
        f"5. **ΔM confirms tangent mechanism?** DEFERRED  ",
        "",
        f"6. **Pairs passing coincidence + specificity:** {n_pairs_passing}/{len(PAIRS)}  ",
        "",
        f"7. **Sign stable across noise models?** {'YES' if sign_stable else 'NO (caveat)'}  ",
        "",
        f"8. **≥8 core points?** {'YES' if n_core_ready >= args.min_core_points else 'NO — underpowered'}  ",
        "",
        f"9. **Decision label:** `{label}`  ",
        f"   Caveats: {', '.join(caveats) if caveats else 'none'}  ",
        f"   {impl}  ",
        "",
        "## Manuscript implication",
        "", f"{impl}  ", "", f"**Next action:** {action}  ",
        "",
        "## Notes",
        "",
        "- v2 results (`G_sep`, smoke2) are **quarantined** — wrong observable.  ",
        "- `d′_geom` = ||ḡ_a−ḡ_b||² / sqrt((ḡ_a−ḡ_b)ᵀ Σ_tot (ḡ_a−ḡ_b)),  ",
        "  Σ_tot = Σ_int/W + Σ_pos^A.  ",
        "- Σ_pos^A = pooled within-orientation covariance of trial responses.  ",
        "- Σ_int = diag(r̄)/W (Poisson primary); σ²I/W (isotropic secondary).  ",
        "- Ideal observer: fixed nearest-class-mean under shared Σ_tot (Monte Carlo).  ",
        "  Parameter-free; NOT the D1 decoder. Firewall intact.  ",
        "- 4-way D1 function source unavailable in cache; using mean-pairwise proxy.  ",
        "  Rerun eoptotype with 4-class accuracy to resolve.  ",
        "- Dense grid (0.025 step, ≥8 core pts) required for continuous/specificity verdicts.  ",
    ]
    (out_dir / "keystone_readme.md").write_text("\n".join(lines) + "\n")

    print(f"\n  === DECISION v3: {label} ===")
    if caveats:
        print(f"  Caveats: {', '.join(caveats)}")
    print(f"  {impl}")
    print(f"  Next: {action}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(
    args: argparse.Namespace,
    geom_rows: list[dict], func_rows: list[dict],
    all_test_rows: list[dict], arm_a_by_lm: dict,
    fig_dir: Path,
) -> None:
    print("[Figures] Generating A–G...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    w = args.primary_window
    C1, C2 = "#2166AC", "#D6604D"

    Ls, dag, dad, _, _, s_best, ic = _extract_arrays(
        geom_rows, func_rows, w, args.render_limit, args.core_range)
    fby = {float(r["L"]): r for r in func_rows if int(r.get("window", 0)) == w}
    ci_lo = np.array([_f(fby.get(lm, {}).get("delta_ci_low")) for lm in Ls])
    ci_hi = np.array([_f(fby.get(lm, {}).get("delta_ci_high")) for lm in Ls])

    # Fig A — keystone overlay
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.axvspan(args.render_limit - 0.02, args.render_limit + 0.02,
                alpha=0.12, color="gray")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.3)
    ax1.plot(Ls, dag, "o-", color=C1, lw=2, label="ΔAcc_geom^A (d′)")
    ax2.plot(Ls, dad, "s--", color=C2, lw=2, label="Δacc_D1 (mean-pair proxy)")
    if np.any(np.isfinite(ci_lo)):
        ax2.fill_between(Ls, ci_lo, ci_hi, alpha=0.2, color=C2)
    ax1.set_ylabel("ΔAcc_geom^A", color=C1, fontsize=10)
    ax2.set_ylabel("Δacc_D1", color=C2, fontsize=10)
    ax1.set_xlabel("LogMAR", fontsize=10)
    ax1.set_title("Fig A: Keystone overlay (Arm A d′_geom)", fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "figA_keystone_overlay.pdf", dpi=200)
    fig.savefig(fig_dir / "figA_keystone_overlay.png", dpi=150)
    plt.close(fig)

    # Fig B — continuous scatter
    valid = np.isfinite(dag) & np.isfinite(dad)
    fig, ax = plt.subplots(figsize=(5, 4))
    if valid.sum() >= 2:
        sc = ax.scatter(dag[valid], dad[valid], c=Ls[valid], cmap="plasma",
                        s=60, edgecolors="k", lw=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label="LogMAR")
        if valid.sum() >= 3:
            sl, ic_f, r, _, _ = scipy_stats.linregress(dag[valid], dad[valid])
            xf = np.linspace(dag[valid].min(), dag[valid].max(), 50)
            ax.plot(xf, sl*xf + ic_f, "k-", lw=1.5, label=f"slope={sl:.3f} R²={r**2:.2f}")
            ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("ΔAcc_geom^A"); ax.set_ylabel("Δacc_D1")
    ax.set_title("Fig B: Continuous ΔAcc_geom^A → Δacc_D1")
    fig.tight_layout()
    fig.savefig(fig_dir / "figB_continuous.pdf", dpi=200)
    fig.savefig(fig_dir / "figB_continuous.png", dpi=150)
    plt.close(fig)

    # Fig C — difficulty control
    valid3 = valid & np.isfinite(s_best)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    if valid3.sum() >= 3:
        def _r(y, x):
            sl, ic_f, _, _, _ = scipy_stats.linregress(x, y)
            return y - (sl*x + ic_f)
        dr = _r(dad[valid3], s_best[valid3])
        gr = _r(dag[valid3], s_best[valid3])
        axes[0].scatter(gr, dr, c=Ls[valid3], cmap="plasma", s=50, edgecolors="k", lw=0.4)
        if len(gr) >= 3:
            sl, ic_f, r, _, _ = scipy_stats.linregress(gr, dr)
            xf = np.linspace(gr.min(), gr.max(), 50)
            axes[0].plot(xf, sl*xf + ic_f, "k-", lw=1.5, label=f"R²={r**2:.2f}")
            axes[0].legend(fontsize=7)
        axes[0].set_xlabel("ΔAcc_geom | S_best"); axes[0].set_ylabel("Δacc | S_best")
        axes[0].set_title("Partial effect (beyond S_best)")
        axes[1].scatter(s_best[valid3], dad[valid3], c=Ls[valid3], cmap="plasma",
                        s=50, edgecolors="k", lw=0.4)
        axes[1].set_xlabel("S_best (d′)"); axes[1].set_ylabel("Δacc_D1")
        axes[1].set_title("S_best alone")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
    fig.suptitle("Fig C: Specificity beyond S_best", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "figC_difficulty.pdf", dpi=200)
    fig.savefig(fig_dir / "figC_difficulty.png", dpi=150)
    plt.close(fig)

    # Fig D — Arm A vs B (placeholder)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(Ls, dag, "o-", color=C1, lw=2, label="Arm A (finite cloud)")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.text(0.5, 0.6, "Arm B (Jacobian): deferred — raw J not cached",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax.set_xlabel("LogMAR"); ax.set_ylabel("ΔAcc_geom")
    ax.set_title("Fig D: Arm A vs Arm B comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figD_armA_vs_B.pdf", dpi=200)
    fig.savefig(fig_dir / "figD_armA_vs_B.png", dpi=150)
    plt.close(fig)

    # Fig E — Tier-2 ΔM (placeholder)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "Tier-2 ΔM: deferred (raw J not cached)",
            ha="center", va="center", transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax.set_title("Fig E: Tier-2 ΔM mechanism"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "figE_tier2.pdf", dpi=200)
    fig.savefig(fig_dir / "figE_tier2.png", dpi=150)
    plt.close(fig)

    # Fig F — pairwise small multiples
    gp_by = {}
    for lm, nm_dict in arm_a_by_lm.items():
        res = nm_dict.get("poisson", {})
        for a, b in PAIRS:
            p = (a, b)
            lbl = _pair_label(a, b)
            gp_by.setdefault(lbl, {})[lm] = _f(res.get("delta_dprime_pair", {}).get(p))
    fp_by = {}
    for r in func_rows:
        if int(r.get("window", 0)) == w and "pair" in r:
            fp_by.setdefault(str(r["pair"]), {})[float(r["L"])] = _f(r.get("delta_acc_pair"))

    ncols = 3
    nrows = math.ceil(len(PAIRS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3), squeeze=False)
    for idx, (a, b) in enumerate(PAIRS):
        ax = axes[idx // ncols][idx % ncols]
        lbl = _pair_label(a, b)
        Lp = sorted(gp_by.get(lbl, {}).keys())
        Lp = [l for l in Lp if abs(l - args.render_limit) > 1e-4]
        gvals = np.array([gp_by.get(lbl, {}).get(l, float("nan")) for l in Lp])
        dvals = np.array([fp_by.get(lbl, {}).get(l, float("nan")) for l in Lp])
        ax2 = ax.twinx()
        ax.plot(Lp, gvals, "o-", color=C1, lw=1.5, ms=4)
        ax2.plot(Lp, dvals, "s--", color=C2, lw=1.5, ms=4)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax2.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(lbl, fontsize=8)
        ax.tick_params(labelsize=6); ax2.tick_params(labelsize=6)
    for idx in range(len(PAIRS), nrows*ncols):
        axes[idx//ncols][idx%ncols].axis("off")
    fig.suptitle("Fig F: Pairwise Δd′_geom vs Δacc_D1", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "figF_pairwise.pdf", dpi=200)
    fig.savefig(fig_dir / "figF_pairwise.png", dpi=150)
    plt.close(fig)

    # Fig G — noise sensitivity
    gby_p = {float(r["L"]): r for r in geom_rows if r.get("noise_model") == "poisson"}
    dag_iso = np.array([
        _f(arm_a_by_lm.get(lm, {}).get("isotropic", {}).get("delta_acc_geom_mean_pairwise"))
        for lm in Ls
    ])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.plot(Ls, dag, "o-", color=C1, lw=2, label="Poisson Σ_int")
    ax.plot(Ls, dag_iso, "^--", color="green", lw=2, label="Isotropic Σ_int")
    ax.set_xlabel("LogMAR"); ax.set_ylabel("ΔAcc_geom")
    ax.set_title("Fig G: Intrinsic-noise sensitivity (Poisson vs isotropic)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figG_noise_sensitivity.pdf", dpi=200)
    fig.savefig(fig_dir / "figG_noise_sensitivity.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keystone geometry crossover v3")
    p.add_argument("--logmar-grid", nargs="+", type=float, default=list(DEFAULT_LOGMAR_GRID))
    p.add_argument("--render-limit", type=float, default=DEFAULT_RENDER_LIMIT)
    p.add_argument("--core-range", nargs=2, type=float, default=list(DEFAULT_CORE_RANGE))
    p.add_argument("--min-core-points", type=int, default=DEFAULT_MIN_CORE_POINTS)
    p.add_argument("--primary-window", type=int, default=DEFAULT_PRIMARY_WINDOW)
    p.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    p.add_argument("--n-mc", type=int, default=DEFAULT_N_MC,
                   help="Monte Carlo samples per class for ideal-observer accuracy")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--specificity-delta-r2", type=float, default=DEFAULT_SPECIFICITY_DELTA_R2)
    p.add_argument("--pairwise-meaningful-delta", type=float, default=0.02,
                   help="Minimum |delta| threshold for counting a pairwise crossover cell as meaningful.")
    p.add_argument("--rvec-dirs", nargs="+", default=list(DEFAULT_RVEC_DIRS))
    p.add_argument("--d1-sweep-path", type=str, default=str(DEFAULT_D1_SWEEP_PATH))
    p.add_argument("--d1-sweep-paths", nargs="+", default=None,
                   help="Optional list of D1 sweep CSVs to merge (later rows overwrite earlier keys).")
    p.add_argument("--d1-sweep-col", type=str, default=DEFAULT_D1_SWEEP_COL)
    p.add_argument("--d1-sweep-readout", type=str, default=DEFAULT_D1_SWEEP_READOUT)
    p.add_argument("--feature-repr", type=str, default=DEFAULT_FEATURE_REPR)
    p.add_argument("--mcfarland-outputs", type=str, default=str(DEFAULT_MCFARLAND_OUTPUTS))
    p.add_argument("--expected-session", type=str, default=DEFAULT_EXPECTED_SESSION)
    p.add_argument("--expected-readout-units", type=int, default=DEFAULT_EXPECTED_READOUT_UNITS)
    p.add_argument("--calibration-mode", choices=["anchor", "loo"],
                   default=DEFAULT_CALIBRATION_MODE)
    p.add_argument("--calibration-anchor-logmar", type=float,
                   default=DEFAULT_CALIBRATION_ANCHOR_LOGMAR)
    p.add_argument("--calibration-max-mae", type=float, default=DEFAULT_CALIBRATION_MAX_MAE)
    p.add_argument("--enforce-calibration-gate", action="store_true",
                   help="Fail the run if observer calibration MAE threshold is not met.")
    p.add_argument("--jac-dir", type=str, default=str(DEFAULT_JAC_DIR))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--run-label", type=str, default="keystone_v3")
    p.add_argument("--cache-audit-only", action="store_true")
    args = p.parse_args()
    args.logmar_grid = tuple(float(x) for x in args.logmar_grid)
    args.core_range = tuple(float(x) for x in args.core_range)
    args.windows = tuple(int(x) for x in args.windows)
    args.rvec_dirs = tuple(str(s) for s in args.rvec_dirs)
    return args


def main() -> None:
    t0 = time.time()
    args = _parse_args()
    rng = np.random.default_rng(args.random_seed)

    out_dir = Path(args.out_dir)
    for sub in ("figures", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"Keystone geometry crossover v3 | run_label={args.run_label}")
    print(f"LogMAR grid: {args.logmar_grid}")
    print(f"Core range: {args.core_range}, min_core_pts={args.min_core_points}")
    print(f"n_mc={args.n_mc}, n_bootstrap={args.n_bootstrap}")

    merged, merged_tidx, acc_data, confirmed_d1, has_win = step0_cache_audit(args, out_dir)

    if args.cache_audit_only:
        print("[--cache-audit-only] Done.")
        return

    # Compute global r_bar for noise model
    r_bar = _compute_r_bar(merged, args.feature_repr)
    print(f"  r_bar (mean rate per neuron): {r_bar.round(4)}")

    all_test_rows: list[dict] = []

    # Step 0.5: observer calibration gate (pre-production)
    calib_label, best_lambda, calib_rows = step_observer_calibration(
        args, merged, r_bar, out_dir
    )
    all_test_rows += calib_rows
    print(f"  Observer calibration label: {calib_label} (lambda={best_lambda:.4g})")

    # Step 1
    geom_rows, geom_pair_rows, arm_a_by_lm = step1_arm_a(
        args, merged, merged_tidx, r_bar, out_dir, rng)

    # Step 2
    func_rows, func_pair_rows = step2_function_curves(
        args, acc_data, confirmed_d1, merged, out_dir, rng)

    # Step 2b
    readout_rows, readout_summary = step2b_readout_validation(
        args, merged, merged_tidx, geom_rows, out_dir
    )
    all_test_rows += readout_rows
    print(
        "  Readout validation: "
        f"{readout_summary['validation_label']} "
        f"(best_model={readout_summary['best_model']}, "
        f"abs_mae={readout_summary['best_abs_mae']:.4f}, "
        f"delta_mae={readout_summary['best_delta_mae']:.4f})"
    )

    # Count ready core points (use primary observable)
    core_ready = [lm for lm in args.logmar_grid
                  if (args.core_range[0]-1e-4) <= lm <= (args.core_range[1]+1e-4)
                  and abs(lm - args.render_limit) > 1e-4
                  and any(np.isfinite(_f(r.get("delta_acc_geom_mean_pairwise")))
                          for r in geom_rows
                          if abs(float(r["L"])-lm) < 1e-4 and r.get("noise_model")=="poisson")]
    n_core_ready = len(core_ready)
    print(f"  Core points with valid ΔAcc_geom: {n_core_ready}/{args.min_core_points} required")

    # Steps 3–5
    coin_rows, coincidence, render_confounded = step3_coincidence(
        args, geom_rows, func_rows, n_core_ready, rng)
    all_test_rows += coin_rows

    cont_rows, cont_sig = step4_continuous(args, geom_rows, func_rows, n_core_ready, rng)
    all_test_rows += cont_rows

    spec_rows, spec_passed = step5_specificity(args, geom_rows, func_rows, n_core_ready, rng)
    all_test_rows += spec_rows

    # Steps 6–7
    all_test_rows += step6_arm_b(args, out_dir)
    all_test_rows += step7_tier2(args)

    # Step 8
    sens_rows, sign_stable = step8_noise_sensitivity(
        args, arm_a_by_lm, func_rows, out_dir, rng)
    all_test_rows += sens_rows

    # Step 9
    pw_rows, n_pairs_passing = step9_pairwise(
        args, merged, merged_tidx, arm_a_by_lm, func_pair_rows, out_dir, rng
    )
    all_test_rows += pw_rows

    # Step 10
    null_rows = step10_nulls(
        args, merged, merged_tidx, r_bar, geom_rows, func_rows, rng)
    all_test_rows += null_rows

    # Step 11
    win_rows, window_stable = step11_window_robustness(
        args, geom_rows, func_rows, n_core_ready)
    all_test_rows += win_rows

    # Extract no_geom_crossing flag
    no_geom_crossing = any(
        r.get("statistic") == "no_geom_crossing" and np.isfinite(_f(r.get("value", 0)))
        and int(_f(r.get("value", 0))) == 1
        for r in all_test_rows
    )
    # nulls_passed extracted inside step12_output from test_rows
    nulls_passed_main = False  # placeholder; step12 re-extracts from rows
    mechanism = False  # Tier-2 deferred

    # Step 12
    step12_output(
        args, geom_rows, func_rows, all_test_rows,
        coincidence, cont_sig, spec_passed,
        mechanism, nulls_passed_main, window_stable, sign_stable,
        n_core_ready, no_geom_crossing, render_confounded,
        n_pairs_passing, confirmed_d1, out_dir
    )

    # Figures
    _make_figures(args, geom_rows, func_rows, all_test_rows, arm_a_by_lm,
                  out_dir / "figures")

    print(f"\nDone in {time.time()-t0:.1f}s. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
