#!/usr/bin/env python3
"""
E-optotype Jacobian field and identity-vector smoothness diagnostics.

Measures:
  1. J(p) subspace alignment as a function of phase separation -> decorrelation length ell_J
  2. d_ab(p) identity-vector cosines as a function of phase separation
  3. Anisotropy of smoothness: radial vs x-axis vs y-axis
  4. Region stratification for identity vectors: near_center / off_center / cross

Scientific context:
  The step-Jacobian result proved local differentiability over drift-sized steps,
  but does NOT imply J(p) remains similar as p moves across the drift cloud.
  This analysis directly measures the Jacobian field decorrelation length and
  distinguishes which mechanism drives the signed-sum cancellation seen in the
  path-integration analysis:

  Case A: smooth J, stable d_ab -> bad reference axis / readout issue
  Case B: smooth J, rotating d_ab -> phase-dependent identity geometry
  Case C: rough J, rotating d_ab -> curved local-field framework
  Case D: ell_J > drift_step_RMS but < drift_cloud_RMS -> local patches reconcile
           step-prediction and trajectory-sum cancellation (likely outcome)

FD-step validity note:
  Jacobian subspace alignment is only reliable for phase distances > 2 × fd_step.
  For distances < 2 × fd_step, the two Jacobian estimates share overlapping FD
  windows and will be artificially similar. Summary rows include
  `reliable_distance_min_arcmin` and a per-bin reliability flag.

Usage:
  # Pilot run (single orientation, dense core only)
  python scripts/eoptotype_jacobian_field_smoothness.py \\
      --logmars=-0.20 --orientations=0 \\
      --grid-core-radius-arcmin=3.0 --grid-core-spacing-arcmin=0.25 \\
      --grid-outer-radius-arcmin=0.0 \\
      --output-dir=outputs/stats/eoptotype_jacobian_field_smoothness_pilot

  # Full run
  python scripts/eoptotype_jacobian_field_smoothness.py \\
      --logmars=-0.20,+0.20 --orientations=0,90,180,270 \\
      --output-dir=outputs/stats/eoptotype_jacobian_field_smoothness_full
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import STATS_DIR
from scripts.jacobian_predictive_framework.run_eoptotype_curvature_scale_match import (
    DEFAULT_N_LAGS,
    DEFAULT_PPD,
    EPS,
    EYE_TRACES_PATH,
    CurvatureScaleMatchRunner,
    _load_eye_traces,
    _parse_csv_floats,
    _parse_csv_ints,
    _pick_device,
    _px_to_arcmin,
    _write_csv,
)
from scripts.fem_step_jacobian_prediction import (
    _prepare_trace_positions,
    _valid_adjacent_step_start_indices,
    _step_arcmin_values_from_indices,
)


DEFAULT_OUTPUT_DIR = STATS_DIR / "eoptotype_jacobian_field_smoothness"
DEFAULT_LOGMARS = (-0.20, 0.20)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_GRID_CORE_RADIUS_ARCMIN = 3.0
DEFAULT_GRID_CORE_SPACING_ARCMIN = 0.25
DEFAULT_GRID_OUTER_RADIUS_ARCMIN = 10.0
DEFAULT_GRID_OUTER_SPACING_ARCMIN = 0.5
# FD step must be < grid_spacing_px / 2 for non-overlapping windows between
# adjacent grid points.  At DEFAULT_PPD = 37.5, 0.25 arcmin = 0.156 px, so
# the maximum overlap-free step is ~0.078 px.  We use 0.1 px (slightly above
# that boundary) as a practical compromise: windows overlap minimally for the
# closest pairs and are non-overlapping for all pairs > ~0.32 arcmin.
DEFAULT_JACOBIAN_STEP_PX = 0.1
DEFAULT_MODEL_BATCH_SIZE = 64
DEFAULT_NEAR_CENTER_RADIUS_ARCMIN = 1.0
DEFAULT_NORM_PRODUCT_MIN = 1e-8
DEFAULT_DRIFT_MAX_ARCMIN = 1.0  # step threshold for "drift" classification

# Distance bins for decorrelation summary (arcmin)
DISTANCE_BIN_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf)
DISTANCE_BIN_LABELS = (
    "0_0p25_arcmin",
    "0p25_0p5_arcmin",
    "0p5_1_arcmin",
    "1_2_arcmin",
    "2_4_arcmin",
    "4_8_arcmin",
)
ALIGNMENT_THRESHOLDS = (0.90, 0.75, 0.50)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    total = max(int(round(float(seconds))), 0)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _arcmin_to_px(value_arcmin: float, pixels_per_degree: float) -> float:
    return float(value_arcmin) / 60.0 * float(pixels_per_degree)


def _pos_key(pos_px: np.ndarray) -> tuple[float, float]:
    return (round(float(pos_px[0]), 6), round(float(pos_px[1]), 6))


# ---------------------------------------------------------------------------
# Drift statistics from eye traces
# ---------------------------------------------------------------------------

def _compute_drift_stats(
    traces_path: Path,
    pixels_per_degree: float,
    drift_max_arcmin: float,
) -> dict[str, float]:
    """
    Load eye traces and compute:
      drift_step_rms_arcmin   - RMS of adjacent steps classified as drift
      drift_step_p90_arcmin   - 90th percentile of drift steps
      drift_cloud_rms_arcmin  - RMS of phase positions during drift-step segments
    """
    if not traces_path.exists():
        return {
            "drift_step_rms_arcmin": float("nan"),
            "drift_step_p90_arcmin": float("nan"),
            "drift_cloud_rms_arcmin": float("nan"),
        }

    traces_deg, durations = _load_eye_traces(traces_path)
    all_drift_steps: list[float] = []
    all_drift_positions_px: list[np.ndarray] = []

    for trace_id in range(len(durations)):
        trace_deg = traces_deg[trace_id, : int(durations[trace_id])]
        trace_px, finite_mask = _prepare_trace_positions(
            trace_deg.astype(np.float64), pixels_per_degree, "subtract_trace_mean"
        )
        valid_adj = _valid_adjacent_step_start_indices(finite_mask)
        if valid_adj.size == 0:
            continue
        step_arcmin = _step_arcmin_values_from_indices(trace_px, valid_adj, pixels_per_degree)
        drift_mask = step_arcmin <= float(drift_max_arcmin)
        all_drift_steps.extend(step_arcmin[drift_mask].tolist())
        # Collect start positions for drift steps as cloud sample
        for idx in valid_adj[drift_mask]:
            if finite_mask[idx]:
                all_drift_positions_px.append(trace_px[idx])

    if not all_drift_steps:
        return {
            "drift_step_rms_arcmin": float("nan"),
            "drift_step_p90_arcmin": float("nan"),
            "drift_cloud_rms_arcmin": float("nan"),
        }

    step_arr = np.asarray(all_drift_steps, dtype=np.float64)
    drift_step_rms = float(np.sqrt(np.mean(step_arr * step_arr)))
    drift_step_p90 = float(np.nanquantile(step_arr, 0.90))

    pos_arr = np.stack(all_drift_positions_px, axis=0)  # (N, 2) in px
    pos_arcmin = pos_arr * 60.0 / float(pixels_per_degree)
    cloud_rms = float(np.sqrt(np.mean(pos_arcmin ** 2)))

    return {
        "drift_step_rms_arcmin": drift_step_rms,
        "drift_step_p90_arcmin": drift_step_p90,
        "drift_cloud_rms_arcmin": cloud_rms,
    }


# ---------------------------------------------------------------------------
# Phase grid construction
# ---------------------------------------------------------------------------

def _build_phase_grid(
    core_radius_arcmin: float,
    core_spacing_arcmin: float,
    outer_radius_arcmin: float,
    outer_spacing_arcmin: float,
    pixels_per_degree: float,
) -> np.ndarray:
    """Return (N, 2) array of grid positions in pixels (circular mask)."""
    positions: dict[tuple[float, float], np.ndarray] = {}

    def _add_grid(radius_arcmin: float, spacing_arcmin: float) -> None:
        if radius_arcmin <= 0.0 or spacing_arcmin <= 0.0:
            return
        radius_px = _arcmin_to_px(radius_arcmin, pixels_per_degree)
        spacing_px = _arcmin_to_px(spacing_arcmin, pixels_per_degree)
        n_steps = int(np.ceil(radius_px / spacing_px))
        coords_px = np.arange(-n_steps, n_steps + 1) * spacing_px
        for x in coords_px:
            for y in coords_px:
                p = np.array([x, y], dtype=np.float64)
                if np.linalg.norm(p) <= radius_px + 1e-9:
                    key = _pos_key(p)
                    if key not in positions:
                        positions[key] = p

    _add_grid(core_radius_arcmin, core_spacing_arcmin)
    _add_grid(outer_radius_arcmin, outer_spacing_arcmin)

    return np.stack(list(positions.values()), axis=0)


def _all_eval_positions(
    grid_px: np.ndarray,
    jacobian_step_px: float,
) -> list[np.ndarray]:
    """Grid positions + FD neighbors (±step in x and y), deduplicated."""
    seen: set[tuple[float, float]] = set()
    unique: list[np.ndarray] = []

    def _add(p: np.ndarray) -> None:
        k = _pos_key(p)
        if k not in seen:
            seen.add(k)
            unique.append(p)

    step = float(jacobian_step_px)
    offsets = np.array([[step, 0.0], [-step, 0.0], [0.0, step], [0.0, -step]], dtype=np.float64)

    for p in grid_px:
        _add(p)
        for off in offsets:
            _add(p + off)

    return unique


# ---------------------------------------------------------------------------
# Jacobian and subspace alignment
# ---------------------------------------------------------------------------

def _rank_trimmed_qr(J: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (Q_trimmed, rank) for J shape (N, k)."""
    if J.shape[1] == 0:
        return np.zeros((J.shape[0], 0), dtype=np.float64), 0
    rank = int(np.linalg.matrix_rank(J))
    if rank == 0:
        return np.zeros((J.shape[0], 0), dtype=np.float64), 0
    Q, _ = np.linalg.qr(J)
    return Q[:, :rank], rank


def _subspace_alignment(
    J_i: np.ndarray,
    J_j: np.ndarray,
) -> dict[str, float | int]:
    """
    Compute subspace alignment between Jacobians J_i and J_j (N_cells, 2).

    J_alignment_mean = trace(P_i P_j) / k, k=2.
    Individual principal angle cosines reported separately — mean can hide
    one-stable/one-rotating behavior.
    """
    Q_i, rank_i = _rank_trimmed_qr(J_i)
    Q_j, rank_j = _rank_trimmed_qr(J_j)

    nan_row: dict = {
        "rank_i": rank_i, "rank_j": rank_j,
        "J_alignment_mean": float("nan"),
        "J_principal_cos_1": float("nan"),
        "J_principal_cos_2": float("nan"),
        "J_principal_angle_1_deg": float("nan"),
        "J_principal_angle_2_deg": float("nan"),
        "J_col_x_cosine": float("nan"),
        "J_col_y_cosine": float("nan"),
    }
    if rank_i == 0 or rank_j == 0:
        return nan_row

    C = Q_i.T @ Q_j  # (rank_i, rank_j)
    _, S, _ = np.linalg.svd(C, full_matrices=False)
    S = np.clip(S, 0.0, 1.0)

    k = 2
    alignment_mean = float(np.sum(S ** 2)) / k

    cos_1 = float(S[0]) if S.size >= 1 else float("nan")
    cos_2 = float(S[1]) if S.size >= 2 else 0.0
    angle_1 = float(np.degrees(np.arccos(np.clip(cos_1, 0.0, 1.0)))) if np.isfinite(cos_1) else float("nan")
    angle_2 = float(np.degrees(np.arccos(np.clip(cos_2, 0.0, 1.0))))

    # Column-wise cosines — NOT rotation-invariant; interpret cautiously.
    norm_xi = np.linalg.norm(J_i[:, 0]) + EPS
    norm_xj = np.linalg.norm(J_j[:, 0]) + EPS
    norm_yi = np.linalg.norm(J_i[:, 1]) + EPS
    norm_yj = np.linalg.norm(J_j[:, 1]) + EPS
    col_x = float(np.dot(J_i[:, 0], J_j[:, 0]) / (norm_xi * norm_xj))
    col_y = float(np.dot(J_i[:, 1], J_j[:, 1]) / (norm_yi * norm_yj))

    return {
        "rank_i": rank_i, "rank_j": rank_j,
        "J_alignment_mean": alignment_mean,
        "J_principal_cos_1": cos_1,
        "J_principal_cos_2": cos_2,
        "J_principal_angle_1_deg": angle_1,
        "J_principal_angle_2_deg": angle_2,
        "J_col_x_cosine": col_x,
        "J_col_y_cosine": col_y,
    }


# ---------------------------------------------------------------------------
# Separation axis classification
# ---------------------------------------------------------------------------

def _separation_axis(
    dx_arcmin: float,
    dy_arcmin: float,
    axis_threshold_arcmin: float,
) -> str:
    ax = abs(dx_arcmin)
    ay = abs(dy_arcmin)
    if ax <= axis_threshold_arcmin and ay <= axis_threshold_arcmin:
        return "same_point"
    if ay <= axis_threshold_arcmin:
        return "x_axis"
    if ax <= axis_threshold_arcmin:
        return "y_axis"
    return "diagonal"


# ---------------------------------------------------------------------------
# Decorrelation length estimation
# ---------------------------------------------------------------------------

def _bin_center_arcmin(lo: float, hi: float) -> float:
    if np.isinf(hi):
        return float(lo)
    return 0.5 * (float(lo) + float(hi))


def _median_per_bin(
    distances: np.ndarray,
    values: np.ndarray,
    bin_edges: tuple,
) -> list[float | None]:
    medians = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (distances >= lo) & (distances < hi) & np.isfinite(values)
        medians.append(float(np.median(values[mask])) if np.any(mask) else None)
    return medians


def _find_decorrelation_length(
    bin_centers: list[float],
    medians: list[float | None],
    threshold: float,
) -> tuple[float, str]:
    """
    Interpolate to find distance where median alignment crosses `threshold`.
    Returns (ell, status) with status in {'resolved', 'below_min', 'above_max'}.
    """
    valid = [(c, m) for c, m in zip(bin_centers, medians) if m is not None]
    if not valid:
        return float("nan"), "below_min"
    if valid[0][1] < threshold:
        return float(valid[0][0]), "below_min"
    if valid[-1][1] >= threshold:
        return float("nan"), "above_max"
    for (c_lo, m_lo), (c_hi, m_hi) in zip(valid[:-1], valid[1:]):
        if m_lo >= threshold > m_hi:
            frac = (threshold - m_lo) / (m_hi - m_lo + 1e-15)
            return float(c_lo + frac * (c_hi - c_lo)), "resolved"
    return float("nan"), "above_max"


def _threshold_key(t: float) -> str:
    return f"{t:.2f}".replace(".", "p")


# ---------------------------------------------------------------------------
# Main per-condition analysis
# ---------------------------------------------------------------------------

def _run_condition(
    runner: CurvatureScaleMatchRunner,
    logmar: float,
    orientations: list[int],
    grid_px: np.ndarray,
    jacobian_step_px: float,
    pixels_per_degree: float,
    near_center_radius_arcmin: float,
    norm_product_min: float,
    axis_threshold_arcmin: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (pair_rows_J, pair_rows_id, jacobian_grid_rows).

    pair_rows_J : one row per (orientation, phase pair) with J alignment metrics
    pair_rows_id: one row per (identity pair, phase pair) with d_ab cosine metrics
    jacobian_grid_rows: per-grid-point Jacobian diagnostics
    """
    t0 = time.time()
    n_grid = len(grid_px)
    print(f"  logmar={logmar:+.2f} {len(orientations)} orientations, {n_grid} grid points")

    eval_positions = _all_eval_positions(grid_px, jacobian_step_px)
    print(f"    Evaluating {len(eval_positions)} unique positions per orientation ...")

    # Evaluate all orientations, store responses keyed by pos_key
    responses_by_ori: dict[int, dict[tuple[float, float], np.ndarray]] = {}
    for ori in orientations:
        t_ori = time.time()
        resp, _support = runner.evaluate_condition(logmar, ori, eval_positions)
        responses_by_ori[ori] = resp
        print(f"    ori={ori}: {len(resp)} responses in {_format_duration(time.time() - t_ori)}")

    # Compute Jacobians at each grid point for each orientation
    jacobians: dict[int, dict[tuple[float, float], np.ndarray]] = {}
    jacobian_grid_rows: list[dict] = []
    for ori in orientations:
        resp = responses_by_ori[ori]
        jac_map: dict[tuple[float, float], np.ndarray] = {}
        for p in grid_px:
            try:
                J = runner.finite_difference_jacobian(resp, p)
            except KeyError:
                # FD neighbor missing — should not happen if _all_eval_positions is correct
                N_cells = next(iter(resp.values())).shape[0]
                J = np.zeros((N_cells, 2), dtype=np.float64)
            jac_map[_pos_key(p)] = J
        jacobians[ori] = jac_map

        for p in grid_px:
            J = jac_map[_pos_key(p)]
            x_am = _px_to_arcmin(float(p[0]), pixels_per_degree)
            y_am = _px_to_arcmin(float(p[1]), pixels_per_degree)
            _, rank = _rank_trimmed_qr(J)
            jacobian_grid_rows.append({
                "logmar": logmar,
                "orientation": ori,
                "phase_x_arcmin": x_am,
                "phase_y_arcmin": y_am,
                "phase_radius_arcmin": _px_to_arcmin(float(np.linalg.norm(p)), pixels_per_degree),
                "rank": rank,
                "J_frobenius_norm": float(np.linalg.norm(J, "fro")),
                "J_col_x_norm": float(np.linalg.norm(J[:, 0])),
                "J_col_y_norm": float(np.linalg.norm(J[:, 1])),
            })

    # -----------------------------------------------------------------
    # Analysis 1: Jacobian field pairwise alignment
    # -----------------------------------------------------------------
    print(f"    Computing J pairwise alignment ({n_grid}x{n_grid}/2 pairs) ...")
    pair_rows_J: list[dict] = []
    grid_list = list(grid_px)
    for ori in orientations:
        jac_map = jacobians[ori]
        for ii in range(n_grid):
            pi = grid_list[ii]
            xi_am = _px_to_arcmin(float(pi[0]), pixels_per_degree)
            yi_am = _px_to_arcmin(float(pi[1]), pixels_per_degree)
            Ji = jac_map[_pos_key(pi)]
            for jj in range(ii + 1, n_grid):
                pj = grid_list[jj]
                xj_am = _px_to_arcmin(float(pj[0]), pixels_per_degree)
                yj_am = _px_to_arcmin(float(pj[1]), pixels_per_degree)
                Jj = jac_map[_pos_key(pj)]

                dx = xj_am - xi_am
                dy = yj_am - yi_am
                dist = float(np.sqrt(dx * dx + dy * dy))
                axis = _separation_axis(dx, dy, axis_threshold_arcmin)

                aln = _subspace_alignment(Ji, Jj)
                row: dict = {
                    "logmar": logmar,
                    "orientation": ori,
                    "phase_i_x_arcmin": xi_am,
                    "phase_i_y_arcmin": yi_am,
                    "phase_j_x_arcmin": xj_am,
                    "phase_j_y_arcmin": yj_am,
                    "delta_x_arcmin": dx,
                    "delta_y_arcmin": dy,
                    "phase_distance_arcmin": dist,
                    "separation_axis": axis,
                }
                row.update(aln)
                pair_rows_J.append(row)

    # -----------------------------------------------------------------
    # Analysis 2: Identity vector pairwise cosines
    # -----------------------------------------------------------------
    pair_rows_id: list[dict] = []
    identity_pairs = list(itertools.combinations(orientations, 2))
    if identity_pairs:
        print(f"    Computing identity vector cosines for {len(identity_pairs)} pairs ...")
        near_r_px = _arcmin_to_px(near_center_radius_arcmin, pixels_per_degree)

        for src_ori, tgt_ori in identity_pairs:
            resp_a = responses_by_ori[src_ori]
            resp_b = responses_by_ori[tgt_ori]
            jac_map_a = jacobians[src_ori]

            # Pre-compute d_ab and f_perp_ab at each grid point
            d_ab: dict[tuple, np.ndarray] = {}
            f_perp_ab: dict[tuple, np.ndarray] = {}
            for p in grid_px:
                k = _pos_key(p)
                ra = resp_a.get(k)
                rb = resp_b.get(k)
                if ra is None or rb is None:
                    continue
                d = (rb - ra).astype(np.float64)
                d_ab[k] = d
                J = jac_map_a[k]
                Q, rank = _rank_trimmed_qr(J)
                f_perp = d - Q[:, :rank] @ (Q[:, :rank].T @ d) if rank > 0 else d.copy()
                f_perp_ab[k] = f_perp

            for ii in range(n_grid):
                pi = grid_list[ii]
                ki = _pos_key(pi)
                if ki not in d_ab:
                    continue
                xi_am = _px_to_arcmin(float(pi[0]), pixels_per_degree)
                yi_am = _px_to_arcmin(float(pi[1]), pixels_per_degree)
                ri_near = np.linalg.norm(pi) <= near_r_px + 1e-9

                d_i = d_ab[ki]
                f_i = f_perp_ab[ki]
                norm_di = float(np.linalg.norm(d_i))
                norm_fi = float(np.linalg.norm(f_i))

                for jj in range(ii + 1, n_grid):
                    pj = grid_list[jj]
                    kj = _pos_key(pj)
                    if kj not in d_ab:
                        continue
                    xj_am = _px_to_arcmin(float(pj[0]), pixels_per_degree)
                    yj_am = _px_to_arcmin(float(pj[1]), pixels_per_degree)
                    rj_near = np.linalg.norm(pj) <= near_r_px + 1e-9

                    dx = xj_am - xi_am
                    dy = yj_am - yi_am
                    dist = float(np.sqrt(dx * dx + dy * dy))
                    axis = _separation_axis(dx, dy, axis_threshold_arcmin)

                    if ri_near and rj_near:
                        region = "near_center"
                    elif (not ri_near) and (not rj_near):
                        region = "off_center"
                    else:
                        region = "cross_center_offcenter"

                    d_j = d_ab[kj]
                    f_j = f_perp_ab[kj]
                    norm_dj = float(np.linalg.norm(d_j))
                    norm_fj = float(np.linalg.norm(f_j))

                    norm_prod = norm_di * norm_dj
                    norm_prod_flag = int(norm_prod < float(norm_product_min))

                    if norm_di > EPS and norm_dj > EPS:
                        id_cos = float(np.dot(d_i, d_j) / (norm_di * norm_dj))
                    else:
                        id_cos = float("nan")

                    norm_prod_f = norm_fi * norm_fj
                    if norm_fi > EPS and norm_fj > EPS:
                        perp_cos = float(np.dot(f_i, f_j) / (norm_fi * norm_fj))
                    else:
                        perp_cos = float("nan")

                    pair_rows_id.append({
                        "logmar": logmar,
                        "source_orientation": src_ori,
                        "target_orientation": tgt_ori,
                        "region_pair_type": region,
                        "phase_i_x_arcmin": xi_am,
                        "phase_i_y_arcmin": yi_am,
                        "phase_j_x_arcmin": xj_am,
                        "phase_j_y_arcmin": yj_am,
                        "delta_x_arcmin": dx,
                        "delta_y_arcmin": dy,
                        "phase_distance_arcmin": dist,
                        "separation_axis": axis,
                        "identity_norm_i": norm_di,
                        "identity_norm_j": norm_dj,
                        "identity_norm_product": norm_prod,
                        "identity_cosine": id_cos,
                        "identity_abs_cosine": abs(id_cos) if np.isfinite(id_cos) else float("nan"),
                        "identity_inner_product": float(np.dot(d_i, d_j)) if np.isfinite(id_cos) else float("nan"),
                        "perp_identity_norm_i": norm_fi,
                        "perp_identity_norm_j": norm_fj,
                        "perp_identity_norm_product": norm_prod_f,
                        "perp_identity_cosine": perp_cos,
                        "perp_identity_abs_cosine": abs(perp_cos) if np.isfinite(perp_cos) else float("nan"),
                        "perp_identity_inner_product": float(np.dot(f_i, f_j)) if np.isfinite(perp_cos) else float("nan"),
                        "norm_product_flag": norm_prod_flag,
                    })

    elapsed = _format_duration(time.time() - t0)
    print(f"    Done in {elapsed}. J pairs={len(pair_rows_J)}, id pairs={len(pair_rows_id)}")
    return pair_rows_J, pair_rows_id, jacobian_grid_rows


# ---------------------------------------------------------------------------
# Decorrelation summary computation
# ---------------------------------------------------------------------------

def _build_decorrelation_summary_J(
    pair_rows: list[dict],
    drift_step_stats: dict,
    jacobian_step_arcmin: float,
) -> list[dict]:
    """One row per (logmar, orientation, separation_axis), including radial (all pairs)."""
    from collections import defaultdict

    # Build groups for each specific axis AND for 'radial' (all pairs)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in pair_rows:
        key_specific = (row["logmar"], row["orientation"], row["separation_axis"])
        key_radial = (row["logmar"], row["orientation"], "radial")
        groups[key_specific].append(row)
        groups[key_radial].append(row)

    bin_edges = DISTANCE_BIN_EDGES
    bin_labels = DISTANCE_BIN_LABELS
    bin_centers = [_bin_center_arcmin(lo, hi) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    # Distance below which J alignment may be artificially inflated by FD window overlap.
    # Two windows ±h around p_i and p_j overlap when |p_i - p_j| < 2h.
    reliable_dist_min = 2.0 * float(jacobian_step_arcmin)

    summary_rows: list[dict] = []
    for (logmar, ori, axis), rows in sorted(groups.items()):
        distances = np.array([float(r["phase_distance_arcmin"]) for r in rows])
        alignments = np.array([float(r["J_alignment_mean"]) for r in rows])
        cos1 = np.array([float(r["J_principal_cos_1"]) for r in rows])
        cos2 = np.array([float(r["J_principal_cos_2"]) for r in rows])

        medians = _median_per_bin(distances, alignments, bin_edges)
        row_out: dict = {
            "logmar": logmar,
            "orientation": ori,
            "separation_axis": axis,
            "n_phase_pairs": len(rows),
            "fd_step_arcmin": jacobian_step_arcmin,
            "reliable_distance_min_arcmin": reliable_dist_min,
        }
        for label, med in zip(bin_labels, medians):
            row_out[f"median_alignment_{label}"] = med if med is not None else float("nan")

        for thresh in ALIGNMENT_THRESHOLDS:
            tkey = _threshold_key(thresh)
            ell, status = _find_decorrelation_length(bin_centers, medians, thresh)
            # Flag if estimate is unreliable due to FD overlap
            if status == "below_min" and (np.isnan(ell) or float(ell) < reliable_dist_min):
                status_annotated = f"below_min_or_fd_overlap"
            elif status == "resolved" and float(ell) < reliable_dist_min:
                status_annotated = f"resolved_but_fd_overlap_suspected"
            else:
                status_annotated = status
            row_out[f"ell_J_{tkey}_arcmin"] = ell
            row_out[f"ell_J_{tkey}_status"] = status_annotated

        # Principal angle medians per bin
        for label, (lo, hi) in zip(bin_labels, zip(bin_edges[:-1], bin_edges[1:])):
            mask = (distances >= lo) & (distances < hi)
            valid_mask = mask & np.isfinite(cos1) & np.isfinite(cos2)
            row_out[f"median_principal_cos_1_{label}"] = float(np.median(cos1[valid_mask])) if np.any(valid_mask) else float("nan")
            row_out[f"median_principal_cos_2_{label}"] = float(np.median(cos2[valid_mask])) if np.any(valid_mask) else float("nan")

        row_out.update(drift_step_stats)
        summary_rows.append(row_out)

    return summary_rows


def _build_decorrelation_summary_id(
    pair_rows: list[dict],
    jacobian_step_arcmin: float,
) -> list[dict]:
    """One row per (logmar, src_ori, tgt_ori, region_pair_type, separation_axis), including radial."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in pair_rows:
        key_specific = (
            row["logmar"], row["source_orientation"], row["target_orientation"],
            row["region_pair_type"], row["separation_axis"],
        )
        key_radial = (
            row["logmar"], row["source_orientation"], row["target_orientation"],
            row["region_pair_type"], "radial",
        )
        groups[key_specific].append(row)
        groups[key_radial].append(row)

    bin_edges = DISTANCE_BIN_EDGES
    bin_labels = DISTANCE_BIN_LABELS
    bin_centers = [_bin_center_arcmin(lo, hi) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    summary_rows: list[dict] = []
    for (logmar, src, tgt, region, axis), rows in sorted(groups.items()):
        distances = np.array([float(r["phase_distance_arcmin"]) for r in rows])
        id_cos = np.array([float(r["identity_cosine"]) for r in rows])
        id_abs = np.array([float(r["identity_abs_cosine"]) for r in rows])
        perp_cos = np.array([float(r["perp_identity_cosine"]) for r in rows])
        perp_abs = np.array([float(r["perp_identity_abs_cosine"]) for r in rows])
        norms_prod = np.array([float(r["identity_norm_product"]) for r in rows])
        flag = np.array([bool(r["norm_product_flag"]) for r in rows])

        valid = np.isfinite(id_cos) & ~flag
        valid_perp = valid & np.isfinite(perp_cos)

        medians_abs = _median_per_bin(distances[valid], id_abs[valid], bin_edges) if np.any(valid) else [None] * len(bin_labels)
        medians_perp_abs = _median_per_bin(distances[valid_perp], perp_abs[valid_perp], bin_edges) if np.any(valid_perp) else [None] * len(bin_labels)

        row_out: dict = {
            "logmar": logmar,
            "source_orientation": src,
            "target_orientation": tgt,
            "region_pair_type": region,
            "separation_axis": axis,
            "n_phase_pairs": len(rows),
            "n_phase_pairs_valid": int(np.sum(valid)),
            "median_identity_cosine": float(np.median(id_cos[valid])) if np.any(valid) else float("nan"),
            "median_identity_abs_cosine": float(np.median(id_abs[valid])) if np.any(valid) else float("nan"),
            "fraction_negative_identity_cosine": float(np.mean(id_cos[valid] < 0.0)) if np.any(valid) else float("nan"),
            "median_identity_inner_product": float(np.median(np.array([float(r["identity_inner_product"]) for r in rows])[valid])) if np.any(valid) else float("nan"),
            "median_perp_identity_cosine": float(np.median(perp_cos[valid_perp])) if np.any(valid_perp) else float("nan"),
            "median_perp_identity_abs_cosine": float(np.median(perp_abs[valid_perp])) if np.any(valid_perp) else float("nan"),
            "fraction_negative_perp_identity_cosine": float(np.mean(perp_cos[valid_perp] < 0.0)) if np.any(valid_perp) else float("nan"),
            "median_perp_identity_inner_product": float(np.median(np.array([float(r["perp_identity_inner_product"]) for r in rows])[valid_perp])) if np.any(valid_perp) else float("nan"),
        }

        for thresh in (0.75, 0.50):
            tkey = _threshold_key(thresh)
            ell_id, st_id = _find_decorrelation_length(bin_centers, medians_abs, thresh)
            ell_perp, st_perp = _find_decorrelation_length(bin_centers, medians_perp_abs, thresh)
            row_out[f"ell_identity_cosine_{tkey}_arcmin"] = ell_id
            row_out[f"ell_identity_cosine_{tkey}_status"] = st_id
            row_out[f"ell_perp_identity_cosine_{tkey}_arcmin"] = ell_perp
            row_out[f"ell_perp_identity_cosine_{tkey}_status"] = st_perp

        summary_rows.append(row_out)

    return summary_rows


# ---------------------------------------------------------------------------
# Interpretation matrix
# ---------------------------------------------------------------------------

def _interpret_case(
    ell_J_0p75: float,
    ell_J_0p75_status: str,
    ell_J_0p50_status: str,
    drift_step_rms: float,
    drift_cloud_rms: float,
    ell_id_0p75: float,
) -> str:
    """
    Case classification from the four pre-registered regimes.

    Key distinction: 'below_min' for ell_J_0.75 can mean either
    (a) the field decorrelates rapidly below the first resolved bin, OR
    (b) the field is smooth but alignment is capped below 0.75 at all distances.
    Case (b) is identified by ell_J_0.50_status == 'above_max' (alignment
    never drops below 0.50 either), indicating a trivially smooth field with
    moderate absolute alignment level.
    """
    # Check if the field is smooth even at the 0.50 threshold
    smooth_at_0p50 = ell_J_0p50_status == "above_max"

    rough_status = ("below_min", "below_min_or_fd_overlap", "resolved_but_fd_overlap_suspected")
    smooth_status = ("above_max",)

    J_unresolved_rough = (ell_J_0p75_status in rough_status) and not smooth_at_0p50
    J_smooth = (ell_J_0p75_status in smooth_status) or smooth_at_0p50 or (
        np.isfinite(ell_J_0p75) and ell_J_0p75 > drift_cloud_rms
    )
    J_cloud_scale = (
        not J_smooth and not J_unresolved_rough
        and np.isfinite(ell_J_0p75)
        and np.isfinite(drift_step_rms)
        and ell_J_0p75 > drift_step_rms
    )

    id_stable = np.isnan(ell_id_0p75) or (
        np.isfinite(ell_id_0p75) and ell_id_0p75 > drift_cloud_rms
    )

    if J_unresolved_rough:
        return "below_resolution_increase_fd_step_or_grid"
    if J_smooth and id_stable:
        return "A_smooth_J_stable_id"
    if J_smooth and not id_stable:
        return "B_smooth_J_rotating_id"
    if J_cloud_scale and not id_stable:
        return "C_rough_J_rotating_id"
    if J_cloud_scale:
        return "D_step_smooth_cloud_rough"
    return "unclassified"


def _build_interpretation_summary(
    summary_J: list[dict],
    summary_id: list[dict],
    drift_step_rms: float,
    drift_cloud_rms: float,
) -> list[dict]:
    rows = []
    for jrow in summary_J:
        if jrow["separation_axis"] != "radial":
            continue
        logmar = jrow["logmar"]
        ori = jrow["orientation"]
        ell_J = float(jrow.get("ell_J_0p75_arcmin", float("nan")))
        ell_J_status = jrow.get("ell_J_0p75_status", "")
        ell_J_0p50_status = jrow.get("ell_J_0p50_status", "")

        ell_ids: list[float] = []
        for irow in summary_id:
            if float(irow["logmar"]) != float(logmar):
                continue
            if irow["separation_axis"] != "radial" or irow["region_pair_type"] != "off_center":
                continue
            ell_ids.append(float(irow.get("ell_identity_cosine_0p75_arcmin", float("nan"))))

        ell_id = float(np.nanmedian(ell_ids)) if ell_ids else float("nan")
        case = _interpret_case(ell_J, ell_J_status, ell_J_0p50_status, drift_step_rms, drift_cloud_rms, ell_id)

        rows.append({
            "logmar": logmar,
            "orientation": ori,
            "ell_J_0p75_arcmin": ell_J,
            "ell_J_0p75_status": ell_J_status,
            "ell_J_0p90_arcmin": jrow.get("ell_J_0p90_arcmin", float("nan")),
            "ell_J_0p50_arcmin": jrow.get("ell_J_0p50_arcmin", float("nan")),
            "median_ell_id_0p75_arcmin_offcenter": ell_id,
            "fd_step_arcmin": jrow.get("fd_step_arcmin", float("nan")),
            "reliable_distance_min_arcmin": jrow.get("reliable_distance_min_arcmin", float("nan")),
            "drift_step_rms_arcmin": drift_step_rms,
            "drift_cloud_rms_arcmin": drift_cloud_rms,
            "case": case,
        })

    return rows


# ---------------------------------------------------------------------------
# Pre-registered predictions
# ---------------------------------------------------------------------------

PREDICTIONS_MD = """\
# Pre-registered predictions: Jacobian field smoothness

Written before running. After running, compare results to the three scenarios
and document which scenario the data support.

## Scenario 1: Pure complex-cell pooling

Expected:
- ell_J ~ 1-3 arcmin (decorrelation at subunit / pooling scale)
- ell_J similar across LogMARs (pooling scale is fixed by the model)
- Primary principal angle (cos_1) decays more slowly than secondary (cos_2)
- Some x vs y anisotropy depending on stroke orientation

Predicted case: C or D depending on relationship to drift-step RMS

## Scenario 2: Curved-field / fine-feature framework

Expected:
- ell_J < 1 arcmin, scaling with stimulus size
- ell_J shorter at LogMAR -0.20 than at LogMAR +0.20
- Identity-vector decorrelation stronger at LogMAR -0.20
- If ell_J < drift_step_RMS: sub-step roughness (Regime 4)
- If drift_step_RMS < ell_J < drift_cloud_RMS: Case D (local patches)

Predicted case: C, D, or below_resolution

## Scenario 3: Trivially smooth field

Expected:
- ell_J >> drift cloud radius at both LogMARs
- J alignment remains high across all tested phase distances
- Signed-sum cancellation must come from identity-vector rotation or readout choice

Predicted case: A or B

## Pre-registered comparison values (from existing step-Jacobian analysis)
- drift_step_RMS ~ 0.35 arcmin (empirical, see prior runs)
- drift_cloud_RMS ~ computed from traces (see run_config.json)
- Grid core spacing: 0.25 arcmin
- FD step: see run_config.json — results reliable for distances > 2 × FD_step

## Notes
- Near-center identity cosines (within 1 arcmin of center at LogMAR -0.20) will
  be sparse due to the center-degeneracy finding (||d_ab(p)|| is very small at
  center). The norm_product_flag will exclude most near-center pairs.
- The 0-0.25 arcmin bin will be empty by construction (minimum pair distance =
  grid spacing = 0.25 arcmin). The first populated bin is 0.25-0.5 arcmin.
"""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_alignment_scatter(
    pair_rows_J: list[dict],
    logmar: float,
    ori: int,
    figures_dir: Path,
) -> None:
    rows = [r for r in pair_rows_J
            if float(r["logmar"]) == float(logmar) and int(r["orientation"]) == int(ori)]
    if not rows:
        return

    distances = np.array([float(r["phase_distance_arcmin"]) for r in rows])
    alignments = np.array([float(r["J_alignment_mean"]) for r in rows])
    cos1 = np.array([float(r["J_principal_cos_1"]) for r in rows])

    bin_edges = DISTANCE_BIN_EDGES
    bin_centers = [_bin_center_arcmin(lo, hi) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    med_aln = _median_per_bin(distances, alignments, bin_edges)
    med_c1 = _median_per_bin(distances, cos1, bin_edges)

    xs = [c for c, m in zip(bin_centers, med_aln) if m is not None and np.isfinite(m)]
    ys_aln = [m for m in med_aln if m is not None and np.isfinite(m)]
    ys_c1 = [m for c, m in zip(bin_centers, med_c1) if m is not None and np.isfinite(m) and c in xs]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    ax = axes[0]
    ax.scatter(distances, alignments, s=1, alpha=0.1, color="tab:blue", rasterized=True)
    if xs:
        ax.plot(xs, ys_aln, "o-", color="tab:red", lw=2, zorder=5, label="median")
    ax.axhline(0.75, color="gray", ls="--", lw=1)
    ax.set_xlabel("Phase separation (arcmin)")
    ax.set_ylabel("J subspace alignment")
    ax.set_title(f"J alignment\nLogMAR={logmar:+.2f}, ori={ori}")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.05)

    ax = axes[1]
    ax.scatter(distances, cos1, s=1, alpha=0.1, color="tab:orange", rasterized=True)
    if xs:
        ax.plot(xs, ys_c1, "o-", color="tab:red", lw=2, zorder=5, label="median cos_1")
    ax.axhline(0.75, color="gray", ls="--", lw=1)
    ax.set_xlabel("Phase separation (arcmin)")
    ax.set_ylabel("J primary principal cosine (cos_1)")
    ax.set_title(f"Primary direction stability\nLogMAR={logmar:+.2f}, ori={ori}")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(figures_dir / f"jacobian_field_alignment_scatter_lm{logmar:+.2f}_ori{ori}.png", dpi=150)
    plt.close(fig)


def _plot_principal_cosines_vs_distance(
    pair_rows_J: list[dict],
    logmar: float,
    ori: int,
    figures_dir: Path,
) -> None:
    rows = [r for r in pair_rows_J
            if float(r["logmar"]) == float(logmar) and int(r["orientation"]) == int(ori)]
    if not rows:
        return

    distances = np.array([float(r["phase_distance_arcmin"]) for r in rows])
    cos1 = np.array([float(r["J_principal_cos_1"]) for r in rows])
    cos2 = np.array([float(r["J_principal_cos_2"]) for r in rows])

    bin_edges = DISTANCE_BIN_EDGES
    bin_centers = [_bin_center_arcmin(lo, hi) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    med1 = _median_per_bin(distances, cos1, bin_edges)
    med2 = _median_per_bin(distances, cos2, bin_edges)

    xs1 = [c for c, m in zip(bin_centers, med1) if m is not None and np.isfinite(m)]
    ys1 = [m for m in med1 if m is not None and np.isfinite(m)]
    xs2 = [c for c, m in zip(bin_centers, med2) if m is not None and np.isfinite(m)]
    ys2 = [m for m in med2 if m is not None and np.isfinite(m)]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if xs1:
        ax.plot(xs1, ys1, "o-", color="tab:blue", lw=1.5, label="cos(theta_1) — primary")
    if xs2:
        ax.plot(xs2, ys2, "s--", color="tab:orange", lw=1.5, label="cos(theta_2) — secondary")
    ax.axhline(0.75, color="gray", ls=":", lw=1)
    ax.set_xlabel("Phase separation (arcmin)")
    ax.set_ylabel("Median principal angle cosine")
    ax.set_title(f"Jacobian principal cosines\nLogMAR={logmar:+.2f}, ori={ori}")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(figures_dir / f"jacobian_principal_cosines_lm{logmar:+.2f}_ori{ori}.png", dpi=150)
    plt.close(fig)


def _plot_alignment_by_axis(
    summary_J: list[dict],
    logmar: float,
    ori: int,
    figures_dir: Path,
) -> None:
    rows_by_axis = {ax: [r for r in summary_J
                         if float(r["logmar"]) == float(logmar)
                         and int(r["orientation"]) == int(ori)
                         and r["separation_axis"] == ax]
                    for ax in ("radial", "x_axis", "y_axis")}

    bin_edges = DISTANCE_BIN_EDGES
    bin_labels = DISTANCE_BIN_LABELS
    bin_centers = [_bin_center_arcmin(lo, hi) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    colors = {"radial": "tab:blue", "x_axis": "tab:orange", "y_axis": "tab:green"}

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for axis_name, color in colors.items():
        rows = rows_by_axis.get(axis_name, [])
        if not rows:
            continue
        row = rows[0]
        medians = [row.get(f"median_alignment_{lbl}") for lbl in bin_labels]
        xs = [c for c, m in zip(bin_centers, medians) if m is not None and np.isfinite(float(m))]
        ys = [float(m) for m in medians if m is not None and np.isfinite(float(m))]
        if xs:
            ax.plot(xs, ys, "o-", color=color, lw=1.5, label=axis_name)

    ax.axhline(0.75, color="gray", ls="--", lw=1, label="0.75 threshold")
    ax.set_xlabel("Phase separation (arcmin)")
    ax.set_ylabel("Median J subspace alignment")
    ax.set_title(f"Jacobian alignment by axis\nLogMAR={logmar:+.2f}, ori={ori}")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    for axis_name in ("radial", "x_axis", "y_axis"):
        fig.savefig(figures_dir / f"jacobian_field_alignment_{axis_name}_lm{logmar:+.2f}_ori{ori}.png", dpi=150)
    plt.close(fig)


def _plot_ell_J_by_logmar(
    summary_J: list[dict],
    orientations: list[int],
    figures_dir: Path,
) -> None:
    logmars = sorted(set(float(r["logmar"]) for r in summary_J))
    radial = [r for r in summary_J if r["separation_axis"] == "radial"]
    if not radial or not logmars:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ori, color in zip(orientations, colors):
        ells = []
        for lm in logmars:
            rows = [r for r in radial if float(r["logmar"]) == lm and int(r["orientation"]) == int(ori)]
            ell = float(rows[0]["ell_J_0p75_arcmin"]) if rows else float("nan")
            ells.append(ell if np.isfinite(ell) else None)
        xs = [lm for lm, e in zip(logmars, ells) if e is not None]
        ys = [e for e in ells if e is not None]
        if xs:
            ax.plot(xs, ys, "o-", color=color, lw=1.5, label=f"ori={ori}")

    ax.set_xlabel("LogMAR")
    ax.set_ylabel("ell_J_0.75 (arcmin)")
    ax.set_title("Jacobian field decorrelation length by LogMAR")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "jacobian_field_decorrelation_length_by_logmar.png", dpi=150)
    plt.close(fig)


def _make_figures(
    all_pair_rows_J: list[dict],
    all_pair_rows_id: list[dict],
    summary_J: list[dict],
    summary_id: list[dict],
    logmars: list[float],
    orientations: list[int],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for logmar in logmars:
        for ori in orientations:
            _plot_alignment_scatter(all_pair_rows_J, logmar, ori, figures_dir)
            _plot_principal_cosines_vs_distance(all_pair_rows_J, logmar, ori, figures_dir)
            _plot_alignment_by_axis(summary_J, logmar, ori, figures_dir)
    _plot_ell_J_by_logmar(summary_J, orientations, figures_dir)


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logmars", type=str,
                        default=",".join(str(lm) for lm in DEFAULT_LOGMARS))
    parser.add_argument("--orientations", type=str,
                        default=",".join(str(o) for o in DEFAULT_ORIENTATIONS))
    parser.add_argument("--grid-core-radius-arcmin", type=float,
                        default=DEFAULT_GRID_CORE_RADIUS_ARCMIN)
    parser.add_argument("--grid-core-spacing-arcmin", type=float,
                        default=DEFAULT_GRID_CORE_SPACING_ARCMIN)
    parser.add_argument("--grid-outer-radius-arcmin", type=float,
                        default=DEFAULT_GRID_OUTER_RADIUS_ARCMIN)
    parser.add_argument("--grid-outer-spacing-arcmin", type=float,
                        default=DEFAULT_GRID_OUTER_SPACING_ARCMIN)
    parser.add_argument("--jacobian-step-px", type=float, default=DEFAULT_JACOBIAN_STEP_PX)
    parser.add_argument("--model-batch-size", type=int, default=DEFAULT_MODEL_BATCH_SIZE)
    parser.add_argument("--near-center-radius-arcmin", type=float,
                        default=DEFAULT_NEAR_CENTER_RADIUS_ARCMIN)
    parser.add_argument("--norm-product-min", type=float, default=DEFAULT_NORM_PRODUCT_MIN)
    parser.add_argument("--drift-max-arcmin", type=float, default=DEFAULT_DRIFT_MAX_ARCMIN)
    parser.add_argument("--pixels-per-degree", type=float, default=DEFAULT_PPD)
    parser.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    parser.add_argument("--eye-traces-path", type=Path, default=EYE_TRACES_PATH)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running model")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    t_main = time.time()

    logmars = list(_parse_csv_floats(args.logmars))
    orientations = [int(o) for o in _parse_csv_ints(args.orientations)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pixels_per_degree = float(args.pixels_per_degree)
    jacobian_step_px = float(args.jacobian_step_px)
    jacobian_step_arcmin = _px_to_arcmin(jacobian_step_px, pixels_per_degree)
    reliable_dist_min = 2.0 * jacobian_step_arcmin
    axis_threshold_arcmin = float(args.grid_core_spacing_arcmin) / 2.0

    # Build phase grid
    grid_px = _build_phase_grid(
        core_radius_arcmin=args.grid_core_radius_arcmin,
        core_spacing_arcmin=args.grid_core_spacing_arcmin,
        outer_radius_arcmin=args.grid_outer_radius_arcmin,
        outer_spacing_arcmin=args.grid_outer_spacing_arcmin,
        pixels_per_degree=pixels_per_degree,
    )
    n_grid = len(grid_px)
    n_eval = len(_all_eval_positions(grid_px, jacobian_step_px))
    n_pairs_J = n_grid * (n_grid - 1) // 2

    # Compute drift statistics from actual eye traces
    print("Computing drift statistics from eye traces ...")
    drift_step_stats = _compute_drift_stats(
        args.eye_traces_path, pixels_per_degree, args.drift_max_arcmin
    )
    drift_step_rms = drift_step_stats["drift_step_rms_arcmin"]
    drift_cloud_rms = drift_step_stats["drift_cloud_rms_arcmin"]
    print(f"  drift_step_rms={drift_step_rms:.3f} arcmin  "
          f"drift_cloud_rms={drift_cloud_rms:.3f} arcmin")

    print(f"Grid: {n_grid} points, {n_eval} eval positions, ~{n_pairs_J} J pairs per condition")
    print(f"FD step: {jacobian_step_px} px = {jacobian_step_arcmin:.4f} arcmin  "
          f"(reliable distances > {reliable_dist_min:.4f} arcmin)")
    print(f"Conditions: {len(logmars)} logmars × {len(orientations)} orientations")

    # Write run config
    run_config = {
        "logmars": logmars,
        "orientations": orientations,
        "grid_core_radius_arcmin": args.grid_core_radius_arcmin,
        "grid_core_spacing_arcmin": args.grid_core_spacing_arcmin,
        "grid_outer_radius_arcmin": args.grid_outer_radius_arcmin,
        "grid_outer_spacing_arcmin": args.grid_outer_spacing_arcmin,
        "n_phase_points": n_grid,
        "n_forward_positions": n_eval,
        "n_pairs_J_per_condition": n_pairs_J,
        "finite_difference_step_px": jacobian_step_px,
        "finite_difference_step_arcmin": jacobian_step_arcmin,
        "reliable_distance_min_arcmin": reliable_dist_min,
        "near_center_radius_arcmin": args.near_center_radius_arcmin,
        "norm_product_min": args.norm_product_min,
        "drift_max_arcmin": args.drift_max_arcmin,
        "drift_step_rms_arcmin": drift_step_rms,
        "drift_step_p90_arcmin": drift_step_stats["drift_step_p90_arcmin"],
        "drift_cloud_rms_arcmin": drift_cloud_rms,
        "pixels_per_degree": pixels_per_degree,
        "n_lags": args.n_lags,
        "model_batch_size": args.model_batch_size,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # Write predictions before running
    (output_dir / "predictions.md").write_text(PREDICTIONS_MD)

    print(json.dumps(run_config, indent=2))

    if args.dry_run:
        print("Dry run — exiting.")
        return

    device = args.device if args.device else _pick_device()
    runner = CurvatureScaleMatchRunner(
        device=device,
        pixels_per_degree=pixels_per_degree,
        n_lags=args.n_lags,
        jacobian_step_px=jacobian_step_px,
        model_batch_size=args.model_batch_size,
    )

    all_pair_rows_J: list[dict] = []
    all_pair_rows_id: list[dict] = []
    all_jacobian_grid_rows: list[dict] = []

    for logmar in logmars:
        print(f"\n=== logmar={logmar:+.2f} ===")
        pair_rows_J, pair_rows_id, jac_grid_rows = _run_condition(
            runner=runner,
            logmar=logmar,
            orientations=orientations,
            grid_px=grid_px,
            jacobian_step_px=jacobian_step_px,
            pixels_per_degree=pixels_per_degree,
            near_center_radius_arcmin=args.near_center_radius_arcmin,
            norm_product_min=args.norm_product_min,
            axis_threshold_arcmin=axis_threshold_arcmin,
        )
        all_pair_rows_J.extend(pair_rows_J)
        all_pair_rows_id.extend(pair_rows_id)
        all_jacobian_grid_rows.extend(jac_grid_rows)

    print("\n=== Writing outputs ===")

    _write_csv(output_dir / "jacobian_field_alignment_by_phase_pair.csv", all_pair_rows_J)
    _write_csv(output_dir / "jacobian_grid_diagnostics.csv", all_jacobian_grid_rows)
    if all_pair_rows_id:
        _write_csv(output_dir / "identity_vector_alignment_by_phase_pair.csv", all_pair_rows_id)

    summary_J = _build_decorrelation_summary_J(all_pair_rows_J, drift_step_stats, jacobian_step_arcmin)
    _write_csv(output_dir / "jacobian_field_decorrelation_summary.csv", summary_J)

    if all_pair_rows_id:
        summary_id = _build_decorrelation_summary_id(all_pair_rows_id, jacobian_step_arcmin)
        _write_csv(output_dir / "identity_vector_decorrelation_summary.csv", summary_id)
    else:
        summary_id = []

    interp_rows = _build_interpretation_summary(
        summary_J, summary_id, drift_step_rms, drift_cloud_rms
    )
    _write_csv(output_dir / "field_smoothness_interpretation_summary.csv", interp_rows)

    _make_figures(all_pair_rows_J, all_pair_rows_id, summary_J, summary_id, logmars, orientations, figures_dir)

    total = _format_duration(time.time() - t_main)
    print(f"\nDone in {total}. Outputs: {output_dir}")

    print(f"\n--- Drift reference values ---")
    print(f"  drift_step_rms = {drift_step_rms:.3f} arcmin  (ell_J should exceed this for Case D)")
    print(f"  drift_cloud_rms = {drift_cloud_rms:.3f} arcmin  (ell_J should be below this for Case D)")
    print(f"  reliable_distance_min = {reliable_dist_min:.4f} arcmin  (2 × FD step)")

    print(f"\n--- Interpretation summary ---")
    for row in interp_rows:
        print(f"  logmar={row['logmar']:+.2f} ori={row['orientation']:3d}  "
              f"ell_J_0.75={row['ell_J_0p75_arcmin']} ({row['ell_J_0p75_status']})  "
              f"case={row['case']}")


if __name__ == "__main__":
    main()
