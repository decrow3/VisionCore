#!/usr/bin/env python3
"""
FEM path-integrated identity separability.

For each E-optotype identity pair (a, b), FEM trace, and sampling condition,
compute whether sampling multiple FEM phases provides complementary identity
information beyond repeated sampling at one stabilized phase.

Primary metrics:
  gain_vs_stabilized_repeated
    = dprime2_orthogonal_path / dprime2_orthogonal(stabilized_repeated, same T)
  gain_vs_stabilized_repeated_idspace
    = dprime2_orthogonal_path_idspace / dprime2_orthogonal_idspace(stabilized_repeated, same T)
    (restricted to the identity subspace U_id, leakage-controlled via center mode)

IMPORTANT — sampling-condition semantics under order-invariant readout:
  F = sum_t f_t is commutative: real_fem_path and phase_shuffled_path are EQUAL
  by construction.  phase_shuffled_path is a phase-set control, NOT an order
  control.  A true temporal-order control requires a lagged response model.

Usage:
  python scripts/fem_path_integrated_separability.py \\
      --logmars=-0.20,+0.20 --orientations=0,90,180,270 \\
      --max-traces=5 --T=60 \\
      --identity-subspace-mode=center \\
      --output-dir=outputs/stats/fem_path_integrated_separability_idspace
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from VisionCore.paths import STATS_DIR
from scripts.jacobian_predictive_framework.run_eoptotype_curvature_scale_match import (
    DEFAULT_N_LAGS,
    DEFAULT_PPD,
    EYE_TRACES_PATH,
    EPS,
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
    _step_arcmin_values_from_indices,
    _valid_adjacent_step_start_indices,
)


DEFAULT_OUTPUT_DIR = STATS_DIR / "fem_path_integrated_separability"
DEFAULT_LOGMARS = (-0.20, 0.20)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_T = 60
DEFAULT_MAX_TRACES = 5
DEFAULT_SIGMA0 = 1.0
DEFAULT_JACOBIAN_STEP_PX = 1.0
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_DRIFT_MAX_ARCMIN = 1.0
DEFAULT_CENTER_MODE = "raw"
DEFAULT_PATH_MODES = ("drift_step_start_phases", "all_steps")
DEFAULT_SAMPLING_CONDITIONS = ("real_fem_path", "stabilized_repeated", "phase_shuffled_path")
DEFAULT_ID_SUBSPACE_MODE = "center"
DEFAULT_ID_SUBSPACE_ENERGY = 0.90
MIN_TRUE_NORM = 1e-8


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    total = max(int(round(float(seconds))), 0)
    m, s = divmod(total, 60)
    return f"{m:d}m{s:02d}s"


def _project_complement(J: np.ndarray, v: np.ndarray) -> np.ndarray:
    """P_perp @ v = v - Q Q^T v, trimmed to numerical rank of J."""
    if J.shape[1] == 0:
        return v.copy()
    rank = int(np.linalg.matrix_rank(J))
    if rank == 0:
        return v.copy()
    Q, _ = np.linalg.qr(J)
    return v - Q[:, :rank] @ (Q[:, :rank].T @ v)


def _project_tangent(J: np.ndarray, v: np.ndarray) -> np.ndarray:
    """P_T @ v = Q Q^T v, trimmed to numerical rank of J."""
    if J.shape[1] == 0:
        return np.zeros_like(v)
    rank = int(np.linalg.matrix_rank(J))
    if rank == 0:
        return np.zeros_like(v)
    Q, _ = np.linalg.qr(J)
    return Q[:, :rank] @ (Q[:, :rank].T @ v)


def _project_onto(U: np.ndarray, v: np.ndarray) -> np.ndarray:
    """P_U @ v = U U^T v.  U must have orthonormal columns."""
    return U @ (U.T @ v)


def _dprime2(F: np.ndarray, T: int, sigma0: float) -> float:
    return float(np.dot(F, F)) / (max(T, 1) * sigma0 * sigma0)


# ---------------------------------------------------------------------------
# Identity subspace construction
# ---------------------------------------------------------------------------

def _build_identity_subspace(
    responses: dict[int, np.ndarray],
    orientations: list[int],
    energy_threshold: float = DEFAULT_ID_SUBSPACE_ENERGY,
) -> tuple[np.ndarray, dict]:
    """Build orthonormal identity subspace U_id from pairwise response differences.

    D[i, :] = r_b - r_a for each pair (a, b) in combinations(orientations, 2).
    SVD of D gives right singular vectors in cell space (U_id columns).
    Rank is chosen by explained energy >= energy_threshold.

    Returns (U_id, metadata_dict).
    U_id shape: (N_cells, rank_id).
    """
    pairs = list(itertools.combinations(orientations, 2))
    diff_vecs = []
    for a, b in pairs:
        r_a = responses.get(a)
        r_b = responses.get(b)
        if r_a is not None and r_b is not None:
            diff_vecs.append(r_b.astype(np.float64) - r_a.astype(np.float64))

    if not diff_vecs:
        N_cells = next(iter(responses.values())).shape[0] if responses else 1
        return np.zeros((N_cells, 0)), {"error": "no_valid_pairs", "rank_id": 0}

    D = np.stack(diff_vecs, axis=0)  # (n_pairs, N_cells)
    # SVD: D = U_D S_D Vt_D.  Vt_D rows are right singular vectors in cell space.
    _, S, Vt = np.linalg.svd(D, full_matrices=False)  # Vt: (n_pairs, N_cells)
    if float(np.max(S)) < EPS:
        N_cells = D.shape[1]
        return np.zeros((N_cells, 0)), {"error": "zero_difference_vectors", "rank_id": 0}

    energy = S ** 2
    total = float(np.sum(energy)) + EPS
    cumvar = np.cumsum(energy) / total

    def _rank_at(threshold: float) -> int:
        idx = np.searchsorted(cumvar, float(threshold))
        return int(min(idx + 1, len(S)))

    rank_80 = _rank_at(0.80)
    rank_90 = _rank_at(0.90)
    rank_95 = _rank_at(0.95)
    rank_id = _rank_at(energy_threshold)

    U_id = Vt[:rank_id, :].T  # (N_cells, rank_id) — orthonormal columns
    meta = {
        "n_vectors_used": len(diff_vecs),
        "n_cells": int(D.shape[1]),
        "rank_id": rank_id,
        "rank_id_80": rank_80,
        "rank_id_90": rank_90,
        "rank_id_95": rank_95,
        "energy_threshold_used": energy_threshold,
        "energy_explained_80": float(cumvar[rank_80 - 1]) if rank_80 > 0 else float("nan"),
        "energy_explained_90": float(cumvar[rank_90 - 1]) if rank_90 > 0 else float("nan"),
        "energy_explained_95": float(cumvar[rank_95 - 1]) if rank_95 > 0 else float("nan"),
        "singular_values_json": json.dumps([float(s) for s in S]),
    }
    return U_id, meta


# ---------------------------------------------------------------------------
# Pair-specific reference directions
# ---------------------------------------------------------------------------

def _compute_pair_reference_directions(
    runner: "SeparabilityRunner",
    logmar: float,
    orientations: list[int],
    stabilized_position: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Unit identity-difference direction at center for each orientation pair.

    u_ab_center = (r_b(p0) - r_a(p0)) / ||r_b(p0) - r_a(p0)||

    Used as a FIXED reference direction for the pair-axis signed scalar:
      signed_orth_pairaxis(p) = u_ab_center^T @ f_perp(p)

    This allows sign variation across phases and a meaningful cancellation index.
    Returns {(source_ori, target_ori): unit_vector}.
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    for s_ori, t_ori in itertools.combinations(orientations, 2):
        r_a = runner.get_response(logmar, s_ori, stabilized_position)
        r_b = runner.get_response(logmar, t_ori, stabilized_position)
        if r_a is None or r_b is None:
            continue
        d_ab = r_b.astype(np.float64) - r_a.astype(np.float64)
        norm = float(np.linalg.norm(d_ab))
        out[(s_ori, t_ori)] = (d_ab / norm) if norm > EPS else np.zeros_like(d_ab)
    return out


def _center_degeneracy_summary(
    runner: "SeparabilityRunner",
    logmar: float,
    orientations: list[int],
    stabilized_position: np.ndarray,
) -> list[dict]:
    """Pairwise identity distance at center, normalized by mean response norm."""
    responses = {ori: runner.get_response(logmar, ori, stabilized_position) for ori in orientations}
    norms = [float(np.linalg.norm(r)) for r in responses.values() if r is not None]
    mean_norm = float(np.mean(norms)) if norms else float("nan")
    out: list[dict] = []
    for s_ori, t_ori in itertools.combinations(orientations, 2):
        r_a = responses.get(s_ori); r_b = responses.get(t_ori)
        if r_a is None or r_b is None:
            continue
        dist = float(np.linalg.norm(r_b.astype(np.float64) - r_a.astype(np.float64)))
        out.append({
            "logmar": logmar, "source_orientation": s_ori, "target_orientation": t_ori,
            "pairwise_distance_center": dist,
            "normalized_pairwise_distance_center": dist / (mean_norm + EPS),
            "mean_response_norm_center": mean_norm,
        })
    return out


# ---------------------------------------------------------------------------
# Trace preparation
# ---------------------------------------------------------------------------

def _prepare_trace(
    trace_deg: np.ndarray,
    pixels_per_degree: float,
    center_mode: str = DEFAULT_CENTER_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace_px, finite_mask = _prepare_trace_positions(trace_deg, pixels_per_degree, center_mode)
    valid_adjacent = _valid_adjacent_step_start_indices(finite_mask)
    return trace_px, finite_mask, valid_adjacent


def _select_phase_positions(
    trace_px: np.ndarray,
    valid_adjacent: np.ndarray,
    pixels_per_degree: float,
    path_mode: str,
    T: int,
    drift_max_arcmin: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[float], dict]:
    if valid_adjacent.size == 0:
        return [], [], {"n_valid_adjacent": 0, "n_after_mode_filter": 0, "n_selected": 0}

    step_arcmins_all = _step_arcmin_values_from_indices(trace_px, valid_adjacent, pixels_per_degree)

    if path_mode == "drift_step_start_phases":
        keep = step_arcmins_all <= float(drift_max_arcmin)
        filtered_indices = valid_adjacent[keep]
        filtered_arcmins = step_arcmins_all[keep]
    elif path_mode == "all_steps":
        filtered_indices = valid_adjacent
        filtered_arcmins = step_arcmins_all
    else:
        raise ValueError(f"Unknown path_mode: {path_mode!r}")

    n_after = int(filtered_indices.size)
    if n_after == 0:
        return [], [], {"n_valid_adjacent": int(valid_adjacent.size), "n_after_mode_filter": 0, "n_selected": 0}

    if n_after > T:
        chosen = np.sort(rng.choice(n_after, size=T, replace=False))
        filtered_indices = filtered_indices[chosen]
        filtered_arcmins = filtered_arcmins[chosen]

    return (
        [trace_px[int(idx)].copy() for idx in filtered_indices],
        [float(a) for a in filtered_arcmins],
        {"n_valid_adjacent": int(valid_adjacent.size), "n_after_mode_filter": n_after, "n_selected": len(filtered_indices)},
    )


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

class SeparabilityRunner:
    def __init__(self, device: str, pixels_per_degree: float, n_lags: int, jacobian_step_px: float, model_batch_size: int = 32) -> None:
        self._runner = CurvatureScaleMatchRunner(
            device=device, pixels_per_degree=pixels_per_degree, n_lags=n_lags,
            jacobian_step_px=jacobian_step_px, model_batch_size=model_batch_size,
        )
        self._jstep = float(jacobian_step_px)
        self._cache: dict[tuple[float, int], dict] = {}

    def _key(self, logmar: float, ori: int) -> tuple[float, int]:
        return (round(float(logmar), 4), int(ori))

    def _jac_offsets(self) -> list[np.ndarray]:
        s = self._jstep
        return [np.array([s, 0.0]), np.array([-s, 0.0]), np.array([0.0, s]), np.array([0.0, -s])]

    def _ensure(self, logmar: float, ori: int, positions: list[np.ndarray]) -> None:
        k = self._key(logmar, ori)
        cached = self._cache.setdefault(k, {})
        missing = [p for p in positions if self._runner._position_key(p) not in cached]
        if missing:
            new_resp, _ = self._runner.evaluate_condition(logmar, ori, missing)
            cached.update(new_resp)

    def precompute(self, logmar: float, orientations: list[int], positions: list[np.ndarray]) -> None:
        all_pos = list(positions)
        for p in positions:
            for off in self._jac_offsets():
                all_pos.append(p + off)
        for ori in orientations:
            self._ensure(logmar, ori, all_pos)

    def get_response(self, logmar: float, ori: int, position: np.ndarray) -> np.ndarray | None:
        k = self._key(logmar, ori)
        return self._cache.get(k, {}).get(self._runner._position_key(position))

    def get_responses_at(self, logmar: float, orientations: list[int], position: np.ndarray) -> dict[int, np.ndarray]:
        out = {}
        for ori in orientations:
            r = self.get_response(logmar, ori, position)
            if r is not None:
                out[ori] = r
        return out

    def jacobian(self, logmar: float, ori: int, anchor: np.ndarray) -> np.ndarray:
        k = self._key(logmar, ori)
        return self._runner.finite_difference_jacobian(self._cache.get(k, {}), anchor)

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Per-phase metrics (extended with idspace and mechanism)
# ---------------------------------------------------------------------------

def _phase_metrics(
    runner: SeparabilityRunner,
    logmar: float,
    source_ori: int,
    target_ori: int,
    position: np.ndarray,
    sigma0: float,
    U_id: np.ndarray | None = None,
    pixels_per_degree: float = DEFAULT_PPD,
    pair_ref_dirs: dict | None = None,
) -> dict | None:
    """Local separability at one phase, including identity-subspace and mechanism metrics."""
    r_a = runner.get_response(logmar, source_ori, position)
    r_b = runner.get_response(logmar, target_ori, position)
    if r_a is None or r_b is None:
        return None

    J = runner.jacobian(logmar, source_ori, position)
    delta_mu = r_b - r_a
    true_norm = float(np.linalg.norm(delta_mu))
    if true_norm < MIN_TRUE_NORM:
        return None

    f_perp = _project_complement(J, delta_mu)
    f_tang = _project_tangent(J, delta_mu)
    tang_norm = float(np.linalg.norm(f_tang))
    perp_norm = float(np.linalg.norm(f_perp))

    try:
        sv = np.linalg.svd(J, compute_uv=False)
        sv1 = float(sv[0]) if sv.size > 0 else float("nan")
        sv2 = float(sv[1]) if sv.size > 1 else float("nan")
        cond = sv1 / (sv2 + EPS) if sv.size > 1 else float("nan")
    except Exception:
        sv1, sv2, cond = float("nan"), float("nan"), float("nan")

    # Mechanism check: alignment with individual Jacobian columns
    J_x, J_y = J[:, 0], J[:, 1]
    nx = float(np.linalg.norm(J_x)); ny = float(np.linalg.norm(J_y))
    dmu_n = float(np.linalg.norm(delta_mu))
    cos_x = float(np.dot(delta_mu, J_x) / (dmu_n * nx + EPS))
    cos_y = float(np.dot(delta_mu, J_y) / (dmu_n * ny + EPS))
    abs_x = abs(cos_x); abs_y = abs(cos_y)
    dominant_axis = "x" if abs_x >= abs_y else "y"
    dominant_abs = max(abs_x, abs_y)

    result: dict = {
        "f_perp": f_perp,
        "f_tang": f_tang,
        "delta_mu": delta_mu,
        "true_norm": true_norm,
        "tang_norm": tang_norm,
        "perp_norm": perp_norm,
        "mimicry_fraction": (tang_norm ** 2) / (true_norm ** 2 + EPS),
        "orthogonal_fraction": (perp_norm ** 2) / (true_norm ** 2 + EPS),
        "jacobian_sv1": sv1,
        "jacobian_sv2": sv2,
        "jacobian_condition_number": cond,
        # Mechanism
        "cos_x": cos_x,
        "cos_y": cos_y,
        "abs_cos_x": abs_x,
        "abs_cos_y": abs_y,
        "dominant_translation_axis": dominant_axis,
        "dominant_axis_abs_cos": dominant_abs,
    }

    # Identity-subspace restricted metrics
    if U_id is not None and U_id.shape[1] > 0:
        # P_id projects onto the identity subspace built from center responses
        delta_id = _project_onto(U_id, delta_mu)
        delta_id_perp = _project_onto(U_id, f_perp)   # id-restricted orthogonal signal
        delta_id_tang = _project_onto(U_id, f_tang)   # id-restricted tangent signal
        id_energy = float(np.dot(delta_id, delta_id))
        id_perp_energy = float(np.dot(delta_id_perp, delta_id_perp))
        id_tang_energy = float(np.dot(delta_id_tang, delta_id_tang))
        # Cross-term: id_energy != id_perp_energy + id_tang_energy because P_id, P_T don't commute.
        # In rank-1 case: id_cross = 2 * s_perp * s_tang (signed, captures interference)
        id_cross = id_energy - id_perp_energy - id_tang_energy
        result["f_perp_id"] = delta_id_perp
        result["identity_energy"] = id_energy
        result["identity_perp_energy"] = id_perp_energy
        result["identity_tangent_energy"] = id_tang_energy
        result["id_cross_residual"] = id_cross
        result["id_cross_residual_fraction"] = id_cross / (id_energy + EPS)
        result["orthogonal_identity_fraction_idspace"] = id_perp_energy / (id_energy + EPS)
        result["mimicry_fraction_idspace"] = id_tang_energy / (id_energy + EPS)
        # Scalar projections onto first identity axis u1 (signed; sign-flip across phases → cancellation)
        u1 = U_id[:, 0]
        result["s_perp_u1"] = float(np.dot(u1, f_perp))
        result["s_raw_u1"] = float(np.dot(u1, delta_mu))
        result["s_tang_u1"] = float(np.dot(u1, f_tang))
    else:
        result["f_perp_id"] = None
        result["identity_energy"] = float("nan")
        result["identity_perp_energy"] = float("nan")
        result["identity_tangent_energy"] = float("nan")
        result["id_cross_residual"] = float("nan")
        result["id_cross_residual_fraction"] = float("nan")
        result["orthogonal_identity_fraction_idspace"] = float("nan")
        result["mimicry_fraction_idspace"] = float("nan")
        result["s_perp_u1"] = float("nan")
        result["s_raw_u1"] = float("nan")
        result["s_tang_u1"] = float("nan")

    # Pair-axis scalar: fixed center reference direction u_ab_center^T @ f_perp(p).
    # Sign CAN vary across phases (u_ab_center is fixed; f_perp direction changes with phase).
    # Meaningful cancellation index = |sum(s)| / sum(|s|) pools these signed values.
    if pair_ref_dirs is not None:
        u_ab = pair_ref_dirs.get((source_ori, target_ori))
        if u_ab is not None and np.any(u_ab != 0):
            s_orth = float(np.dot(u_ab, f_perp))
            s_tang = float(np.dot(u_ab, f_tang))
            result["signed_orth_pairaxis"] = s_orth
            result["unsigned_orth_pairaxis"] = abs(s_orth)
            result["signed_tang_pairaxis"] = s_tang
            result["unsigned_tang_pairaxis"] = abs(s_tang)
        else:
            result["signed_orth_pairaxis"] = float("nan")
            result["unsigned_orth_pairaxis"] = float("nan")
            result["signed_tang_pairaxis"] = float("nan")
            result["unsigned_tang_pairaxis"] = float("nan")
    else:
        result["signed_orth_pairaxis"] = float("nan")
        result["unsigned_orth_pairaxis"] = float("nan")
        result["signed_tang_pairaxis"] = float("nan")
        result["unsigned_tang_pairaxis"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Per-path metrics (full-space and idspace)
# ---------------------------------------------------------------------------

def _path_metrics(
    valid_phase_metrics: list[dict],
    step_arcmins: list[float],
    T: int,
    sigma0: float,
    sampling_condition: str,
    stabilized_pm: dict | None,
    rng: np.random.Generator,
) -> dict | None:
    """Aggregate phase metrics into a path-level row (full-space and idspace)."""
    if not valid_phase_metrics:
        return None

    if sampling_condition == "stabilized_repeated":
        if stabilized_pm is None:
            return None
        used_pm = [stabilized_pm] * T
        used_steps = [0.0] * T
        T_used = T
    elif sampling_condition in ("real_fem_path", "phase_shuffled_path"):
        T_avail = len(valid_phase_metrics)
        T_used = min(T, T_avail)
        if T_used == 0:
            return None
        if sampling_condition == "phase_shuffled_path":
            order = rng.permutation(T_avail)[:T_used]
        else:
            order = np.arange(T_used)
        used_pm = [valid_phase_metrics[int(i)] for i in order]
        used_steps = [step_arcmins[int(i)] for i in (order if len(step_arcmins) >= T_avail else range(T_used))]
    else:
        return None

    # --- Full-space integrated vectors ---
    f_perp_list = [pm["f_perp"] for pm in used_pm]
    F_perp = sum(f_perp_list)
    F_raw = sum(pm["delta_mu"] for pm in used_pm)
    F_tang = sum(pm["f_tang"] for pm in used_pm)

    d2_perp = _dprime2(F_perp, T_used, sigma0)
    d2_raw = _dprime2(F_raw, T_used, sigma0)
    d2_tang = _dprime2(F_tang, T_used, sigma0)

    d2_perp_stab = float("nan"); gain_vs_stab = float("nan")
    if stabilized_pm is not None:
        f0p = stabilized_pm["f_perp"]
        d2_perp_stab = float(T_used) * float(np.dot(f0p, f0p)) / (sigma0 ** 2)
        if d2_perp_stab > EPS:
            gain_vs_stab = d2_perp / d2_perp_stab

    mean_d2_single = float(np.mean([np.dot(pm["f_perp"], pm["f_perp"]) for pm in used_pm])) / (sigma0 ** 2)
    comp_ratio = (d2_perp / (T_used * mean_d2_single)) if mean_d2_single > EPS else float("nan")

    pairwise: list[float] = []
    for i in range(min(len(f_perp_list), 10)):
        for j in range(i + 1, min(len(f_perp_list), 10)):
            ni = float(np.linalg.norm(f_perp_list[i])); nj = float(np.linalg.norm(f_perp_list[j]))
            if ni > EPS and nj > EPS:
                pairwise.append(float(np.dot(f_perp_list[i], f_perp_list[j]) / (ni * nj)))
    mean_pairwise = float(np.mean(pairwise)) if pairwise else float("nan")

    # --- Identity-subspace integrated vector ---
    has_idspace = all(pm.get("f_perp_id") is not None for pm in used_pm)
    if has_idspace:
        f_perp_id_list = [pm["f_perp_id"] for pm in used_pm]
        F_perp_id = sum(f_perp_id_list)
        d2_perp_id = _dprime2(F_perp_id, T_used, sigma0)

        d2_perp_id_stab = float("nan"); gain_vs_stab_id = float("nan")
        if stabilized_pm is not None and stabilized_pm.get("f_perp_id") is not None:
            f0p_id = stabilized_pm["f_perp_id"]
            d2_perp_id_stab = float(T_used) * float(np.dot(f0p_id, f0p_id)) / (sigma0 ** 2)
            if d2_perp_id_stab > EPS:
                gain_vs_stab_id = d2_perp_id / d2_perp_id_stab

        mean_d2_id_single = float(np.mean([np.dot(pm["f_perp_id"], pm["f_perp_id"]) for pm in used_pm])) / (sigma0 ** 2)
        comp_ratio_id = (d2_perp_id / (T_used * mean_d2_id_single)) if mean_d2_id_single > EPS else float("nan")

        pairwise_id: list[float] = []
        for i in range(min(len(f_perp_id_list), 10)):
            for j in range(i + 1, min(len(f_perp_id_list), 10)):
                ni = float(np.linalg.norm(f_perp_id_list[i])); nj = float(np.linalg.norm(f_perp_id_list[j]))
                if ni > EPS and nj > EPS:
                    pairwise_id.append(float(np.dot(f_perp_id_list[i], f_perp_id_list[j]) / (ni * nj)))
        mean_pairwise_id = float(np.mean(pairwise_id)) if pairwise_id else float("nan")
    else:
        d2_perp_id = float("nan"); d2_perp_id_stab = float("nan")
        gain_vs_stab_id = float("nan"); comp_ratio_id = float("nan"); mean_pairwise_id = float("nan")

    # Step regime
    valid_steps = [s for s in used_steps if np.isfinite(s) and s >= 0]
    frac_lte_1 = float(np.mean([s <= 1.0 for s in valid_steps])) if valid_steps else float("nan")
    frac_1_to_1p5 = float(np.mean([1.0 < s <= 1.5 for s in valid_steps])) if valid_steps else float("nan")
    frac_gte_2 = float(np.mean([s >= 2.0 for s in valid_steps])) if valid_steps else float("nan")
    mean_step = float(np.mean(valid_steps)) if valid_steps else float("nan")
    step_rms = float(np.sqrt(np.mean([s * s for s in valid_steps]))) if valid_steps else float("nan")
    step_p90 = float(np.quantile(valid_steps, 0.90)) if valid_steps else float("nan")

    return {
        "T": T_used, "n_valid_samples": len(valid_phase_metrics), "sigma0": sigma0,
        # Full-space
        "dprime2_orthogonal_path": d2_perp,
        "dprime2_raw_path": d2_raw,
        "dprime2_tangent_path": d2_tang,
        "dprime2_orthogonal_stabilized_repeated": d2_perp_stab,
        "gain_vs_stabilized_repeated": gain_vs_stab,
        "complementarity_ratio": comp_ratio,
        "mean_pairwise_cosine_delta_perp": mean_pairwise,
        "mean_local_mimicry_fraction": float(np.mean([pm["mimicry_fraction"] for pm in used_pm])),
        "mean_local_orthogonal_identity_fraction": float(np.mean([pm["orthogonal_fraction"] for pm in used_pm])),
        # Idspace
        "dprime2_orthogonal_path_idspace": d2_perp_id,
        "dprime2_orthogonal_stabilized_repeated_idspace": d2_perp_id_stab,
        "gain_vs_stabilized_repeated_idspace": gain_vs_stab_id,
        "complementarity_ratio_idspace": comp_ratio_id,
        "mean_pairwise_cosine_delta_perp_idspace": mean_pairwise_id,
        "mean_local_mimicry_fraction_idspace": float(np.mean([pm["mimicry_fraction_idspace"] for pm in used_pm])) if has_idspace else float("nan"),
        "mean_local_orthogonal_identity_fraction_idspace": float(np.mean([pm["orthogonal_identity_fraction_idspace"] for pm in used_pm])) if has_idspace else float("nan"),
        # Step regime
        "fraction_steps_lte_1arcmin": frac_lte_1,
        "fraction_steps_1_to_1p5arcmin": frac_1_to_1p5,
        "fraction_steps_gte_2arcmin": frac_gte_2,
        "mean_step_arcmin": mean_step,
        "step_rms_arcmin": step_rms,
        "step_p90_arcmin": step_p90,
    }


def _verify_shuffled_equals_real(phase_metrics: list[dict | None], rng: np.random.Generator, T: int) -> dict:
    valid = [pm for pm in phase_metrics if pm is not None]
    if len(valid) < 2:
        return {"invariance_verified": None, "max_abs_error": None, "reason": "too_few_valid_phases"}
    T_used = min(T, len(valid))
    F_real = sum(valid[i]["f_perp"] for i in range(T_used))
    F_shuf = sum(valid[int(i)]["f_perp"] for i in rng.permutation(T_used))
    max_err = float(np.max(np.abs(F_real - F_shuf)))
    return {"invariance_verified": bool(max_err < 1e-10), "max_abs_error": max_err,
            "reason": "sum_commutative_as_expected" if max_err < 1e-10 else "unexpected_error"}


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def _run_logmar(
    runner: SeparabilityRunner,
    logmar: float,
    orientations: list[int],
    traces_deg: np.ndarray,
    durations: np.ndarray,
    T: int,
    sigma0: float,
    path_modes: list[str],
    sampling_conditions: list[str],
    pixels_per_degree: float,
    jacobian_step_px: float,
    drift_max_arcmin: float,
    center_mode: str,
    max_traces: int | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    identity_subspace_mode: str,
    identity_subspace_energy: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """Run separability analysis for one logmar.

    Returns (pair_rows, phase_rows, mech_rows, sanity_rows, idspace_meta).
    """
    rng = np.random.default_rng(int(bootstrap_seed))
    stabilized_position = np.zeros(2, dtype=np.float64)
    identity_pairs = list(itertools.combinations(orientations, 2))
    n_traces = int(traces_deg.shape[0]) if max_traces is None else min(int(max_traces), int(traces_deg.shape[0]))

    # --- Build identity subspace before trace loop ---
    U_id: np.ndarray | None = None
    idspace_meta: dict = {"identity_subspace_mode": identity_subspace_mode, "logmar": logmar}

    if identity_subspace_mode == "center":
        print(f"  Building center identity subspace (energy={identity_subspace_energy}) ...", flush=True)
        runner.precompute(logmar=logmar, orientations=orientations, positions=[stabilized_position])
        center_responses = runner.get_responses_at(logmar, orientations, stabilized_position)
        U_id, meta = _build_identity_subspace(center_responses, orientations, energy_threshold=identity_subspace_energy)
        idspace_meta.update(meta)
        print(f"  U_id shape: {U_id.shape} (rank_id={meta.get('rank_id')}, n_cells={meta.get('n_cells')})", flush=True)
    elif identity_subspace_mode in ("none", ""):
        idspace_meta["rank_id"] = 0
    else:
        print(f"  WARNING: identity_subspace_mode={identity_subspace_mode!r} not implemented; using none.", flush=True)
        idspace_meta["rank_id"] = 0

    pair_rows: list[dict] = []
    phase_rows: list[dict] = []
    mech_rows: list[dict] = []
    sanity_rows: list[dict] = []
    ori_proj_rows: list[dict] = []
    center_deg_rows: list[dict] = []
    energy_rows: list[dict] = []     # Task 3: per-trace energy readout
    cosine_rows: list[dict] = []     # Task 2: per-trace phase cosine summary
    unit_tuning_rows: list[dict] = []  # Task 4: single-unit tuning
    total_start = time.perf_counter()

    # Always precompute center responses (needed for pair reference directions)
    runner.precompute(logmar=logmar, orientations=orientations, positions=[stabilized_position])
    pair_ref_dirs = _compute_pair_reference_directions(runner, logmar, orientations, stabilized_position)
    center_deg_rows.extend(_center_degeneracy_summary(runner, logmar, orientations, stabilized_position))

    # Task 4: single-unit orientation tuning at center
    unit_tuning_rows.extend(_single_unit_tuning(runner, logmar, orientations, stabilized_position))

    # Task 1: center phase metrics for each pair (once per logmar)
    center_pair_pm = _collect_center_pair_metrics(runner, logmar, orientations, stabilized_position,
                                                   sigma0, U_id, pair_ref_dirs, pixels_per_degree)

    # Collect center orientation projections onto u1 (one-time, before trace loop)
    if U_id is not None and U_id.shape[1] > 0:
        ori_proj_rows.extend(_center_orientation_projections(runner, logmar, orientations, U_id, stabilized_position))

    for trace_id in range(n_traces):
        trace_start = time.perf_counter()
        trace_deg_t = traces_deg[trace_id, : int(durations[trace_id])]
        trace_px, finite_mask, valid_adjacent = _prepare_trace(trace_deg_t, pixels_per_degree, center_mode)

        n_raw = int(trace_deg_t.shape[0])
        n_finite = int(np.sum(finite_mask))
        n_valid_adj = int(valid_adjacent.size)
        print(f"  Trace {trace_id + 1}/{n_traces}: n_raw={n_raw} n_finite={n_finite} n_excluded={n_raw - n_finite} n_valid_adjacent={n_valid_adj}", flush=True)

        for path_mode in path_modes:
            positions, step_arcmins, sel_diag = _select_phase_positions(
                trace_px, valid_adjacent, pixels_per_degree, path_mode, T, drift_max_arcmin, rng)
            if not positions:
                print(f"    {path_mode}: no valid positions, skipping.", flush=True)
                continue

            all_positions = list(positions) + [stabilized_position]
            print(f"    {path_mode}: {len(positions)} path positions, precomputing ...", flush=True)
            runner.precompute(logmar=logmar, orientations=orientations, positions=all_positions)

            for source_ori, target_ori in identity_pairs:
                phase_pm: list[dict | None] = []
                for pos in positions:
                    pm = _phase_metrics(runner, logmar, source_ori, target_ori, pos, sigma0, U_id, pixels_per_degree, pair_ref_dirs)
                    phase_pm.append(pm)
                    if pm is not None:
                        phase_rows.append({
                            "logmar": logmar, "source_orientation": source_ori, "target_orientation": target_ori,
                            "trace_id": trace_id, "path_mode": path_mode,
                            "phase_x_px": float(pos[0]), "phase_y_px": float(pos[1]),
                            "phase_x_arcmin": _px_to_arcmin(float(pos[0]), pixels_per_degree),
                            "phase_y_arcmin": _px_to_arcmin(float(pos[1]), pixels_per_degree),
                            "delta_mu_norm": pm["true_norm"],
                            "tangent_component_norm": pm["tang_norm"],
                            "orthogonal_component_norm": pm["perp_norm"],
                            "mimicry_fraction": pm["mimicry_fraction"],
                            "orthogonal_identity_fraction": pm["orthogonal_fraction"],
                            "jacobian_condition_number": pm["jacobian_condition_number"],
                            "jacobian_singular_value_1": pm["jacobian_sv1"],
                            "jacobian_singular_value_2": pm["jacobian_sv2"],
                            "identity_subspace_mode": identity_subspace_mode,
                            "identity_subspace_rank": idspace_meta.get("rank_id", 0),
                            "identity_energy": pm["identity_energy"],
                            "identity_perp_energy": pm["identity_perp_energy"],
                            "identity_tangent_energy": pm["identity_tangent_energy"],
                            "id_cross_residual": pm["id_cross_residual"],
                            "id_cross_residual_fraction": pm["id_cross_residual_fraction"],
                            "mimicry_fraction_idspace": pm["mimicry_fraction_idspace"],
                            "orthogonal_identity_fraction_idspace": pm["orthogonal_identity_fraction_idspace"],
                            "s_perp_u1": pm["s_perp_u1"],
                            "s_raw_u1": pm["s_raw_u1"],
                            "s_tang_u1": pm["s_tang_u1"],
                            "signed_orth_pairaxis": pm["signed_orth_pairaxis"],
                            "unsigned_orth_pairaxis": pm["unsigned_orth_pairaxis"],
                            "signed_tang_pairaxis": pm["signed_tang_pairaxis"],
                            "unsigned_tang_pairaxis": pm["unsigned_tang_pairaxis"],
                        })
                        mech_rows.append({
                            "logmar": logmar, "source_orientation": source_ori, "target_orientation": target_ori,
                            "trace_id": trace_id, "path_mode": path_mode,
                            "phase_x_px": float(pos[0]), "phase_y_px": float(pos[1]),
                            "cos_x": pm["cos_x"], "cos_y": pm["cos_y"],
                            "abs_cos_x": pm["abs_cos_x"], "abs_cos_y": pm["abs_cos_y"],
                            "dominant_translation_axis": pm["dominant_translation_axis"],
                            "dominant_axis_abs_cos": pm["dominant_axis_abs_cos"],
                            "tangent_mimicry": pm["mimicry_fraction"],
                        })

                valid_pm = [pm for pm in phase_pm if pm is not None]
                stab_pm = _phase_metrics(runner, logmar, source_ori, target_ori, stabilized_position, sigma0, U_id, pixels_per_degree, pair_ref_dirs)

                # Task 2: pairwise cosine of d_ab across phases (cap at 30 phases for speed)
                delta_mus = [pm["delta_mu"] for pm in valid_pm]
                n_cos = min(len(delta_mus), 30)
                cos_vals: list[float] = []
                for ci in range(n_cos):
                    for cj in range(ci + 1, n_cos):
                        ni = float(np.linalg.norm(delta_mus[ci])); nj = float(np.linalg.norm(delta_mus[cj]))
                        if ni > EPS and nj > EPS:
                            cos_vals.append(float(np.dot(delta_mus[ci], delta_mus[cj]) / (ni * nj)))
                import json as _json_mod
                cosine_rows.append({
                    "logmar": logmar, "source_orientation": source_ori, "target_orientation": target_ori,
                    "trace_id": trace_id, "path_mode": path_mode,
                    "n_valid_phases": len(valid_pm), "n_phase_pairs_sampled": len(cos_vals),
                    "median_cos_dab": float(np.nanmedian(cos_vals)) if cos_vals else float("nan"),
                    "fraction_negative": float(np.mean([c < 0 for c in cos_vals])) if cos_vals else float("nan"),
                    "pairwise_cosines_json": _json_mod.dumps(cos_vals[:50]),  # store up to 50
                })

                # Task 3: energy readout (sum of squared norms, not sum of vectors)
                T_used_e = min(T, len(valid_pm))
                if T_used_e > 0:
                    orth_e = float(sum(pm["perp_norm"] ** 2 for pm in valid_pm[:T_used_e]))
                    raw_e = float(sum(pm["true_norm"] ** 2 for pm in valid_pm[:T_used_e]))
                    cpm = center_pair_pm.get((source_ori, target_ori))
                    center_orth_e = float(cpm["perp_norm"] ** 2) if cpm else float("nan")
                    center_raw_e = float(cpm["true_norm"] ** 2) if cpm else float("nan")
                    gain_orth = (orth_e / (T_used_e * center_orth_e)) if (cpm and center_orth_e > EPS) else float("nan")
                    gain_raw = (raw_e / (T_used_e * center_raw_e)) if (cpm and center_raw_e > EPS) else float("nan")
                    energy_rows.append({
                        "logmar": logmar, "source_orientation": source_ori, "target_orientation": target_ori,
                        "trace_id": trace_id, "path_mode": path_mode, "T": T_used_e,
                        "orthogonal_energy_readout": orth_e,
                        "raw_energy_readout": raw_e,
                        "center_orthogonal_energy": center_orth_e,
                        "gain_orthogonal_vs_stabilized": gain_orth,
                        "gain_raw_vs_stabilized": gain_raw,
                    })

                if trace_id == 0 and source_ori == identity_pairs[0][0] and target_ori == identity_pairs[0][1]:
                    check = _verify_shuffled_equals_real(phase_pm, rng, T)
                    sanity_rows.append({"logmar": logmar, "path_mode": path_mode,
                                        "source_orientation": source_ori, "target_orientation": target_ori,
                                        "trace_id": trace_id, "n_valid_phases": len(valid_pm),
                                        "check_type": "phase_shuffled_equals_real_fem_by_construction", **check})

                for cond in sampling_conditions:
                    row = _path_metrics(valid_pm, step_arcmins, T, sigma0, cond, stab_pm, rng)
                    if row is None:
                        continue
                    pair_rows.append({
                        "logmar": logmar, "source_orientation": source_ori, "target_orientation": target_ori,
                        "trace_id": trace_id, "path_mode": path_mode, "sampling_condition": cond,
                        "identity_subspace_mode": identity_subspace_mode,
                        "identity_subspace_rank": idspace_meta.get("rank_id", 0),
                        "n_raw_samples_trace": n_raw, "n_finite_samples_trace": n_finite,
                        "n_excluded_samples_trace": n_raw - n_finite, "n_valid_adjacent_trace": n_valid_adj,
                        **row,
                    })

        trace_elapsed = time.perf_counter() - trace_start
        print(f"  Trace {trace_id + 1} done in {_format_duration(trace_elapsed)}, total {_format_duration(time.perf_counter() - total_start)}", flush=True)

    return (pair_rows, phase_rows, mech_rows, sanity_rows, idspace_meta,
            ori_proj_rows, center_deg_rows, energy_rows, cosine_rows,
            unit_tuning_rows, center_pair_pm)


# ---------------------------------------------------------------------------
# Aggregation: bootstrap summary and per-pair summary
# ---------------------------------------------------------------------------

_METRIC_KEYS_FULL = (
    "dprime2_orthogonal_path", "dprime2_raw_path",
    "gain_vs_stabilized_repeated", "complementarity_ratio",
    "mean_local_mimicry_fraction", "mean_local_orthogonal_identity_fraction",
    "mean_pairwise_cosine_delta_perp",
)
_METRIC_KEYS_ID = (
    "dprime2_orthogonal_path_idspace",
    "gain_vs_stabilized_repeated_idspace", "complementarity_ratio_idspace",
    "mean_local_mimicry_fraction_idspace", "mean_local_orthogonal_identity_fraction_idspace",
    "mean_pairwise_cosine_delta_perp_idspace",
)
_ALL_METRIC_KEYS = _METRIC_KEYS_FULL + _METRIC_KEYS_ID


def _bootstrap_summary(
    pair_rows: list[dict],
    bootstrap_samples: int,
    bootstrap_seed: int,
    groupby_keys: tuple[str, ...] = ("logmar", "sampling_condition", "path_mode", "identity_subspace_mode"),
    metric_keys: tuple[str, ...] = _ALL_METRIC_KEYS,
) -> list[dict]:
    rng = np.random.default_rng(int(bootstrap_seed))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in pair_rows:
        key = tuple(str(row.get(k, "")) for k in groupby_keys)
        groups[key].append(row)

    summary_rows: list[dict] = []
    for group_key, rows in sorted(groups.items()):
        units: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            unit = (int(row["source_orientation"]), int(row["target_orientation"]), int(row["trace_id"]))
            units[unit].append(row)
        unit_list = sorted(units)
        n_units = len(unit_list)

        def _med(mk: str) -> float:
            vals = [float(r[mk]) for r in rows if str(r.get(mk, "")) not in ("", "nan", "NaN")]
            return float(np.nanmedian(vals)) if vals else float("nan")

        boot: dict[str, list[float]] = {mk: [] for mk in metric_keys}
        for _ in range(max(int(bootstrap_samples), 0)):
            samp = rng.choice(n_units, size=n_units, replace=True)
            samp_rows: list[dict] = []
            for ui in samp:
                samp_rows.extend(units[unit_list[int(ui)]])
            for mk in metric_keys:
                vals = [float(r[mk]) for r in samp_rows if str(r.get(mk, "")) not in ("", "nan", "NaN")]
                boot[mk].append(float(np.nanmedian(vals)) if vals else float("nan"))

        row_out: dict = {}
        for k, v in zip(groupby_keys, group_key, strict=False):
            row_out[k] = v
        row_out["n_rows"] = len(rows)
        row_out["n_traces"] = len({int(r["trace_id"]) for r in rows})
        row_out["n_pairs"] = len({(int(r["source_orientation"]), int(r["target_orientation"])) for r in rows})
        for mk in metric_keys:
            row_out[f"median_{mk}"] = _med(mk)
            arr = np.asarray([v for v in boot[mk] if np.isfinite(v)])
            row_out[f"ci_low_{mk}"] = float(np.nanquantile(arr, 0.025)) if arr.size else float("nan")
            row_out[f"ci_high_{mk}"] = float(np.nanquantile(arr, 0.975)) if arr.size else float("nan")
        summary_rows.append(row_out)
    return summary_rows


def _bootstrap_summary_by_pair(
    pair_rows: list[dict],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    """Per-pair bootstrap summary grouped by (logmar, source_ori, target_ori, cond, path_mode, id_mode)."""
    return _bootstrap_summary(
        pair_rows, bootstrap_samples, bootstrap_seed,
        groupby_keys=("logmar", "source_orientation", "target_orientation",
                      "sampling_condition", "path_mode", "identity_subspace_mode"),
        metric_keys=_ALL_METRIC_KEYS,
    )


def _mechanism_summary(mech_rows: list[dict]) -> list[dict]:
    """Aggregate mechanism rows by (logmar, source_ori, target_ori, path_mode)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in mech_rows:
        key = (float(row["logmar"]), int(row["source_orientation"]), int(row["target_orientation"]), str(row["path_mode"]))
        groups[key].append(row)

    out: list[dict] = []
    for (logmar, s_ori, t_ori, path_mode), rows in sorted(groups.items()):
        abs_cos_x = np.asarray([float(r["abs_cos_x"]) for r in rows])
        abs_cos_y = np.asarray([float(r["abs_cos_y"]) for r in rows])
        dom_abs = np.asarray([float(r["dominant_axis_abs_cos"]) for r in rows])
        mimicry = np.asarray([float(r["tangent_mimicry"]) for r in rows])
        frac_x = float(np.mean([r["dominant_translation_axis"] == "x" for r in rows]))
        out.append({
            "logmar": logmar, "source_orientation": s_ori, "target_orientation": t_ori,
            "path_mode": path_mode, "n_phases": len(rows),
            "median_abs_cos_x": float(np.nanmedian(abs_cos_x)),
            "median_abs_cos_y": float(np.nanmedian(abs_cos_y)),
            "fraction_x_dominant": frac_x,
            "fraction_y_dominant": 1.0 - frac_x,
            "median_dominant_axis_abs_cos": float(np.nanmedian(dom_abs)),
            "median_tangent_mimicry": float(np.nanmedian(mimicry)),
        })
    return out


def _sign_summary(phase_rows: list[dict]) -> list[dict]:
    """Per-pair sign-flip diagnostics on the u1 scalar projection s_perp_u1.

    The key question: does the (0,180) pair fail because s_perp_u1 changes
    sign across phases, causing F_perp_id = u1 * sum(s_perp) ≈ 0?

    signed_cancellation_index = |sum(s)| / (sum(|s|) + eps).
    Near 0 → strong sign-flip cancellation.
    Near 1 → consistent-sign accumulation.
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in phase_rows:
        v = row.get("s_perp_u1", "")
        if str(v) in ("", "nan", "NaN"):
            continue
        key = (float(row["logmar"]), int(row["source_orientation"]),
               int(row["target_orientation"]), str(row["path_mode"]))
        groups[key].append(float(v))

    out: list[dict] = []
    for (logmar, s_ori, t_ori, path_mode), vals in sorted(groups.items()):
        arr = np.asarray(vals)
        n = len(arr)
        frac_pos = float(np.mean(arr > 0))
        signed_cancellation = float(abs(np.sum(arr)) / (np.sum(np.abs(arr)) + EPS))
        out.append({
            "logmar": logmar, "source_orientation": s_ori, "target_orientation": t_ori,
            "path_mode": path_mode, "n_phases": n,
            "median_abs_s_perp_u1": float(np.nanmedian(np.abs(arr))),
            "median_s_perp_u1": float(np.nanmedian(arr)),
            "fraction_positive_s_perp_u1": frac_pos,
            "fraction_negative_s_perp_u1": 1.0 - frac_pos,
            "sign_consistency": float(max(frac_pos, 1.0 - frac_pos)),
            "signed_cancellation_index": signed_cancellation,
        })
    return out


def _idspace_fraction_summary(phase_rows: list[dict], bootstrap_samples: int, bootstrap_seed: int) -> list[dict]:
    """Bootstrap idspace fraction metrics grouped by (logmar, path_mode, source_ori, target_ori).

    Reports orth_id, mim_id, cross_residual_fraction with trace-level bootstrap CIs.
    """
    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in phase_rows:
        key = (float(row["logmar"]), str(row["path_mode"]),
               int(row["source_orientation"]), int(row["target_orientation"]))
        groups[key][int(row["trace_id"])].append(row)

    rng = np.random.default_rng(int(bootstrap_seed))
    metrics = ("orthogonal_identity_fraction_idspace", "mimicry_fraction_idspace", "id_cross_residual_fraction")
    out: list[dict] = []

    for key, by_trace in sorted(groups.items()):
        logmar, path_mode, s_ori, t_ori = key
        trace_ids = sorted(by_trace)
        n_traces = len(trace_ids)
        all_rows = [r for rows in by_trace.values() for r in rows]
        n_phases = len(all_rows)

        def _med(mk: str, rows: list[dict]) -> float:
            vals = [float(r[mk]) for r in rows if str(r.get(mk, "")) not in ("", "nan", "NaN")]
            return float(np.nanmedian(vals)) if vals else float("nan")

        boot: dict[str, list[float]] = {mk: [] for mk in metrics}
        for _ in range(max(int(bootstrap_samples), 0)):
            samp_ids = rng.choice(n_traces, size=n_traces, replace=True)
            samp_rows = [r for ti in samp_ids for r in by_trace[trace_ids[int(ti)]]]
            for mk in metrics:
                boot[mk].append(_med(mk, samp_rows))

        row_out: dict = {
            "logmar": logmar, "path_mode": path_mode,
            "source_orientation": s_ori, "target_orientation": t_ori,
            "n_phases": n_phases, "n_traces": n_traces,
        }
        for mk in metrics:
            row_out[f"median_{mk}"] = _med(mk, all_rows)
            arr = np.asarray([v for v in boot[mk] if np.isfinite(v)])
            row_out[f"ci_low_{mk}"] = float(np.nanquantile(arr, 0.025)) if arr.size else float("nan")
            row_out[f"ci_high_{mk}"] = float(np.nanquantile(arr, 0.975)) if arr.size else float("nan")
        out.append(row_out)
    return out


def _headline_decomposition_summary(
    phase_rows: list[dict],
    bootstrap_samples: int,
    bootstrap_seed: int,
    groupby_pair: bool = True,
) -> list[dict]:
    """Bootstrap-CI summary of absolute identity energy, orthogonal/tangent energies, and fractions.

    groupby_pair=True: one row per (logmar, path_mode, source_ori, target_ori).
    groupby_pair=False: pooled across pairs — one row per (logmar, path_mode).
    Bootstrap unit: trace_id within each group.
    """
    metrics = {
        "identity_energy": lambda r: float(r["delta_mu_norm"]) ** 2,
        "orthogonal_energy": lambda r: float(r["orthogonal_component_norm"]) ** 2,
        "tangent_energy": lambda r: float(r["tangent_component_norm"]) ** 2,
        "orthogonal_fraction": lambda r: float(r["orthogonal_identity_fraction"]),
        "mimicry_fraction": lambda r: float(r["mimicry_fraction"]),
    }
    rng = np.random.default_rng(int(bootstrap_seed))

    def _group_key(row: dict) -> tuple:
        base = (float(row["logmar"]), str(row["path_mode"]))
        if groupby_pair:
            return base + (int(row["source_orientation"]), int(row["target_orientation"]))
        return base

    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in phase_rows:
        key = _group_key(row)
        try:
            tid = int(row["trace_id"])
        except (KeyError, ValueError):
            continue
        groups[key][tid].append(row)

    out: list[dict] = []
    for key, by_trace in sorted(groups.items()):
        trace_ids = sorted(by_trace)
        n_traces = len(trace_ids)
        all_rows = [r for rows in by_trace.values() for r in rows]

        def _safe_val(r: dict, mk: str) -> float | None:
            try:
                v = metrics[mk](r)
                return v if np.isfinite(v) else None
            except Exception:
                return None

        def _med(mk: str, rows: list[dict]) -> float:
            vals = [v for r in rows for v in [_safe_val(r, mk)] if v is not None]
            return float(np.nanmedian(vals)) if vals else float("nan")

        boot: dict[str, list[float]] = {mk: [] for mk in metrics}
        for _ in range(max(int(bootstrap_samples), 0)):
            samp = rng.choice(n_traces, size=n_traces, replace=True)
            samp_rows = [r for ti in samp for r in by_trace[trace_ids[int(ti)]]]
            for mk in metrics:
                boot[mk].append(_med(mk, samp_rows))

        row_out: dict = {}
        if groupby_pair:
            row_out["logmar"], row_out["path_mode"], row_out["source_orientation"], row_out["target_orientation"] = key
        else:
            row_out["logmar"], row_out["path_mode"] = key
        row_out["n_traces"] = n_traces
        row_out["n_phases"] = len(all_rows)
        for mk in metrics:
            row_out[f"median_{mk}"] = _med(mk, all_rows)
            arr = np.asarray([v for v in boot[mk] if np.isfinite(v)])
            row_out[f"ci_low_{mk}"] = float(np.nanquantile(arr, 0.025)) if arr.size else float("nan")
            row_out[f"ci_high_{mk}"] = float(np.nanquantile(arr, 0.975)) if arr.size else float("nan")
        out.append(row_out)
    return out


def _pair_axis_summary(phase_rows: list[dict]) -> list[dict]:
    """Per-pair sign-flip and accumulation summary using the fixed center-reference scalar.

    cancellation_index = |sum(signed_orth)| / (sum(|signed_orth|) + eps)
    Near 0 → sign-flip cancellation across phases.
    Near 1 → coherent accumulation.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in phase_rows:
        s = row.get("signed_orth_pairaxis", "")
        if str(s) in ("", "nan", "NaN"):
            continue
        key = (float(row["logmar"]), str(row["path_mode"]),
               int(row["source_orientation"]), int(row["target_orientation"]))
        groups[key].append(row)

    out: list[dict] = []
    for (logmar, path_mode, s_ori, t_ori), rows in sorted(groups.items()):
        s_orth = np.asarray([float(r["signed_orth_pairaxis"]) for r in rows])
        u_orth = np.abs(s_orth)
        s_tang = np.asarray([float(r.get("signed_tang_pairaxis", float("nan"))) for r in rows])
        n_traces = len({int(r["trace_id"]) for r in rows})
        frac_pos = float(np.mean(s_orth > 0))
        cancel_idx = float(abs(np.sum(s_orth)) / (np.sum(u_orth) + EPS))
        out.append({
            "logmar": logmar, "path_mode": path_mode,
            "source_orientation": s_ori, "target_orientation": t_ori,
            "n_phases": len(rows), "n_traces": n_traces,
            # Energy metrics (from full-space, same rows)
            "median_identity_energy": float(np.nanmedian([float(r["delta_mu_norm"]) ** 2 for r in rows])),
            "median_orthogonal_energy": float(np.nanmedian([float(r["orthogonal_component_norm"]) ** 2 for r in rows])),
            "median_orthogonal_fraction": float(np.nanmedian([float(r["orthogonal_identity_fraction"]) for r in rows])),
            "median_mimicry_fraction": float(np.nanmedian([float(r["mimicry_fraction"]) for r in rows])),
            # Pair-axis scalars
            "mean_signed_identity_orth_scalar": float(np.nanmean(s_orth)),
            "median_signed_identity_orth_scalar": float(np.nanmedian(s_orth)),
            "mean_unsigned_identity_orth_scalar": float(np.nanmean(u_orth)),
            "median_unsigned_identity_orth_scalar": float(np.nanmedian(u_orth)),
            "sum_signed_identity_orth_scalar": float(np.sum(s_orth)),
            "sum_unsigned_identity_orth_scalar": float(np.sum(u_orth)),
            "cancellation_index": cancel_idx,
            "fraction_positive": frac_pos,
            "fraction_negative": 1.0 - frac_pos,
            "sign_consistency": float(max(frac_pos, 1.0 - frac_pos)),
        })
    return out


def _offcenter_identity_summary(phase_rows: list[dict], bootstrap_samples: int, bootstrap_seed: int) -> list[dict]:
    """Normalized pairwise identity distance (||d_ab||/mean_response) at FEM phases.

    Uses delta_mu_norm (already computed per phase) as the pairwise distance.
    The 'normalized' version divides by the phase-averaged response norm.
    Note: we only have delta_mu_norm here, not individual response norms.
    We compute median_delta_mu_norm grouped by (logmar, path_mode, pair).
    """
    rng = np.random.default_rng(int(bootstrap_seed))
    groups: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in phase_rows:
        try:
            v = float(row["delta_mu_norm"])
        except (KeyError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        key = (float(row["logmar"]), str(row["path_mode"]),
               int(row["source_orientation"]), int(row["target_orientation"]))
        groups[key][int(row["trace_id"])].append(v)

    out: list[dict] = []
    for (logmar, path_mode, s_ori, t_ori), by_trace in sorted(groups.items()):
        trace_ids = sorted(by_trace)
        all_vals = [v for vals in by_trace.values() for v in vals]
        boot_meds = []
        for _ in range(max(int(bootstrap_samples), 0)):
            samp = [v for ti in np.random.default_rng().choice(len(trace_ids), size=len(trace_ids), replace=True)
                    for v in by_trace[trace_ids[int(ti)]]]
            boot_meds.append(float(np.nanmedian(samp)) if samp else float("nan"))
        arr = np.asarray([v for v in boot_meds if np.isfinite(v)])
        out.append({
            "logmar": logmar, "path_mode": path_mode,
            "source_orientation": s_ori, "target_orientation": t_ori,
            "n_traces": len(trace_ids), "n_phases": len(all_vals),
            "median_delta_mu_norm": float(np.nanmedian(all_vals)),
            "ci_low_delta_mu_norm": float(np.nanquantile(arr, 0.025)) if arr.size else float("nan"),
            "ci_high_delta_mu_norm": float(np.nanquantile(arr, 0.975)) if arr.size else float("nan"),
        })
    return out


# ---------------------------------------------------------------------------
# Task 1. fem_sampling_identity_magnitude_summary
# ---------------------------------------------------------------------------

def _collect_center_pair_metrics(
    runner: "SeparabilityRunner",
    logmar: float,
    orientations: list[int],
    stabilized_position: np.ndarray,
    sigma0: float,
    U_id: "np.ndarray | None",
    pair_ref_dirs: dict,
    pixels_per_degree: float,
) -> dict[tuple[int, int], dict]:
    """Compute full _phase_metrics for each identity pair at the stabilized center.

    Returns {(source_ori, target_ori): pm_dict}.  Called once per logmar before
    the trace loop — the center position is identical for every trace.
    """
    out: dict[tuple[int, int], dict] = {}
    for s_ori, t_ori in itertools.combinations(orientations, 2):
        pm = _phase_metrics(runner, logmar, s_ori, t_ori, stabilized_position,
                            sigma0, U_id, pixels_per_degree, pair_ref_dirs)
        if pm is not None:
            out[(s_ori, t_ori)] = pm
    return out


def _fem_sampling_identity_summary(
    center_pair_pm: dict[tuple[int, int], dict],
    phase_rows: list[dict],
    logmar: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict]:
    """Task 1 — magnitude/orthogonality table with center as a phase_condition.

    Groups:
      phase_condition = center | drift_step_start_phases | all_steps
    Bootstrap: for center, over identity pairs; for drift/all_steps, over traces.
    """
    rng = np.random.default_rng(int(bootstrap_seed))
    metrics_keys = ("identity_norm", "identity_energy", "orthogonal_fraction",
                    "mimicry_fraction", "orthogonal_energy", "tangent_energy")

    def _pm_to_vals(pm: dict) -> dict[str, float]:
        tn = float(pm["true_norm"]); pn = float(pm["perp_norm"]); tgn = float(pm["tang_norm"])
        return {
            "identity_norm": tn,
            "identity_energy": tn ** 2,
            "orthogonal_energy": pn ** 2,
            "tangent_energy": tgn ** 2,
            "orthogonal_fraction": float(pm["orthogonal_fraction"]),
            "mimicry_fraction": float(pm["mimicry_fraction"]),
        }

    def _bootstrap_ci(vals_list: list[float], n_boot: int, rng_local: np.random.Generator) -> tuple[float, float]:
        if not vals_list:
            return float("nan"), float("nan")
        if n_boot <= 0:
            return float("nan"), float("nan")
        arr = np.asarray(vals_list)
        boot_meds = [float(np.nanmedian(rng_local.choice(arr, size=len(arr), replace=True)))
                     for _ in range(n_boot)]
        ba = np.asarray([v for v in boot_meds if np.isfinite(v)])
        return (float(np.nanquantile(ba, 0.025)), float(np.nanquantile(ba, 0.975))) if ba.size else (float("nan"), float("nan"))

    out: list[dict] = []

    # --- Center ---
    for (s_ori, t_ori), pm in sorted(center_pair_pm.items()):
        vals = _pm_to_vals(pm)
        row: dict = {
            "logmar": logmar, "phase_condition": "center",
            "source_orientation": s_ori, "target_orientation": t_ori,
            "n_traces": 1, "n_phases": 1,
        }
        for mk in metrics_keys:
            row[f"median_{mk}"] = vals.get(mk, float("nan"))
            row[f"ci_low_{mk}"] = float("nan")
            row[f"ci_high_{mk}"] = float("nan")
        out.append(row)

    # Also produce pooled center row across pairs (bootstrap over pairs)
    all_center_vals: dict[str, list[float]] = defaultdict(list)
    for pm in center_pair_pm.values():
        for mk, v in _pm_to_vals(pm).items():
            all_center_vals[mk].append(v)
    row_pool: dict = {"logmar": logmar, "phase_condition": "center",
                      "source_orientation": -1, "target_orientation": -1,
                      "n_traces": 1, "n_phases": len(center_pair_pm)}
    for mk in metrics_keys:
        vals_list = all_center_vals.get(mk, [])
        row_pool[f"median_{mk}"] = float(np.nanmedian(vals_list)) if vals_list else float("nan")
        ci_lo, ci_hi = _bootstrap_ci(vals_list, bootstrap_samples, rng)
        row_pool[f"ci_low_{mk}"] = ci_lo
        row_pool[f"ci_high_{mk}"] = ci_hi
    out.append(row_pool)

    # --- Drift / all_steps: bootstrap over traces ---
    def _get_metric(row: dict, mk: str) -> float | None:
        field_map = {
            "identity_norm": "delta_mu_norm",
            "identity_energy": None,   # computed below
            "orthogonal_fraction": "orthogonal_identity_fraction",
            "mimicry_fraction": "mimicry_fraction",
            "orthogonal_energy": None,
            "tangent_energy": None,
        }
        if mk == "identity_energy":
            try: return float(row["delta_mu_norm"]) ** 2
            except Exception: return None
        if mk == "orthogonal_energy":
            try: return float(row["orthogonal_component_norm"]) ** 2
            except Exception: return None
        if mk == "tangent_energy":
            try: return float(row["tangent_component_norm"]) ** 2
            except Exception: return None
        if mk == "identity_norm":
            try: v = float(row["delta_mu_norm"]); return v if np.isfinite(v) else None
            except Exception: return None
        field = field_map.get(mk)
        if field is None: return None
        try:
            v = float(row[field])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    # Group by (phase_condition, source_ori, target_ori) with trace-level bootstrap
    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in phase_rows:
        if abs(float(row["logmar"]) - float(logmar)) > 0.01:
            continue
        key = (str(row["path_mode"]), int(row["source_orientation"]), int(row["target_orientation"]))
        groups[key][int(row["trace_id"])].append(row)

    for (path_mode, s_ori, t_ori), by_trace in sorted(groups.items()):
        trace_ids = sorted(by_trace)
        all_rows = [r for rs in by_trace.values() for r in rs]
        row_out: dict = {
            "logmar": logmar, "phase_condition": path_mode,
            "source_orientation": s_ori, "target_orientation": t_ori,
            "n_traces": len(trace_ids), "n_phases": len(all_rows),
        }
        for mk in metrics_keys:
            vals_all = [v for r in all_rows for v in [_get_metric(r, mk)] if v is not None]
            row_out[f"median_{mk}"] = float(np.nanmedian(vals_all)) if vals_all else float("nan")
            # Trace bootstrap
            if bootstrap_samples > 0 and len(trace_ids) > 0:
                boot: list[float] = []
                for _ in range(int(bootstrap_samples)):
                    samp_ids = rng.choice(len(trace_ids), size=len(trace_ids), replace=True)
                    samp_rows = [r for ti in samp_ids for r in by_trace[trace_ids[int(ti)]]]
                    v_samp = [v for r in samp_rows for v in [_get_metric(r, mk)] if v is not None]
                    boot.append(float(np.nanmedian(v_samp)) if v_samp else float("nan"))
                ba = np.asarray([v for v in boot if np.isfinite(v)])
                row_out[f"ci_low_{mk}"] = float(np.nanquantile(ba, 0.025)) if ba.size else float("nan")
                row_out[f"ci_high_{mk}"] = float(np.nanquantile(ba, 0.975)) if ba.size else float("nan")
            else:
                row_out[f"ci_low_{mk}"] = float("nan")
                row_out[f"ci_high_{mk}"] = float("nan")
        out.append(row_out)

    # Pooled rows across pairs for each path_mode
    pooled_groups: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in phase_rows:
        if abs(float(row["logmar"]) - float(logmar)) > 0.01:
            continue
        pooled_groups[str(row["path_mode"])][int(row["trace_id"])].append(row)
    for path_mode, by_trace in sorted(pooled_groups.items()):
        trace_ids = sorted(by_trace)
        all_rows = [r for rs in by_trace.values() for r in rs]
        row_out = {
            "logmar": logmar, "phase_condition": path_mode,
            "source_orientation": -1, "target_orientation": -1,
            "n_traces": len(trace_ids), "n_phases": len(all_rows),
        }
        for mk in metrics_keys:
            vals_all = [v for r in all_rows for v in [_get_metric(r, mk)] if v is not None]
            row_out[f"median_{mk}"] = float(np.nanmedian(vals_all)) if vals_all else float("nan")
            if bootstrap_samples > 0 and len(trace_ids) > 0:
                boot = []
                for _ in range(int(bootstrap_samples)):
                    samp_ids = rng.choice(len(trace_ids), size=len(trace_ids), replace=True)
                    samp_rows = [r for ti in samp_ids for r in by_trace[trace_ids[int(ti)]]]
                    v_samp = [v for r in samp_rows for v in [_get_metric(r, mk)] if v is not None]
                    boot.append(float(np.nanmedian(v_samp)) if v_samp else float("nan"))
                ba = np.asarray([v for v in boot if np.isfinite(v)])
                row_out[f"ci_low_{mk}"] = float(np.nanquantile(ba, 0.025)) if ba.size else float("nan")
                row_out[f"ci_high_{mk}"] = float(np.nanquantile(ba, 0.975)) if ba.size else float("nan")
            else:
                row_out[f"ci_low_{mk}"] = float("nan"); row_out[f"ci_high_{mk}"] = float("nan")
        out.append(row_out)

    return out


# ---------------------------------------------------------------------------
# Task 2. Identity-vector cosine across phases
# ---------------------------------------------------------------------------

def _identity_vector_cosine_summary(cosine_rows: list[dict], bootstrap_samples: int, bootstrap_seed: int) -> list[dict]:
    """Aggregate per-trace phase-pair cosines by (logmar, source_ori, target_ori, path_mode)."""
    rng = np.random.default_rng(int(bootstrap_seed))
    groups: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in cosine_rows:
        key = (float(row["logmar"]), str(row["path_mode"]),
               int(row["source_orientation"]), int(row["target_orientation"]))
        groups[key][int(row["trace_id"])].extend(
            [float(v) for v in str(row.get("pairwise_cosines_json", "[]")).strip("[]").split(",")
             if v.strip() and v.strip() not in ("", "nan")]
        )

    out: list[dict] = []
    for (logmar, path_mode, s_ori, t_ori), by_trace in sorted(groups.items()):
        trace_ids = sorted(by_trace)
        all_cosines = [v for vals in by_trace.values() for v in vals]
        if not all_cosines:
            continue
        arr = np.asarray(all_cosines)
        frac_neg = float(np.mean(arr < 0))

        boot_meds: list[float] = []
        for _ in range(max(int(bootstrap_samples), 0)):
            samp_idx = rng.choice(len(trace_ids), size=len(trace_ids), replace=True)
            samp = [v for ti in samp_idx for v in by_trace[trace_ids[int(ti)]]]
            boot_meds.append(float(np.nanmedian(samp)) if samp else float("nan"))
        ba = np.asarray([v for v in boot_meds if np.isfinite(v)])
        out.append({
            "logmar": logmar, "path_mode": path_mode,
            "source_orientation": s_ori, "target_orientation": t_ori,
            "n_traces": len(trace_ids),
            "n_phase_pairs_total": len(all_cosines),
            "median_cos_dab_phase_pair": float(np.nanmedian(arr)),
            "ci_low": float(np.nanquantile(ba, 0.025)) if ba.size else float("nan"),
            "ci_high": float(np.nanquantile(ba, 0.975)) if ba.size else float("nan"),
            "fraction_negative_cosine": frac_neg,
            "fraction_positive_cosine": 1.0 - frac_neg,
        })
    return out


# ---------------------------------------------------------------------------
# Task 3. Nonlinear energy readout
# ---------------------------------------------------------------------------

def _identity_energy_readout_summary(energy_rows: list[dict], bootstrap_samples: int, bootstrap_seed: int) -> list[dict]:
    """Aggregate per-trace energy readout by (logmar, source_ori, target_ori, path_mode)."""
    rng = np.random.default_rng(int(bootstrap_seed))
    metric_keys = ("raw_energy_readout", "orthogonal_energy_readout",
                   "gain_raw_vs_stabilized", "gain_orthogonal_vs_stabilized")
    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in energy_rows:
        key = (float(row["logmar"]), str(row["path_mode"]),
               int(row["source_orientation"]), int(row["target_orientation"]))
        groups[key][int(row["trace_id"])].append(row)

    out: list[dict] = []
    for (logmar, path_mode, s_ori, t_ori), by_trace in sorted(groups.items()):
        trace_ids = sorted(by_trace)
        all_rows = [r for rs in by_trace.values() for r in rs]

        def _med(mk: str, rows: list[dict]) -> float:
            vals = [float(r[mk]) for r in rows if str(r.get(mk, "")) not in ("", "nan", "NaN")]
            return float(np.nanmedian(vals)) if vals else float("nan")

        boot: dict[str, list[float]] = {mk: [] for mk in metric_keys}
        for _ in range(max(int(bootstrap_samples), 0)):
            samp_idx = rng.choice(len(trace_ids), size=len(trace_ids), replace=True)
            samp_rows = [r for ti in samp_idx for r in by_trace[trace_ids[int(ti)]]]
            for mk in metric_keys:
                boot[mk].append(_med(mk, samp_rows))

        row_out: dict = {
            "logmar": logmar, "path_mode": path_mode,
            "source_orientation": s_ori, "target_orientation": t_ori,
            "n_traces": len(trace_ids), "n_rows": len(all_rows),
        }
        for mk in metric_keys:
            row_out[f"median_{mk}"] = _med(mk, all_rows)
            ba = np.asarray([v for v in boot[mk] if np.isfinite(v)])
            row_out[f"ci_low_{mk}"] = float(np.nanquantile(ba, 0.025)) if ba.size else float("nan")
            row_out[f"ci_high_{mk}"] = float(np.nanquantile(ba, 0.975)) if ba.size else float("nan")
        out.append(row_out)
    return out


# ---------------------------------------------------------------------------
# Task 4. Single-unit orientation tuning
# ---------------------------------------------------------------------------

def _single_unit_tuning(
    runner: "SeparabilityRunner",
    logmar: float,
    orientations: list[int],
    stabilized_position: np.ndarray,
) -> list[dict]:
    """Per-unit orientation tuning at center phase.

    OSI = response_range / (response_mean + eps).
    Identifies whether center degeneracy is from weak single-unit tuning or
    population geometry averaging despite tuned units.
    """
    responses = {ori: runner.get_response(logmar, ori, stabilized_position)
                 for ori in orientations}
    valid = {ori: r for ori, r in responses.items() if r is not None}
    if not valid:
        return []

    n_units = next(iter(valid.values())).shape[0]
    oris = sorted(valid)
    R = np.stack([valid[o].astype(np.float64) for o in oris], axis=1)  # (n_units, n_oris)

    out: list[dict] = []
    for unit_id in range(n_units):
        r_u = R[unit_id, :]
        r_range = float(r_u.max() - r_u.min())
        r_mean = float(r_u.mean())
        osi = r_range / (r_mean + EPS)
        pref_idx = int(np.argmax(r_u))
        out.append({
            "unit_id": unit_id,
            "logmar": logmar,
            "response_range": r_range,
            "response_mean": r_mean,
            "response_range_over_mean": osi,
            "orientation_selectivity_index": osi,
            "preferred_orientation": oris[pref_idx],
            **{f"mean_response_{o}": float(R[unit_id, i]) for i, o in enumerate(oris)},
        })
    return out


def _center_orientation_projections(runner: "SeparabilityRunner", logmar: float,
                                    orientations: list[int], U_id: np.ndarray,
                                    stabilized_position: np.ndarray) -> list[dict]:
    """Project each orientation's center response onto the first identity axis u1."""
    if U_id.shape[1] == 0:
        return []
    u1 = U_id[:, 0]
    out: list[dict] = []
    for ori in orientations:
        r = runner.get_response(logmar, ori, stabilized_position)
        if r is not None:
            out.append({"logmar": logmar, "orientation": ori, "u1_projection": float(np.dot(u1, r))})
    return out


def _make_figures(
    output_dir: Path,
    figures_dir: Path,
    idspace_summary_rows: list[dict],
    orientation_proj_rows: list[dict],
    fraction_summary_rows: list[dict],
    sign_summary_rows: list[dict],
    headline_rows: list[dict],
    headline_pooled_rows: list[dict],
    pair_axis_rows: list[dict],
    center_deg_rows: list[dict],
    offcenter_rows: list[dict],
    skip_figures: bool,
    energy_rows_arg: list[dict] | None = None,
    cosine_summary_rows: list[dict] | None = None,
    unit_tuning_rows_arg: list[dict] | None = None,
) -> None:
    if skip_figures:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energy_rows_arg = energy_rows_arg or []
    cosine_summary_rows = cosine_summary_rows or []
    unit_tuning_rows_arg = unit_tuning_rows_arg or []

    _COLOR = {"drift_step_start_phases": "tab:blue", "all_steps": "tab:orange"}

    def _pool_by_lm_mode(rows: list[dict], metric: str) -> dict[tuple, list[float]]:
        grouped: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            v = r.get(f"median_{metric}", "")
            if str(v) not in ("", "nan", "NaN"):
                grouped[(float(r["logmar"]), str(r["path_mode"]))].append(float(v))
        return grouped

    # --- Figure 1: Orthogonal fraction by logmar × path_mode (MAIN FIGURE) ---
    if headline_pooled_rows:
        keys = sorted({(float(r["logmar"]), str(r["path_mode"])) for r in headline_pooled_rows})
        labels = [f"lm={k[0]:+.2f}\n{k[1].replace('_',' ')}" for k in keys]
        vals, ci_lo, ci_hi = [], [], []
        for k in keys:
            matching = [r for r in headline_pooled_rows if float(r["logmar"]) == k[0] and r["path_mode"] == k[1]]
            if matching:
                r = matching[0]
                vals.append(float(r.get("median_orthogonal_fraction", float("nan"))))
                ci_lo.append(float(r.get("ci_low_orthogonal_fraction", float("nan"))))
                ci_hi.append(float(r.get("ci_high_orthogonal_fraction", float("nan"))))
            else:
                vals.append(float("nan")); ci_lo.append(float("nan")); ci_hi.append(float("nan"))
        vals_a = np.asarray(vals); ci_lo_a = np.asarray(ci_lo); ci_hi_a = np.asarray(ci_hi)
        yerr = np.vstack([np.maximum(0, vals_a - ci_lo_a), np.maximum(0, ci_hi_a - vals_a)])
        yerr[~np.isfinite(yerr)] = 0.0
        colors = [_COLOR.get(k[1], "gray") for k in keys]
        fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(keys)), 4.5))
        ax.bar(range(len(keys)), vals, color=colors, alpha=0.85, yerr=yerr, capsize=4)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Median orthogonal fraction (pooled across pairs)")
        ax.set_ylim(0, 1)
        ax.set_title("Identity energy outside local translation tangent\n(blue=drift, orange=all_steps)")
        fig.tight_layout()
        fig.savefig(figures_dir / "orthogonal_fraction_by_logmar_pathmode.png", dpi=180)
        plt.close(fig)

    # --- Figure 2: Absolute identity energy by logmar × path_mode ---
    if headline_pooled_rows:
        grouped_e = _pool_by_lm_mode(headline_pooled_rows, "identity_energy")
        keys = sorted(grouped_e)
        vals = [float(np.nanmedian(grouped_e[k])) for k in keys]
        labels = [f"lm={k[0]:+.2f}\n{k[1].replace('_',' ')}" for k in keys]
        colors = [_COLOR.get(k[1], "gray") for k in keys]
        fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(keys)), 4.5))
        ax.bar(range(len(keys)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Median ||d_ab||² (identity energy, pooled across pairs)")
        ax.set_title("Absolute identity difference energy by regime")
        fig.tight_layout()
        fig.savefig(figures_dir / "identity_energy_by_logmar_pathmode.png", dpi=180)
        plt.close(fig)

    # --- Figure 3: Orthogonal energy by logmar × path_mode ---
    if headline_pooled_rows:
        grouped_o = _pool_by_lm_mode(headline_pooled_rows, "orthogonal_energy")
        keys = sorted(grouped_o)
        vals = [float(np.nanmedian(grouped_o[k])) for k in keys]
        labels = [f"lm={k[0]:+.2f}\n{k[1].replace('_',' ')}" for k in keys]
        colors = [_COLOR.get(k[1], "gray") for k in keys]
        fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(keys)), 4.5))
        ax.bar(range(len(keys)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Median ||P_perp d_ab||² (orthogonal energy, pooled)")
        ax.set_title("Absolute translation-discounted identity energy by regime")
        fig.tight_layout()
        fig.savefig(figures_dir / "orthogonal_energy_by_logmar_pathmode.png", dpi=180)
        plt.close(fig)

    # --- Figure 4: Pair-specific cancellation index (drift mode) ---
    if pair_axis_rows:
        rows_drift = [r for r in pair_axis_rows if "drift" in str(r.get("path_mode", ""))]
        if rows_drift:
            logmars_lm = sorted({float(r["logmar"]) for r in rows_drift})
            fig, axes = plt.subplots(1, len(logmars_lm), figsize=(5.5 * len(logmars_lm), 4.5), squeeze=False)
            for ax, lm in zip(axes[0], logmars_lm):
                sub = sorted([r for r in rows_drift if float(r["logmar"]) == lm],
                              key=lambda r: (int(r["source_orientation"]), int(r["target_orientation"])))
                pairs = [f"({r['source_orientation']},{r['target_orientation']})" for r in sub]
                ci = [float(r["cancellation_index"]) for r in sub]
                colors_ci = ["tab:red" if c < 0.3 else ("tab:olive" if c < 0.7 else "tab:green") for c in ci]
                ax.barh(pairs, ci, color=colors_ci, alpha=0.85)
                ax.axvline(0.5, color="0.4", linestyle="--", linewidth=0.8)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Cancellation index\n(0=sign-flip cancels, 1=coherent)")
                ax.set_title(f"logmar={lm:+.2f}\nPair-axis cancellation (drift)")
            fig.tight_layout()
            fig.savefig(figures_dir / "pair_specific_cancellation_index_by_pair.png", dpi=180)
            plt.close(fig)

    # --- Figure 5: Pair-specific unsigned identity signal (drift mode) ---
    if pair_axis_rows:
        rows_drift = [r for r in pair_axis_rows if "drift" in str(r.get("path_mode", ""))]
        if rows_drift:
            logmars_lm = sorted({float(r["logmar"]) for r in rows_drift})
            fig, axes = plt.subplots(1, len(logmars_lm), figsize=(5.5 * len(logmars_lm), 4.5), squeeze=False)
            for ax, lm in zip(axes[0], logmars_lm):
                sub = sorted([r for r in rows_drift if float(r["logmar"]) == lm],
                              key=lambda r: (int(r["source_orientation"]), int(r["target_orientation"])))
                pairs = [f"({r['source_orientation']},{r['target_orientation']})" for r in sub]
                u_sig = [float(r["median_unsigned_identity_orth_scalar"]) for r in sub]
                ax.barh(pairs, u_sig, color="tab:purple", alpha=0.85)
                ax.set_xlabel("|u_ab_center^T f_perp| (unsigned orth signal, drift)")
                ax.set_title(f"logmar={lm:+.2f}\nPair-axis unsigned signal")
            fig.tight_layout()
            fig.savefig(figures_dir / "pair_specific_unsigned_identity_signal_by_pair.png", dpi=180)
            plt.close(fig)

    # --- Figure 6: Center vs off-center identity distance ---
    if center_deg_rows and offcenter_rows:
        all_lm_pairs = sorted({(float(r["logmar"]), int(r["source_orientation"]), int(r["target_orientation"]))
                                for r in center_deg_rows})
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        x_labels = [f"({s},{t})\nlm={lm:+.2f}" for lm, s, t in all_lm_pairs]
        xs = np.arange(len(x_labels))
        # Center distances
        center_dist = []
        for lm, s, t in all_lm_pairs:
            match = [r for r in center_deg_rows if float(r["logmar"]) == lm
                     and int(r["source_orientation"]) == s and int(r["target_orientation"]) == t]
            center_dist.append(float(match[0]["pairwise_distance_center"]) if match else float("nan"))
        # Drift median delta_mu_norm
        drift_dist = []
        for lm, s, t in all_lm_pairs:
            match = [r for r in offcenter_rows if float(r["logmar"]) == lm
                     and str(r.get("path_mode", "")) == "drift_step_start_phases"
                     and int(r["source_orientation"]) == s and int(r["target_orientation"]) == t]
            drift_dist.append(float(match[0]["median_delta_mu_norm"]) if match else float("nan"))
        w = 0.35
        ax.bar(xs - w / 2, center_dist, width=w, label="center (p0)", color="tab:gray", alpha=0.85)
        ax.bar(xs + w / 2, drift_dist, width=w, label="drift phases", color="tab:blue", alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("||d_ab|| (identity distance)")
        ax.set_title("Center vs drift-phase pairwise identity distance")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / "center_vs_offcenter_identity_distance.png", dpi=180)
        plt.close(fig)

    # --- Figure: center_vs_drift_identity_norm_by_logmar (MAIN FIGURE) ---
    # Pool across pairs: compare center vs drift vs all_steps median ||d_ab||
    if "fem_summary_rows" in dir():  # populated externally
        pass
    elif center_deg_rows and offcenter_rows:
        logmars_seen = sorted({float(r["logmar"]) for r in center_deg_rows})
        conditions = ["center", "drift_step_start_phases", "all_steps"]
        cond_labels = ["center (stabilized)", "drift phases", "all steps"]
        cond_colors = ["tab:gray", "tab:blue", "tab:orange"]
        fig, axes = plt.subplots(1, len(logmars_seen), figsize=(5.5 * len(logmars_seen), 4.5), squeeze=False)
        for ax, lm in zip(axes[0], logmars_seen):
            y_vals: list[float] = []
            for cond in conditions:
                if cond == "center":
                    vals = [float(r["pairwise_distance_center"]) for r in center_deg_rows
                            if abs(float(r["logmar"]) - lm) < 0.01]
                else:
                    vals = [float(r["median_delta_mu_norm"]) for r in offcenter_rows
                            if abs(float(r["logmar"]) - lm) < 0.01 and str(r.get("path_mode", "")) == cond]
                y_vals.append(float(np.nanmedian(vals)) if vals else float("nan"))
            ax.bar(range(len(conditions)), y_vals, color=cond_colors, alpha=0.85)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(cond_labels, fontsize=9, rotation=15, ha="right")
            ax.set_ylabel("Median ||d_ab|| (identity distance)")
            ax.set_title(f"logmar={lm:+.2f}")
        fig.suptitle("Identity distance: center vs FEM phases")
        fig.tight_layout()
        fig.savefig(figures_dir / "center_vs_drift_identity_norm_by_logmar.png", dpi=180)
        plt.close(fig)

    # --- Figure: energy readout by condition ---
    if energy_rows_arg:
        from collections import defaultdict as _dd3
        erg: dict[tuple, list[float]] = _dd3(list)
        for row in energy_rows_arg:
            key = (float(row["logmar"]), str(row["path_mode"]))
            v = row.get("gain_orthogonal_vs_stabilized", "")
            if str(v) not in ("", "nan", "NaN"):
                erg[key].append(float(v))
        keys = sorted(erg)
        labels = [f"lm={k[0]:+.2f}\n{k[1].replace('_',' ')}" for k in keys]
        vals = [float(np.nanmedian(erg[k])) for k in keys]
        colors = [_COLOR.get(k[1], "gray") for k in keys]
        fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(keys)), 4.5))
        ax.bar(range(len(keys)), vals, color=colors, alpha=0.85)
        ax.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Median gain (energy readout / center)")
        ax.set_title("Energy readout gain vs stabilized center\n(blue=drift, orange=all_steps)")
        fig.tight_layout()
        fig.savefig(figures_dir / "identity_energy_readout_by_condition.png", dpi=180)
        plt.close(fig)

    # --- Figure: identity vector phase cosine by pair ---
    if cosine_summary_rows:
        rows_drift = [r for r in cosine_summary_rows if "drift" in str(r.get("path_mode", ""))]
        if rows_drift:
            logmars_lm = sorted({float(r["logmar"]) for r in rows_drift})
            fig, axes = plt.subplots(1, len(logmars_lm), figsize=(5.5 * len(logmars_lm), 4.5), squeeze=False)
            for ax, lm in zip(axes[0], logmars_lm):
                sub = sorted([r for r in rows_drift if float(r["logmar"]) == lm],
                              key=lambda r: (int(r["source_orientation"]), int(r["target_orientation"])))
                pairs = [f"({r['source_orientation']},{r['target_orientation']})" for r in sub]
                med_cos = [float(r["median_cos_dab_phase_pair"]) for r in sub]
                bar_colors = ["tab:green" if c > 0.5 else ("tab:olive" if c > 0 else "tab:red") for c in med_cos]
                ax.barh(pairs, med_cos, color=bar_colors, alpha=0.85)
                ax.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
                ax.set_xlim(-1, 1)
                ax.set_xlabel("Median pairwise cosine(d_ab(p_i), d_ab(p_j))")
                ax.set_title(f"logmar={lm:+.2f}\nIdentity-vector phase consistency (drift)")
            fig.tight_layout()
            fig.savefig(figures_dir / "identity_vector_phase_cosine_by_pair.png", dpi=180)
            plt.close(fig)

    # --- Figure: single-unit OSI distribution ---
    if unit_tuning_rows_arg:
        logmars_lm = sorted({float(r["logmar"]) for r in unit_tuning_rows_arg})
        fig, axes = plt.subplots(1, len(logmars_lm), figsize=(5.0 * len(logmars_lm), 4.0), squeeze=False)
        for ax, lm in zip(axes[0], logmars_lm):
            sub = [r for r in unit_tuning_rows_arg if abs(float(r["logmar"]) - lm) < 0.01]
            osi = [float(r["orientation_selectivity_index"]) for r in sub]
            ax.hist(osi, bins=30, color="tab:purple", alpha=0.8, edgecolor="none")
            ax.axvline(float(np.nanmedian(osi)), color="k", linestyle="--", linewidth=1.0, label=f"median={np.nanmedian(osi):.2f}")
            ax.set_xlabel("OSI = response_range / response_mean")
            ax.set_ylabel("Units")
            ax.set_title(f"logmar={lm:+.2f}")
            ax.legend(frameon=False)
        fig.suptitle("Single-unit E-optotype orientation selectivity at center")
        fig.tight_layout()
        fig.savefig(figures_dir / "single_unit_orientation_selectivity_distribution.png", dpi=180)
        plt.close(fig)

    # --- Legacy spectrum and orientation projection figures ---
    if idspace_summary_rows:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        import json as _json
        for idx, row in enumerate(idspace_summary_rows):
            sv = _json.loads(row.get("singular_values_json", "[]"))
            if not sv:
                continue
            lm = float(row["logmar"])
            energy = np.asarray([s ** 2 for s in sv], dtype=np.float64)
            frac = energy / (energy.sum() + EPS)
            xs_sv = np.arange(1, len(frac) + 1)
            ax.bar(xs_sv - 0.15 + 0.3 * idx, frac, width=0.3, label=f"logmar={lm:+.2f}", alpha=0.8)
        ax.set_xlabel("Singular value rank"); ax.set_ylabel("Fraction of identity energy")
        ax.set_title("Center identity subspace spectrum"); ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / "identity_subspace_spectrum_by_logmar.png", dpi=180)
        plt.close(fig)

    if orientation_proj_rows:
        logmars_seen = sorted({float(r["logmar"]) for r in orientation_proj_rows})
        fig, axes = plt.subplots(1, len(logmars_seen), figsize=(4.5 * len(logmars_seen), 4.0), squeeze=False)
        for ax, lm in zip(axes[0], logmars_seen):
            rows_lm = [r for r in orientation_proj_rows if float(r["logmar"]) == lm]
            oris = [int(r["orientation"]) for r in rows_lm]
            projs = [float(r["u1_projection"]) for r in rows_lm]
            ax.bar([str(o) + "°" for o in oris], projs, color="tab:blue", alpha=0.8)
            ax.axhline(0, color="0.4", linewidth=0.8)
            ax.set_title(f"logmar={lm:+.2f}"); ax.set_xlabel("Orientation"); ax.set_ylabel("u₁ projection")
        fig.suptitle("Center response projected onto first identity axis (u₁)")
        fig.tight_layout()
        fig.savefig(figures_dir / "identity_center_projection_by_orientation.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# README and output helpers
# ---------------------------------------------------------------------------

def _write_readme(output_dir: Path, config: dict) -> None:
    n_pairs = len(list(itertools.combinations(config.get("orientations", []), 2)))
    lines = [
        "# FEM path-integrated separability",
        "",
        "## Scope",
        f"- LogMARs: {config.get('logmars')}",
        f"- Orientations: {config.get('orientations')}",
        f"- Identity pairs: all pairwise ({n_pairs} pairs)",
        f"- Max traces: {config.get('max_traces')}",
        f"- T (samples per path): {config.get('T')}",
        f"- Sampling conditions: {config.get('sampling_conditions')}",
        f"- Path modes: {config.get('path_modes')}",
        f"- sigma0: {config.get('sigma0')}",
        f"- Identity subspace mode: {config.get('identity_subspace_mode')} (energy threshold: {config.get('identity_subspace_energy')})",
        "",
        "## Identity subspace (center mode)",
        "  U_id is built from pairwise identity differences at the stabilized center position [0,0].",
        "  This is leakage-controlled: U_id does not depend on the FEM trace positions being evaluated.",
        "  P_id projects onto the identity-relevant directions visible from the canonical stabilized view.",
        "",
        "## IMPORTANT — phase_shuffled_path = real_fem_path by construction",
        "  F = sum_t f_t is commutative; shuffled order gives the same result.",
        "  This is a phase-set control, NOT a temporal-order control.",
        "",
        "## Primary metric: gain_vs_stabilized_repeated_idspace",
        "  > 1: drift-scale FEM phase sampling provides more translation-discounted",
        "       identity signal than T repeated samples at the stabilized center.",
        "  Low complementarity_ratio_idspace: gain is magnitude-driven (stronger",
        "  per-phase signal), not coherent-accumulation-driven.",
        "",
        "## Orthogonal complement note",
        "  Full-space P_perp is (N-2)-dimensional and includes high-dimensional residual.",
        "  idspace metrics restrict to U_id (rank ~3 for 4 orientations) and are identity-specific.",
        "  Use idspace metrics for manuscript claims.",
        "",
        "## Guardrails",
        "  Do NOT claim: FEMs are optimized, trajectory order matters, or coherent accumulation.",
        "  Preferred wording if gain_idspace > 1:",
        "    Drift-scale FEM sampling increases identity readout by visiting phases with",
        "    stronger translation-discounted identity signal, rather than by coherently",
        "    accumulating aligned identity vectors across phases.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logmars", default=",".join(f"{x:+.2f}" for x in DEFAULT_LOGMARS))
    parser.add_argument("--orientations", default=",".join(str(x) for x in DEFAULT_ORIENTATIONS))
    parser.add_argument("--max-traces", type=int, default=DEFAULT_MAX_TRACES)
    parser.add_argument("--T", type=int, default=DEFAULT_T, dest="T")
    parser.add_argument("--sigma0", type=float, default=DEFAULT_SIGMA0)
    parser.add_argument("--drift-max-arcmin", type=float, default=DEFAULT_DRIFT_MAX_ARCMIN)
    parser.add_argument("--jacobian-step-px", type=float, default=DEFAULT_JACOBIAN_STEP_PX)
    parser.add_argument("--pixels-per-degree", type=float, default=DEFAULT_PPD)
    parser.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS)
    parser.add_argument("--center-mode", default=DEFAULT_CENTER_MODE,
                        choices=["raw", "subtract_trace_mean", "subtract_first_sample"])
    parser.add_argument("--eye-traces-path", type=Path, default=EYE_TRACES_PATH)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--path-modes", default=",".join(DEFAULT_PATH_MODES))
    parser.add_argument("--sampling-conditions", default=",".join(DEFAULT_SAMPLING_CONDITIONS))
    parser.add_argument("--identity-subspace-mode", default="none",
                        choices=["center", "none"])
    parser.add_argument("--identity-subspace-energy", type=float, default=DEFAULT_ID_SUBSPACE_ENERGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logmars = _parse_csv_floats(args.logmars)
    orientations = list(_parse_csv_ints(args.orientations))
    path_modes = [m.strip() for m in args.path_modes.split(",")]
    sampling_conditions = [c.strip() for c in args.sampling_conditions.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and eye traces ...", flush=True)
    device = _pick_device()
    runner = SeparabilityRunner(device=device, pixels_per_degree=float(args.pixels_per_degree),
                                n_lags=int(args.n_lags), jacobian_step_px=float(args.jacobian_step_px))
    traces_deg, durations = _load_eye_traces(args.eye_traces_path)
    print(f"Loaded {traces_deg.shape[0]} traces (max length {traces_deg.shape[1]}).", flush=True)

    all_pair_rows: list[dict] = []
    all_phase_rows: list[dict] = []
    all_mech_rows: list[dict] = []
    all_sanity_rows: list[dict] = []
    all_ori_proj_rows: list[dict] = []
    all_center_deg_rows: list[dict] = []
    all_energy_rows: list[dict] = []
    all_cosine_rows: list[dict] = []
    all_unit_tuning_rows: list[dict] = []
    all_fem_sampling_summary: list[dict] = []
    idspace_summary_rows: list[dict] = []

    for logmar in logmars:
        print(f"\n=== LogMAR {logmar:+.2f} ===", flush=True)
        runner.clear_cache()
        (pair_rows, phase_rows, mech_rows, sanity_rows, idspace_meta,
         ori_proj_rows, center_deg_rows, energy_rows, cosine_rows,
         unit_tuning_rows, center_pair_pm) = _run_logmar(
            runner=runner, logmar=float(logmar), orientations=orientations,
            traces_deg=traces_deg, durations=durations,
            T=int(args.T), sigma0=float(args.sigma0),
            path_modes=path_modes, sampling_conditions=sampling_conditions,
            pixels_per_degree=float(args.pixels_per_degree), jacobian_step_px=float(args.jacobian_step_px),
            drift_max_arcmin=float(args.drift_max_arcmin), center_mode=args.center_mode,
            max_traces=int(args.max_traces) if args.max_traces is not None else None,
            bootstrap_samples=int(args.bootstrap_samples), bootstrap_seed=int(args.bootstrap_seed),
            identity_subspace_mode=args.identity_subspace_mode,
            identity_subspace_energy=float(args.identity_subspace_energy),
        )
        # Task 1: magnitude summary for this logmar
        all_fem_sampling_summary.extend(
            _fem_sampling_identity_summary(center_pair_pm, phase_rows, float(logmar),
                                           int(args.bootstrap_samples), int(args.bootstrap_seed)))
        all_pair_rows.extend(pair_rows)
        all_phase_rows.extend(phase_rows)
        all_mech_rows.extend(mech_rows)
        all_sanity_rows.extend(sanity_rows)
        all_ori_proj_rows.extend(ori_proj_rows)
        all_center_deg_rows.extend(center_deg_rows)
        all_energy_rows.extend(energy_rows)
        all_cosine_rows.extend(cosine_rows)
        all_unit_tuning_rows.extend(unit_tuning_rows)
        idspace_summary_rows.append(idspace_meta)
        print(f"  LogMAR {logmar:+.2f}: {len(pair_rows)} pair rows, {len(phase_rows)} phase rows, "
              f"{len(energy_rows)} energy rows, {len(cosine_rows)} cosine rows.", flush=True)

    print("\nComputing summaries ...", flush=True)
    summary_rows = _bootstrap_summary(all_pair_rows, int(args.bootstrap_samples), int(args.bootstrap_seed))
    summary_by_pair_rows = _bootstrap_summary_by_pair(all_pair_rows, int(args.bootstrap_samples), int(args.bootstrap_seed))
    mech_summary_rows = _mechanism_summary(all_mech_rows)
    sign_summary_rows = _sign_summary(all_phase_rows)
    fraction_summary_rows = _idspace_fraction_summary(all_phase_rows, int(args.bootstrap_samples), int(args.bootstrap_seed))
    headline_rows = _headline_decomposition_summary(all_phase_rows, int(args.bootstrap_samples), int(args.bootstrap_seed), groupby_pair=True)
    headline_pooled_rows = _headline_decomposition_summary(all_phase_rows, int(args.bootstrap_samples), int(args.bootstrap_seed), groupby_pair=False)
    pair_axis_rows = _pair_axis_summary(all_phase_rows)
    offcenter_rows = _offcenter_identity_summary(all_phase_rows, int(args.bootstrap_samples), int(args.bootstrap_seed))
    # Tasks 2, 3, 4
    cosine_summary_rows = _identity_vector_cosine_summary(all_cosine_rows, int(args.bootstrap_samples), int(args.bootstrap_seed))
    energy_summary_rows = _identity_energy_readout_summary(all_energy_rows, int(args.bootstrap_samples), int(args.bootstrap_seed))

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    _write_csv(output_dir / "local_separability_by_phase.csv", all_phase_rows)
    _write_csv(output_dir / "path_separability_by_pair.csv", all_pair_rows)
    _write_csv(output_dir / "path_separability_summary.csv", summary_rows)
    _write_csv(output_dir / "path_separability_summary_by_pair.csv", summary_by_pair_rows)
    _write_csv(output_dir / "translation_mimicry_mechanism_by_phase.csv", all_mech_rows)
    _write_csv(output_dir / "translation_mimicry_mechanism_summary.csv", mech_summary_rows)
    _write_csv(output_dir / "identity_subspace_summary.csv", idspace_summary_rows)
    _write_csv(output_dir / "identity_axis_sign_summary_by_pair.csv", sign_summary_rows)
    _write_csv(output_dir / "idspace_fraction_summary_by_group.csv", fraction_summary_rows)
    _write_csv(output_dir / "identity_center_projection_by_orientation.csv", all_ori_proj_rows)
    _write_csv(output_dir / "identity_translation_decomposition_summary.csv", headline_rows)
    _write_csv(output_dir / "identity_translation_decomposition_summary_pooled.csv", headline_pooled_rows)
    _write_csv(output_dir / "pair_specific_identity_axis_summary.csv", pair_axis_rows)
    _write_csv(output_dir / "center_identity_degeneracy_summary.csv", all_center_deg_rows)
    _write_csv(output_dir / "offcenter_identity_distance_summary.csv", offcenter_rows)
    _write_csv(output_dir / "sanity_checks.csv", all_sanity_rows)
    # Tasks 1, 2, 3, 4
    _write_csv(output_dir / "fem_sampling_identity_magnitude_summary.csv", all_fem_sampling_summary)
    _write_csv(output_dir / "identity_vector_phase_cosine_by_pair.csv", all_cosine_rows)
    _write_csv(output_dir / "identity_vector_phase_cosine_summary.csv", cosine_summary_rows)
    _write_csv(output_dir / "identity_energy_readout_by_pair.csv", all_energy_rows)
    _write_csv(output_dir / "identity_energy_readout_summary.csv", energy_summary_rows)
    _write_csv(output_dir / "single_unit_eoptotype_orientation_tuning.csv", all_unit_tuning_rows)

    _make_figures(output_dir, figures_dir, idspace_summary_rows, all_ori_proj_rows,
                  fraction_summary_rows, sign_summary_rows,
                  headline_rows, headline_pooled_rows, pair_axis_rows,
                  all_center_deg_rows, offcenter_rows, args.skip_figures,
                  energy_rows_arg=all_energy_rows,
                  cosine_summary_rows=cosine_summary_rows,
                  unit_tuning_rows_arg=all_unit_tuning_rows)

    invariance_ok = all(bool(r.get("invariance_verified", False)) for r in all_sanity_rows)
    if all_sanity_rows:
        max_err = max(float(r.get("max_abs_error") or 0.0) for r in all_sanity_rows)
        print(f"\nSanity check (sum invariance): {'PASS' if invariance_ok else 'FAIL'}, max_abs_error={max_err:.2e}", flush=True)

    config: dict = {
        "logmars": list(logmars), "orientations": orientations,
        "max_traces": int(args.max_traces) if args.max_traces is not None else None,
        "T": int(args.T), "sigma0": float(args.sigma0),
        "drift_max_arcmin": float(args.drift_max_arcmin),
        "jacobian_step_px": float(args.jacobian_step_px),
        "pixels_per_degree": float(args.pixels_per_degree), "n_lags": int(args.n_lags),
        "center_mode": args.center_mode, "eye_traces_path": str(args.eye_traces_path),
        "path_modes": path_modes, "sampling_conditions": sampling_conditions,
        "bootstrap_samples": int(args.bootstrap_samples), "bootstrap_seed": int(args.bootstrap_seed),
        "identity_subspace_mode": args.identity_subspace_mode,
        "identity_subspace_energy": float(args.identity_subspace_energy),
        "identity_subspace_construction": "center_position_pairwise_differences" if args.identity_subspace_mode == "center" else "none",
        "leakage_controlled": args.identity_subspace_mode == "center",
        "orthogonal_mode": "full_population_space_and_idspace",
        "jacobian_mode": "finite_difference_at_phase_position",
        "stabilized_position_px": [0.0, 0.0],
        "trace_prep": {"duration_trim": True, "finite_mask": True, "no_gap_bridging": True, "center_mode": args.center_mode},
        "sampling_condition_notes": {
            "real_fem_path": "positions in original trace order",
            "stabilized_repeated": "screen center (0,0) repeated T times",
            "phase_shuffled_path": "same phase set as real_fem_path, shuffled order; equals real_fem_path by construction (order-invariant sum readout)",
        },
        "path_mode_definitions": {
            "drift_step_start_phases": "p_t for adjacent steps <= drift_max_arcmin; step starts, not contiguous",
            "all_steps": "p_t for all valid adjacent step-start positions",
        },
        "sum_invariance_verified": invariance_ok,
        "min_true_norm": MIN_TRUE_NORM,
        "n_pair_rows": len(all_pair_rows), "n_phase_rows": len(all_phase_rows),
        "n_mech_rows": len(all_mech_rows), "n_summary_rows": len(summary_rows),
        "n_sign_summary_rows": len(sign_summary_rows),
        "n_fraction_summary_rows": len(fraction_summary_rows),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    _write_readme(output_dir, config)

    print(f"\nSaved to {output_dir}")
    if summary_rows:
        print("\n=== Summary: gain_vs_stabilized_repeated and gain_vs_stabilized_repeated_idspace ===")
        for row in summary_rows:
            cond = row.get("sampling_condition", "?"); mode = row.get("path_mode", "?")
            if str(cond) not in ("real_fem_path", "phase_shuffled_path"):
                continue
            g = float(row.get("median_gain_vs_stabilized_repeated", float("nan")))
            g_id = float(row.get("median_gain_vs_stabilized_repeated_idspace", float("nan")))
            ci_lo = float(row.get("ci_low_gain_vs_stabilized_repeated_idspace", float("nan")))
            ci_hi = float(row.get("ci_high_gain_vs_stabilized_repeated_idspace", float("nan")))
            note = " [= real_fem]" if cond == "phase_shuffled_path" else ""
            lm = row.get("logmar", "?")
            print(f"  lm={lm} {cond}/{mode}: gain_full={g:.3f}  gain_id={g_id:.3f} [{ci_lo:.3f},{ci_hi:.3f}]{note}")

    if mech_summary_rows:
        print("\n=== Mechanism summary (all pairs, drift mode) ===")
        print(f"  {'pair':>12}  {'med|cos_x|':>10}  {'med|cos_y|':>10}  {'frac_x':>7}  {'med_mimicry':>11}")
        for row in mech_summary_rows:
            if str(row.get("path_mode", "")) != "drift_step_start_phases":
                continue
            pair = f"({row['source_orientation']},{row['target_orientation']})"
            print(f"  {pair:>12}  {row['median_abs_cos_x']:>10.3f}  {row['median_abs_cos_y']:>10.3f}  "
                  f"{row['fraction_x_dominant']:>7.3f}  {row['median_tangent_mimicry']:>11.3f}")

    if sign_summary_rows:
        print("\n=== Sign-flip diagnostics (s_perp_u1, drift mode) ===")
        print(f"  {'logmar':>6}  {'pair':>10}  {'sign_cons':>10}  {'cancel_idx':>11}  {'med_abs_s':>10}  {'frac_pos':>9}")
        for row in sign_summary_rows:
            if str(row.get("path_mode", "")) != "drift_step_start_phases":
                continue
            pair = f"({row['source_orientation']},{row['target_orientation']})"
            lm = float(row["logmar"])
            print(f"  {lm:+6.2f}  {pair:>10}  {float(row['sign_consistency']):>10.3f}  "
                  f"{float(row['signed_cancellation_index']):>11.3f}  "
                  f"{float(row['median_abs_s_perp_u1']):>10.4f}  "
                  f"{float(row['fraction_positive_s_perp_u1']):>9.3f}")


if __name__ == "__main__":
    main()
