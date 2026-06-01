#!/usr/bin/env python3
"""
Focused reconciliation for Phase 1 noise-correlation discrepancy.

Goal:
- Determine whether the historical negative corrected correlation result is
  estimator-specific or due to preprocessing (masks/units/residuals/binning).
- Regenerate Phase 1 summaries with strict sign-flip semantics.

Outputs:
  outputs/phase1_fem_covariance/noise_correlations/reconciliation/
    old_vs_phase1_method_comparison.md
    noise_corr_reconciliation_table.csv
    estimator_comparison_by_session.csv
    synthetic_control_numeric_summary.csv
    raw_vs_corrected_distributions_by_estimator.png
    correction_strength_comparison.png

Also updates:
  outputs/phase1_fem_covariance/summaries/phase1_master_summary.csv
  outputs/phase1_fem_covariance/summaries/phase1_decision_table.csv
  outputs/phase1_fem_covariance/summaries/phase1_readme.md
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT
from scripts.phase1_fem.estimators import fit_b_emp, mcfarland_fem_covariance
from scripts.phase1_fem.noise_correlations import run_noise_correlations
from scripts.phase1_fem.preprocessing import (
    build_valid_mask,
    compute_loto_residuals,
    load_session,
    select_primary_units,
)


def _resolve_config(cfg_arg: str) -> str:
    p = Path(cfg_arg)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = VISIONCORE_ROOT / "experiments" / "dataset_configs" / cfg_arg
    if candidate.exists():
        return str(candidate)
    if p.exists():
        return str(p.resolve())
    raise FileNotFoundError(f"Cannot find dataset config '{cfg_arg}'")


def _flatten_valid(R: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    NT, T, U = R.shape
    X = R.reshape(NT * T, U).astype(np.float64)
    vm = valid_mask.reshape(NT * T)
    X = X[vm]
    good = np.all(np.isfinite(X), axis=1)
    return X[good]


def _flatten_valid_pair(R: np.ndarray, eye: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten rows where valid_mask is true and both R and eye are finite."""
    NT, T, U = R.shape
    X = R.reshape(NT * T, U).astype(np.float64)
    E = eye.reshape(NT * T, 2).astype(np.float64)
    vm = valid_mask.reshape(NT * T)
    X = X[vm]
    E = E[vm]
    good = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(E), axis=1)
    return X[good], E[good]


def _cov(X: np.ndarray) -> np.ndarray:
    if X.shape[0] < 3:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    return np.cov(X, rowvar=False)


def _psd_project(C: np.ndarray) -> np.ndarray:
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 0.0, None)
    return (vecs * vals[None, :]) @ vecs.T


def _corr_from_cov(C: np.ndarray) -> np.ndarray:
    d = np.diag(C)
    d = np.where(d > 1e-12, np.sqrt(d), np.nan)
    denom = d[:, None] * d[None, :]
    R = np.divide(C, denom, out=np.full_like(C, np.nan), where=np.isfinite(denom) & (denom > 0))
    np.fill_diagonal(R, 1.0)
    return R


def _median_upper(M: np.ndarray) -> float:
    idx = np.triu_indices(M.shape[0], k=1)
    v = M[idx]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _corr_upper(A: np.ndarray, B: np.ndarray) -> float:
    idx = np.triu_indices(A.shape[0], k=1)
    a = A[idx]
    b = B[idx]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _fisher_reduction(raw_corr: float, corr_corr: float) -> float:
    if not np.isfinite(raw_corr) or not np.isfinite(corr_corr):
        return float("nan")
    r0 = float(np.clip(raw_corr, -0.999999, 0.999999))
    r1 = float(np.clip(corr_corr, -0.999999, 0.999999))
    z0 = np.arctanh(r0)
    z1 = np.arctanh(r1)
    if abs(z0) < 1e-8:
        return float("nan")
    return float(1.0 - (z1 / z0))


def _psd_diagnostics(C_pre: np.ndarray, C_post: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh(0.5 * (C_pre + C_pre.T))
    neg = eig[eig < 0]
    return {
        "min_eig_C_psth_minus_C_FEM": float(np.min(eig)) if eig.size else float("nan"),
        "n_negative_eigs_before_psd": int(neg.size),
        "sum_negative_eigs": float(np.sum(neg)) if neg.size else 0.0,
        "trace_before_psd": float(np.trace(C_pre)),
        "trace_after_psd": float(np.trace(C_post)),
        "median_corr_before_psd_if_defined": _median_upper(_corr_from_cov(C_pre)),
        "median_corr_after_psd": _median_upper(_corr_from_cov(C_post)),
    }


def _embed_valid_rows(X_valid: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Embed (N_valid, U) matrix back into (NT, T, U) with NaN on invalid bins."""
    NT, T = valid_mask.shape
    U = X_valid.shape[1]
    out = np.full((NT, T, U), np.nan, dtype=np.float64)
    idx = np.where(valid_mask.reshape(NT * T))[0]
    flat = out.reshape(NT * T, U)
    flat[idx, :] = X_valid
    return out


def _run_estimator_b_synthetic_controls(
    session: str,
    X_psth: np.ndarray,
    eye_valid: np.ndarray,
    valid_mask_embed: np.ndarray,
    image_ids: np.ndarray,
    seed: int,
) -> list[dict]:
    """
    Run key synthetic controls through estimator-B style correction.

    Control correction is done with estimated McFarland C_FEM from synthetic data,
    followed by subtraction and PSD projection, matching estimator B mechanics.
    """
    rng = np.random.default_rng(seed)
    n, u = X_psth.shape

    E = eye_valid.astype(np.float64)
    E_c = E - E.mean(axis=0, keepdims=True)

    b_scale = np.maximum(np.std(X_psth), 1e-3)
    Bsyn = rng.normal(0.0, 0.25 * b_scale, size=(u, 2))
    eye_comp = E_c @ Bsyn.T

    rows = []
    specs = [
        ("independent_poisson_plus_eye", 0.0, "near_zero_after_correction"),
        ("zero_intrinsic_estimated_eye", 0.0, "near_zero_not_negative"),
        ("positive_latent_plus_eye", 0.04, "remains_positive_after_correction"),
    ]

    for name, gt, expected in specs:
        mu = np.maximum(np.mean(X_psth, axis=0), 0.3)
        if name == "positive_latent_plus_eye":
            z = rng.standard_normal((n, 1))
            loading = rng.uniform(0.05, 0.16, size=(1, u))
            intrinsic = z @ loading
        else:
            intrinsic = np.zeros((n, u), dtype=np.float64)

        lam = np.clip(mu[None, :] + 0.2 * eye_comp + intrinsic, 1e-4, None)
        X = rng.poisson(lam).astype(np.float64)

        R_syn = _embed_valid_rows(X, valid_mask_embed)
        eye_syn = _embed_valid_rows(E, valid_mask_embed)
        mc_syn = mcfarland_fem_covariance(R_syn, eye_syn, image_ids, valid_mask_embed, seed=seed)
        C_fem_est = mc_syn.get("C_FEM", np.full((u, u), np.nan))

        C_raw = _cov(X)
        C_corr = _psd_project(C_raw - C_fem_est)
        raw_med = _median_upper(_corr_from_cov(C_raw))
        corr_med = _median_upper(_corr_from_cov(C_corr))

        passed, expected_direction, failure_reason = _evaluate_control(name, raw_med, corr_med)
        rows.append(
            {
                "session": session,
                "estimator": "B_old_mcfarland_or_law_total_cov_current_masks",
                "control_name": name,
                "ground_truth_intrinsic_corr": gt,
                "raw_median_corr": raw_med,
                "corrected_median_corr": corr_med,
                "expected_direction": expected_direction if expected_direction else expected,
                "passed": str(bool(passed)).lower(),
                "failure_reason": failure_reason,
                "n_contexts_mcfarland": mc_syn.get("n_contexts", "nan"),
            }
        )
    return rows


def _evaluate_control(control_name: str, raw: float, corrected: float) -> tuple[bool, str, str]:
    expected = ""
    if control_name == "independent_poisson_plus_eye":
        expected = "corrected near zero"
        passed = np.isfinite(corrected) and abs(corrected) <= 0.01
        reason = "" if passed else f"|corrected|={abs(corrected):.4f} > 0.01"
        return passed, expected, reason

    if control_name == "zero_intrinsic_estimated_eye":
        expected = "corrected near zero, not strongly negative"
        passed = np.isfinite(corrected) and corrected >= -0.01 and abs(corrected) <= 0.02
        reason = "" if passed else f"corrected={corrected:.4f} outside [-0.01, 0.02]"
        return passed, expected, reason

    if control_name == "positive_latent_plus_eye":
        expected = "corrected remains positive"
        passed = np.isfinite(corrected) and corrected >= 0.02
        reason = "" if passed else f"corrected={corrected:.4f} < 0.02"
        return passed, expected, reason

    if control_name == "shuffled_eye_correction_empirical":
        expected = "little/no reduction"
        if np.isfinite(raw) and np.isfinite(corrected):
            passed = abs(corrected - raw) <= 0.0035
            reason = "" if passed else f"|delta|={abs(corrected - raw):.4f} > 0.0035"
            return passed, expected, reason
        return False, expected, "raw/corrected not finite"

    if control_name == "cross_validated_eye_correction_empirical":
        expected = "reduction survives cross-validation"
        if np.isfinite(raw) and np.isfinite(corrected):
            passed = corrected < raw
            reason = "" if passed else f"corrected={corrected:.4f} >= raw={raw:.4f}"
            return passed, expected, reason
        return False, expected, "raw/corrected not finite"

    return False, "unknown", "unknown control"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _build_method_comparison_md(path: Path, old_number_source: str) -> None:
    text = f"""# Old vs Phase 1 method comparison

## Summary
This note reconciles Phase 1 Analysis 2 with the legacy LOTC/McFarland path.

## Historical number lookup
- Target historical claim searched: `+0.0492 -> -0.0179`
- Result: {old_number_source}

## Legacy LOTC path (from repository code)
- Primary code path inspected:
  - `scripts/fixrsvp_lotc_model_declan.py`
  - `scripts/figures_noisecorr.py`
  - `scripts/mcfarland_sim.py`
- Core decomposition used in that path:
  - `Ctotal`: total covariance
  - `Cpsth`: PSTH covariance
  - `Crate`: McFarland intercept covariance
  - `CnoiseU = Ctotal - Cpsth`
  - `CnoiseC = Ctotal - Crate`
- PSD projection appears in legacy path before downstream covariance displays/stats.
- Noise-correlation summary statistics are computed from pairwise correlations and Fisher-z summaries over datasets/windows.

## Phase 1 path used here
- Data/masks/units/residuals:
  - same trial-time valid mask and primary units as Phase 1
  - same LOTO image-time residuals
- Estimator A (current): linear `B_emp` correction from PSTH residuals.
- Estimator B (old-style on current preprocessing): McFarland covariance subtraction on the same residuals.

## Reconciliation logic
- Sign flip is defined strictly as:
  - `median_corr_eye_corrected < 0`
- Reduction is tracked separately via:
  - `noise_corr_reduction = median_corrected - median_raw`
  - `noise_corr_reduction_fraction = 1 - median_corrected / median_raw`

## Interpretation guardrails
- If corrected median remains positive:
  - FEM correction reduces positive noise correlations but does not reveal negative residual correlations.
- If old-style estimator sign-flips but current estimator does not:
  - The negative residual-correlation result is estimator-dependent.
- If sign flip appears only under old masks/unit sets:
  - The result is preprocessing-dependent.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _regenerate_phase1_summaries(out_root: Path, synth_numeric_rows: list[dict]) -> None:
    cov_path = out_root / "covariance_geometry" / "covariance_geometry_session_metrics.csv"
    noise_path = out_root / "noise_correlations" / "noise_correlation_session_metrics.csv"

    if not cov_path.exists() or not noise_path.exists():
        return

    cov_rows = list(csv.DictReader(cov_path.open()))
    noise_rows = list(csv.DictReader(noise_path.open()))

    cov_by = {r["session"]: r for r in cov_rows}
    noise_by = {r["session"]: r for r in noise_rows}

    ctrl_pass_by = {}
    for r in synth_numeric_rows:
        if r.get("estimator") != "A_phase1_linear_Bemp_current":
            continue
        sess = r["session"]
        ok = (r.get("passed", "").lower() == "true")
        ctrl_pass_by.setdefault(sess, []).append(ok)

    sessions = sorted(cov_by.keys())
    master = []
    for s in sessions:
        c = cov_by[s]
        n = noise_by.get(s, {})

        raw = float(n.get("median_corr_raw", "nan")) if n else float("nan")
        corr = float(n.get("median_corr_eye_corrected", "nan")) if n else float("nan")
        reduction = corr - raw if np.isfinite(corr) and np.isfinite(raw) else float("nan")
        reduction_fraction = 1.0 - corr / raw if np.isfinite(corr) and np.isfinite(raw) and abs(raw) > 1e-9 else float("nan")
        sign_flip = np.isfinite(corr) and (corr < 0)

        ctrl_ok = False
        if s in ctrl_pass_by and ctrl_pass_by[s]:
            ctrl_ok = bool(np.all(np.array(ctrl_pass_by[s], dtype=bool)))

        fisher_reduction = _fisher_reduction(raw, corr)

        if sign_flip and ctrl_ok and np.isfinite(fisher_reduction) and fisher_reduction > 0.5:
            rec = "headline"
        elif np.isfinite(reduction) and reduction < 0 and np.isfinite(corr) and corr > 0:
            rec = "supporting"
        else:
            rec = "inconclusive"

        master.append(
            {
                "session": s,
                "n_units": c.get("n_units_primary", "nan"),
                "n_valid_windows": c.get("n_valid_bins", "nan"),
                "cov_top2_fraction": c.get("mc_top2_fraction", c.get("b_emp_top2_fraction", "nan")),
                "cov_participation_ratio": c.get("mc_participation_ratio", c.get("b_emp_participation_ratio", "nan")),
                "model_alignment": c.get("model_alignment", "nan"),
                "image_shuffle_alignment": c.get("model_shuffle_alignment", "nan"),
                "reliability_ceiling": c.get("reliability_ceiling", "nan"),
                "ceiling_normalized_alignment": c.get("ceiling_normalized_alignment", "nan"),
                "alignment_norm_status": c.get("alignment_norm_status", "not_available"),
                "shared_vs_image_specific_delta": c.get("shared_vs_img_r2_delta", "nan"),
                "primary_vs_sensitivity_cov_corr": c.get("primary_vs_sensitivity_cov_corr", "nan"),
                "raw_noise_corr_median": n.get("median_corr_raw", "nan"),
                "psth_noise_corr_median": n.get("median_corr_psth", "nan"),
                "eye_corrected_corr_median": n.get("median_corr_eye_corrected", "nan"),
                "noise_corr_reduction": reduction,
                "noise_corr_reduction_fraction": reduction_fraction,
                "noise_corr_sign_flip": str(bool(sign_flip)).lower() if np.isfinite(corr) else "nan",
                "noise_corr_fisher_reduction_fraction": fisher_reduction,
                "synthetic_controls_pass": "yes" if ctrl_ok else "no",
                "aggregation_reliability_full": "nan",
                "aggregation_n_half": "nan",
                "stage4_explained_by_aggregation": "nan",
                "phase1_recommendation": rec,
                "status": "analysis2_complete",
            }
        )

    sum_path = out_root / "summaries" / "phase1_master_summary.csv"
    _write_csv(sum_path, master)

    # Decision labels for Analysis 2
    sign_flip_count = sum(r["noise_corr_sign_flip"] == "true" for r in master)
    fisher_good = [
        (float(r["noise_corr_fisher_reduction_fraction"]) > 0.5)
        for r in master
        if r["noise_corr_fisher_reduction_fraction"] not in ("nan", "", None)
    ]
    fisher_ok = (len(fisher_good) > 0) and bool(np.median(np.array(fisher_good, dtype=float)) >= 1.0)
    controls_pass_sessions = sum(r["synthetic_controls_pass"] == "yes" for r in master)

    consistent_reduction = all(
        (r["noise_corr_reduction"] != "nan") and (float(r["noise_corr_reduction"]) < 0)
        for r in master
    )
    corrected_all_positive = all(
        (r["eye_corrected_corr_median"] != "nan") and (float(r["eye_corrected_corr_median"]) > 0)
        for r in master
    )

    if sign_flip_count >= 3 and fisher_ok and controls_pass_sessions >= 3:
        nc_headline = "yes"
        nc_supporting = "yes"
        nc_null = "no"
        nc_implication = "noise_correlation_sign_flip_robust"
    elif consistent_reduction and corrected_all_positive and controls_pass_sessions >= 3:
        nc_headline = "no"
        nc_supporting = "yes"
        nc_null = "no"
        nc_implication = "reduction_without_sign_flip"
    else:
        nc_headline = "no"
        nc_supporting = "no"
        nc_null = "yes"
        nc_implication = "inconclusive_or_artifactual"

    dec_rows = [
        {
            "row": "2D_covariance_geometry",
            "headline_worthy": "no",
            "supporting": "yes",
            "null": "no",
            "reason": "Analysis 1 completed; low-dimensionality landed, but cross-metric strength is mixed.",
            "sessions_supporting": ";".join(r["session"] for r in master),
            "controls_passed": "partial",
            "manuscript_implication": "covariance_geometry_support",
            "next_action": "run_analysis3",
        },
        {
            "row": "noise_correlation_sign_flip",
            "headline_worthy": nc_headline,
            "supporting": nc_supporting,
            "null": nc_null,
            "reason": "Sign flip requires corrected median < 0; negative delta alone is not sign flip.",
            "sessions_supporting": ";".join(r["session"] for r in master if float(r["noise_corr_reduction"]) < 0),
            "controls_passed": "yes" if controls_pass_sessions >= 3 else "partial",
            "manuscript_implication": nc_implication,
            "next_action": "run_analysis3",
        },
        {
            "row": "aggregation_scaling",
            "headline_worthy": "unknown",
            "supporting": "unknown",
            "null": "unknown",
            "reason": "Analysis 3 not run yet.",
            "sessions_supporting": "",
            "controls_passed": "not_run",
            "manuscript_implication": "pending",
            "next_action": "implement_and_run_analysis3",
        },
        {
            "row": "overall_phase1",
            "headline_worthy": "no" if nc_headline == "no" else "yes",
            "supporting": "yes" if nc_supporting == "yes" else "no",
            "null": "no",
            "reason": "Analyses 1 and 2 complete with strict sign-flip labeling; Analysis 3 pending.",
            "sessions_supporting": ";".join(r["session"] for r in master),
            "controls_passed": "partial",
            "manuscript_implication": "preliminary_supporting",
            "next_action": "complete_phase1_analysis3",
        },
    ]
    dec_path = out_root / "summaries" / "phase1_decision_table.csv"
    _write_csv(dec_path, dec_rows)

    md = out_root / "summaries" / "phase1_readme.md"
    lines = [
        "# Phase 1 interim summary",
        "",
        "Completed Analyses 1 and 2 sessions (strict sign-flip semantics):",
    ]
    for r in master:
        lines.append(
            f"- {r['session']}: raw_noise={r['raw_noise_corr_median']}, corrected_noise={r['eye_corrected_corr_median']}, reduction={r['noise_corr_reduction']}, sign_flip={r['noise_corr_sign_flip']}, controls_pass={r['synthetic_controls_pass']}"
        )
    lines += [
        "",
        "Interpretation:",
        "- Negative delta means reduction, not sign flip.",
        "- Sign flip requires corrected median < 0.",
        "",
        "Pending analysis in this milestone:",
        "- Analysis 3 (aggregation scaling)",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase 1 noise-correlation reconciliation")
    ap.add_argument("--dataset-configs-path", default="multi_basic_120_long_legacy.yaml")
    ap.add_argument(
        "--sessions",
        nargs="+",
        default=["Allen:2022-02-16", "Allen:2022-02-24", "Allen:2022-03-04", "Allen:2022-04-08"],
    )
    ap.add_argument("--use-cached-data", action="store_true", default=False)
    ap.add_argument("--out-dir", default="outputs/phase1_fem_covariance")
    ap.add_argument("--min-rate-hz", type=float, default=0.5)
    ap.add_argument("--min-valid-fraction", type=float, default=0.2)
    ap.add_argument("--fixation-radius", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    cfg_path = _resolve_config(args.dataset_configs_path)
    out_root = VISIONCORE_ROOT / args.out_dir
    rec_dir = out_root / "noise_correlations" / "reconciliation"
    rec_dir.mkdir(parents=True, exist_ok=True)

    recon_rows: list[dict] = []
    comp_rows: list[dict] = []
    synth_numeric_rows: list[dict] = []

    pooled_raw = []
    pooled_a_corr = []
    pooled_b_corr = []

    for spec in args.sessions:
        subject, date = spec.split(":", 1)
        session = f"{subject}_{date}"

        data = load_session(
            subject,
            date,
            cfg_path,
            use_cached=args.use_cached_data,
            fixation_degree_radius=args.fixation_radius,
            verbose=False,
        )

        robs = data["robs"]
        dfs = data["dfs"]
        eyepos = data["eyepos"]
        image_ids = data["image_ids"]
        bin_ms = data["bin_ms"]

        valid_mask = build_valid_mask(robs, dfs, eyepos, image_ids)
        unit_mask, _ = select_primary_units(
            robs,
            dfs,
            valid_mask,
            bin_ms,
            min_rate_hz=args.min_rate_hz,
            min_valid_fraction=args.min_valid_fraction,
        )

        R_resid = compute_loto_residuals(robs, image_ids, valid_mask)

        R_raw_u = robs[:, :, unit_mask]
        R_resid_u = R_resid[:, :, unit_mask]
        eye = eyepos

        X_raw = _flatten_valid(R_raw_u, valid_mask)
        X_psth = _flatten_valid(R_resid_u, valid_mask)

        C_raw = _cov(X_raw)
        C_psth = _cov(X_psth)
        R_raw_corr = _corr_from_cov(C_raw)
        R_psth_corr = _corr_from_cov(C_psth)

        raw_med = _median_upper(R_raw_corr)
        psth_med = _median_upper(R_psth_corr)

        # Estimator A: Phase 1 linear B_emp
        b = fit_b_emp(R_resid_u, eye, valid_mask)
        R_eye_pred = b["R_eye_pred"]
        X_eye_pred = _flatten_valid(R_eye_pred, valid_mask)
        C_eye_pred = _cov(X_eye_pred)
        X_a_corr = _flatten_valid(R_resid_u - R_eye_pred, valid_mask)
        C_a_corr = _cov(X_a_corr)
        R_a_corr = _corr_from_cov(C_a_corr)
        a_med = _median_upper(R_a_corr)

        var_raw_total = float(np.trace(C_raw))
        var_psth_resid = float(np.trace(C_psth))
        var_eye_pred = float(np.trace(C_eye_pred))
        var_eye_corrected = float(np.trace(C_a_corr))
        frac_var_eye = var_eye_pred / var_psth_resid if np.isfinite(var_psth_resid) and abs(var_psth_resid) > 1e-12 else float("nan")

        # Estimator B: old-style McFarland subtraction on same masks/residuals
        mc = mcfarland_fem_covariance(R_resid_u, eye, image_ids, valid_mask, seed=args.seed)
        C_old_fem = mc.get("C_FEM", np.full_like(C_psth, np.nan))
        C_b_pre = C_psth - C_old_fem
        C_b_corr = _psd_project(C_b_pre)
        R_b_corr = _corr_from_cov(C_b_corr)
        b_med = _median_upper(R_b_corr)
        psd_diag = _psd_diagnostics(C_b_pre, C_b_corr)

        corr_C_raw_C_eye_pred = _corr_upper(C_raw, C_eye_pred)
        corr_C_old_CFEM_C_phase1_Ceye = _corr_upper(C_old_fem, C_eye_pred)

        # Synthetic controls (numeric, with pass/failure) from current estimator path
        noise_out = run_noise_correlations(
            session=session,
            R_raw=robs,
            R_resid_psth=R_resid,
            eye=eyepos,
            valid_mask=valid_mask,
            unit_mask=unit_mask,
            seed=args.seed,
        )
        ctrl_rows = noise_out.get("synthetic_control_rows", [])
        ctrl_passes = []
        for r in ctrl_rows:
            passed, expected, failure_reason = _evaluate_control(
                r.get("control_name", ""),
                float(r.get("median_corr_raw", "nan")),
                float(r.get("median_corr_corrected", "nan")),
            )
            ctrl_passes.append(passed)
            synth_numeric_rows.append(
                {
                    "session": session,
                    "estimator": "A_phase1_linear_Bemp_current",
                    "control_name": r.get("control_name", ""),
                    "ground_truth_intrinsic_corr": r.get("ground_truth_intrinsic_median_corr", "nan"),
                    "raw_median_corr": r.get("median_corr_raw", "nan"),
                    "corrected_median_corr": r.get("median_corr_corrected", "nan"),
                    "expected_direction": expected,
                    "passed": str(bool(passed)).lower(),
                    "failure_reason": failure_reason,
                }
            )
        controls_pass = bool(ctrl_passes) and bool(np.all(np.array(ctrl_passes, dtype=bool)))

        # Estimator-B matched synthetic controls
        X_psth_pair, eye_valid = _flatten_valid_pair(R_resid_u, eye, valid_mask)
        valid_mask_embed = valid_mask & np.all(np.isfinite(R_resid_u), axis=-1) & np.all(np.isfinite(eye), axis=-1)
        b_ctrl_rows = _run_estimator_b_synthetic_controls(
            session=session,
            X_psth=X_psth_pair,
            eye_valid=eye_valid,
            valid_mask_embed=valid_mask_embed,
            image_ids=image_ids,
            seed=args.seed,
        )
        for r in b_ctrl_rows:
            synth_numeric_rows.append(r)
        b_ctrl_pass = bool(b_ctrl_rows) and all(x.get("passed") == "true" for x in b_ctrl_rows)

        n_units = int(unit_mask.sum())
        n_pairs = int(n_units * (n_units - 1) // 2)

        rows_this_session = [
            {
                "session": session,
                "pipeline_label": "phase1_reconciliation",
                "estimator": "A_phase1_linear_Bemp_current",
                "unit_set": "phase1_primary_units_current",
                "valid_mask": "phase1_trialtime_anyunit_finite_eye_img_valid",
                "residual_type": "loto_image_time",
                "bin_ms": bin_ms,
                "n_units": n_units,
                "n_pairs": n_pairs,
                "raw_median_corr": raw_med,
                "psth_median_corr": psth_med,
                "corrected_median_corr": a_med,
                "delta_raw_to_corrected": a_med - raw_med,
                "delta_psth_to_corrected": a_med - psth_med,
                "reduction_fraction": (1.0 - a_med / raw_med) if np.isfinite(raw_med) and abs(raw_med) > 1e-12 else float("nan"),
                "sign_flip": str(bool(np.isfinite(a_med) and a_med < 0)).lower(),
                "synthetic_controls_pass": "yes" if controls_pass else "no",
                "var_raw_total": var_raw_total,
                "var_psth_resid": var_psth_resid,
                "var_eye_pred": var_eye_pred,
                "var_eye_corrected": var_eye_corrected,
                "fraction_residual_variance_explained_by_eye": frac_var_eye,
                "corr_C_raw_C_eye_pred": corr_C_raw_C_eye_pred,
                "corr_C_old_CFEM_C_phase1_Ceye": corr_C_old_CFEM_C_phase1_Ceye,
                "C_residual_pre_psd_min_eig": "nan",
                "C_residual_pre_psd_n_negative_eigs": "nan",
                "C_residual_pre_psd_trace": "nan",
                "C_residual_post_psd_trace": "nan",
                "min_eig_C_psth_minus_C_FEM": "nan",
                "n_negative_eigs_before_psd": "nan",
                "sum_negative_eigs": "nan",
                "trace_before_psd": "nan",
                "trace_after_psd": "nan",
                "median_corr_before_psd_if_defined": "nan",
                "median_corr_after_psd": "nan",
                "notes": "Current Phase 1 linear estimator on current preprocessing.",
            },
            {
                "session": session,
                "pipeline_label": "phase1_reconciliation",
                "estimator": "B_old_mcfarland_or_law_total_cov_current_masks",
                "unit_set": "phase1_primary_units_current",
                "valid_mask": "phase1_trialtime_anyunit_finite_eye_img_valid",
                "residual_type": "loto_image_time",
                "bin_ms": bin_ms,
                "n_units": n_units,
                "n_pairs": n_pairs,
                "raw_median_corr": raw_med,
                "psth_median_corr": psth_med,
                "corrected_median_corr": b_med,
                "delta_raw_to_corrected": b_med - raw_med,
                "delta_psth_to_corrected": b_med - psth_med,
                "reduction_fraction": (1.0 - b_med / raw_med) if np.isfinite(raw_med) and abs(raw_med) > 1e-12 else float("nan"),
                "sign_flip": str(bool(np.isfinite(b_med) and b_med < 0)).lower(),
                "synthetic_controls_pass": "yes" if b_ctrl_pass else "no",
                "var_raw_total": var_raw_total,
                "var_psth_resid": var_psth_resid,
                "var_eye_pred": float(np.trace(C_old_fem)) if np.isfinite(C_old_fem).any() else float("nan"),
                "var_eye_corrected": float(np.trace(C_b_corr)) if np.isfinite(C_b_corr).any() else float("nan"),
                "fraction_residual_variance_explained_by_eye": (
                    float(np.trace(C_old_fem)) / var_psth_resid
                    if np.isfinite(var_psth_resid) and abs(var_psth_resid) > 1e-12 and np.isfinite(C_old_fem).any()
                    else float("nan")
                ),
                "corr_C_raw_C_eye_pred": _corr_upper(C_raw, C_old_fem),
                "corr_C_old_CFEM_C_phase1_Ceye": corr_C_old_CFEM_C_phase1_Ceye,
                "C_residual_pre_psd_min_eig": psd_diag["min_eig_C_psth_minus_C_FEM"],
                "C_residual_pre_psd_n_negative_eigs": psd_diag["n_negative_eigs_before_psd"],
                "C_residual_pre_psd_trace": psd_diag["trace_before_psd"],
                "C_residual_post_psd_trace": psd_diag["trace_after_psd"],
                "min_eig_C_psth_minus_C_FEM": psd_diag["min_eig_C_psth_minus_C_FEM"],
                "n_negative_eigs_before_psd": psd_diag["n_negative_eigs_before_psd"],
                "sum_negative_eigs": psd_diag["sum_negative_eigs"],
                "trace_before_psd": psd_diag["trace_before_psd"],
                "trace_after_psd": psd_diag["trace_after_psd"],
                "median_corr_before_psd_if_defined": psd_diag["median_corr_before_psd_if_defined"],
                "median_corr_after_psd": psd_diag["median_corr_after_psd"],
                "notes": "Legacy-style covariance subtraction using McFarland estimator on current masks.",
            },
            {
                "session": session,
                "pipeline_label": "phase1_reconciliation",
                "estimator": "C_phase1_linear_Bemp_old_masks_if_available",
                "unit_set": "old_masks_unavailable",
                "valid_mask": "old_masks_unavailable",
                "residual_type": "unknown",
                "bin_ms": bin_ms,
                "n_units": "nan",
                "n_pairs": "nan",
                "raw_median_corr": "nan",
                "psth_median_corr": "nan",
                "corrected_median_corr": "nan",
                "delta_raw_to_corrected": "nan",
                "delta_psth_to_corrected": "nan",
                "reduction_fraction": "nan",
                "sign_flip": "nan",
                "synthetic_controls_pass": "not_run",
                "var_raw_total": "nan",
                "var_psth_resid": "nan",
                "var_eye_pred": "nan",
                "var_eye_corrected": "nan",
                "fraction_residual_variance_explained_by_eye": "nan",
                "corr_C_raw_C_eye_pred": "nan",
                "corr_C_old_CFEM_C_phase1_Ceye": "nan",
                "C_residual_pre_psd_min_eig": "nan",
                "C_residual_pre_psd_n_negative_eigs": "nan",
                "C_residual_pre_psd_trace": "nan",
                "C_residual_post_psd_trace": "nan",
                "min_eig_C_psth_minus_C_FEM": "nan",
                "n_negative_eigs_before_psd": "nan",
                "sum_negative_eigs": "nan",
                "trace_before_psd": "nan",
                "trace_after_psd": "nan",
                "median_corr_before_psd_if_defined": "nan",
                "median_corr_after_psd": "nan",
                "notes": "Old masks/unit set artifact not available in current workspace outputs.",
            },
            {
                "session": session,
                "pipeline_label": "phase1_reconciliation",
                "estimator": "D_old_mcfarland_old_masks_if_available",
                "unit_set": "old_masks_unavailable",
                "valid_mask": "old_masks_unavailable",
                "residual_type": "unknown",
                "bin_ms": bin_ms,
                "n_units": "nan",
                "n_pairs": "nan",
                "raw_median_corr": "nan",
                "psth_median_corr": "nan",
                "corrected_median_corr": "nan",
                "delta_raw_to_corrected": "nan",
                "delta_psth_to_corrected": "nan",
                "reduction_fraction": "nan",
                "sign_flip": "nan",
                "synthetic_controls_pass": "not_run",
                "var_raw_total": "nan",
                "var_psth_resid": "nan",
                "var_eye_pred": "nan",
                "var_eye_corrected": "nan",
                "fraction_residual_variance_explained_by_eye": "nan",
                "corr_C_raw_C_eye_pred": "nan",
                "corr_C_old_CFEM_C_phase1_Ceye": "nan",
                "C_residual_pre_psd_min_eig": "nan",
                "C_residual_pre_psd_n_negative_eigs": "nan",
                "C_residual_pre_psd_trace": "nan",
                "C_residual_post_psd_trace": "nan",
                "min_eig_C_psth_minus_C_FEM": "nan",
                "n_negative_eigs_before_psd": "nan",
                "sum_negative_eigs": "nan",
                "trace_before_psd": "nan",
                "trace_after_psd": "nan",
                "median_corr_before_psd_if_defined": "nan",
                "median_corr_after_psd": "nan",
                "notes": "Old masks/unit set artifact not available in current workspace outputs.",
            },
        ]
        recon_rows.extend(rows_this_session)

        comp_rows.append(
            {
                "session": session,
                "A_corrected_median_corr": a_med,
                "B_corrected_median_corr": b_med,
                "A_sign_flip": str(bool(np.isfinite(a_med) and a_med < 0)).lower(),
                "B_sign_flip": str(bool(np.isfinite(b_med) and b_med < 0)).lower(),
                "A_reduction_fraction": (1.0 - a_med / raw_med) if np.isfinite(raw_med) and abs(raw_med) > 1e-12 else float("nan"),
                "B_reduction_fraction": (1.0 - b_med / raw_med) if np.isfinite(raw_med) and abs(raw_med) > 1e-12 else float("nan"),
                "A_controls_pass": "yes" if controls_pass else "no",
                "B_controls_pass": "yes" if b_ctrl_pass else "no",
                "estimator_difference": b_med - a_med if np.isfinite(a_med) and np.isfinite(b_med) else float("nan"),
                "B_minus_A_corrected_median": b_med - a_med if np.isfinite(a_med) and np.isfinite(b_med) else float("nan"),
                "B_minus_A_reduction_fraction": (
                    ((1.0 - b_med / raw_med) - (1.0 - a_med / raw_med))
                    if np.isfinite(raw_med) and abs(raw_med) > 1e-12 and np.isfinite(a_med) and np.isfinite(b_med)
                    else float("nan")
                ),
                "B_creates_more_negative_residual_than_A": (
                    str(bool(np.isfinite(a_med) and np.isfinite(b_med) and (b_med < a_med))).lower()
                ),
                "C_residual_pre_psd_min_eig": psd_diag["min_eig_C_psth_minus_C_FEM"],
                "C_residual_pre_psd_n_negative_eigs": psd_diag["n_negative_eigs_before_psd"],
                "C_residual_pre_psd_trace": psd_diag["trace_before_psd"],
                "C_residual_post_psd_trace": psd_diag["trace_after_psd"],
            }
        )

        # Plot buffers
        idx = np.triu_indices(R_raw_corr.shape[0], k=1)
        pooled_raw.extend(R_raw_corr[idx][np.isfinite(R_raw_corr[idx])].tolist())
        pooled_a_corr.extend(R_a_corr[idx][np.isfinite(R_a_corr[idx])].tolist())
        pooled_b_corr.extend(R_b_corr[idx][np.isfinite(R_b_corr[idx])].tolist())

    # Write tables
    _write_csv(rec_dir / "noise_corr_reconciliation_table.csv", recon_rows)
    _write_csv(rec_dir / "estimator_comparison_by_session.csv", comp_rows)
    _write_csv(rec_dir / "synthetic_control_numeric_summary.csv", synth_numeric_rows)

    # Plots
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bins = np.linspace(-0.1, 0.15, 120)
    axs[0].hist(pooled_raw, bins=bins, alpha=0.45, label="Raw", density=True)
    axs[0].hist(pooled_a_corr, bins=bins, alpha=0.45, label="Corrected A (B_emp)", density=True)
    axs[0].hist(pooled_b_corr, bins=bins, alpha=0.45, label="Corrected B (McFarland)", density=True)
    axs[0].axvline(0, color="k", linewidth=1)
    axs[0].set_title("Pairwise correlation distributions")
    axs[0].set_xlabel("Pairwise correlation")
    axs[0].set_ylabel("Density")
    axs[0].legend(frameon=False)

    sess = [r["session"] for r in comp_rows]
    a_vals = [float(r["A_corrected_median_corr"]) for r in comp_rows]
    b_vals = [float(r["B_corrected_median_corr"]) for r in comp_rows]
    x = np.arange(len(sess))
    axs[1].plot(x, a_vals, "o-", label="A corrected median")
    axs[1].plot(x, b_vals, "o-", label="B corrected median")
    axs[1].axhline(0, color="k", linewidth=1)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(sess, rotation=25, ha="right")
    axs[1].set_ylabel("Median corrected correlation")
    axs[1].set_title("By-session estimator comparison")
    axs[1].legend(frameon=False)
    fig.savefig(rec_dir / "raw_vs_corrected_distributions_by_estimator.png", dpi=180)
    plt.close(fig)

    # Correction strength plot
    a_rows = [r for r in recon_rows if r["estimator"] == "A_phase1_linear_Bemp_current"]
    b_rows = [r for r in recon_rows if r["estimator"] == "B_old_mcfarland_or_law_total_cov_current_masks"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(a_rows))
    axs[0].bar(x - 0.18, [float(r["fraction_residual_variance_explained_by_eye"]) for r in a_rows], width=0.36, label="A")
    axs[0].bar(x + 0.18, [float(r["fraction_residual_variance_explained_by_eye"]) for r in b_rows], width=0.36, label="B")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([r["session"] for r in a_rows], rotation=25, ha="right")
    axs[0].set_ylabel("Fraction residual variance explained")
    axs[0].set_title("Correction strength")
    axs[0].legend(frameon=False)

    axs[1].plot(
        x,
        [float(r["corr_C_old_CFEM_C_phase1_Ceye"]) for r in a_rows],
        "o-",
        label="corr(C_old_CFEM, C_phase1_Ceye)",
    )
    axs[1].plot(
        x,
        [float(r["corr_C_raw_C_eye_pred"]) for r in a_rows],
        "o-",
        label="corr(C_raw, C_eye_pred)",
    )
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([r["session"] for r in a_rows], rotation=25, ha="right")
    axs[1].set_ylabel("Upper-triangle matrix correlation")
    axs[1].set_title("Covariance alignment diagnostics")
    axs[1].legend(frameon=False)
    fig.savefig(rec_dir / "correction_strength_comparison.png", dpi=180)
    plt.close(fig)

    # Old number source note
    old_number_source = (
        "unresolved_not_found: no exact in-repo match found for +0.0492 -> -0.0179 in scripts/, "
        "declan/, or current outputs; prior artifact/log source remains unresolved."
    )
    _build_method_comparison_md(rec_dir / "old_vs_phase1_method_comparison.md", old_number_source)

    # Update phase1 summary files with strict sign-flip criteria
    _regenerate_phase1_summaries(out_root, synth_numeric_rows)

    print(f"Wrote reconciliation artifacts to {rec_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
