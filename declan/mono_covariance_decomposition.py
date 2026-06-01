#!/usr/bin/env python3
"""Covariance decomposition on mono rate-cache population.

Implements mechanism-focused observers on the 756-dim Model A features:
- full_empirical
- mean_only
- covariance_only
- positional_covariance
- residual_covariance
- jacobian_positional (optional if Arm B cache is available)

Outputs per-LogMAR absolute accuracies, deltas, bootstrap CIs, sign agreement,
and mechanism-level decision labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ORIENTATIONS = (0, 90, 180, 270)
LOGMARS_DEFAULT = (-0.40, -0.35, -0.30, -0.25, -0.20)
CANONICAL_TRACES = 471
EPS = 1e-12
INTERIM_LABEL = "empirical_observer_reproduces_modelA_FEM_benefit__geometry_mechanism_pending"
MAX_DELTA_MULTIPLIER_FOR_PLAUSIBLE_REPRO = 2.0
MIN_RELATIVE_ACCURACY_FLOOR = 0.75


@dataclass(frozen=True)
class Bundle:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    eye: np.ndarray
    n_groups: int
    feature_dim: int


def _fmt_lm(v: float) -> str:
    return f"{v:.2f}"


def _cache_path(rates_dir: Path, logmar: float, ori: int, cond: str) -> Path:
    return rates_dir / f"rates_hires_lm{_fmt_lm(logmar)}_ori{int(ori)}_{cond}.npz"


def _load_trial_means(path: Path, n_traces: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    rates = d["rates"]
    lengths = d["lengths"].astype(int)
    n_use = min(int(n_traces), int(rates.shape[0]))
    feats = []
    for i in range(n_use):
        t = int(lengths[i])
        feats.append(np.asarray(rates[i, :t], dtype=np.float64).mean(axis=0))
    return np.asarray(feats, dtype=np.float64), lengths[:n_use]


def _load_eye_means(eye_npz: Path, n_traces: int) -> np.ndarray:
    d = np.load(eye_npz, allow_pickle=True)
    traces = d["traces"].astype(np.float64)
    durations = d["durations"].astype(int)
    n_use = min(int(n_traces), int(traces.shape[0]))
    out = []
    for i in range(n_use):
        t = int(durations[i])
        out.append(np.asarray(traces[i, :t], dtype=np.float64).mean(axis=0))
    return np.asarray(out, dtype=np.float64)


def build_bundle(rates_dir: Path, eye_npz: Path, logmar: float, cond: str, n_traces: int) -> Bundle:
    eye_means = _load_eye_means(eye_npz, n_traces)
    by_class = []
    n_per_class = []
    for ori in ORIENTATIONS:
        p = _cache_path(rates_dir, logmar, ori, cond)
        if not p.exists():
            raise FileNotFoundError(str(p))
        feats, _ = _load_trial_means(p, n_traces=n_traces)
        by_class.append(feats)
        n_per_class.append(feats.shape[0])

    n_use = min(min(n_per_class), eye_means.shape[0])
    by_class = [x[:n_use] for x in by_class]
    eye_use = eye_means[:n_use]

    X = np.concatenate(by_class, axis=0)
    y = np.concatenate([np.full(n_use, i, dtype=int) for i in range(len(ORIENTATIONS))])
    groups = np.tile(np.arange(n_use, dtype=int), len(ORIENTATIONS))
    eye = np.tile(eye_use, (len(ORIENTATIONS), 1))

    return Bundle(
        X=X,
        y=y,
        groups=groups,
        eye=eye,
        n_groups=n_use,
        feature_dim=int(X.shape[1]),
    )


def _pooled_cov(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    d = X.shape[1]
    S = np.zeros((d, d), dtype=np.float64)
    means = []
    n_tot = 0
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        means.append(mu)
        C = Xc - mu
        S += C.T @ C
        n_tot += Xc.shape[0]
    denom = max(n_tot - len(classes), 1)
    return np.stack(means, axis=0), S / float(denom)


def _decompose_cov(X: np.ndarray, y: np.ndarray, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y)
    d = X.shape[1]
    S_tot = np.zeros((d, d), dtype=np.float64)
    S_pos = np.zeros((d, d), dtype=np.float64)
    S_res = np.zeros((d, d), dtype=np.float64)
    means = []
    n_tot = 0

    for c in classes:
        idx = (y == c)
        Xc = X[idx]
        Ec = eye[idx]
        mu = Xc.mean(axis=0)
        means.append(mu)

        Ctot = Xc - mu
        S_tot += Ctot.T @ Ctot

        E_aug = np.concatenate([np.ones((Ec.shape[0], 1), dtype=np.float64), Ec], axis=1)
        B = np.linalg.pinv(E_aug) @ Xc
        Xhat = E_aug @ B
        Xhat_c = Xhat - Xhat.mean(axis=0)
        Xres = Xc - Xhat
        Xres_c = Xres - Xres.mean(axis=0)

        S_pos += Xhat_c.T @ Xhat_c
        S_res += Xres_c.T @ Xres_c
        n_tot += Xc.shape[0]

    denom = max(n_tot - len(classes), 1)
    return np.stack(means, axis=0), S_tot / float(denom), S_pos / float(denom), S_res / float(denom)


def _reg_inverse(cov: np.ndarray, ridge_frac: float) -> np.ndarray:
    d = cov.shape[0]
    ridge = ridge_frac * float(np.trace(cov) / max(d, 1) + EPS)
    return np.linalg.pinv(cov + ridge * np.eye(d))


def _predict_gaussian(X: np.ndarray, means: np.ndarray, cov: np.ndarray, ridge_frac: float) -> np.ndarray:
    inv = _reg_inverse(cov, ridge_frac=ridge_frac)
    quad = 0.5 * np.sum((means @ inv) * means, axis=1)[None, :]
    scores = X @ inv @ means.T - quad
    return np.argmax(scores, axis=1)


def _predict_modelA_logreg(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int) -> np.ndarray:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=seed)
    clf.fit(Xtr_s, ytr)
    return clf.predict(Xte_s)


def _bootstrap_ci(correct_real: np.ndarray, correct_stab: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = correct_real.shape[0]
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.nanmean(correct_real[idx]) - np.nanmean(correct_stab[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def _correct_matrix(y: np.ndarray, groups: np.ndarray, pred: np.ndarray) -> np.ndarray:
    n_groups = int(np.max(groups)) + 1
    n_classes = int(np.max(y)) + 1
    mat = np.full((n_groups, n_classes), np.nan, dtype=np.float64)
    for i in range(y.shape[0]):
        mat[int(groups[i]), int(y[i])] = float(pred[i] == y[i])
    return mat


def _sign(v: float, tol: float = 1e-9) -> int:
    if v > tol:
        return 1
    if v < -tol:
        return -1
    return 0


def _jacobian_cov_if_available(logmar: float, jac_dir: Path, d: int) -> np.ndarray | None:
    if not jac_dir.exists():
        return None
    # Optional Arm B cache convention: jacobian_pos_cov_lm-0.35.npy
    p = jac_dir / f"jacobian_pos_cov_lm{_fmt_lm(logmar)}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (d, d):
        return None
    return arr


def run(args: argparse.Namespace) -> None:
    rates_dir = Path(args.rates_dir)
    eye_npz = Path(args.eye_traces)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jac_dir = Path(args.jacobian_cov_dir)

    logmars = [float(x) for x in args.logmars]

    # Revise current decision label as requested.
    for p in [
        Path("outputs/jacobian_predictive_framework/mono_empirical_keystone_fullgrid_20260601_regemp/mono_empirical_decision_table.csv"),
        Path("outputs/jacobian_predictive_framework/mono_empirical_keystone_fullgrid_20260601_regemp_sensitivity/mono_empirical_decision_table.csv"),
    ]:
        if p.exists():
            df = pd.read_csv(p)
            if "final_label" in df.columns:
                df.loc[:, "final_label"] = INTERIM_LABEL
                df.to_csv(p, index=False)

    all_rows = []
    observer_names = [
        "full_empirical",
        "mean_only",
        "covariance_only",
        "real_means_plus_stabilized_covariance",
        "stabilized_means_plus_real_covariance",
        "positional_covariance",
        "residual_covariance",
        "jacobian_positional",
    ]

    per_obs_lm_delta = {k: {} for k in observer_names if k != "jacobian_positional"}
    modelA_delta_by_lm = {}
    jac_available_any = False

    for lm in logmars:
        b_real = build_bundle(rates_dir, eye_npz, lm, "real", args.n_traces)
        b_stab = build_bundle(rates_dir, eye_npz, lm, "stabilized", args.n_traces)

        if b_real.feature_dim != b_stab.feature_dim:
            raise RuntimeError("Feature dimension mismatch between conditions")
        d = b_real.feature_dim

        # OOF predictions storage.
        pred_modelA = {
            "real": np.full_like(b_real.y, -1),
            "stabilized": np.full_like(b_stab.y, -1),
        }
        pred = {
            obs: {
                "real": np.full_like(b_real.y, -1),
                "stabilized": np.full_like(b_stab.y, -1),
            }
            for obs in observer_names
        }

        gkf = GroupKFold(n_splits=args.n_splits)
        for fold_idx, (tr, te) in enumerate(gkf.split(b_real.X, b_real.y, groups=b_real.groups)):
            # Model A baseline (condition-specific grouped CV logistic regression).
            pred_modelA["real"][te] = _predict_modelA_logreg(b_real.X[tr], b_real.y[tr], b_real.X[te], args.seed + fold_idx)
            pred_modelA["stabilized"][te] = _predict_modelA_logreg(b_stab.X[tr], b_stab.y[tr], b_stab.X[te], args.seed + 100 + fold_idx)

            # Training stats by condition.
            mu_r, cov_r, cov_pos_r, cov_res_r = _decompose_cov(b_real.X[tr], b_real.y[tr], b_real.eye[tr])
            mu_s, cov_s, cov_pos_s, cov_res_s = _decompose_cov(b_stab.X[tr], b_stab.y[tr], b_stab.eye[tr])

            # Shared stats.
            X_cat = np.concatenate([b_real.X[tr], b_stab.X[tr]], axis=0)
            y_cat = np.concatenate([b_real.y[tr], b_stab.y[tr]], axis=0)
            mu_shared, cov_shared = _pooled_cov(X_cat, y_cat)
            cov_pos_shared = 0.5 * (cov_pos_r + cov_pos_s)
            cov_res_shared = 0.5 * (cov_res_r + cov_res_s)

            # Optional Jacobian positional covariance.
            jac_cov = _jacobian_cov_if_available(lm, jac_dir, d)
            if jac_cov is not None:
                jac_available_any = True

            for cond in ("real", "stabilized"):
                Xte = b_real.X[te] if cond == "real" else b_stab.X[te]
                if cond == "real":
                    mu_cond, cov_cond, cov_pos_cond, cov_res_cond = mu_r, cov_r, cov_pos_r, cov_res_r
                else:
                    mu_cond, cov_cond, cov_pos_cond, cov_res_cond = mu_s, cov_s, cov_pos_s, cov_res_s

                pred["full_empirical"][cond][te] = _predict_gaussian(Xte, mu_cond, cov_cond, args.ridge_frac)
                pred["mean_only"][cond][te] = _predict_gaussian(Xte, mu_cond, cov_shared, args.ridge_frac)
                pred["covariance_only"][cond][te] = _predict_gaussian(Xte, mu_shared, cov_cond, args.ridge_frac)
                pred["real_means_plus_stabilized_covariance"][cond][te] = _predict_gaussian(Xte, mu_r, cov_s, args.ridge_frac)
                pred["stabilized_means_plus_real_covariance"][cond][te] = _predict_gaussian(Xte, mu_s, cov_r, args.ridge_frac)

                cov_pos_only = cov_res_shared + cov_pos_cond
                pred["positional_covariance"][cond][te] = _predict_gaussian(Xte, mu_shared, cov_pos_only, args.ridge_frac)

                cov_res_only = cov_pos_shared + cov_res_cond
                pred["residual_covariance"][cond][te] = _predict_gaussian(Xte, mu_shared, cov_res_only, args.ridge_frac)

                if jac_cov is not None:
                    cov_jac = cov_res_shared + jac_cov
                    pred["jacobian_positional"][cond][te] = _predict_gaussian(Xte, mu_shared, cov_jac, args.ridge_frac)

        # Aggregate model A baseline.
        mat_m_real = _correct_matrix(b_real.y, b_real.groups, pred_modelA["real"])
        mat_m_stab = _correct_matrix(b_stab.y, b_stab.groups, pred_modelA["stabilized"])
        m_real_acc = float(np.mean(pred_modelA["real"] == b_real.y))
        m_stab_acc = float(np.mean(pred_modelA["stabilized"] == b_stab.y))
        m_delta = m_real_acc - m_stab_acc
        modelA_delta_by_lm[lm] = m_delta

        for obs in observer_names:
            if obs == "jacobian_positional" and not jac_available_any:
                continue
            if np.any(pred[obs]["real"] < 0) or np.any(pred[obs]["stabilized"] < 0):
                continue

            mat_r = _correct_matrix(b_real.y, b_real.groups, pred[obs]["real"])
            mat_s = _correct_matrix(b_stab.y, b_stab.groups, pred[obs]["stabilized"])
            r_acc = float(np.mean(pred[obs]["real"] == b_real.y))
            s_acc = float(np.mean(pred[obs]["stabilized"] == b_stab.y))
            dlt = r_acc - s_acc
            ci_lo, ci_hi = _bootstrap_ci(mat_r, mat_s, args.n_boot, args.seed + int(round(2000 * (lm + 2))) + hash(obs) % 1000)

            per_obs_lm_delta.setdefault(obs, {})[lm] = dlt
            all_rows.append({
                "observer": obs,
                "logmar": lm,
                "feature_dim": d,
                "n_traces": int(b_real.n_groups),
                "task_n_way": 4,
                "modelA_acc_real": m_real_acc,
                "modelA_acc_stabilized": m_stab_acc,
                "modelA_delta_real_minus_stabilized": m_delta,
                "observer_acc_real": r_acc,
                "observer_acc_stabilized": s_acc,
                "observer_delta_real_minus_stabilized": dlt,
                "observer_delta_ci_lo": ci_lo,
                "observer_delta_ci_hi": ci_hi,
                "carrier_contrast": dlt,
                "fraction_modelA_delta_preserved": float(dlt / m_delta) if abs(m_delta) > EPS else np.nan,
                "sign_agreement_with_modelA": int(_sign(dlt) == _sign(m_delta)),
                "fine_scale_benefit_reproduced_at_neg0p35": int(False),
                "larger_size_decay_toward_zero_reproduced": int(False),
                "is_render_limit_control": int(abs(lm - float(args.render_limit)) < 1e-9),
            })

    df = pd.DataFrame(all_rows).sort_values(["observer", "logmar"]).reset_index(drop=True)

    # Observer-level reproduction flags.
    summary_rows = []
    for obs, dmap in per_obs_lm_delta.items():
        if -0.35 not in dmap or -0.20 not in dmap:
            continue
        fine = dmap[-0.35] > 0.0
        decay = abs(dmap[-0.20]) < abs(dmap[-0.35])
        sign_rate = float(np.mean([
            _sign(dmap[lm]) == _sign(modelA_delta_by_lm.get(lm, np.nan))
            for lm in dmap.keys()
            if lm in modelA_delta_by_lm and abs(lm - float(args.render_limit)) > 1e-9
        ]))
        summary_rows.append({
            "observer": obs,
            "fine_scale_benefit_reproduced_at_neg0p35": int(fine),
            "larger_size_decay_toward_zero_reproduced": int(decay),
            "evidence_sign_agreement_rate_noncontrol": sign_rate,
            "reproduces_benefit_pattern": int(fine and decay),
        })

    summary_df = pd.DataFrame(summary_rows)

    def _repro(obs: str) -> bool:
        if summary_df.empty or obs not in set(summary_df["observer"]):
            return False
        row = summary_df[summary_df["observer"] == obs].iloc[0]
        return bool(row["reproduces_benefit_pattern"])

    def _plausible_covariance_observer(obs: str) -> bool:
        """Reject p>>n covariance observers that show implausible overfit behavior.

        A mechanism reproduction is considered implausible if it relies on deltas
        that are many times larger than Model A (below threshold) or if absolute
        decoding accuracy is far below Model A where Model A is strong.
        """
        if df.empty or obs not in set(df["observer"]):
            return False

        t = df[df["observer"] == obs].copy()
        non_ctrl = t[t["is_render_limit_control"] == 0].copy()
        if non_ctrl.empty:
            return False

        # Focus plausibility checks on the below-threshold side where Model A is strong.
        below = non_ctrl[non_ctrl["logmar"] <= -0.30].copy()
        check = below if not below.empty else non_ctrl

        # 1) Delta magnitude sanity against Model A.
        ratio_ok = True
        for _, r in check.iterrows():
            m = float(abs(r["modelA_delta_real_minus_stabilized"]))
            dlt = float(abs(r["observer_delta_real_minus_stabilized"]))
            if m > EPS and dlt > MAX_DELTA_MULTIPLIER_FOR_PLAUSIBLE_REPRO * m:
                ratio_ok = False
                break

        # 2) Absolute accuracy floor relative to Model A (prevent degraded decoders).
        acc_ok = True
        for _, r in check.iterrows():
            m_real = float(r["modelA_acc_real"])
            m_stab = float(r["modelA_acc_stabilized"])
            o_real = float(r["observer_acc_real"])
            o_stab = float(r["observer_acc_stabilized"])
            if m_real > 0.85 and o_real < MIN_RELATIVE_ACCURACY_FLOOR * m_real:
                acc_ok = False
                break
            if m_stab > 0.80 and o_stab < MIN_RELATIVE_ACCURACY_FLOOR * m_stab:
                acc_ok = False
                break

        return bool(ratio_ok and acc_ok)

    jac_repro = jac_available_any and _repro("jacobian_positional") and _plausible_covariance_observer("jacobian_positional")
    pos_repro = _repro("positional_covariance") and _plausible_covariance_observer("positional_covariance")
    mean_repro = _repro("mean_only")
    cov_only_repro = _repro("covariance_only") and _plausible_covariance_observer("covariance_only")
    swap_realmeans_stabcov_repro = _repro("real_means_plus_stabilized_covariance") and _plausible_covariance_observer("real_means_plus_stabilized_covariance")
    swap_stabmeans_realcov_repro = _repro("stabilized_means_plus_real_covariance") and _plausible_covariance_observer("stabilized_means_plus_real_covariance")
    res_repro = _repro("residual_covariance") and _plausible_covariance_observer("residual_covariance")
    full_repro = _repro("full_empirical")

    mechanism_hits = int(jac_repro) + int(pos_repro) + int(mean_repro) + int(res_repro)

    # User-requested primary decision rule for first-order mean mechanism.
    if mean_repro and not cov_only_repro:
        final_label = "first_order_mean_geometry_explains_benefit"
    elif jac_repro and mechanism_hits == 1:
        final_label = "jacobian_positional_covariance_reproduces_modelA_benefit"
    elif pos_repro and mechanism_hits == 1:
        final_label = "positional_covariance_reproduces_modelA_benefit"
    elif mean_repro and mechanism_hits == 1:
        final_label = "mean_only_explains_benefit"
    elif res_repro and mechanism_hits == 1:
        final_label = "residual_covariance_explains_benefit"
    elif full_repro and mechanism_hits == 0:
        final_label = "full_empirical_only_geometry_not_isolated"
    elif full_repro:
        # Avoid concluding "inconclusive" solely because multiple covariance observers pass.
        final_label = "full_empirical_only_geometry_not_isolated"
    else:
        final_label = "mechanism_inconclusive"

    confirmation_label = (
        "benefit_carried_by_condition_specific_class_means"
        if (swap_realmeans_stabcov_repro and not swap_stabmeans_realcov_repro)
        else "not_confirmed"
    )

    # Fill per-row booleans from summary.
    if not df.empty and not summary_df.empty:
        for obs in df["observer"].unique():
            if obs in set(summary_df["observer"]):
                row = summary_df[summary_df["observer"] == obs].iloc[0]
                df.loc[df["observer"] == obs, "fine_scale_benefit_reproduced_at_neg0p35"] = int(row["fine_scale_benefit_reproduced_at_neg0p35"])
                df.loc[df["observer"] == obs, "larger_size_decay_toward_zero_reproduced"] = int(row["larger_size_decay_toward_zero_reproduced"])

    df.to_csv(out_dir / "mono_covariance_decomposition_table.csv", index=False)
    summary_df.to_csv(out_dir / "mono_covariance_decomposition_summary.csv", index=False)

    decision = pd.DataFrame([{
        "interim_label_for_full_empirical_run": INTERIM_LABEL,
        "final_label": final_label,
        "confirmation_label": confirmation_label,
        "feature_dim": int(df["feature_dim"].iloc[0]) if not df.empty else np.nan,
        "n_traces": int(df["n_traces"].iloc[0]) if not df.empty else np.nan,
        "task_n_way": 4,
        "render_limit_control_logmar": float(args.render_limit),
        "jacobian_arm_b_available": int(jac_available_any),
        "notes": "No inherited 49-unit calibration metadata used; rate-cache-only observer analysis.",
    }])
    decision.to_csv(out_dir / "mono_covariance_decomposition_decision.csv", index=False)

    print("Mono covariance decomposition complete")
    print(f"  out_dir={out_dir}")
    print(f"  final_label={final_label}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rates-dir", type=str, default="scripts/temporal_decoding/data/rates")
    p.add_argument("--eye-traces", type=str, default="scripts/temporal_decoding/data/eye_traces.npz")
    p.add_argument("--out-dir", type=str, default="outputs/jacobian_predictive_framework/mono_covariance_decomposition_20260601")
    p.add_argument("--logmars", type=float, nargs="+", default=list(LOGMARS_DEFAULT))
    p.add_argument("--render-limit", type=float, default=-0.40)
    p.add_argument("--n-traces", type=int, default=CANONICAL_TRACES)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ridge-frac", type=float, default=1e-3)
    p.add_argument("--jacobian-cov-dir", type=str, default="outputs/stats/eoptotype_jacobian_field_smoothness_pilot3")
    run(p.parse_args())
