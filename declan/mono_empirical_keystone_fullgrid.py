#!/usr/bin/env python3
"""Mono full-grid empirical-covariance Keystone analysis (rate-cache only).

This script evaluates 4-way E-optotype decoding on cached mono rates using:
- Model A baseline: grouped-CV logistic regression on time-averaged rates.
- Empirical-covariance observers: Fisher LDA, shrinkage LDA, and regularized
  empirical covariance classifier.

It is intentionally independent of Keystone calibration metadata and uses only
rate-cache trial responses, preserving the 756-dim mono feature space.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_LOGMARS = (-0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10)
CANONICAL_TRACES = 471
EPS = 1e-12
INTERIM_LABEL = "empirical_observer_reproduces_modelA_FEM_benefit__geometry_mechanism_pending"


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    n_groups: int
    feature_dim: int


def _fmt_lm(lm: float) -> str:
    return f"{lm:.2f}"


def _cache_path(rates_dir: Path, logmar: float, ori: int, condition: str) -> Path:
    return rates_dir / f"rates_hires_lm{_fmt_lm(logmar)}_ori{int(ori)}_{condition}.npz"


def _load_npz_trial_means(path: Path, n_traces: int) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    rates = d["rates"]
    lengths = d["lengths"].astype(int)
    n_use = min(int(n_traces), int(rates.shape[0]))
    feats = []
    for i in range(n_use):
        t = int(lengths[i])
        feats.append(np.asarray(rates[i, :t], dtype=np.float64).mean(axis=0))
    return np.asarray(feats, dtype=np.float64)


def build_dataset(
    rates_dir: Path,
    logmar: float,
    condition: str,
    orientations: tuple[int, ...],
    n_traces: int,
) -> Dataset:
    by_class = []
    for ori in orientations:
        p = _cache_path(rates_dir, logmar, ori, condition)
        if not p.exists():
            raise FileNotFoundError(str(p))
        by_class.append(_load_npz_trial_means(p, n_traces=n_traces))

    n_use = min(x.shape[0] for x in by_class)
    by_class = [x[:n_use] for x in by_class]
    feat_dim = int(by_class[0].shape[1])

    X = np.concatenate(by_class, axis=0)
    y = np.concatenate([
        np.full(n_use, i, dtype=int) for i in range(len(orientations))
    ])
    groups = np.tile(np.arange(n_use, dtype=int), len(orientations))
    return Dataset(X=X, y=y, groups=groups, n_groups=n_use, feature_dim=feat_dim)


def _fit_predict_modelA(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, seed: int) -> np.ndarray:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=seed)
    clf.fit(X_tr_s, y_tr)
    return clf.predict(X_te_s)


def _fit_predict_lda_empirical(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, _: int) -> np.ndarray:
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def _fit_predict_lda_shrinkage(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, _: int) -> np.ndarray:
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)


def _fit_predict_reg_empirical(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, _: int) -> np.ndarray:
    classes = np.unique(y_tr)
    means = []
    pooled = np.zeros((X_tr.shape[1], X_tr.shape[1]), dtype=np.float64)
    n_tot = 0

    for c in classes:
        xc = X_tr[y_tr == c]
        means.append(xc.mean(axis=0))
        centered = xc - means[-1]
        pooled += centered.T @ centered
        n_tot += xc.shape[0]

    denom = max(n_tot - len(classes), 1)
    cov = pooled / float(denom)
    ridge = 1e-3 * float(np.trace(cov) / max(cov.shape[0], 1) + EPS)
    cov_reg = cov + ridge * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov_reg)

    means = np.stack(means, axis=0)
    scores = X_te @ inv @ means.T - 0.5 * np.sum((means @ inv) * means, axis=1)[None, :]
    pred_idx = np.argmax(scores, axis=1)
    return classes[pred_idx]


def grouped_cv_eval(
    ds: Dataset,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray],
    n_splits: int,
    seed: int,
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(ds.y.shape[0], -1, dtype=int)
    fold_acc = []

    for fold_idx, (tr, te) in enumerate(gkf.split(ds.X, ds.y, groups=ds.groups)):
        yhat = fit_predict(ds.X[tr], ds.y[tr], ds.X[te], seed + fold_idx)
        oof_pred[te] = yhat
        fold_acc.append(float(np.mean(yhat == ds.y[te])))

    if np.any(oof_pred < 0):
        raise RuntimeError("Some samples did not receive OOF predictions")

    correct = (oof_pred == ds.y).astype(np.float64)
    mat = np.full((ds.n_groups, len(np.unique(ds.y))), np.nan, dtype=np.float64)
    for i in range(ds.y.shape[0]):
        g = int(ds.groups[i])
        c = int(ds.y[i])
        mat[g, c] = correct[i]

    return {
        "acc_mean": float(np.mean(correct)),
        "acc_std": float(np.std(fold_acc)),
        "fold_acc": np.asarray(fold_acc, dtype=np.float64),
        "correct_matrix": mat,
    }


def bootstrap_delta_ci(
    correct_real: np.ndarray,
    correct_stab: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if correct_real.shape != correct_stab.shape:
        raise ValueError("real/stabilized correctness matrices must have same shape")
    n_groups = correct_real.shape[0]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_groups, size=n_groups)
        deltas[i] = float(np.nanmean(correct_real[idx]) - np.nanmean(correct_stab[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def _sign(x: float, tol: float = 1e-9) -> int:
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0


def run(args: argparse.Namespace) -> None:
    rates_dir = Path(args.rates_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logmars = [float(x) for x in args.logmars]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if "real" not in conditions or "stabilized" not in conditions:
        raise ValueError("conditions must include real and stabilized")

    methods = {
        "modelA_logreg": _fit_predict_modelA,
        "observer_lda_empirical": _fit_predict_lda_empirical,
        "observer_lda_shrinkage": _fit_predict_lda_shrinkage,
        "observer_reg_empirical": _fit_predict_reg_empirical,
    }
    observer_methods = [m for m in methods if m.startswith("observer_")]

    available_fixed_center = {}
    all_rows = []
    eval_store: dict[tuple[float, str, str], dict] = {}
    feature_dims = set()

    anchor = float(args.anchor_logmar)

    # Step 1: evaluate all empirical observers at anchor only to choose primary method.
    for cond in conditions:
        ds_anchor = build_dataset(
            rates_dir=rates_dir,
            logmar=anchor,
            condition=cond,
            orientations=ORIENTATIONS,
            n_traces=args.n_traces,
        )
        feature_dims.add(ds_anchor.feature_dim)
        for method_name in ["modelA_logreg", *observer_methods]:
            ev = grouped_cv_eval(ds_anchor, methods[method_name], n_splits=args.n_splits, seed=args.seed)
            eval_store[(anchor, cond, method_name)] = ev
            all_rows.append({
                "logmar": anchor,
                "condition": cond,
                "method": method_name,
                "accuracy": ev["acc_mean"],
                "accuracy_std": ev["acc_std"],
                "n_groups": ds_anchor.n_groups,
                "feature_dim": ds_anchor.feature_dim,
                "n_classes": len(ORIENTATIONS),
            })

    anchor_scores = {}
    for m in observer_methods:
        r = float(eval_store[(anchor, "real", m)]["acc_mean"])
        s = float(eval_store[(anchor, "stabilized", m)]["acc_mean"])
        anchor_scores[m] = 0.5 * (r + s)
    primary_obs = max(anchor_scores, key=anchor_scores.get)
    if args.primary_observer:
        if args.primary_observer not in observer_methods:
            raise ValueError(f"Unknown primary observer: {args.primary_observer}")
        primary_obs = str(args.primary_observer)

    # Step 2: run full grid for Model A + selected empirical observer.
    for lm in logmars:
        for cond in conditions:
            for method_name in ["modelA_logreg", primary_obs]:
                if (lm, cond, method_name) in eval_store:
                    continue
                ds = build_dataset(
                    rates_dir=rates_dir,
                    logmar=lm,
                    condition=cond,
                    orientations=ORIENTATIONS,
                    n_traces=args.n_traces,
                )
                feature_dims.add(ds.feature_dim)
                ev = grouped_cv_eval(ds, methods[method_name], n_splits=args.n_splits, seed=args.seed)
                eval_store[(lm, cond, method_name)] = ev
                all_rows.append({
                    "logmar": lm,
                    "condition": cond,
                    "method": method_name,
                    "accuracy": ev["acc_mean"],
                    "accuracy_std": ev["acc_std"],
                    "n_groups": ds.n_groups,
                    "feature_dim": ds.feature_dim,
                    "n_classes": len(ORIENTATIONS),
                })

        fc_ok = all(_cache_path(rates_dir, lm, int(ori), "fixed_center").exists() for ori in ORIENTATIONS)
        available_fixed_center[lm] = bool(fc_ok)
        if fc_ok:
            for method_name in ["modelA_logreg", primary_obs]:
                ds_fc = build_dataset(
                    rates_dir=rates_dir,
                    logmar=lm,
                    condition="fixed_center",
                    orientations=ORIENTATIONS,
                    n_traces=args.n_traces,
                )
                feature_dims.add(ds_fc.feature_dim)
                ev = grouped_cv_eval(ds_fc, methods[method_name], n_splits=args.n_splits, seed=args.seed)
                eval_store[(lm, "fixed_center", method_name)] = ev
                all_rows.append({
                    "logmar": lm,
                    "condition": "fixed_center",
                    "method": method_name,
                    "accuracy": ev["acc_mean"],
                    "accuracy_std": ev["acc_std"],
                    "n_groups": ds_fc.n_groups,
                    "feature_dim": ds_fc.feature_dim,
                    "n_classes": len(ORIENTATIONS),
                })

    if len(feature_dims) != 1:
        raise RuntimeError(f"Expected one feature_dim, found: {sorted(feature_dims)}")
    feature_dim = int(next(iter(feature_dims)))

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_dir / "mono_empirical_fullgrid_metrics.csv", index=False)

    # Keep anchor observer scores in decision table for transparency.

    crossover_rows = []
    for lm in logmars:
        m_real = eval_store[(lm, "real", "modelA_logreg")]
        m_stab = eval_store[(lm, "stabilized", "modelA_logreg")]
        o_real = eval_store[(lm, "real", primary_obs)]
        o_stab = eval_store[(lm, "stabilized", primary_obs)]

        m_delta = float(m_real["acc_mean"] - m_stab["acc_mean"])
        o_delta = float(o_real["acc_mean"] - o_stab["acc_mean"])
        m_ci_lo, m_ci_hi = bootstrap_delta_ci(
            m_real["correct_matrix"], m_stab["correct_matrix"], n_boot=args.n_boot, seed=args.seed + int(round(1000 * (lm + 2))),
        )
        o_ci_lo, o_ci_hi = bootstrap_delta_ci(
            o_real["correct_matrix"], o_stab["correct_matrix"], n_boot=args.n_boot, seed=args.seed + int(round(2000 * (lm + 2))),
        )

        fc_model = np.nan
        fc_obs = np.nan
        if available_fixed_center.get(lm, False):
            fc_model = float(eval_store[(lm, "fixed_center", "modelA_logreg")]["acc_mean"])
            fc_obs = float(eval_store[(lm, "fixed_center", primary_obs)]["acc_mean"])

        s_agree = int(_sign(m_delta) == _sign(o_delta))
        crossover_rows.append({
            "logmar": lm,
            "feature_dim": feature_dim,
            "modelA_acc_real": float(m_real["acc_mean"]),
            "modelA_acc_stabilized": float(m_stab["acc_mean"]),
            "modelA_delta_real_minus_stabilized": m_delta,
            "modelA_delta_ci_lo": m_ci_lo,
            "modelA_delta_ci_hi": m_ci_hi,
            "empirical_method": primary_obs,
            "empirical_acc_real": float(o_real["acc_mean"]),
            "empirical_acc_stabilized": float(o_stab["acc_mean"]),
            "empirical_delta_real_minus_stabilized": o_delta,
            "empirical_delta_ci_lo": o_ci_lo,
            "empirical_delta_ci_hi": o_ci_hi,
            "sign_agreement": s_agree,
            "is_render_limit_control": int(abs(lm - float(args.render_limit)) < 1e-9),
            "fixed_center_available": int(available_fixed_center.get(lm, False)),
            "modelA_acc_fixed_center": fc_model,
            "empirical_acc_fixed_center": fc_obs,
        })

    crossover_df = pd.DataFrame(crossover_rows).sort_values("logmar").reset_index(drop=True)
    crossover_df.to_csv(out_dir / "mono_empirical_crossover_table.csv", index=False)

    # Primary sanity check at -0.35.
    sanity_row = crossover_df[np.isclose(crossover_df["logmar"], anchor)].iloc[0].to_dict()
    sanity_df = pd.DataFrame([{
        "feature_dim": feature_dim,
        "logmar": anchor,
        "modelA_acc_real": sanity_row["modelA_acc_real"],
        "modelA_acc_stabilized": sanity_row["modelA_acc_stabilized"],
        "modelA_delta_real_minus_stabilized": sanity_row["modelA_delta_real_minus_stabilized"],
        "empirical_method": sanity_row["empirical_method"],
        "empirical_acc_real": sanity_row["empirical_acc_real"],
        "empirical_acc_stabilized": sanity_row["empirical_acc_stabilized"],
        "empirical_delta_real_minus_stabilized": sanity_row["empirical_delta_real_minus_stabilized"],
    }])
    sanity_df.to_csv(out_dir / "mono_empirical_sanity_logmar_neg0p35.csv", index=False)

    # Gate and final label logic.
    emp_real_035 = float(sanity_row["empirical_acc_real"])
    emp_stab_035 = float(sanity_row["empirical_acc_stabilized"])
    emp_delta_035 = float(sanity_row["empirical_delta_real_minus_stabilized"])

    near_chance = (emp_real_035 < args.near_chance_max) and (emp_stab_035 < args.near_chance_max)
    reaches_high = (max(emp_real_035, emp_stab_035) >= args.high_acc_min)
    reproduces_benefit = emp_delta_035 > 0.0

    gate_label = "proceed_to_crossover_test"
    if near_chance:
        gate_label = "mono_observer_pipeline_bug"
    elif reaches_high and not reproduces_benefit:
        gate_label = "empirical_geometry_does_not_capture_modelA_FEM_benefit"

    evidence_df = crossover_df[crossover_df["is_render_limit_control"] == 0].copy()
    if evidence_df.shape[0] < 2:
        final_label = "render_limit_confounded"
    elif gate_label == "mono_observer_pipeline_bug":
        final_label = "mono_observer_pipeline_bug"
    elif reaches_high and reproduces_benefit:
        final_label = INTERIM_LABEL
    else:
        final_label = "empirical_geometry_does_not_capture_modelA_FEM_benefit"

    decision_row = {
        "feature_dim": feature_dim,
        "render_limit_control_logmar": float(args.render_limit),
        "anchor_logmar": anchor,
        "empirical_method_primary": primary_obs,
        "empirical_method_anchor_scores": json.dumps(anchor_scores, sort_keys=True),
        "gate_label": gate_label,
        "final_label": final_label,
        "empirical_acc_real_at_anchor": emp_real_035,
        "empirical_acc_stabilized_at_anchor": emp_stab_035,
        "empirical_delta_at_anchor": emp_delta_035,
        "modelA_acc_real_at_anchor": float(sanity_row["modelA_acc_real"]),
        "modelA_acc_stabilized_at_anchor": float(sanity_row["modelA_acc_stabilized"]),
        "modelA_delta_at_anchor": float(sanity_row["modelA_delta_real_minus_stabilized"]),
        "near_chance_threshold": float(args.near_chance_max),
        "high_accuracy_threshold": float(args.high_acc_min),
        "n_logmar_total": int(crossover_df.shape[0]),
        "n_logmar_evidence": int(evidence_df.shape[0]),
    }
    decision_df = pd.DataFrame([decision_row])
    decision_df.to_csv(out_dir / "mono_empirical_decision_table.csv", index=False)

    print("Mono empirical full-grid analysis complete")
    print(f"  out_dir={out_dir}")
    print(f"  feature_dim={feature_dim}")
    print(f"  primary_observer={primary_obs}")
    print(f"  gate_label={gate_label}")
    print(f"  final_label={final_label}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rates-dir",
        type=str,
        default="scripts/temporal_decoding/data/rates",
        help="Directory containing rates_hires_lm*.npz caches.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/jacobian_predictive_framework/mono_empirical_keystone_fullgrid_20260601",
    )
    p.add_argument("--logmars", type=float, nargs="+", default=list(DEFAULT_LOGMARS))
    p.add_argument("--render-limit", type=float, default=-0.40)
    p.add_argument("--anchor-logmar", type=float, default=-0.35)
    p.add_argument("--conditions", type=str, default="real,stabilized")
    p.add_argument("--n-traces", type=int, default=CANONICAL_TRACES)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--primary-observer",
        type=str,
        default="",
        help="Optional override for grid observer method (observer_lda_empirical|observer_lda_shrinkage|observer_reg_empirical).",
    )
    p.add_argument("--near-chance-max", type=float, default=0.35)
    p.add_argument("--high-acc-min", type=float, default=0.85)
    p.add_argument("--min-sign-agreement-for-crossover", type=float, default=0.60)
    run(p.parse_args())
