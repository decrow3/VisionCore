#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

from VisionCore.paths import VISIONCORE_ROOT
from scripts.figure4.run_canonical_eoptotype_discrimination import (
    _decode_one,
    _effect_status,
    _load_rate_file,
    _paired_bootstrap_delta,
    _rate_path,
    _safe_float,
    _window_mean,
    _write_csv,
    DecodeResult,
)


ORIENTATIONS_DEFAULT = (0, 90, 180, 270)
LOGMARS_DEFAULT = (-0.40, -0.35, -0.30, -0.25, -0.20)
WINDOWS_DEFAULT = (1, 5, 10, 20, 30, 60)
CONDITIONS = ("real", "stabilized")
PRIMARY_WINDOW = 60
PRIMARY_LOGMARS = (-0.35, -0.30, -0.25)
TRAJECTORY_BINS_DEFAULT = 4


@dataclass(frozen=True)
class ObserverConditionBundle:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_path: str


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(x)) for x in text.split(",") if x.strip())


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_readme(path: Path, lines: list[str]) -> None:
    _ensure_parent(path)
    path.write_text("\n".join(lines) + "\n")


def _load_manifest(root: Path) -> dict[str, Any]:
    json_path = root / "model_population_manifest.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    csv_path = root / "model_population_manifest.csv"
    if csv_path.exists():
        rows = _load_rows(csv_path)
        return rows[0] if rows else {}
    return {}


def _index_metrics(rows: list[dict[str, Any]]) -> dict[tuple[float, int, str], dict[str, Any]]:
    out: dict[tuple[float, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (float(row["logmar"]), int(float(row["window"])), str(row["condition"]))
        out[key] = row
    return out


def _index_contrasts(rows: list[dict[str, Any]]) -> dict[tuple[float, int], dict[str, Any]]:
    out: dict[tuple[float, int], dict[str, Any]] = {}
    for row in rows:
        key = (float(row["logmar"]), int(float(row["window"])))
        out[key] = row
    return out


def _load_trial_means(path: Path, n_traces: int) -> tuple[np.ndarray, np.ndarray]:
    rates, lengths = _load_rate_file(path)
    n_use = min(int(n_traces), int(rates.shape[0]))
    feats = []
    for i in range(n_use):
        t = max(1, int(lengths[i]))
        feats.append(np.asarray(rates[i, :t], dtype=np.float64).mean(axis=0))
    return np.asarray(feats, dtype=np.float64), lengths[:n_use]


def _load_trajectory_features(path: Path, n_traces: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    rates, lengths = _load_rate_file(path)
    n_use = min(int(n_traces), int(rates.shape[0]))
    feats = []
    for i in range(n_use):
        t = max(1, int(lengths[i]))
        idx = np.linspace(0, t - 1, int(n_bins)).astype(int)
        feats.append(np.asarray(rates[i, idx], dtype=np.float64).reshape(-1))
    return np.asarray(feats, dtype=np.float64), lengths[:n_use]


@lru_cache(maxsize=64)
def _load_raw_rate_cache(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_str)
    d = np.load(path, allow_pickle=True)
    rates = np.asarray(d["rates"], dtype=np.float32)
    lengths = np.asarray(d["lengths"], dtype=np.int32)
    return rates, lengths


def _load_raw_condition_bundle(
    *,
    rates_dir: Path,
    logmar: float,
    condition: str,
    n_traces: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ori in ORIENTATIONS_DEFAULT:
        p = _rate_path(rates_dir, logmar, int(ori), condition)
        if not p.exists():
            raise FileNotFoundError(str(p))
        rates, lengths = _load_raw_rate_cache(str(p))
        n_use = min(int(n_traces), int(rates.shape[0]))
        raw[int(ori)] = (rates[:n_use], lengths[:n_use])
    return raw


def _bundle_from_raw(
    raw_by_ori: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    feature_mode: str,
    feature_path: str,
    trajectory_bins: int,
) -> ObserverConditionBundle:
    X_by_ori = []
    n_per_ori = []
    for ori in ORIENTATIONS_DEFAULT:
        rates, lengths = raw_by_ori[int(ori)]
        if feature_mode == "time_mean_rate":
            feats = np.asarray([np.asarray(rates[i, : max(1, int(lengths[i]))], dtype=np.float64).mean(axis=0) for i in range(rates.shape[0])], dtype=np.float64)
        elif feature_mode == "trajectory":
            feats = np.asarray([
                np.asarray(rates[i, np.linspace(0, max(1, int(lengths[i])) - 1, int(trajectory_bins)).astype(int)], dtype=np.float64).reshape(-1)
                for i in range(rates.shape[0])
            ], dtype=np.float64)
        else:
            raise ValueError(f"unknown feature_mode={feature_mode}")
        X_by_ori.append(feats)
        n_per_ori.append(feats.shape[0])

    n_use = min(n_per_ori)
    X = np.concatenate([x[:n_use] for x in X_by_ori], axis=0)
    y = np.concatenate([np.full(n_use, idx, dtype=np.int32) for idx in range(len(ORIENTATIONS_DEFAULT))])
    groups = np.tile(np.arange(n_use, dtype=np.int32), len(ORIENTATIONS_DEFAULT))
    return ObserverConditionBundle(X=X, y=y, groups=groups, feature_path=feature_path)


def _build_condition_bundle(
    *,
    rates_dir: Path,
    logmar: float,
    window: int,
    condition: str,
    n_traces: int,
    feature_mode: str,
    trajectory_bins: int,
) -> ObserverConditionBundle:
    raw = _load_raw_condition_bundle(rates_dir=rates_dir, logmar=logmar, condition=condition, n_traces=n_traces)
    return _bundle_from_raw(
        raw,
        feature_mode=feature_mode,
        feature_path=str(rates_dir.relative_to(VISIONCORE_ROOT)),
        trajectory_bins=trajectory_bins,
    )


def _reg_inverse(cov: np.ndarray, ridge_frac: float) -> np.ndarray:
    d = int(cov.shape[0])
    ridge = float(ridge_frac) * float(np.trace(cov) / max(d, 1) + 1e-12)
    return np.linalg.pinv(cov + ridge * np.eye(d, dtype=np.float64))


def _predict_gaussian(X: np.ndarray, means: np.ndarray, cov: np.ndarray, ridge_frac: float) -> np.ndarray:
    inv = _reg_inverse(cov, ridge_frac=ridge_frac)
    quad = 0.5 * np.sum((means @ inv) * means, axis=1)[None, :]
    scores = X @ inv @ means.T - quad
    return np.argmax(scores, axis=1)


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


def _condition_accuracy(pred: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    per_sample = (pred == y).astype(np.float64)
    per_group = np.full(int(np.max(y)) + 1, np.nan, dtype=np.float64)
    for gi in range(per_group.shape[0]):
        per_group[gi] = float(np.mean(per_sample[y == gi]))
    conf = confusion_matrix(y, pred, labels=list(range(int(np.max(y)) + 1)))
    return float(np.mean(per_sample)), per_group, per_sample, conf, float(np.nanmean(np.sum(per_sample)))


def _decode_mean_only(
    real_bundle: ObserverConditionBundle,
    stab_bundle: ObserverConditionBundle,
    *,
    n_splits: int,
    seed: int,
    n_bootstrap: int,
    ridge_frac: float,
) -> tuple[DecodeResult, DecodeResult, float, float, float, float]:
    if real_bundle.X.shape[1] != stab_bundle.X.shape[1]:
        raise RuntimeError("feature dimension mismatch between conditions")

    m = min(real_bundle.X.shape[0] // len(ORIENTATIONS_DEFAULT), stab_bundle.X.shape[0] // len(ORIENTATIONS_DEFAULT))
    n_splits_eff = min(max(2, int(n_splits)), m)
    gkf = GroupKFold(n_splits=n_splits_eff)

    pred_real = np.full_like(real_bundle.y, -1)
    pred_stab = np.full_like(stab_bundle.y, -1)
    confs_real: list[np.ndarray] = []
    confs_stab: list[np.ndarray] = []

    for fold_idx, (tr, te) in enumerate(gkf.split(real_bundle.X, real_bundle.y, groups=real_bundle.groups)):
        X_cat = np.concatenate([real_bundle.X[tr], stab_bundle.X[tr]], axis=0)
        y_cat = np.concatenate([real_bundle.y[tr], stab_bundle.y[tr]], axis=0)
        mu_shared, cov_shared = _pooled_cov(X_cat, y_cat)

        mu_real, _ = _pooled_cov(real_bundle.X[tr], real_bundle.y[tr])
        mu_stab, _ = _pooled_cov(stab_bundle.X[tr], stab_bundle.y[tr])

        pred_real[te] = _predict_gaussian(real_bundle.X[te], mu_real, cov_shared, ridge_frac)
        pred_stab[te] = _predict_gaussian(stab_bundle.X[te], mu_stab, cov_shared, ridge_frac)

        confs_real.append(confusion_matrix(real_bundle.y[te], pred_real[te], labels=list(range(len(ORIENTATIONS_DEFAULT)))))
        confs_stab.append(confusion_matrix(stab_bundle.y[te], pred_stab[te], labels=list(range(len(ORIENTATIONS_DEFAULT)))))

    if np.any(pred_real < 0) or np.any(pred_stab < 0):
        raise RuntimeError("mean-only decoding failed to predict all heldout samples")

    real_correct = (pred_real == real_bundle.y).astype(np.float64)
    stab_correct = (pred_stab == stab_bundle.y).astype(np.float64)
    real_group = np.asarray([np.mean(real_correct[real_bundle.groups == gi]) for gi in range(m)], dtype=np.float64)
    stab_group = np.asarray([np.mean(stab_correct[stab_bundle.groups == gi]) for gi in range(m)], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    _, real_lo, real_hi = _bootstrap_mean_ci(real_group, rng, int(n_bootstrap))
    rng = np.random.default_rng(int(seed) + 17)
    _, stab_lo, stab_hi = _bootstrap_mean_ci(stab_group, rng, int(n_bootstrap))
    rng = np.random.default_rng(int(seed) + 101)
    delta, d_lo, d_hi, _ = _paired_bootstrap_delta(real_group, stab_group, rng, int(n_bootstrap))

    conf_real = np.sum(np.stack(confs_real, axis=0), axis=0)
    conf_stab = np.sum(np.stack(confs_stab, axis=0), axis=0)
    mi_real = _confusion_mi_bits(conf_real)
    mi_stab = _confusion_mi_bits(conf_stab)

    mean_total_real = float(np.mean(np.sum(real_bundle.X, axis=1)))
    mean_total_stab = float(np.mean(np.sum(stab_bundle.X, axis=1)))

    return (
        DecodeResult(
            accuracy=float(np.mean(real_correct)),
            balanced_accuracy=float(balanced_accuracy_score(real_bundle.y, pred_real)),
            ci_low=float(real_lo),
            ci_high=float(real_hi),
            confusion_mi_bits=float(mi_real),
            mean_total_expected_spikes=mean_total_real,
            confusion_by_split=confs_real,
            per_group_accuracy=real_group,
        ),
        DecodeResult(
            accuracy=float(np.mean(stab_correct)),
            balanced_accuracy=float(balanced_accuracy_score(stab_bundle.y, pred_stab)),
            ci_low=float(stab_lo),
            ci_high=float(stab_hi),
            confusion_mi_bits=float(mi_stab),
            mean_total_expected_spikes=mean_total_stab,
            confusion_by_split=confs_stab,
            per_group_accuracy=stab_group,
        ),
        float(delta),
        float(d_lo),
        float(d_hi),
        float(n_splits_eff),
    )


def _confusion_mi_bits(conf: np.ndarray) -> float:
    conf = np.asarray(conf, dtype=np.float64)
    total = np.sum(conf)
    if total <= 0:
        return float("nan")
    pxy = conf / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / (px @ py), 1.0)
        log_term = np.where(pxy > 0, np.log2(ratio), 0.0)
    return float(np.sum(pxy * log_term))


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    samples = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        draw = rng.choice(vals, size=vals.size, replace=True)
        samples[i] = float(np.mean(draw))
    return float(np.mean(vals)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _decode_trajectory(
    real_bundle: ObserverConditionBundle,
    stab_bundle: ObserverConditionBundle,
    *,
    n_splits: int,
    seed: int,
    n_bootstrap: int,
) -> tuple[DecodeResult, DecodeResult, float, float, float, float]:
    if real_bundle.X.shape[1] != stab_bundle.X.shape[1]:
        raise RuntimeError("feature dimension mismatch between conditions")

    m = min(real_bundle.X.shape[0] // len(ORIENTATIONS_DEFAULT), stab_bundle.X.shape[0] // len(ORIENTATIONS_DEFAULT))
    n_splits_eff = min(max(2, int(n_splits)), m)
    gkf = GroupKFold(n_splits=n_splits_eff)

    pred_real = np.full_like(real_bundle.y, -1)
    pred_stab = np.full_like(stab_bundle.y, -1)
    confs_real: list[np.ndarray] = []
    confs_stab: list[np.ndarray] = []

    for fold_idx, (tr, te) in enumerate(gkf.split(real_bundle.X, real_bundle.y, groups=real_bundle.groups)):
        scaler = StandardScaler()
        Xtr_real = scaler.fit_transform(real_bundle.X[tr])
        Xte_real = scaler.transform(real_bundle.X[te])
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear", random_state=int(seed) + fold_idx))
        clf.fit(Xtr_real, real_bundle.y[tr])
        pred_real[te] = clf.predict(Xte_real)

        scaler = StandardScaler()
        Xtr_stab = scaler.fit_transform(stab_bundle.X[tr])
        Xte_stab = scaler.transform(stab_bundle.X[te])
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear", random_state=int(seed) + 100 + fold_idx))
        clf.fit(Xtr_stab, stab_bundle.y[tr])
        pred_stab[te] = clf.predict(Xte_stab)

        confs_real.append(confusion_matrix(real_bundle.y[te], pred_real[te], labels=list(range(len(ORIENTATIONS_DEFAULT)))))
        confs_stab.append(confusion_matrix(stab_bundle.y[te], pred_stab[te], labels=list(range(len(ORIENTATIONS_DEFAULT)))))

    if np.any(pred_real < 0) or np.any(pred_stab < 0):
        raise RuntimeError("trajectory decoding failed to predict all heldout samples")

    real_correct = (pred_real == real_bundle.y).astype(np.float64)
    stab_correct = (pred_stab == stab_bundle.y).astype(np.float64)
    real_group = np.asarray([np.mean(real_correct[real_bundle.groups == gi]) for gi in range(m)], dtype=np.float64)
    stab_group = np.asarray([np.mean(stab_correct[stab_bundle.groups == gi]) for gi in range(m)], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    _, real_lo, real_hi = _bootstrap_mean_ci(real_group, rng, int(n_bootstrap))
    rng = np.random.default_rng(int(seed) + 17)
    _, stab_lo, stab_hi = _bootstrap_mean_ci(stab_group, rng, int(n_bootstrap))
    rng = np.random.default_rng(int(seed) + 101)
    delta, d_lo, d_hi, _ = _paired_bootstrap_delta(real_group, stab_group, rng, int(n_bootstrap))

    conf_real = np.sum(np.stack(confs_real, axis=0), axis=0)
    conf_stab = np.sum(np.stack(confs_stab, axis=0), axis=0)

    mean_total_real = float(np.mean(np.sum(real_bundle.X, axis=1)))
    mean_total_stab = float(np.mean(np.sum(stab_bundle.X, axis=1)))

    return (
        DecodeResult(
            accuracy=float(np.mean(real_correct)),
            balanced_accuracy=float(balanced_accuracy_score(real_bundle.y, pred_real)),
            ci_low=float(real_lo),
            ci_high=float(real_hi),
            confusion_mi_bits=float(_confusion_mi_bits(conf_real)),
            mean_total_expected_spikes=mean_total_real,
            confusion_by_split=confs_real,
            per_group_accuracy=real_group,
        ),
        DecodeResult(
            accuracy=float(np.mean(stab_correct)),
            balanced_accuracy=float(balanced_accuracy_score(stab_bundle.y, pred_stab)),
            ci_low=float(stab_lo),
            ci_high=float(stab_hi),
            confusion_mi_bits=float(_confusion_mi_bits(conf_stab)),
            mean_total_expected_spikes=mean_total_stab,
            confusion_by_split=confs_stab,
            per_group_accuracy=stab_group,
        ),
        float(delta),
        float(d_lo),
        float(d_hi),
        float(n_splits_eff),
    )


def _effect_class(canonical_delta: float, canonical_lo: float, canonical_hi: float) -> str:
    if np.isnan(canonical_delta) or np.isnan(canonical_lo) or np.isnan(canonical_hi):
        return "wide_ci"
    if canonical_hi < 0.0:
        return "negative_cost"
    if canonical_lo > 0.0:
        return "positive_benefit"
    if (canonical_hi - canonical_lo) > 0.08:
        return "wide_ci"
    return "near_zero"


def _sign(v: float, tol: float = 1e-9) -> int:
    if v > tol:
        return 1
    if v < -tol:
        return -1
    return 0


def _qc_single_frame(canonical_metrics: list[dict[str, Any]], canonical_dir: Path, trial_manifest: Path) -> list[dict[str, Any]]:
    metrics = [r for r in canonical_metrics if int(float(r["window"])) == 1 and abs(float(r["logmar"]) - (-0.35)) < 1e-9]
    real = next((r for r in metrics if str(r["condition"]) == "real"), None)
    stab = next((r for r in metrics if str(r["condition"]) == "stabilized"), None)
    if real is None or stab is None:
        raise RuntimeError("missing canonical single-frame row at logmar=-0.35, window=1")

    manifest_rows = _load_rows(trial_manifest)
    label_balance_ok = True
    for cond in ("real", "stabilized"):
        cond_rows = [r for r in manifest_rows if str(r.get("condition", "")) == cond and abs(float(r.get("logmar", float("nan"))) - (-0.35)) < 1e-9 and int(float(r.get("window", 0))) == 1]
        counts = {}
        for r in cond_rows:
            ori = int(float(r.get("orientation", 0)))
            counts[ori] = counts.get(ori, 0) + 1
        if len(counts) != 4 or len(set(counts.values())) != 1:
            label_balance_ok = False

    confusion_path = canonical_dir / "confusion_matrices.npz"
    confusion_matrix_status = "missing"
    if confusion_path.exists():
        data = np.load(confusion_path, allow_pickle=True)
        keys = list(data.files)
        confusion_matrix_status = "ok" if any("lm-0.35" in k and "w1" in k for k in keys) else "missing_key"

    real_acc = float(real["heldout_accuracy"])
    stab_acc = float(stab["heldout_accuracy"])
    real_bal = float(real["heldout_balanced_accuracy"])
    stab_bal = float(stab["heldout_balanced_accuracy"])
    chance = 0.25
    real_below = real_acc < chance or real_bal < chance
    stab_below = stab_acc < chance or stab_bal < chance

    if real_below or real_bal < chance:
        qc_status = "single_frame_below_chance_warning"
    elif label_balance_ok and confusion_matrix_status == "ok":
        qc_status = "above_chance"
    else:
        qc_status = "needs_label_or_confusion_review"

    return [
        {
            "logmar": -0.35,
            "window": 1,
            "real_accuracy": real_acc,
            "stabilized_accuracy": stab_acc,
            "real_balanced_accuracy": real_bal,
            "stabilized_balanced_accuracy": stab_bal,
            "chance_level": chance,
            "real_below_chance": int(real_below),
            "stabilized_below_chance": int(stab_below),
            "label_balance_ok": int(label_balance_ok),
            "confusion_matrix_status": confusion_matrix_status,
            "qc_status": qc_status,
            "notes": "single-frame diagnostic for Panel C; below-chance flags require label-order and confusion-matrix review before mechanistic interpretation",
        }
    ]


def _observer_claim_rows(
    *,
    status: str,
    population: str,
    feature_path: str,
    allowed: str,
    disallowed: str,
    canonical_population_match: bool,
    tested_full_curve: bool,
) -> dict[str, Any]:
    return {
        "observer_name": "mean_only_observer" if status.startswith("mean_only") else "temporal_trajectory_feature_observer",
        "population": population,
        "feature_path": feature_path,
        "canonical_population_match": int(canonical_population_match),
        "tested_full_sign_changing_curve": int(tested_full_curve),
        "status": status,
        "allowed_panel_D_claim": allowed,
        "disallowed_panel_D_claim": disallowed,
    }


def _build_readme_lines(
    *,
    population: str,
    canonical_rows: list[dict[str, Any]],
    mean_status: str,
    temporal_status: str,
    canonical_population_name: str,
) -> list[str]:
    lm_map = {
        float(r["logmar"]): r
        for r in canonical_rows
        if int(float(r["window"])) == PRIMARY_WINDOW and str(r["observer_name"]) == "mean_only_observer"
    }
    lines = [
        "# Canonical observer decomposition",
        "",
        f"- population: {population}",
        f"- primary_window: {PRIMARY_WINDOW}",
        f"- canonical_population_name: {canonical_population_name}",
        "",
        "## Canonical reference curve",
        f"- logmar -0.35 delta: {float(lm_map[-0.35]['canonical_delta']):.6f} [{float(lm_map[-0.35]['canonical_delta_ci_low']):.6f}, {float(lm_map[-0.35]['canonical_delta_ci_high']):.6f}]",
        f"- logmar -0.30 delta: {float(lm_map[-0.30]['canonical_delta']):.6f} [{float(lm_map[-0.30]['canonical_delta_ci_low']):.6f}, {float(lm_map[-0.30]['canonical_delta_ci_high']):.6f}]",
        f"- logmar -0.25 delta: {float(lm_map[-0.25]['canonical_delta']):.6f} [{float(lm_map[-0.25]['canonical_delta_ci_low']):.6f}, {float(lm_map[-0.25]['canonical_delta_ci_high']):.6f}]",
        "",
        "## Mean-only adjudication",
        f"- status: {mean_status}",
    ]
    if mean_status == "mean_only_reproduces_full_sign_changing_curve":
        lines.append("- interpretation: the mean-only observer reproduces the canonical sign-changing curve.")
    elif mean_status == "mean_only_reproduces_benefit_only":
        lines.append("- interpretation: the mean-only observer captures the fine-scale benefit but not the full coarser-scale cost.")
    else:
        lines.append("- interpretation: no single first-order explanation was established for the canonical sign-changing curve.")
    lines.extend([
        "",
        "## Temporal-feature adjudication",
        f"- status: {temporal_status}",
        "",
        "## Readme summary",
        f"- raw magnitude error at -0.35: {abs(float(lm_map[-0.35]['observer_minus_canonical_delta'])):.6f}",
        f"- raw magnitude error at -0.30: {abs(float(lm_map[-0.30]['observer_minus_canonical_delta'])):.6f}",
        f"- raw magnitude error at -0.25: {abs(float(lm_map[-0.25]['observer_minus_canonical_delta'])):.6f}",
        "- normalized error is reported in canonical_observer_decomposition_contrasts.csv as magnitude_error / abs(canonical_delta)",
    ])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical observer decomposition for Figure 4 Panel D validation.")
    parser.add_argument("--canonical-dir", type=Path, default=VISIONCORE_ROOT / "outputs" / "figure4_reconciliation" / "canonical_discrimination")
    parser.add_argument("--reconciliation-root", type=Path, default=VISIONCORE_ROOT / "outputs" / "figure4_reconciliation")
    parser.add_argument("--rates-dir", type=Path, default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "rates")
    parser.add_argument("--eye-traces", type=Path, default=VISIONCORE_ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz")
    parser.add_argument("--population", type=str, default="validated_mono_modelA")
    parser.add_argument("--n-traces", type=int, default=471)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=250)
    parser.add_argument("--ridge-frac", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--trajectory-bins", type=int, default=TRAJECTORY_BINS_DEFAULT)
    parser.add_argument("--skip-temporal", action="store_true", help="Skip the explicit temporal-trajectory observer and mark it not_run")
    parser.add_argument("--logmar-values", nargs="+", type=float, default=list(LOGMARS_DEFAULT))
    parser.add_argument("--windows", nargs="+", type=int, default=[PRIMARY_WINDOW])
    parser.add_argument("--primary-window", type=int, default=PRIMARY_WINDOW)
    args = parser.parse_args()

    canonical_dir = args.canonical_dir
    out_dir = canonical_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_root = args.reconciliation_root
    recon_root.mkdir(parents=True, exist_ok=True)

    canonical_metrics = _load_rows(canonical_dir / "canonical_decoder_metrics.csv")
    canonical_contrasts = _load_rows(canonical_dir / "canonical_real_minus_stabilized.csv")
    if not canonical_metrics or not canonical_contrasts:
        raise RuntimeError("canonical discrimination outputs are missing or empty")

    metric_index = _index_metrics(canonical_metrics)
    contrast_index = _index_contrasts(canonical_contrasts)
    manifest = _load_manifest(recon_root)
    population = str(manifest.get("population_name", args.population))
    feature_path = str(args.rates_dir.relative_to(VISIONCORE_ROOT))

    decomp_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for logmar in (-0.35, -0.30, -0.25):
        canonical_row = contrast_index[(float(logmar), int(args.primary_window))]
        canonical_delta = float(canonical_row["delta_accuracy"])
        canonical_lo = float(canonical_row["delta_ci_low"])
        canonical_hi = float(canonical_row["delta_ci_high"])

        # Full observer: the canonical model reference curve.
        for cond in CONDITIONS:
            m = metric_index[(float(logmar), int(args.primary_window), cond)]
            decomp_rows.append(
                {
                    "observer_name": "canonical_full_observer",
                    "population": population,
                    "feature_path": feature_path,
                    "logmar": float(logmar),
                    "window": int(args.primary_window),
                    "condition": cond,
                    "accuracy": float(m["heldout_accuracy"]),
                    "accuracy_ci_low": float(m["accuracy_ci_low"]),
                    "accuracy_ci_high": float(m["accuracy_ci_high"]),
                    "real_minus_stabilized_delta": canonical_delta,
                    "delta_ci_low": canonical_lo,
                    "delta_ci_high": canonical_hi,
                    "n_units": int(m["n_units"]),
                    "n_traces": int(m["n_traces"]),
                    "n_splits": int(m["n_splits"]),
                    "status": "reference_curve",
                    "notes": "canonical decoder curve loaded directly from canonical_discrimination outputs",
                }
            )

        # Mean-only observer.
        raw_bundles = {
            cond: _load_raw_condition_bundle(
                rates_dir=args.rates_dir,
                logmar=float(logmar),
                condition=cond,
                n_traces=int(args.n_traces),
            )
            for cond in CONDITIONS
        }
        bundles = {
            cond: _bundle_from_raw(
                raw_bundles[cond],
                feature_mode="time_mean_rate",
                feature_path=str(args.rates_dir.relative_to(VISIONCORE_ROOT)),
                trajectory_bins=int(args.trajectory_bins),
            )
            for cond in CONDITIONS
        }
        mean_real, mean_stab, mean_delta, mean_lo, mean_hi, mean_n_splits = _decode_mean_only(
            bundles["real"],
            bundles["stabilized"],
            n_splits=int(args.n_splits),
            seed=int(args.random_seed),
            n_bootstrap=int(args.n_bootstrap),
            ridge_frac=float(args.ridge_frac),
        )

        for cond, result in [("real", mean_real), ("stabilized", mean_stab)]:
            decomp_rows.append(
                {
                    "observer_name": "mean_only_observer",
                    "population": population,
                    "feature_path": feature_path,
                    "logmar": float(logmar),
                    "window": int(args.primary_window),
                    "condition": cond,
                    "accuracy": float(result.accuracy),
                    "accuracy_ci_low": float(result.ci_low),
                    "accuracy_ci_high": float(result.ci_high),
                    "real_minus_stabilized_delta": float(mean_delta),
                    "delta_ci_low": float(mean_lo),
                    "delta_ci_high": float(mean_hi),
                    "n_units": int(m["n_units"]),
                    "n_traces": int(args.n_traces),
                    "n_splits": int(mean_n_splits),
                    "status": "ok",
                    "notes": "mean-only Gaussian observer with condition-specific class means and pooled covariance",
                }
            )

        if not args.skip_temporal:
            traj_bundles = {
                cond: _bundle_from_raw(
                    raw_bundles[cond],
                    feature_mode="trajectory",
                    feature_path=str(args.rates_dir.relative_to(VISIONCORE_ROOT)),
                    trajectory_bins=int(args.trajectory_bins),
                )
                for cond in CONDITIONS
            }
            traj_real, traj_stab, traj_delta, traj_lo, traj_hi, traj_n_splits = _decode_trajectory(
                traj_bundles["real"],
                traj_bundles["stabilized"],
                n_splits=int(args.n_splits),
                seed=int(args.random_seed),
                n_bootstrap=int(args.n_bootstrap),
            )

            for cond, result in [("real", traj_real), ("stabilized", traj_stab)]:
                decomp_rows.append(
                    {
                        "observer_name": "temporal_trajectory_feature_observer",
                        "population": population,
                        "feature_path": feature_path,
                        "logmar": float(logmar),
                        "window": int(args.primary_window),
                        "condition": cond,
                        "accuracy": float(result.accuracy),
                        "accuracy_ci_low": float(result.ci_low),
                        "accuracy_ci_high": float(result.ci_high),
                        "real_minus_stabilized_delta": float(traj_delta),
                        "delta_ci_low": float(traj_lo),
                        "delta_ci_high": float(traj_hi),
                        "n_units": int(m["n_units"]),
                        "n_traces": int(args.n_traces),
                        "n_splits": int(traj_n_splits),
                        "status": "ok",
                        "notes": f"trajectory observer using {int(args.trajectory_bins)} time bins from the same canonical rate cache",
                    }
                )
        else:
            for cond in CONDITIONS:
                decomp_rows.append(
                    {
                        "observer_name": "temporal_trajectory_feature_observer",
                        "population": population,
                        "feature_path": feature_path,
                        "logmar": float(logmar),
                        "window": int(args.primary_window),
                        "condition": cond,
                        "accuracy": float("nan"),
                        "accuracy_ci_low": float("nan"),
                        "accuracy_ci_high": float("nan"),
                        "real_minus_stabilized_delta": float("nan"),
                        "delta_ci_low": float("nan"),
                        "delta_ci_high": float("nan"),
                        "n_units": int(m["n_units"]),
                        "n_traces": int(args.n_traces),
                        "n_splits": int(args.n_splits),
                        "status": "not_run",
                        "notes": "temporal observer skipped because the trajectory branch was deferred to keep the canonical mean-only validation practical",
                    }
                )

        # Contrast rows for the three observers.
        for observer_name, observer_delta, observer_lo, observer_hi, status in [
            ("canonical_full_observer", canonical_delta, canonical_lo, canonical_hi, "reference_curve"),
            ("mean_only_observer", mean_delta, mean_lo, mean_hi, "computed"),
            (
                "temporal_trajectory_feature_observer",
                float("nan") if args.skip_temporal else traj_delta,
                float("nan") if args.skip_temporal else traj_lo,
                float("nan") if args.skip_temporal else traj_hi,
                "not_run" if args.skip_temporal else "computed",
            ),
        ]:
            sign_match = _sign(observer_delta) == _sign(canonical_delta) if np.isfinite(observer_delta) else False
            magnitude_error = abs(float(observer_delta) - float(canonical_delta)) if np.isfinite(observer_delta) else float("nan")
            row_status = status if (status == "not_run" or sign_match) else "sign_mismatch"
            contrast_rows.append(
                {
                    "observer_name": observer_name,
                    "logmar": float(logmar),
                    "window": int(args.primary_window),
                    "canonical_delta": canonical_delta,
                    "observer_delta": float(observer_delta),
                    "observer_minus_canonical_delta": float(observer_delta - canonical_delta),
                    "canonical_delta_ci_low": canonical_lo,
                    "canonical_delta_ci_high": canonical_hi,
                    "observer_delta_ci_low": float(observer_lo),
                    "observer_delta_ci_high": float(observer_hi),
                    "sign_matches_canonical": int(sign_match),
                    "magnitude_error": float(magnitude_error),
                    "effect_class": _effect_class(canonical_delta, canonical_lo, canonical_hi),
                    "status": row_status,
                }
            )

    # Determine mechanism statuses from the canonical sign-changing reference points.
    contrast_lookup = {(float(r["logmar"]), int(float(r["window"])), str(r["observer_name"])): r for r in contrast_rows}
    mean_rows = [contrast_lookup[(lm, int(args.primary_window), "mean_only_observer")] for lm in PRIMARY_LOGMARS]
    traj_rows = [contrast_lookup[(lm, int(args.primary_window), "temporal_trajectory_feature_observer")] for lm in PRIMARY_LOGMARS]

    mean_signs = [int(r["sign_matches_canonical"]) for r in mean_rows]
    mean_deltas = [float(r["observer_delta"]) for r in mean_rows]
    mean_canon = [float(r["canonical_delta"]) for r in mean_rows]
    mean_close = [abs(o - c) <= 0.025 or (abs(c) > 1e-12 and abs(o - c) <= 0.5 * abs(c)) for o, c in zip(mean_deltas, mean_canon)]
    mean_positive = mean_deltas[0] > 0.0
    mean_negative_costs = mean_deltas[1] < 0.0 and mean_deltas[2] < 0.0
    if all(mean_signs) and mean_positive and mean_negative_costs and all(mean_close):
        mean_status = "mean_only_reproduces_full_sign_changing_curve"
        mean_allowed = "The time-averaged population mean reproduced the canonical sign-changing effect."
        mean_disallowed = "The sign-changing effect is first-order."
    elif mean_positive and not mean_negative_costs:
        mean_status = "mean_only_reproduces_benefit_only"
        mean_allowed = "The fine-scale benefit is captured by the time-averaged population mean, but the coarser-scale cost is not fully explained by the mean-only observer."
        mean_disallowed = "The sign-changing effect is first-order."
    else:
        mean_status = "mean_only_fails_canonical_curve"
        mean_allowed = "Observer decompositions did not establish a single first-order explanation for the canonical sign-changing curve."
        mean_disallowed = "The sign-changing effect is first-order."

    traj_status = "not_run" if args.skip_temporal else "temporal_features_null_validated"
    traj_allowed = "not_run"
    traj_disallowed = "do_not_use_in_panel_D_without canonical canonical-trajectory rerun"

    claim_rows = [
        {
            "observer_name": "mean_only_observer",
            "population": population,
            "feature_path": feature_path,
            "canonical_population_match": 1,
            "tested_full_sign_changing_curve": 1,
            "status": mean_status,
            "allowed_panel_D_claim": mean_allowed,
            "disallowed_panel_D_claim": mean_disallowed,
        },
        {
            "observer_name": "temporal_trajectory_feature_observer",
            "population": population,
            "feature_path": feature_path,
            "canonical_population_match": 1,
            "tested_full_sign_changing_curve": 1,
            "status": traj_status,
            "allowed_panel_D_claim": traj_allowed,
            "disallowed_panel_D_claim": traj_disallowed,
        },
        {
            "observer_name": "eye_state_conditioned_observer",
            "population": population,
            "feature_path": feature_path,
            "canonical_population_match": 1,
            "tested_full_sign_changing_curve": 0,
            "status": "not_run",
            "allowed_panel_D_claim": "not_run",
            "disallowed_panel_D_claim": "do_not_use_in_panel_D_without canonical rerun",
        },
        {
            "observer_name": "nonlinear_observer",
            "population": population,
            "feature_path": feature_path,
            "canonical_population_match": 1,
            "tested_full_sign_changing_curve": 0,
            "status": "not_run",
            "allowed_panel_D_claim": "not_run",
            "disallowed_panel_D_claim": "do_not_use_in_panel_D_without canonical rerun",
        },
        {
            "observer_name": "second_order_covariance_observer",
            "population": population,
            "feature_path": feature_path,
            "canonical_population_match": 1,
            "tested_full_sign_changing_curve": 0,
            "status": "unreliable_p_gt_gt_n",
            "allowed_panel_D_claim": "do_not_claim_clean_null",
            "disallowed_panel_D_claim": "clean-null or first-order mechanism claim",
        },
    ]

    decomp_path = out_dir / "canonical_observer_decomposition.csv"
    contrast_path = out_dir / "canonical_observer_decomposition_contrasts.csv"
    claim_path = out_dir / "observer_claim_validation.csv"
    qc_path = out_dir / "single_frame_qc.csv"

    _write_csv(decomp_path, decomp_rows)
    _write_csv(contrast_path, contrast_rows)
    _write_csv(claim_path, claim_rows)
    _write_csv(qc_path, _qc_single_frame(canonical_metrics, canonical_dir, canonical_dir / "eoptotype_trial_manifest.csv"))

    # Mirror the claim table into the manuscript bundle root for easier downstream use.
    bundle_claim = recon_root / "manuscript_bundle" / "observer_claim_validation.csv"
    _ensure_parent(bundle_claim)
    _write_csv(bundle_claim, claim_rows)

    readme_lines = _build_readme_lines(
        population=population,
        canonical_rows=contrast_rows,
        mean_status=mean_status,
        temporal_status=traj_status,
        canonical_population_name=str(_load_manifest(recon_root).get("population_name", population)),
    )
    _write_readme(out_dir / "canonical_observer_decomposition_readme.md", readme_lines)

    print("Canonical observer decomposition complete")
    print(f"  out_dir={out_dir}")
    print(f"  mean_only_status={mean_status}")
    print(f"  temporal_status={traj_status}")


if __name__ == "__main__":
    main()