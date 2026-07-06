"""Conditional model for local empirical-vs-matched BackImage pairing effects.

The input is a completed local feature-space screen. The outcome is the
per-window contrast:

    actual paired motion gain - matched-unpaired motion gain

For the current cache this outcome is the per-window decoder-gain contrast
(`incremental_gain_delta_neg_mse`), not a per-window bit estimate. The script
tests whether that contrast is conditionally organized by image coherence,
drift geometry, motion scale, and edge/drift alignment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


DEFAULT_SCREEN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1"
    / "local_feature_space_screen_v1"
)

BASE_PREDICTORS = (
    "image_coherence_z",
    "drift_anisotropy_z",
    "log_rms_z",
    "edge_drift_alignment_z",
)
INTERACTION_PREDICTORS = (
    "image_coherence_x_alignment",
    "image_coherence_x_log_rms",
    "drift_anisotropy_x_alignment",
    "image_coherence_x_drift_anisotropy",
)


def _parse_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return (arr - mean) / sd


def _edge_drift_alignment(edge_axis_deg: np.ndarray, drift_axis_deg: np.ndarray) -> np.ndarray:
    return np.cos(np.radians(2.0 * (np.asarray(edge_axis_deg, dtype=np.float64) - np.asarray(drift_axis_deg, dtype=np.float64))))


def _source_trial_groups(df: pd.DataFrame) -> np.ndarray:
    return (df["session"].astype(str) + "::trial_" + df["trial_idx"].astype(int).astype(str)).to_numpy()


def _session_demean(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sessions = np.asarray(sessions)
    out = arr.copy()
    if out.ndim == 1:
        for session in np.unique(sessions):
            idx = sessions == session
            out[idx] = out[idx] - np.nanmean(out[idx])
        return out
    for session in np.unique(sessions):
        idx = sessions == session
        out[idx, :] = out[idx, :] - np.nanmean(out[idx, :], axis=0, keepdims=True)
    return out


def _add_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = sorted(
        {
            "image_orientation_coherence",
            "drift_anisotropy",
            "image_edge_axis_deg",
            "drift_orientation_deg",
            "actual_observed_rms_deg",
        }.difference(out.columns)
    )
    if missing:
        raise ValueError(f"Conditional model requires missing columns: {missing}")
    out["edge_drift_alignment"] = _edge_drift_alignment(out["image_edge_axis_deg"], out["drift_orientation_deg"])
    out["log_actual_observed_rms_deg"] = np.log(np.clip(out["actual_observed_rms_deg"].to_numpy(dtype=np.float64), 1e-9, np.inf))
    out["image_coherence_z"] = _zscore(out["image_orientation_coherence"].to_numpy(dtype=np.float64))
    out["drift_anisotropy_z"] = _zscore(out["drift_anisotropy"].to_numpy(dtype=np.float64))
    out["log_rms_z"] = _zscore(out["log_actual_observed_rms_deg"].to_numpy(dtype=np.float64))
    out["edge_drift_alignment_z"] = _zscore(out["edge_drift_alignment"].to_numpy(dtype=np.float64))
    out["image_coherence_x_alignment"] = _zscore(
        out["image_coherence_z"].to_numpy(dtype=np.float64) * out["edge_drift_alignment_z"].to_numpy(dtype=np.float64)
    )
    out["image_coherence_x_log_rms"] = _zscore(
        out["image_coherence_z"].to_numpy(dtype=np.float64) * out["log_rms_z"].to_numpy(dtype=np.float64)
    )
    out["drift_anisotropy_x_alignment"] = _zscore(
        out["drift_anisotropy_z"].to_numpy(dtype=np.float64) * out["edge_drift_alignment_z"].to_numpy(dtype=np.float64)
    )
    out["image_coherence_x_drift_anisotropy"] = _zscore(
        out["image_coherence_z"].to_numpy(dtype=np.float64) * out["drift_anisotropy_z"].to_numpy(dtype=np.float64)
    )
    return out


def _fit_ols(X: np.ndarray, y: np.ndarray, *, add_intercept: bool) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if add_intercept:
        X_fit = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])
    else:
        X_fit = X
    beta, *_ = np.linalg.lstsq(X_fit, y, rcond=None)
    pred = X_fit @ beta
    resid = y - pred
    mse = float(np.nanmean(resid * resid))
    return beta, pred, mse


def _fit_session_centered(df: pd.DataFrame, predictors: list[str], outcome_col: str) -> dict[str, Any]:
    sessions = df["session"].to_numpy()
    X = df[predictors].to_numpy(dtype=np.float64)
    y = df[outcome_col].to_numpy(dtype=np.float64)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    sessions = sessions[ok]
    Xc = _session_demean(X, sessions)
    yc = _session_demean(y, sessions)
    beta, pred, mse = _fit_ols(Xc, yc, add_intercept=False)
    null_mse = float(np.nanmean(yc * yc))
    ss_res = float(np.nansum((yc - pred) ** 2))
    ss_tot = float(np.nansum((yc - np.nanmean(yc)) ** 2))
    return {
        "beta": beta,
        "predictors": predictors,
        "n": int(y.size),
        "n_sessions": int(np.unique(sessions).size),
        "mse": mse,
        "null_mse": null_mse,
        "delta_neg_mse_vs_session_mean": float(null_mse - mse),
        "r2_within_session": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "predicted_centered": pred,
        "observed_centered": yc,
    }


def _bootstrap_coefficients(
    df: pd.DataFrame,
    predictors: list[str],
    outcome_col: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    sessions = np.asarray(sorted(df["session"].astype(str).unique()))
    coefs = np.full((int(n_bootstrap), len(predictors)), np.nan, dtype=np.float64)
    if int(n_bootstrap) <= 0 or sessions.size < 2:
        return coefs[:0]
    by_session = {session: df[df["session"].astype(str) == session] for session in sessions}
    for b in range(int(n_bootstrap)):
        parts = []
        for draw_idx, session in enumerate(rng.choice(sessions, size=sessions.size, replace=True)):
            part = by_session[str(session)].copy()
            part["session"] = f"{session}::boot{draw_idx}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        try:
            coefs[b, :] = _fit_session_centered(sample, predictors, outcome_col)["beta"]
        except Exception:
            continue
    return coefs


def _split_groups(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if unique.size >= 2:
        n = min(int(n_splits), unique.size)
        if n >= 2:
            return list(GroupKFold(n_splits=n).split(np.zeros(groups.size), groups=groups))
    n = min(int(n_splits), groups.size)
    if n < 2:
        return [(np.arange(groups.size), np.arange(groups.size))]
    return list(KFold(n_splits=n, shuffle=True, random_state=int(seed)).split(np.zeros(groups.size)))


def _train_standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0, keepdims=True)
    sd = np.nanstd(train, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (train - mean) / sd, (test - mean) / sd


def _cross_validated_model(
    df: pd.DataFrame,
    predictors: list[str],
    outcome_col: str,
    *,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    X = df[predictors].to_numpy(dtype=np.float64)
    y = df[outcome_col].to_numpy(dtype=np.float64)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & pd.notna(groups)
    X = X[ok]
    y = y[ok]
    groups = np.asarray(groups)[ok]
    pred = np.full(y.shape, np.nan, dtype=np.float64)
    pred_null = np.full(y.shape, np.nan, dtype=np.float64)
    for train_idx, test_idx in _split_groups(groups, int(n_folds), int(seed)):
        X_train, X_test = _train_standardize(X[train_idx], X[test_idx])
        y_train = y[train_idx]
        beta, _train_pred, _mse = _fit_ols(X_train, y_train, add_intercept=True)
        pred[test_idx] = np.column_stack([np.ones(test_idx.size), X_test]) @ beta
        pred_null[test_idx] = float(np.nanmean(y_train))
    mse = float(np.nanmean((y - pred) ** 2))
    null_mse = float(np.nanmean((y - pred_null) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    ss_res = float(np.nansum((y - pred) ** 2))
    return {
        "cv_mse": mse,
        "cv_null_mse": null_mse,
        "cv_delta_neg_mse_vs_null": float(null_mse - mse),
        "cv_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "n_cv_rows": int(y.size),
        "n_cv_groups": int(np.unique(groups).size),
    }


def _session_bootstrap_mean(values: np.ndarray, sessions: np.ndarray, *, n_bootstrap: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    sessions = np.asarray(sessions)
    ok = np.isfinite(values) & pd.notna(sessions)
    values = values[ok]
    sessions = sessions[ok]
    if values.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0, "n_sessions": 0}
    session_values = pd.DataFrame({"session": sessions, "value": values}).groupby("session")["value"].mean().to_numpy()
    mean = float(np.nanmean(session_values))
    if int(n_bootstrap) <= 0 or session_values.size <= 1:
        lo = hi = mean
    else:
        rng = np.random.default_rng(int(seed))
        boot = np.asarray(
            [np.nanmean(rng.choice(session_values, size=session_values.size, replace=True)) for _ in range(int(n_bootstrap))],
            dtype=np.float64,
        )
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "mean": mean,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n": int(values.size),
        "n_sessions": int(session_values.size),
    }


def _target_selection(screen_dir: Path, targets_arg: str, top_n: int) -> pd.DataFrame:
    best_path = screen_dir / "screen_best_empirical_vs_matched_by_target.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing target selection file: {best_path}")
    best = pd.read_csv(best_path)
    if best.empty:
        raise ValueError(f"Target selection file is empty: {best_path}")
    requested = _parse_list(targets_arg)
    if requested and "top" not in requested and "all" not in requested:
        selected = best[best["latent"].astype(str).isin(requested)].copy()
    elif requested and "all" in requested:
        selected = best.copy()
    else:
        selected = best.sort_values("screen_score", ascending=False).head(int(top_n)).copy()
    if selected.empty:
        raise ValueError(f"No target rows selected from {best_path} with --targets={targets_arg!r}")
    return selected


def _load_contrast_rows(args: argparse.Namespace, selected: pd.DataFrame) -> pd.DataFrame:
    screen_dir = Path(args.screen_dir)
    path = screen_dir / "incremental_decode" / "incremental_gain_contrasts_by_window.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-window contrast file: {path}")
    df = pd.read_csv(path)
    df = df[
        (df["motion_summary"].astype(str) == str(args.motion_summary))
        & (df["lhs_family"].astype(str) == str(args.lhs_family))
        & (df["rhs_family"].astype(str) == str(args.rhs_family))
        & (df["scale_id"].astype(str) == str(args.scale_id))
    ].copy()
    keys = selected[["latent", "k"]].drop_duplicates()
    df = df.merge(keys, on=["latent", "k"], how="inner")
    if df.empty:
        raise ValueError("No per-window contrast rows remain after filtering by selected latent/k rows")
    return _add_predictors(df)


def _coefficient_rows(
    df: pd.DataFrame,
    *,
    predictors: list[str],
    model_name: str,
    outcome_col: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    fits: dict[str, dict[str, Any]] = {}
    for latent, work in df.groupby("latent", sort=False):
        work = work.copy()
        fit = _fit_session_centered(work, predictors, outcome_col)
        boots = _bootstrap_coefficients(
            work,
            predictors,
            outcome_col,
            n_bootstrap=int(n_bootstrap),
            seed=int(seed) + sum(ord(ch) for ch in f"{latent}:{model_name}") % 100000,
        )
        fits[str(latent)] = fit
        for idx, predictor in enumerate(predictors):
            coef = float(fit["beta"][idx])
            if boots.size:
                boot_values = boots[:, idx]
                lo, hi = np.nanpercentile(boot_values, [2.5, 97.5])
                p_sign = 2.0 * min(
                    float(np.nanmean(boot_values <= 0.0)),
                    float(np.nanmean(boot_values >= 0.0)),
                )
            else:
                lo = hi = coef
                p_sign = float("nan")
            rows.append(
                {
                    "latent": str(latent),
                    "model": model_name,
                    "predictor": predictor,
                    "coefficient": coef,
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "bootstrap_sign_p": float(min(1.0, p_sign)) if np.isfinite(p_sign) else float("nan"),
                    "n": int(fit["n"]),
                    "n_sessions": int(fit["n_sessions"]),
                    "r2_within_session": float(fit["r2_within_session"]),
                    "delta_neg_mse_vs_session_mean": float(fit["delta_neg_mse_vs_session_mean"]),
                }
            )
    return rows, fits


def _cv_rows(
    df: pd.DataFrame,
    *,
    predictor_sets: dict[str, list[str]],
    outcome_col: str,
    n_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for latent, work in df.groupby("latent", sort=False):
        groups = _source_trial_groups(work)
        for model_name, predictors in predictor_sets.items():
            cv = _cross_validated_model(
                work,
                predictors,
                outcome_col,
                groups=groups,
                n_folds=int(n_folds),
                seed=int(seed),
            )
            rows.append({"latent": str(latent), "model": model_name, **cv})
    return rows


def _binned_rows(
    df: pd.DataFrame,
    fits: dict[str, dict[str, Any]],
    *,
    predictors: list[str],
    outcome_col: str,
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for latent, work in df.groupby("latent", sort=False):
        fit = fits[str(latent)]
        beta = np.asarray(fit["beta"], dtype=np.float64)
        score = work[predictors].to_numpy(dtype=np.float64) @ beta
        work = work.copy()
        work["conditional_score"] = score
        ranks = pd.Series(score).rank(method="first")
        bins = pd.qcut(ranks, q=int(min(n_bins, work.shape[0])), labels=False, duplicates="drop")
        work["conditional_score_bin"] = np.asarray(bins, dtype=int)
        for bin_id, part in work.groupby("conditional_score_bin"):
            stats = _session_bootstrap_mean(
                part[outcome_col].to_numpy(dtype=np.float64),
                part["session"].to_numpy(),
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + int(bin_id),
            )
            rows.append(
                {
                    "latent": str(latent),
                    "bin": int(bin_id),
                    "score_mean": float(np.nanmean(part["conditional_score"])),
                    "score_min": float(np.nanmin(part["conditional_score"])),
                    "score_max": float(np.nanmax(part["conditional_score"])),
                    "effect_mean": stats["mean"],
                    "effect_ci95_low": stats["ci95_low"],
                    "effect_ci95_high": stats["ci95_high"],
                    "n": stats["n"],
                    "n_sessions": stats["n_sessions"],
                }
            )
    return rows


def _predictor_bin_rows(
    df: pd.DataFrame,
    *,
    predictor: str,
    outcome_col: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for latent, work in df.groupby("latent", sort=False):
        values = work[predictor].to_numpy(dtype=np.float64)
        if predictor == "edge_drift_alignment":
            labels = np.where(values >= 0.0, "aligned", "orthogonal")
        else:
            threshold = float(np.nanmedian(values))
            labels = np.where(values >= threshold, "high", "low")
        work = work.copy()
        work["bin_label"] = labels
        for label, part in work.groupby("bin_label"):
            stats = _session_bootstrap_mean(
                part[outcome_col].to_numpy(dtype=np.float64),
                part["session"].to_numpy(),
                n_bootstrap=int(n_bootstrap),
                seed=int(seed) + len(rows),
            )
            rows.append(
                {
                    "latent": str(latent),
                    "predictor": predictor,
                    "bin": str(label),
                    "effect_mean": stats["mean"],
                    "effect_ci95_low": stats["ci95_low"],
                    "effect_ci95_high": stats["ci95_high"],
                    "n": stats["n"],
                    "n_sessions": stats["n_sessions"],
                }
            )
    return rows


def _plot_coefficients(rows: list[dict[str, Any]], out_dir: Path, *, model_name: str) -> None:
    df = pd.DataFrame(rows)
    df = df[df["model"] == model_name].copy()
    if df.empty:
        return
    predictors = list(dict.fromkeys(df["predictor"].astype(str)))
    latents = list(dict.fromkeys(df["latent"].astype(str)))
    fig, axes = plt.subplots(
        len(latents),
        1,
        figsize=(9.5, max(2.4, 1.9 * len(latents))),
        sharex=True,
        constrained_layout=True,
    )
    if len(latents) == 1:
        axes = [axes]
    for ax, latent in zip(axes, latents, strict=True):
        part = df[df["latent"] == latent].set_index("predictor").loc[predictors].reset_index()
        y = np.arange(part.shape[0])
        x = part["coefficient"].to_numpy(dtype=float)
        lo = part["ci95_low"].to_numpy(dtype=float)
        hi = part["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", color="#4C78A8", ecolor="0.35", capsize=2)
        ax.axvline(0.0, color="0.2", linewidth=0.8)
        ax.set_yticks(y, part["predictor"].astype(str))
        ax.set_title(str(latent), loc="left", fontsize=10)
        ax.grid(axis="x", color="0.9")
    axes[-1].set_xlabel("within-session coefficient on empirical-vs-matched decoder-gain contrast")
    fig.savefig(out_dir / f"conditional_{model_name}_coefficients.png", dpi=180)
    plt.close(fig)


def _plot_bins(rows: list[dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    latents = list(dict.fromkeys(df["latent"].astype(str)))
    fig, axes = plt.subplots(
        len(latents),
        1,
        figsize=(8.5, max(2.4, 1.9 * len(latents))),
        sharex=True,
        constrained_layout=True,
    )
    if len(latents) == 1:
        axes = [axes]
    for ax, latent in zip(axes, latents, strict=True):
        part = df[df["latent"] == latent].sort_values("bin")
        x = part["bin"].to_numpy(dtype=int)
        y = part["effect_mean"].to_numpy(dtype=float)
        lo = part["effect_ci95_low"].to_numpy(dtype=float)
        hi = part["effect_ci95_high"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="o-", color="#4C78A8", capsize=2)
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.set_title(str(latent), loc="left", fontsize=10)
        ax.set_ylabel("effect")
        ax.grid(axis="y", color="0.9")
    axes[-1].set_xlabel("conditional-score quartile")
    fig.savefig(out_dir / "conditional_score_bins.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--targets", default="top")
    parser.add_argument("--top-n-targets", type=int, default=4)
    parser.add_argument("--motion-summary", default="delta_mean")
    parser.add_argument("--lhs-family", default="actual_paired_empirical")
    parser.add_argument("--rhs-family", default="matched_unpaired_empirical")
    parser.add_argument("--scale-id", default="rel_1x")
    parser.add_argument("--outcome-col", default="incremental_gain_delta_neg_mse")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    screen_dir = Path(args.screen_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else screen_dir / "conditional_local_pairing_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _target_selection(screen_dir, str(args.targets), int(args.top_n_targets))
    selected.to_csv(out_dir / "target_selection.csv", index=False)
    df = _load_contrast_rows(args, selected)
    df.to_csv(out_dir / "conditional_model_input_rows.csv", index=False)

    predictor_sets = {
        "main_effects": list(BASE_PREDICTORS),
        "main_plus_interactions": list(BASE_PREDICTORS + INTERACTION_PREDICTORS),
    }
    all_coef_rows: list[dict[str, Any]] = []
    fits_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, predictors in predictor_sets.items():
        rows, fits = _coefficient_rows(
            df,
            predictors=predictors,
            model_name=model_name,
            outcome_col=str(args.outcome_col),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
        )
        all_coef_rows.extend(rows)
        fits_by_model[model_name] = fits
    _write_csv(out_dir / "conditional_coefficients.csv", all_coef_rows)

    cv = _cv_rows(
        df,
        predictor_sets=predictor_sets,
        outcome_col=str(args.outcome_col),
        n_folds=int(args.n_folds),
        seed=int(args.seed),
    )
    _write_csv(out_dir / "conditional_cv_summary.csv", cv)

    binned = _binned_rows(
        df,
        fits_by_model["main_plus_interactions"],
        predictors=predictor_sets["main_plus_interactions"],
        outcome_col=str(args.outcome_col),
        n_bins=int(args.n_bins),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    _write_csv(out_dir / "conditional_score_bins.csv", binned)
    predictor_bins: list[dict[str, Any]] = []
    for predictor in (
        "image_orientation_coherence",
        "drift_anisotropy",
        "actual_observed_rms_deg",
        "edge_drift_alignment",
    ):
        predictor_bins.extend(
            _predictor_bin_rows(
                df,
                predictor=predictor,
                outcome_col=str(args.outcome_col),
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed),
            )
        )
    _write_csv(out_dir / "conditional_single_predictor_bins.csv", predictor_bins)

    _plot_coefficients(all_coef_rows, out_dir, model_name="main_plus_interactions")
    _plot_bins(binned, out_dir)
    _write_json(
        out_dir / "run_metadata.json",
        {
            "screen_dir": screen_dir,
            "per_window_contrast_file": screen_dir / "incremental_decode" / "incremental_gain_contrasts_by_window.csv",
            "target_selection": out_dir / "target_selection.csv",
            "outcome_col": str(args.outcome_col),
            "outcome_interpretation": (
                "per-window empirical-minus-matched decoder-gain contrast; "
                "this is a local information proxy, not a per-window bit estimate"
            ),
            "filters": {
                "motion_summary": str(args.motion_summary),
                "lhs_family": str(args.lhs_family),
                "rhs_family": str(args.rhs_family),
                "scale_id": str(args.scale_id),
            },
            "predictor_sets": predictor_sets,
            "coefficient_model": "OLS after demeaning outcome and predictors within session",
            "coefficient_ci": f"session bootstrap, n={int(args.n_bootstrap)}",
            "predictive_check": f"source-trial grouped {int(args.n_folds)}-fold CV without session fixed effects",
            "n_rows": int(df.shape[0]),
            "n_images_per_target": int(df.groupby('latent')["image_index"].nunique().median()),
            "n_sessions": int(df["session"].nunique()),
            "n_source_trials": int(pd.Series(_source_trial_groups(df)).nunique()),
            "seed": int(args.seed),
        },
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Conditional Local Pairing Model",
                "",
                "This posthoc tests whether local empirical-vs-matched pairing effects are conditionally organized by image and drift structure.",
                "",
                "Outcome:",
                f"- `{args.outcome_col}` from `incremental_gain_contrasts_by_window.csv`.",
                "- This is a per-window decoder-gain contrast, not a per-window bit estimate.",
                "",
                "Predictors:",
                "- image orientation coherence",
                "- drift anisotropy",
                "- log observed drift RMS",
                "- edge/drift alignment, `cos(2 * (edge_axis - drift_axis))`",
                "- pre-specified two-way interactions with coherence/alignment/RMS/anisotropy",
                "",
                "Primary files:",
                "- `conditional_coefficients.csv`",
                "- `conditional_cv_summary.csv`",
                "- `conditional_score_bins.csv`",
                "- `conditional_single_predictor_bins.csv`",
                "- `conditional_main_plus_interactions_coefficients.png`",
                "- `conditional_score_bins.png`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote conditional local pairing model to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
