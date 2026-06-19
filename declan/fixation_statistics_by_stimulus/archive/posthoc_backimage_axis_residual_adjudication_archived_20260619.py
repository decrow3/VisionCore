"""Archived cache-only residual adjudication for BackImage drift-axis alignment.

Archived 2026-06-19 because it is partially redundant with existing
BackImage residual-prediction summaries and because the preservation-audit
windows overlap sparsely with the newer observer/feature-posterior caches.

Do not treat this as a canonical analysis. The intended path forward is to
extend the existing residual-prediction/posthoc summary code instead of
maintaining this parallel wide-table adjudicator.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
DEFAULT_DRIFT = ROOT / "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
DEFAULT_STABILITY = ROOT / "backimage_twin_stability_metric_audit/twin_stability_metric_by_window.csv"
DEFAULT_OUT_DIR = ROOT / "backimage_axis_residual_adjudication"

DEFAULT_OBSERVER_CACHE_SPECS = (
    "observer_matched_static_n64="
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1/observer_trials.csv",
    "observer_hard_negative_n64="
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n64_c4_k16_v1/observer_trials.csv",
    "observer_hard_negative_n128_multiscale="
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/observer_trials.csv",
)

DEFAULT_FEATURE_CACHE_SPECS = (
    "feature_matched_static_n64="
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2/"
    "feature_posterior_trials.csv",
    "feature_hard_negative_n64="
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_hard_negative_feature_posterior_gabor_pyramid_k4_8_uncertainty_v1/"
    "feature_posterior_trials.csv",
    "feature_hard_negative_n128_multiscale="
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_uncertainty_v1/"
    "feature_posterior_trials.csv",
)

BASELINE_CONTROLS = [
    "image_orientation_coherence",
    "drift_anisotropy",
    "local_contrast",
    "edge_strength",
    "image_edge_density",
    "image_spectrum_anisotropy",
    "pixel_relative_advantage",
    "pixel_stability_advantage",
]

OBSERVER_METRICS = [
    "joint_correct",
    "joint_true_margin",
    "joint_true_score",
    "joint_minus_zero_true_score",
    "known_minus_zero_true_score",
    "best_single_tau_minus_joint_true_score",
    "known_minus_joint_pose_cost",
    "posterior_concentration",
]

FEATURE_METRICS = [
    "candidate_posterior_true_mass",
    "posterior_concentration",
    "score_true_margin",
    "score_true_value",
    "feature_neg_mse",
    "feature_cosine",
    "joint_minus_zero_feature_gain",
    "known_minus_joint_pose_cost",
    "motion_delta_minus_zero_feature_gain",
]


def _parse_named_paths(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" not in str(item):
            raise ValueError(f"Expected NAME=PATH cache spec, got {item!r}")
        name, path = str(item).split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty cache name in {item!r}")
        out.append((name, Path(path)))
    return out


def _slug(parts: list[Any]) -> str:
    text = "_".join(str(part) for part in parts if str(part) != "")
    text = text.replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=bool)
    ok = np.ones(np.asarray(arrays[0]).shape[0], dtype=bool)
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            ok &= np.isfinite(arr)
        else:
            ok &= np.all(np.isfinite(arr), axis=1)
    return ok


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(values))
    sd = float(np.nanstd(values))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / sd


def _demean_by_session(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return (series - series.groupby(pd.Series(sessions)).transform("mean")).to_numpy(dtype=np.float64)


def _fit_ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    ok = _finite_mask(y, X)
    y = y[ok]
    X = X[ok]
    if y.size <= X.shape[1] + 2:
        return {
            "coef": np.full(X.shape[1], np.nan),
            "r2": float("nan"),
            "pred": np.full(y.size, np.nan),
            "resid": np.full(y.size, np.nan),
            "ok": ok,
            "n": int(y.size),
        }
    yz = _zscore(y)
    Xz = np.column_stack([_zscore(X[:, j]) for j in range(X.shape[1])]) if X.shape[1] else np.empty((y.size, 0))
    design = np.column_stack([np.ones(y.size), Xz])
    beta, *_ = np.linalg.lstsq(design, yz, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((yz - pred) ** 2))
    ss_tot = float(np.sum((yz - np.mean(yz)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"coef": beta[1:], "r2": float(r2), "pred": pred, "resid": yz - pred, "ok": ok, "n": int(y.size)}


def _control_matrix(df: pd.DataFrame, controls: list[str]) -> np.ndarray:
    cols = []
    sessions = df["session_id"].to_numpy()
    for col in controls:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=np.float64)
        nonnull = np.isfinite(vals)
        if np.count_nonzero(nonnull) < max(8, min(16, df.shape[0] // 4)):
            continue
        if len(pd.unique(df.loc[nonnull, "session_id"])) < 2:
            continue
        vals = _demean_by_session(vals, sessions)
        if np.nanstd(vals) <= 1e-12:
            continue
        cols.append(np.nan_to_num(vals, nan=0.0))
    if not cols:
        return np.empty((df.shape[0], 0), dtype=np.float64)
    return np.column_stack(cols).astype(np.float64)


def _fit_block_model(
    df: pd.DataFrame,
    y_col: str,
    controls: list[str],
    predictors: list[str],
    *,
    max_components: int = 8,
    require_multi_session_predictor: bool = True,
    min_predictor_obs: int | None = None,
) -> dict[str, Any]:
    needed = ["session_id", y_col]
    work = df.dropna(subset=needed).copy()
    if work.empty:
        return {"coef": {}, "r2": float("nan"), "n_windows": 0, "n_sessions": 0, "pred": np.array([]), "resid": np.array([]), "predictors_used": []}
    sessions = work["session_id"].to_numpy()
    y = _demean_by_session(work[y_col].to_numpy(dtype=np.float64), sessions)
    X_control = _control_matrix(work, controls)
    pred_cols = []
    predictors_used = []
    for col in predictors:
        if col not in work.columns:
            continue
        values = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=np.float64)
        nonnull = np.isfinite(values)
        min_obs = int(min_predictor_obs) if min_predictor_obs is not None else max(8, len(controls) + 4)
        if np.count_nonzero(nonnull) < min_obs:
            continue
        if require_multi_session_predictor and len(pd.unique(work.loc[nonnull, "session_id"])) < 2:
            continue
        demeaned = _demean_by_session(values, sessions)
        if np.nanstd(demeaned) <= 1e-12:
            continue
        pred_cols.append(np.nan_to_num(demeaned, nan=0.0))
        predictors_used.append(col)
    if len(pred_cols) > 1:
        X_raw = np.column_stack([_zscore(col) for col in pred_cols])
        ok_rows = np.any(np.abs(X_raw) > 1e-12, axis=1)
        work = work.loc[ok_rows].copy()
        sessions = work["session_id"].to_numpy()
        y = _demean_by_session(work[y_col].to_numpy(dtype=np.float64), sessions)
        X_control = _control_matrix(work, controls)
        X_raw = X_raw[ok_rows]
        u, s, _ = np.linalg.svd(X_raw, full_matrices=False)
        n_components = int(min(max_components, u.shape[1], max(1, work.shape[0] - X_control.shape[1] - 2)))
        X_pred = u[:, :n_components] * s[:n_components]
        X = np.column_stack([X_control, X_pred])
    elif len(pred_cols) == 1:
        ok_rows = np.abs(pred_cols[0]) > 1e-12
        work = work.loc[ok_rows].copy()
        sessions = work["session_id"].to_numpy()
        y = _demean_by_session(work[y_col].to_numpy(dtype=np.float64), sessions)
        X_control = _control_matrix(work, controls)
        X = np.column_stack([X_control, pred_cols[0][ok_rows]])
    else:
        X = X_control
    fit = _fit_ols(y, X)
    coefs = {}
    if len(predictors_used) == 1 and np.asarray(fit["coef"]).size >= 1:
        for col, val in zip(predictors_used, np.asarray(fit["coef"])[-1:]):
            coefs[col] = float(val)
    return {
        "coef": coefs,
        "r2": float(fit["r2"]),
        "n_windows": int(fit["n"]),
        "n_sessions": int(work["session_id"].nunique()),
        "pred": fit["pred"],
        "resid": fit["resid"],
        "index": work.index.to_numpy(),
        "predictors_used": predictors_used,
    }


def _extract_source_row(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    parsed = text.str.extract(r"source_row:(\d+)", expand=False)
    return pd.to_numeric(parsed, errors="coerce")


def _load_base_table(drift_path: Path, stability_path: Path) -> pd.DataFrame:
    drift = pd.read_csv(drift_path).reset_index(names="source_window_id")
    drift["session_id"] = drift["session"].astype(str)
    drift["image_id"] = drift["trial_idx"].astype(int)
    if "drift_anisotropy" not in drift.columns:
        drift["drift_anisotropy"] = drift.get("anisotropy", np.nan)
    if "local_contrast" not in drift.columns:
        drift["local_contrast"] = drift.get("image_patch_rms_contrast", np.nan)
    if "edge_strength" not in drift.columns:
        drift["edge_strength"] = drift.get("image_gradient_energy", np.nan)
    stability = pd.read_csv(stability_path)
    stability["source_window_id"] = pd.to_numeric(stability["source_window_id"], errors="coerce").astype("Int64")
    keep = [
        "source_window_id",
        "window_row",
        "window_id",
        "drift_edge_align_signed",
        "alignment_weight",
        "pixel_parallel_cost",
        "pixel_orthogonal_cost",
        "pixel_stability_advantage",
        "pixel_relative_advantage",
        "twin_parallel_cost",
        "twin_orthogonal_cost",
        "twin_stability_advantage",
        "twin_relative_advantage",
        "raw_mse_stability_advantage",
        "raw_mse_relative_advantage",
        "response_norm_mse_stability_advantage",
        "diag_whitened_mse_stability_advantage",
        "full_cov_whitened_mse_stability_advantage",
    ]
    keep = [col for col in keep if col in stability.columns]
    base = drift.merge(stability[keep], on="source_window_id", how="left", validate="one_to_one")
    base["drift_edge_cos2_session_demeaned"] = _demean_by_session(base["drift_edge_cos2"].to_numpy(dtype=float), base["session_id"].to_numpy())
    base["drift_edge_align_signed_session_demeaned"] = _demean_by_session(
        base["drift_edge_align_signed"].to_numpy(dtype=float), base["session_id"].to_numpy()
    )
    return base


def _parallel_minus_orthogonal(df: pd.DataFrame, metrics: list[str], group_cols: list[str]) -> pd.DataFrame:
    if "prior_condition" not in df.columns:
        return pd.DataFrame()
    rows = []
    metric_cols = [col for col in metrics if col in df.columns]
    for key, block in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec: dict[str, Any] = dict(zip(group_cols, key))
        par = block[block["prior_condition"].astype(str) == "axis_edge_parallel"]
        orth = block[block["prior_condition"].astype(str) == "axis_edge_orthogonal"]
        if par.empty or orth.empty:
            continue
        for col in metric_cols:
            rec[f"{col}_parallel_minus_orthogonal"] = float(pd.to_numeric(par[col], errors="coerce").mean() - pd.to_numeric(orth[col], errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def _load_observer_cache(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "observation_source_row" not in df.columns:
        return pd.DataFrame()
    df["source_window_id"] = pd.to_numeric(df["observation_source_row"], errors="coerce")
    if "posterior_N_eff_true_image" in df.columns:
        df["posterior_concentration"] = 1.0 - pd.to_numeric(df["posterior_N_eff_true_image"], errors="coerce") / pd.to_numeric(df["n_candidates"], errors="coerce")
    if {"known_true_score", "joint_true_score"}.issubset(df.columns):
        df["known_minus_joint_pose_cost"] = pd.to_numeric(df["known_true_score"], errors="coerce") - pd.to_numeric(df["joint_true_score"], errors="coerce")
    group_cols = ["source_window_id"]
    for col in ["observation_scale", "prior_scale", "likelihood_scale"]:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            group_cols.append(col)
    wide = _parallel_minus_orthogonal(df, OBSERVER_METRICS, group_cols)
    if wide.empty:
        return wide
    nuisance_cols = set(group_cols).difference({"source_window_id"})
    if len(nuisance_cols) == 0:
        rename = {col: _slug([name, col]) for col in wide.columns if col not in group_cols}
        wide = wide.rename(columns=rename)
        return wide
    pieces = []
    for key, block in wide.groupby(list(nuisance_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(list(nuisance_cols), key))
        value_cols = [col for col in block.columns if col not in group_cols]
        rename = {
            col: _slug([name, "all_window_global_axis_nuisance", *[f"{k}_{key_map[k]}" for k in sorted(key_map)], col])
            for col in value_cols
        }
        pieces.append(block[["source_window_id", *value_cols]].rename(columns=rename))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="source_window_id", how="outer", validate="one_to_one")
    return out


def _load_feature_cache(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "source_window_id" not in df.columns:
        if "observation_source_row" in df.columns:
            df["source_window_id"] = pd.to_numeric(df["observation_source_row"], errors="coerce")
        elif "true_image_id" in df.columns:
            df["source_window_id"] = _extract_source_row(df["true_image_id"])
    if "source_window_id" not in df.columns:
        return pd.DataFrame()
    if "candidate_posterior_N_eff_fraction" in df.columns:
        df["posterior_concentration"] = 1.0 - pd.to_numeric(df["candidate_posterior_N_eff_fraction"], errors="coerce")
    key_cols = [
        "source_window_id",
        "latent",
        "requested_k",
    ]
    for col in ["observation_scale", "prior_scale", "likelihood_scale"]:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            key_cols.append(col)
    mode_metric = df.pivot_table(
        index=[*key_cols, "prior_condition"],
        columns="observer_mode",
        values=["feature_neg_mse", "candidate_posterior_true_mass", "posterior_concentration", "score_true_margin", "score_true_value", "feature_cosine"],
        aggfunc="mean",
    )
    mode_metric.columns = [f"{metric}_{mode}" for metric, mode in mode_metric.columns]
    mode_metric = mode_metric.reset_index()
    if {"feature_neg_mse_joint", "feature_neg_mse_zero"}.issubset(mode_metric.columns):
        mode_metric["joint_minus_zero_feature_gain"] = mode_metric["feature_neg_mse_joint"] - mode_metric["feature_neg_mse_zero"]
    if {"feature_neg_mse_known", "feature_neg_mse_joint"}.issubset(mode_metric.columns):
        mode_metric["known_minus_joint_pose_cost"] = mode_metric["feature_neg_mse_known"] - mode_metric["feature_neg_mse_joint"]
    if {"feature_neg_mse_motion_delta", "feature_neg_mse_zero"}.issubset(mode_metric.columns):
        mode_metric["motion_delta_minus_zero_feature_gain"] = mode_metric["feature_neg_mse_motion_delta"] - mode_metric["feature_neg_mse_zero"]
    metrics = [
        "candidate_posterior_true_mass_joint",
        "posterior_concentration_joint",
        "score_true_margin_joint",
        "score_true_value_joint",
        "feature_neg_mse_joint",
        "feature_cosine_joint",
        "joint_minus_zero_feature_gain",
        "known_minus_joint_pose_cost",
        "motion_delta_minus_zero_feature_gain",
    ]
    wide = _parallel_minus_orthogonal(mode_metric, metrics, key_cols)
    if wide.empty:
        return wide
    pieces = []
    nuisance_cols = [col for col in key_cols if col != "source_window_id"]
    for key, block in wide.groupby(nuisance_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(nuisance_cols, key))
        value_cols = [col for col in block.columns if col not in key_cols]
        rename = {
            col: _slug(
                [
                    name,
                    "all_window_global_axis_nuisance",
                    *[f"{k}_{key_map[k]}" for k in nuisance_cols],
                    col,
                ]
            )
            for col in value_cols
        }
        pieces.append(block[["source_window_id", *value_cols]].rename(columns=rename))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="source_window_id", how="outer", validate="one_to_one")
    return out


def _merge_optional_tables(base: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for table in tables:
        if table.empty:
            continue
        table = table.copy()
        table["source_window_id"] = pd.to_numeric(table["source_window_id"], errors="coerce")
        out = out.merge(table, on="source_window_id", how="left", validate="one_to_one")
    return out


def _predictor_blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    observer_cols = [c for c in df.columns if c.startswith("observer_") and c.endswith("_parallel_minus_orthogonal")]
    feature_cols = [c for c in df.columns if c.startswith("feature_") and c.endswith("_parallel_minus_orthogonal")]
    compact_cols = [c for c in df.columns if "compact" in c and pd.api.types.is_numeric_dtype(df[c])]
    blocks = {
        "observer": observer_cols,
        "feature": feature_cols,
        "compact": compact_cols,
        "observer_feature_compact": observer_cols + feature_cols + compact_cols,
    }
    return {name: cols for name, cols in blocks.items() if cols}


def _model_rows(
    df: pd.DataFrame,
    y_col: str,
    controls: list[str],
    blocks: dict[str, list[str]],
    *,
    include_predictor_coefs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    inc_rows = []
    pred_table = pd.DataFrame(index=df.index)
    pred_table["source_window_id"] = df["source_window_id"].to_numpy()
    base_fit = _fit_block_model(df, y_col, controls, [])
    rows.append(
        {
            "model": "baseline",
            "target": y_col,
            "predictor_block": "baseline",
            "predictors": "",
            "n_predictors": 0,
            "r2": base_fit["r2"],
            "incremental_r2_vs_baseline": 0.0,
            "n_windows": base_fit["n_windows"],
            "n_sessions": base_fit["n_sessions"],
        }
    )
    for block_name, predictors in blocks.items():
        fit = _fit_block_model(df, y_col, controls, predictors)
        block_base = _fit_block_model(df.loc[fit.get("index", [])], y_col, controls, []) if fit.get("index") is not None else base_fit
        delta = float(fit["r2"] - block_base["r2"]) if np.isfinite(fit["r2"]) and np.isfinite(block_base["r2"]) else float("nan")
        rows.append(
            {
                "model": f"baseline_plus_{block_name}",
                "target": y_col,
                "predictor_block": block_name,
                "predictors": "+".join(fit.get("predictors_used", predictors)),
                "n_predictors": len(predictors),
                "n_predictors_used": len(fit.get("predictors_used", [])),
                "r2": fit["r2"],
                "incremental_r2_vs_baseline": delta,
                "n_windows": fit["n_windows"],
                "n_sessions": fit["n_sessions"],
                "baseline_r2_same_rows": block_base["r2"],
            }
        )
        inc_rows.append(
            {
                "target": y_col,
                "predictor_block": block_name,
                "incremental_r2": delta,
                "full_r2": fit["r2"],
                "baseline_r2": block_base["r2"],
                "n_predictors": len(predictors),
                "n_predictors_used": len(fit.get("predictors_used", [])),
                "n_windows": fit["n_windows"],
                "n_sessions": fit["n_sessions"],
            }
        )
        if block_name == "observer_feature_compact" and fit.get("index") is not None:
            pred_table.loc[fit["index"], "observed_y_demeaned_z"] = _zscore(
                _demean_by_session(df.loc[fit["index"], y_col].to_numpy(dtype=float), df.loc[fit["index"], "session_id"].to_numpy())
            )
            pred_table.loc[fit["index"], "predicted_y_demeaned_z"] = fit["pred"]
            pred_table.loc[fit["index"], "model_residual_z"] = fit["resid"]
        if not include_predictor_coefs:
            continue
        for predictor in predictors:
            single = _fit_block_model(df, y_col, controls, [predictor])
            coef = single["coef"].get(predictor, np.nan)
            if not np.isfinite(coef):
                continue
            single_base = _fit_block_model(df.loc[single.get("index", [])], y_col, controls, [])
            rows.append(
                {
                    "model": f"baseline_plus_{block_name}",
                    "target": y_col,
                    "predictor_block": block_name,
                    "predictor": predictor,
                    "standardized_coef": coef,
                    "predictor_flag": "nuisance_control" if "all_window_global_axis_nuisance" in predictor else "window_joined",
                    "r2": single["r2"],
                    "incremental_r2_vs_baseline": float(single["r2"] - single_base["r2"]) if np.isfinite(single["r2"]) and np.isfinite(single_base["r2"]) else float("nan"),
                    "n_windows": single["n_windows"],
                    "n_sessions": single["n_sessions"],
                    "baseline_r2_same_rows": single_base["r2"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(inc_rows), pred_table


def _session_bootstrap(
    df: pd.DataFrame,
    y_col: str,
    controls: list[str],
    blocks: dict[str, list[str]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    sessions = np.asarray(sorted(df["session_id"].dropna().unique()))
    rows = []
    if sessions.size < 2 or int(n_bootstrap) <= 0:
        return pd.DataFrame()
    observed_rows, observed_inc, _ = _model_rows(df, y_col, controls, blocks, include_predictor_coefs=False)
    coef_obs = observed_rows.dropna(subset=["standardized_coef"]) if "standardized_coef" in observed_rows.columns else pd.DataFrame()
    coef_keys = [(r["predictor_block"], r["predictor"]) for _, r in coef_obs.iterrows()]
    boot_inc: dict[str, list[float]] = {name: [] for name in blocks}
    boot_coef: dict[tuple[str, str], list[float]] = {key: [] for key in coef_keys}
    for _ in range(int(n_bootstrap)):
        draw = rng.choice(sessions, size=sessions.size, replace=True)
        pieces = []
        for j, sess in enumerate(draw):
            piece = df[df["session_id"] == sess].copy()
            piece["session_id"] = f"{sess}__boot{j}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        summary, inc, _ = _model_rows(sample, y_col, controls, blocks, include_predictor_coefs=False)
        for _, row in inc.iterrows():
            boot_inc[str(row["predictor_block"])].append(float(row["incremental_r2"]))
        coefs = summary.dropna(subset=["standardized_coef"]) if "standardized_coef" in summary.columns else pd.DataFrame()
        for _, row in coefs.iterrows():
            key = (str(row["predictor_block"]), str(row["predictor"]))
            if key in boot_coef:
                boot_coef[key].append(float(row["standardized_coef"]))
    for _, row in observed_inc.iterrows():
        vals = np.asarray(boot_inc[str(row["predictor_block"])], dtype=float)
        rows.append(
            {
                "target": y_col,
                "statistic": "incremental_r2",
                "predictor_block": row["predictor_block"],
                "predictor": "",
                "observed": float(row["incremental_r2"]),
                "ci_low": float(np.nanpercentile(vals, 2.5)),
                "ci_high": float(np.nanpercentile(vals, 97.5)),
                "bootstrap_n": int(n_bootstrap),
            }
        )
    for _, row in coef_obs.iterrows():
        key = (str(row["predictor_block"]), str(row["predictor"]))
        vals = np.asarray(boot_coef.get(key, []), dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "target": y_col,
                "statistic": "standardized_coef",
                "predictor_block": key[0],
                "predictor": key[1],
                "observed": float(row["standardized_coef"]),
                "ci_low": float(np.nanpercentile(vals, 2.5)),
                "ci_high": float(np.nanpercentile(vals, 97.5)),
                "bootstrap_n": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows)


def _session_sign_counts(df: pd.DataFrame, y_col: str, controls: list[str], blocks: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for block_name, predictors in blocks.items():
        for predictor in predictors:
            vals = []
            for session, sub in df.groupby("session_id", sort=True):
                if sub.shape[0] < len(controls) + 4:
                    continue
                fit = _fit_block_model(
                    sub,
                    y_col,
                    controls,
                    [predictor],
                    require_multi_session_predictor=False,
                    min_predictor_obs=3,
                )
                coef = fit["coef"].get(predictor, np.nan)
                if np.isfinite(coef):
                    vals.append((session, coef))
            arr = np.asarray([v for _, v in vals], dtype=float)
            rows.append(
                {
                    "target": y_col,
                    "predictor_block": block_name,
                    "predictor": predictor,
                    "n_sessions_with_fit": int(arr.size),
                    "sessions_positive": int(np.count_nonzero(arr > 0)),
                    "sessions_negative": int(np.count_nonzero(arr < 0)),
                    "fraction_positive": float(np.mean(arr > 0)) if arr.size else float("nan"),
                    "median_session_coef": float(np.nanmedian(arr)) if arr.size else float("nan"),
                    "predictor_flag": "nuisance_control" if "all_window_global_axis_nuisance" in predictor else "window_joined",
                }
            )
    return pd.DataFrame(rows)


def _write_plots(out_dir: Path, model_summary: pd.DataFrame, inc: pd.DataFrame, pred: pd.DataFrame, boot: pd.DataFrame) -> None:
    coef = model_summary.dropna(subset=["standardized_coef"]).copy() if "standardized_coef" in model_summary.columns else pd.DataFrame()
    if not coef.empty:
        focus = coef.sort_values("standardized_coef").tail(40)
        err = boot[boot["statistic"] == "standardized_coef"] if (not boot.empty and "statistic" in boot.columns) else pd.DataFrame()
        if {"target", "predictor", "ci_low", "ci_high"}.issubset(err.columns):
            focus = focus.merge(err[["target", "predictor", "ci_low", "ci_high"]], on=["target", "predictor"], how="left")
        fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.18 * focus.shape[0])), dpi=150)
        y = np.arange(focus.shape[0])
        colors = ["#2f6f73" if v >= 0 else "#b65c35" for v in focus["standardized_coef"]]
        ax.barh(y, focus["standardized_coef"], color=colors)
        if {"ci_low", "ci_high"}.issubset(focus.columns):
            lo = focus["standardized_coef"] - focus["ci_low"]
            hi = focus["ci_high"] - focus["standardized_coef"]
            ax.errorbar(focus["standardized_coef"], y, xerr=[lo, hi], fmt="none", color="black", linewidth=0.7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(focus["predictor"], fontsize=6)
        ax.set_xlabel("standardized within-session coefficient")
        fig.tight_layout()
        fig.savefig(out_dir / "coefficient_forest.png", dpi=150)
        plt.close(fig)
    if not inc.empty:
        fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
        x = np.arange(inc.shape[0])
        ax.bar(x, inc["incremental_r2"], color="#4d79a8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(inc["predictor_block"], rotation=30, ha="right")
        ax.set_ylabel("Delta R2 beyond baseline")
        fig.tight_layout()
        fig.savefig(out_dir / "incremental_r2_barplot.png", dpi=150)
        plt.close(fig)
    if {"observed_y_demeaned_z", "predicted_y_demeaned_z"}.issubset(pred.columns):
        ok = pred[["observed_y_demeaned_z", "predicted_y_demeaned_z"]].dropna()
        if not ok.empty:
            fig, ax = plt.subplots(figsize=(4.4, 4.1), dpi=150)
            ax.scatter(ok["predicted_y_demeaned_z"], ok["observed_y_demeaned_z"], s=14, alpha=0.7, color="#596f3d")
            lim = float(np.nanmax(np.abs(ok.to_numpy()))) if ok.size else 1.0
            ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.8)
            ax.set_xlabel("predicted residual z")
            ax.set_ylabel("observed residual z")
            fig.tight_layout()
            fig.savefig(out_dir / "observed_vs_predicted_residuals.png", dpi=150)
            plt.close(fig)


def _write_report(
    out_dir: Path,
    table: pd.DataFrame,
    inc: pd.DataFrame,
    boot: pd.DataFrame,
    sign_counts: pd.DataFrame,
    paths: dict[str, Any],
) -> None:
    primary = inc.sort_values("incremental_r2", ascending=False).head(8) if not inc.empty else pd.DataFrame()
    boot_inc = boot[boot["statistic"] == "incremental_r2"] if (not boot.empty and "statistic" in boot.columns) else pd.DataFrame()
    if not primary.empty and {"target", "predictor_block", "ci_low", "ci_high"}.issubset(boot_inc.columns):
        best = primary.merge(boot_inc[["target", "predictor_block", "ci_low", "ci_high"]], on=["target", "predictor_block"], how="left")
    else:
        best = primary.copy()
    success = bool((best["incremental_r2"] > 0).any()) if not best.empty else False
    stable = bool((best.get("ci_low", pd.Series(dtype=float)) > 0).any()) if not best.empty and "ci_low" in best.columns else False
    model_cols = [col for col in table.columns if col.startswith(("observer_", "feature_"))]
    preservation_n = int(table["pixel_relative_advantage"].notna().sum()) if "pixel_relative_advantage" in table.columns else 0
    model_n = int(table[model_cols].notna().any(axis=1).sum()) if model_cols else 0
    lines = [
        "# BackImage Axis Residual Adjudication",
        "",
        "## Claim Boundary",
        "",
        "This is a cache-only posthoc. It tests whether observer, feature-posterior, and compact-like cached predictors explain within-session signed drift-edge alignment beyond low-level image geometry and pixel edge-parallel preservation controls. Predictors tagged `all_window_global_axis_nuisance` are global/axis-cache diagnostics, not fresh per-window measurements.",
        "",
        "## Inputs",
        "",
        f"- Drift windows: `{paths['drift']}`",
        f"- Pixel/V1 preservation: `{paths['stability']}`",
        f"- Joined windows: `{table.shape[0]}` across `{table['session_id'].nunique()}` sessions.",
        f"- Windows with pixel-preservation cache: `{preservation_n}`.",
        f"- Windows with observer/feature model cache: `{model_n}`.",
        "- Baseline controls are included when they have enough finite coverage on the fitted rows; sparse pixel-preservation controls are therefore a coverage caveat when observer/feature windows do not overlap the preservation audit.",
        "",
        "## Incremental R2",
        "",
    ]
    if best.empty:
        lines.append("- No eligible model-derived predictor blocks were available after joins.")
    else:
        for _, row in best.iterrows():
            ci = ""
            if "ci_low" in row and np.isfinite(row["ci_low"]):
                ci = f", session-bootstrap CI `[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]`"
            lines.append(f"- `{row['predictor_block']}`: Delta R2 `{row['incremental_r2']:+.4f}`{ci}.")
    lines.extend(["", "## Session Sign Counts", ""])
    sign_focus = sign_counts[sign_counts["n_sessions_with_fit"] > 0].copy() if not sign_counts.empty else pd.DataFrame()
    if sign_focus.empty:
        lines.append("- No per-session coefficient fits were available.")
    else:
        for _, row in sign_focus.sort_values(["fraction_positive", "n_sessions_with_fit"], ascending=False).head(12).iterrows():
            lines.append(
                f"- `{row['predictor']}`: + `{row['sessions_positive']}` / - `{row['sessions_negative']}` "
                f"of `{row['n_sessions_with_fit']}` sessions; median coef `{row['median_session_coef']:+.4f}`."
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "Provisional success criterion: model-derived predictor blocks add positive, session-stable Delta R2 beyond the available baseline controls. Treat this as provisional when preservation overlap is sparse."
                if success and stable
                else "Failure/hold criterion: current model-derived quantities do not yet provide session-stable incremental explanatory power beyond raw edge geometry and pixel preservation baselines."
            ),
            "",
            "Generated outputs: `residual_adjudication_window_table.csv`, `residual_model_summary.csv`, `predictor_incremental_r2.csv`, `session_bootstrap_summary.csv`, `coefficient_forest.png`, `incremental_r2_barplot.png`, and `observed_vs_predicted_residuals.png`.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    base = _load_base_table(Path(args.drift_windows), Path(args.stability_table))
    tables = []
    used: dict[str, str] = {}
    for name, path in _parse_named_paths(list(args.observer_cache)):
        if path.exists():
            tables.append(_load_observer_cache(name, path))
            used[name] = str(path)
    for name, path in _parse_named_paths(list(args.feature_cache)):
        if path.exists():
            tables.append(_load_feature_cache(name, path))
            used[name] = str(path)
    table = _merge_optional_tables(base, tables)
    table.to_csv(out_dir / "residual_adjudication_window_table.csv", index=False)
    controls = [col for col in BASELINE_CONTROLS if col in table.columns and pd.api.types.is_numeric_dtype(table[col])]
    blocks = _predictor_blocks(table)
    model_summary, inc, pred = _model_rows(table, str(args.target), controls, blocks)
    boot = _session_bootstrap(table, str(args.target), controls, blocks, rng=rng, n_bootstrap=int(args.n_bootstrap))
    sign_counts = _session_sign_counts(table, str(args.target), controls, blocks)
    model_summary.to_csv(out_dir / "residual_model_summary.csv", index=False)
    inc.to_csv(out_dir / "predictor_incremental_r2.csv", index=False)
    boot.to_csv(out_dir / "session_bootstrap_summary.csv", index=False)
    sign_counts.to_csv(out_dir / "session_sign_count_summary.csv", index=False)
    pred.to_csv(out_dir / "observed_vs_predicted_residuals.csv", index=False)
    _write_plots(out_dir, model_summary, inc, pred, boot)
    paths = {"drift": str(args.drift_windows), "stability": str(args.stability_table), "used_model_caches": used}
    (out_dir / "posthoc_metadata.json").write_text(
        json.dumps(
            {
                **paths,
                "target": str(args.target),
                "baseline_controls": controls,
                "predictor_blocks": {k: v for k, v in blocks.items()},
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, table, inc, boot, sign_counts, paths)
    print(f"Wrote BackImage axis residual adjudication to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-windows", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--stability-table", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--observer-cache", action="append", default=list(DEFAULT_OBSERVER_CACHE_SPECS), help="NAME=observer_trials.csv")
    parser.add_argument("--feature-cache", action="append", default=list(DEFAULT_FEATURE_CACHE_SPECS), help="NAME=feature_posterior_trials.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target", default="drift_edge_cos2", choices=["drift_edge_cos2", "drift_edge_align_signed"])
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
