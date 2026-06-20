"""Cache-first residual adjudication for BackImage raw-edge alignment.

This script asks whether cached preservation, joint-posterior, or
feature-posterior variables explain residual variation in signed BackImage
drift-edge alignment after raw image geometry and simple FEM confidence
variables have been accounted for.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
DEFAULT_WINDOWS = ROOT / "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
DEFAULT_STABILITY = ROOT / "backimage_twin_stability_metric_audit/twin_stability_metric_by_window.csv"
DEFAULT_FEATURE_PRESERVATION = (
    ROOT / "backimage_twin_stability_metric_audit/endpoint_feature_preservation_static_decoder/feature_preservation_by_window.csv"
)
DEFAULT_OBSERVER = (
    ROOT
    / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1/observer_trials.csv"
)
DEFAULT_FEATURE_POSTERIOR = (
    ROOT
    / "backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_16_uncertainty_v1/"
    "feature_posterior_trials.csv"
)
DEFAULT_FEATURE_AXIS_CONTRASTS = (
    ROOT
    / "backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_16_uncertainty_v1/"
    "feature_axis_contrasts.csv"
)
DEFAULT_OUT_DIR = ROOT / "backimage_raw_edge_roadblock_residual_adjudication_v1"

TARGET = "drift_edge_cos2"
SESSION_COL = "session"
SOURCE_COL = "raw_window_row"
AXIS_FAMILIES = ("axis_edge_parallel", "axis_edge_orthogonal")

RAW_PREDICTORS = [
    "image_orientation_coherence",
    "image_edge_density",
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_spectrum_anisotropy",
    "image_high_freq_power_fraction",
    "image_patch_distance_to_image_border_px",
    "rms_radius_deg",
    "anisotropy",
    "duration_s",
    "path_length_deg",
    "phase_late_fixation",
]

PRESERVATION_PREDICTORS = [
    "pixel_stability_advantage",
    "twin_stability_advantage",
    "raw_mse_stability_advantage",
    "response_norm_mse_stability_advantage",
    "per_rate_mse_stability_advantage",
    "diag_whitened_mse_stability_advantage",
    "full_cov_whitened_mse_stability_advantage",
    "edge_parallel_preservation_minus_orthogonal_mean",
    "edge_parallel_preservation_minus_orthogonal_gabor_local_field_k4",
    "edge_parallel_preservation_minus_orthogonal_pyramid_local_field_k4",
]

OBSERVER_METRICS = [
    "joint_correct",
    "joint_true_score",
    "joint_true_margin",
    "joint_minus_zero_true_score",
    "N_eff_true_image_fraction",
    "posterior_entropy_true_image",
    "max_tau_posterior_true_image",
    "posterior_concentration_true_image",
    "known_minus_joint_pose_cost",
]

OBSERVER_HARDNESS_METRICS = [
    "static_response_distance_to_nearest_distractor",
    "mean_rate_distance_to_nearest_distractor",
    "contrast_distance_to_nearest_distractor",
    "n_matched_distractors",
    "n_random_fallback_distractors",
    "random_fallback_used",
]

FEATURE_DERIVED_METRICS = [
    "joint_feature_recovery",
    "joint_minus_zero_feature_gain",
    "known_minus_joint_pose_cost",
    "motion_delta_minus_zero_feature_gain",
    "joint_candidate_true_mass",
    "joint_posterior_concentration",
]


@dataclass(frozen=True)
class FitResult:
    model_name: str
    predictors: list[str]
    predictors_used: list[str]
    n_windows: int
    n_sessions: int
    r2: float
    loos_within_session_r2: float
    y: np.ndarray
    pred: np.ndarray
    resid: np.ndarray
    index: np.ndarray


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p")
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


def _zscore(values: np.ndarray, *, mean: float | None = None, sd: float | None = None) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if mean is None:
        mean = float(np.nanmean(values))
    if sd is None:
        sd = float(np.nanstd(values))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(values, dtype=np.float64), float(mean), 1.0
    return (values - mean) / sd, float(mean), float(sd)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 3:
        return float("nan")
    if np.nanstd(x[ok]) <= 1e-12 or np.nanstd(y[ok]) <= 1e-12:
        return float("nan")
    return float(spearmanr(x[ok], y[ok], nan_policy="omit").statistic)


def _demean_by_session(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return (series - series.groupby(pd.Series(sessions)).transform("mean")).to_numpy(dtype=np.float64)


def _axial_abs_delta_deg(angle_a: pd.Series, angle_b: pd.Series) -> np.ndarray:
    delta = (pd.to_numeric(angle_a, errors="coerce") - pd.to_numeric(angle_b, errors="coerce") + 90.0) % 180.0 - 90.0
    return np.abs(delta.to_numpy(dtype=np.float64))


def _session_bootstrap_mean(values: np.ndarray, sessions: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    sessions = np.asarray(sessions)
    ok = np.isfinite(values) & pd.notna(sessions)
    values = values[ok]
    sessions = sessions[ok]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_windows": 0,
            "n_sessions": 0,
        }
    unique = np.asarray(sorted(pd.unique(sessions)))
    session_means = pd.Series(values).groupby(pd.Series(sessions)).mean().reindex(unique).to_numpy(dtype=np.float64)
    observed = float(np.nanmean(session_means))
    if int(n_bootstrap) <= 0 or unique.size < 2:
        return {
            "mean": observed,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_windows": int(values.size),
            "n_sessions": int(unique.size),
        }
    draws = rng.choice(session_means, size=(int(n_bootstrap), unique.size), replace=True)
    boot = np.nanmean(draws, axis=1)
    return {
        "mean": observed,
        "ci_low": float(np.nanpercentile(boot, 2.5)),
        "ci_high": float(np.nanpercentile(boot, 97.5)),
        "n_windows": int(values.size),
        "n_sessions": int(unique.size),
    }


def _read_raw_windows(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path).reset_index(names=SOURCE_COL)
    raw[SESSION_COL] = raw["session"].astype(str)
    raw["session_id"] = raw[SESSION_COL]
    raw["drift_anisotropy"] = pd.to_numeric(raw.get("anisotropy"), errors="coerce")
    raw["phase_late_fixation"] = (raw.get("phase", "").astype(str) == "late_fixation").astype(float)
    raw["reliable_axis"] = (
        (pd.to_numeric(raw["image_orientation_coherence"], errors="coerce") >= 0.20)
        & (pd.to_numeric(raw["anisotropy"], errors="coerce") >= 0.20)
    )
    raw["high_confidence"] = (
        (pd.to_numeric(raw["image_orientation_coherence"], errors="coerce") >= 0.50)
        & (pd.to_numeric(raw["anisotropy"], errors="coerce") >= 0.50)
    )
    if "image_patch_radius_px" in raw.columns and "image_patch_distance_to_image_border_px" in raw.columns:
        raw["near_image_border"] = (
            pd.to_numeric(raw["image_patch_distance_to_image_border_px"], errors="coerce")
            < pd.to_numeric(raw["image_patch_radius_px"], errors="coerce")
        )
    else:
        raw["near_image_border"] = False
    raw["drift_edge_abs_delta_deg"] = _axial_abs_delta_deg(raw["drift_orientation_deg"], raw["image_edge_axis_deg"])
    key_cols = [
        "session",
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "image_patch_center_x_px",
        "image_patch_center_y_px",
    ]
    raw["raw_window_key"] = raw[key_cols].astype(str).agg("|".join, axis=1)
    return raw


def _summarize_alignment(df: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "all": df[TARGET].notna(),
        "reliable_axis": df["reliable_axis"].astype(bool),
        "high_confidence": df["high_confidence"].astype(bool),
    }
    for name, mask in subsets.items():
        sub = df.loc[mask].copy()
        boot = _session_bootstrap_mean(sub[TARGET].to_numpy(dtype=np.float64), sub[SESSION_COL].to_numpy(), rng=rng, n_bootstrap=n_bootstrap)
        rows.append(
            {
                "subset": name,
                "n_windows": int(sub.shape[0]),
                "n_sessions": int(sub[SESSION_COL].nunique()),
                "window_mean_drift_edge_cos2": float(np.nanmean(sub[TARGET])),
                "session_mean_drift_edge_cos2": boot["mean"],
                "session_bootstrap_ci_low": boot["ci_low"],
                "session_bootstrap_ci_high": boot["ci_high"],
                "positive_sessions": int((sub.groupby(SESSION_COL)[TARGET].mean() > 0).sum()),
                "median_abs_drift_edge_delta_deg": float(np.nanmedian(sub["drift_edge_abs_delta_deg"])),
                "fraction_within_30deg_parallel": float(np.nanmean(sub["drift_edge_abs_delta_deg"] <= 30.0)),
                "fraction_within_30deg_orthogonal": float(np.nanmean(sub["drift_edge_abs_delta_deg"] >= 60.0)),
            }
        )
    return pd.DataFrame(rows)


def _load_preservation_tables(
    raw: pd.DataFrame,
    stability_path: Path,
    feature_preservation_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_cols = ["session", "trial_idx", "global_start", "global_stop", "local_start", "local_stop"]
    joined = raw[[SOURCE_COL, "raw_window_key", TARGET, "image_edge_axis_deg", *key_cols]].copy()
    qc_rows: list[dict[str, Any]] = []

    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        if not set(key_cols).issubset(stability.columns):
            missing = sorted(set(key_cols).difference(stability.columns))
            raise ValueError(f"Missing stability join keys {missing} in {stability_path}")
        keep = [*key_cols, "source_window_id", "window_row", "window_id", "session_id", "image_id"]
        keep.extend([col for col in PRESERVATION_PREDICTORS if col in stability.columns])
        keep.extend(
            [
                col
                for col in [
                    "edge_axis_deg",
                    "real_drift_axis_deg",
                    "drift_edge_align_signed",
                    "image_orientation_coherence",
                    "drift_anisotropy",
                    "global_start",
                    "global_stop",
                    "local_start",
                    "local_stop",
                    "image_edge_axis_deg",
                    "trial_idx",
                    "session",
                ]
                if col in stability.columns and col not in keep
            ]
        )
        stability_keep = stability[keep].copy().rename(
            columns={
                "source_window_id": "stability_source_window_id",
                "window_row": "stability_window_row",
                "window_id": "stability_window_id",
                "session_id": "stability_session_id",
                "image_id": "stability_image_id",
                "image_edge_axis_deg": "stability_image_edge_axis_deg",
                "image_orientation_coherence": "stability_image_orientation_coherence",
                "drift_anisotropy": "stability_drift_anisotropy",
            }
        )
        joined = joined.merge(stability_keep, on=key_cols, how="left", validate="one_to_one")
        overlap = (
            joined["stability_source_window_id"].notna()
            if "stability_source_window_id" in joined.columns
            else pd.Series(False, index=joined.index)
        )
        qc_rows.append(
            {
                "join": "stability_by_composite_window_key",
                "left_rows": int(raw.shape[0]),
                "right_rows": int(stability.shape[0]),
                "matched_left_rows": int(overlap.sum()),
                "matched_right_unique_keys": int(stability[key_cols].drop_duplicates().shape[0]),
                "duplicate_right_keys": int(stability.duplicated(key_cols).sum()),
            }
        )
    else:
        qc_rows.append({"join": "stability_by_composite_window_key", "status": "missing", "path": str(stability_path)})

    if feature_preservation_path.exists():
        feature = pd.read_csv(feature_preservation_path)
        if not set(key_cols).issubset(feature.columns):
            missing = sorted(set(key_cols).difference(feature.columns))
            raise ValueError(f"Missing feature-preservation join keys {missing} in {feature_preservation_path}")
        piv = feature.pivot_table(
            index=key_cols,
            columns=["latent_name", "pca_k"],
            values="edge_parallel_preservation_minus_orthogonal",
            aggfunc="mean",
        )
        piv.columns = [
            f"edge_parallel_preservation_minus_orthogonal_{_slug(latent)}_k{int(k)}" for latent, k in piv.columns
        ]
        mean_by_key = (
            feature.groupby(key_cols, dropna=False)["edge_parallel_preservation_minus_orthogonal"]
            .mean()
            .rename("edge_parallel_preservation_minus_orthogonal_mean")
        )
        feature_wide = pd.concat([mean_by_key, piv], axis=1).reset_index()
        joined = joined.merge(feature_wide, on=key_cols, how="left", validate="one_to_one")
        qc_rows.append(
            {
                "join": "feature_preservation_by_composite_window_key",
                "left_rows": int(raw.shape[0]),
                "right_rows": int(feature.shape[0]),
                "matched_left_rows": int(joined["edge_parallel_preservation_minus_orthogonal_mean"].notna().sum()),
                "matched_right_unique_keys": int(feature[key_cols].drop_duplicates().shape[0]),
                "duplicate_right_keys": int(feature.duplicated(key_cols).sum()),
            }
        )
    else:
        qc_rows.append({"join": "feature_preservation_by_composite_window_key", "status": "missing", "path": str(feature_preservation_path)})

    audit_cols = [
        col
        for col in [
            SOURCE_COL,
            SESSION_COL,
            "stability_session_id",
            "trial_idx",
            "stability_image_id",
            "global_start",
            "global_stop",
            "local_start",
            "local_stop",
            "image_edge_axis_deg",
            "stability_image_edge_axis_deg",
            "edge_axis_deg",
            "drift_edge_cos2",
            "drift_edge_align_signed",
        ]
        if col in joined.columns
    ]
    audit = joined[audit_cols].copy()
    if "edge_axis_deg" in audit.columns and "image_edge_axis_deg" in audit.columns:
        delta = np.nanmax(_axial_abs_delta_deg(audit["edge_axis_deg"], audit["image_edge_axis_deg"]))
        qc_rows.append({"join": "stability_axis_audit", "max_abs_edge_axis_delta_deg": float(delta)})
    return joined, pd.DataFrame(qc_rows), audit


def _parallel_minus_orthogonal(
    df: pd.DataFrame,
    *,
    index_cols: list[str],
    metrics: list[str],
    family_col: str = "prior_family",
) -> pd.DataFrame:
    metric_cols = [col for col in metrics if col in df.columns]
    if not metric_cols:
        return pd.DataFrame(columns=index_cols)
    grouped = df.groupby([*index_cols, family_col], dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    wide = grouped.pivot_table(index=index_cols, columns=family_col, values=metric_cols, aggfunc="mean")
    if wide.empty:
        return pd.DataFrame(columns=index_cols)
    records = pd.DataFrame(index=wide.index)
    for metric in metric_cols:
        par_key = (metric, AXIS_FAMILIES[0])
        orth_key = (metric, AXIS_FAMILIES[1])
        if par_key in wide.columns and orth_key in wide.columns:
            records[f"{metric}_parallel_minus_orthogonal"] = wide[par_key] - wide[orth_key]
    records = records.reset_index()
    return records


def _load_observer_axis_deltas(
    path: Path,
    *,
    include_scales: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame([{"join": "observer_axis_delta", "status": "missing", "path": str(path)}])
    obs = pd.read_csv(path)
    obs = obs[obs["prior_family"].isin(AXIS_FAMILIES)].copy()
    obs = obs[pd.to_numeric(obs["observation_scale"], errors="coerce").isin(include_scales)].copy()
    if "axis_shared_source_catalog" in obs.columns:
        obs = obs[obs["axis_shared_source_catalog"].astype(bool)].copy()
    obs[SOURCE_COL] = pd.to_numeric(obs["observation_source_row"], errors="coerce").astype("Int64")
    obs["known_minus_joint_pose_cost"] = pd.to_numeric(obs["known_true_score"], errors="coerce") - pd.to_numeric(
        obs["joint_true_score"], errors="coerce"
    )
    obs["posterior_concentration_true_image"] = 1.0 - pd.to_numeric(obs["N_eff_true_image_fraction"], errors="coerce")
    index_cols = [SOURCE_COL, "trial_id", "candidate_set_mode", "observation_scale", "prior_scale", "likelihood_scale"]
    delta = _parallel_minus_orthogonal(obs, index_cols=index_cols, metrics=OBSERVER_METRICS)
    hardness_cols = [col for col in OBSERVER_HARDNESS_METRICS if col in obs.columns]
    if hardness_cols:
        hardness = obs.groupby(index_cols, dropna=False)[hardness_cols].mean(numeric_only=True).reset_index()
        delta = delta.merge(hardness, on=index_cols, how="left", validate="one_to_one")
    pair_counts = obs.groupby(index_cols, dropna=False)["prior_family"].nunique().rename("n_axis_families").reset_index()
    delta = delta.merge(pair_counts, on=index_cols, how="left", validate="one_to_one")
    qc_rows.append(
        {
            "join": "observer_axis_delta",
            "input_rows": int(obs.shape[0]),
            "delta_rows": int(delta.shape[0]),
            "unique_source_windows": int(delta[SOURCE_COL].nunique(dropna=True)) if not delta.empty else 0,
            "complete_axis_pairs": int((delta.get("n_axis_families", 0) == 2).sum()) if not delta.empty else 0,
        }
    )
    return delta, pd.DataFrame(qc_rows)


def _extract_source_row(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(r"source_row:(\d+)", expand=False)
    return pd.to_numeric(parsed, errors="coerce").astype("Int64")


def _load_feature_posterior_deltas(
    path: Path,
    *,
    include_scales: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame([{"join": "feature_posterior_axis_delta", "status": "missing", "path": str(path)}])
    df = pd.read_csv(path)
    if SOURCE_COL not in df.columns:
        df[SOURCE_COL] = _extract_source_row(df["true_image_id"])
    df = df[df["prior_family"].isin(AXIS_FAMILIES)].copy()
    df = df[pd.to_numeric(df["observation_scale"], errors="coerce").isin(include_scales)].copy()
    if "axis_shared_source_catalog" in df.columns:
        df = df[df["axis_shared_source_catalog"].astype(bool)].copy()
    mode_values = [
        "feature_neg_mse",
        "feature_cosine",
        "candidate_posterior_true_mass",
        "candidate_posterior_N_eff_fraction",
        "score_true_margin",
        "score_true_value",
    ]
    mode_values = [col for col in mode_values if col in df.columns]
    index_cols = [
        SOURCE_COL,
        "trial_id",
        "candidate_set_mode",
        "observation_scale",
        "prior_scale",
        "likelihood_scale",
        "latent",
        "requested_k",
    ]
    mode = df.pivot_table(
        index=[*index_cols, "prior_family"],
        columns="observer_mode",
        values=mode_values,
        aggfunc="mean",
    )
    mode.columns = [f"{metric}_{observer}" for metric, observer in mode.columns]
    mode = mode.reset_index()
    if "feature_neg_mse_joint" in mode.columns:
        mode["joint_feature_recovery"] = mode["feature_neg_mse_joint"]
    if {"feature_neg_mse_joint", "feature_neg_mse_zero"}.issubset(mode.columns):
        mode["joint_minus_zero_feature_gain"] = mode["feature_neg_mse_joint"] - mode["feature_neg_mse_zero"]
    if {"feature_neg_mse_known", "feature_neg_mse_joint"}.issubset(mode.columns):
        mode["known_minus_joint_pose_cost"] = mode["feature_neg_mse_known"] - mode["feature_neg_mse_joint"]
    if {"feature_neg_mse_motion_delta", "feature_neg_mse_zero"}.issubset(mode.columns):
        mode["motion_delta_minus_zero_feature_gain"] = mode["feature_neg_mse_motion_delta"] - mode["feature_neg_mse_zero"]
    if "candidate_posterior_true_mass_joint" in mode.columns:
        mode["joint_candidate_true_mass"] = mode["candidate_posterior_true_mass_joint"]
    if "candidate_posterior_N_eff_fraction_joint" in mode.columns:
        mode["joint_posterior_concentration"] = 1.0 - mode["candidate_posterior_N_eff_fraction_joint"]
    delta = _parallel_minus_orthogonal(mode, index_cols=index_cols, metrics=FEATURE_DERIVED_METRICS)
    pair_counts = mode.groupby(index_cols, dropna=False)["prior_family"].nunique().rename("n_axis_families").reset_index()
    delta = delta.merge(pair_counts, on=index_cols, how="left", validate="one_to_one")
    qc_rows.append(
        {
            "join": "feature_posterior_axis_delta",
            "input_rows": int(df.shape[0]),
            "delta_rows": int(delta.shape[0]),
            "unique_source_windows": int(delta[SOURCE_COL].nunique(dropna=True)) if not delta.empty else 0,
            "complete_axis_pairs": int((delta.get("n_axis_families", 0) == 2).sum()) if not delta.empty else 0,
        }
    )
    return delta, pd.DataFrame(qc_rows)


def _wide_observer_predictors(delta: pd.DataFrame, *, primary_scale: float, primary_likelihood_scale: float) -> tuple[pd.DataFrame, list[str], list[str], list[dict[str, Any]]]:
    if delta.empty:
        return pd.DataFrame(columns=[SOURCE_COL]), [], [], []
    pieces: list[pd.DataFrame] = []
    predictor_cols: list[str] = []
    hardness_cols: list[str] = []
    dictionary: list[dict[str, Any]] = []
    value_cols = [col for col in delta.columns if col.endswith("_parallel_minus_orthogonal")]
    nuisance_cols = [col for col in OBSERVER_HARDNESS_METRICS if col in delta.columns]
    for (scale, likelihood), block in delta.groupby(["observation_scale", "likelihood_scale"], dropna=False, sort=True):
        prefix = f"observer_scale{_slug(scale)}_likelihood{_slug(likelihood)}"
        cols = [SOURCE_COL]
        rename: dict[str, str] = {}
        for col in value_cols:
            name = f"{prefix}_{col}"
            rename[col] = name
            cols.append(col)
            dictionary.append({"predictor": name, "block": "observer", "source": "observer_trials", "description": col})
            if np.isclose(float(scale), primary_scale) and np.isclose(float(likelihood), primary_likelihood_scale):
                predictor_cols.append(name)
        for col in nuisance_cols:
            name = f"{prefix}_{col}"
            rename[col] = name
            cols.append(col)
            dictionary.append({"predictor": name, "block": "candidate_hardness", "source": "observer_trials", "description": col})
            if np.isclose(float(scale), primary_scale) and np.isclose(float(likelihood), primary_likelihood_scale):
                hardness_cols.append(name)
        piece = block[cols].rename(columns=rename)
        pieces.append(piece)
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.merge(piece, on=SOURCE_COL, how="outer", validate="one_to_one")
    return wide, predictor_cols, hardness_cols, dictionary


def _wide_feature_predictors(delta: pd.DataFrame, *, primary_scale: float, primary_likelihood_scale: float) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    if delta.empty:
        return pd.DataFrame(columns=[SOURCE_COL]), [], []
    pieces: list[pd.DataFrame] = []
    predictor_cols: list[str] = []
    dictionary: list[dict[str, Any]] = []
    value_cols = [col for col in delta.columns if col.endswith("_parallel_minus_orthogonal")]
    for (scale, likelihood, latent, requested_k), block in delta.groupby(
        ["observation_scale", "likelihood_scale", "latent", "requested_k"],
        dropna=False,
        sort=True,
    ):
        prefix = f"feature_scale{_slug(scale)}_likelihood{_slug(likelihood)}_{_slug(latent)}_k{int(requested_k)}"
        cols = [SOURCE_COL]
        rename: dict[str, str] = {}
        for col in value_cols:
            name = f"{prefix}_{col}"
            rename[col] = name
            cols.append(col)
            dictionary.append({"predictor": name, "block": "feature_posterior", "source": "feature_posterior_trials", "description": col})
            is_primary = (
                np.isclose(float(scale), primary_scale)
                and np.isclose(float(likelihood), primary_likelihood_scale)
                and str(latent) in {"gabor_local_field", "pyramid_local_field"}
                and int(requested_k) in {8, 16}
            )
            if is_primary and col in {
                "joint_feature_recovery_parallel_minus_orthogonal",
                "joint_minus_zero_feature_gain_parallel_minus_orthogonal",
                "known_minus_joint_pose_cost_parallel_minus_orthogonal",
                "motion_delta_minus_zero_feature_gain_parallel_minus_orthogonal",
            }:
                predictor_cols.append(name)
        pieces.append(block[cols].rename(columns=rename))
    wide = pieces[0]
    for piece in pieces[1:]:
        wide = wide.merge(piece, on=SOURCE_COL, how="outer", validate="one_to_one")
    return wide, predictor_cols, dictionary


def _numeric_predictors(df: pd.DataFrame, predictors: Iterable[str]) -> list[str]:
    out: list[str] = []
    for col in predictors:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)).any():
                out.append(col)
    return out


def _unique_predictors(predictors: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(predictors))


def _complete_case_subset(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    used = _numeric_predictors(df, predictors)
    needed = [SESSION_COL, TARGET, *used]
    work = df.dropna(subset=needed).copy()
    if work.empty:
        return work
    arrays = [pd.to_numeric(work[TARGET], errors="coerce").to_numpy(dtype=np.float64)]
    arrays.extend(pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=np.float64) for col in used)
    ok = _finite_mask(*arrays)
    return work.loc[ok].copy()


def _build_design(df: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    used = _numeric_predictors(df, predictors)
    needed = [SESSION_COL, TARGET, *used]
    work = df.dropna(subset=needed).copy()
    if work.empty:
        return work, np.array([]), np.empty((0, 0)), np.array([]), []
    sessions = work[SESSION_COL].astype(str).to_numpy()
    y_raw = pd.to_numeric(work[TARGET], errors="coerce").to_numpy(dtype=np.float64)
    y = _demean_by_session(y_raw, sessions)
    y, _, _ = _zscore(y)
    x_cols: list[np.ndarray] = []
    final_used: list[str] = []
    for col in used:
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=np.float64)
        vals = _demean_by_session(vals, sessions)
        vals, _, _ = _zscore(vals)
        if np.nanstd(vals) <= 1e-12:
            continue
        x_cols.append(vals)
        final_used.append(col)
    X = np.column_stack(x_cols).astype(np.float64) if x_cols else np.empty((work.shape[0], 0), dtype=np.float64)
    ok = _finite_mask(y, X)
    work = work.loc[ok].copy()
    return work, y[ok], X[ok], sessions[ok], final_used


def _ols_predict(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if X.ndim == 1:
        X = X[:, None]
    design = np.column_stack([np.ones(y.shape[0], dtype=np.float64), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    resid = y - pred
    ss_tot = float(np.sum((y - np.nanmean(y)) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return pred, resid, float(r2)


def _fit_model(df: pd.DataFrame, predictors: list[str], *, model_name: str, min_sessions: int = 3) -> FitResult:
    work, y, X, sessions, used = _build_design(df, predictors)
    n_sessions = int(pd.Series(sessions).nunique()) if sessions.size else 0
    if y.size < max(8, X.shape[1] + 3) or n_sessions < min_sessions:
        return FitResult(model_name, predictors, used, int(y.size), n_sessions, float("nan"), float("nan"), y, np.full_like(y, np.nan), np.full_like(y, np.nan), work.index.to_numpy())
    pred, resid, r2 = _ols_predict(y, X)
    loos_within_session_r2 = _leave_one_session_out_within_session_r2(y, X, sessions)
    return FitResult(model_name, predictors, used, int(y.size), n_sessions, r2, loos_within_session_r2, y, pred, resid, work.index.to_numpy())


def _leave_one_session_out_within_session_r2(y: np.ndarray, X: np.ndarray, sessions: np.ndarray) -> float:
    unique = np.asarray(sorted(pd.unique(pd.Series(sessions))))
    if unique.size < 3 or y.size < max(8, X.shape[1] + 3):
        return float("nan")
    pred = np.full(y.shape[0], np.nan, dtype=np.float64)
    for sess in unique:
        test = sessions == sess
        train = ~test
        if np.count_nonzero(train) < max(8, X.shape[1] + 3) or np.count_nonzero(test) == 0:
            continue
        train_x = X[train]
        train_y = y[train]
        means = train_x.mean(axis=0) if train_x.shape[1] else np.array([])
        sds = train_x.std(axis=0) if train_x.shape[1] else np.array([])
        sds[sds <= 1e-12] = 1.0
        x_train = (train_x - means) / sds if train_x.shape[1] else train_x
        x_test = (X[test] - means) / sds if X.shape[1] else X[test]
        design_train = np.column_stack([np.ones(np.count_nonzero(train)), x_train])
        beta, *_ = np.linalg.lstsq(design_train, train_y, rcond=None)
        pred[test] = np.column_stack([np.ones(np.count_nonzero(test)), x_test]) @ beta
    ok = np.isfinite(pred)
    if np.count_nonzero(ok) < 3:
        return float("nan")
    ss_tot = float(np.sum((y[ok] - np.nanmean(y[ok])) ** 2))
    ss_res = float(np.sum((y[ok] - pred[ok]) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _fit_same_row_comparison(
    df: pd.DataFrame,
    previous_predictors: list[str],
    current_predictors: list[str],
) -> tuple[FitResult, FitResult, pd.DataFrame]:
    compare_predictors = _unique_predictors([*previous_predictors, *current_predictors])
    same_rows = _complete_case_subset(df, compare_predictors)
    previous = _fit_model(same_rows, previous_predictors, model_name="previous")
    current = _fit_model(same_rows, current_predictors, model_name="current")
    return previous, current, same_rows


def _model_sequence(
    df: pd.DataFrame,
    *,
    cohort: str,
    stratum: str,
    sequence: list[tuple[str, str, list[str]]],
) -> tuple[pd.DataFrame, dict[str, FitResult]]:
    fits: dict[str, FitResult] = {}
    rows: list[dict[str, Any]] = []
    prev_predictors: list[str] | None = None
    raw_fit: FitResult | None = None
    raw_predictors: list[str] | None = None
    for label, block_name, predictors in sequence:
        fit = _fit_model(df, predictors, model_name=label)
        fits[label] = fit
        if label in {"M1", "R1"}:
            raw_fit = fit
            raw_predictors = predictors
        if prev_predictors is not None:
            prev_same, curr_same, same_rows = _fit_same_row_comparison(df, prev_predictors, predictors)
            delta_prev = curr_same.r2 - prev_same.r2 if np.isfinite(curr_same.r2) and np.isfinite(prev_same.r2) else float("nan")
            previous_r2_same_rows = prev_same.r2
            current_r2_same_rows = curr_same.r2
            comparison_n_windows = int(same_rows.shape[0])
            comparison_n_sessions = int(same_rows[SESSION_COL].nunique()) if SESSION_COL in same_rows.columns else 0
        else:
            delta_prev = float("nan")
            previous_r2_same_rows = float("nan")
            current_r2_same_rows = float("nan")
            comparison_n_windows = 0
            comparison_n_sessions = 0
        if raw_predictors is not None:
            raw_same, curr_raw_same, raw_same_rows = _fit_same_row_comparison(df, raw_predictors, predictors)
            delta_raw = curr_raw_same.r2 - raw_same.r2 if np.isfinite(curr_raw_same.r2) and np.isfinite(raw_same.r2) else float("nan")
        else:
            delta_raw = float("nan")
        rows.append(
            {
                "cohort": cohort,
                "stratum": stratum,
                "model": label,
                "block": block_name,
                "n_windows": fit.n_windows,
                "n_sessions": fit.n_sessions,
                "n_predictors_requested": int(len(predictors)),
                "n_predictors_used": int(len(fit.predictors_used)),
                "predictors": ";".join(fit.predictors_used),
                "in_sample_r2": fit.r2,
                "leave_one_session_out_within_session_r2": fit.loos_within_session_r2,
                "delta_r2_vs_previous": delta_prev,
                "delta_r2_vs_M1": delta_raw,
                "previous_r2_same_rows": previous_r2_same_rows,
                "current_r2_same_rows": current_r2_same_rows,
                "comparison_n_windows": comparison_n_windows,
                "comparison_n_sessions": comparison_n_sessions,
                "status": "ok" if np.isfinite(fit.r2) else "insufficient_rows_or_rank",
            }
        )
        prev_predictors = predictors
    return pd.DataFrame(rows), fits


def _session_bootstrap_delta(
    df: pd.DataFrame,
    *,
    current_predictors: list[str],
    previous_predictors: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> np.ndarray:
    sessions = np.asarray(sorted(df[SESSION_COL].dropna().astype(str).unique()))
    if int(n_bootstrap) <= 0 or sessions.size < 2:
        return np.array([], dtype=np.float64)
    out = np.full(int(n_bootstrap), np.nan, dtype=np.float64)
    grouped = {sess: df[df[SESSION_COL].astype(str) == sess].copy() for sess in sessions}
    for i in range(int(n_bootstrap)):
        draw = rng.choice(sessions, size=sessions.size, replace=True)
        pieces: list[pd.DataFrame] = []
        for j, sess in enumerate(draw):
            piece = grouped[str(sess)].copy()
            piece[SESSION_COL] = f"{sess}__boot{j}"
            piece["session_id"] = piece[SESSION_COL]
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        prev, curr, _same_rows = _fit_same_row_comparison(sample, previous_predictors, current_predictors)
        if np.isfinite(prev.r2) and np.isfinite(curr.r2):
            out[i] = curr.r2 - prev.r2
    return out


def _bootstrap_model_rows(
    master: pd.DataFrame,
    model_specs: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    runnable_specs = [spec for spec in model_specs if spec.get("previous_predictors") is not None and not spec.get("df", pd.DataFrame()).empty]
    total = len(runnable_specs)
    done = 0
    for spec in model_specs:
        if spec["model"] == "M0":
            continue
        df = spec["df"]
        if df.empty or spec["previous_predictors"] is None:
            continue
        done += 1
        if int(n_bootstrap) >= 100:
            print(
                f"[raw-edge-roadblock] bootstrap {done}/{total}: "
                f"{spec['cohort']} {spec['stratum']} {spec['model']} ({int(n_bootstrap)} draws)",
                flush=True,
            )
        observed = spec["observed_delta"]
        vals = _session_bootstrap_delta(
            df,
            current_predictors=spec["predictors"],
            previous_predictors=spec["previous_predictors"],
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        finite = vals[np.isfinite(vals)]
        rows.append(
            {
                "cohort": spec["cohort"],
                "stratum": spec["stratum"],
                "model": spec["model"],
                "comparison": spec["comparison"],
                "observed_delta_r2": observed,
                "ci_low": float(np.nanpercentile(finite, 2.5)) if finite.size else float("nan"),
                "ci_high": float(np.nanpercentile(finite, 97.5)) if finite.size else float("nan"),
                "bootstrap_n_requested": int(n_bootstrap),
                "bootstrap_n_finite": int(finite.size),
            }
        )
    return pd.DataFrame(rows)


def _find_predictor(predictors: list[str], *needles: str) -> str | None:
    for predictor in predictors:
        if all(needle in predictor for needle in needles):
            return predictor
    return None


def _reduced_sequences(
    *,
    raw_predictors: list[str],
    preservation_predictors: list[str],
    hardness_predictors: list[str],
    observer_predictors: list[str],
    feature_predictors: list[str],
) -> dict[str, list[tuple[str, str, list[str]]]]:
    pixel = _find_predictor(preservation_predictors, "pixel_stability_advantage")
    pyramid_pres = _find_predictor(preservation_predictors, "pyramid_local_field_k4")
    observer_score = _find_predictor(observer_predictors, "joint_minus_zero_true_score")
    observer_margin = _find_predictor(observer_predictors, "joint_true_margin")
    feature_pyramid_k8 = _find_predictor(feature_predictors, "pyramid_local_field_k8", "joint_minus_zero_feature_gain")
    feature_gabor_k8 = _find_predictor(feature_predictors, "gabor_local_field_k8", "joint_minus_zero_feature_gain")

    preservation_added = [col for col in [pixel, pyramid_pres] if col is not None]
    observer_added = [col for col in [observer_score, observer_margin] if col is not None]
    feature_added = [col for col in [feature_pyramid_k8, feature_gabor_k8] if col is not None]

    sequences: dict[str, list[tuple[str, str, list[str]]]] = {}
    preservation_sequence: list[tuple[str, str, list[str]]] = [
        ("R0", "session_only", []),
        ("R1", "raw_edge_confidence", raw_predictors),
    ]
    if pixel is not None:
        preservation_sequence.append(("R2_pixel", "raw_plus_pixel_preservation", [*raw_predictors, pixel]))
    if preservation_added:
        preservation_sequence.append(("R2_preservation_pair", "raw_plus_two_preservation_predictors", [*raw_predictors, *preservation_added]))
    sequences["preservation_cache"] = preservation_sequence

    all_cache_sequence: list[tuple[str, str, list[str]]] = [
        ("R0", "session_only", []),
        ("R1", "raw_edge_confidence", raw_predictors),
    ]
    if pixel is not None:
        all_cache_sequence.append(("R2_pixel", "raw_plus_pixel_preservation", [*raw_predictors, pixel]))
    if hardness_predictors:
        all_cache_sequence.append(("R2h", "raw_pixel_plus_candidate_hardness", [*raw_predictors, *([pixel] if pixel else []), *hardness_predictors]))
    if observer_score is not None:
        all_cache_sequence.append(
            (
                "R3_score",
                "plus_joint_minus_zero_true_score",
                [*raw_predictors, *([pixel] if pixel else []), *hardness_predictors, observer_score],
            )
        )
    if observer_added:
        all_cache_sequence.append(
            (
                "R3_score_margin",
                "plus_observer_score_and_margin",
                [*raw_predictors, *([pixel] if pixel else []), *hardness_predictors, *observer_added],
            )
        )
    if feature_added and observer_score is not None:
        all_cache_sequence.append(
            (
                "R4_feature_pair",
                "plus_two_feature_posterior_predictors",
                [*raw_predictors, *([pixel] if pixel else []), *hardness_predictors, observer_score, *feature_added],
            )
        )
    sequences["all_block_intersection"] = all_cache_sequence

    observer_sequence: list[tuple[str, str, list[str]]] = [
        ("R0", "session_only", []),
        ("R1", "raw_edge_confidence", raw_predictors),
    ]
    if hardness_predictors:
        observer_sequence.append(("R1h", "raw_plus_candidate_hardness", [*raw_predictors, *hardness_predictors]))
    if observer_score is not None:
        observer_sequence.append(("R3_score", "plus_joint_minus_zero_true_score", [*raw_predictors, *hardness_predictors, observer_score]))
    if observer_added:
        observer_sequence.append(("R3_score_margin", "plus_observer_score_and_margin", [*raw_predictors, *hardness_predictors, *observer_added]))
    if feature_added and observer_score is not None:
        observer_sequence.append(
            (
                "R4_feature_pair",
                "plus_two_feature_posterior_predictors",
                [*raw_predictors, *hardness_predictors, observer_score, *feature_added],
            )
        )
    sequences["observer_feature_cache"] = observer_sequence
    return sequences


def _run_reduced_validation(
    master: pd.DataFrame,
    *,
    raw_predictors: list[str],
    preservation_predictors: list[str],
    hardness_predictors: list[str],
    observer_predictors: list[str],
    feature_predictors: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequences = _reduced_sequences(
        raw_predictors=raw_predictors,
        preservation_predictors=preservation_predictors,
        hardness_predictors=hardness_predictors,
        observer_predictors=observer_predictors,
        feature_predictors=feature_predictors,
    )
    masks = {
        "preservation_cache": master[preservation_predictors].notna().any(axis=1)
        if preservation_predictors
        else pd.Series(False, index=master.index),
        "observer_feature_cache": master[[*observer_predictors, *feature_predictors]].notna().any(axis=1)
        if (observer_predictors or feature_predictors)
        else pd.Series(False, index=master.index),
        "all_block_intersection": (
            (master[preservation_predictors].notna().any(axis=1) if preservation_predictors else pd.Series(False, index=master.index))
            & (
                master[[*observer_predictors, *feature_predictors]].notna().any(axis=1)
                if (observer_predictors or feature_predictors)
                else pd.Series(False, index=master.index)
            )
        ),
    }
    summaries: list[pd.DataFrame] = []
    bootstrap_specs: list[dict[str, Any]] = []
    for cohort, sequence in sequences.items():
        for stratum in ["all", "reliable_axis", "high_confidence"]:
            df = master.loc[masks[cohort] & _mask_for_stratum(master, stratum)].copy()
            summary, _fits = _model_sequence(df, cohort=cohort, stratum=stratum, sequence=sequence)
            summaries.append(summary)
            prev_predictors: list[str] | None = None
            prev_label: str | None = None
            for label, _block, predictors in sequence:
                row = summary[summary["model"] == label]
                observed = float(row["delta_r2_vs_previous"].iloc[0]) if not row.empty else float("nan")
                bootstrap_specs.append(
                    {
                        "cohort": cohort,
                        "stratum": stratum,
                        "model": label,
                        "comparison": f"{label}_vs_{prev_label}" if prev_label else "",
                        "df": df,
                        "predictors": predictors,
                        "previous_predictors": prev_predictors,
                        "observed_delta": observed,
                    }
                )
                prev_predictors = predictors
                prev_label = label
    reduced_summary = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    reduced_bootstrap = _bootstrap_model_rows(master, bootstrap_specs, rng=rng, n_bootstrap=n_bootstrap)
    return reduced_summary, reduced_bootstrap


def _single_predictor_coefficients(
    df: pd.DataFrame,
    *,
    cohort: str,
    stratum: str,
    baseline_predictors: list[str],
    predictor_block: str,
    predictors: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    def _coef_for(sample: pd.DataFrame, predictor: str) -> tuple[float, int, int, float]:
        fit = _fit_model(sample, [*baseline_predictors, predictor], model_name="single")
        base = _fit_model(sample.loc[fit.index] if fit.index.size else sample.iloc[0:0], baseline_predictors, model_name="base")
        coef = float("nan")
        work, y, X, _sessions, used = _build_design(sample, [*baseline_predictors, predictor])
        if predictor in used and y.size >= max(8, X.shape[1] + 3):
            design = np.column_stack([np.ones(y.shape[0]), X])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            coef = float(beta[1 + used.index(predictor)])
        delta = fit.r2 - base.r2 if np.isfinite(fit.r2) and np.isfinite(base.r2) else float("nan")
        return coef, fit.n_windows, fit.n_sessions, delta

    rows: list[dict[str, Any]] = []
    sessions = np.asarray(sorted(df[SESSION_COL].dropna().astype(str).unique()))
    grouped = {sess: df[df[SESSION_COL].astype(str) == sess].copy() for sess in sessions}
    for predictor in predictors:
        if predictor not in df.columns:
            continue
        coef, n_windows, n_sessions, delta = _coef_for(df, predictor)
        vals: list[float] = []
        if int(n_bootstrap) > 0 and sessions.size >= 2:
            for _ in range(int(n_bootstrap)):
                draw = rng.choice(sessions, size=sessions.size, replace=True)
                pieces: list[pd.DataFrame] = []
                for j, sess in enumerate(draw):
                    piece = grouped[str(sess)].copy()
                    piece[SESSION_COL] = f"{sess}__boot{j}"
                    piece["session_id"] = piece[SESSION_COL]
                    pieces.append(piece)
                sample = pd.concat(pieces, ignore_index=True)
                boot_coef, _n, _s, _delta = _coef_for(sample, predictor)
                vals.append(boot_coef)
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        rows.append(
            {
                "cohort": cohort,
                "stratum": stratum,
                "predictor_block": predictor_block,
                "predictor": predictor,
                "n_windows": n_windows,
                "n_sessions": n_sessions,
                "standardized_coefficient": coef,
                "standardized_coefficient_ci_low": float(np.nanpercentile(arr, 2.5)) if arr.size else float("nan"),
                "standardized_coefficient_ci_high": float(np.nanpercentile(arr, 97.5)) if arr.size else float("nan"),
                "single_predictor_delta_r2_vs_baseline": delta,
                "coefficient_bootstrap_n_finite": int(arr.size),
            }
        )
    return pd.DataFrame(rows)


def _m1_residual_table(df: pd.DataFrame, baseline_predictors: list[str]) -> pd.DataFrame:
    fit = _fit_model(df, baseline_predictors, model_name="M1")
    out = pd.DataFrame({SOURCE_COL: df.loc[fit.index, SOURCE_COL].to_numpy() if fit.index.size else []})
    if fit.index.size:
        out["m1_y_session_demeaned_z"] = fit.y
        out["m1_pred_session_demeaned_z"] = fit.pred
        out["m1_residual_z"] = fit.resid
    return out


def _spearman_summary(
    df: pd.DataFrame,
    *,
    cohort: str,
    stratum: str,
    baseline_predictors: list[str],
    predictor_block: str,
    predictors: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    residuals = _m1_residual_table(df, baseline_predictors)
    if residuals.empty:
        return pd.DataFrame()
    work = df.merge(residuals[[SOURCE_COL, "m1_residual_z"]], on=SOURCE_COL, how="inner")
    rows: list[dict[str, Any]] = []
    sessions = np.asarray(sorted(work[SESSION_COL].dropna().astype(str).unique()))
    for predictor in predictors:
        if predictor not in work.columns:
            continue
        sub = work[[SESSION_COL, predictor, "m1_residual_z"]].dropna().copy()
        if sub.shape[0] < 8 or sub[SESSION_COL].nunique() < 3:
            rho = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
            finite = 0
        else:
            x = _demean_by_session(pd.to_numeric(sub[predictor], errors="coerce").to_numpy(dtype=np.float64), sub[SESSION_COL].to_numpy())
            y = sub["m1_residual_z"].to_numpy(dtype=np.float64)
            rho = _safe_spearman(x, y)
            vals = []
            grouped = {sess: sub[sub[SESSION_COL].astype(str) == sess].copy() for sess in sessions if sess in set(sub[SESSION_COL].astype(str))}
            available = np.asarray(sorted(grouped))
            if int(n_bootstrap) > 0 and available.size >= 2:
                for _ in range(int(n_bootstrap)):
                    draw = rng.choice(available, size=available.size, replace=True)
                    pieces = []
                    for j, sess in enumerate(draw):
                        piece = grouped[str(sess)].copy()
                        piece[SESSION_COL] = f"{sess}__boot{j}"
                        pieces.append(piece)
                    boot = pd.concat(pieces, ignore_index=True)
                    bx = _demean_by_session(
                        pd.to_numeric(boot[predictor], errors="coerce").to_numpy(dtype=np.float64),
                        boot[SESSION_COL].to_numpy(),
                    )
                    by = boot["m1_residual_z"].to_numpy(dtype=np.float64)
                    vals.append(_safe_spearman(bx, by))
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            ci_low = float(np.nanpercentile(arr, 2.5)) if arr.size else float("nan")
            ci_high = float(np.nanpercentile(arr, 97.5)) if arr.size else float("nan")
            finite = int(arr.size)
        rows.append(
            {
                "cohort": cohort,
                "stratum": stratum,
                "predictor_block": predictor_block,
                "predictor": predictor,
                "spearman_rho_with_M1_residual": rho,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_windows": int(sub.shape[0]) if "sub" in locals() else 0,
                "n_sessions": int(sub[SESSION_COL].nunique()) if "sub" in locals() else 0,
                "bootstrap_n_finite": finite,
            }
        )
    return pd.DataFrame(rows)


def _session_sign_counts(
    df: pd.DataFrame,
    *,
    cohort: str,
    stratum: str,
    baseline_predictors: list[str],
    predictor_block: str,
    predictors: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        coefs: list[float] = []
        for _session, sub in df.groupby(SESSION_COL, sort=True):
            if sub.shape[0] < max(5, len(baseline_predictors) // 2):
                continue
            fit = _fit_model(sub, [*baseline_predictors, predictor], model_name="session", min_sessions=1)
            work, y, X, _sessions, used = _build_design(sub, [*baseline_predictors, predictor])
            if predictor not in used or y.size < max(4, X.shape[1] + 1):
                continue
            design = np.column_stack([np.ones(y.shape[0]), X])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            coefs.append(float(beta[1 + used.index(predictor)]))
        arr = np.asarray(coefs, dtype=np.float64)
        rows.append(
            {
                "cohort": cohort,
                "stratum": stratum,
                "predictor_block": predictor_block,
                "predictor": predictor,
                "n_sessions_with_fit": int(arr.size),
                "sessions_positive": int(np.count_nonzero(arr > 0)),
                "sessions_negative": int(np.count_nonzero(arr < 0)),
                "fraction_positive": float(np.mean(arr > 0)) if arr.size else float("nan"),
                "median_session_coefficient": float(np.nanmedian(arr)) if arr.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _mask_for_stratum(df: pd.DataFrame, stratum: str) -> pd.Series:
    if stratum == "all":
        return df[TARGET].notna()
    if stratum == "reliable_axis":
        return df["reliable_axis"].astype(bool)
    if stratum == "high_confidence":
        return df["high_confidence"].astype(bool)
    raise ValueError(f"Unknown stratum {stratum!r}")


def _figure_alignment(out_dir: Path, alignment: pd.DataFrame) -> None:
    if alignment.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    x = np.arange(alignment.shape[0])
    y = alignment["session_mean_drift_edge_cos2"].to_numpy(dtype=float)
    lo = np.maximum(0.0, y - alignment["session_bootstrap_ci_low"].to_numpy(dtype=float))
    hi = np.maximum(0.0, alignment["session_bootstrap_ci_high"].to_numpy(dtype=float) - y)
    ax.bar(x, y, color="#5d7f8c")
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(alignment["subset"], rotation=20, ha="right")
    ax.set_ylabel("session mean cos(2 drift-edge)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_raw_edge_alignment_by_confidence.png")
    plt.close(fig)


def _figure_predictor_residual(
    out_dir: Path,
    df: pd.DataFrame,
    *,
    predictor: str,
    baseline_predictors: list[str],
    filename: str,
    title: str,
) -> None:
    if predictor not in df.columns:
        return
    residuals = _m1_residual_table(df, baseline_predictors)
    if residuals.empty:
        return
    work = df.merge(residuals[[SOURCE_COL, "m1_residual_z"]], on=SOURCE_COL, how="inner")
    work = work[[predictor, "m1_residual_z", SESSION_COL]].dropna()
    if work.shape[0] < 5:
        return
    x = _demean_by_session(pd.to_numeric(work[predictor], errors="coerce").to_numpy(dtype=float), work[SESSION_COL].to_numpy())
    y = work["m1_residual_z"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    ax.scatter(x, y, s=18, alpha=0.75, color="#597b53")
    if np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
        beta = np.polyfit(x, y, deg=1)
        grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 80)
        ax.plot(grid, beta[0] * grid + beta[1], color="black", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("session-demeaned predictor")
    ax.set_ylabel("M1 residual z")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def _figure_incremental_r2(out_dir: Path, boot: pd.DataFrame) -> None:
    if boot.empty:
        return
    focus = boot.dropna(subset=["observed_delta_r2"]).copy()
    focus = focus[focus["model"].isin(["M1", "M2", "M3", "M4"])]
    if focus.empty:
        return
    focus["label"] = focus["cohort"] + "\n" + focus["stratum"] + " " + focus["model"]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.45 * focus.shape[0]), 4.2), dpi=150)
    x = np.arange(focus.shape[0])
    y = focus["observed_delta_r2"].to_numpy(dtype=float)
    ax.bar(x, y, color="#6f789b")
    if {"ci_low", "ci_high"}.issubset(focus.columns):
        lo = np.maximum(0.0, y - focus["ci_low"].to_numpy(dtype=float))
        hi = np.maximum(0.0, focus["ci_high"].to_numpy(dtype=float) - y)
        finite = np.isfinite(lo) & np.isfinite(hi)
        if np.any(finite):
            ax.errorbar(x[finite], y[finite], yerr=[lo[finite], hi[finite]], fmt="none", color="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(focus["label"], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Delta R2 vs previous block")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_incremental_r2_by_block.png")
    plt.close(fig)


def _figure_session_signs(out_dir: Path, signs: pd.DataFrame) -> None:
    if signs.empty:
        return
    focus = signs.dropna(subset=["fraction_positive"]).copy()
    focus = focus.sort_values(["cohort", "stratum", "fraction_positive"], ascending=[True, True, False]).head(30)
    if focus.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, max(4.0, 0.22 * focus.shape[0])), dpi=150)
    y = np.arange(focus.shape[0])
    ax.barh(y, focus["fraction_positive"], color="#8d6d58")
    ax.axvline(0.5, color="black", linewidth=0.8)
    labels = [f"{r.predictor_block}:{r.predictor[:42]}" for r in focus.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("fraction positive across sessions")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_session_delta_r2_signs.png")
    plt.close(fig)


def _write_join_qc_markdown(out_dir: Path, qc: pd.DataFrame) -> None:
    lines = ["# Join QC", ""]
    if qc.empty:
        lines.append("No join QC rows were produced.")
    else:
        for _, row in qc.iterrows():
            name = row.get("join", "join")
            lines.append(f"## {name}")
            for col, value in row.items():
                if col == "join" or pd.isna(value):
                    continue
                lines.append(f"- `{col}`: `{value}`")
            lines.append("")
    (out_dir / "join_qc.md").write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    out_dir: Path,
    *,
    alignment: pd.DataFrame,
    join_qc: pd.DataFrame,
    model_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    reduced_model_summary: pd.DataFrame,
    reduced_bootstrap: pd.DataFrame,
    spearman: pd.DataFrame,
    master: pd.DataFrame,
    feature_axis_contrasts: pd.DataFrame,
    primary_scale: float,
) -> None:
    def _fmt_ci(row: pd.Series, low: str = "ci_low", high: str = "ci_high") -> str:
        if low in row and high in row and np.isfinite(row[low]) and np.isfinite(row[high]):
            return f" CI [{row[low]:+.4f}, {row[high]:+.4f}]"
        return ""

    raw = alignment.set_index("subset") if not alignment.empty else pd.DataFrame()
    obs_n = int(master.filter(regex=r"^observer_scale").notna().any(axis=1).sum())
    feat_n = int(master.filter(regex=r"^feature_scale").notna().any(axis=1).sum())
    preservation_cols = [col for col in PRESERVATION_PREDICTORS if col in master.columns]
    preservation_mask = (
        master[preservation_cols].notna().any(axis=1)
        if preservation_cols
        else pd.Series(False, index=master.index)
    )
    pres_n = int(preservation_mask.sum())
    overlap_cols = [
        preservation_mask,
        master.filter(regex=r"^observer_scale").notna().any(axis=1) if obs_n else pd.Series(False, index=master.index),
        master.filter(regex=r"^feature_scale").notna().any(axis=1) if feat_n else pd.Series(False, index=master.index),
    ]
    all_overlap = int((overlap_cols[0] & overlap_cols[1] & overlap_cols[2]).sum())
    lines = [
        "# BackImage Raw Edge Roadblock Residual Adjudication",
        "",
        "## 1. Join Interpretability",
        "",
        f"- Raw reviewed windows: `{master.shape[0]}` across `{master[SESSION_COL].nunique()}` sessions.",
        f"- Windows with preservation predictors: `{pres_n}`.",
        f"- Windows with observer-axis predictors: `{obs_n}`.",
        f"- Windows with feature-posterior predictors: `{feat_n}`.",
        f"- Windows with preservation, observer, and feature predictors together: `{all_overlap}`.",
        "",
    ]
    if all_overlap < 20:
        lines.append(
            "The preservation and observer caches have too little overlap for a literal M2 -> M3 -> M4 all-block ladder to be interpreted. The report therefore treats preservation and observer/feature residual tests as separate cache-cohort adjudications, with raw-edge controls fitted first in each cohort."
        )
        lines.append("")
    lines.append("All reported block Delta R2 values refit the previous and current models on the same complete-case rows for that comparison.")
    lines.append("")

    lines.extend(["## 2. Raw Edge Alignment", ""])
    if not raw.empty:
        for subset in ["all", "reliable_axis", "high_confidence"]:
            if subset in raw.index:
                r = raw.loc[subset]
                lines.append(
                    f"- `{subset}`: session mean `{r['session_mean_drift_edge_cos2']:+.4f}`"
                    f" [{r['session_bootstrap_ci_low']:+.4f}, {r['session_bootstrap_ci_high']:+.4f}],"
                    f" n=`{int(r['n_windows'])}`, positive sessions `{int(r['positive_sessions'])}/{int(r['n_sessions'])}`."
                )
    lines.extend(["", "## 3. Preservation Residuals", ""])
    pres = bootstrap[(bootstrap["cohort"] == "preservation_cache") & (bootstrap["model"] == "M2")] if not bootstrap.empty else pd.DataFrame()
    if pres.empty:
        lines.append("- No interpretable preservation block fit was available.")
    else:
        for _, row in pres.iterrows():
            lines.append(f"- `{row['stratum']}` M2 over M1: Delta R2 `{row['observed_delta_r2']:+.4f}`{_fmt_ci(row)}.")

    lines.extend(["", "## 4. Joint-Posterior Axis Residuals", ""])
    observer_report_cohort = "all_block_intersection" if all_overlap >= 20 else "observer_feature_cache"
    obs = bootstrap[(bootstrap["cohort"] == observer_report_cohort) & (bootstrap["model"] == "M3")] if not bootstrap.empty else pd.DataFrame()
    if obs.empty:
        lines.append("- No interpretable observer block fit was available.")
    else:
        for _, row in obs.iterrows():
            lines.append(
                f"- `{row['stratum']}` primary scale `{primary_scale:g}` M3 over previous nuisance block: Delta R2 `{row['observed_delta_r2']:+.4f}`{_fmt_ci(row)}."
            )

    lines.extend(["", "## 5. Feature-Posterior Residuals", ""])
    feat = bootstrap[(bootstrap["cohort"] == observer_report_cohort) & (bootstrap["model"] == "M4")] if not bootstrap.empty else pd.DataFrame()
    if feat.empty:
        lines.append("- No interpretable feature-posterior block fit was available.")
    else:
        for _, row in feat.iterrows():
            lines.append(f"- `{row['stratum']}` M4 over M3: Delta R2 `{row['observed_delta_r2']:+.4f}`{_fmt_ci(row)}.")

    lines.extend(["", "## 6. Stability Across Subsets", ""])
    stable_rows = bootstrap[
        bootstrap["stratum"].isin(["reliable_axis", "high_confidence"]) & bootstrap["model"].isin(["M2", "M3", "M4"])
    ] if not bootstrap.empty else pd.DataFrame()
    positive_stable = stable_rows[(stable_rows["observed_delta_r2"] > 0) & (stable_rows["ci_low"] > 0)] if not stable_rows.empty and "ci_low" in stable_rows else pd.DataFrame()
    if positive_stable.empty:
        lines.append("- No model-derived block has a positive session-bootstrap CI excluding zero in the reliable-axis or high-confidence subsets.")
    else:
        for _, row in positive_stable.iterrows():
            lines.append(f"- `{row['cohort']}/{row['stratum']}/{row['model']}` clears zero with Delta R2 `{row['observed_delta_r2']:+.4f}`{_fmt_ci(row)}.")

    lines.extend(["", "## 7. Decision", ""])
    observer_success = not obs.empty and bool(((obs["observed_delta_r2"] > 0) & (obs["ci_low"] > 0)).any())
    feature_success = not feat.empty and bool(((feat["observed_delta_r2"] > 0) & (feat["ci_low"] > 0)).any())
    if observer_success or feature_success:
        if all_overlap >= 20:
            lines.append(
                "Model-derived observer/preservation variables explain positive session-bootstrap residual variation in observed drift-edge alignment beyond raw local edge geometry on the shared all-cache cohort. This supports a mechanistic bridge candidate between BackImage along-contour drift and trajectory-aware V1-twin sampling."
            )
        else:
            lines.append(
                "Model-derived observer/preservation variables explain positive session-bootstrap residual variation in observed drift-edge alignment beyond raw local edge geometry in at least one cache cohort. Because cache overlap is sparse, this should be treated as a mechanistic bridge candidate rather than a fully promoted all-block result."
            )
        cv_rows = model_summary[
            (model_summary["cohort"] == observer_report_cohort)
            & (model_summary["stratum"] == "all")
            & (model_summary["model"].isin(["M3", "M4"]))
        ]
        if not cv_rows.empty and (pd.to_numeric(cv_rows["leave_one_session_out_within_session_r2"], errors="coerce") <= 0).any():
            lines.append(
                "However, the high-dimensional M3/M4 leave-one-session-out within-session R2 is not positive in this first pass. This diagnostic uses session-demeaned held-out targets and predictors, so it is a residual-transfer check rather than literal new-session prediction."
            )
    else:
        lines.append(
            "Raw edge geometry and local preservation remain the best explanation of observed BackImage drift axes. The joint-posterior observer is still evidence for trajectory-aware feature recovery, but not yet an explanation of the biological along-contour drift bias."
        )

    if not feature_axis_contrasts.empty:
        lines.extend(["", "## Feature-Axis Cache Context", ""])
        cols = [
            "observation_scale",
            "latent",
            "requested_k",
            "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal",
            "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_low",
            "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_high",
        ]
        cols = [col for col in cols if col in feature_axis_contrasts.columns]
        for _, row in feature_axis_contrasts[cols].head(12).iterrows():
            val = row.get("mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal", np.nan)
            lo = row.get("mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_low", np.nan)
            hi = row.get("mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal_ci_high", np.nan)
            lines.append(
                f"- scale `{row.get('observation_scale')}`, `{row.get('latent')}` k=`{row.get('requested_k')}`: joint-minus-zero axis contrast `{val:+.4f}`"
                + (f" [{lo:+.4f}, {hi:+.4f}]." if np.isfinite(lo) and np.isfinite(hi) else ".")
            )

    lines.extend(["", "## Reduced-Model Validation", ""])
    if reduced_model_summary.empty:
        lines.append("- Reduced-model validation was not available.")
    else:
        focus = reduced_model_summary[
            (reduced_model_summary["cohort"] == "all_block_intersection")
            & (reduced_model_summary["stratum"] == "all")
            & (reduced_model_summary["model"].isin(["R2_pixel", "R3_score", "R3_score_margin", "R4_feature_pair"]))
        ].copy()
        if not focus.empty and not reduced_bootstrap.empty:
            focus = focus.merge(
                reduced_bootstrap[
                    ["cohort", "stratum", "model", "observed_delta_r2", "ci_low", "ci_high", "bootstrap_n_finite"]
                ],
                on=["cohort", "stratum", "model"],
                how="left",
            )
        if focus.empty:
            lines.append("- No all-cache reduced model rows were available.")
        else:
            for _, row in focus.iterrows():
                delta = row.get("observed_delta_r2", row.get("delta_r2_vs_previous", np.nan))
                lines.append(
                    f"- `{row['model']}` (`{row['block']}`): in-sample R2 `{row['in_sample_r2']:+.4f}`, "
                    f"leave-one-session-out within-session R2 `{row['leave_one_session_out_within_session_r2']:+.4f}`, "
                    f"Delta R2 `{delta:+.4f}`{_fmt_ci(row)}."
                )
        cv_positive = reduced_model_summary[
            (reduced_model_summary["cohort"] == "all_block_intersection")
            & (reduced_model_summary["stratum"] == "all")
            & (pd.to_numeric(reduced_model_summary["leave_one_session_out_within_session_r2"], errors="coerce") > 0)
            & (reduced_model_summary["model"].str.startswith("R3", na=False))
        ]
        if cv_positive.empty:
            lines.append(
                "Reduced observer models also do not yet clear the leave-one-session-out within-session residual-transfer check on the all-cache cohort; this keeps the current result in the bridge-candidate category."
            )
        else:
            best = cv_positive.sort_values("leave_one_session_out_within_session_r2", ascending=False).iloc[0]
            lines.append(
                f"Best reduced observer residual-transfer row is `{best['model']}` with leave-one-session-out within-session R2 `{best['leave_one_session_out_within_session_r2']:+.4f}`."
            )

    lines.extend(
        [
            "",
            "Generated files include `raw_edge_residual_master_table.csv`, `model_block_summary.csv`, `incremental_r2_session_bootstrap.csv`, `reduced_model_summary.csv`, `reduced_model_session_bootstrap.csv`, `standardized_coefficients.csv`, `spearman_predictor_summary.csv`, and the diagnostic figures in this directory.",
            "",
        ]
    )
    (out_dir / "raw_edge_roadblock_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.random_seed))
    include_scales = _parse_float_list(args.include_scales)

    raw = _read_raw_windows(Path(args.windows_csv))
    alignment = _summarize_alignment(raw, rng=rng, n_bootstrap=int(args.session_bootstrap_n))
    raw.to_csv(out_dir / "joined_raw_edge_baseline_table.csv", index=False)

    preservation, preservation_qc, _preservation_audit = _load_preservation_tables(
        raw,
        Path(args.stability_window_csv),
        Path(args.feature_preservation_window_csv),
    )
    preservation.to_csv(out_dir / "joined_preservation_table.csv", index=False)

    observer_delta, observer_qc = _load_observer_axis_deltas(Path(args.observer_trials_csv), include_scales=include_scales)
    observer_delta.to_csv(out_dir / "observer_axis_delta_by_window.csv", index=False)
    observer_wide, observer_predictors, hardness_predictors, observer_dict = _wide_observer_predictors(
        observer_delta,
        primary_scale=float(args.primary_scale),
        primary_likelihood_scale=float(args.primary_likelihood_scale),
    )

    feature_delta, feature_qc = _load_feature_posterior_deltas(Path(args.feature_posterior_trials_csv), include_scales=include_scales)
    feature_delta.to_csv(out_dir / "feature_posterior_axis_delta_by_window.csv", index=False)
    feature_wide, feature_predictors, feature_dict = _wide_feature_predictors(
        feature_delta,
        primary_scale=float(args.primary_scale),
        primary_likelihood_scale=float(args.primary_likelihood_scale),
    )

    preservation_master_cols = [SOURCE_COL, *[col for col in PRESERVATION_PREDICTORS if col in preservation.columns]]
    master = raw.merge(preservation[preservation_master_cols], on=SOURCE_COL, how="left", validate="one_to_one")
    master = master.merge(observer_wide, on=SOURCE_COL, how="left", validate="one_to_one")
    master = master.merge(feature_wide, on=SOURCE_COL, how="left", validate="one_to_one")
    master.to_csv(out_dir / "raw_edge_residual_master_table.csv", index=False)

    join_qc = pd.concat([preservation_qc, observer_qc, feature_qc], ignore_index=True, sort=False)
    if not observer_delta.empty:
        preservation_cols = [col for col in PRESERVATION_PREDICTORS if col in preservation.columns]
        preservation_mask = (
            preservation[preservation_cols].notna().any(axis=1)
            if preservation_cols
            else pd.Series(False, index=preservation.index)
        )
        preservation_ids = set(pd.to_numeric(preservation.loc[preservation_mask, SOURCE_COL], errors="coerce").dropna().astype(int))
        observer_ids = set(pd.to_numeric(observer_delta[SOURCE_COL], errors="coerce").dropna().astype(int))
        feature_ids = set(pd.to_numeric(feature_delta[SOURCE_COL], errors="coerce").dropna().astype(int)) if not feature_delta.empty else set()
        join_qc = pd.concat(
            [
                join_qc,
                pd.DataFrame(
                    [
                        {
                            "join": "cache_overlap",
                            "preservation_unique_windows": len(preservation_ids),
                            "observer_unique_windows": len(observer_ids),
                            "feature_unique_windows": len(feature_ids),
                            "preservation_observer_overlap": len(preservation_ids & observer_ids),
                            "observer_feature_overlap": len(observer_ids & feature_ids),
                            "all_three_overlap": len(preservation_ids & observer_ids & feature_ids),
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    join_qc.to_csv(out_dir / "join_qc.csv", index=False)
    _write_join_qc_markdown(out_dir, join_qc)

    predictor_dictionary = pd.DataFrame(
        [
            *[
                {"predictor": col, "block": "raw_edge_confidence", "source": "backimage_image_fem_windows", "description": col}
                for col in RAW_PREDICTORS
                if col in master.columns
            ],
            *[
                {"predictor": col, "block": "preservation", "source": "twin_stability_metric_by_window", "description": col}
                for col in PRESERVATION_PREDICTORS
                if col in master.columns
            ],
            *observer_dict,
            *feature_dict,
        ]
    )
    predictor_dictionary.to_csv(out_dir / "predictor_dictionary.csv", index=False)

    raw_predictors = _numeric_predictors(master, RAW_PREDICTORS)
    preservation_predictors = _numeric_predictors(master, PRESERVATION_PREDICTORS)
    observer_predictors = _numeric_predictors(master, observer_predictors)
    hardness_predictors = _numeric_predictors(master, hardness_predictors)
    feature_predictors = _numeric_predictors(master, feature_predictors)

    model_tables: list[pd.DataFrame] = []
    stratified_tables: list[pd.DataFrame] = []
    coef_tables: list[pd.DataFrame] = []
    spearman_tables: list[pd.DataFrame] = []
    sign_tables: list[pd.DataFrame] = []
    bootstrap_specs: list[dict[str, Any]] = []

    cohort_defs = [
        {
            "cohort": "all_windows",
            "mask": master[TARGET].notna(),
            "sequence": [
                ("M0", "session_only", []),
                ("M1", "raw_edge_confidence", raw_predictors),
            ],
        },
        {
            "cohort": "preservation_cache",
            "mask": master[preservation_predictors].notna().any(axis=1) if preservation_predictors else pd.Series(False, index=master.index),
            "sequence": [
                ("M0", "session_only", []),
                ("M1", "raw_edge_confidence", raw_predictors),
                ("M2", "raw_plus_preservation", [*raw_predictors, *preservation_predictors]),
            ],
        },
        {
            "cohort": "observer_feature_cache",
            "mask": master[[*observer_predictors, *feature_predictors]].notna().any(axis=1)
            if (observer_predictors or feature_predictors)
            else pd.Series(False, index=master.index),
            "sequence": [
                ("M0", "session_only", []),
                ("M1", "raw_edge_confidence", raw_predictors),
                ("M1h", "raw_plus_candidate_hardness", [*raw_predictors, *hardness_predictors]),
                ("M3", "raw_hardness_plus_joint_posterior", [*raw_predictors, *hardness_predictors, *observer_predictors]),
                (
                    "M4",
                    "raw_hardness_joint_plus_feature_posterior",
                    [*raw_predictors, *hardness_predictors, *observer_predictors, *feature_predictors],
                ),
            ],
        },
        {
            "cohort": "all_block_intersection",
            "mask": (
                (master[preservation_predictors].notna().any(axis=1) if preservation_predictors else pd.Series(False, index=master.index))
                & (
                    master[[*observer_predictors, *feature_predictors]].notna().any(axis=1)
                    if (observer_predictors or feature_predictors)
                    else pd.Series(False, index=master.index)
                )
            ),
            "sequence": [
                ("M0", "session_only", []),
                ("M1", "raw_edge_confidence", raw_predictors),
                ("M2", "raw_plus_preservation", [*raw_predictors, *preservation_predictors]),
                ("M2h", "M2_plus_candidate_hardness", [*raw_predictors, *preservation_predictors, *hardness_predictors]),
                ("M3", "M2h_plus_joint_posterior", [*raw_predictors, *preservation_predictors, *hardness_predictors, *observer_predictors]),
                (
                    "M4",
                    "M3_plus_feature_posterior",
                    [*raw_predictors, *preservation_predictors, *hardness_predictors, *observer_predictors, *feature_predictors],
                ),
            ],
        },
    ]

    for cohort_def in cohort_defs:
        for stratum in ["all", "reliable_axis", "high_confidence"]:
            mask = cohort_def["mask"] & _mask_for_stratum(master, stratum)
            df = master.loc[mask].copy()
            summary, fits = _model_sequence(
                df,
                cohort=cohort_def["cohort"],
                stratum=stratum,
                sequence=cohort_def["sequence"],
            )
            model_tables.append(summary)
            stratified_tables.append(summary)
            prev_predictors: list[str] | None = None
            prev_label: str | None = None
            for label, _block, predictors in cohort_def["sequence"]:
                row = summary[summary["model"] == label]
                observed = float(row["delta_r2_vs_previous"].iloc[0]) if not row.empty else float("nan")
                bootstrap_specs.append(
                    {
                        "cohort": cohort_def["cohort"],
                        "stratum": stratum,
                        "model": label,
                        "comparison": f"{label}_vs_{prev_label}" if prev_label else "",
                        "df": df,
                        "predictors": predictors,
                        "previous_predictors": prev_predictors,
                        "observed_delta": observed,
                    }
                )
                prev_predictors = predictors
                prev_label = label

            if cohort_def["cohort"] == "preservation_cache":
                coef_tables.append(
                    _single_predictor_coefficients(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=raw_predictors,
                        predictor_block="preservation",
                        predictors=preservation_predictors,
                        rng=rng,
                        n_bootstrap=int(args.predictor_bootstrap_n),
                    )
                )
                spearman_tables.append(
                    _spearman_summary(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=raw_predictors,
                        predictor_block="preservation",
                        predictors=preservation_predictors,
                        rng=rng,
                        n_bootstrap=int(args.predictor_bootstrap_n),
                    )
                )
                sign_tables.append(
                    _session_sign_counts(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=raw_predictors,
                        predictor_block="preservation",
                        predictors=preservation_predictors,
                    )
                )
            if cohort_def["cohort"] == "observer_feature_cache":
                observer_baseline = [*raw_predictors, *hardness_predictors]
                coef_tables.append(
                    _single_predictor_coefficients(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=observer_baseline,
                        predictor_block="observer",
                        predictors=observer_predictors,
                        rng=rng,
                        n_bootstrap=int(args.predictor_bootstrap_n),
                    )
                )
                coef_tables.append(
                    _single_predictor_coefficients(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=[*observer_baseline, *observer_predictors],
                        predictor_block="feature_posterior",
                        predictors=feature_predictors,
                        rng=rng,
                        n_bootstrap=int(args.predictor_bootstrap_n),
                    )
                )
                spearman_tables.append(
                    _spearman_summary(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=observer_baseline,
                        predictor_block="observer",
                        predictors=observer_predictors,
                        rng=rng,
                        n_bootstrap=int(args.predictor_bootstrap_n),
                    )
                )
                spearman_tables.append(
                    _spearman_summary(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=[*observer_baseline, *observer_predictors],
                        predictor_block="feature_posterior",
                        predictors=feature_predictors,
                        rng=rng,
                        n_bootstrap=int(args.predictor_bootstrap_n),
                    )
                )
                sign_tables.append(
                    _session_sign_counts(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=observer_baseline,
                        predictor_block="observer",
                        predictors=observer_predictors,
                    )
                )
                sign_tables.append(
                    _session_sign_counts(
                        df,
                        cohort=cohort_def["cohort"],
                        stratum=stratum,
                        baseline_predictors=[*observer_baseline, *observer_predictors],
                        predictor_block="feature_posterior",
                        predictors=feature_predictors,
                    )
                )

    model_summary = pd.concat(model_tables, ignore_index=True, sort=False) if model_tables else pd.DataFrame()
    model_summary.to_csv(out_dir / "model_block_summary.csv", index=False)
    stratified_model_summary = pd.concat(stratified_tables, ignore_index=True, sort=False) if stratified_tables else pd.DataFrame()
    stratified_model_summary.to_csv(out_dir / "stratified_model_summary.csv", index=False)

    bootstrap = _bootstrap_model_rows(master, bootstrap_specs, rng=rng, n_bootstrap=int(args.session_bootstrap_n))
    bootstrap.to_csv(out_dir / "incremental_r2_session_bootstrap.csv", index=False)

    reduced_summary, reduced_bootstrap = _run_reduced_validation(
        master,
        raw_predictors=raw_predictors,
        preservation_predictors=preservation_predictors,
        hardness_predictors=hardness_predictors,
        observer_predictors=observer_predictors,
        feature_predictors=feature_predictors,
        rng=rng,
        n_bootstrap=int(args.reduced_bootstrap_n),
    )
    reduced_summary.to_csv(out_dir / "reduced_model_summary.csv", index=False)
    reduced_bootstrap.to_csv(out_dir / "reduced_model_session_bootstrap.csv", index=False)

    coefficients = pd.concat([table for table in coef_tables if table is not None and not table.empty], ignore_index=True, sort=False) if coef_tables else pd.DataFrame()
    coefficients.to_csv(out_dir / "standardized_coefficients.csv", index=False)
    spearman = pd.concat([table for table in spearman_tables if table is not None and not table.empty], ignore_index=True, sort=False) if spearman_tables else pd.DataFrame()
    spearman.to_csv(out_dir / "spearman_predictor_summary.csv", index=False)
    signs = pd.concat([table for table in sign_tables if table is not None and not table.empty], ignore_index=True, sort=False) if sign_tables else pd.DataFrame()
    signs.to_csv(out_dir / "session_predictor_sign_counts.csv", index=False)

    feature_axis_contrasts = pd.DataFrame()
    if Path(args.feature_axis_contrasts_csv).exists():
        feature_axis_contrasts = pd.read_csv(args.feature_axis_contrasts_csv)
        feature_axis_contrasts.to_csv(out_dir / "feature_axis_contrasts_context.csv", index=False)

    run_metadata = {
        "windows_csv": str(args.windows_csv),
        "stability_window_csv": str(args.stability_window_csv),
        "feature_preservation_window_csv": str(args.feature_preservation_window_csv),
        "observer_trials_csv": str(args.observer_trials_csv),
        "feature_posterior_trials_csv": str(args.feature_posterior_trials_csv),
        "feature_axis_contrasts_csv": str(args.feature_axis_contrasts_csv),
        "out_dir": str(out_dir),
        "primary_scale": float(args.primary_scale),
        "include_scales": include_scales,
        "primary_likelihood_scale": float(args.primary_likelihood_scale),
        "session_bootstrap_n": int(args.session_bootstrap_n),
        "predictor_bootstrap_n": int(args.predictor_bootstrap_n),
        "reduced_bootstrap_n": int(args.reduced_bootstrap_n),
        "random_seed": int(args.random_seed),
        "raw_predictors": raw_predictors,
        "preservation_predictors": preservation_predictors,
        "observer_predictors": observer_predictors,
        "candidate_hardness_predictors": hardness_predictors,
        "feature_predictors": feature_predictors,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _figure_alignment(out_dir, alignment)
    preservation_focus = next((col for col in ["full_cov_whitened_mse_stability_advantage", "pixel_stability_advantage"] if col in master.columns), "")
    if preservation_focus:
        _figure_predictor_residual(
            out_dir,
            master.loc[master[preservation_focus].notna()].copy(),
            predictor=preservation_focus,
            baseline_predictors=raw_predictors,
            filename="fig_preservation_predicts_residual_alignment.png",
            title=preservation_focus,
        )
    observer_focus = next((col for col in observer_predictors if "joint_minus_zero_true_score" in col), observer_predictors[0] if observer_predictors else "")
    if observer_focus:
        _figure_predictor_residual(
            out_dir,
            master.loc[master[observer_focus].notna()].copy(),
            predictor=observer_focus,
            baseline_predictors=[*raw_predictors, *hardness_predictors],
            filename="fig_observer_axis_delta_predicts_residual_alignment.png",
            title=observer_focus,
        )
    _figure_incremental_r2(out_dir, bootstrap)
    _figure_session_signs(out_dir, signs)

    alignment.to_csv(out_dir / "raw_edge_alignment_summary.csv", index=False)
    _write_report(
        out_dir,
        alignment=alignment,
        join_qc=join_qc,
        model_summary=model_summary,
        bootstrap=bootstrap,
        reduced_model_summary=reduced_summary,
        reduced_bootstrap=reduced_bootstrap,
        spearman=spearman,
        master=master,
        feature_axis_contrasts=feature_axis_contrasts,
        primary_scale=float(args.primary_scale),
    )
    print(f"Wrote BackImage raw-edge roadblock adjudication to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-csv", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--stability-window-csv", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--feature-preservation-window-csv", type=Path, default=DEFAULT_FEATURE_PRESERVATION)
    parser.add_argument("--observer-trials-csv", type=Path, default=DEFAULT_OBSERVER)
    parser.add_argument("--feature-posterior-trials-csv", type=Path, default=DEFAULT_FEATURE_POSTERIOR)
    parser.add_argument("--feature-axis-contrasts-csv", type=Path, default=DEFAULT_FEATURE_AXIS_CONTRASTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--primary-scale", type=float, default=1.0)
    parser.add_argument("--include-scales", default="0.5,1.0,2.0")
    parser.add_argument("--primary-likelihood-scale", type=float, default=1.0)
    parser.add_argument("--session-bootstrap-n", type=int, default=200)
    parser.add_argument("--predictor-bootstrap-n", type=int, default=20)
    parser.add_argument("--reduced-bootstrap-n", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
