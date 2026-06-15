"""Posthoc controls for BackImage edge-parallel stability alignment.

This script works from an existing ``run_backimage_edge_parallel_stability_screen``
output directory. It tests whether the relationship between edge-parallel
stability advantage and observed drift-edge axis alignment survives controls for
image orientation coherence, drift anisotropy, and local contrast/edge strength.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_STABILITY_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
)
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_stability_wrong_direction_controls"
)


MODEL_SPECS = (
    ("model0_unadjusted", []),
    ("model1_coherence", ["image_orientation_coherence"]),
    ("model2_coherence_anisotropy", ["image_orientation_coherence", "drift_anisotropy"]),
    (
        "model3_full_low_level",
        ["image_orientation_coherence", "drift_anisotropy", "edge_strength"],
    ),
)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
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


def _design_from_controls(df: pd.DataFrame, controls: list[str], *, within_session: bool) -> np.ndarray:
    if not controls:
        return np.empty((df.shape[0], 0), dtype=np.float64)
    cols = []
    sessions = df["session_id"].to_numpy()
    for col in controls:
        values = df[col].to_numpy(dtype=np.float64)
        if within_session:
            values = _demean_by_session(values, sessions)
        cols.append(values)
    return np.column_stack(cols).astype(np.float64)


def _fit_standardized_ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    ok = _finite_mask(y, X)
    y = y[ok]
    X = X[ok]
    if y.size <= X.shape[1] + 2:
        return {"coef": np.full(X.shape[1], np.nan), "r2": float("nan"), "resid": np.full(y.size, np.nan), "n": int(y.size)}
    y_z = _zscore(y)
    X_z = np.column_stack([_zscore(X[:, j]) for j in range(X.shape[1])]) if X.shape[1] else np.empty((y.size, 0))
    design = np.column_stack([np.ones(y.size), X_z])
    beta, *_ = np.linalg.lstsq(design, y_z, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y_z - pred) ** 2))
    ss_tot = float(np.sum((y_z - np.mean(y_z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"coef": beta[1:], "r2": float(r2), "resid": y_z - pred, "n": int(y.size)}


def _fit_model(df: pd.DataFrame, predictor: str, controls: list[str]) -> dict[str, float]:
    sessions = df["session_id"].to_numpy()
    y = _demean_by_session(df["drift_edge_align_signed"].to_numpy(dtype=np.float64), sessions)
    pred = _demean_by_session(df[predictor].to_numpy(dtype=np.float64), sessions)
    X_control = _design_from_controls(df, controls, within_session=True)
    X_full = np.column_stack([X_control, pred])
    control_fit = _fit_standardized_ols(y, X_control)
    full_fit = _fit_standardized_ols(y, X_full)
    return {
        "coef_stability": float(full_fit["coef"][-1]) if len(full_fit["coef"]) else float("nan"),
        "full_r2": float(full_fit["r2"]),
        "control_r2": float(control_fit["r2"]),
        "incremental_r2": float(full_fit["r2"] - control_fit["r2"]),
        "n_windows": int(full_fit["n"]),
        "n_sessions": int(pd.Series(sessions[_finite_mask(y, X_full)]).nunique()) if df.shape[0] else 0,
    }


def _session_bootstrap_model(
    df: pd.DataFrame,
    predictor: str,
    controls: list[str],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    sessions = np.asarray(sorted(df["session_id"].dropna().unique()))
    if sessions.size < 2 or int(n_bootstrap) <= 0:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    coefs = []
    for _ in range(int(n_bootstrap)):
        draw = rng.choice(sessions, size=sessions.size, replace=True)
        pieces = []
        for j, sess in enumerate(draw):
            block = df[df["session_id"] == sess].copy()
            block["session_id"] = f"{sess}__boot{j}"
            pieces.append(block)
        sample = pd.concat(pieces, ignore_index=True)
        coefs.append(_fit_model(sample, predictor, controls)["coef_stability"])
    arr = np.asarray(coefs, dtype=np.float64)
    return {"ci_low": float(np.nanpercentile(arr, 2.5)), "ci_high": float(np.nanpercentile(arr, 97.5))}


def _permute_model(
    df: pd.DataFrame,
    predictor: str,
    controls: list[str],
    observed_coef: float,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, float]:
    if int(n_permutations) <= 0:
        return {"perm_p_two_sided": float("nan"), "perm_p_coef_le_observed": float("nan"), "perm_p_coef_ge_observed": float("nan")}
    null = np.empty(int(n_permutations), dtype=np.float64)
    for j in range(int(n_permutations)):
        shuf = df.copy()
        values = shuf[predictor].to_numpy(dtype=np.float64).copy()
        sessions = shuf["session_id"].to_numpy()
        for sess in np.unique(sessions):
            idx = np.flatnonzero(sessions == sess)
            if idx.size > 1:
                values[idx] = rng.permutation(values[idx])
        shuf[predictor] = values
        null[j] = _fit_model(shuf, predictor, controls)["coef_stability"]
    return {
        "perm_p_two_sided": float((1.0 + np.count_nonzero(np.abs(null) >= abs(float(observed_coef)))) / (null.size + 1.0)),
        "perm_p_coef_le_observed": float((1.0 + np.count_nonzero(null <= float(observed_coef))) / (null.size + 1.0)),
        "perm_p_coef_ge_observed": float((1.0 + np.count_nonzero(null >= float(observed_coef))) / (null.size + 1.0)),
    }


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    ok = _finite_mask(values, controls)
    resid = np.full(values.shape[0], np.nan, dtype=np.float64)
    if np.count_nonzero(ok) <= controls.shape[1] + 2:
        return resid
    fit = _fit_standardized_ols(values[ok], controls[ok])
    resid[ok] = fit["resid"]
    return resid


def _partial_corr(df: pd.DataFrame, predictor: str, controls: list[str]) -> dict[str, Any]:
    sessions = df["session_id"].to_numpy()
    y = _demean_by_session(df["drift_edge_align_signed"].to_numpy(dtype=np.float64), sessions)
    p = _demean_by_session(df[predictor].to_numpy(dtype=np.float64), sessions)
    X = _design_from_controls(df, controls, within_session=True)
    y_resid = _residualize(y, X)
    p_resid = _residualize(p, X)
    ok = _finite_mask(y_resid, p_resid)
    r = float(np.corrcoef(y_resid[ok], p_resid[ok])[0, 1]) if np.count_nonzero(ok) > 3 else float("nan")
    return {"partial_r": r, "y_resid": y_resid, "predictor_resid": p_resid, "n_windows": int(np.count_nonzero(ok))}


def _bootstrap_partial_corr(
    df: pd.DataFrame,
    predictor: str,
    controls: list[str],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    sessions = np.asarray(sorted(df["session_id"].dropna().unique()))
    if sessions.size < 2 or int(n_bootstrap) <= 0:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    vals = []
    for _ in range(int(n_bootstrap)):
        draw = rng.choice(sessions, size=sessions.size, replace=True)
        pieces = []
        for j, sess in enumerate(draw):
            block = df[df["session_id"] == sess].copy()
            block["session_id"] = f"{sess}__boot{j}"
            pieces.append(block)
        vals.append(_partial_corr(pd.concat(pieces, ignore_index=True), predictor, controls)["partial_r"])
    arr = np.asarray(vals, dtype=np.float64)
    return {"ci_low": float(np.nanpercentile(arr, 2.5)), "ci_high": float(np.nanpercentile(arr, 97.5))}


def _permute_partial_corr(
    df: pd.DataFrame,
    predictor: str,
    controls: list[str],
    observed_r: float,
    *,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, float]:
    if int(n_permutations) <= 0:
        return {"perm_p_two_sided": float("nan"), "perm_p_r_le_observed": float("nan"), "perm_p_r_ge_observed": float("nan")}
    null = np.empty(int(n_permutations), dtype=np.float64)
    for j in range(int(n_permutations)):
        shuf = df.copy()
        values = shuf[predictor].to_numpy(dtype=np.float64).copy()
        sessions = shuf["session_id"].to_numpy()
        for sess in np.unique(sessions):
            idx = np.flatnonzero(sessions == sess)
            if idx.size > 1:
                values[idx] = rng.permutation(values[idx])
        shuf[predictor] = values
        null[j] = _partial_corr(shuf, predictor, controls)["partial_r"]
    return {
        "perm_p_two_sided": float((1.0 + np.count_nonzero(np.abs(null) >= abs(float(observed_r)))) / (null.size + 1.0)),
        "perm_p_r_le_observed": float((1.0 + np.count_nonzero(null <= float(observed_r))) / (null.size + 1.0)),
        "perm_p_r_ge_observed": float((1.0 + np.count_nonzero(null >= float(observed_r))) / (null.size + 1.0)),
    }


def _load_merged(input_path: Path, stability_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    stability = pd.read_csv(stability_dir / "edge_parallel_stability_by_window.csv")
    metadata_path = stability_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"config": {}}
    cfg = metadata.get("config", {})
    source = pd.read_csv(input_path)
    if "duration_s" not in source.columns:
        source["duration_s"] = source.get("epoch_duration_s", np.nan)
    margin = float(cfg.get("min_patch_image_margin_px", cfg.get("patch_size_px", 540) / 2.0))
    keep = (
        np.isfinite(source["drift_orientation_deg"].astype(float))
        & np.isfinite(source["image_edge_axis_deg"].astype(float))
        & (source["anisotropy"].astype(float) >= float(cfg.get("reliable_drift_anisotropy_min", 0.2)))
        & (source["image_orientation_coherence"].astype(float) >= float(cfg.get("reliable_image_coherence_min", 0.2)))
        & (source["duration_s"].astype(float) >= float(cfg.get("min_duration_s", 0.1)))
        & (source["image_patch_distance_to_image_border_px"].astype(float) >= margin)
    )
    work = source.loc[keep].copy()
    work["source_window_id"] = np.arange(work.shape[0], dtype=int)
    keep_cols = [
        "source_window_id",
        "session",
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "duration_s",
        "image_patch_distance_to_image_border_px",
        "image_patch_rms_contrast",
        "image_patch_std",
        "image_gradient_energy",
        "image_edge_density",
        "image_high_freq_power_fraction",
        "rms_radius_deg",
    ]
    keep_cols = [col for col in keep_cols if col in work.columns]
    merged = stability.merge(
        work[keep_cols],
        left_on=["window_id", "session", "trial_idx"],
        right_on=["source_window_id", "session", "trial_idx"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.rename(
        columns={
            "session": "session_id",
            "trial_idx": "image_id",
            "drift_edge_cos2": "drift_edge_align_signed",
            "anisotropy": "drift_anisotropy",
        }
    )
    if "image_patch_rms_contrast" in merged.columns:
        merged["local_contrast"] = merged["image_patch_rms_contrast"].astype(float)
    elif "image_patch_std" in merged.columns:
        merged["local_contrast"] = merged["image_patch_std"].astype(float)
    else:
        merged["local_contrast"] = np.nan
    if "image_gradient_energy" in merged.columns:
        merged["edge_strength"] = merged["image_gradient_energy"].astype(float)
    elif "image_edge_density" in merged.columns:
        merged["edge_strength"] = merged["image_edge_density"].astype(float)
    else:
        merged["edge_strength"] = merged["local_contrast"]
    merged["edge_parallel_stability_advantage_pixel"] = merged["pixel_stability_advantage"].astype(float)
    merged["edge_parallel_stability_advantage_twin"] = merged["twin_stability_advantage"].astype(float) if "twin_stability_advantage" in merged.columns else np.nan
    merged["edge_parallel_relative_advantage_pixel"] = merged["pixel_relative_advantage"].astype(float)
    merged["edge_parallel_relative_advantage_twin"] = merged["twin_relative_advantage"].astype(float) if "twin_relative_advantage" in merged.columns else np.nan
    return merged, metadata


def _subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {"all_available_stability_windows": df.copy()}
    reliable = df[
        (df["duration_s"].astype(float) >= 0.10)
        & (df["drift_anisotropy"].astype(float) >= 0.20)
        & (df["image_orientation_coherence"].astype(float) >= 0.20)
        & (df["image_patch_distance_to_image_border_px"].astype(float) >= 270.0)
    ].copy()
    out["reliable_windows"] = reliable
    q = df["image_orientation_coherence"].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    out["coherence_low_tertile"] = df[df["image_orientation_coherence"] <= q[0]].copy()
    out["coherence_middle_tertile"] = df[(df["image_orientation_coherence"] > q[0]) & (df["image_orientation_coherence"] <= q[1])].copy()
    out["coherence_high_tertile"] = df[df["image_orientation_coherence"] > q[1]].copy()
    anis_q = float(df["drift_anisotropy"].quantile(2 / 3))
    out["high_anisotropy_top_tertile"] = df[df["drift_anisotropy"] >= anis_q].copy()
    return out


def _run_models(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    sensitivity_bootstrap: int,
    sensitivity_permutations: int,
) -> pd.DataFrame:
    rows = []
    predictors = [
        ("pixel", "edge_parallel_relative_advantage_pixel"),
        ("twin", "edge_parallel_relative_advantage_twin"),
    ]
    for subset_name, sub in _subsets(df).items():
        for predictor_label, predictor in predictors:
            if predictor not in sub.columns or sub[predictor].notna().sum() < 8:
                continue
            for model_name, controls in MODEL_SPECS:
                needed = ["drift_edge_align_signed", predictor, "session_id", *controls]
                block = sub.dropna(subset=needed).copy()
                if block["session_id"].nunique() < 2 or block.shape[0] < len(controls) + 8:
                    continue
                fit = _fit_model(block, predictor, controls)
                primary = subset_name == "all_available_stability_windows"
                boot_n = int(n_bootstrap) if primary else int(sensitivity_bootstrap)
                perm_n = int(n_permutations) if primary else int(sensitivity_permutations)
                boot = _session_bootstrap_model(block, predictor, controls, rng=rng, n_bootstrap=boot_n)
                perm = _permute_model(block, predictor, controls, fit["coef_stability"], rng=rng, n_permutations=perm_n)
                rows.append(
                    {
                        "subset": subset_name,
                        "predictor_family": predictor_label,
                        "predictor": predictor,
                        "model": model_name,
                        "controls": "+".join(controls) if controls else "none",
                        **fit,
                        "ci_low": boot["ci_low"],
                        "ci_high": boot["ci_high"],
                        "n_bootstrap": boot_n,
                        "n_permutations": perm_n,
                        **perm,
                    }
                )
    return pd.DataFrame(rows)


def _run_partial_corr(df: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int, n_permutations: int) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    residuals: dict[str, dict[str, np.ndarray]] = {}
    controls = ["image_orientation_coherence", "drift_anisotropy", "edge_strength"]
    for predictor_label, predictor in (
        ("pixel", "edge_parallel_relative_advantage_pixel"),
        ("twin", "edge_parallel_relative_advantage_twin"),
    ):
        block = df.dropna(subset=["drift_edge_align_signed", predictor, "session_id", *controls]).copy()
        if block["session_id"].nunique() < 2 or block.shape[0] < len(controls) + 8:
            continue
        result = _partial_corr(block, predictor, controls)
        boot = _bootstrap_partial_corr(block, predictor, controls, rng=rng, n_bootstrap=n_bootstrap)
        perm = _permute_partial_corr(block, predictor, controls, result["partial_r"], rng=rng, n_permutations=n_permutations)
        rows.append(
            {
                "subset": "all_available_stability_windows",
                "predictor_family": predictor_label,
                "predictor": predictor,
                "controls": "+".join(controls),
                "partial_r": result["partial_r"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                **perm,
                "n_windows": result["n_windows"],
                "n_sessions": int(block["session_id"].nunique()),
            }
        )
        residuals[predictor_label] = {
            "window_row": block["window_row"].to_numpy(),
            "y_resid": result["y_resid"],
            "predictor_resid": result["predictor_resid"],
        }
    return pd.DataFrame(rows), residuals


def _coherence_stratified(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = df["image_orientation_coherence"].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    bins = [
        ("low", df[df["image_orientation_coherence"] <= q[0]].copy()),
        ("middle", df[(df["image_orientation_coherence"] > q[0]) & (df["image_orientation_coherence"] <= q[1])].copy()),
        ("high", df[df["image_orientation_coherence"] > q[1]].copy()),
    ]
    for bin_name, block in bins:
        for predictor_label, predictor in (
            ("pixel", "edge_parallel_relative_advantage_pixel"),
            ("twin", "edge_parallel_relative_advantage_twin"),
        ):
            sub = block.dropna(subset=["drift_edge_align_signed", predictor, "session_id"]).copy()
            if sub.shape[0] < 8:
                continue
            fit = _fit_model(sub, predictor, [])
            rows.append(
                {
                    "coherence_bin": bin_name,
                    "predictor_family": predictor_label,
                    "coherence_min": float(sub["image_orientation_coherence"].min()),
                    "coherence_max": float(sub["image_orientation_coherence"].max()),
                    "mean_drift_edge_align_signed": float(sub.groupby("session_id")["drift_edge_align_signed"].mean().mean()),
                    "mean_stability_advantage": float(sub.groupby("session_id")[predictor].mean().mean()),
                    "slope_y_on_stability_within_session": fit["coef_stability"],
                    "incremental_r2": fit["incremental_r2"],
                    "n_windows": int(sub.shape[0]),
                    "n_sessions": int(sub["session_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _session_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    controls = ["image_orientation_coherence", "drift_anisotropy", "edge_strength"]
    for session, block in df.groupby("session_id"):
        for label, predictor in (
            ("pixel", "edge_parallel_relative_advantage_pixel"),
            ("twin", "edge_parallel_relative_advantage_twin"),
        ):
            sub = block.dropna(subset=["drift_edge_align_signed", predictor, *controls]).copy()
            if sub.shape[0] < len(controls) + 4:
                continue
            y = sub["drift_edge_align_signed"].to_numpy(dtype=np.float64)
            X = np.column_stack([sub[col].to_numpy(dtype=np.float64) for col in [*controls, predictor]])
            fit = _fit_standardized_ols(y, X)
            rows.append(
                {
                    "session_id": session,
                    "predictor_family": label,
                    "coef_stability": float(fit["coef"][-1]),
                    "r2": float(fit["r2"]),
                    "n_windows": int(sub.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _write_plots(out_dir: Path, df: pd.DataFrame, model_summary: pd.DataFrame, partial: pd.DataFrame, residuals: dict[str, dict[str, np.ndarray]], stratified: pd.DataFrame, slopes: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"pixel": "#4878a8", "twin": "#c16622"}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), dpi=150, sharey=True)
    for ax, (label, pred) in zip(axes, (("pixel", "edge_parallel_relative_advantage_pixel"), ("twin", "edge_parallel_relative_advantage_twin")), strict=True):
        ax.scatter(df[pred], df["drift_edge_align_signed"], s=13, alpha=0.5, color=colors[label])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel(f"{label} stability advantage")
        ax.set_title(label, loc="left", fontsize=10)
    axes[0].set_ylabel("drift-edge alignment cos(2 delta)")
    fig.tight_layout()
    fig.savefig(out_dir / "drift_edge_alignment_vs_stability_raw.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), dpi=150, sharey=True)
    for ax, label in zip(axes, ("pixel", "twin"), strict=True):
        if label not in residuals:
            ax.axis("off")
            continue
        ax.scatter(residuals[label]["predictor_resid"], residuals[label]["y_resid"], s=13, alpha=0.5, color=colors[label])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel(f"{label} residual stability")
        ax.set_title(label, loc="left", fontsize=10)
    axes[0].set_ylabel("residual drift-edge alignment")
    fig.tight_layout()
    fig.savefig(out_dir / "drift_edge_alignment_vs_stability_residualized.png", dpi=150)
    plt.close(fig)

    if not stratified.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
        bins = ["low", "middle", "high"]
        x = np.arange(len(bins))
        width = 0.34
        for offset, label in [(-width / 2, "pixel"), (width / 2, "twin")]:
            block = stratified[stratified["predictor_family"] == label].set_index("coherence_bin").reindex(bins)
            ax.bar(x + offset, block["slope_y_on_stability_within_session"], width=width, color=colors[label], label=label)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_ylabel("within-session standardized slope")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "stability_slope_by_coherence_bin.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), dpi=150)
        block = stratified[stratified["predictor_family"] == "twin"].set_index("coherence_bin").reindex(bins)
        if block["mean_drift_edge_align_signed"].isna().all():
            block = stratified[stratified["predictor_family"] == "pixel"].set_index("coherence_bin").reindex(bins)
        axes[0].bar(x, block["mean_drift_edge_align_signed"], color="#555555")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(bins)
        axes[0].set_ylabel("mean alignment")
        for label in ("pixel", "twin"):
            b = stratified[stratified["predictor_family"] == label].set_index("coherence_bin").reindex(bins)
            axes[1].plot(x, b["mean_stability_advantage"], marker="o", color=colors[label], label=label)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(bins)
        axes[1].set_ylabel("mean stability advantage")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "mean_alignment_and_stability_by_coherence_bin.png", dpi=150)
        plt.close(fig)

    if not slopes.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=150)
        plot = slopes.sort_values(["predictor_family", "coef_stability"]).copy()
        labels = [f"{r.session_id}\n{r.predictor_family}" for r in plot.itertuples()]
        ax.bar(np.arange(plot.shape[0]), plot["coef_stability"], color=[colors[x] for x in plot["predictor_family"]])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(plot.shape[0]))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_ylabel("session controlled slope")
        fig.tight_layout()
        fig.savefig(out_dir / "session_slopes_stability_controlled.png", dpi=150)
        plt.close(fig)


def _decision_label(model_summary: pd.DataFrame, partial: pd.DataFrame, merged: pd.DataFrame) -> tuple[str, list[str]]:
    notes = []
    mean_align = float(merged.groupby("session_id")["drift_edge_align_signed"].mean().mean())
    pixel_adv = float(merged.groupby("session_id")["edge_parallel_relative_advantage_pixel"].mean().mean())
    twin_adv = float(merged.groupby("session_id")["edge_parallel_relative_advantage_twin"].mean().mean())
    first_order = mean_align > 0 and pixel_adv > 0 and twin_adv > 0
    if first_order:
        notes.append("stability_consistent_first_order: real drift is edge-parallel and edge-parallel motion is less disruptive.")
    full = model_summary[
        (model_summary["subset"] == "all_available_stability_windows")
        & (model_summary["model"] == "model3_full_low_level")
    ].copy()
    robust_negative = False
    robust_positive = False
    for _, row in full.iterrows():
        if float(row["coef_stability"]) < 0 and float(row["ci_high"]) < 0 and float(row["perm_p_coef_le_observed"]) <= 0.05:
            robust_negative = True
        if float(row["coef_stability"]) > 0 and float(row["ci_low"]) > 0 and float(row["perm_p_coef_ge_observed"]) <= 0.05:
            robust_positive = True
    model0 = model_summary[
        (model_summary["subset"] == "all_available_stability_windows")
        & (model_summary["model"] == "model0_unadjusted")
    ]
    model1 = model_summary[
        (model_summary["subset"] == "all_available_stability_windows")
        & (model_summary["model"] == "model1_coherence")
    ]
    unadj_negative = bool((model0["coef_stability"] < 0).any())
    coherence_removed = unadj_negative and bool((model1["coef_stability"] >= 0).any() or (model1["ci_high"] >= 0).all())
    if robust_negative:
        label = "stability_rejected"
        notes.append("wrong_direction_survives_controls: controlled wrong-direction slope is robust.")
    elif robust_positive and bool((full["coef_stability"] > 0).all()):
        label = "stability_supported"
        notes.append("stability_supported: stability advantage positively predicts drift-edge alignment after controls for all tested stability predictors.")
    elif coherence_removed:
        label = "wrong_direction_coherence_artifact"
        notes.append("wrong_direction_coherence_artifact: unadjusted negative relationship disappears or weakens after coherence controls.")
    else:
        label = "stability_inconclusive"
        notes.append("stability_inconclusive: first-order signs are consistent with stability, but controlled slopes differ between pixel and twin stability predictors.")
    return label, notes


def _write_decision_table(out_dir: Path, model_summary: pd.DataFrame, partial: pd.DataFrame, stratified: pd.DataFrame, merged: pd.DataFrame) -> None:
    label, notes = _decision_label(model_summary, partial, merged)
    lines = [
        "# BackImage Stability Wrong-Direction Controls",
        "",
        f"Decision label: `{label}`",
        "",
        "## Decision Notes",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## First-Order Signs",
        "",
        f"- Mean session drift-edge alignment `cos(2 * delta_theta)`: `{merged.groupby('session_id')['drift_edge_align_signed'].mean().mean():+.4f}`.",
        f"- Mean session pixel relative stability advantage: `{merged.groupby('session_id')['edge_parallel_relative_advantage_pixel'].mean().mean():+.4f}`.",
        f"- Mean session twin relative stability advantage: `{merged.groupby('session_id')['edge_parallel_relative_advantage_twin'].mean().mean():+.4f}`.",
        "",
        "## Primary Controlled Models",
        "",
    ]
    focus = model_summary[
        (model_summary["subset"] == "all_available_stability_windows")
        & model_summary["model"].isin([name for name, _ in MODEL_SPECS])
    ].copy()
    for _, row in focus.sort_values(["predictor_family", "model"]).iterrows():
        lines.append(
            f"- `{row['predictor_family']}` `{row['model']}` controls `{row['controls']}`: "
            f"coef `{row['coef_stability']:+.4f}` CI `[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]`, "
            f"dR2 `{row['incremental_r2']:+.4f}`, perm p two-sided `{row['perm_p_two_sided']:.4f}`."
        )
    lines.extend(["", "## Partial Correlations", ""])
    for _, row in partial.iterrows():
        lines.append(
            f"- `{row['predictor_family']}`: partial r `{row['partial_r']:+.4f}` "
            f"CI `[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]`, "
            f"perm p two-sided `{row['perm_p_two_sided']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Do not use the phrase \"generic stability is not the right free-viewing objective\" unless the wrong-direction relationship survives the controlled models.",
            "",
        ]
    )
    (out_dir / "decision_table.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    merged, metadata = _load_merged(Path(args.input), Path(args.stability_dir))
    merged.to_csv(out_dir / "merged_window_stability_alignment_table.csv", index=False)
    model_summary = _run_models(
        merged,
        rng=rng,
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        sensitivity_bootstrap=int(args.sensitivity_bootstrap),
        sensitivity_permutations=int(args.sensitivity_permutations),
    )
    model_summary.to_csv(out_dir / "stability_wrong_direction_model_summary.csv", index=False)
    partial, residuals = _run_partial_corr(merged, rng=rng, n_bootstrap=int(args.n_bootstrap), n_permutations=int(args.n_permutations))
    partial.to_csv(out_dir / "stability_wrong_direction_partial_corr_summary.csv", index=False)
    stratified = _coherence_stratified(merged)
    stratified.to_csv(out_dir / "stability_wrong_direction_coherence_stratified_summary.csv", index=False)
    slopes = _session_slopes(merged)
    slopes.to_csv(out_dir / "stability_wrong_direction_session_slopes.csv", index=False)
    _write_plots(out_dir, merged, model_summary, partial, residuals, stratified, slopes)
    _write_decision_table(out_dir, model_summary, partial, stratified, merged)
    (out_dir / "posthoc_metadata.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "stability_dir": str(args.stability_dir),
                "source_stability_metadata": metadata,
                "n_bootstrap": int(args.n_bootstrap),
                "n_permutations": int(args.n_permutations),
                "sensitivity_bootstrap": int(args.sensitivity_bootstrap),
                "sensitivity_permutations": int(args.sensitivity_permutations),
                "seed": int(args.seed),
                "notes": (
                    "All analyses are limited to windows with existing edge-parallel stability outputs. "
                    "Positive drift_edge_align_signed is edge-parallel alignment under cos(2 * delta_theta). "
                    "Positive stability advantage means orthogonal motion was more disruptive than parallel motion."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote BackImage stability wrong-direction controls to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--sensitivity-bootstrap", type=int, default=250)
    parser.add_argument("--sensitivity-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
