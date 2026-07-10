"""Analyze saved Figure 4C feature predictions by feature group.

The saved prediction artifact is meant to answer a narrower question than the
main pooled score: whether the hidden-joint observer helps specific feature
groups after calibration, even when the full pooled target is dominated by
static/coarse structure.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.feature_recovery_scores import R2_CV_METHOD


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTION_DIR = REPO_ROOT / "outputs" / "figure4_joint_decoder_prediction_saved_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "figure4_feature_group_calibration_diagnostics_v1"

DEFAULT_KNOWN_MODE = "true_tau_interactions"
DEFAULT_JOINT_MODE = "hidden_joint_forward_model"
DEFAULT_ZERO_MODE = "zero_static"

MODE_LABELS = {
    DEFAULT_KNOWN_MODE: "known",
    DEFAULT_JOINT_MODE: "joint",
    DEFAULT_ZERO_MODE: "zero",
}
MODE_COLORS = {
    DEFAULT_KNOWN_MODE: "#be123c",
    DEFAULT_JOINT_MODE: "#0f766e",
    DEFAULT_ZERO_MODE: "#b45309",
}
CONTRAST_COLORS = {
    "joint_minus_zero": "#0f766e",
    "known_minus_joint": "#7c2d12",
    "known_minus_zero": "#be123c",
}
SCORE_VARIANTS = ("uncalibrated", "crossfold_scalar_gain", "crossfold_diagonal_gain")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--prediction-rows", type=Path, default=None)
    parser.add_argument("--prediction-arrays", type=Path, default=None)
    parser.add_argument("--raw-feature-metadata", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--known-mode", default=DEFAULT_KNOWN_MODE)
    parser.add_argument("--joint-mode", default=DEFAULT_JOINT_MODE)
    parser.add_argument("--zero-mode", default=DEFAULT_ZERO_MODE)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "font.size": 9.5,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf, dpi=dpi)
    plt.close(fig)
    return png


def _finite_limits(values: Iterable[float], *, pad_fraction: float = 0.08) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if math.isclose(lo, hi):
        delta = 1.0 if math.isclose(lo, 0.0) else abs(lo) * 0.2
        return lo - delta, hi + delta
    pad = (hi - lo) * float(pad_fraction)
    return lo - pad, hi + pad


def _mode_label(mode: str, mode_labels: dict[str, str]) -> str:
    return mode_labels.get(str(mode), str(mode).replace("_", " "))


def _prior_label(value: object) -> str:
    return str(value).replace("axis_edge_", "").replace("_", " ")


def _score_label(value: str) -> str:
    return str(value).replace("crossfold_", "").replace("_", " ")


def _read_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    rows_path = (
        args.prediction_rows
        if args.prediction_rows is not None
        else args.prediction_dir / "linear_synthetic_prior_feature_observer_prediction_rows.csv"
    )
    arrays_path = (
        args.prediction_arrays
        if args.prediction_arrays is not None
        else args.prediction_dir / "linear_synthetic_prior_feature_observer_prediction_arrays.npz"
    )
    metadata_path = (
        args.raw_feature_metadata
        if args.raw_feature_metadata is not None
        else args.prediction_dir / "linear_synthetic_prior_feature_observer_raw_feature_dim_metadata.csv"
    )
    rows = pd.read_csv(rows_path)
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {key: data[key].astype(np.float64, copy=False) for key in data.files}
    metadata = pd.read_csv(metadata_path)
    required_arrays = {
        "z_hat",
        "z_true",
        "z_train_mean",
        "raw_hat_projected",
        "raw_true_projected",
        "raw_train_mean_projected",
    }
    missing = sorted(required_arrays.difference(arrays))
    if missing:
        raise ValueError(f"{arrays_path} is missing arrays: {missing}")
    n = int(rows.shape[0])
    for key, arr in arrays.items():
        if arr.shape[0] != n:
            raise ValueError(f"Array {key!r} rows {arr.shape[0]} do not match prediction rows {n}")
    if "prediction_row" in rows.columns:
        order = rows["prediction_row"].to_numpy(dtype=int)
        if not np.array_equal(order, np.arange(n, dtype=int)):
            raise ValueError("prediction_row must match array row order for this diagnostic")
    return rows, arrays, metadata


def _group_specs(rows: pd.DataFrame, metadata: pd.DataFrame) -> list[dict[str, object]]:
    z_dim = int(rows["feature_dim"].dropna().iloc[0])
    specs: list[dict[str, object]] = [
        {
            "feature_space": "locked_feature_space",
            "group_kind": "all",
            "group_value": "all",
            "dims": np.arange(z_dim, dtype=int),
        }
    ]
    for dim in range(z_dim):
        specs.append(
            {
                "feature_space": "locked_feature_space",
                "group_kind": "pc",
                "group_value": f"pc{dim}",
                "dims": np.asarray([dim], dtype=int),
            }
        )
    raw_dim = int(rows["raw_feature_dim"].dropna().iloc[0])
    if metadata.shape[0] != raw_dim:
        raise ValueError(f"raw feature metadata rows {metadata.shape[0]} do not match raw dim {raw_dim}")
    if "channel" in metadata.columns:
        for channel in ["real", "imag", "magnitude"]:
            dims = metadata.index[metadata["channel"].astype(str).eq(channel)].to_numpy(dtype=int)
            if dims.size:
                specs.append(
                    {
                        "feature_space": "raw_projected_feature_space",
                        "group_kind": "channel",
                        "group_value": channel,
                        "dims": dims,
                    }
                )
    if "band" in metadata.columns:
        for band in sorted(pd.to_numeric(metadata["band"], errors="coerce").dropna().astype(int).unique()):
            if int(band) < 0:
                continue
            dims = metadata.index[pd.to_numeric(metadata["band"], errors="coerce").astype("Int64").eq(int(band))].to_numpy(dtype=int)
            if dims.size:
                specs.append(
                    {
                        "feature_space": "raw_projected_feature_space",
                        "group_kind": "band",
                        "group_value": f"band{int(band)}",
                        "dims": dims,
                    }
                )
    if {"band", "orientation"}.issubset(metadata.columns):
        band_values = sorted(pd.to_numeric(metadata["band"], errors="coerce").dropna().astype(int).unique())
        orientation_values = sorted(pd.to_numeric(metadata["orientation"], errors="coerce").dropna().astype(int).unique())
        band_series = pd.to_numeric(metadata["band"], errors="coerce").astype("Int64")
        orientation_series = pd.to_numeric(metadata["orientation"], errors="coerce").astype("Int64")
        for band in band_values:
            if int(band) < 0:
                continue
            for orientation in orientation_values:
                if int(orientation) < 0:
                    continue
                mask = band_series.eq(int(band)) & orientation_series.eq(int(orientation))
                dims = metadata.index[mask].to_numpy(dtype=int)
                if dims.size:
                    specs.append(
                        {
                            "feature_space": "raw_projected_feature_space",
                            "group_kind": "band_orientation",
                            "group_value": f"band{int(band)}_ori{int(orientation)}",
                            "band": int(band),
                            "orientation": int(orientation),
                            "dims": dims,
                        }
                    )
    return specs


def _arrays_for_space(arrays: dict[str, np.ndarray], feature_space: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_space == "locked_feature_space":
        return arrays["z_hat"], arrays["z_true"], arrays["z_train_mean"]
    if feature_space == "raw_projected_feature_space":
        return arrays["raw_hat_projected"], arrays["raw_true_projected"], arrays["raw_train_mean_projected"]
    raise ValueError(f"Unknown feature_space {feature_space!r}")


def _calibrated_predictions(
    *,
    pred: np.ndarray,
    true: np.ndarray,
    folds: np.ndarray,
    variant: str,
) -> np.ndarray:
    if variant == "uncalibrated":
        return pred.copy()
    out = np.full_like(pred, np.nan, dtype=np.float64)
    unique_folds = sorted(set(int(value) for value in folds.tolist()))
    for fold in unique_folds:
        test = folds == int(fold)
        calib = ~test
        if not np.any(test) or not np.any(calib):
            continue
        calib_pred = pred[calib]
        calib_true = true[calib]
        good_rows = np.isfinite(calib_pred).all(axis=1) & np.isfinite(calib_true).all(axis=1)
        calib_pred = calib_pred[good_rows]
        calib_true = calib_true[good_rows]
        if calib_pred.size == 0:
            continue
        if variant == "crossfold_scalar_gain":
            denom = float(np.sum(calib_pred * calib_pred))
            if denom <= 1e-12:
                continue
            gain = float(np.sum(calib_pred * calib_true) / denom)
            out[test] = gain * pred[test]
        elif variant == "crossfold_diagonal_gain":
            denom = np.sum(calib_pred * calib_pred, axis=0)
            numer = np.sum(calib_pred * calib_true, axis=0)
            gain = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
            out[test] = pred[test] * gain[None, :]
        else:
            raise ValueError(f"Unknown score variant {variant!r}")
    return out


def _row_sse_sst(pred: np.ndarray, true: np.ndarray, train_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    good = np.isfinite(pred).all(axis=1) & np.isfinite(true).all(axis=1) & np.isfinite(train_mean).all(axis=1)
    sse = np.full(pred.shape[0], np.nan, dtype=np.float64)
    sst = np.full(pred.shape[0], np.nan, dtype=np.float64)
    if np.any(good):
        sse[good] = np.sum((true[good] - pred[good]) ** 2, axis=1)
        sst[good] = np.sum((true[good] - train_mean[good]) ** 2, axis=1)
    return sse, sst


def _score_from_sse_sst(sse: np.ndarray, sst: np.ndarray) -> tuple[float, float, float, int]:
    sse_arr = np.asarray(sse, dtype=np.float64)
    sst_arr = np.asarray(sst, dtype=np.float64)
    good = np.isfinite(sse_arr) & np.isfinite(sst_arr)
    total_sse = float(np.sum(sse_arr[good])) if np.any(good) else float("nan")
    total_sst = float(np.sum(sst_arr[good])) if np.any(good) else float("nan")
    r2 = float(1.0 - total_sse / total_sst) if np.isfinite(total_sst) and total_sst > 1e-12 else float("nan")
    return r2, total_sse, total_sst, int(np.sum(good))


def _compute_row_scores(
    rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    specs: list[dict[str, object]],
    *,
    modes: list[str],
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for spec in specs:
        feature_space = str(spec["feature_space"])
        dims = np.asarray(spec["dims"], dtype=int)
        pred_all, true_all, mean_all = _arrays_for_space(arrays, feature_space)
        for prior, prior_group in rows.groupby("prior_family", dropna=False, sort=True):
            for mode in modes:
                mode_group = prior_group[prior_group["observer_mode"].astype(str).eq(mode)]
                if mode_group.empty:
                    continue
                row_indices = mode_group.index.to_numpy(dtype=int)
                pred = pred_all[row_indices][:, dims]
                true = true_all[row_indices][:, dims]
                mean = mean_all[row_indices][:, dims]
                folds = mode_group["fold"].to_numpy(dtype=int)
                for variant in SCORE_VARIANTS:
                    pred_cal = _calibrated_predictions(pred=pred, true=true, folds=folds, variant=variant)
                    sse, sst = _row_sse_sst(pred_cal, true, mean)
                    for local_index, row_index in enumerate(row_indices):
                        base = rows.iloc[int(row_index)]
                        out_rows.append(
                            {
                                "prediction_row": int(base.get("prediction_row", row_index)),
                                "table_index": int(base["table_index"]),
                                "trial_id": int(base["trial_id"]),
                                "true_source_row": int(base["true_source_row"]),
                                "observation_scale": float(base["observation_scale"]),
                                "prior_family": prior,
                                "observer_mode": mode,
                                "fold": int(base["fold"]),
                                "feature_space": feature_space,
                                "group_kind": str(spec["group_kind"]),
                                "group_value": str(spec["group_value"]),
                                "band": spec.get("band", np.nan),
                                "orientation": spec.get("orientation", np.nan),
                                "n_dims": int(dims.size),
                                "score_variant": variant,
                                "calibration_scope": (
                                    "none"
                                    if variant == "uncalibrated"
                                    else "other_outer_folds_same_scale_prior_observer_from_saved_oof_predictions"
                                ),
                                "sse": float(sse[local_index]),
                                "sst": float(sst[local_index]),
                            }
                        )
    return pd.DataFrame(out_rows)


def _summarize_row_scores(row_scores: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "observation_scale",
        "prior_family",
        "observer_mode",
        "feature_space",
        "group_kind",
        "group_value",
        "band",
        "orientation",
        "n_dims",
        "score_variant",
        "calibration_scope",
    ]
    summary = (
        row_scores.groupby(group_cols, dropna=False, as_index=False)
        .agg(n_rows=("sse", "size"), sse=("sse", "sum"), sst=("sst", "sum"))
        .sort_values(group_cols)
    )
    summary["R2_cv"] = 1.0 - summary["sse"] / summary["sst"]
    summary.loc[summary["sst"] <= 1e-12, "R2_cv"] = np.nan
    summary["score_method"] = R2_CV_METHOD
    return summary


def _bootstrap_contrasts(
    frame: pd.DataFrame,
    *,
    known_mode: str,
    joint_mode: str,
    zero_mode: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {"bootstrap_n": 0}
    key_cols = ["table_index", "trial_id", "true_source_row"]
    grouped = frame.groupby(key_cols + ["observer_mode"], dropna=False, sort=False)[["sse", "sst"]].sum().reset_index()
    wide_sse = grouped.pivot(index=key_cols, columns="observer_mode", values="sse")
    wide_sst = grouped.pivot(index=key_cols, columns="observer_mode", values="sst")
    modes = [known_mode, joint_mode, zero_mode]
    if any(mode not in wide_sse.columns for mode in modes):
        return {"bootstrap_n": 0, "bootstrap_skipped_missing_mode": 1}
    complete = wide_sse[modes].notna().all(axis=1) & wide_sst[modes].notna().all(axis=1)
    wide_sse = wide_sse.loc[complete]
    wide_sst = wide_sst.loc[complete]
    if wide_sse.empty:
        return {"bootstrap_n": 0, "bootstrap_skipped_no_complete_triplets": 1}
    rng = np.random.default_rng(seed)
    n = len(wide_sse)
    sse_arrays = {mode: wide_sse[mode].to_numpy(dtype=np.float64) for mode in modes}
    sst_arrays = {mode: wide_sst[mode].to_numpy(dtype=np.float64) for mode in modes}
    values: dict[str, list[float]] = {
        "joint_minus_zero": [],
        "known_minus_joint": [],
        "known_minus_zero": [],
    }
    for _ in range(int(n_bootstrap)):
        sample = rng.integers(0, n, size=n)
        scores: dict[str, float] = {}
        for mode in modes:
            sse = float(np.sum(sse_arrays[mode][sample]))
            sst = float(np.sum(sst_arrays[mode][sample]))
            scores[mode] = float(1.0 - sse / sst) if np.isfinite(sst) and sst > 1e-12 else np.nan
        values["joint_minus_zero"].append(float(scores[joint_mode] - scores[zero_mode]))
        values["known_minus_joint"].append(float(scores[known_mode] - scores[joint_mode]))
        values["known_minus_zero"].append(float(scores[known_mode] - scores[zero_mode]))
    out: dict[str, float] = {"bootstrap_n": int(n_bootstrap), "bootstrap_trial_units": int(n)}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        out[f"{name}_ci_low"] = float(np.quantile(arr, 0.025)) if arr.size else np.nan
        out[f"{name}_ci_high"] = float(np.quantile(arr, 0.975)) if arr.size else np.nan
        out[f"{name}_bootstrap_mean"] = float(np.mean(arr)) if arr.size else np.nan
    return out


def _contrast_summary(
    summary: pd.DataFrame,
    row_scores: pd.DataFrame,
    *,
    known_mode: str,
    joint_mode: str,
    zero_mode: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    group_cols = [
        "observation_scale",
        "prior_family",
        "feature_space",
        "group_kind",
        "group_value",
        "band",
        "orientation",
        "n_dims",
        "score_variant",
        "calibration_scope",
    ]
    rows = []
    for offset, (values, group) in enumerate(summary.groupby(group_cols, dropna=False, sort=True)):
        if not isinstance(values, tuple):
            values = (values,)
        by_mode = {str(row["observer_mode"]): row for _, row in group.iterrows()}
        if not {known_mode, joint_mode, zero_mode}.issubset(by_mode):
            continue
        row = dict(zip(group_cols, values))
        s_known = float(by_mode[known_mode]["R2_cv"])
        s_joint = float(by_mode[joint_mode]["R2_cv"])
        s_zero = float(by_mode[zero_mode]["R2_cv"])
        row.update(
            {
                "S_known": s_known,
                "S_joint": s_joint,
                "S_zero": s_zero,
                "known_mode": known_mode,
                "joint_mode": joint_mode,
                "zero_mode": zero_mode,
                "joint_minus_zero": s_joint - s_zero,
                "known_minus_joint": s_known - s_joint,
                "known_minus_zero": s_known - s_zero,
                "incremental_error_reduction_joint_over_zero": (
                    (float(by_mode[zero_mode]["sse"]) - float(by_mode[joint_mode]["sse"]))
                    / float(by_mode[zero_mode]["sst"])
                    if float(by_mode[zero_mode]["sst"]) > 1e-12
                    else np.nan
                ),
                "score_method": R2_CV_METHOD,
            }
        )
        mask = np.ones(row_scores.shape[0], dtype=bool)
        for col, value in row.items():
            if col in row_scores.columns and col in group_cols:
                if pd.isna(value):
                    mask &= row_scores[col].isna().to_numpy()
                else:
                    mask &= row_scores[col].astype(str).eq(str(value)).to_numpy()
        row_frame = row_scores.loc[mask].copy()
        row.update(
            _bootstrap_contrasts(
                row_frame,
                known_mode=known_mode,
                joint_mode=joint_mode,
                zero_mode=zero_mode,
                n_bootstrap=n_bootstrap,
                seed=seed + offset,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_channel_contrasts(contrasts: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path:
    frame = contrasts[
        contrasts["feature_space"].eq("raw_projected_feature_space")
        & contrasts["group_kind"].eq("channel")
    ].copy()
    if frame.empty:
        raise ValueError("No raw channel contrasts available")
    priors = sorted(frame["prior_family"].dropna().unique().tolist())
    variants = list(SCORE_VARIANTS)
    channels = ["real", "imag", "magnitude"]
    fig, axes = plt.subplots(
        len(variants),
        len(priors),
        figsize=(5.4 * len(priors), 3.25 * len(variants)),
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(len(variants), len(priors))
    all_values = []
    for row_i, variant in enumerate(variants):
        for col_i, prior in enumerate(priors):
            ax = axes_arr[row_i, col_i]
            subset = frame[frame["score_variant"].eq(variant) & frame["prior_family"].eq(prior)]
            values = []
            lows = []
            highs = []
            for channel in channels:
                row = subset[subset["group_value"].eq(channel)]
                value = float(row["joint_minus_zero"].iloc[0]) if not row.empty else np.nan
                values.append(value)
                lows.append(float(row["joint_minus_zero_ci_low"].iloc[0]) if not row.empty else np.nan)
                highs.append(float(row["joint_minus_zero_ci_high"].iloc[0]) if not row.empty else np.nan)
            x = np.arange(len(channels))
            ax.bar(x, values, color="#0f766e", alpha=0.9)
            lower = [value - low if np.isfinite(low) else 0.0 for value, low in zip(values, lows)]
            upper = [high - value if np.isfinite(high) else 0.0 for value, high in zip(values, highs)]
            ax.errorbar(x, values, yerr=np.asarray([lower, upper]), fmt="none", ecolor="#111827", capsize=4)
            ax.axhline(0.0, color="#111827", linewidth=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=20, ha="right")
            ax.set_title(f"{_prior_label(prior)} | {_score_label(variant)}", loc="left")
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
            all_values.extend(values + lows + highs)
    lo, hi = _finite_limits(all_values, pad_fraction=0.16)
    for ax in axes_arr.reshape(-1):
        ax.set_ylim(min(lo, -0.1), max(hi, 0.1))
    for ax in axes_arr[:, 0]:
        ax.set_ylabel("joint - zero R2_cv")
    fig.suptitle("1x raw-projected feature-channel contrasts", x=0.01, y=1.02, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "figure4c_1x_feature_channel_joint_minus_zero", dpi=dpi)


def _plot_channel_observer_scores(summary: pd.DataFrame, out_dir: Path, *, modes: list[str], mode_labels: dict[str, str], dpi: int) -> Path:
    frame = summary[
        summary["feature_space"].eq("raw_projected_feature_space")
        & summary["group_kind"].eq("channel")
        & summary["score_variant"].eq("crossfold_diagonal_gain")
    ].copy()
    if frame.empty:
        raise ValueError("No raw channel diagonal-gain scores available")
    priors = sorted(frame["prior_family"].dropna().unique().tolist())
    channels = ["real", "imag", "magnitude"]
    fig, axes = plt.subplots(1, len(priors), figsize=(5.8 * len(priors), 4.3), sharey=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    width = 0.23
    all_values = []
    for ax, prior in zip(axes_arr, priors):
        subset = frame[frame["prior_family"].eq(prior)]
        x = np.arange(len(channels), dtype=float)
        for mode_i, mode in enumerate(modes):
            values = []
            for channel in channels:
                row = subset[subset["observer_mode"].astype(str).eq(mode) & subset["group_value"].eq(channel)]
                values.append(float(row["R2_cv"].iloc[0]) if not row.empty else np.nan)
            all_values.extend(values)
            ax.bar(
                x + (mode_i - 1) * width,
                values,
                width=width,
                color=MODE_COLORS.get(mode, "#6b7280"),
                alpha=0.9,
                label=_mode_label(mode, mode_labels),
            )
        ax.axhline(0.0, color="#111827", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=20, ha="right")
        ax.set_title(_prior_label(prior), loc="left")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    lo, hi = _finite_limits(all_values, pad_fraction=0.14)
    for ax in axes_arr:
        ax.set_ylim(min(lo, -0.2), max(hi, 0.2))
    axes_arr[0].set_ylabel("cross-fold diagonal-gain R2_cv")
    axes_arr[-1].legend(frameon=False, loc="best")
    fig.suptitle("1x raw-projected channel scores after diagonal calibration", x=0.01, y=1.02, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "figure4c_1x_feature_channel_diagonal_r2_by_observer", dpi=dpi)


def _plot_band_orientation_heatmap(contrasts: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path:
    frame = contrasts[
        contrasts["feature_space"].eq("raw_projected_feature_space")
        & contrasts["group_kind"].eq("band_orientation")
        & contrasts["score_variant"].eq("crossfold_diagonal_gain")
    ].copy()
    if frame.empty:
        raise ValueError("No band/orientation diagonal contrasts available")
    priors = sorted(frame["prior_family"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(priors), figsize=(5.3 * len(priors), 4.2), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    vals = frame["joint_minus_zero"].to_numpy(dtype=np.float64)
    max_abs = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.1
    max_abs = max(max_abs, 0.05)
    image = None
    for ax, prior in zip(axes_arr, priors):
        subset = frame[frame["prior_family"].eq(prior)]
        bands = sorted(pd.to_numeric(subset["band"], errors="coerce").dropna().astype(int).unique())
        orientations = sorted(pd.to_numeric(subset["orientation"], errors="coerce").dropna().astype(int).unique())
        mat = np.full((len(bands), len(orientations)), np.nan, dtype=np.float64)
        for i, band in enumerate(bands):
            for j, orientation in enumerate(orientations):
                row = subset[
                    pd.to_numeric(subset["band"], errors="coerce").eq(int(band))
                    & pd.to_numeric(subset["orientation"], errors="coerce").eq(int(orientation))
                ]
                if not row.empty:
                    mat[i, j] = float(row["joint_minus_zero"].iloc[0])
        image = ax.imshow(mat, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_title(_prior_label(prior), loc="left")
        ax.set_xlabel("orientation")
        ax.set_ylabel("band")
        ax.set_xticks(np.arange(len(orientations)))
        ax.set_xticklabels([str(v) for v in orientations])
        ax.set_yticks(np.arange(len(bands)))
        ax.set_yticklabels([str(v) for v in bands])
    if image is not None:
        fig.colorbar(image, ax=axes_arr.ravel().tolist(), shrink=0.82, label="joint - zero R2_cv")
    fig.suptitle("1x band/orientation joint-zero contrast after diagonal calibration", x=0.01, y=1.02, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "figure4c_1x_band_orientation_diagonal_joint_minus_zero", dpi=dpi)


def _write_readme(path: Path, *, args: argparse.Namespace, outputs: list[Path]) -> None:
    lines = [
        "# Feature-group calibration diagnostics",
        "",
        "This diagnostic consumes saved out-of-fold feature predictions from the Figure 4C observer.",
        "It computes pooled multi-output R2 by locked-space and raw-projected feature groups.",
        "",
        "Calibration variants:",
        "- `uncalibrated`: saved prediction directly.",
        "- `crossfold_scalar_gain`: one gain fitted from other outer-fold predictions in the same scale/prior/observer subset.",
        "- `crossfold_diagonal_gain`: one gain per feature dimension fitted from other outer-fold predictions in the same scale/prior/observer subset.",
        "",
        "Important limitation: raw-space scores use the inverse projection of the locked PCA/whitened feature scores. They diagnose the retained feature subspace, not discarded raw-pyramid variance.",
        "",
        f"Scale: `{float(args.scale):g}x`",
        f"Known mode: `{args.known_mode}`",
        f"Joint mode: `{args.joint_mode}`",
        f"Zero mode: `{args.zero_mode}`",
        "",
        "Outputs:",
    ]
    for output in outputs:
        lines.append(f"- `{output.name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _configure_matplotlib()
    rows, arrays, metadata = _read_inputs(args)
    modes = [str(args.known_mode), str(args.joint_mode), str(args.zero_mode)]
    mode_labels = {
        str(args.known_mode): "known",
        str(args.joint_mode): "joint",
        str(args.zero_mode): "zero",
    }
    rows = rows[
        np.isclose(rows["observation_scale"].to_numpy(dtype=np.float64), float(args.scale))
        & rows["observer_mode"].astype(str).isin(modes)
    ].copy()
    if rows.empty:
        raise ValueError(f"No saved predictions found for scale={args.scale:g} and modes={modes}")
    row_index = rows.index.to_numpy(dtype=int)
    rows = rows.reset_index(drop=True)
    arrays = {key: value[row_index] for key, value in arrays.items()}
    specs = _group_specs(rows, metadata)
    row_scores = _compute_row_scores(rows, arrays, specs, modes=modes)
    summary = _summarize_row_scores(row_scores)
    contrasts = _contrast_summary(
        summary,
        row_scores,
        known_mode=str(args.known_mode),
        joint_mode=str(args.joint_mode),
        zero_mode=str(args.zero_mode),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    row_scores_path = args.out_dir / "feature_group_row_scores.csv"
    summary_path = args.out_dir / "feature_group_r2_summary.csv"
    contrasts_path = args.out_dir / "feature_group_contrasts.csv"
    row_scores.to_csv(row_scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    figures = [
        _plot_channel_contrasts(contrasts, args.out_dir, dpi=int(args.dpi)),
        _plot_channel_observer_scores(
            summary,
            args.out_dir,
            modes=modes,
            mode_labels=mode_labels,
            dpi=int(args.dpi),
        ),
        _plot_band_orientation_heatmap(contrasts, args.out_dir, dpi=int(args.dpi)),
    ]
    outputs = [row_scores_path, summary_path, contrasts_path, *figures]
    _write_readme(args.out_dir / "README.md", args=args, outputs=outputs)
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
