"""Plot one-scale calibration diagnostics for the Figure 4C observer run.

This script is intentionally narrow: it uses the aggregate trial artifact from
the current candidate-free feature observer and asks whether the nominal 1x
joint advantage is a direction/cosine effect, a magnitude calibration effect,
or an artifact of the scalar-gain diagnostic.
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

from declan.feature_recovery_scores import R2_CV_METHOD, pooled_multioutput_r2_from_sse_sst


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOUR_C_DIR = REPO_ROOT / "outputs" / "figure4_joint_decoder_known_residual_r2cal_affine_v4"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "figure4_one_scale_calibration_diagnostics_v4"

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
CONTRAST_LABELS = {
    "joint_minus_zero": "joint - zero",
    "known_minus_joint": "known - joint",
    "known_minus_zero": "known - zero",
}
CONTRAST_COLORS = {
    "joint_minus_zero": "#0f766e",
    "known_minus_joint": "#7c2d12",
    "known_minus_zero": "#be123c",
}
SCORE_SPECS = {
    "uncalibrated": ("feature_sse", "feature_sst_train_baseline", "uncalibrated"),
    "cv_scalar_gain": (
        "feature_sse_cv_gain_calibrated",
        "feature_sst_cv_gain_calibrated_train_baseline",
        "CV scalar gain",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-c-dir", type=Path, default=DEFAULT_FOUR_C_DIR)
    parser.add_argument("--trials-csv", type=Path, default=None)
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
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return paths


def _mode_label(mode: str, mode_labels: dict[str, str]) -> str:
    return mode_labels.get(mode, mode.replace("_", " "))


def _prior_label(prior_family: object) -> str:
    text = str(prior_family).replace("axis_edge_", "").replace("_", " ")
    return text


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
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _trial_key_columns(frame: pd.DataFrame) -> list[str]:
    candidates = ["table_index", "trial_id", "true_source_row"]
    cols = [col for col in candidates if col in frame.columns]
    if cols:
        return cols
    return [col for col in ("response_cache_path",) if col in frame.columns]


def _pooled_r2(frame: pd.DataFrame, *, sse_col: str, sst_col: str) -> float:
    score = pooled_multioutput_r2_from_sse_sst(
        frame[sse_col].to_numpy(dtype=np.float64),
        frame[sst_col].to_numpy(dtype=np.float64),
    )
    return float(score.r2)


def _score_modes(
    frame: pd.DataFrame,
    *,
    modes: list[str],
    sse_col: str,
    sst_col: str,
) -> dict[str, float]:
    return {
        mode: _pooled_r2(
            frame[frame["observer_mode"].astype(str).eq(mode)],
            sse_col=sse_col,
            sst_col=sst_col,
        )
        for mode in modes
    }


def _bootstrap_contrasts(
    frame: pd.DataFrame,
    *,
    known_mode: str,
    joint_mode: str,
    zero_mode: str,
    sse_col: str,
    sst_col: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {"bootstrap_n": 0}
    key_cols = _trial_key_columns(frame)
    if not key_cols:
        return {"bootstrap_n": 0, "bootstrap_skipped_no_shared_trial_keys": 1}

    modes = [known_mode, joint_mode, zero_mode]
    tmp = frame[frame["observer_mode"].astype(str).isin(modes)].copy()
    grouped = (
        tmp.groupby(key_cols + ["observer_mode"], dropna=False, sort=False)[[sse_col, sst_col]]
        .sum()
        .reset_index()
    )
    wide_sse = grouped.pivot(index=key_cols, columns="observer_mode", values=sse_col)
    wide_sst = grouped.pivot(index=key_cols, columns="observer_mode", values=sst_col)
    complete = wide_sse[modes].notna().all(axis=1) & wide_sst[modes].notna().all(axis=1)
    wide_sse = wide_sse.loc[complete]
    wide_sst = wide_sst.loc[complete]
    if wide_sse.empty:
        return {"bootstrap_n": 0, "bootstrap_skipped_no_complete_triplets": 1}

    rng = np.random.default_rng(seed)
    n = len(wide_sse)
    values: dict[str, list[float]] = {
        "joint_minus_zero": [],
        "known_minus_joint": [],
        "known_minus_zero": [],
    }
    sse_arrays = {mode: wide_sse[mode].to_numpy(dtype=np.float64) for mode in modes}
    sst_arrays = {mode: wide_sst[mode].to_numpy(dtype=np.float64) for mode in modes}
    for _ in range(int(n_bootstrap)):
        sample = rng.integers(0, n, size=n)
        scores = {}
        for mode in modes:
            sse = float(np.nansum(sse_arrays[mode][sample]))
            sst = float(np.nansum(sst_arrays[mode][sample]))
            scores[mode] = float(1.0 - sse / sst) if np.isfinite(sst) and sst > 1e-12 else np.nan
        values["joint_minus_zero"].append(float(scores[joint_mode] - scores[zero_mode]))
        values["known_minus_joint"].append(float(scores[known_mode] - scores[joint_mode]))
        values["known_minus_zero"].append(float(scores[known_mode] - scores[zero_mode]))

    out: dict[str, float] = {"bootstrap_n": int(n_bootstrap), "bootstrap_trial_units": int(n)}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[f"{name}_ci_low"] = np.nan
            out[f"{name}_ci_high"] = np.nan
            out[f"{name}_bootstrap_mean"] = np.nan
        else:
            out[f"{name}_ci_low"] = float(np.quantile(arr, 0.025))
            out[f"{name}_ci_high"] = float(np.quantile(arr, 0.975))
            out[f"{name}_bootstrap_mean"] = float(np.mean(arr))
    return out


def _condition_summary(
    frame: pd.DataFrame,
    *,
    modes: list[str],
    mode_labels: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for prior, prior_group in frame.groupby("prior_family", dropna=False, sort=True):
        for mode in modes:
            subset = prior_group[prior_group["observer_mode"].astype(str).eq(mode)].copy()
            if subset.empty:
                continue
            finite_score = subset[
                np.isfinite(subset["feature_sse"])
                & np.isfinite(subset["feature_sst_train_baseline"])
                & np.isfinite(subset["feature_sse_cv_gain_calibrated"])
                & np.isfinite(subset["feature_sst_cv_gain_calibrated_train_baseline"])
            ]
            true_norm = subset["feature_true_norm"].to_numpy(dtype=np.float64)
            pred_norm = subset["feature_pred_norm"].to_numpy(dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_ratio = pred_norm / true_norm
                cv_norm_ratio = subset["feature_pred_norm_cv_gain_calibrated"].to_numpy(dtype=np.float64) / true_norm
            row = {
                "prior_family": prior,
                "prior_label": _prior_label(prior),
                "observer_mode": mode,
                "observer_label": _mode_label(mode, mode_labels),
                "n_rows": int(len(subset)),
                "n_score_rows": int(len(finite_score)),
                "mean_feature_cosine": float(np.nanmean(subset["feature_cosine"])),
                "median_feature_cosine": float(np.nanmedian(subset["feature_cosine"])),
                "median_pred_norm": float(np.nanmedian(pred_norm)),
                "median_true_norm": float(np.nanmedian(true_norm)),
                "median_pred_true_norm_ratio": float(np.nanmedian(norm_ratio)),
                "median_cv_gain_pred_true_norm_ratio": float(np.nanmedian(cv_norm_ratio)),
                "median_cv_scalar_gain": float(np.nanmedian(subset["feature_cv_scalar_gain"])),
                "R2_cv_uncalibrated": _pooled_r2(
                    finite_score,
                    sse_col="feature_sse",
                    sst_col="feature_sst_train_baseline",
                ),
                "R2_cv_cv_scalar_gain": _pooled_r2(
                    finite_score,
                    sse_col="feature_sse_cv_gain_calibrated",
                    sst_col="feature_sst_cv_gain_calibrated_train_baseline",
                ),
                "score_method": R2_CV_METHOD,
                "score_space": "locked_train_normalized_feature_space",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _contrast_summary(
    frame: pd.DataFrame,
    *,
    modes: list[str],
    known_mode: str,
    joint_mode: str,
    zero_mode: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for score_index, (score_name, (sse_col, sst_col, score_label)) in enumerate(SCORE_SPECS.items()):
        for prior_index, (prior, prior_group) in enumerate(frame.groupby("prior_family", dropna=False, sort=True)):
            finite = prior_group[np.isfinite(prior_group[sse_col]) & np.isfinite(prior_group[sst_col])].copy()
            scores = _score_modes(finite, modes=modes, sse_col=sse_col, sst_col=sst_col)
            row = {
                "prior_family": prior,
                "prior_label": _prior_label(prior),
                "score_variant": score_name,
                "score_label": score_label,
                "sse_column": sse_col,
                "sst_column": sst_col,
                "S_known": scores[known_mode],
                "S_joint": scores[joint_mode],
                "S_zero": scores[zero_mode],
                "joint_minus_zero": scores[joint_mode] - scores[zero_mode],
                "known_minus_joint": scores[known_mode] - scores[joint_mode],
                "known_minus_zero": scores[known_mode] - scores[zero_mode],
                "score_method": R2_CV_METHOD,
            }
            row.update(
                _bootstrap_contrasts(
                    finite,
                    known_mode=known_mode,
                    joint_mode=joint_mode,
                    zero_mode=zero_mode,
                    sse_col=sse_col,
                    sst_col=sst_col,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 100 * score_index + prior_index,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_distribution(
    frame: pd.DataFrame,
    *,
    modes: list[str],
    mode_labels: dict[str, str],
    value_col: str,
    y_label: str,
    title: str,
    out_dir: Path,
    stem: str,
    dpi: int,
    reference_line: float | None = None,
) -> Path:
    priors = sorted(frame["prior_family"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(priors), figsize=(5.1 * len(priors), 4.3), sharey=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    rng = np.random.default_rng(12345)
    all_values = []
    for ax, prior in zip(axes_arr, priors):
        subset = frame[frame["prior_family"].eq(prior)]
        for idx, mode in enumerate(modes):
            values = subset.loc[subset["observer_mode"].astype(str).eq(mode), value_col].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            all_values.extend(values.tolist())
            x = idx + rng.normal(0.0, 0.035, size=values.shape[0])
            ax.scatter(
                x,
                values,
                s=18,
                color=MODE_COLORS.get(mode, "#6b7280"),
                alpha=0.55,
                linewidth=0.0,
            )
            if values.size:
                median = float(np.median(values))
                ax.plot([idx - 0.24, idx + 0.24], [median, median], color="#111827", linewidth=2.0)
        if reference_line is not None:
            ax.axhline(reference_line, color="#111827", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_title(_prior_label(prior), loc="left")
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels([_mode_label(mode, mode_labels) for mode in modes], rotation=20, ha="right")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    lo, hi = _finite_limits(all_values, pad_fraction=0.12)
    if reference_line is not None:
        lo = min(lo, reference_line - 0.1 * max(1.0, abs(reference_line)))
        hi = max(hi, reference_line + 0.1 * max(1.0, abs(reference_line)))
    for ax in axes_arr:
        ax.set_ylim(lo, hi)
    axes_arr[0].set_ylabel(y_label)
    fig.suptitle(title, x=0.01, y=1.03, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, stem, dpi=dpi)[0]


def _plot_r2_comparison(
    summary: pd.DataFrame,
    *,
    modes: list[str],
    mode_labels: dict[str, str],
    out_dir: Path,
    dpi: int,
) -> Path:
    priors = sorted(summary["prior_family"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(priors), figsize=(5.5 * len(priors), 4.6), sharey=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    all_values = []
    width = 0.36
    for ax, prior in zip(axes_arr, priors):
        subset = summary[summary["prior_family"].eq(prior)]
        x = np.arange(len(modes), dtype=float)
        uncal = []
        cal = []
        for mode in modes:
            row = subset[subset["observer_mode"].astype(str).eq(mode)]
            uncal.append(float(row["R2_cv_uncalibrated"].iloc[0]) if not row.empty else np.nan)
            cal.append(float(row["R2_cv_cv_scalar_gain"].iloc[0]) if not row.empty else np.nan)
        all_values.extend(uncal)
        all_values.extend(cal)
        ax.bar(x - width / 2, uncal, width=width, color="#94a3b8", label="uncalibrated")
        ax.bar(x + width / 2, cal, width=width, color="#2563eb", label="CV scalar gain")
        ax.axhline(0.0, color="#111827", linewidth=0.9)
        ax.set_title(_prior_label(prior), loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([_mode_label(mode, mode_labels) for mode in modes], rotation=20, ha="right")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        for xi, value in zip(x - width / 2, uncal):
            if np.isfinite(value):
                ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        for xi, value in zip(x + width / 2, cal):
            if np.isfinite(value):
                ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    lo, hi = _finite_limits(all_values, pad_fraction=0.14)
    for ax in axes_arr:
        ax.set_ylim(min(lo, -0.2), max(hi, 0.2))
    axes_arr[0].set_ylabel("pooled R2_cv")
    axes_arr[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("1x uncalibrated versus scalar-gain calibrated score", x=0.01, y=1.03, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "figure4c_1x_uncalibrated_vs_cv_gain_r2", dpi=dpi)[0]


def _plot_contrasts(
    contrasts: pd.DataFrame,
    *,
    out_dir: Path,
    dpi: int,
) -> Path:
    priors = sorted(contrasts["prior_family"].dropna().unique().tolist())
    variants = ["uncalibrated", "cv_scalar_gain"]
    fig, axes = plt.subplots(
        len(variants),
        len(priors),
        figsize=(5.7 * len(priors), 3.6 * len(variants)),
        sharex=False,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(len(variants), len(priors))
    contrast_order = ["joint_minus_zero", "known_minus_joint", "known_minus_zero"]
    for row_idx, variant in enumerate(variants):
        for col_idx, prior in enumerate(priors):
            ax = axes_arr[row_idx, col_idx]
            subset = contrasts[
                contrasts["score_variant"].eq(variant)
                & contrasts["prior_family"].eq(prior)
            ]
            if subset.empty:
                ax.set_axis_off()
                continue
            data = subset.iloc[0]
            values = [float(data[name]) for name in contrast_order]
            lows = [float(data.get(f"{name}_ci_low", np.nan)) for name in contrast_order]
            highs = [float(data.get(f"{name}_ci_high", np.nan)) for name in contrast_order]
            y = np.arange(len(contrast_order))
            colors = [CONTRAST_COLORS[name] for name in contrast_order]
            ax.barh(y, values, color=colors, alpha=0.92)
            lower = [value - low if np.isfinite(low) else 0.0 for value, low in zip(values, lows)]
            upper = [high - value if np.isfinite(high) else 0.0 for value, high in zip(values, highs)]
            ax.errorbar(values, y, xerr=np.asarray([lower, upper]), fmt="none", ecolor="#111827", capsize=4)
            ax.axvline(0.0, color="#111827", linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels([CONTRAST_LABELS[name] for name in contrast_order])
            ax.invert_yaxis()
            label = str(data["score_label"])
            ax.set_title(f"{_prior_label(prior)} | {label}", loc="left")
            ax.set_xlabel("R2_cv contrast")
            all_vals = values + lows + highs
            lo, hi = _finite_limits(all_vals, pad_fraction=0.16)
            ax.set_xlim(min(lo, -0.15), max(hi, 0.15))
            ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
            for yi, value in zip(y, values):
                ha = "left" if value >= 0 else "right"
                offset = 0.03 if value >= 0 else -0.03
                ax.text(
                    value + offset,
                    yi,
                    f"{value:+.2f}",
                    ha=ha,
                    va="center",
                    fontsize=8,
                    zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                )
    fig.suptitle("1x known/joint/zero contrasts with bootstrap intervals", x=0.01, y=1.02, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "figure4c_1x_contrasts_with_ci", dpi=dpi)[0]


def _write_feature_group_status(path: Path) -> pd.DataFrame:
    status = pd.DataFrame(
        [
            {
                "requested_diagnostic": "per_feature_group_R2",
                "requested_groups": "Re, Im, magnitude, scale/orientation",
                "available_from_current_artifact": False,
                "reason": (
                    "The trial CSV stores row-level aggregate SSE/SST, cosine, and norms after the locked "
                    "feature transform. It does not store per-dimension z_true/z_hat vectors or raw pyramid "
                    "channel identities, so group-specific R2 and diagonal calibration cannot be recovered "
                    "without rerunning or saving prediction arrays."
                ),
                "needed_next_artifact": (
                    "Save out-of-fold z_true, z_hat, train_mean, fold, observer_mode, prior_family, "
                    "observation_scale, and feature metadata mapping dimensions to Re/Im/magnitude and "
                    "scale/orientation groups."
                ),
            },
            {
                "requested_diagnostic": "train_fold_diagonal_calibration",
                "requested_groups": "feature_dimension",
                "available_from_current_artifact": False,
                "reason": (
                    "A diagonal gain requires per-feature predicted and true values on train/held-out folds; "
                    "the current artifact only has norms and dot products."
                ),
                "needed_next_artifact": (
                    "Persist per-dimension predictions or add an in-run diagnostic before reducing to "
                    "row-level SSE/SST."
                ),
            },
        ]
    )
    status.to_csv(path, index=False)
    return status


def _write_readme(
    path: Path,
    *,
    args: argparse.Namespace,
    trials_csv: Path,
    summary_csv: Path,
    contrasts_csv: Path,
    feature_group_status_csv: Path,
    figure_paths: list[Path],
) -> None:
    lines = [
        "# One-scale calibration diagnostics",
        "",
        f"Input trial artifact: `{trials_csv}`",
        f"Scale filter: `{float(args.scale):g}x`",
        f"Known mode: `{args.known_mode}`",
        f"Joint mode: `{args.joint_mode}`",
        f"Zero mode: `{args.zero_mode}`",
        "",
        "The figures ask why scalar gain calibration erases the nominal 1x joint advantage.",
        "Scores are pooled multi-output `R2_cv` in the locked, train-normalized feature space.",
        "",
        "Outputs:",
        f"- `{summary_csv.name}`: per-condition cosine, norm-ratio, scalar-gain, and R2 summaries.",
        f"- `{contrasts_csv.name}`: uncalibrated and CV scalar-gain known/joint/zero contrasts with bootstrap intervals.",
        f"- `{feature_group_status_csv.name}`: availability audit for per-feature-group R2 and diagonal calibration.",
    ]
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path.name}`")
    lines.extend(
        [
            "",
            "Limitation: this pass cannot compute Re/Im/magnitude or scale/orientation R2 from the existing CSV, because predictions were already reduced to aggregate row metrics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _configure_matplotlib()
    trials_csv = (
        args.trials_csv
        if args.trials_csv is not None
        else args.four_c_dir / "linear_synthetic_prior_feature_observer_trials.csv"
    )
    trials = pd.read_csv(trials_csv)
    modes = [str(args.known_mode), str(args.joint_mode), str(args.zero_mode)]
    mode_labels = {
        str(args.known_mode): "known",
        str(args.joint_mode): "joint",
        str(args.zero_mode): "zero",
    }
    required = {
        "observer_mode",
        "prior_family",
        "observation_scale",
        "feature_cosine",
        "feature_true_norm",
        "feature_pred_norm",
        "feature_sse",
        "feature_sst_train_baseline",
        "feature_sse_cv_gain_calibrated",
        "feature_sst_cv_gain_calibrated_train_baseline",
        "feature_pred_norm_cv_gain_calibrated",
        "feature_cv_scalar_gain",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"{trials_csv} lacks required columns: {missing}")

    scale_trials = trials[
        np.isclose(trials["observation_scale"].to_numpy(dtype=np.float64), float(args.scale))
        & trials["observer_mode"].astype(str).isin(modes)
    ].copy()
    if scale_trials.empty:
        raise ValueError(f"No rows found for scale={args.scale:g} and modes={modes}")
    observed_modes = set(scale_trials["observer_mode"].astype(str))
    missing_modes = [mode for mode in modes if mode not in observed_modes]
    if missing_modes:
        raise ValueError(f"Missing requested modes at scale {args.scale:g}: {missing_modes}")

    scale_trials["pred_true_norm_ratio"] = (
        scale_trials["feature_pred_norm"].to_numpy(dtype=np.float64)
        / scale_trials["feature_true_norm"].to_numpy(dtype=np.float64)
    )
    scale_trials["cv_gain_pred_true_norm_ratio"] = (
        scale_trials["feature_pred_norm_cv_gain_calibrated"].to_numpy(dtype=np.float64)
        / scale_trials["feature_true_norm"].to_numpy(dtype=np.float64)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = _condition_summary(scale_trials, modes=modes, mode_labels=mode_labels)
    contrasts = _contrast_summary(
        scale_trials,
        modes=modes,
        known_mode=str(args.known_mode),
        joint_mode=str(args.joint_mode),
        zero_mode=str(args.zero_mode),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )

    summary_csv = args.out_dir / "one_scale_condition_summary.csv"
    contrasts_csv = args.out_dir / "one_scale_contrasts.csv"
    feature_group_status_csv = args.out_dir / "feature_group_diagnostic_status.csv"
    summary.to_csv(summary_csv, index=False)
    contrasts.to_csv(contrasts_csv, index=False)
    _write_feature_group_status(feature_group_status_csv)

    figures = [
        _plot_distribution(
            scale_trials,
            modes=modes,
            mode_labels=mode_labels,
            value_col="feature_cosine",
            y_label="feature cosine",
            title=f"{args.scale:g}x feature direction diagnostic",
            out_dir=args.out_dir,
            stem="figure4c_1x_feature_cosine_by_condition",
            dpi=int(args.dpi),
        ),
        _plot_distribution(
            scale_trials,
            modes=modes,
            mode_labels=mode_labels,
            value_col="pred_true_norm_ratio",
            y_label="||zhat|| / ||z||",
            title=f"{args.scale:g}x prediction magnitude diagnostic",
            out_dir=args.out_dir,
            stem="figure4c_1x_prediction_norm_ratio",
            dpi=int(args.dpi),
            reference_line=1.0,
        ),
        _plot_distribution(
            scale_trials,
            modes=modes,
            mode_labels=mode_labels,
            value_col="cv_gain_pred_true_norm_ratio",
            y_label="gain-calibrated ||zhat|| / ||z||",
            title=f"{args.scale:g}x magnitude after cross-fold scalar gain",
            out_dir=args.out_dir,
            stem="figure4c_1x_cv_gain_prediction_norm_ratio",
            dpi=int(args.dpi),
            reference_line=1.0,
        ),
        _plot_r2_comparison(
            summary,
            modes=modes,
            mode_labels=mode_labels,
            out_dir=args.out_dir,
            dpi=int(args.dpi),
        ),
        _plot_contrasts(contrasts, out_dir=args.out_dir, dpi=int(args.dpi)),
    ]
    _write_readme(
        args.out_dir / "README.md",
        args=args,
        trials_csv=trials_csv,
        summary_csv=summary_csv,
        contrasts_csv=contrasts_csv,
        feature_group_status_csv=feature_group_status_csv,
        figure_paths=figures,
    )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {contrasts_csv}")
    print(f"Wrote {feature_group_status_csv}")
    for path in figures:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
