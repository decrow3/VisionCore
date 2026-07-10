"""Plot diagnostics for the unified Figure 4 feature-observer R2_cv artifacts."""

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


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOUR_C_DIR = REPO_ROOT / "outputs" / "figure4_joint_decoder_candidate_free_larger_r2cv_affine_fix_known"
DEFAULT_FOUR_B_DIR = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_latent_information_r2cv_smoke16"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "figure4_unified_feature_observer_plots"


OBSERVER_LABELS = {
    "hidden_joint_forward_model": "hidden joint",
    "zero_static": "zero static",
    "response_only": "response only",
    "feature_conditioned_tau_interactions": "estimated tau x response",
    "pose_known_nested_tau_interactions": "recorded tau residual",
    "pose_known_forward_model": "recorded tau forward",
    "true_tau_interactions": "true tau interactions",
    "pose_aware_flat": "pose aware flat",
    "pose_blind_mean": "pose blind mean",
    "pose_aware_delta_flat": "pose aware delta",
    "pose_blind_delta_mean": "pose blind delta",
}

OBSERVER_COLORS = {
    "hidden_joint_forward_model": "#0f766e",
    "zero_static": "#b45309",
    "response_only": "#4b5563",
    "feature_conditioned_tau_interactions": "#7c3aed",
    "pose_known_nested_tau_interactions": "#be123c",
    "pose_known_forward_model": "#dc2626",
    "true_tau_interactions": "#64748b",
}

CONTRAST_COLORS = {
    "joint_minus_zero": "#0f766e",
    "joint_minus_response": "#2563eb",
    "known_minus_zero": "#be123c",
    "known_minus_joint": "#7c2d12",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-c-dir", type=Path, default=DEFAULT_FOUR_C_DIR)
    parser.add_argument("--four-b-dir", type=Path, default=DEFAULT_FOUR_B_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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


def _label_observer(name: str) -> str:
    return OBSERVER_LABELS.get(str(name), str(name).replace("_", " "))


def _scale_label(value: object) -> str:
    text = str(value)
    if text == "all":
        return "all"
    try:
        return f"{float(text):g}x"
    except ValueError:
        return text


def _prior_label(value: object) -> str:
    text = str(value)
    text = text.replace("axis_edge_", "")
    text = text.replace("_", " ")
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
    pad = (hi - lo) * float(pad_fraction)
    return lo - pad, hi + pad


def plot_four_c_observer_ranking(summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path:
    all_rows = summary[
        summary["observation_scale"].astype(str).eq("all")
        & summary["prior_family"].astype(str).eq("all")
    ].copy()
    if all_rows.empty:
        raise ValueError("4C summary has no all/all rows")
    all_rows = all_rows.sort_values("R2_cv", ascending=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    y = np.arange(len(all_rows))
    colors = [OBSERVER_COLORS.get(mode, "#6b7280") for mode in all_rows["observer_mode"]]
    ax.barh(y, all_rows["R2_cv"], color=colors, alpha=0.92)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([_label_observer(mode) for mode in all_rows["observer_mode"]])
    ax.set_xlabel("pooled held-out R2_cv")
    ax.set_title("4C candidate-free observer ranking", loc="left", pad=18)
    ax.text(
        0.0,
        1.005,
        "Higher is better. Negative values are worse than the train-fold feature mean.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    lo, hi = _finite_limits(all_rows["R2_cv"], pad_fraction=0.12)
    ax.set_xlim(min(lo, -0.2), max(hi, 0.2))
    for yi, value in zip(y, all_rows["R2_cv"]):
        ax.text(
            value + 0.04,
            yi,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#111827",
        )
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    return _save(fig, out_dir, "figure4c_observer_r2cv_ranking", dpi=dpi)[0]


def plot_four_c_gate_contrasts(gate: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path:
    all_row = gate[gate["group_kind"].astype(str).eq("all")]
    if all_row.empty:
        raise ValueError("Gate table has no all row")
    row = all_row.iloc[0]
    contrasts = [
        ("joint_minus_zero", "joint - zero"),
        ("joint_minus_response", "joint - response"),
        ("known_minus_zero", "known - zero"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 3.9), constrained_layout=True)
    y = np.arange(len(contrasts))
    values = []
    lower = []
    upper = []
    colors = []
    for key, _label in contrasts:
        value = float(row.get(key, np.nan))
        values.append(value)
        lo = float(row.get(f"{key}_ci_low", np.nan))
        hi = float(row.get(f"{key}_ci_high", np.nan))
        lower.append(value - lo if np.isfinite(lo) else 0.0)
        upper.append(hi - value if np.isfinite(hi) else 0.0)
        colors.append(CONTRAST_COLORS.get(key, "#4b5563"))

    ax.barh(y, values, color=colors, alpha=0.9)
    xerr = np.asarray([lower, upper], dtype=np.float64)
    ax.errorbar(values, y, xerr=xerr, fmt="none", ecolor="#111827", elinewidth=1.0, capsize=4)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _key, label in contrasts])
    ax.invert_yaxis()
    ax.set_xlabel("R2_cv contrast")
    ax.set_title("4C promotion gate contrasts", loc="left", pad=18)
    known = _label_observer(str(row.get("known_mode", "known")))
    ax.text(
        0.0,
        1.005,
        f"Known reference = {known}. Gap is not reportable if known - zero is not positive.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    lo, hi = _finite_limits(values + [float(row.get(f"{key}_ci_low", np.nan)) for key, _ in contrasts] + [float(row.get(f"{key}_ci_high", np.nan)) for key, _ in contrasts])
    ax.set_xlim(min(lo, -0.4), max(hi, 0.4))
    for yi, value in zip(y, values):
        ha = "left" if value >= 0 else "right"
        offset = 0.05 if value >= 0 else -0.05
        ax.text(value + offset, yi, f"{value:+.2f}", ha=ha, va="center", fontsize=9)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    return _save(fig, out_dir, "figure4c_gate_contrasts", dpi=dpi)[0]


def plot_four_c_scale_prior(summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path:
    grouped = summary[~summary["observation_scale"].astype(str).eq("all")].copy()
    if grouped.empty:
        raise ValueError("4C summary has no scale/prior rows")
    all_rows = summary[summary["observation_scale"].astype(str).eq("all")].copy()
    order = (
        all_rows.sort_values("R2_cv", ascending=False)["observer_mode"].astype(str).tolist()
        if not all_rows.empty
        else sorted(grouped["observer_mode"].astype(str).unique())
    )
    col_frame = grouped[["observation_scale", "prior_family"]].drop_duplicates()
    col_frame = col_frame.sort_values(["observation_scale", "prior_family"], key=lambda s: s.astype(str))
    columns = list(col_frame.itertuples(index=False, name=None))
    matrix = np.full((len(order), len(columns)), np.nan, dtype=np.float64)
    for i, mode in enumerate(order):
        for j, (scale, prior) in enumerate(columns):
            subset = grouped[
                grouped["observer_mode"].astype(str).eq(mode)
                & grouped["observation_scale"].astype(str).eq(str(scale))
                & grouped["prior_family"].astype(str).eq(str(prior))
            ]
            if not subset.empty:
                matrix[i, j] = float(subset["R2_cv"].iloc[0])

    fig, ax = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.min(finite)) if finite.size else -1.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_label_observer(mode) for mode in order])
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        [f"{_scale_label(scale)}\n{_prior_label(prior)}" for scale, prior in columns],
        rotation=0,
        ha="center",
    )
    ax.set_title("4C R2_cv across motion scale and prior family")
    ax.set_xlabel("condition")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value < (vmin + vmax) / 2.0 else "#111827"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("pooled held-out R2_cv")
    return _save(fig, out_dir, "figure4c_r2cv_by_scale_prior", dpi=dpi)[0]


def _candidate_order(candidates: Iterable[str]) -> list[str]:
    preferred = ["static", "real_drift_axis", "edge", "edge_orthogonal", "random_axis_0"]
    observed = [str(c) for c in candidates]
    out = [candidate for candidate in preferred if candidate in observed]
    out.extend(sorted(candidate for candidate in observed if candidate not in set(out)))
    return out


def plot_four_b_smoke(summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path:
    if summary.empty:
        raise ValueError("4B summary is empty")
    pca_values = sorted(summary["pca_k"].dropna().astype(int).unique().tolist())
    candidates = _candidate_order(summary["candidate"].astype(str).unique())
    observers = [
        observer
        for observer in [
            "pose_aware_flat",
            "pose_blind_mean",
            "pose_aware_delta_flat",
            "pose_blind_delta_mean",
        ]
        if observer in set(summary["observer"].astype(str))
    ]
    if not observers:
        observers = sorted(summary["observer"].astype(str).unique())

    fig, axes = plt.subplots(
        1,
        len(pca_values),
        figsize=(5.4 * len(pca_values), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    finite = summary["R2_cv"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    vmin = float(np.min(finite)) if finite.size else -1.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    for panel_index, (ax, k) in enumerate(zip(axes[0], pca_values)):
        subset = summary[summary["pca_k"].astype(int).eq(int(k))]
        matrix = np.full((len(observers), len(candidates)), np.nan, dtype=np.float64)
        for i, observer in enumerate(observers):
            for j, candidate in enumerate(candidates):
                rows = subset[
                    subset["observer"].astype(str).eq(observer)
                    & subset["candidate"].astype(str).eq(candidate)
                ]
                if not rows.empty:
                    matrix[i, j] = float(rows["R2_cv"].iloc[0])
        im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"4B smoke R2_cv, pca_k={k}")
        ax.set_xticks(np.arange(len(candidates)))
        ax.set_xticklabels([candidate.replace("_", "\n") for candidate in candidates], fontsize=8)
        ax.set_yticks(np.arange(len(observers)))
        if panel_index == 0:
            ax.set_yticklabels([_label_observer(observer) for observer in observers])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    color = "white" if value < (vmin + vmax) / 2.0 else "#111827"
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("pooled held-out R2_cv")
    fig.suptitle("4B replay: shared score contract smoke check", fontweight="bold")
    return _save(fig, out_dir, "figure4b_r2cv_smoke_heatmap", dpi=dpi)[0]


def _observer_order_from_trials(trials: pd.DataFrame) -> list[str]:
    preferred = [
        "true_tau_interactions",
        "pose_known_nested_tau_interactions",
        "pose_known_forward_model",
        "hidden_joint_forward_model",
        "zero_static",
        "response_only",
    ]
    observed = set(trials["observer_mode"].astype(str))
    out = [mode for mode in preferred if mode in observed]
    out.extend(sorted(mode for mode in observed if mode not in set(out)))
    return out


def plot_norm_calibration(trials: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path | None:
    required = {"observer_mode", "feature_true_norm", "feature_pred_norm"}
    if not required.issubset(trials.columns):
        return None
    order = _observer_order_from_trials(trials)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    for mode in order:
        subset = trials[trials["observer_mode"].astype(str).eq(mode)]
        x = subset["feature_true_norm"].to_numpy(dtype=np.float64)
        y = subset["feature_pred_norm"].to_numpy(dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        if not np.any(good):
            continue
        ax.scatter(
            x[good],
            y[good],
            s=16,
            alpha=0.48,
            color=OBSERVER_COLORS.get(mode, "#6b7280"),
            label=_label_observer(mode),
        )
    lo, hi = _finite_limits(
        list(trials["feature_true_norm"].to_numpy(dtype=np.float64))
        + list(trials["feature_pred_norm"].to_numpy(dtype=np.float64))
    )
    lo = max(0.0, lo)
    ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1.0, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true feature norm")
    ax.set_ylabel("predicted feature norm")
    ax.set_title("Feature magnitude calibration")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return _save(fig, out_dir, "figure4c_predicted_norm_vs_true_norm", dpi=dpi)[0]


def plot_cosine_vs_r2(trials: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path | None:
    required = {"observer_mode", "feature_cosine", "feature_r2_row_diagnostic"}
    if not required.issubset(trials.columns):
        return None
    order = _observer_order_from_trials(trials)
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    for mode in order:
        subset = trials[trials["observer_mode"].astype(str).eq(mode)]
        x = subset["feature_cosine"].to_numpy(dtype=np.float64)
        y = subset["feature_r2_row_diagnostic"].to_numpy(dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        if not np.any(good):
            continue
        ax.scatter(
            x[good],
            y[good],
            s=15,
            alpha=0.45,
            color=OBSERVER_COLORS.get(mode, "#6b7280"),
            label=_label_observer(mode),
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel("feature cosine")
    ax.set_ylabel("row diagnostic R2_cv")
    ax.set_title("Metric mismatch diagnostic")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(color="#e5e7eb", linewidth=0.8)
    return _save(fig, out_dir, "figure4c_feature_cosine_vs_r2cv", dpi=dpi)[0]


def _pooled_r2(frame: pd.DataFrame, *, sse_col: str = "feature_sse", sst_col: str = "feature_sst_train_baseline") -> float:
    if frame.empty or sse_col not in frame.columns or sst_col not in frame.columns:
        return float("nan")
    sse = float(np.nansum(frame[sse_col].to_numpy(dtype=np.float64)))
    sst = float(np.nansum(frame[sst_col].to_numpy(dtype=np.float64)))
    return float(1.0 - sse / sst) if np.isfinite(sst) and sst > 1e-12 else float("nan")


def plot_known_correction_delta(
    trials: pd.DataFrame,
    out_dir: Path,
    *,
    known_mode: str = "pose_known_nested_tau_interactions",
    dpi: int,
) -> Path | None:
    required = {"observer_mode", "fold", "observation_scale", "feature_sse", "feature_sst_train_baseline"}
    if not required.issubset(trials.columns) or known_mode not in set(trials["observer_mode"].astype(str)):
        return None
    rows = []
    for (fold, scale), group in trials.groupby(["fold", "observation_scale"], dropna=False):
        known = _pooled_r2(group[group["observer_mode"].astype(str).eq(known_mode)])
        response = _pooled_r2(group[group["observer_mode"].astype(str).eq("response_only")])
        rows.append({"fold": fold, "observation_scale": scale, "delta": known - response})
    frame = pd.DataFrame(rows).dropna()
    if frame.empty:
        return None
    scales = sorted(frame["observation_scale"].unique(), key=lambda value: float(value))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    width = 0.13
    for i, scale in enumerate(scales):
        sub = frame[frame["observation_scale"].eq(scale)].sort_values("fold")
        x = np.arange(sub.shape[0]) + (i - (len(scales) - 1) / 2.0) * width
        ax.bar(x, sub["delta"], width=width, label=_scale_label(scale), alpha=0.86)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(np.arange(frame["fold"].nunique()))
    ax.set_xticklabels(sorted(frame["fold"].unique()))
    ax.set_xlabel("outer fold")
    ax.set_ylabel("R2_cv known residual - response-only")
    ax.set_title("Known-tau correction delta by fold and scale")
    ax.legend(frameon=False, title="scale")
    return _save(fig, out_dir, "figure4c_known_correction_delta_by_fold_scale", dpi=dpi)[0]


def plot_shrinkage_comparison(models: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path | None:
    required = {"observer_mode", "fold", "residual_shrinkage_lambda", "residual_shrinkage_lambda_cosine_selected"}
    if not required.issubset(models.columns):
        return None
    frame = models[models["observer_mode"].astype(str).eq("pose_known_nested_tau_interactions")].copy()
    frame = frame.dropna(subset=["residual_shrinkage_lambda", "residual_shrinkage_lambda_cosine_selected"])
    if frame.empty:
        return None
    frame = frame.sort_values("fold")
    x = np.arange(frame.shape[0])
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.bar(x - 0.18, frame["residual_shrinkage_lambda"], width=0.34, color="#0f766e", label="R2-selected alpha")
    ax.bar(
        x + 0.18,
        frame["residual_shrinkage_lambda_cosine_selected"],
        width=0.34,
        color="#be123c",
        label="cosine-selected alpha",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(frame["fold"].astype(int).astype(str))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("selected shrinkage alpha")
    ax.set_title("Known-tau shrinkage: cosine vs R2_cv validation")
    ax.legend(frameon=False)
    return _save(fig, out_dir, "figure4c_shrinkage_cosine_vs_r2_alpha", dpi=dpi)[0]


def plot_forward_residual_by_scale(trials: pd.DataFrame, out_dir: Path, *, dpi: int) -> Path | None:
    required = {"observer_mode", "observation_scale", "forward_response_residual_mse"}
    if not required.issubset(trials.columns):
        return None
    modes = ["pose_known_forward_model", "hidden_joint_forward_model"]
    frame = trials[trials["observer_mode"].astype(str).isin(modes)].copy()
    frame = frame[np.isfinite(frame["forward_response_residual_mse"].to_numpy(dtype=np.float64))]
    if frame.empty:
        return None
    summary = (
        frame.groupby(["observer_mode", "observation_scale"], as_index=False)
        .agg(mean_forward_response_residual_mse=("forward_response_residual_mse", "mean"))
    )
    scales = sorted(summary["observation_scale"].unique(), key=lambda value: float(value))
    x = np.arange(len(scales))
    fig, ax = plt.subplots(figsize=(6.8, 3.9), constrained_layout=True)
    width = 0.34
    for i, mode in enumerate(modes):
        sub = summary[summary["observer_mode"].astype(str).eq(mode)]
        vals = [
            float(sub[sub["observation_scale"].eq(scale)]["mean_forward_response_residual_mse"].iloc[0])
            if np.any(sub["observation_scale"].eq(scale))
            else np.nan
            for scale in scales
        ]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=_label_observer(mode), color=OBSERVER_COLORS.get(mode))
    ax.set_xticks(x)
    ax.set_xticklabels([_scale_label(scale) for scale in scales])
    ax.set_xlabel("motion scale")
    ax.set_ylabel("mean compact residual MSE")
    ax.set_title("Forward-model residual: true tau vs optimized tau")
    ax.legend(frameon=False)
    return _save(fig, out_dir, "figure4c_forward_residual_by_scale", dpi=dpi)[0]


def _write_index(out_dir: Path, paths: list[Path]) -> None:
    lines = [
        "# Unified Feature Observer Diagnostic Plots",
        "",
        "These plots summarize the current R2_cv verification artifacts.",
        "",
    ]
    descriptions = {
        "figure4c_observer_r2cv_ranking.png": "All-scale 4C observer ranking. The hidden-joint observer is best in this pass, but all scores are negative relative to the train-fold feature mean.",
        "figure4c_gate_contrasts.png": "All-scale gate contrasts. The joint observer beats zero and response-only, while the current known-trajectory reference fails as a ceiling.",
        "figure4c_r2cv_by_scale_prior.png": "4C condition heatmap by motion scale and prior family.",
        "figure4b_r2cv_smoke_heatmap.png": "4B smoke replay heatmap showing that the BackImage screen now emits the same pooled R2_cv score.",
        "figure4c_predicted_norm_vs_true_norm.png": "Predicted feature norm versus true feature norm, to reveal output-scale mismatch.",
        "figure4c_feature_cosine_vs_r2cv.png": "Per-row feature cosine versus row diagnostic R2_cv, to expose metric mismatch.",
        "figure4c_known_correction_delta_by_fold_scale.png": "Known-tau residual correction gain/loss relative to response-only by fold and scale.",
        "figure4c_shrinkage_cosine_vs_r2_alpha.png": "Shrinkage selected by pooled R2_cv validation compared with the alpha that cosine validation would select.",
        "figure4c_forward_residual_by_scale.png": "Compact-response residual of true-tau forward inversion versus hidden optimized tau by scale.",
    }
    for path in paths:
        lines.append(f"- `{path.name}`: {descriptions.get(path.name, '')}")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _configure_matplotlib()

    four_c_summary = pd.read_csv(args.four_c_dir / "linear_synthetic_prior_feature_observer_summary.csv")
    four_c_gate = pd.read_csv(args.four_c_dir / "gates" / "unified_feature_observer_gate_table.csv")
    four_c_trials = pd.read_csv(args.four_c_dir / "linear_synthetic_prior_feature_observer_trials.csv")
    four_c_models = pd.read_csv(args.four_c_dir / "linear_synthetic_prior_feature_observer_models.csv")
    four_b_summary = pd.read_csv(args.four_b_dir / "decode_summary_by_candidate.csv")

    paths: list[Path | None] = [
        plot_four_c_observer_ranking(four_c_summary, args.out_dir, dpi=int(args.dpi)),
        plot_four_c_gate_contrasts(four_c_gate, args.out_dir, dpi=int(args.dpi)),
        plot_four_c_scale_prior(four_c_summary, args.out_dir, dpi=int(args.dpi)),
        plot_norm_calibration(four_c_trials, args.out_dir, dpi=int(args.dpi)),
        plot_cosine_vs_r2(four_c_trials, args.out_dir, dpi=int(args.dpi)),
        plot_known_correction_delta(four_c_trials, args.out_dir, dpi=int(args.dpi)),
        plot_shrinkage_comparison(four_c_models, args.out_dir, dpi=int(args.dpi)),
        plot_forward_residual_by_scale(four_c_trials, args.out_dir, dpi=int(args.dpi)),
        plot_four_b_smoke(four_b_summary, args.out_dir, dpi=int(args.dpi)),
    ]
    written_paths = [path for path in paths if path is not None]
    _write_index(args.out_dir, written_paths)
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
