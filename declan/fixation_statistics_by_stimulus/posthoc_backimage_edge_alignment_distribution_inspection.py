"""Inspect BackImage drift-axis alignment relative to local image edge axes.

This posthoc script regenerates the exploratory distribution plots saved under
``backimage_edge_alignment_distribution_inspection``.  The headline signed
alignment index is ``cos(2 * drift-edge delta)``: +1 means edge-parallel motion,
0 means 45 degrees from the edge, and -1 means edge-orthogonal motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
DEFAULT_IMAGE_STRUCTURE_DIR = BASE_DIR / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
DEFAULT_DRIFT_GEOMETRY_DIR = BASE_DIR / "backimage_twin_drift_geometry_scaled_n256_twin_axis_only_yfix"
DEFAULT_OUT_DIR = BASE_DIR / "backimage_edge_alignment_distribution_inspection"

COLORS = {
    "All windows": "#31566b",
    "Reliable axes": "#2f9d8c",
    "High confidence": "#eb6a4a",
    "Objective": "#9d7bb8",
    "Null": "#8e9aa6",
}

OBJECTIVE_ORDER = [
    "raw_edge_axis",
    "optimized_pixel_isophote",
    "optimized_PA",
    "optimized_PB",
    "optimized_response_stability",
    "optimized_response_refresh_lambda_0.25",
    "optimized_refresh_only",
    "raw_gradient_axis",
    "raw_spectrum_axis",
]

OBJECTIVE_LABELS = {
    "raw_edge_axis": "raw edge",
    "optimized_pixel_isophote": "pixel isophote",
    "optimized_PA": "PA",
    "optimized_PB": "PB",
    "optimized_response_stability": "response stability",
    "optimized_response_refresh_lambda_0.25": "response refresh .25",
    "optimized_refresh_only": "refresh only",
    "raw_gradient_axis": "raw gradient",
    "raw_spectrum_axis": "raw spectrum",
}

PREDICTED_AXIS_ORDER = [
    "raw_edge_axis",
    "optimized_pixel_isophote",
    "optimized_PA",
    "optimized_PB",
    "optimized_response_stability",
    "optimized_response_refresh_lambda_0.25",
    "optimized_refresh_only",
]


def _require_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing required columns: {joined}")


def _finite(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out = out[np.isfinite(pd.to_numeric(out[col], errors="coerce"))]
    return out.copy()


def _axial_abs_delta(delta_deg: np.ndarray | pd.Series) -> np.ndarray:
    delta = np.asarray(delta_deg, dtype=np.float64)
    wrapped = (delta + 90.0) % 180.0 - 90.0
    return np.abs(wrapped)


def _signed_axial_delta(delta_deg: np.ndarray | pd.Series) -> np.ndarray:
    delta = np.asarray(delta_deg, dtype=np.float64)
    return (delta + 90.0) % 180.0 - 90.0


def _resultant_abs(signed_delta_deg: np.ndarray | pd.Series) -> float:
    delta_rad = np.radians(np.asarray(signed_delta_deg, dtype=np.float64))
    vals = np.exp(2j * delta_rad)
    return float(np.abs(np.nanmean(vals)))


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1 or int(n_boot) <= 0:
        return float(vals[0]), float(vals[0])
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _wilson_interval(count: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = float(count) / float(total)
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _binomial_ci_for_mean(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        mean = float(np.nanmean(vals)) if vals.size else float("nan")
        return mean, mean
    mean = float(vals.mean())
    half = 1.96 * float(vals.std(ddof=1)) / np.sqrt(vals.size)
    return mean - half, mean + half


def _load_edge_windows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(
        df,
        path,
        [
            "session",
            "drift_edge_delta_deg",
            "drift_edge_cos2",
            "image_orientation_coherence",
            "anisotropy",
        ],
    )
    work = _finite(
        df,
        ["drift_edge_delta_deg", "drift_edge_cos2", "image_orientation_coherence", "anisotropy"],
    )
    work["signed_edge_delta_deg"] = _signed_axial_delta(work["drift_edge_delta_deg"])
    work["abs_edge_delta_deg"] = np.abs(work["signed_edge_delta_deg"])
    return work


def _make_subsets(
    df: pd.DataFrame,
    *,
    reliable_coherence: float,
    reliable_anisotropy: float,
    high_confidence_coherence: float,
    high_confidence_anisotropy: float,
) -> dict[str, pd.DataFrame]:
    reliable = df[
        (df["image_orientation_coherence"].astype(float) >= float(reliable_coherence))
        & (df["anisotropy"].astype(float) >= float(reliable_anisotropy))
    ].copy()
    high_confidence = df[
        (df["image_orientation_coherence"].astype(float) >= float(high_confidence_coherence))
        & (df["anisotropy"].astype(float) >= float(high_confidence_anisotropy))
    ].copy()
    return {
        "All windows": df.copy(),
        "Reliable axes": reliable,
        "High confidence": high_confidence,
    }


def _edge_alignment_summary(
    subsets: dict[str, pd.DataFrame],
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, sub in subsets.items():
        session_means = sub.groupby("session", dropna=False)["drift_edge_cos2"].mean().to_numpy(dtype=np.float64)
        ci_low, ci_high = _bootstrap_ci(session_means, rng, n_boot)
        abs_delta = sub["abs_edge_delta_deg"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "subset": name,
                "n_windows": int(len(sub)),
                "n_sessions": int(sub["session"].nunique()),
                "mean_edge_alignment_index_window": float(sub["drift_edge_cos2"].mean()),
                "mean_edge_alignment_index_session": float(np.mean(session_means)) if session_means.size else float("nan"),
                "ci95_low_session_mean": ci_low,
                "ci95_high_session_mean": ci_high,
                "median_abs_delta_deg": float(np.median(abs_delta)) if abs_delta.size else float("nan"),
                "fraction_within_15deg_parallel": float(np.mean(abs_delta <= 15.0)) if abs_delta.size else float("nan"),
                "fraction_within_30deg_parallel": float(np.mean(abs_delta <= 30.0)) if abs_delta.size else float("nan"),
                "fraction_within_15deg_orthogonal": float(np.mean(abs_delta >= 75.0)) if abs_delta.size else float("nan"),
                "fraction_within_30deg_orthogonal": float(np.mean(abs_delta >= 60.0)) if abs_delta.size else float("nan"),
                "resultant_R_abs_mean_exp2i": _resultant_abs(sub["signed_edge_delta_deg"]),
            }
        )
    return pd.DataFrame(rows)


def _endpoint_zone_summary(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zones = [
        ("parallel <=15 deg", lambda x: x <= 15.0, 15.0 / 90.0),
        ("orthogonal >=75 deg", lambda x: x >= 75.0, 15.0 / 90.0),
        ("mid 30-60 deg", lambda x: (x >= 30.0) & (x <= 60.0), 30.0 / 90.0),
    ]
    rows: list[dict[str, object]] = []
    for subset_name, sub in subsets.items():
        abs_delta = sub["abs_edge_delta_deg"].to_numpy(dtype=np.float64)
        total = int(abs_delta.size)
        for zone_name, mask_fn, expected in zones:
            count = int(np.count_nonzero(mask_fn(abs_delta)))
            frac = float(count / total) if total else float("nan")
            lo, hi = _wilson_interval(count, total)
            rows.append(
                {
                    "subset": subset_name,
                    "zone": zone_name,
                    "n_windows": total,
                    "count": count,
                    "fraction": frac,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "uniform_expected_fraction": expected,
                    "excess_fraction_points": frac - expected if np.isfinite(frac) else float("nan"),
                    "observed_expected_ratio": frac / expected if expected > 0.0 and np.isfinite(frac) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def _format_percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(lambda val, _pos: f"{100.0 * val:.0f}%")


def _plot_edge_alignment_window_session(subsets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    bins_cos = np.linspace(-1.0, 1.0, 41)
    bins_delta = np.linspace(0.0, 90.0, 31)

    ax = axes[0, 0]
    for name, sub in subsets.items():
        vals = sub["drift_edge_cos2"].to_numpy(dtype=np.float64)
        ax.hist(vals, bins=bins_cos, density=True, histtype="step", linewidth=2.0, color=COLORS[name], label=f"{name} (n={len(sub):,})")
    ax.axvline(0.0, color="0.35", linewidth=1.2)
    ax.axvline(subsets["All windows"]["drift_edge_cos2"].mean(), color=COLORS["All windows"], linestyle="--", linewidth=1.8)
    ax.set_title("Edge Alignment Index")
    ax.set_xlabel("cos(2 * drift-edge delta): +1 parallel, -1 orthogonal")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "A")

    ax = axes[0, 1]
    for name, sub in subsets.items():
        vals = sub["abs_edge_delta_deg"].to_numpy(dtype=np.float64)
        ax.hist(vals, bins=bins_delta, density=True, histtype="step", linewidth=2.0, color=COLORS[name], label=name)
    ax.axhline(1.0 / 90.0, color=COLORS["Null"], linestyle=":", linewidth=1.8, label="Uniform axial baseline")
    ax.set_xlim(0.0, 90.0)
    ax.set_title("Absolute Drift-Edge Axis Difference")
    ax.set_xlabel("|delta| in axial degrees (0 parallel, 90 orthogonal)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "B")

    ax = axes[1, 0]
    rng = np.random.default_rng(12)
    for idx, (name, sub) in enumerate(subsets.items(), start=1):
        means = sub.groupby("session", dropna=False)["drift_edge_cos2"].mean().to_numpy(dtype=np.float64)
        jitter = rng.normal(0.0, 0.045, size=means.size)
        ax.scatter(np.full(means.size, idx) + jitter, means, s=26, color=COLORS[name], alpha=0.75, edgecolor="none")
        lo, hi = _binomial_ci_for_mean(means)
        mean = float(np.nanmean(means))
        ax.errorbar([idx], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o", color="black", capsize=5, markersize=7)
    ax.axhline(0.0, color="0.35", linewidth=1.0)
    ax.set_xticks(
        [1, 2, 3],
        [f"{name}\n{subsets[name]['session'].nunique()} sessions" for name in subsets],
    )
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Session Mean Edge Alignment")
    ax.set_ylabel("Mean edge alignment index per session")
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "C")

    ax = axes[1, 1]
    thresholds = np.linspace(0.0, 90.0, 181)
    for name, sub in subsets.items():
        vals = sub["abs_edge_delta_deg"].to_numpy(dtype=np.float64)
        cum = np.asarray([np.mean(vals <= t) for t in thresholds], dtype=np.float64)
        ax.plot(thresholds, cum, color=COLORS[name], linewidth=2.0, label=name)
    ax.plot(thresholds, thresholds / 90.0, color=COLORS["Null"], linestyle=":", linewidth=1.8, label="Uniform axial baseline")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, 1.0)
    _format_percent_axis(ax)
    ax.set_title("Cumulative Parallel Preference")
    ax.set_xlabel("|delta| threshold in degrees")
    ax.set_ylabel("Fraction of windows")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "D")

    fig.savefig(out_dir / "edge_alignment_window_and_session_distributions.png", dpi=180)
    plt.close(fig)


def _binned_mean(df: pd.DataFrame, x_col: str, y_col: str, bins: np.ndarray) -> pd.DataFrame:
    work = df[[x_col, y_col]].copy()
    work["bin"] = pd.cut(work[x_col], bins=bins, include_lowest=True, right=False)
    rows: list[dict[str, float]] = []
    for interval, sub in work.groupby("bin", observed=False):
        if sub.empty:
            continue
        vals = sub[y_col].to_numpy(dtype=np.float64)
        mean = float(np.nanmean(vals))
        lo, hi = _binomial_ci_for_mean(vals)
        rows.append(
            {
                "x": float((interval.left + interval.right) / 2.0),
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "count": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def _plot_confidence_signed_delta(df: pd.DataFrame, subsets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    bins01 = np.linspace(0.0, 1.0, 11)

    for ax, x_col, title, panel in [
        (axes[0, 0], "image_orientation_coherence", "Alignment by Image Orientation Coherence", "A"),
        (axes[0, 1], "anisotropy", "Alignment by FEM Anisotropy", "B"),
    ]:
        summary = _binned_mean(df, x_col, "drift_edge_cos2", bins01)
        counts, edges = np.histogram(df[x_col].to_numpy(dtype=np.float64), bins=bins01)
        ax_count = ax.twinx()
        ax_count.bar((edges[:-1] + edges[1:]) / 2.0, counts, width=0.08, color="#cfd5da", alpha=0.45, zorder=0)
        ax_count.set_ylabel("Window count")
        ax.errorbar(
            summary["x"],
            summary["mean"],
            yerr=[summary["mean"] - summary["ci_low"], summary["ci_high"] - summary["mean"]],
            color=COLORS["All windows"],
            marker="o",
            linewidth=2.0,
            capsize=3,
            zorder=2,
        )
        ax.axhline(0.0, color="0.35", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(title)
        ax.set_xlabel("Image orientation coherence" if x_col == "image_orientation_coherence" else "FEM anisotropy")
        ax.set_ylabel("Mean edge alignment index")
        ax.grid(alpha=0.25)
        _annotate_panel(ax, panel)

    ax = axes[1, 0]
    work = df.copy()
    work["coh_bin"] = pd.cut(work["image_orientation_coherence"], bins=bins01, include_lowest=True, right=False)
    work["ani_bin"] = pd.cut(work["anisotropy"], bins=bins01, include_lowest=True, right=False)
    heat = work.pivot_table(
        values="drift_edge_cos2",
        index="ani_bin",
        columns="coh_bin",
        aggfunc="mean",
        observed=False,
    )
    arr = heat.to_numpy(dtype=np.float64)
    im = ax.imshow(arr, origin="lower", extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    ax.set_title("Mean Alignment in Confidence Grid")
    ax.set_xlabel("Image orientation coherence")
    ax.set_ylabel("FEM anisotropy")
    fig.colorbar(im, ax=ax, label="Mean edge alignment index")
    _annotate_panel(ax, "C")

    ax = axes[1, 1]
    reliable = subsets["Reliable axes"]
    bins_signed = np.linspace(-90.0, 90.0, 61)
    vals = reliable["signed_edge_delta_deg"].to_numpy(dtype=np.float64)
    ax.hist(vals, bins=bins_signed, density=True, color="#8aa5b8", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0 / 180.0, color=COLORS["Null"], linestyle=":", linewidth=1.8, label="Uniform signed axial baseline")
    ax.axvline(0.0, color="0.55", linewidth=1.0)
    ax.axvline(-45.0, color="0.8", linewidth=0.8)
    ax.axvline(45.0, color="0.8", linewidth=0.8)
    ax.set_xlim(-90.0, 90.0)
    ax.set_title("Signed Drift-Edge Delta, Reliable Axes")
    ax.set_xlabel("Signed axial delta in degrees")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "D")

    fig.savefig(out_dir / "edge_alignment_confidence_and_signed_delta.png", dpi=180)
    plt.close(fig)


def _objective_session_table(objective_path: Path) -> pd.DataFrame:
    df = pd.read_csv(objective_path)
    _require_columns(df, objective_path, ["session", "objective", "cos2_alignment", "predicted_axis_deg"])
    df = _finite(df, ["cos2_alignment", "predicted_axis_deg"])
    return df


def _write_objective_summary(objective_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for objective in OBJECTIVE_ORDER:
        sub = objective_df[objective_df["objective"] == objective]
        if sub.empty:
            continue
        sess = sub.groupby("session", dropna=False)["cos2_alignment"].mean()
        rows.append(
            {
                "objective": objective,
                "mean": float(sess.mean()),
                "median": float(sess.median()),
                "count": int(sess.size),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "objective_session_alignment_distribution_summary.csv", index=False)
    return out


def _plot_model_objective_alignment(objective_df: pd.DataFrame, out_dir: Path) -> None:
    session_means = (
        objective_df[objective_df["objective"].isin(OBJECTIVE_ORDER)]
        .groupby(["objective", "session"], dropna=False)["cos2_alignment"]
        .mean()
        .reset_index()
    )
    raw = session_means[session_means["objective"] == "raw_edge_axis"].set_index("session")["cos2_alignment"]
    labels = [OBJECTIVE_LABELS[obj] for obj in OBJECTIVE_ORDER if obj in set(session_means["objective"])]
    present = [obj for obj in OBJECTIVE_ORDER if obj in set(session_means["objective"])]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )
    ax = axes[0]
    data = [session_means[session_means["objective"] == obj]["cos2_alignment"].to_numpy(dtype=np.float64) for obj in present]
    colors = ["#d9e3ea" if obj == "raw_edge_axis" else "#eadff0" for obj in present]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor("0.3")
    rng = np.random.default_rng(24)
    for idx, (obj, vals) in enumerate(zip(present, data, strict=False), start=1):
        jitter = rng.normal(0.0, 0.035, size=len(vals))
        color = COLORS["All windows"] if obj == "raw_edge_axis" else COLORS["Objective"]
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=22, color=color, alpha=0.7, edgecolor="none")
        ax.scatter([idx], [np.mean(vals)], marker="D", s=50, color="black", zorder=5)
    ax.axhline(0.0, color="0.25", linewidth=1.0)
    if not raw.empty:
        ax.axhline(float(raw.mean()), color=COLORS["All windows"], linestyle="--", linewidth=1.8, label="Raw-edge session mean")
        ax.legend(frameon=False, loc="upper right")
    ax.set_title("Session-Level Axis Alignment by Objective")
    ax.set_ylabel("Mean cos(2 * predicted-real delta) per session")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "A")

    ax = axes[1]
    rows: list[dict[str, object]] = []
    for obj in present:
        if obj == "raw_edge_axis":
            continue
        obj_sess = session_means[session_means["objective"] == obj].set_index("session")["cos2_alignment"]
        aligned = pd.concat([obj_sess.rename("obj"), raw.rename("raw")], axis=1).dropna()
        if aligned.empty:
            continue
        delta = aligned["obj"] - aligned["raw"]
        lo, hi = _binomial_ci_for_mean(delta.to_numpy(dtype=np.float64))
        rows.append(
            {
                "objective": obj,
                "label": OBJECTIVE_LABELS[obj],
                "mean_delta": float(delta.mean()),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    delta_df = pd.DataFrame(rows)
    y = np.arange(delta_df.shape[0])
    colors_delta = np.where(delta_df["mean_delta"].to_numpy(dtype=np.float64) >= 0.0, "#4da89b", "#cf6558")
    ax.barh(y, delta_df["mean_delta"], color=colors_delta, alpha=0.95)
    ax.errorbar(
        delta_df["mean_delta"],
        y,
        xerr=[delta_df["mean_delta"] - delta_df["ci_low"], delta_df["ci_high"] - delta_df["mean_delta"]],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.3,
    )
    ax.axvline(0.0, color="0.25", linewidth=1.0)
    ax.set_yticks(y, delta_df["label"])
    ax.invert_yaxis()
    ax.set_title("Paired Session Delta vs Raw Edge")
    ax.set_xlabel("Objective minus raw-edge session alignment")
    ax.grid(axis="x", alpha=0.25)
    _annotate_panel(ax, "B")

    fig.savefig(out_dir / "model_objective_alignment_distributions.png", dpi=180)
    plt.close(fig)


def _plot_predicted_axis_distributions(objective_df: pd.DataFrame, out_dir: Path) -> None:
    present = [obj for obj in PREDICTED_AXIS_ORDER if obj in set(objective_df["objective"])]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(11, 1.35 * len(present) + 1.0), sharex=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    bins = np.linspace(0.0, 180.0, 37)
    for ax, obj in zip(axes_arr, present, strict=False):
        vals = objective_df.loc[objective_df["objective"] == obj, "predicted_axis_deg"].to_numpy(dtype=np.float64)
        vals = vals % 180.0
        color = COLORS["All windows"] if obj == "raw_edge_axis" else COLORS["Objective"]
        ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(OBJECTIVE_LABELS[obj], rotation=0, ha="right", va="center", labelpad=50)
        ax.grid(alpha=0.2)
        if vals.size and np.nanmax(vals) == np.nanmin(vals):
            ax.text(0.99, 0.1, f"all {vals[0]:.1f} deg", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox={"facecolor": "white", "edgecolor": "0.8"})
    axes_arr[0].set_title("Predicted Axis Distributions by Objective")
    axes_arr[-1].set_xlabel("Predicted axial axis in degrees")
    axes_arr[-1].set_xlim(0.0, 180.0)
    fig.savefig(out_dir / "predicted_axis_distributions_by_objective.png", dpi=180)
    plt.close(fig)


def _uniform_cos2_bin_mass(edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(edges, -1.0, 1.0)
    theta = np.arccos(clipped)
    mass = (theta[:-1] - theta[1:]) / np.pi
    return np.maximum(mass, 0.0)


def _plot_endpoint_null_diagnostic(subsets: dict[str, pd.DataFrame], endpoint_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    cos_bins = np.linspace(-1.0, 1.0, 41)
    cos_centers = (cos_bins[:-1] + cos_bins[1:]) / 2.0
    expected_cos = _uniform_cos2_bin_mass(cos_bins)
    delta_bins = np.linspace(0.0, 90.0, 31)
    delta_centers = (delta_bins[:-1] + delta_bins[1:]) / 2.0
    expected_delta = np.diff(delta_bins) / 90.0

    ax = axes[0, 0]
    for name, sub in subsets.items():
        vals = sub["drift_edge_cos2"].to_numpy(dtype=np.float64)
        hist, _ = np.histogram(vals, bins=cos_bins)
        prob = hist / hist.sum()
        ax.plot(cos_centers, prob, color=COLORS[name], linewidth=2.0, label=name)
    ax.plot(cos_centers, expected_cos, color=COLORS["Null"], linestyle=":", linewidth=2.0, label="Uniform angular null")
    ax.set_title("Cos2 Histogram: Raw Bin Probability")
    ax.set_xlabel("Edge alignment index, cos(2 delta)")
    ax.set_ylabel("Probability per bin")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "A")

    ax = axes[0, 1]
    for name, sub in subsets.items():
        vals = sub["drift_edge_cos2"].to_numpy(dtype=np.float64)
        hist, _ = np.histogram(vals, bins=cos_bins)
        prob = hist / hist.sum()
        ratio = np.divide(prob, expected_cos, out=np.full_like(prob, np.nan, dtype=np.float64), where=expected_cos > 0.0)
        ax.plot(cos_centers, ratio, color=COLORS[name], linewidth=2.0, label=name)
    ax.axhline(1.0, color="0.35", linewidth=1.0)
    ax.set_title("Cos2 Histogram: Observed / Uniform-Angle Expected")
    ax.set_xlabel("Edge alignment index, cos(2 delta)")
    ax.set_ylabel("Observed / expected bin mass")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "B")

    ax = axes[1, 0]
    for name, sub in subsets.items():
        vals = sub["abs_edge_delta_deg"].to_numpy(dtype=np.float64)
        hist, _ = np.histogram(vals, bins=delta_bins)
        prob = hist / hist.sum()
        ratio = np.divide(prob, expected_delta, out=np.full_like(prob, np.nan, dtype=np.float64), where=expected_delta > 0.0)
        ax.plot(delta_centers, ratio, color=COLORS[name], linewidth=2.0, label=name)
    ax.axhline(1.0, color="0.35", linewidth=1.0)
    ax.axvspan(0.0, 15.0, color=COLORS["Reliable axes"], alpha=0.08)
    ax.axvspan(75.0, 90.0, color=COLORS["High confidence"], alpha=0.08)
    ax.set_xlim(0.0, 90.0)
    ax.set_title("Absolute Delta: Observed / Uniform-Angle Expected")
    ax.set_xlabel("|drift-edge delta| degrees")
    ax.set_ylabel("Observed / expected bin mass")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _annotate_panel(ax, "C")

    ax = axes[1, 1]
    zones = ["parallel <=15 deg", "orthogonal >=75 deg", "mid 30-60 deg"]
    zone_labels = ["Parallel\n<=15 deg", "Orthogonal\n>=75 deg", "Mid\n30-60 deg"]
    x = np.arange(len(zones), dtype=np.float64)
    width = 0.22
    for offset, subset_name in zip([-width, 0.0, width], subsets.keys(), strict=False):
        block = endpoint_summary[endpoint_summary["subset"] == subset_name].set_index("zone").loc[zones]
        frac = block["fraction"].to_numpy(dtype=np.float64)
        lo = block["ci95_low"].to_numpy(dtype=np.float64)
        hi = block["ci95_high"].to_numpy(dtype=np.float64)
        ax.bar(x + offset, frac, width=width, color=COLORS[subset_name], alpha=0.8, label=subset_name)
        ax.errorbar(x + offset, frac, yerr=[frac - lo, hi - frac], fmt="none", ecolor="black", capsize=2, linewidth=1.0)
    expected = [15.0 / 90.0, 15.0 / 90.0, 30.0 / 90.0]
    for xi, exp in zip(x, expected, strict=False):
        ax.hlines(exp, xi - 0.42, xi + 0.42, color="black", linestyle=":", linewidth=1.8)
    ax.set_xticks(x, zone_labels)
    ax.set_ylim(0.0, max(0.4, float(endpoint_summary["ci95_high"].max()) * 1.15))
    _format_percent_axis(ax)
    ax.set_title("Endpoint and Mid-Angle Zone Fractions")
    ax.set_ylabel("Fraction of windows")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _annotate_panel(ax, "D")

    fig.savefig(out_dir / "edge_alignment_endpoint_null_diagnostic.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    image_windows_path = Path(args.image_structure_dir) / "backimage_image_fem_windows.csv"
    edge_df = _load_edge_windows(image_windows_path)
    subsets = _make_subsets(
        edge_df,
        reliable_coherence=float(args.reliable_coherence),
        reliable_anisotropy=float(args.reliable_anisotropy),
        high_confidence_coherence=float(args.high_confidence_coherence),
        high_confidence_anisotropy=float(args.high_confidence_anisotropy),
    )

    edge_summary = _edge_alignment_summary(subsets, rng=rng, n_boot=int(args.n_bootstrap))
    edge_summary.to_csv(out_dir / "edge_alignment_distribution_summary.csv", index=False)
    endpoint_summary = _endpoint_zone_summary(subsets)
    endpoint_summary.to_csv(out_dir / "endpoint_zone_enrichment_summary.csv", index=False)

    _plot_edge_alignment_window_session(subsets, out_dir)
    _plot_confidence_signed_delta(edge_df, subsets, out_dir)
    _plot_endpoint_null_diagnostic(subsets, endpoint_summary, out_dir)

    objective_path = Path(args.drift_geometry_dir) / "real_vs_predicted_axis_alignment.csv"
    if objective_path.exists():
        objective_df = _objective_session_table(objective_path)
        _write_objective_summary(objective_df, out_dir)
        _plot_model_objective_alignment(objective_df, out_dir)
        _plot_predicted_axis_distributions(objective_df, out_dir)
    elif not bool(args.allow_missing_objective_tables):
        raise FileNotFoundError(f"Missing objective table: {objective_path}")

    metadata = {
        "image_windows_path": str(image_windows_path),
        "drift_geometry_dir": str(args.drift_geometry_dir),
        "out_dir": str(out_dir),
        "reliable_coherence": float(args.reliable_coherence),
        "reliable_anisotropy": float(args.reliable_anisotropy),
        "high_confidence_coherence": float(args.high_confidence_coherence),
        "high_confidence_anisotropy": float(args.high_confidence_anisotropy),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
    }
    (out_dir / "edge_alignment_distribution_inspection_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-structure-dir", type=Path, default=DEFAULT_IMAGE_STRUCTURE_DIR)
    parser.add_argument("--drift-geometry-dir", type=Path, default=DEFAULT_DRIFT_GEOMETRY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reliable-coherence", type=float, default=0.2)
    parser.add_argument("--reliable-anisotropy", type=float, default=0.2)
    parser.add_argument("--high-confidence-coherence", type=float, default=0.5)
    parser.add_argument("--high-confidence-anisotropy", type=float, default=0.5)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-missing-objective-tables", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
