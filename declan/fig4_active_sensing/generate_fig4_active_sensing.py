#!/usr/bin/env python3
"""Generate the Figure 4 active-sensing headline figure.

This figure supersedes the initial wrapper around the older natural-image
movie-information figure.  It is cache-first: the generator reads the cleaned
BackImage aggregate FEM-information run and local image-geometry support tables
without rerunning the V1 twin.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
BACKIMAGE_BASE = ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_AGGREGATE_DIR = (
    BACKIMAGE_BASE
    / "backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched"
)
DEFAULT_INCREMENTAL_DIR = DEFAULT_AGGREGATE_DIR / "incremental_static_plus_motion_relids"
DEFAULT_GEOMETRY_DIR = BACKIMAGE_BASE / "backimage_image_structure_reviewed_v2_screenfiltered"
DEFAULT_STABILITY_DIR = BACKIMAGE_BASE / "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
DEFAULT_OUT = ROOT / "outputs" / "fig4_active_sensing" / "active_sensing_headline_figure"
DEFAULT_THUMBNAIL = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "active_sensing_movie_information_figure_frozen_20260615_pre_backimage_collab_pack"
    / "active_sensing_movie_information_figure.png"
)

FIGURE_STEM = "fig4_active_sensing_headline"
CAPTION_TITLE = (
    "Figure 4. Self-generated drift adds feature-relevant temporal structure "
    "and follows image-stable axes."
)
SCALE_AXIS_LABEL = "scaled eye trajectories (observed-RMS multiplier)"
MOTION_ORDER = ["empirical", "ou", "brownian", "rotated"]

COLORS = {
    "empirical": "#244f7a",
    "ou": "#d07a22",
    "brownian": "#707070",
    "rotated": "#8064a2",
    "gabor": "#244f7a",
    "pyramid": "#2f8f6a",
    "edge": "#2f8f6a",
    "pixel": "#4f7fb7",
    "twin": "#7a5ea8",
    "light": "#eef2f4",
    "dark": "#2a2a2a",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.08,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title(title, loc="left", pad=8, fontweight="bold")


def _scale_value(scale_id: str) -> float:
    return float(scale_id.replace("rel_", "").replace("p", ".").replace("x", ""))


def _scale_label(scale: float) -> str:
    return f"{scale:g}x"


def _errbar(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    y_col: str,
    lo_col: str,
    hi_col: str,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    block = df.sort_values("scale")
    x = block["scale"].to_numpy(dtype=float)
    y = block[y_col].to_numpy(dtype=float)
    lo = block[lo_col].to_numpy(dtype=float)
    hi = block[hi_col].to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        marker=marker,
        markersize=4,
        linewidth=1.9,
        capsize=3,
        color=color,
        label=label,
    )


def _motion_qc(aggregate_dir: Path) -> dict[str, float | int]:
    motion = _read_csv(aggregate_dir / "aggregate_motion_summary.csv")
    metadata = _read_csv(aggregate_dir / "aggregate_motion_metadata.csv")
    nonstatic = metadata[metadata["family"].astype(str) != "static"].copy()
    return {
        "n_trace_sources": int(nonstatic["trace_source_row"].dropna().nunique()),
        "median_effective_to_requested_rms": float(
            motion["median_effective_to_requested_rms"].astype(float).median()
        ),
        "max_clipped_fraction": float(motion["clipped_fraction"].astype(float).max()),
    }


def _edge_axis_session_bootstrap_ci(
    windows: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 11,
) -> dict[str, dict[str, float]]:
    work = windows.dropna(
        subset=[
            "session",
            "drift_orientation_deg",
            "image_edge_axis_deg",
            "image_orientation_coherence",
            "anisotropy",
        ]
    ).copy()
    out: dict[str, dict[str, float]] = {}
    subsets = {
        "all_windows": work,
        "reliable_axes_coh_ge_0p20_aniso_ge_0p20": work[
            (work["image_orientation_coherence"].astype(float) >= 0.20)
            & (work["anisotropy"].astype(float) >= 0.20)
        ].copy(),
    }
    rng = np.random.default_rng(seed)
    for subset, sub in subsets.items():
        if sub.empty:
            continue
        session_sums: list[tuple[float, float]] = []
        for _, sess_df in sub.groupby("session", sort=True):
            drift = sess_df["drift_orientation_deg"].to_numpy(dtype=np.float64)
            edge = sess_df["image_edge_axis_deg"].to_numpy(dtype=np.float64)
            weights = (
                sess_df["image_orientation_coherence"].astype(float).to_numpy(dtype=np.float64)
                * sess_df["anisotropy"].astype(float).to_numpy(dtype=np.float64)
            )
            weights[~np.isfinite(weights) | (weights < 0)] = 0.0
            values = np.cos(2.0 * np.radians(drift - edge))
            session_sums.append((float(np.sum(weights * values)), float(np.sum(weights))))
        sums = np.asarray(session_sums, dtype=np.float64)
        valid = sums[:, 1] > 0
        sums = sums[valid]
        if sums.size == 0:
            continue
        choices = rng.integers(0, sums.shape[0], size=(n_bootstrap, sums.shape[0]))
        boot_num = sums[choices, 0].sum(axis=1)
        boot_den = sums[choices, 1].sum(axis=1)
        boot = boot_num / boot_den
        out[subset] = {
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "n_bootstrap": int(n_bootstrap),
            "n_sessions": int(sums.shape[0]),
        }
    return out


def _synthetic_patch(size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(41)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    shore = 0.28 * np.tanh(9.0 * (x * 0.25 + y * 0.92 + 0.08))
    ridge = 0.20 * np.exp(-((x + 0.48) ** 2 / 0.10 + (y + 0.18) ** 2 / 0.32))
    texture = 0.12 * np.sin(18 * (x * 0.78 - y * 0.35))
    texture += 0.08 * np.sin(34 * (x * 0.18 + y * 0.98))
    patch = 0.52 + shore + ridge + texture + 0.05 * rng.normal(size=(size, size))
    return np.clip(patch, 0.0, 1.0)


def _thumbnail_patch(path: Path = DEFAULT_THUMBNAIL) -> np.ndarray:
    if not path.exists():
        return _synthetic_patch()
    try:
        image = plt.imread(path)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        image = np.asarray(image, dtype=np.float32)
        if image.max(initial=0.0) > 1.5:
            image = image / 255.0
        # Crop the clean tree/coast portion of the frozen Figure 4 thumbnail.
        # This keeps the visual exemplar stable without reloading raw session assets.
        h, w = image.shape[:2]
        crop = image[
            int(round(0.135 * h)) : int(round(0.255 * h)),
            int(round(0.050 * w)) : int(round(0.132 * w)),
        ]
        lo, hi = np.nanpercentile(crop, [1, 99])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            crop = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
        return crop
    except Exception:
        return _synthetic_patch()


def _draw_node(ax: plt.Axes, xy: tuple[float, float], text: str, *, width: float = 0.19) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - 0.055),
        width,
        0.11,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8,
        edgecolor="#c5ccd2",
        facecolor="#f8fafb",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=8)


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.0,
        color="#616a70",
        transform=ax.transAxes,
    )
    ax.add_patch(arr)


def _motion_trace(kind: str, n: int = 50) -> np.ndarray:
    rng = np.random.default_rng({"empirical": 3, "ou": 5, "brownian": 7, "rotated": 11}[kind])
    if kind == "ou":
        steps = rng.normal(scale=0.045, size=(n, 2))
        trace = np.zeros((n, 2))
        for i in range(1, n):
            trace[i] = 0.86 * trace[i - 1] + steps[i]
    elif kind == "brownian":
        trace = np.cumsum(rng.normal(scale=0.028, size=(n, 2)), axis=0)
    else:
        t = np.linspace(0, 1, n)
        trace = np.c_[
            0.18 * np.sin(2.2 * np.pi * t) + 0.035 * rng.normal(size=n),
            0.07 * np.sin(5.2 * np.pi * t + 0.4) + 0.025 * rng.normal(size=n),
        ]
        if kind == "rotated":
            rot = np.array([[0.0, -1.0], [1.0, 0.0]])
            trace = trace @ rot.T
    trace -= trace.mean(axis=0, keepdims=True)
    denom = np.max(np.abs(trace)) or 1.0
    return trace / denom


def _plot_concept(ax: plt.Axes) -> None:
    _panel_label(ax, "A", "Does FEM-induced reafference carry image information?")
    ax.set_axis_off()
    patch_ax = ax.inset_axes([0.02, 0.18, 0.23, 0.58])
    patch_ax.imshow(_thumbnail_patch(), cmap="gray", vmin=0, vmax=1)
    patch_ax.set_xticks([])
    patch_ax.set_yticks([])
    for spine in patch_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#c5ccd2")
    ax.text(0.135, 0.09, "example natural-image patch", transform=ax.transAxes, ha="center", fontsize=7.8)

    trace_ax = ax.inset_axes([0.31, 0.17, 0.25, 0.60])
    trace_ax.set_facecolor("#fbfcfd")
    for kind, color, offset in [
        ("empirical", COLORS["empirical"], 0.27),
        ("ou", COLORS["ou"], 0.09),
        ("brownian", COLORS["brownian"], -0.09),
        ("rotated", COLORS["rotated"], -0.27),
    ]:
        trace = _motion_trace(kind)
        trace_ax.plot(0.16 * trace[:, 0], 0.16 * trace[:, 1] + offset, color=color, lw=1.2)
        label = "OU" if kind == "ou" else kind
        trace_ax.text(0.24, offset, label, color=color, va="center", fontsize=7.2)
    trace_ax.set_xlim(-0.24, 0.55)
    trace_ax.set_ylim(-0.43, 0.43)
    trace_ax.set_xticks([])
    trace_ax.set_yticks([])
    for spine in trace_ax.spines.values():
        spine.set_visible(False)
    ax.text(0.435, 0.09, "motion distributions", transform=ax.transAxes, ha="center", fontsize=7.8)

    _draw_node(ax, (0.69, 0.64), "V1-twin\nresponse movies", width=0.22)
    _draw_node(ax, (0.69, 0.36), "temporal-PC\nresponse summaries", width=0.24)
    _draw_node(ax, (0.91, 0.50), "image-feature\ndecoding", width=0.20)
    _draw_arrow(ax, (0.26, 0.50), (0.31, 0.50))
    _draw_arrow(ax, (0.56, 0.50), (0.61, 0.58))
    _draw_arrow(ax, (0.69, 0.58), (0.69, 0.43))
    _draw_arrow(ax, (0.80, 0.50), (0.82, 0.50))
    ax.text(
        0.83,
        0.18,
        "decode $z$ from\n$R_{static}+R_{motion}$\nvs $R_{static}$ only",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#3f484f",
    )
    ax.text(
        0.66,
        0.06,
        "from FEM-linked shared variability to feature-relevant temporal samples",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#3f484f",
    )


def _plot_gain_vs_static(ax: plt.Axes, gain: pd.DataFrame, qc: dict[str, float | int]) -> None:
    _panel_label(ax, "B", "Empirical drift adds information beyond static")
    block = gain[
        (gain["motion_summary"] == "temporal_pca")
        & (gain["family"] == "empirical")
        & (
            ((gain["latent"] == "gabor_local_field") & (gain["k"] == 4))
            | ((gain["latent"] == "pyramid_local_field") & (gain["k"] == 8))
        )
    ].copy()
    if block.empty:
        raise ValueError("No empirical temporal_pca gain rows found for panel B")
    block["scale"] = block["scale_id"].map(_scale_value)
    labels = {
        ("gabor_local_field", 4): ("Gabor local field, k=4", COLORS["gabor"], "o"),
        ("pyramid_local_field", 8): ("Pyramid local field, k=8", COLORS["pyramid"], "s"),
    }
    for (latent, k), (label, color, marker) in labels.items():
        sub = block[(block["latent"] == latent) & (block["k"] == k)]
        _errbar(
            ax,
            sub,
            y_col="incremental_gain_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=label,
            color=color,
            marker=marker,
        )
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels([_scale_label(x) for x in [0.25, 0.5, 1.0, 1.5, 2.0]])
    ax.set_xlabel(SCALE_AXIS_LABEL)
    ax.set_ylabel("incremental decoding gain over static (-MSE)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.03,
        0.06,
        "256 images; K=4; 756-unit twin\n"
        f"grouped-by-image CV; {qc['n_trace_sources']} drift-only sources\n"
        f"RMS ratio={qc['median_effective_to_requested_rms']:.1f}; clipping={qc['max_clipped_fraction']:.1f}",
        transform=ax.transAxes,
        fontsize=7.1,
        color="#50585f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
    )
    _clean_axis(ax)


def _plot_control_contrasts(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    _panel_label(ax, "D", "Empirical drift beats OU-like motion")
    block = contrasts[
        (contrasts["motion_summary"] == "temporal_pca")
        & (contrasts["lhs_family"] == "empirical")
        & (contrasts["latent"] == "gabor_local_field")
        & (contrasts["k"] == 4)
        & (contrasts["rhs_family"].isin(["ou", "brownian", "rotated"]))
    ].copy()
    if block.empty:
        raise ValueError("No empirical control-contrast rows found for panel C")
    block["scale"] = block["scale_id"].map(_scale_value)
    for rhs, label, color, marker in [
        ("ou", "empirical - OU", COLORS["ou"], "o"),
        ("brownian", "empirical - Brownian", COLORS["brownian"], "s"),
        ("rotated", "empirical - rotated", COLORS["rotated"], "^"),
    ]:
        sub = block[block["rhs_family"] == rhs]
        _errbar(
            ax,
            sub,
            y_col="incremental_gain_delta_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=label,
            color=color,
            marker=marker,
        )
    ax.axhline(0, color="#222222", lw=0.8)
    ax.axvspan(0.2, 0.55, color="#e8f3ec", alpha=0.75, zorder=-5)
    ax.text(0.27, 28.0, "clearest\nmulti-control\nregime", fontsize=7.4, color="#2f6f52")
    ax.text(
        0.97,
        0.08,
        "Gabor k=4, temporal-PC readout\nBrownian/rotated narrow at large scales",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#50585f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
    )
    ax.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels([_scale_label(x) for x in [0.25, 0.5, 1.0, 1.5, 2.0]])
    ax.set_xlabel(SCALE_AXIS_LABEL)
    ax.set_ylabel("contrast in incremental gain (-MSE)")
    ax.legend(frameon=False, loc="upper right")
    _clean_axis(ax)


def _plot_absolute_temporal_pca_gains(ax: plt.Axes, gain: pd.DataFrame) -> None:
    _panel_label(ax, "C", "Temporal-PC gains by trajectory family")
    block = gain[
        (gain["motion_summary"] == "temporal_pca")
        & (gain["latent"] == "gabor_local_field")
        & (gain["k"] == 4)
        & (gain["family"].isin(MOTION_ORDER))
    ].copy()
    if block.empty:
        raise ValueError("No Gabor k=4 temporal_pca absolute-gain rows found for panel C")
    block["scale"] = block["scale_id"].map(_scale_value)
    for family, label, color, marker in [
        ("empirical", "empirical drift", COLORS["empirical"], "o"),
        ("ou", "OU-like", COLORS["ou"], "o"),
        ("brownian", "Brownian", COLORS["brownian"], "s"),
        ("rotated", "rotated drift", COLORS["rotated"], "^"),
    ]:
        sub = block[block["family"] == family]
        _errbar(
            ax,
            sub,
            y_col="incremental_gain_neg_mse",
            lo_col="ci95_low",
            hi_col="ci95_high",
            label=label,
            color=color,
            marker=marker,
        )
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels([_scale_label(x) for x in [0.25, 0.5, 1.0, 1.5, 2.0]])
    ax.set_xlabel(SCALE_AXIS_LABEL)
    ax.set_ylabel("incremental decoding gain over static (-MSE)")
    ax.text(
        0.03,
        0.08,
        "Gabor k=4, temporal-PC readout",
        transform=ax.transAxes,
        fontsize=7.4,
        color="#50585f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
    )
    ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.8)
    _clean_axis(ax)


def _plot_local_geometry(
    ax: plt.Axes,
    geometry: pd.DataFrame,
    stability: pd.DataFrame,
    edge_ci: dict[str, dict[str, float]],
) -> None:
    _panel_label(ax, "E", "Predicted image-stable directions match real drift behavior")
    ax.set_axis_off()

    edge = geometry[
        (geometry["alignment_reference"] == "edge_axis")
        & (
            geometry["analysis_subset"].isin(
                ["all_windows", "reliable_axes_coh_ge_0p20_aniso_ge_0p20"]
            )
        )
    ].copy()
    if edge.empty:
        raise ValueError("No edge-axis alignment rows found for panel D")
    edge["label"] = edge["analysis_subset"].map(
        {
            "all_windows": "all windows",
            "reliable_axes_coh_ge_0p20_aniso_ge_0p20": "reliable axes",
        }
    )

    ax1 = ax.inset_axes([0.06, 0.58, 0.86, 0.32])
    st = stability[stability["screen"].isin(["pixel", "twin"])].copy()
    if st.empty:
        raise ValueError("No pixel/twin stability rows found for panel D")
    st["label"] = st["screen"].map({"pixel": "pixels", "twin": "V1 twin"})
    y2 = np.arange(len(st))
    colors = [COLORS["pixel"] if s == "pixel" else COLORS["twin"] for s in st["screen"]]
    ax1.barh(y2, st["fraction_windows_positive_advantage"], color=colors, alpha=0.90)
    ax1.axvline(0.5, color="#222222", lw=0.8, ls="--")
    ax1.set_title("Prediction: edge-parallel motion preserves local structure", loc="left", fontsize=8.0, pad=2)
    ax1.set_yticks(y2)
    ax1.set_yticklabels(st["label"])
    ax1.invert_yaxis()
    ax1.set_xlabel("fraction of windows with edge-parallel advantage")
    ax1.set_xlim(0, 1.18)
    ax1.text(
        0.03,
        0.96,
        "bars = window fraction",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#50585f",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.2},
    )
    for yi, (_, row) in zip(y2, st.iterrows(), strict=False):
        ax1.text(
            float(row["fraction_windows_positive_advantage"]) + 0.025,
            yi,
            f"sessions positive: {int(row['n_sessions_positive_advantage'])}/{int(row['n_sessions'])}",
            va="center",
            fontsize=7.2,
        )
    _clean_axis(ax1)

    ax2 = ax.inset_axes([0.06, 0.10, 0.86, 0.30])
    y = np.arange(len(edge))
    ax2.barh(y, edge["weighted_mean_cos2_delta"], color=COLORS["edge"], alpha=0.88)
    ax2.axvline(0, color="#222222", lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(edge["label"])
    ax2.invert_yaxis()
    ax2.set_title("Behavior: real drift axes align with predicted stable axes", loc="left", fontsize=8.0, pad=2)
    max_ci = max(
        [edge_ci.get(subset, {}).get("ci95_high", np.nan) for subset in edge["analysis_subset"]]
        + [float(edge["weighted_mean_cos2_delta"].max())]
    )
    ax2.set_xlim(0, max(0.27, float(max_ci) * 1.35))
    ax2.set_xlabel("edge-axis alignment (weighted cos2 delta)")
    for yi, (_, row) in zip(y, edge.iterrows(), strict=False):
        p = row.get("shuffle_p_abs_cos_ge_observed", np.nan)
        ci = edge_ci.get(str(row["analysis_subset"]))
        value = float(row["weighted_mean_cos2_delta"])
        if ci:
            ax2.errorbar(
                value,
                yi,
                xerr=np.asarray([[value - ci["ci95_low"]], [ci["ci95_high"] - value]]),
                color="#246b50",
                capsize=3,
                lw=1.2,
                fmt="none",
                zorder=5,
            )
        ax2.text(
            (ci["ci95_high"] if ci else value) + 0.006,
            yi,
            f"{value:.3f}, p={p:.3g}",
            va="center",
            fontsize=7.2,
        )
    _clean_axis(ax2)


def _collect_key_stats(
    gain: pd.DataFrame,
    contrasts: pd.DataFrame,
    aggregate_dir: Path,
    geometry: pd.DataFrame,
    stability: pd.DataFrame,
    edge_ci: dict[str, dict[str, float]],
    qc: dict[str, float | int],
) -> dict[str, Any]:
    gain_block = gain[
        (gain["motion_summary"] == "temporal_pca")
        & (gain["family"] == "empirical")
        & (gain["latent"] == "gabor_local_field")
        & (gain["k"] == 4)
    ].copy()
    gain_block["scale"] = gain_block["scale_id"].map(_scale_value)
    control_block = contrasts[
        (contrasts["motion_summary"] == "temporal_pca")
        & (contrasts["lhs_family"] == "empirical")
        & (contrasts["latent"] == "gabor_local_field")
        & (contrasts["k"] == 4)
    ].copy()
    control_block["scale"] = control_block["scale_id"].map(_scale_value)
    metadata = _read_json(aggregate_dir / "run_metadata.json")
    config = metadata.get("config", {})
    trace = _read_csv(aggregate_dir / "trace_bank_metadata.csv")

    edge = geometry[
        (geometry["alignment_reference"] == "edge_axis")
        & (geometry["analysis_subset"] == "all_windows")
    ].iloc[0]
    reliable_edge = geometry[
        (geometry["alignment_reference"] == "edge_axis")
        & (geometry["analysis_subset"] == "reliable_axes_coh_ge_0p20_aniso_ge_0p20")
    ].iloc[0]

    return {
        "aggregate_run": str(aggregate_dir),
        "n_images": int(gain_block["n_images"].max()),
        "n_sessions": int(gain_block["n_sessions"].max()),
        "trace_samples_per_condition": int(config.get("trace_samples_per_condition", 4)),
        "population": "canonical 756-unit V1 twin",
        "cv": "grouped by image, 5 outer folds",
        "motion_bookkeeping": {
            **qc,
            "n_trace_bank_rows": int(len(trace)),
        },
        "gabor_empirical_gain": gain_block.sort_values("scale")[
            ["scale_id", "incremental_gain_neg_mse", "ci95_low", "ci95_high"]
        ].to_dict(orient="records"),
        "gabor_temporal_pca_absolute_gains": gain[
            (gain["motion_summary"] == "temporal_pca")
            & (gain["latent"] == "gabor_local_field")
            & (gain["k"] == 4)
            & (gain["family"].isin(MOTION_ORDER))
        ]
        .assign(scale=lambda df: df["scale_id"].map(_scale_value))
        .sort_values(["family", "scale"])[
            ["family", "scale_id", "incremental_gain_neg_mse", "ci95_low", "ci95_high"]
        ]
        .to_dict(orient="records"),
        "gabor_empirical_control_contrasts": control_block.sort_values(["rhs_family", "scale"])[
            [
                "rhs_family",
                "scale_id",
                "incremental_gain_delta_neg_mse",
                "ci95_low",
                "ci95_high",
            ]
        ].to_dict(orient="records"),
        "edge_axis_alignment": {
            "all_windows_weighted_mean_cos2_delta": float(edge["weighted_mean_cos2_delta"]),
            "all_windows_shuffle_p_abs_cos_ge_observed": float(
                edge["shuffle_p_abs_cos_ge_observed"]
            ),
            "reliable_weighted_mean_cos2_delta": float(
                reliable_edge["weighted_mean_cos2_delta"]
            ),
            "reliable_shuffle_p_abs_cos_ge_observed": float(
                reliable_edge["shuffle_p_abs_cos_ge_observed"]
            ),
            "session_bootstrap_ci": edge_ci,
        },
        "edge_parallel_stability": stability.to_dict(orient="records"),
        "thumbnail_source": str(DEFAULT_THUMBNAIL),
    }


def _plain_caption_text(markdown_text: str) -> str:
    return " ".join(markdown_text.replace("**", "").replace("# Figure 4 Caption", "").split())


def _write_caption(stats: dict[str, Any], out_dir: Path) -> Path:
    caption_path = out_dir / f"{FIGURE_STEM}_caption.md"
    best_gain = stats["gabor_empirical_gain"][0]
    controls = stats["gabor_empirical_control_contrasts"]
    absolute = stats["gabor_temporal_pca_absolute_gains"]
    empirical_small = next(
        r for r in absolute if r["family"] == "empirical" and r["scale_id"] == "rel_0p25x"
    )
    ou_small_abs = next(r for r in absolute if r["family"] == "ou" and r["scale_id"] == "rel_0p25x")
    brownian_large_abs = next(
        r for r in absolute if r["family"] == "brownian" and r["scale_id"] == "rel_2x"
    )
    small_ou = next(r for r in controls if r["rhs_family"] == "ou" and r["scale_id"] == "rel_0p25x")
    small_brownian = next(
        r for r in controls if r["rhs_family"] == "brownian" and r["scale_id"] == "rel_0p25x"
    )
    edge = stats["edge_axis_alignment"]
    qc = stats["motion_bookkeeping"]
    edge_ci = edge["session_bootstrap_ci"]
    all_ci = edge_ci.get("all_windows", {})
    reliable_ci = edge_ci.get("reliable_axes_coh_ge_0p20_aniso_ge_0p20", {})
    lines = [
        "# Figure 4 Caption",
        "",
        f"**{CAPTION_TITLE}**",
        "(A) Functional bridge from reafferent variability to active sensing. After establishing that FEM-linked reafference contributes to shared V1 variability, this assay asks whether that self-generated retinal motion carries natural-image feature information. Natural-image BackImage patches were paired with empirical, OU-like, Brownian, and rotated drift-like motion distributions, passed through the canonical 756-unit V1 twin, summarized with temporal PCs, and evaluated by incremental image-feature decoding beyond the static response.",
        "(B) Empirical drift adds feature information beyond the static response. In the primary Gabor k=4 temporal-PC readout, the static-plus-motion gain over the static-only decoder at 0.25x observed-RMS scale was "
        f"{best_gain['incremental_gain_neg_mse']:.2f} in -MSE units "
        f"(95% CI {best_gain['ci95_low']:.2f} to {best_gain['ci95_high']:.2f}); "
        "the pyramid k=8 readout showed the same positive sign at every tested scale. "
        f"The aggregate run used grouped-by-image CV, {qc['n_trace_sources']} sampled drift-only trace sources, "
        f"median effective/requested RMS {qc['median_effective_to_requested_rms']:.1f}, "
        f"and maximum clipping fraction {qc['max_clipped_fraction']:.1f}.",
        "(C) Absolute temporal-PC gains for the primary Gabor k=4 readout show the scale-dependent family structure. Empirical drift was positive at small scale "
        f"({empirical_small['incremental_gain_neg_mse']:.2f}; "
        f"95% CI {empirical_small['ci95_low']:.2f} to {empirical_small['ci95_high']:.2f}), "
        f"whereas OU-like motion was negative at the same scale ({ou_small_abs['incremental_gain_neg_mse']:.2f}). "
        f"Brownian motion caught up at the largest scale ({brownian_large_abs['incremental_gain_neg_mse']:.2f}), "
        "calibrating the control caveat.",
        "(D) Empirical drift is not merely matched random confinement. Empirical drift outperformed OU-like confined motion across scale; at 0.25x the empirical-minus-OU contrast was "
        f"{small_ou['incremental_gain_delta_neg_mse']:.2f} "
        f"(95% CI {small_ou['ci95_low']:.2f} to {small_ou['ci95_high']:.2f}). "
        "The advantage over Brownian/generic and rotated motion was strongest at small scales "
        f"(0.25x empirical-minus-Brownian {small_brownian['incremental_gain_delta_neg_mse']:.2f}) "
        "and narrowed at larger scales.",
        "(E) Behavioral payoff: predicted image-stable directions match real drift behavior. The model-side stability screen first asks which local axes preserve the image and V1-twin response: edge-parallel motion was more stable than edge-orthogonal motion in both pixel and V1-twin metrics. The behavioral test then asks whether measured drift follows those axes. Real drift axes were aligned with local edge geometry "
        f"(all-window weighted cos2 delta {edge['all_windows_weighted_mean_cos2_delta']:.3f}; "
        f"session-bootstrap 95% CI {all_ci.get('ci95_low', float('nan')):.3f} to {all_ci.get('ci95_high', float('nan')):.3f}; "
        f"reliable-axis delta {edge['reliable_weighted_mean_cos2_delta']:.3f}; "
        f"CI {reliable_ci.get('ci95_low', float('nan')):.3f} to {reliable_ci.get('ci95_high', float('nan')):.3f}), "
        "consistent with a functional constraint that drift should add temporal samples while avoiding maximally disruptive local directions. "
        "Panel E top bars show window fractions; text labels report sessions with positive advantage.",
        "",
        "Interpretation: FEM-induced variability is not only a confound to subtract. In the V1 twin, empirical drift-like motion supplies feature-relevant temporal samples of natural images, while local stability predicts image-preserving axes that match measured drift geometry. Guardrail: the aggregate decoder does not predict the exact real drift trajectory. The safer claim is a functional constraint: drift should generate feature-relevant temporal samples while preserving local structure. The deterministic ridge-decoding endpoint is a V1-twin feature-decoding proxy, not a literal mutual-information estimate or a claim that exact measured trajectory order is uniquely optimal. The result is an incremental static-plus-motion versus static-only claim, not a claim that moving responses globally beat static responses. Motion sanity checks are kept out of the main panels but remain in the manifest: effective/requested RMS median 1.0 and maximum clipping 0.0 across generated families/scales.",
        "",
    ]
    caption_path.write_text("\n".join(lines), encoding="utf-8")
    return caption_path


def _write_readme(
    out_dir: Path,
    *,
    aggregate_dir: Path,
    incremental_dir: Path,
    geometry_dir: Path,
    stability_dir: Path,
    outputs: dict[str, str],
) -> Path:
    readme = out_dir / f"{FIGURE_STEM}_README.md"
    readme.write_text(
        "\n".join(
            [
                "# Figure 4 Active-Sensing Headline",
                "",
                "Generated by `declan/fig4_active_sensing/generate_fig4_active_sensing.py`.",
                "",
                "This v0 replaces the initial wrapper around the older active-sensing movie-information figure.",
                "The main result now frames BackImage aggregate FEM information as the functional counterpart of FEM-linked reafferent V1 variability.",
                "",
                "## Outputs",
                "",
                f"- `{Path(outputs['pdf']).name}`",
                f"- `{Path(outputs['png']).name}`",
                f"- `{Path(outputs['svg']).name}`",
                f"- `{Path(outputs['caption_md']).name}`",
                f"- `{Path(outputs['stats_json']).name}`",
                "",
                "## Source Tables",
                "",
                f"- Aggregate run: `{aggregate_dir}`",
                f"- Incremental gains: `{incremental_dir / 'incremental_gain_vs_static.csv'}`",
                f"- Control contrasts: `{incremental_dir / 'incremental_gain_contrasts.csv'}`",
                f"- Motion QC: `{aggregate_dir / 'aggregate_motion_metadata.csv'}`",
                f"- Local edge geometry: `{geometry_dir / 'orientation_alignment_summary.csv'}`",
                f"- Local edge-geometry CIs: `{geometry_dir / 'backimage_image_fem_windows.csv'}`",
                f"- Edge-parallel stability: `{stability_dir / 'stability_summary.csv'}`",
                f"- Schematic thumbnail source: `{DEFAULT_THUMBNAIL}`",
                "",
                "## Interpretation Boundary",
                "",
                "- The figure is canonical 756-unit V1-twin evidence, not the older 16-channel natural-image movie-information endpoint.",
                "- The endpoint is deterministic static-plus-motion feature-decoding gain over a static-only decoder in ridge -MSE units, not literal mutual information.",
                "- The supported claim is distributional and scale/readout scoped: empirical drift-like motion supplies feature-relevant temporal samples beyond static responses and robustly beats OU-like controls, while the absolute temporal-PC family panel keeps the Brownian/rotated caveat visible.",
                "- The local-geometry panel is the payoff: edge-parallel motion is predicted to preserve local image/V1-twin structure, and measured drift axes are biased toward those stable directions.",
                "- Do not read this as exact trajectory prediction; the supported claim is a functional constraint on drift geometry.",
                "- Motion sanity is documented in the stats manifest rather than plotted as a main panel.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return readme


def make_figure(
    *,
    aggregate_dir: Path = DEFAULT_AGGREGATE_DIR,
    incremental_dir: Path = DEFAULT_INCREMENTAL_DIR,
    geometry_dir: Path = DEFAULT_GEOMETRY_DIR,
    stability_dir: Path = DEFAULT_STABILITY_DIR,
    out_dir: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """Create the active-sensing headline figure from cached BackImage outputs."""
    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    gain = _read_csv(incremental_dir / "incremental_gain_vs_static.csv")
    contrasts = _read_csv(incremental_dir / "incremental_gain_contrasts.csv")
    geometry = _read_csv(geometry_dir / "orientation_alignment_summary.csv")
    geometry_windows = _read_csv(geometry_dir / "backimage_image_fem_windows.csv")
    stability = _read_csv(stability_dir / "stability_summary.csv")
    qc = _motion_qc(aggregate_dir)
    edge_ci = _edge_axis_session_bootstrap_ci(geometry_windows)

    fig = plt.figure(figsize=(12.2, 10.4), constrained_layout=False)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        left=0.07,
        right=0.985,
        bottom=0.065,
        top=0.92,
        wspace=0.28,
        hspace=0.52,
        height_ratios=[1.0, 1.0, 0.92],
    )
    fig.suptitle(CAPTION_TITLE, x=0.07, y=0.975, ha="left", fontsize=13, fontweight="bold")

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    _plot_concept(ax_a)
    _plot_gain_vs_static(ax_b, gain, qc)
    _plot_absolute_temporal_pca_gains(ax_c, gain)
    _plot_control_contrasts(ax_d, contrasts)
    _plot_local_geometry(ax_e, geometry, stability, edge_ci)

    stats = _collect_key_stats(gain, contrasts, aggregate_dir, geometry, stability, edge_ci, qc)
    caption_path = _write_caption(stats, out_dir)
    caption_text = caption_path.read_text(encoding="utf-8")
    plain_caption = _plain_caption_text(caption_text)

    pdf = out_dir / f"{FIGURE_STEM}.pdf"
    png = out_dir / f"{FIGURE_STEM}.png"
    svg = out_dir / f"{FIGURE_STEM}.svg"
    metadata = {
        "Title": CAPTION_TITLE,
        "Subject": plain_caption,
        "Creator": "VisionCore fig4 active-sensing generator",
        "Keywords": "active sensing; BackImage; FEM; V1 twin; feature decoding",
    }
    svg_metadata = {
        "Title": CAPTION_TITLE,
        "Description": plain_caption,
        "Creator": metadata["Creator"],
        "Keywords": metadata["Keywords"],
    }
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata=metadata)
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata=svg_metadata)
    plt.close(fig)

    outputs = {
        "pdf": str(pdf),
        "png": str(png),
        "svg": str(svg),
        "caption_md": str(caption_path),
        "stats_json": str(out_dir / f"{FIGURE_STEM}_stats.json"),
    }
    stats["outputs"] = outputs
    stats["source_files"] = {
        "aggregate_run_metadata": str(aggregate_dir / "run_metadata.json"),
        "incremental_gain_vs_static": str(incremental_dir / "incremental_gain_vs_static.csv"),
        "incremental_gain_contrasts": str(incremental_dir / "incremental_gain_contrasts.csv"),
        "aggregate_motion_summary": str(aggregate_dir / "aggregate_motion_summary.csv"),
        "aggregate_motion_metadata": str(aggregate_dir / "aggregate_motion_metadata.csv"),
        "orientation_alignment_summary": str(geometry_dir / "orientation_alignment_summary.csv"),
        "image_fem_windows": str(geometry_dir / "backimage_image_fem_windows.csv"),
        "stability_summary": str(stability_dir / "stability_summary.csv"),
        "schematic_thumbnail": str(DEFAULT_THUMBNAIL),
    }
    Path(outputs["stats_json"]).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    readme_path = _write_readme(
        out_dir,
        aggregate_dir=aggregate_dir,
        incremental_dir=incremental_dir,
        geometry_dir=geometry_dir,
        stability_dir=stability_dir,
        outputs=outputs,
    )
    outputs["readme_md"] = str(readme_path)
    stats["outputs"] = outputs
    Path(outputs["stats_json"]).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=DEFAULT_AGGREGATE_DIR)
    parser.add_argument("--incremental-dir", type=Path, default=DEFAULT_INCREMENTAL_DIR)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = make_figure(
        aggregate_dir=args.aggregate_dir,
        incremental_dir=args.incremental_dir,
        geometry_dir=args.geometry_dir,
        stability_dir=args.stability_dir,
        out_dir=args.out_dir,
    )
    outputs = stats["outputs"]
    print(f"Wrote {outputs['pdf']}")
    print(f"Wrote {outputs['png']}")
    print(f"Wrote {outputs['svg']}")
    print(f"Wrote {outputs['caption_md']}")
    print(f"Wrote {outputs['stats_json']}")


if __name__ == "__main__":
    main()
