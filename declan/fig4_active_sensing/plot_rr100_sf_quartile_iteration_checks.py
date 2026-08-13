#!/usr/bin/env python3
"""Regenerate the pre-Figure-4 tuning checks using new RR100 SF quartiles."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from declan.fig4_active_sensing.plot_rr100_sf_tf_parametric_preferences import (
    DEFAULT_NPZ,
    DEFAULT_UNIT_TABLE,
    SF_SUPPORT,
    TF_SUPPORT,
    configure_matplotlib,
    file_identity,
    load_data,
    save_figure,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_sf_quartile_iteration_checks_v1"
QUARTILE_ORDER = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
QUARTILE_LABELS = {
    "sf_q1": "SF Q1 (lowest)",
    "sf_q2": "SF Q2",
    "sf_q3": "SF Q3",
    "sf_q4": "SF Q4 (highest)",
}
QUARTILE_COLORS = {
    "sf_q1": "#46327E",
    "sf_q2": "#2A788E",
    "sf_q3": "#2FB47C",
    "sf_q4": "#BDDF26",
}
FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNIT_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--contour-level", type=float, default=0.68)
    return parser.parse_args()


def assign_sf_quartiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["sf_quartile"] = "invalid_model"
    valid = out[out["model_valid"]].copy().sort_values(["preferred_sf_cpd", "rr100_index"])
    labels = pd.qcut(valid["preferred_sf_cpd"], q=4, labels=QUARTILE_ORDER)
    out.loc[valid.index, "sf_quartile"] = labels.astype(str)
    out["sf_quartile_label"] = out["sf_quartile"].map(QUARTILE_LABELS).fillna("invalid model")
    out["sf_quartile_rank"] = out["sf_quartile"].map({q: i + 1 for i, q in enumerate(QUARTILE_ORDER)})

    rows: list[dict[str, Any]] = []
    for group in QUARTILE_ORDER:
        sub = out[out["sf_quartile"].eq(group)]
        rows.append(
            {
                "sf_quartile": group,
                "sf_quartile_label": QUARTILE_LABELS[group],
                "n_units": int(len(sub)),
                "preferred_sf_min_cpd": float(sub["preferred_sf_cpd"].min()),
                "preferred_sf_max_cpd": float(sub["preferred_sf_cpd"].max()),
                "preferred_sf_median_cpd": float(sub["preferred_sf_cpd"].median()),
                "preferred_tf_median_hz": float(sub["preferred_tf_hz"].median()),
                "joint_surface_r2_median": float(sub["joint_parametric_surface_r2"].median()),
            }
        )
    return out, pd.DataFrame(rows)


def sampled_indices(grid: np.ndarray, upper: float, stride: int = 4) -> np.ndarray:
    support = np.flatnonzero(np.asarray(grid, dtype=float) <= upper * (1.0 + 1e-10))
    return np.unique(np.r_[support[::stride], support[-1]])


def normalized_unit_surfaces(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf_grid = arrays["sf_evaluation_grid_cpd"].astype(float)
    tf_grid = arrays["tf_evaluation_grid_hz"].astype(float)
    sf_idx = sampled_indices(sf_grid, SF_SUPPORT[1])
    tf_idx = sampled_indices(tf_grid, TF_SUPPORT[1])
    surfaces = np.full((len(arrays["rr100_index"]), len(tf_idx), len(sf_idx)), np.nan)
    for unit in np.flatnonzero(arrays["model_valid"]):
        surface = np.outer(
            arrays["tf_factor_normalized_curves"][unit, tf_idx],
            arrays["sf_factor_normalized_curves"][unit, sf_idx],
        ).astype(float)
        surfaces[unit] = surface / np.nanmax(surface)
    return sf_grid[sf_idx], tf_grid[tf_idx], surfaces


def setup_frequency_axis(ax: plt.Axes, *, joint: bool = False) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.9, 12.2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    if joint:
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.44, 36)
        ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.grid(True, color="0.91", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def representative_units(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in QUARTILE_ORDER:
        sub = frame[frame["sf_quartile"].eq(group)].copy()
        target = float(sub["preferred_sf_cpd"].median())
        sub["selection_distance_cpd"] = np.abs(sub["preferred_sf_cpd"] - target)
        row = sub.sort_values(
            ["selection_distance_cpd", "joint_parametric_surface_r2", "rr100_index"],
            ascending=[True, False, True],
        ).iloc[0]
        rows.append(
            {
                "selection_role": f"{group}_median_sf_representative",
                "selection_kind": "algorithmic",
                "criterion_name": "minimum_absolute_distance_to_within_quartile_median_preferred_sf_then_highest_joint_r2",
                "criterion_value": float(row["selection_distance_cpd"]),
                "rr100_index": int(row["rr100_index"]),
                "sf_quartile": group,
                "preferred_sf_cpd": float(row["preferred_sf_cpd"]),
                "preferred_tf_hz": float(row["preferred_tf_hz"]),
                "joint_parametric_surface_r2": float(row["joint_parametric_surface_r2"]),
            }
        )
    return pd.DataFrame(rows)


def plot_quartile_definition(
    arrays: dict[str, np.ndarray],
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    representatives: pd.DataFrame,
    out_dir: Path,
) -> None:
    sf_grid = arrays["sf_evaluation_grid_cpd"].astype(float)
    keep = sf_grid <= SF_SUPPORT[1] * (1.0 + 1e-10)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.5))

    ax = axes[0, 0]
    for group in QUARTILE_ORDER:
        ids = frame.loc[frame["sf_quartile"].eq(group), "rr100_index"].astype(int).to_numpy()
        curves = arrays["sf_factor_normalized_curves"][ids][:, keep].astype(float)
        curves /= np.nanmax(curves, axis=1, keepdims=True)
        for curve in curves:
            ax.plot(sf_grid[keep], curve, color=QUARTILE_COLORS[group], alpha=0.11, lw=0.7)
        mean = np.nanmean(curves, axis=0)
        sem = np.nanstd(curves, axis=0, ddof=1) / math.sqrt(len(ids))
        ax.plot(sf_grid[keep], mean, color=QUARTILE_COLORS[group], lw=2.2, label=f"{QUARTILE_LABELS[group]} (n={len(ids)})")
        ax.fill_between(sf_grid[keep], mean - sem, mean + sem, color=QUARTILE_COLORS[group], alpha=0.14, lw=0)
    setup_frequency_axis(ax)
    ax.set(xlabel="spatial frequency (cycles/deg)", ylabel="within-unit response / fitted maximum")
    ax.set_title("A. Parametric SF factors by new quartile", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[0, 1]
    rng = np.random.default_rng(47)
    for i, group in enumerate(QUARTILE_ORDER, start=1):
        sub = frame[frame["sf_quartile"].eq(group)]
        y = i + rng.uniform(-0.10, 0.10, len(sub))
        ax.scatter(sub["preferred_sf_cpd"], y, color=QUARTILE_COLORS[group], s=28, alpha=0.8, edgecolor="white", lw=0.3)
    boundaries = summary["preferred_sf_max_cpd"].iloc[:3].to_numpy(float)
    for value in boundaries:
        ax.axvline(value, color="0.35", ls="--", lw=0.9)
    setup_frequency_axis(ax)
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks(range(1, 5), [QUARTILE_LABELS[q] for q in QUARTILE_ORDER])
    ax.set_xlabel("new preferred SF (cycles/deg)")
    ax.set_title("B. Quartile assignments and empirical boundaries", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for i, group in enumerate(QUARTILE_ORDER, start=1):
        vals = frame.loc[frame["sf_quartile"].eq(group), "preferred_tf_hz"].to_numpy(float)
        parts = ax.violinplot(np.log2(vals), positions=[i], widths=0.72, showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(QUARTILE_COLORS[group])
            body.set_edgecolor(QUARTILE_COLORS[group])
            body.set_alpha(0.34)
        parts["cmedians"].set_color(QUARTILE_COLORS[group])
        ax.scatter(i + rng.uniform(-0.11, 0.11, len(vals)), np.log2(vals), s=12, color=QUARTILE_COLORS[group], alpha=0.52, lw=0)
    tf_ticks = np.asarray([0.5, 1, 2, 4, 8, 16, 32], dtype=float)
    ax.set_xticks(range(1, 5), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_yticks(np.log2(tf_ticks), [f"{v:g}" for v in tf_ticks])
    ax.set(xlabel="SF quartile", ylabel="preferred |TF| (Hz)")
    ax.set_title("C. TF preferences within SF quartiles", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="0.91", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    for row in representatives.itertuples(index=False):
        unit = int(row.rr100_index)
        group = str(row.sf_quartile)
        curve = arrays["sf_factor_normalized_curves"][unit, keep].astype(float)
        curve /= np.nanmax(curve)
        ax.plot(
            sf_grid[keep],
            curve,
            color=QUARTILE_COLORS[group],
            lw=2.0,
            label=f"RR100 {unit:02d}: {row.preferred_sf_cpd:.2g} cpd, {row.preferred_tf_hz:.2g} Hz",
        )
        ax.axvline(row.preferred_sf_cpd, color=QUARTILE_COLORS[group], alpha=0.55, lw=0.85)
    setup_frequency_axis(ax)
    ax.set(xlabel="spatial frequency (cycles/deg)", ylabel="response / fitted maximum")
    ax.set_title("D. Algorithmic median-SF representatives", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5)

    fig.suptitle("Iteration 1: new RR100 SF quartile definition", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(
        0.02,
        0.935,
        "Quartiles are computed only across 85 valid positive-dynamic-F0 models; 15 invalid models remain unassigned.",
        color="0.35",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.6, w_pad=2.2)
    save_figure(fig, out_dir, "01_sf_quartile_definition")


def group_surfaces(frame: pd.DataFrame, surfaces: np.ndarray) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    maps: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for group in QUARTILE_ORDER:
        ids = frame.loc[frame["sf_quartile"].eq(group), "rr100_index"].astype(int).to_numpy()
        mean = np.nanmean(surfaces[ids], axis=0)
        maps[group] = mean
        peak = np.unravel_index(int(np.nanargmax(mean)), mean.shape)
        rows.append(
            {
                "sf_quartile": group,
                "n_units": int(len(ids)),
                "mean_surface_min": float(np.nanmin(mean)),
                "mean_surface_max": float(np.nanmax(mean)),
                "peak_tf_index": int(peak[0]),
                "peak_sf_index": int(peak[1]),
            }
        )
    return maps, pd.DataFrame(rows)


def plot_group_surfaces(
    sf: np.ndarray,
    tf: np.ndarray,
    maps: dict[str, np.ndarray],
    map_summary: pd.DataFrame,
    out_dir: Path,
    contour_level: float,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.3), sharex=True, sharey=True)
    mesh = None
    rows: list[dict[str, Any]] = []
    for col, group in enumerate(QUARTILE_ORDER):
        mean = maps[group]
        normalized = (mean - np.nanpercentile(mean, 5)) / max(
            np.nanpercentile(mean, 98) - np.nanpercentile(mean, 5), 1e-12
        )
        normalized = np.clip(normalized, 0, 1)
        mesh = axes[0, col].pcolormesh(sf, tf, mean, shading="auto", cmap="viridis", vmin=0, vmax=1)
        axes[1, col].pcolormesh(sf, tf, normalized, shading="auto", cmap="viridis", vmin=0, vmax=1)
        axes[1, col].contour(sf, tf, normalized, levels=[contour_level], colors=[QUARTILE_COLORS[group]], linewidths=2.0)
        peak = np.unravel_index(int(np.nanargmax(mean)), mean.shape)
        rows.append(
            {
                "sf_quartile": group,
                "n_units": int(map_summary.loc[map_summary["sf_quartile"].eq(group), "n_units"].iloc[0]),
                "mean_surface_peak_sf_cpd": float(sf[peak[1]]),
                "mean_surface_peak_tf_hz": float(tf[peak[0]]),
                "contour_level_after_5_98_percentile_normalization": float(contour_level),
            }
        )
        axes[0, col].set_title(f"{QUARTILE_LABELS[group]}\nn={rows[-1]['n_units']}, peak {sf[peak[1]]:.2g} cpd / {tf[peak[0]]:.2g} Hz", loc="left", fontweight="bold")
        for ax in axes[:, col]:
            setup_frequency_axis(ax, joint=True)
        axes[1, col].set_xlabel("SF (cycles/deg)")
    axes[0, 0].set_ylabel("mean unit-normalized surface\n|TF| (Hz)")
    axes[1, 0].set_ylabel("5-98% normalized + contour\n|TF| (Hz)")
    assert mesh is not None
    cax = fig.add_axes([0.925, 0.18, 0.014, 0.64])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("mean response / within-unit fitted maximum")
    fig.suptitle("Iteration 2: SF-quartile mean parametric SFxTF surfaces", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.90, bottom=0.09, top=0.88, wspace=0.18, hspace=0.28)
    save_figure(fig, out_dir, "02_sf_quartile_group_surfaces")
    return pd.DataFrame(rows)


def plot_unit_contours(
    sf: np.ndarray,
    tf: np.ndarray,
    surfaces: np.ndarray,
    frame: pd.DataFrame,
    out_dir: Path,
    contour_level: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.7), sharex=True, sharey=True)
    for ax, group in zip(axes.ravel(), QUARTILE_ORDER):
        sub = frame[frame["sf_quartile"].eq(group)]
        for row in sub.itertuples(index=False):
            surface = surfaces[int(row.rr100_index)]
            ax.contour(sf, tf, surface, levels=[contour_level], colors=[QUARTILE_COLORS[group]], linewidths=0.8, alpha=0.36)
            ax.scatter(row.preferred_sf_cpd, row.preferred_tf_hz, s=9, color=QUARTILE_COLORS[group], alpha=0.42, lw=0)
        ax.scatter(
            sub["preferred_sf_cpd"].median(),
            sub["preferred_tf_hz"].median(),
            marker="D",
            s=48,
            color=QUARTILE_COLORS[group],
            edgecolor="black",
            lw=0.5,
            zorder=5,
        )
        setup_frequency_axis(ax, joint=True)
        ax.set_title(f"{QUARTILE_LABELS[group]}: n={len(sub)}", loc="left", fontweight="bold")
    for ax in axes[-1, :]:
        ax.set_xlabel("spatial frequency (cycles/deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("|temporal frequency| (Hz)")
    legend = [
        Line2D([0], [0], color="0.35", lw=1, alpha=0.5, label=f"per-unit {contour_level:.2f} contour"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.35", markersize=4, label="fitted preference"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="0.55", markeredgecolor="black", markersize=5, label="quartile median preference"),
    ]
    axes[0, 0].legend(handles=legend, frameon=False, fontsize=6.6, loc="lower left")
    fig.suptitle("Iteration 3: individual SFxTF tuning contours within each SF quartile", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.1, w_pad=1.7)
    save_figure(fig, out_dir, "03_sf_quartile_unit_contours")


def plot_fit_ellipses(arrays: dict[str, np.ndarray], frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    sf_params = arrays["sf_parameters"].astype(float)
    tf_params = arrays["tf_parameters"].astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.7), sharex=True, sharey=True)
    rows: list[dict[str, Any]] = []
    for ax, group in zip(axes.ravel(), QUARTILE_ORDER):
        sub = frame[frame["sf_quartile"].eq(group)].copy()
        for row in sub.itertuples(index=False):
            unit = int(row.rr100_index)
            sf_fwhm = FWHM_FACTOR * sf_params[unit, 3]
            tf_fwhm = FWHM_FACTOR * tf_params[unit, 3]
            boundary = (
                np.isclose(row.preferred_sf_cpd, SF_SUPPORT[0])
                or np.isclose(row.preferred_sf_cpd, SF_SUPPORT[1])
                or np.isclose(row.preferred_tf_hz, TF_SUPPORT[0])
                or np.isclose(row.preferred_tf_hz, TF_SUPPORT[1])
            )
            ellipse = Ellipse(
                (np.log2(row.preferred_sf_cpd), np.log2(row.preferred_tf_hz)),
                width=sf_fwhm,
                height=tf_fwhm,
                fill=False,
                edgecolor=QUARTILE_COLORS[group],
                lw=0.75 if boundary else 0.95,
                ls="--" if boundary else "-",
                alpha=0.16 if boundary else 0.30,
            )
            ax.add_patch(ellipse)
            ellipse.set_clip_path(ax.patch)
            ax.scatter(np.log2(row.preferred_sf_cpd), np.log2(row.preferred_tf_hz), s=10, color=QUARTILE_COLORS[group], alpha=0.55, lw=0)
            rows.append(
                {
                    "rr100_index": unit,
                    "sf_quartile": group,
                    "preferred_sf_cpd": float(row.preferred_sf_cpd),
                    "preferred_tf_hz": float(row.preferred_tf_hz),
                    "sf_fwhm_octaves": float(sf_fwhm),
                    "tf_fwhm_octaves": float(tf_fwhm),
                    "preference_on_fit_support_boundary": bool(boundary),
                    "joint_parametric_surface_r2": float(row.joint_parametric_surface_r2),
                }
            )
        med_sf = float(np.log2(sub["preferred_sf_cpd"]).median())
        med_tf = float(np.log2(sub["preferred_tf_hz"]).median())
        ids = sub["rr100_index"].astype(int).to_numpy()
        med_sf_w = float(np.nanmedian(FWHM_FACTOR * sf_params[ids, 3]))
        med_tf_w = float(np.nanmedian(FWHM_FACTOR * tf_params[ids, 3]))
        median_ellipse = Ellipse(
            (med_sf, med_tf), med_sf_w, med_tf_w, fill=False,
            edgecolor=QUARTILE_COLORS[group], lw=2.2, alpha=0.95,
        )
        ax.add_patch(median_ellipse)
        median_ellipse.set_clip_path(ax.patch)
        ax.scatter(med_sf, med_tf, marker="D", s=46, color=QUARTILE_COLORS[group], edgecolor="black", lw=0.5, zorder=5)
        ax.set_xlim(np.log2([0.9, 12.2]))
        ax.set_ylim(np.log2([0.44, 36]))
        ax.set_xticks(np.log2([1, 2, 4, 8]), ["1", "2", "4", "8"])
        ax.set_yticks(np.log2([0.5, 1, 2, 4, 8, 16, 32]), ["0.5", "1", "2", "4", "8", "16", "32"])
        ax.grid(True, color="0.91", lw=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{QUARTILE_LABELS[group]}: n={len(sub)}", loc="left", fontweight="bold")
    for ax in axes[-1, :]:
        ax.set_xlabel("preferred SF (cycles/deg; log2 axis)")
    for ax in axes[:, 0]:
        ax.set_ylabel("preferred |TF| (Hz; log2 axis)")
    legend = [
        Line2D([0], [0], color="0.35", lw=1, label="interior preference"),
        Line2D([0], [0], color="0.35", lw=1, ls="--", label="support-boundary preference"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="0.55", markeredgecolor="black", markersize=5, label="median center/FWHM"),
    ]
    axes[0, 0].legend(handles=legend, frameon=False, fontsize=6.6, loc="lower left")
    fig.suptitle("Iteration 4: parametric SF/TF FWHM ellipses by new SF quartile", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.935, "Ellipses use separable factor sigma values; dashed fits have a preferred SF or TF on the declared fit-support boundary.", color="0.35", fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.1, w_pad=1.7)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(out_dir / f"04_sf_quartile_fit_ellipses.{suffix}", **kwargs)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    arrays, frame = load_data(args.npz, args.unit_table)
    frame, quartile_summary = assign_sf_quartiles(frame)
    representatives = representative_units(frame)
    sf, tf, surfaces = normalized_unit_surfaces(arrays)

    plot_quartile_definition(arrays, frame, quartile_summary, representatives, args.out_dir)
    maps, map_index_summary = group_surfaces(frame, surfaces)
    surface_summary = plot_group_surfaces(sf, tf, maps, map_index_summary, args.out_dir, float(args.contour_level))
    plot_unit_contours(sf, tf, surfaces, frame, args.out_dir, float(args.contour_level))
    ellipse_table = plot_fit_ellipses(arrays, frame, args.out_dir)

    frame.to_csv(args.out_dir / "sf_quartile_unit_assignments.csv", index=False)
    quartile_summary.to_csv(args.out_dir / "sf_quartile_summary.csv", index=False)
    representatives.to_csv(args.out_dir / "sf_quartile_representative_units.csv", index=False)
    surface_summary.to_csv(args.out_dir / "sf_quartile_group_surface_summary.csv", index=False)
    ellipse_table.to_csv(args.out_dir / "sf_quartile_fit_ellipse_unit_table.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "map_first_iterated_tuning_checks",
        "source_npz": file_identity(args.npz),
        "historical_unit_table": file_identity(args.unit_table),
        "quartile_contract": "pandas.qcut on preferred_sf_cpd across valid models only; no imputation of invalid models",
        "n_units_total": int(len(frame)),
        "n_valid_models": int(frame["model_valid"].sum()),
        "n_invalid_models": int((~frame["model_valid"]).sum()),
        "quartile_counts": frame[frame["model_valid"]]["sf_quartile"].value_counts().sort_index().to_dict(),
        "quartile_boundaries_cpd": quartile_summary["preferred_sf_max_cpd"].iloc[:3].tolist(),
        "contour_level": float(args.contour_level),
        "surface_scaling": "each unit normalized to its fitted maximum on declared support; group surfaces are arithmetic means",
        "artifacts": [
            "README.md",
            "01_sf_quartile_definition.{png,pdf,svg}",
            "02_sf_quartile_group_surfaces.{png,pdf,svg}",
            "03_sf_quartile_unit_contours.{png,pdf,svg}",
            "04_sf_quartile_fit_ellipses.{png,pdf,svg}",
            "sf_quartile_unit_assignments.csv",
            "sf_quartile_summary.csv",
            "sf_quartile_representative_units.csv",
            "sf_quartile_group_surface_summary.csv",
            "sf_quartile_fit_ellipse_unit_table.csv",
        ],
        "not_run": "No Figure 4 SSI panels or retinal-power/passband-overlap summaries were regenerated at this checkpoint.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    valid = frame[frame["model_valid"]]
    sf_tf_rho = float(valid[["preferred_sf_cpd", "preferred_tf_hz"]].corr(method="spearman").iloc[0, 1])
    boundary_counts = ellipse_table.groupby("sf_quartile")["preference_on_fit_support_boundary"].sum().to_dict()
    readme_lines = [
        "# RR100 SF-quartile iteration checks",
        "",
        "This is a pre-Figure-4, map-first checkpoint using the new separable positive-dynamic-F0 parametric models.",
        "It does not regenerate or overwrite the Figure 4 SSI panels.",
        "",
        "## Quartile contract",
        "",
        "Quartiles use `pandas.qcut(preferred_sf_cpd, 4)` across the 85 valid models only. The 15 invalid models remain explicit and unassigned.",
        "",
    ]
    for row in quartile_summary.itertuples(index=False):
        readme_lines.append(
            f"- {row.sf_quartile_label}: n={int(row.n_units)}, {row.preferred_sf_min_cpd:.3f}-{row.preferred_sf_max_cpd:.3f} cpd; "
            f"median TF={row.preferred_tf_median_hz:.2f} Hz; support-boundary fits={int(boundary_counts.get(row.sf_quartile, 0))}."
        )
    readme_lines.extend(
        [
            "",
            "## Check sequence",
            "",
            "1. `01_sf_quartile_definition`: marginal SF factors, assignments, within-quartile TF preferences, and auditable representatives.",
            "2. `02_sf_quartile_group_surfaces`: quartile-mean parametric SF-by-TF surfaces and normalized contours.",
            "3. `03_sf_quartile_unit_contours`: individual unit half-height contours within each quartile.",
            "4. `04_sf_quartile_fit_ellipses`: factor-derived FWHM ellipses, retaining support-boundary cases as dashed controls.",
            "",
            "## Visible checkpoint observations",
            "",
            f"- SF and TF preference are modestly anticorrelated across valid units (Spearman rho={sf_tf_rho:+.3f}).",
            "- Group-mean surface peaks move monotonically from low SF/high TF in Q1 toward high SF/lower TF in Q4.",
            "- Q1's group contour reaches the 1-cpd lower SF support, so its low-SF extent is support-limited.",
            "- Individual contours remain heterogeneous and broad; quartile membership is not a narrow joint SF/TF subtype.",
            "",
            "## Scope caution",
            "",
            "These plots reconstruct the new parametric factors. They are not the older sampled dense-probe response surfaces. Retinal-power/passband overlap and SSI-by-motion checks remain intentionally unrun until this grouping checkpoint is accepted.",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(quartile_summary.to_string(index=False))


if __name__ == "__main__":
    main()
