#!/usr/bin/env python3
"""Bridge SSI figure panels to SF-TF power redistribution.

This diagnostic uses only cached SSI-v3 outputs.  It keeps the Rucci-style
motion transfer term Q separate from the image-weighted retinal movie power
I(k)Q(k, f_t), then places those spectral summaries next to the SSI dose curves
already used in the figure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_eye_movement_joint_sftf_power import (
    DEFAULT_OUT_DIR as DEFAULT_SFTF_DIR,
    SPATIAL_BANDS,
    db,
    spatial_edges_from_centers,
    temporal_edges,
)


ROOT = Path(__file__).resolve().parents[4]
PANEL_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
DEFAULT_OUT_DIR = PANEL_DIR / "ssi_sftf_mechanism_bridge"
DEFAULT_RADIAL_CSV = DEFAULT_SFTF_DIR / "eye_movement_joint_sftf_radial_temporal_summary.csv"
DEFAULT_BCEF_CSV = PANEL_DIR / "panel_bcef_path_bins_values.csv"
DEFAULT_H_CSV = PANEL_DIR / "panel_g_alternative_x_axes_diagnostic_values.csv"
DEFAULT_TUNING_POINTS_CSV = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_rr100_dense_sf_tf_speed_pref_groups_v1"
    / "cycle_valid_dense_sf_tf_points.csv"
)

CONDITION_ORDER = ("short_drift", "mid_drift", "long_drift", "microsaccade")
CONDITION_LABELS = {
    "short_drift": "short drift",
    "mid_drift": "mid drift",
    "long_drift": "long drift",
    "microsaccade": "micro.",
}
BAND_LABELS = {"low_sf": "low SF", "mid_sf": "mid SF", "high_sf": "high SF"}
BAND_COLORS = {"low_sf": "#0072B2", "mid_sf": "#559F76", "high_sf": "#D55E00"}
TUNING_GROUP_COMBINE = {"low_sf": "low_mid_sf", "middle_sf": "low_mid_sf", "high_sf": "high_sf"}
TUNING_GROUP_ORDER = ("low_mid_sf", "high_sf")
TUNING_GROUP_LABELS = {"low_mid_sf": "low+mid SF units", "high_sf": "high SF units"}
TUNING_GROUP_COLORS = {"low_mid_sf": "#0072B2", "high_sf": "#D55E00"}
SSI_COLORS = {
    "low_all": "#0072B2",
    "high_all": "#D55E00",
    "low_aligned": "#56B4E9",
    "high_aligned": "#E69F00",
}


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def matrix_from_column(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.asarray(sorted(frame["spatial_frequency_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(frame["temporal_frequency_hz"].unique()), dtype=np.float64)
    mat = np.full((tf.size, sf.size), np.nan, dtype=np.float64)
    sf_index = {float(v): i for i, v in enumerate(sf)}
    tf_index = {float(v): i for i, v in enumerate(tf)}
    for row in frame.itertuples(index=False):
        mat[tf_index[float(row.temporal_frequency_hz)], sf_index[float(row.spatial_frequency_cpd)]] = float(
            getattr(row, column)
        )
    return sf, tf, mat


def load_tuning_surfaces(points_csv: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = Path(points_csv)
    if not path.exists():
        return {}
    points = pd.read_csv(path)
    required = {"sf_group", "spatial_cpd", "temporal_hz", "unit_surface_z"}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"Tuning points file is missing columns: {sorted(missing)}")
    points["tuning_group"] = points["sf_group"].map(TUNING_GROUP_COMBINE).fillna(points["sf_group"])
    grouped = (
        points.dropna(subset=["sf_group", "spatial_cpd", "temporal_hz", "unit_surface_z"])
        .groupby(["tuning_group", "spatial_cpd", "temporal_hz"], sort=True)["unit_surface_z"]
        .mean()
        .reset_index()
    )
    surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group_id, group in grouped.groupby("tuning_group", sort=False):
        sf = np.asarray(sorted(group["spatial_cpd"].unique()), dtype=np.float64)
        tf = np.asarray(sorted(group["temporal_hz"].unique()), dtype=np.float64)
        mat = np.full((tf.size, sf.size), np.nan, dtype=np.float64)
        sf_index = {float(v): i for i, v in enumerate(sf)}
        tf_index = {float(v): i for i, v in enumerate(tf)}
        for row in group.itertuples(index=False):
            mat[tf_index[float(row.temporal_hz)], sf_index[float(row.spatial_cpd)]] = float(row.unit_surface_z)
        finite = mat[np.isfinite(mat)]
        if finite.size == 0:
            continue
        lo = float(np.nanpercentile(finite, 5.0))
        hi = float(np.nanpercentile(finite, 98.0))
        if hi <= lo:
            hi = float(np.nanmax(finite))
            lo = float(np.nanmin(finite))
        if hi > lo:
            mat = np.clip((mat - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        surfaces[str(group_id)] = (sf, tf, mat)
    return surfaces


def overlay_tuning_contours(
    ax: plt.Axes,
    tuning_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    level: float = 0.68,
    linewidth: float = 1.25,
) -> None:
    for group_id in TUNING_GROUP_ORDER:
        surface = tuning_surfaces.get(group_id)
        if surface is None:
            continue
        sf, tf, mat = surface
        if mat.shape[0] < 2 or mat.shape[1] < 2 or not np.any(np.isfinite(mat)):
            continue
        try:
            ax.contour(
                sf,
                tf,
                mat,
                levels=[float(level)],
                colors=[TUNING_GROUP_COLORS[group_id]],
                linewidths=float(linewidth),
                alpha=0.95,
            )
        except ValueError:
            continue


def add_tuning_legend(
    ax: plt.Axes,
    tuning_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    loc: str = "lower left",
) -> None:
    handles = [
        Line2D([0], [0], color=TUNING_GROUP_COLORS[group_id], linewidth=1.6, label=TUNING_GROUP_LABELS[group_id])
        for group_id in TUNING_GROUP_ORDER
        if group_id in tuning_surfaces
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=6.5, loc=loc, title="unit tuning", title_fontsize=6.5)


def compute_band_metrics(radial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, condition_frame in radial.groupby("condition", sort=False):
        for band_id, band_label, low_cpd, high_cpd in SPATIAL_BANDS:
            band_frame = condition_frame[
                (condition_frame["spatial_frequency_cpd"] >= float(low_cpd))
                & (condition_frame["spatial_frequency_cpd"] < float(high_cpd))
            ].copy()
            if band_frame.empty:
                continue
            for quantity, label in (
                ("motion_q_mean", "motion transfer Q"),
                ("modulation_power_mean", "image-weighted retinal power"),
            ):
                by_tf = band_frame.groupby("temporal_frequency_hz", sort=True)[quantity].mean()
                freq = by_tf.index.to_numpy(dtype=np.float64)
                power = by_tf.to_numpy(dtype=np.float64)
                total = float(np.nansum(power))
                centroid = float(np.nansum(freq * power) / max(total, 1e-30))
                fast_fraction = float(np.nansum(power[freq >= 30.0]) / max(total, 1e-30))
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": CONDITION_LABELS.get(condition, condition.replace("_", " ")),
                        "spatial_band": band_id,
                        "spatial_band_label": band_label,
                        "spatial_low_cpd": float(low_cpd),
                        "spatial_high_cpd": float(high_cpd),
                        "quantity": quantity,
                        "quantity_label": label,
                        "total_nonzero_tf_power": total,
                        "tf_power_centroid_hz": centroid,
                        "fast_tf_fraction": fast_fraction,
                    }
                )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    baselines = (
        metrics[metrics["condition"].eq("all_real_fem")]
        .loc[:, ["spatial_band", "quantity", "total_nonzero_tf_power"]]
        .rename(columns={"total_nonzero_tf_power": "all_real_total_nonzero_tf_power"})
    )
    metrics = metrics.merge(baselines, on=["spatial_band", "quantity"], how="left")
    metrics["total_power_over_all_real_same_band"] = (
        metrics["total_nonzero_tf_power"] / metrics["all_real_total_nonzero_tf_power"]
    )
    return metrics


def endpoint_table(curves: Iterable[tuple[str, pd.DataFrame, str, str]]) -> pd.DataFrame:
    rows = []
    for curve_id, frame, x_col, y_col in curves:
        clean = frame.sort_values(x_col)
        if clean.empty:
            continue
        first = clean.iloc[0]
        last = clean.iloc[-1]
        rows.append(
            {
                "curve": curve_id,
                "first_x": float(first[x_col]),
                "last_x": float(last[x_col]),
                "first_y": float(first[y_col]),
                "last_y": float(last[y_col]),
                "last_minus_first_y": float(last[y_col] - first[y_col]),
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(
    ax: plt.Axes,
    radial: pd.DataFrame,
    column: str,
    title: str,
    cbar_label: str,
    *,
    tuning_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    show_tuning_legend: bool = False,
) -> None:
    sub = radial[radial["condition"].eq("all_real_fem")].copy()
    sf, tf, mat = matrix_from_column(sub, column)
    db_mat = db(mat)
    sf_edges = spatial_edges_from_centers(sf)
    tf_edges = temporal_edges(tf)
    mesh = ax.pcolormesh(
        sf_edges,
        tf_edges,
        db_mat,
        shading="auto",
        cmap="hot",
        vmin=float(np.nanpercentile(db_mat, 5)),
        vmax=float(np.nanpercentile(db_mat, 98)),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")
    ax.set_xticks([0.5, 1, 3, 10], ["0.5", "1", "3", "10"])
    ax.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
    ax.set_title(title, loc="left", fontweight="bold")
    if tuning_surfaces:
        overlay_tuning_contours(ax, tuning_surfaces)
        if show_tuning_legend:
            add_tuning_legend(ax, tuning_surfaces)
        ax.set_xlim(float(sf_edges[0]), float(sf_edges[-1]))
        ax.set_ylim(float(tf_edges[0]), float(tf_edges[-1]))
    cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.015, shrink=0.9)
    cbar.set_label(cbar_label)


def plot_band_ratios(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    sub = metrics[
        metrics["quantity"].eq("modulation_power_mean") & metrics["condition"].isin(CONDITION_ORDER)
    ].copy()
    x = np.arange(len(CONDITION_ORDER))
    for band_id in ("low_sf", "mid_sf", "high_sf"):
        y = []
        for condition in CONDITION_ORDER:
            row = sub[sub["condition"].eq(condition) & sub["spatial_band"].eq(band_id)]
            y.append(float(row["total_power_over_all_real_same_band"].iloc[0]) if not row.empty else np.nan)
        ax.plot(x, y, marker="o", linewidth=2.0, color=BAND_COLORS[band_id], label=BAND_LABELS[band_id])
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_yscale("log")
    ax.set_ylabel("relative I(k)Q power")
    ax.set_title("C. Movement scale changes low-SF power most", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)


def plot_band_centroids(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    sub = metrics[
        metrics["quantity"].eq("modulation_power_mean") & metrics["condition"].isin(CONDITION_ORDER)
    ].copy()
    x = np.arange(len(CONDITION_ORDER))
    for band_id in ("low_sf", "mid_sf", "high_sf"):
        y = []
        for condition in CONDITION_ORDER:
            row = sub[sub["condition"].eq(condition) & sub["spatial_band"].eq(band_id)]
            y.append(float(row["tf_power_centroid_hz"].iloc[0]) if not row.empty else np.nan)
        ax.plot(x, y, marker="o", linewidth=2.0, color=BAND_COLORS[band_id], label=BAND_LABELS[band_id])
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel("TF centroid (Hz)")
    ax.set_title("D. Low-SF power shifts to slower TFs", loc="left", fontweight="bold")


def plot_ssi_path_curves(ax: plt.Axes, bcef: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("low_all", "Low-SF", "low_lt0p5", "strong_contours_no_osi"),
        ("high_all", "High-SF", "high_ge0p75", "strong_contours_no_osi"),
        ("low_aligned", "Low-SF aligned", "low_lt0p5", "contour_matched"),
        ("high_aligned", "High-SF aligned", "high_ge0p75", "contour_matched"),
    )
    endpoint_inputs = []
    for curve_id, label, sf_group, relation in specs:
        sub = bcef[
            bcef["context"].eq("drift_only")
            & bcef["sf_group"].eq(sf_group)
            & bcef["relation"].eq(relation)
        ].copy()
        sub = sub.sort_values("path_median_arcmin")
        endpoint_inputs.append((curve_id, sub, "path_median_arcmin", "ssi_percent_vs_cell_baseline"))
        linestyle = "--" if "aligned" in curve_id else "-"
        ax.plot(
            sub["path_median_arcmin"],
            sub["ssi_percent_vs_cell_baseline"],
            marker="o",
            linewidth=2.0,
            linestyle=linestyle,
            color=SSI_COLORS[curve_id],
            label=label,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlim(82.0, 166.0)
    ax.set_xticks([90, 120, 150])
    ax.set_xlabel("total FEM path (arcmin)")
    ax.set_ylabel("SSI change (%)")
    ax.set_title("E. SSI path-dose split in ssi_figure", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8, ncol=2)
    return endpoint_table(endpoint_inputs)


def plot_ssi_rms_curves(ax: plt.Axes, h_values: pd.DataFrame) -> pd.DataFrame:
    sub = h_values[
        h_values["population_key"].eq("high_sf_aligned") & h_values["metric_family"].eq("component_rms")
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    keep_through = int(sub["component_bin_order"].max()) - 1
    sub = sub[sub["component_bin_order"] <= keep_through]
    endpoint_inputs = []
    for component, color, label in (
        ("across", "#B24A3B", "contour-normal"),
        ("along", "#2B6CB0", "contour-parallel"),
    ):
        comp = sub[sub["component"].eq(component)].sort_values("plot_median")
        endpoint_inputs.append((component, comp, "plot_median", "ssi_percent_vs_cell_baseline"))
        ax.plot(
            comp["plot_median"],
            comp["ssi_percent_vs_cell_baseline"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlabel("component RMS excursion (arcmin)")
    ax.set_ylabel("SSI change (%)")
    ax.set_title("F. High-SF aligned SSI is axis-sensitive", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    return endpoint_table(endpoint_inputs)


def write_summary(
    out_dir: Path,
    *,
    radial_csv: Path,
    bcef_csv: Path,
    h_csv: Path,
    tuning_points_csv: Path,
    metrics: pd.DataFrame,
    path_endpoints: pd.DataFrame,
    rms_endpoints: pd.DataFrame,
) -> None:
    def metric_line(condition: str, band: str) -> str:
        row = metrics[
            metrics["quantity"].eq("modulation_power_mean")
            & metrics["condition"].eq(condition)
            & metrics["spatial_band"].eq(band)
        ].iloc[0]
        return (
            f"{float(row['total_power_over_all_real_same_band']):.3g}x all-real, "
            f"centroid {float(row['tf_power_centroid_hz']):.3g} Hz"
        )

    endpoint_lookup = {row["curve"]: row for row in path_endpoints.to_dict("records")}
    rms_lookup = {row["curve"]: row for row in rms_endpoints.to_dict("records")}
    lines = [
        "# SSI/SF-TF Mechanism Bridge",
        "",
        "This diagnostic ties the cached SSI figure-v3 dose curves to the SF-TF redistribution analysis.",
        "",
        f"- SF-TF source: `{_relative(radial_csv)}`",
        f"- SF-TF unit-tuning contours: `{_relative(tuning_points_csv)}`",
        "- Unit-tuning contours use the final-figure population split: original low-SF and middle-SF units are combined into one low+mid group.",
        f"- SSI path source: `{_relative(bcef_csv)}`",
        f"- SSI contour-axis source: `{_relative(h_csv)}`",
        "- Spectral short/mid/long drift classes are q25/q25-q75/q75 trace-bank summaries, not the exact eight SSI path bins.",
        "",
        "## Spectral Summary",
        "",
        f"- Short-to-long drift low-SF retinal power: {metric_line('short_drift', 'low_sf')} -> {metric_line('long_drift', 'low_sf')}.",
        f"- Short-to-long drift high-SF retinal power: {metric_line('short_drift', 'high_sf')} -> {metric_line('long_drift', 'high_sf')}.",
        f"- Microsaccade windows: low-SF {metric_line('microsaccade', 'low_sf')}; high-SF {metric_line('microsaccade', 'high_sf')}.",
        "",
        "## SSI Endpoints",
        "",
    ]
    for key, label in (
        ("low_all", "Low-SF all"),
        ("high_all", "High-SF all"),
        ("low_aligned", "Low-SF aligned"),
        ("high_aligned", "High-SF aligned"),
    ):
        row = endpoint_lookup.get(key)
        if row is None:
            continue
        lines.append(
            f"- {label}: {row['first_y']:.3g}% -> {row['last_y']:.3g}% "
            f"({row['last_minus_first_y']:+.3g} pp across the drift path bins)."
        )
    for key, label in (("across", "Contour-normal RMS"), ("along", "Contour-parallel RMS")):
        row = rms_lookup.get(key)
        if row is None:
            continue
        lines.append(
            f"- {label}, high-SF aligned: {row['first_y']:.3g}% -> {row['last_y']:.3g}% "
            f"({row['last_minus_first_y']:+.3g} pp across displayed RMS bins)."
        )
    lines.extend(
        [
            "",
            "## Working Interpretation",
            "",
            "Path length mainly buys low-SF movie power.  High-SF movie power is already near saturation across short/mid/long drift, so more path does not supply a comparable high-SF benefit.  For high-SF contour-aligned units, the important variable becomes trajectory shape: contour-normal excursion disrupts the aligned contour signal more than contour-parallel excursion.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    radial = pd.read_csv(args.radial_csv)
    bcef = pd.read_csv(args.bcef_csv)
    h_values = pd.read_csv(args.h_csv)
    tuning_surfaces = load_tuning_surfaces(Path(args.tuning_points_csv))
    metrics = compute_band_metrics(radial)
    metrics.to_csv(out_dir / "ssi_sftf_mechanism_bridge_band_metrics.csv", index=False)

    fig = plt.figure(figsize=(12.8, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.36)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    plot_heatmap(
        axes[0],
        radial,
        "motion_q_mean",
        "A. Motion-only transfer",
        "10 log10 Q",
        tuning_surfaces=tuning_surfaces,
    )
    plot_heatmap(
        axes[1],
        radial,
        "modulation_power_mean",
        "B. Image-weighted movie power",
        "10 log10 I(k)Q",
        tuning_surfaces=tuning_surfaces,
        show_tuning_legend=True,
    )
    plot_band_ratios(axes[2], metrics)
    plot_band_centroids(axes[3], metrics)
    path_endpoints = plot_ssi_path_curves(axes[4], bcef)
    rms_endpoints = plot_ssi_rms_curves(axes[5], h_values)
    path_endpoints.to_csv(out_dir / "ssi_sftf_mechanism_bridge_ssi_path_endpoints.csv", index=False)
    rms_endpoints.to_csv(out_dir / "ssi_sftf_mechanism_bridge_ssi_rms_endpoints.csv", index=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Mechanism bridge: FEM redistributes SF-TF power before SSI splits by population",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    png = out_dir / "ssi_sftf_mechanism_bridge.png"
    pdf = out_dir / "ssi_sftf_mechanism_bridge.pdf"
    svg = out_dir / "ssi_sftf_mechanism_bridge.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    write_summary(
        out_dir,
        radial_csv=Path(args.radial_csv),
        bcef_csv=Path(args.bcef_csv),
        h_csv=Path(args.h_csv),
        tuning_points_csv=Path(args.tuning_points_csv),
        metrics=metrics,
        path_endpoints=path_endpoints,
        rms_endpoints=rms_endpoints,
    )
    return {"png": png, "pdf": pdf, "svg": svg, "metrics_csv": out_dir / "ssi_sftf_mechanism_bridge_band_metrics.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radial-csv", type=Path, default=DEFAULT_RADIAL_CSV)
    parser.add_argument("--bcef-csv", type=Path, default=DEFAULT_BCEF_CSV)
    parser.add_argument("--h-csv", type=Path, default=DEFAULT_H_CSV)
    parser.add_argument("--tuning-points-csv", type=Path, default=DEFAULT_TUNING_POINTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(f"Wrote {paths['png']}")


if __name__ == "__main__":
    main()
