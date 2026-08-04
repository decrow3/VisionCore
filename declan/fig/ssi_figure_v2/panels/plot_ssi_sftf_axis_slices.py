#!/usr/bin/env python3
"""Contour-axis SF/TF slices for the SSI-v3 image and eye-movement banks.

The radial SF/TF diagnostic averages over all Fourier orientations.  This
script splits the same factorization into contour-parallel and contour-normal
spatial components using ``image_edge_axis_deg``.  It estimates:

    directional movie power(k, f_t) ~= I_axis(k) * Q_axis(k, f_t)

where ``I_axis`` is static image power within an angular wedge around the local
contour axis or its normal, and ``Q_axis`` is the temporal spectrum of eye-trace
displacement projected onto that component.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_eye_movement_joint_sftf_power import (
    SPATIAL_BANDS,
    abs_temporal_frequency_bins,
    condition_label,
    db,
    spatial_edges_from_centers,
    temporal_edges,
)
from plot_eye_movement_power_spectrum_shift import (
    DEFAULT_IMAGE_TABLE,
    DEFAULT_TRACE_TABLE,
    DEFAULT_TRACE_XY,
    DT_S,
    PATCH_SIZE_PX,
    ROOT,
    frequency_grid,
    load_image_powers,
    radial_edges,
    select_trace_groups,
)
from plot_ssi_sftf_mechanism_bridge import (
    DEFAULT_TUNING_POINTS_CSV,
    add_tuning_legend,
    load_tuning_surfaces,
    overlay_tuning_contours,
)


PANEL_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
DEFAULT_OUT_DIR = PANEL_DIR / "ssi_sftf_axis_slices"
DEFAULT_CONDITIONS = ("all_real_fem", "long_drift", "microsaccade")
COMPONENT_LABELS = {
    "along": "contour-parallel SF",
    "across": "contour-normal SF",
}


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def geometric_centers(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    return np.sqrt(values[:-1] * values[1:])


def axial_angle_delta_deg(angle_deg: np.ndarray, target_deg: float) -> np.ndarray:
    return np.abs(((np.asarray(angle_deg, dtype=np.float64) - float(target_deg) + 90.0) % 180.0) - 90.0)


def directional_image_power(
    power2d: np.ndarray,
    fx_cpd: np.ndarray,
    fy_cpd: np.ndarray,
    rr_cpd: np.ndarray,
    edges: np.ndarray,
    *,
    contour_axis_deg: float,
    component: str,
    wedge_half_width_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    if component not in {"along", "across"}:
        raise ValueError(f"Unknown component {component!r}")
    target_axis = float(contour_axis_deg) if component == "along" else float(contour_axis_deg) + 90.0
    wave_angle = (np.degrees(np.arctan2(fy_cpd, fx_cpd)) % 180.0).astype(np.float64)
    axis_delta = axial_angle_delta_deg(wave_angle, target_axis)
    arr = np.asarray(power2d, dtype=np.float64)
    rr = np.asarray(rr_cpd, dtype=np.float64)
    values: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (rr >= float(lo)) & (rr < float(hi)) & (axis_delta <= float(wedge_half_width_deg))
        counts.append(int(np.sum(mask)))
        values.append(float(np.nanmean(arr[mask])) if np.any(mask) else float("nan"))
    return np.asarray(values, dtype=np.float64), np.asarray(counts, dtype=int)


def projected_q_spectrum(
    traces_xy_deg: np.ndarray,
    sf_centers_cpd: np.ndarray,
    *,
    contour_axis_deg: float,
    component: str,
    dt_s: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces_xy_deg, dtype=np.float32)
    if traces.ndim != 3 or traces.shape[1] < 2 or traces.shape[2] != 2:
        raise ValueError(f"Expected traces with shape (n,T,2); got {traces.shape}")
    if component not in {"along", "across"}:
        raise ValueError(f"Unknown component {component!r}")
    sf = np.asarray(sf_centers_cpd, dtype=np.float32)
    freq_hz, freq_bins = abs_temporal_frequency_bins(traces.shape[1], float(dt_s))
    q = np.zeros((sf.size, freq_hz.size), dtype=np.float64)
    theta = math.radians(float(contour_axis_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    n_trace = 0
    for start in range(0, traces.shape[0], int(chunk_size)):
        chunk = traces[start : start + int(chunk_size)].astype(np.float32, copy=True)
        chunk -= np.nanmean(chunk, axis=1, keepdims=True)
        # Match the contour-component convention in panel_g_alternative_x_axes_diagnostic.py.
        if component == "along":
            displacement = chunk[:, :, 0] * cos_t + chunk[:, :, 1] * sin_t
        else:
            displacement = -chunk[:, :, 0] * sin_t + chunk[:, :, 1] * cos_t
        phase = np.exp((-2j * np.pi) * displacement[:, :, None] * sf[None, None, :])
        phase -= np.mean(phase, axis=1, keepdims=True)
        spec = np.fft.fft(phase, axis=1, norm="ortho")
        power = np.sum(np.abs(spec) ** 2, axis=0)
        for freq_i, bins in enumerate(freq_bins):
            q[:, freq_i] += np.sum(power[bins, :], axis=0)
        n_trace += int(chunk.shape[0])
    if n_trace == 0:
        return freq_hz, q.astype(np.float32)
    return freq_hz, (q / float(n_trace)).astype(np.float32)


def parse_conditions(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not parts:
        raise ValueError("At least one condition is required.")
    return parts


def compute_axis_summary(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    image_table = pd.read_csv(args.image_table)
    trace_table = pd.read_csv(args.trace_table)
    traces = np.load(args.trace_xy)
    source_powers, image_rows, _example_patch = load_image_powers(
        image_table,
        max_images=int(args.max_images),
        patch_size_px=int(args.patch_size_px),
    )
    ppd = float(np.nanmedian([row["ppd"] for row in image_rows]))
    fx, fy, rr = frequency_grid(int(args.patch_size_px), ppd)
    edges = radial_edges(rr, int(args.n_sf_bins))
    sf_centers = geometric_centers(edges)
    trace_groups, _trace_group_rows = select_trace_groups(
        trace_table,
        max_traces_per_condition=int(args.max_traces_per_condition),
        seed=int(args.seed),
    )
    conditions = parse_conditions(args.conditions)
    rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for condition in conditions:
        if condition not in trace_groups:
            raise ValueError(f"Unknown condition {condition!r}; available: {sorted(trace_groups)}")
        selected_traces = np.asarray(traces[np.asarray(trace_groups[condition], dtype=int)], dtype=np.float32)
        print(f"[axis] {condition}: {selected_traces.shape[0]} traces x {len(image_rows)} image axes", flush=True)
        for component in ("along", "across"):
            acc_source = np.zeros(sf_centers.size, dtype=np.float64)
            acc_q = np.zeros((sf_centers.size, 0), dtype=np.float64)
            acc_mod = np.zeros((sf_centers.size, 0), dtype=np.float64)
            acc_counts = np.zeros(sf_centers.size, dtype=np.float64)
            n_image = 0
            freq_hz: np.ndarray | None = None
            for image_power, image_row in zip(source_powers, image_rows, strict=True):
                contour_axis_deg = float(image_row["image_edge_axis_deg"])
                if not math.isfinite(contour_axis_deg):
                    continue
                source_axis, coefficient_counts = directional_image_power(
                    image_power,
                    fx,
                    fy,
                    rr,
                    edges,
                    contour_axis_deg=contour_axis_deg,
                    component=component,
                    wedge_half_width_deg=float(args.wedge_half_width_deg),
                )
                freq_hz_i, q_axis = projected_q_spectrum(
                    selected_traces,
                    sf_centers,
                    contour_axis_deg=contour_axis_deg,
                    component=component,
                    dt_s=float(args.dt_s),
                    chunk_size=int(args.chunk_size),
                )
                if freq_hz is None:
                    freq_hz = freq_hz_i
                    acc_q = np.zeros((sf_centers.size, freq_hz.size), dtype=np.float64)
                    acc_mod = np.zeros((sf_centers.size, freq_hz.size), dtype=np.float64)
                acc_source += np.nan_to_num(source_axis, nan=0.0)
                acc_counts += coefficient_counts.astype(np.float64)
                acc_q += np.asarray(q_axis, dtype=np.float64)
                acc_mod += np.nan_to_num(source_axis[:, None], nan=0.0) * np.asarray(q_axis, dtype=np.float64)
                n_image += 1
            if freq_hz is None or n_image == 0:
                continue
            source_mean = acc_source / float(n_image)
            q_mean = acc_q / float(n_image)
            modulation_mean = acc_mod / float(n_image)
            n_coeff_mean = acc_counts / float(n_image)
            for sf_i, sf in enumerate(sf_centers):
                for tf_i, tf in enumerate(freq_hz):
                    rows.append(
                        {
                            "condition": condition,
                            "condition_label": condition_label(condition),
                            "component": component,
                            "component_label": COMPONENT_LABELS[component],
                            "spatial_frequency_cpd": float(sf),
                            "temporal_frequency_hz": float(tf),
                            "source_power_mean": float(source_mean[sf_i]),
                            "motion_q_mean": float(q_mean[sf_i, tf_i]),
                            "modulation_power_mean": float(modulation_mean[sf_i, tf_i]),
                            "mean_directional_coefficients_per_image": float(n_coeff_mean[sf_i]),
                            "n_images": int(n_image),
                            "n_traces": int(selected_traces.shape[0]),
                            "wedge_half_width_deg": float(args.wedge_half_width_deg),
                        }
                    )
            for band_id, band_label, low_cpd, high_cpd in SPATIAL_BANDS:
                keep_sf = (sf_centers >= float(low_cpd)) & (sf_centers < float(high_cpd))
                if not np.any(keep_sf):
                    continue
                band_power_by_tf = np.nansum(modulation_mean[keep_sf, :], axis=0)
                total = float(np.nansum(band_power_by_tf))
                centroid = float(np.nansum(freq_hz * band_power_by_tf) / max(total, 1e-30))
                band_rows.append(
                    {
                        "condition": condition,
                        "condition_label": condition_label(condition),
                        "component": component,
                        "component_label": COMPONENT_LABELS[component],
                        "spatial_band": band_id,
                        "spatial_band_label": band_label,
                        "spatial_low_cpd": float(low_cpd),
                        "spatial_high_cpd": float(high_cpd),
                        "total_nonzero_tf_power": total,
                        "tf_power_centroid_hz": centroid,
                        "fast_tf_fraction": float(np.nansum(band_power_by_tf[freq_hz >= 30.0]) / max(total, 1e-30)),
                    }
                )
    summary = pd.DataFrame(rows)
    bands = pd.DataFrame(band_rows)
    if not bands.empty:
        pivot = bands.pivot_table(
            index=["condition", "spatial_band"],
            columns="component",
            values="total_nonzero_tf_power",
            aggfunc="first",
        ).reset_index()
        pivot["across_over_along_total_power"] = pivot.get("across", np.nan) / np.maximum(pivot.get("along", np.nan), 1e-30)
        bands = bands.merge(
            pivot[["condition", "spatial_band", "across_over_along_total_power"]],
            on=["condition", "spatial_band"],
            how="left",
        )
    return summary, bands, sf_centers, freq_hz if "freq_hz" in locals() and freq_hz is not None else np.asarray([])


def matrix_from_summary(summary: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.asarray(sorted(summary["spatial_frequency_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(summary["temporal_frequency_hz"].unique()), dtype=np.float64)
    mat = np.full((tf.size, sf.size), np.nan, dtype=np.float64)
    sf_index = {float(v): i for i, v in enumerate(sf)}
    tf_index = {float(v): i for i, v in enumerate(tf)}
    for row in summary.itertuples(index=False):
        mat[tf_index[float(row.temporal_frequency_hz)], sf_index[float(row.spatial_frequency_cpd)]] = float(
            getattr(row, value_col)
        )
    return sf, tf, mat


def plot_axis_heatmaps(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    conditions: tuple[str, ...],
    tuning_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Path:
    db_values = db(summary["modulation_power_mean"].to_numpy(dtype=np.float64))
    finite = db_values[np.isfinite(db_values)]
    vmin = float(np.percentile(finite, 5.0))
    vmax = float(np.percentile(finite, 98.0))
    fig, axes = plt.subplots(2, len(conditions), figsize=(3.6 * len(conditions) + 1.0, 7.6), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.08, top=0.83, hspace=0.72, wspace=0.24)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    panel_i = 0
    for row_i, component in enumerate(("along", "across")):
        for col_i, condition in enumerate(conditions):
            ax = axes[row_i, col_i]
            sub = summary[summary["condition"].eq(condition) & summary["component"].eq(component)]
            sf, tf, mat = matrix_from_summary(sub, "modulation_power_mean")
            sf_edges = spatial_edges_from_centers(sf)
            tf_edges = temporal_edges(tf)
            mesh = ax.pcolormesh(
                sf_edges,
                tf_edges,
                db(mat),
                shading="auto",
                cmap="hot",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.5, 1, 3, 10], ["0.5", "1", "3", "10"])
            ax.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
            ax.set_title(
                f"{letters[panel_i]}. {condition_label(condition)}\n{COMPONENT_LABELS[component]}",
                loc="left",
                fontweight="bold",
                fontsize=8.5,
            )
            if row_i == 1:
                ax.set_xlabel("spatial frequency (cycles/deg)")
            if col_i == 0:
                ax.set_ylabel("temporal frequency (Hz)")
            overlay_tuning_contours(ax, tuning_surfaces, linewidth=1.05)
            if row_i == 0 and col_i == len(conditions) - 1:
                add_tuning_legend(ax, tuning_surfaces, loc="lower left")
            ax.set_xlim(float(sf_edges[0]), float(sf_edges[-1]))
            ax.set_ylim(float(tf_edges[0]), float(tf_edges[-1]))
            ax.spines[["top", "right"]].set_visible(False)
            panel_i += 1
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.012, shrink=0.84)
    cbar.set_label("10 log10 directional I(k)Q")
    fig.suptitle(
        "Contour-axis SF/TF power slices from SSI-v3 images and FEM traces",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    png = out_dir / "ssi_sftf_axis_slices.png"
    pdf = out_dir / "ssi_sftf_axis_slices.pdf"
    svg = out_dir / "ssi_sftf_axis_slices.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def ratio_summary(summary: pd.DataFrame) -> pd.DataFrame:
    along = summary[summary["component"].eq("along")].rename(
        columns={
            "motion_q_mean": "along_motion_q_mean",
            "modulation_power_mean": "along_modulation_power_mean",
            "source_power_mean": "along_source_power_mean",
        }
    )
    across = summary[summary["component"].eq("across")].rename(
        columns={
            "motion_q_mean": "across_motion_q_mean",
            "modulation_power_mean": "across_modulation_power_mean",
            "source_power_mean": "across_source_power_mean",
        }
    )
    merged = across.merge(
        along[
            [
                "condition",
                "spatial_frequency_cpd",
                "temporal_frequency_hz",
                "along_motion_q_mean",
                "along_modulation_power_mean",
                "along_source_power_mean",
            ]
        ],
        on=["condition", "spatial_frequency_cpd", "temporal_frequency_hz"],
        how="inner",
    )
    merged["across_minus_along_modulation_db"] = db(
        merged["across_modulation_power_mean"].to_numpy(dtype=np.float64)
    ) - db(merged["along_modulation_power_mean"].to_numpy(dtype=np.float64))
    merged["across_over_along_modulation"] = merged["across_modulation_power_mean"] / np.maximum(
        merged["along_modulation_power_mean"], 1e-30
    )
    merged["across_over_along_motion_q"] = merged["across_motion_q_mean"] / np.maximum(
        merged["along_motion_q_mean"], 1e-30
    )
    merged["across_over_along_source_power"] = merged["across_source_power_mean"] / np.maximum(
        merged["along_source_power_mean"], 1e-30
    )
    return merged


def plot_ratio_heatmaps(
    ratios: pd.DataFrame,
    out_dir: Path,
    *,
    conditions: tuple[str, ...],
    tuning_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Path:
    finite = ratios["across_minus_along_modulation_db"].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    vmax = float(np.percentile(np.abs(finite), 98.0))
    vmax = max(vmax, 1.0)
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.6 * len(conditions) + 1.0, 3.25), squeeze=False)
    mesh = None
    for col_i, condition in enumerate(conditions):
        ax = axes[0, col_i]
        sub = ratios[ratios["condition"].eq(condition)]
        sf, tf, mat = matrix_from_summary(sub, "across_minus_along_modulation_db")
        sf_edges = spatial_edges_from_centers(sf)
        tf_edges = temporal_edges(tf)
        mesh = ax.pcolormesh(
            sf_edges,
            tf_edges,
            mat,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.5, 1, 3, 10], ["0.5", "1", "3", "10"])
        ax.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
        ax.set_xlabel("spatial frequency (cycles/deg)")
        if col_i == 0:
            ax.set_ylabel("temporal frequency (Hz)")
        ax.set_title(f"{condition_label(condition)}", loc="left", fontweight="bold")
        overlay_tuning_contours(ax, tuning_surfaces, linewidth=1.05)
        ax.set_xlim(float(sf_edges[0]), float(sf_edges[-1]))
        ax.set_ylim(float(tf_edges[0]), float(tf_edges[-1]))
        ax.spines[["top", "right"]].set_visible(False)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.012, shrink=0.9)
        cbar.set_label("contour-normal minus parallel power (dB)")
    fig.suptitle(
        "Directional imbalance: contour-normal vs contour-parallel movie power",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    png = out_dir / "ssi_sftf_axis_ratio.png"
    pdf = out_dir / "ssi_sftf_axis_ratio.pdf"
    svg = out_dir / "ssi_sftf_axis_ratio.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def write_summary_readme(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    bands: pd.DataFrame,
    axis_png: Path,
    ratio_png: Path,
) -> None:
    lines = [
        "# SSI SF/TF Axis Slices",
        "",
        "Directional SF/TF diagnostic using the SSI-v3 image bank and fixation-trace bank.",
        "",
        f"- Axis-slice figure: `{_relative(axis_png)}`",
        f"- Directional-ratio figure: `{_relative(ratio_png)}`",
        f"- Image source: `{_relative(Path(args.image_table))}`",
        f"- Trace source: `{_relative(Path(args.trace_table))}`",
        f"- Tuning contour source: `{_relative(Path(args.tuning_points_csv))}`",
        f"- Directional image power uses +/-{float(args.wedge_half_width_deg):.1f} deg Fourier wedges around `image_edge_axis_deg` and its normal.",
        "- Trace projections use the same contour-component convention as `panel_g_alternative_x_axes_diagnostic.py`.",
        "",
        "## Band Summary",
        "",
    ]
    if not bands.empty:
        for condition in parse_conditions(args.conditions):
            rows = bands[bands["condition"].eq(condition)]
            if rows.empty:
                continue
            lines.append(f"### {condition_label(condition)}")
            for band_id in ("low_sf", "mid_sf", "high_sf"):
                sub = rows[rows["spatial_band"].eq(band_id)]
                if sub.empty:
                    continue
                along = sub[sub["component"].eq("along")]
                across = sub[sub["component"].eq("across")]
                if along.empty or across.empty:
                    continue
                ratio = float(across["across_over_along_total_power"].iloc[0])
                lines.append(
                    f"- {sub['spatial_band_label'].iloc[0]}: normal/parallel total power {ratio:.3g}x; "
                    f"normal centroid {float(across['tf_power_centroid_hz'].iloc[0]):.3g} Hz, "
                    f"parallel centroid {float(along['tf_power_centroid_hz'].iloc[0]):.3g} Hz."
                )
            lines.append("")
    lines.extend(
        [
            "## Working Interpretation",
            "",
            "The radial heatmap says how much SF/TF power the movement bank makes available in aggregate.  These slices ask whether that power lands on contour-parallel or contour-normal Fourier components.  The contour-normal slice is the one most directly tied to edge phase changes for contour-aligned high-SF units, so a normal-biased high-SF band is the spectral counterpart of the panel-G component-RMS split.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, bands, _sf_centers, _freq_hz = compute_axis_summary(args)
    summary_csv = out_dir / "ssi_sftf_axis_slices_summary.csv"
    band_csv = out_dir / "ssi_sftf_axis_band_summary.csv"
    ratio_csv = out_dir / "ssi_sftf_axis_ratio_summary.csv"
    summary.to_csv(summary_csv, index=False)
    bands.to_csv(band_csv, index=False)
    ratios = ratio_summary(summary)
    ratios.to_csv(ratio_csv, index=False)
    conditions = parse_conditions(args.conditions)
    tuning_surfaces = load_tuning_surfaces(Path(args.tuning_points_csv))
    axis_png = plot_axis_heatmaps(summary, out_dir, conditions=conditions, tuning_surfaces=tuning_surfaces)
    ratio_png = plot_ratio_heatmaps(ratios, out_dir, conditions=conditions, tuning_surfaces=tuning_surfaces)
    write_summary_readme(out_dir, args=args, bands=bands, axis_png=axis_png, ratio_png=ratio_png)
    return {
        "axis_png": axis_png,
        "ratio_png": ratio_png,
        "summary_csv": summary_csv,
        "band_csv": band_csv,
        "ratio_csv": ratio_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_IMAGE_TABLE)
    parser.add_argument("--trace-table", type=Path, default=DEFAULT_TRACE_TABLE)
    parser.add_argument("--trace-xy", type=Path, default=DEFAULT_TRACE_XY)
    parser.add_argument("--tuning-points-csv", type=Path, default=DEFAULT_TUNING_POINTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--max-traces-per-condition", type=int, default=192)
    parser.add_argument("--patch-size-px", type=int, default=PATCH_SIZE_PX)
    parser.add_argument("--n-sf-bins", type=int, default=17)
    parser.add_argument("--wedge-half-width-deg", type=float, default=22.5)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    print(f"Wrote {paths['axis_png']}")
    print(f"Wrote {paths['ratio_png']}")


if __name__ == "__main__":
    main()
