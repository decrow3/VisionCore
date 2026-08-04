#!/usr/bin/env python3
"""High-coherence contour SF/TF heatmaps using each window's full eye trace."""

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

from plot_eye_movement_joint_sftf_power import abs_temporal_frequency_bins, db, spatial_edges_from_centers, temporal_edges
from plot_eye_movement_power_spectrum_shift import (
    DEFAULT_IMAGE_TABLE,
    DT_S,
    PATCH_SIZE_PX,
    ROOT,
    clip_patch_subpixel,
    frequency_grid,
    patch_power2d,
)
from plot_ssi_sftf_mechanism_bridge import (
    DEFAULT_TUNING_POINTS_CSV,
    add_tuning_legend,
    load_tuning_surfaces,
    overlay_tuning_contours,
)

from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import _window_trace


PANEL_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
DEFAULT_OUT_DIR = PANEL_DIR / "high_coherence_full_trace_sftf_components"

MOTION_MODES = ("full_2d", "along_only", "across_only")
MODE_LABELS = {
    "full_2d": "full 2D motion",
    "along_only": "contour-parallel only",
    "across_only": "contour-normal only",
}
MODE_COLORS = {
    "full_2d": "#222222",
    "along_only": "#2B6CB0",
    "across_only": "#B24A3B",
}
SPATIAL_BANDS = (
    ("low_sf", "low SF", 0.3, 2.0),
    ("mid_sf", "mid SF", 2.0, 8.0),
    ("high_sf", "high SF", 8.0, 16.0),
)
EPS = 1e-30


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


def _axis_vectors(axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(float(axis_deg))
    along = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    across = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    return along, across


def spatial_edges(rr: np.ndarray, *, low_cpd: float, high_cpd: float, n_bins: int) -> np.ndarray:
    high = min(float(high_cpd), float(np.nanmax(rr)))
    low = max(float(low_cpd), 1e-4)
    if high <= low:
        raise ValueError(f"Invalid spatial-frequency range: {low:g}-{high:g} cpd")
    return np.geomspace(low, high, int(n_bins) + 1)


def geometric_centers(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    return np.sqrt(values[:-1] * values[1:])


def select_high_coherence_windows(image_table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = image_table.copy()
    frame["image_orientation_coherence"] = pd.to_numeric(frame["image_orientation_coherence"], errors="coerce")
    frame["image_patch_fraction_background"] = pd.to_numeric(
        frame.get("image_patch_fraction_background", np.nan), errors="coerce"
    )
    frame["image_patch_distance_to_image_border_px"] = pd.to_numeric(
        frame.get("image_patch_distance_to_image_border_px", np.nan), errors="coerce"
    )
    keep = frame["image_feature_ok"].astype(bool) if "image_feature_ok" in frame.columns else pd.Series(True, index=frame.index)
    keep &= frame["image_orientation_coherence"] >= float(args.coherence_min)
    keep &= frame["image_patch_fraction_background"].fillna(0.0) <= float(args.max_background_fraction)
    keep &= frame["image_patch_distance_to_image_border_px"].fillna(np.inf) >= int(args.patch_size_px) // 2
    if bool(args.require_contour_reliable) and "image_contour_reliable" in frame.columns:
        keep &= frame["image_contour_reliable"].astype(bool)
    if bool(args.require_contour_strong) and "image_contour_strong" in frame.columns:
        keep &= frame["image_contour_strong"].astype(bool)
    selected = frame[keep].copy().sort_values("image_orientation_coherence", ascending=False)
    if int(args.max_images) > 0:
        selected = selected.head(int(args.max_images)).copy()
    selected = selected.sort_values(["session", "trial_idx", "global_start"]).reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No high-coherence windows passed the selection filters.")
    return selected


def motion_q_single_trace(
    trace_xy_deg: np.ndarray,
    fx_cpd: np.ndarray,
    fy_cpd: np.ndarray,
    *,
    mode: str,
    contour_axis_deg: float,
    dt_s: float,
    coeff_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal phase spectrum Q(k,ft) for one full fixation-window trace."""
    trace = np.asarray(trace_xy_deg, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[0] < 2 or trace.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T,2); got {trace.shape}")
    trace = trace - np.nanmean(trace, axis=0, keepdims=True)
    flat_fx = np.asarray(fx_cpd, dtype=np.float32).ravel()
    flat_fy = np.asarray(fy_cpd, dtype=np.float32).ravel()
    freq_hz, freq_bins = abs_temporal_frequency_bins(trace.shape[0], float(dt_s))
    q = np.zeros((flat_fx.size, freq_hz.size), dtype=np.float64)

    if mode == "full_2d":
        screen_x = -trace[:, 0]
        screen_y = trace[:, 1]
        scalar_disp = None
        flat_keff = None
    elif mode in {"along_only", "across_only"}:
        along, across = _axis_vectors(float(contour_axis_deg))
        u = along if mode == "along_only" else across
        scalar_disp = trace @ u.astype(np.float32)
        flat_keff = (-float(u[0])) * flat_fx + float(u[1]) * flat_fy
        screen_x = None
        screen_y = None
    else:
        raise ValueError(f"Unknown motion mode {mode!r}")

    for start in range(0, flat_fx.size, int(coeff_chunk_size)):
        stop = min(start + int(coeff_chunk_size), flat_fx.size)
        if mode == "full_2d":
            assert screen_x is not None and screen_y is not None
            phase_arg = screen_x[:, None] * flat_fx[None, start:stop] + screen_y[:, None] * flat_fy[None, start:stop]
        else:
            assert scalar_disp is not None and flat_keff is not None
            phase_arg = scalar_disp[:, None] * flat_keff[None, start:stop]
        phase = np.exp((-2j * np.pi) * phase_arg)
        phase -= np.mean(phase, axis=0, keepdims=True)
        spec = np.fft.fft(phase, axis=0, norm="ortho")
        power = np.abs(spec) ** 2
        for freq_i, bins in enumerate(freq_bins):
            q[start:stop, freq_i] = np.sum(power[bins, :], axis=0)
    return freq_hz, q.astype(np.float32)


def bin_modulation(
    source_power2d: np.ndarray,
    q_flat: np.ndarray,
    rr: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_source = np.asarray(source_power2d, dtype=np.float64).ravel()
    q = np.asarray(q_flat, dtype=np.float64)
    radius = np.asarray(rr, dtype=np.float64).ravel()
    n_sf = int(edges.size - 1)
    n_tf = int(q.shape[1])
    source = np.full(n_sf, np.nan, dtype=np.float64)
    q_mean = np.full((n_sf, n_tf), np.nan, dtype=np.float64)
    modulation = np.full((n_sf, n_tf), np.nan, dtype=np.float64)
    counts = np.zeros(n_sf, dtype=np.int64)
    for sf_i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (radius >= float(lo)) & (radius < float(hi))
        counts[sf_i] = int(np.sum(mask))
        if not np.any(mask):
            continue
        source[sf_i] = float(np.nanmean(flat_source[mask]))
        q_mean[sf_i, :] = np.nanmean(q[mask, :], axis=0)
        modulation[sf_i, :] = np.nanmean(flat_source[mask, None] * q[mask, :], axis=0)
    return source, q_mean, modulation, counts


def compute_summary(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    image_table = pd.read_csv(args.image_table)
    windows = select_high_coherence_windows(image_table, args)
    rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    freq_hz: np.ndarray | None = None
    sf_centers: np.ndarray | None = None
    edges: np.ndarray | None = None
    rr: np.ndarray | None = None

    for image_pos, row in windows.iterrows():
        key = (str(row["session"]), int(row["trial_idx"]))
        print(
            f"[window] {image_pos + 1}/{len(windows)} {key[0]} trial {key[1]} "
            f"coh={float(row['image_orientation_coherence']):.3f}",
            flush=True,
        )
        if key not in canvas_cache:
            canvas_cache[key] = _backimage_canvas(key[0], key[1])
        canvas, ppd, _screen_shape = canvas_cache[key]
        if rr is None:
            fx, fy, rr_i = frequency_grid(int(args.patch_size_px), float(ppd))
            rr = rr_i
            edges = spatial_edges(
                rr,
                low_cpd=float(args.low_cpd),
                high_cpd=float(args.high_cpd),
                n_bins=int(args.n_sf_bins),
            )
            sf_centers = geometric_centers(edges)
        else:
            fx, fy, _rr_i = frequency_grid(int(args.patch_size_px), float(ppd))
        center = (float(row["image_patch_center_x_px"]), float(row["image_patch_center_y_px"]))
        patch = clip_patch_subpixel(canvas, center, int(args.patch_size_px))
        source_power = patch_power2d(patch)
        trace = _window_trace(row)
        trace = np.asarray(trace, dtype=np.float32)
        if int(args.trace_samples) > 0 and trace.shape[0] != int(args.trace_samples):
            print(f"[skip] expected {int(args.trace_samples)} samples, got {trace.shape[0]}", flush=True)
            continue
        axis_deg = float(row["image_edge_axis_deg"])
        if not np.isfinite(axis_deg):
            continue
        assert edges is not None and sf_centers is not None and rr is not None
        for mode in MOTION_MODES:
            freq_i, q_flat = motion_q_single_trace(
                trace,
                fx,
                fy,
                mode=mode,
                contour_axis_deg=axis_deg,
                dt_s=float(args.dt_s),
                coeff_chunk_size=int(args.coeff_chunk_size),
            )
            if freq_hz is None:
                freq_hz = freq_i
            elif not np.allclose(freq_hz, freq_i):
                raise RuntimeError("Temporal frequency support changed across windows.")
            source_radial, q_mean, modulation, counts = bin_modulation(source_power, q_flat, rr, edges)
            for sf_i, sf in enumerate(sf_centers):
                for tf_i, tf in enumerate(freq_i):
                    rows.append(
                        {
                            "image_pos": int(image_pos),
                            "image_index": int(row["image_index"]),
                            "session": str(row["session"]),
                            "trial_idx": int(row["trial_idx"]),
                            "global_start": int(row["global_start"]),
                            "global_stop": int(row["global_stop"]),
                            "trace_n_samples": int(trace.shape[0]),
                            "image_orientation_coherence": float(row["image_orientation_coherence"]),
                            "image_edge_axis_deg": axis_deg,
                            "motion_mode": mode,
                            "motion_mode_label": MODE_LABELS[mode],
                            "spatial_frequency_cpd": float(sf),
                            "temporal_frequency_hz": float(tf),
                            "source_power_mean": float(source_radial[sf_i]),
                            "motion_q_mean": float(q_mean[sf_i, tf_i]),
                            "modulation_power_mean": float(modulation[sf_i, tf_i]),
                            "n_spatial_coefficients": int(counts[sf_i]),
                        }
                    )
        centered = trace - np.nanmean(trace, axis=0, keepdims=True)
        along, across = _axis_vectors(axis_deg)
        along_pos = centered @ along
        across_pos = centered @ across
        image_rows.append(
            {
                "image_pos": int(image_pos),
                "image_index": int(row["image_index"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "global_start": int(row["global_start"]),
                "global_stop": int(row["global_stop"]),
                "trace_n_samples": int(trace.shape[0]),
                "trace_duration_s": float(trace.shape[0] * float(args.dt_s)),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "image_edge_axis_deg": axis_deg,
                "image_patch_center_x_px": float(row["image_patch_center_x_px"]),
                "image_patch_center_y_px": float(row["image_patch_center_y_px"]),
                "image_patch_size_px": int(args.patch_size_px),
                "ppd": float(ppd),
                "rms_full_arcmin": float(60.0 * np.sqrt(np.nanmean(np.sum(centered * centered, axis=1)))),
                "rms_along_arcmin": float(60.0 * np.sqrt(np.nanmean(along_pos * along_pos))),
                "rms_across_arcmin": float(60.0 * np.sqrt(np.nanmean(across_pos * across_pos))),
                "path_full_arcmin": float(60.0 * np.nansum(np.linalg.norm(np.diff(trace, axis=0), axis=1))),
                "path_along_arcmin": float(60.0 * np.nansum(np.abs(np.diff(along_pos)))),
                "path_across_arcmin": float(60.0 * np.nansum(np.abs(np.diff(across_pos)))),
            }
        )

    detail = pd.DataFrame(rows)
    images = pd.DataFrame(image_rows)
    if detail.empty or freq_hz is None or sf_centers is None:
        raise RuntimeError("No full-trace spectra were computed.")
    summary = (
        detail.groupby(
            ["motion_mode", "motion_mode_label", "spatial_frequency_cpd", "temporal_frequency_hz"],
            sort=False,
        )
        .agg(
            source_power_mean=("source_power_mean", "mean"),
            motion_q_mean=("motion_q_mean", "mean"),
            modulation_power_mean=("modulation_power_mean", "mean"),
            n_images=("image_index", "nunique"),
            n_spatial_coefficients=("n_spatial_coefficients", "mean"),
        )
        .reset_index()
    )
    return detail, summary, images, sf_centers, freq_hz


def matrix_from_summary(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def mode_matrix(summary: pd.DataFrame, mode: str, column: str = "modulation_power_mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return matrix_from_summary(summary[summary["motion_mode"].eq(mode)], column)


def setup_heat_axis(ax: plt.Axes, sf: np.ndarray, tf: np.ndarray) -> None:
    sf_edges = spatial_edges_from_centers(sf)
    tf_edges = temporal_edges(tf)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(sf_edges[0]), float(sf_edges[-1]))
    ax.set_ylim(max(float(tf_edges[0]), 0.75), float(tf_edges[-1]))
    ax.set_xticks([0.2, 0.5, 1, 2, 4, 8, 16], ["0.2", "0.5", "1", "2", "4", "8", "16"])
    ax.set_yticks([1, 3, 10, 30, 60], ["1", "3", "10", "30", "60"])
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")
    ax.spines[["top", "right"]].set_visible(False)


def plot_main_heatmaps(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    tuning_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Path:
    heat_values = db(summary["modulation_power_mean"].to_numpy(dtype=np.float64))
    finite = heat_values[np.isfinite(heat_values)]
    vmin = float(np.percentile(finite, 5.0))
    vmax = float(np.percentile(finite, 98.0))
    _sf, _tf, along = mode_matrix(summary, "along_only")
    sf, tf, across = mode_matrix(summary, "across_only")
    diff = db(across) - db(along)
    diff_vals = diff[np.isfinite(diff)]
    diff_vmax = max(float(np.percentile(np.abs(diff_vals), 98.0)), 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.55), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.875, bottom=0.18, top=0.73, wspace=0.28)
    heat_mesh = None
    for ax, mode in zip(axes[:3], MOTION_MODES, strict=True):
        sf, tf, mat = mode_matrix(summary, mode)
        heat_mesh = ax.pcolormesh(
            spatial_edges_from_centers(sf),
            temporal_edges(tf),
            db(mat),
            shading="auto",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        )
        setup_heat_axis(ax, sf, tf)
        ax.set_title(MODE_LABELS[mode], loc="left", fontweight="bold", color=MODE_COLORS[mode])
        overlay_tuning_contours(ax, tuning_surfaces, linewidth=0.95)
    add_tuning_legend(axes[0], tuning_surfaces, loc="lower left")

    ratio_mesh = axes[3].pcolormesh(
        spatial_edges_from_centers(sf),
        temporal_edges(tf),
        diff,
        shading="auto",
        cmap="RdBu_r",
        vmin=-diff_vmax,
        vmax=diff_vmax,
    )
    setup_heat_axis(axes[3], sf, tf)
    axes[3].set_title("normal - parallel", loc="left", fontweight="bold")
    cbar = fig.colorbar(heat_mesh, ax=axes[:3].tolist(), pad=0.012, shrink=0.9)
    cbar.set_label("10 log10 image-weighted motion power")
    cbar2 = fig.colorbar(ratio_mesh, ax=[axes[3]], pad=0.028, shrink=0.9)
    cbar2.set_label("normal minus parallel (dB)")
    fig.suptitle(
        "High-coherence contour windows: full-resolution trace SF/TF power",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    png = out_dir / "high_coherence_full_trace_sftf_heatmaps.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_sftf_heatmaps.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_sftf_heatmaps.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def relative_frame(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary[summary["motion_mode"].eq("full_2d")].rename(
        columns={"modulation_power_mean": "full_modulation_power_mean"}
    )
    rows: list[pd.DataFrame] = []
    for mode in ("along_only", "across_only"):
        sub = summary[summary["motion_mode"].eq(mode)].merge(
            full[["spatial_frequency_cpd", "temporal_frequency_hz", "full_modulation_power_mean"]],
            on=["spatial_frequency_cpd", "temporal_frequency_hz"],
            how="inner",
        )
        sub["mode_minus_full_db"] = db(sub["modulation_power_mean"].to_numpy(dtype=np.float64)) - db(
            sub["full_modulation_power_mean"].to_numpy(dtype=np.float64)
        )
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    return out


def plot_relative_heatmaps(relative: pd.DataFrame, out_dir: Path) -> Path:
    vals = relative["mode_minus_full_db"].to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vmax = max(float(np.percentile(np.abs(vals), 98.0)), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.87, bottom=0.18, top=0.76, wspace=0.28)
    mesh = None
    for ax, mode in zip(axes, ("along_only", "across_only"), strict=True):
        sub = relative[relative["motion_mode"].eq(mode)]
        sf, tf, mat = matrix_from_summary(sub, "mode_minus_full_db")
        mesh = ax.pcolormesh(
            spatial_edges_from_centers(sf),
            temporal_edges(tf),
            mat,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        setup_heat_axis(ax, sf, tf)
        ax.set_title(f"{MODE_LABELS[mode]} - full", loc="left", fontweight="bold", color=MODE_COLORS[mode])
    cbar = fig.colorbar(mesh, ax=axes.tolist(), pad=0.016, shrink=0.9)
    cbar.set_label("component minus full (dB)")
    fig.suptitle(
        "How much of full-motion SF/TF power is captured by one component?",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    png = out_dir / "high_coherence_full_trace_component_minus_full.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_component_minus_full.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_component_minus_full.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def band_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, sub in summary.groupby("motion_mode", sort=False):
        for band_id, band_label, low_cpd, high_cpd in SPATIAL_BANDS:
            band = sub[(sub["spatial_frequency_cpd"] >= low_cpd) & (sub["spatial_frequency_cpd"] < high_cpd)]
            if band.empty:
                continue
            by_tf = band.groupby("temporal_frequency_hz", sort=True)["modulation_power_mean"].mean()
            freq = by_tf.index.to_numpy(dtype=np.float64)
            power = by_tf.to_numpy(dtype=np.float64)
            total = float(np.nansum(power))
            rows.append(
                {
                    "motion_mode": mode,
                    "motion_mode_label": MODE_LABELS[str(mode)],
                    "spatial_band": band_id,
                    "spatial_band_label": band_label,
                    "spatial_low_cpd": float(low_cpd),
                    "spatial_high_cpd": float(high_cpd),
                    "total_nonzero_tf_power": total,
                    "tf_power_centroid_hz": float(np.nansum(freq * power) / max(total, EPS)),
                    "fast_tf_fraction": float(np.nansum(power[freq >= 30.0]) / max(total, EPS)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    full = out[out["motion_mode"].eq("full_2d")][["spatial_band", "total_nonzero_tf_power"]].rename(
        columns={"total_nonzero_tf_power": "full_total_nonzero_tf_power"}
    )
    out = out.merge(full, on="spatial_band", how="left")
    out["total_power_over_full_same_band"] = out["total_nonzero_tf_power"] / np.maximum(
        out["full_total_nonzero_tf_power"], EPS
    )
    return out


def plot_band_summary(bands: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.78, wspace=0.34)
    x = np.arange(len(SPATIAL_BANDS))
    labels = [label for _bid, label, _lo, _hi in SPATIAL_BANDS]
    width = 0.23
    for mode_i, mode in enumerate(MOTION_MODES):
        sub = bands[bands["motion_mode"].eq(mode)].set_index("spatial_band")
        vals = [float(sub.loc[band_id, "total_power_over_full_same_band"]) for band_id, _label, _lo, _hi in SPATIAL_BANDS]
        axes[0].bar(x + (mode_i - 1) * width, vals, width=width, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        cent = [float(sub.loc[band_id, "tf_power_centroid_hz"]) for band_id, _label, _lo, _hi in SPATIAL_BANDS]
        fast = [float(sub.loc[band_id, "fast_tf_fraction"]) for band_id, _label, _lo, _hi in SPATIAL_BANDS]
        axes[1].plot(x, cent, marker="o", linewidth=1.8, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        axes[2].plot(x, fast, marker="o", linewidth=1.8, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
    axes[0].axhline(1.0, color="0.55", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("total power / full")
    axes[1].set_ylabel("TF centroid (Hz)")
    axes[2].set_ylabel("fraction >= 30 Hz")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Band summaries for high-coherence full-trace heatmaps",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    png = out_dir / "high_coherence_full_trace_band_summary.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_band_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_band_summary.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def plot_slices(summary: pd.DataFrame, out_dir: Path) -> Path:
    sf, tf, full_mat = mode_matrix(summary, "full_2d")
    _sf, _tf, along_mat = mode_matrix(summary, "along_only")
    _sf, _tf, across_mat = mode_matrix(summary, "across_only")
    mats = {"full_2d": full_mat, "along_only": along_mat, "across_only": across_mat}
    sf_targets = (0.5, 1.0, 2.0, 4.0, 8.0)
    tf_targets = (3.0, 8.0, 16.0, 32.0)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.86, hspace=0.38)
    for target in sf_targets:
        sf_i = int(np.argmin(np.abs(sf - float(target))))
        for mode, mat in mats.items():
            axes[0].plot(
                tf,
                db(mat[:, sf_i]),
                color=MODE_COLORS[mode],
                linewidth=1.2,
                alpha=0.95 if mode == "full_2d" else 0.75,
                linestyle="-" if mode == "full_2d" else ("--" if mode == "along_only" else ":"),
            )
        axes[0].text(float(tf[-1]) * 1.03, db(full_mat[-1, sf_i]), f"{sf[sf_i]:.2g} cpd", fontsize=6.5, va="center")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("temporal frequency (Hz)")
    axes[0].set_ylabel("spectral density (dB)")
    axes[0].set_title("A. Temporal slices at selected spatial frequencies", loc="left", fontweight="bold")
    axes[0].set_xlim(float(tf[0]), float(tf[-1]) * 1.45)

    for target in tf_targets:
        tf_i = int(np.argmin(np.abs(tf - float(target))))
        for mode, mat in mats.items():
            axes[1].plot(
                sf,
                db(mat[tf_i, :]),
                color=MODE_COLORS[mode],
                linewidth=1.2,
                alpha=0.95 if mode == "full_2d" else 0.75,
                linestyle="-" if mode == "full_2d" else ("--" if mode == "along_only" else ":"),
            )
        axes[1].text(float(sf[-1]) * 1.03, db(full_mat[tf_i, -1]), f"{tf[tf_i]:.0f} Hz", fontsize=6.5, va="center")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("spatial frequency (cycles/deg)")
    axes[1].set_ylabel("spectral density (dB)")
    axes[1].set_title("B. Spatial slices at selected temporal frequencies", loc="left", fontweight="bold")
    axes[1].set_xlim(float(sf[0]), float(sf[-1]) * 1.45)
    handles = [
        plt.Line2D([0], [0], color=MODE_COLORS[mode], linestyle="-" if mode == "full_2d" else ("--" if mode == "along_only" else ":"), label=MODE_LABELS[mode])
        for mode in MOTION_MODES
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Full-trace SF/TF slices on high-coherence contour windows",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    png = out_dir / "high_coherence_full_trace_sftf_slices.png"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_sftf_slices.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "high_coherence_full_trace_sftf_slices.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def write_readme(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    images: pd.DataFrame,
    bands: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    lines = [
        "# High-Coherence Full-Trace SF/TF Components",
        "",
        "This diagnostic uses high-coherence BackImage contour windows and each selected window's own full fixation trace.",
        "",
        f"- Image table: `{_relative(Path(args.image_table))}`",
        f"- Tuning contours: `{_relative(Path(args.tuning_points_csv))}`",
        f"- Coherence minimum: `{float(args.coherence_min):.3g}`",
        f"- Patch size: `{int(args.patch_size_px)}` px",
        f"- Spatial bins: `{int(args.n_sf_bins)}` from `{float(args.low_cpd):.3g}` to `{float(args.high_cpd):.3g}` cpd",
        f"- Trace samples required: `{int(args.trace_samples)}`",
        "",
        "## Figures",
        "",
    ]
    for key, path in paths.items():
        lines.append(f"- {key}: `{_relative(path)}`")
    lines.extend(
        [
            "",
            "## Selected Windows",
            "",
            f"- n windows: `{int(images.shape[0])}`",
            f"- median orientation coherence: `{float(images['image_orientation_coherence'].median()):.3g}`",
            f"- median full RMS: `{float(images['rms_full_arcmin'].median()):.3g}` arcmin",
            f"- median contour-parallel RMS: `{float(images['rms_along_arcmin'].median()):.3g}` arcmin",
            f"- median contour-normal RMS: `{float(images['rms_across_arcmin'].median()):.3g}` arcmin",
            "",
            "## Band Summary",
            "",
        ]
    )
    for band_id, band_label, _lo, _hi in SPATIAL_BANDS:
        rows = bands[bands["spatial_band"].eq(band_id)].set_index("motion_mode")
        if rows.empty:
            continue
        lines.append(f"### {band_label}")
        for mode in MOTION_MODES:
            if mode not in rows.index:
                continue
            row = rows.loc[mode]
            lines.append(
                f"- {MODE_LABELS[mode]}: total/full `{float(row['total_power_over_full_same_band']):.3g}`, "
                f"TF centroid `{float(row['tf_power_centroid_hz']):.3g}` Hz, "
                f">=30 Hz fraction `{float(row['fast_tf_fraction']):.3g}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "The full panel uses the complete 2D image spectrum and the full 2D trace.  The component panels keep the same image spectrum, but reconstruct retinal motion using only the contour-parallel or contour-normal projection of the same trace.  This is therefore a direct projected-motion control, not merely a rotated/wedged image-spectrum summary.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail, summary, images, _sf, _tf = compute_summary(args)
    detail_csv = out_dir / "high_coherence_full_trace_sftf_detail.csv"
    summary_csv = out_dir / "high_coherence_full_trace_sftf_summary.csv"
    image_csv = out_dir / "high_coherence_full_trace_image_sample.csv"
    band_csv = out_dir / "high_coherence_full_trace_band_summary.csv"
    relative_csv = out_dir / "high_coherence_full_trace_component_minus_full.csv"
    detail.to_csv(detail_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    images.to_csv(image_csv, index=False)
    bands = band_summary(summary)
    bands.to_csv(band_csv, index=False)
    relative = relative_frame(summary)
    relative.to_csv(relative_csv, index=False)
    tuning_surfaces = load_tuning_surfaces(Path(args.tuning_points_csv))
    paths = {
        "main_heatmaps": plot_main_heatmaps(summary, out_dir, tuning_surfaces=tuning_surfaces),
        "component_minus_full": plot_relative_heatmaps(relative, out_dir),
        "band_summary": plot_band_summary(bands, out_dir),
        "slices": plot_slices(summary, out_dir),
    }
    write_readme(out_dir, args=args, images=images, bands=bands, paths=paths)
    paths.update(
        {
            "detail_csv": detail_csv,
            "summary_csv": summary_csv,
            "image_csv": image_csv,
            "band_csv": band_csv,
            "relative_csv": relative_csv,
        }
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_IMAGE_TABLE)
    parser.add_argument("--tuning-points-csv", type=Path, default=DEFAULT_TUNING_POINTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--coherence-min", type=float, default=0.5)
    parser.add_argument("--max-background-fraction", type=float, default=0.05)
    parser.add_argument("--require-contour-reliable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-contour-strong", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--patch-size-px", type=int, default=257)
    parser.add_argument("--trace-samples", type=int, default=128)
    parser.add_argument("--low-cpd", type=float, default=0.25)
    parser.add_argument("--high-cpd", type=float, default=18.0)
    parser.add_argument("--n-sf-bins", type=int, default=24)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--coeff-chunk-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
