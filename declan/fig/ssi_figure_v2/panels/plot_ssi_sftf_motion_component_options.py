#!/usr/bin/env python3
"""Exploratory SF/TF plots for contour-relative FEM motion components.

This script is deliberately a sandbox.  It generates several candidate views
for separating the image-frequency coordinate system from the motion component:

* fast contour-coordinate factorization, split by RMS-excursion trace bins;
* contour-normal minus contour-parallel ratios;
* a smaller direct coefficient-level control where the full image spectrum is
  driven by full, contour-parallel-only, or contour-normal-only trajectories.
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
    sample_indices,
)
from plot_ssi_sftf_axis_slices import (
    directional_image_power,
    geometric_centers,
    projected_q_spectrum,
)
from plot_ssi_sftf_mechanism_bridge import (
    DEFAULT_TUNING_POINTS_CSV,
    add_tuning_legend,
    load_tuning_surfaces,
    overlay_tuning_contours,
)


PANEL_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
DEFAULT_OUT_DIR = PANEL_DIR / "ssi_sftf_motion_component_options"

TRACE_GROUP_ORDER = ("drift_rms_low", "drift_rms_mid", "drift_rms_high", "microsaccade", "all_real_fem")
PLOT_TRACE_GROUPS = ("drift_rms_low", "drift_rms_mid", "drift_rms_high", "microsaccade")
TRACE_GROUP_LABELS = {
    "drift_rms_low": "low RMS drift",
    "drift_rms_mid": "mid RMS drift",
    "drift_rms_high": "high RMS drift",
    "microsaccade": "microsaccade",
    "all_real_fem": "all real FEM",
}
TRACE_GROUP_COLORS = {
    "drift_rms_low": "#276fbf",
    "drift_rms_mid": "#559f76",
    "drift_rms_high": "#c36d1d",
    "microsaccade": "#b83b5e",
    "all_real_fem": "#222222",
}
COMPONENT_LABELS = {
    "along": "contour-parallel",
    "across": "contour-normal",
}
MOTION_MODE_LABELS = {
    "full_2d": "full 2D motion",
    "along_only": "parallel-only motion",
    "across_only": "normal-only motion",
}
AXIS_METHOD_LABELS = {
    "projection": "projected |k component|",
    "wedge": "oriented Fourier wedge",
}
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


def _component_vector(axis_deg: float, component: str) -> np.ndarray:
    along, across = _axis_vectors(float(axis_deg))
    if component == "along":
        return along
    if component == "across":
        return across
    raise ValueError(f"unknown component {component!r}")


def projected_image_power(
    power2d: np.ndarray,
    fx_cpd: np.ndarray,
    fy_cpd: np.ndarray,
    edges: np.ndarray,
    *,
    contour_axis_deg: float,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin static image power by |k projected onto the contour component|."""
    u = _component_vector(float(contour_axis_deg), component)
    k_component = np.abs(np.asarray(fx_cpd, dtype=np.float64) * u[0] + np.asarray(fy_cpd, dtype=np.float64) * u[1])
    arr = np.asarray(power2d, dtype=np.float64)
    values: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (k_component >= float(lo)) & (k_component < float(hi))
        counts.append(int(np.sum(mask)))
        values.append(float(np.nanmean(arr[mask])) if np.any(mask) else float("nan"))
    return np.asarray(values, dtype=np.float64), np.asarray(counts, dtype=np.int64)


def select_rms_trace_groups(
    trace_table: pd.DataFrame,
    *,
    max_traces_per_condition: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    table = trace_table.copy()
    table["trace_bank_index"] = pd.to_numeric(table["trace_bank_index"], errors="coerce").astype(int)
    table["rms_arcmin"] = pd.to_numeric(table["rendered_rms_radius_arcmin"], errors="coerce")
    table["path_arcmin"] = pd.to_numeric(table["rendered_path_length_arcmin"], errors="coerce")
    table["n_ms"] = pd.to_numeric(table["rendered_n_microsaccade_events"], errors="coerce").fillna(0).astype(int)
    valid = table[np.isfinite(table["rms_arcmin"])].copy()
    drift = valid[valid["n_ms"].eq(0)].copy()
    with_ms = valid[valid["n_ms"].gt(0)].copy()
    q1 = float(drift["rms_arcmin"].quantile(1.0 / 3.0))
    q2 = float(drift["rms_arcmin"].quantile(2.0 / 3.0))
    frames = {
        "drift_rms_low": drift[drift["rms_arcmin"] <= q1],
        "drift_rms_mid": drift[(drift["rms_arcmin"] > q1) & (drift["rms_arcmin"] <= q2)],
        "drift_rms_high": drift[drift["rms_arcmin"] > q2],
        "microsaccade": with_ms,
        "all_real_fem": valid,
    }
    groups: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for offset, condition in enumerate(TRACE_GROUP_ORDER):
        frame = frames[condition]
        available = frame["trace_bank_index"].to_numpy(dtype=np.int64)
        selected = sample_indices(available, max_n=int(max_traces_per_condition), seed=int(seed) + 2003 * offset)
        groups[condition] = selected
        chosen = table[table["trace_bank_index"].isin(selected)]
        rows.append(
            {
                "condition": condition,
                "condition_label": TRACE_GROUP_LABELS[condition],
                "n_available_traces": int(available.size),
                "n_selected_traces": int(selected.size),
                "rms_q25_arcmin": float(chosen["rms_arcmin"].quantile(0.25)) if not chosen.empty else float("nan"),
                "rms_median_arcmin": float(chosen["rms_arcmin"].median()) if not chosen.empty else float("nan"),
                "rms_q75_arcmin": float(chosen["rms_arcmin"].quantile(0.75)) if not chosen.empty else float("nan"),
                "path_median_arcmin": float(chosen["path_arcmin"].median()) if not chosen.empty else float("nan"),
                "microsaccade_fraction": float(np.mean(chosen["n_ms"].to_numpy(dtype=float) > 0.0))
                if not chosen.empty
                else float("nan"),
            }
        )
    return groups, pd.DataFrame(rows)


def _selected_traces(traces: np.ndarray, indices: np.ndarray, *, max_n: int, seed: int) -> np.ndarray:
    sampled = sample_indices(np.asarray(indices, dtype=np.int64), max_n=int(max_n), seed=int(seed))
    return np.asarray(traces[sampled], dtype=np.float32)


def compute_factorized_summary(
    *,
    source_powers: np.ndarray,
    image_rows: list[dict[str, Any]],
    fx: np.ndarray,
    fy: np.ndarray,
    rr: np.ndarray,
    traces: np.ndarray,
    trace_groups: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = radial_edges(rr, int(args.n_sf_bins))
    sf_centers = geometric_centers(edges)
    rows: list[dict[str, Any]] = []
    for axis_method in ("projection", "wedge"):
        for condition in TRACE_GROUP_ORDER:
            selected = _selected_traces(
                traces,
                trace_groups[condition],
                max_n=int(args.max_traces_per_condition),
                seed=int(args.seed) + 5011 + TRACE_GROUP_ORDER.index(condition),
            )
            print(f"[factorized] {axis_method} {condition}: {selected.shape[0]} traces x {len(image_rows)} images", flush=True)
            for component in ("along", "across"):
                acc_source = np.zeros(sf_centers.size, dtype=np.float64)
                acc_q: np.ndarray | None = None
                acc_mod: np.ndarray | None = None
                acc_counts = np.zeros(sf_centers.size, dtype=np.float64)
                n_image = 0
                freq_hz: np.ndarray | None = None
                for power2d, image_row in zip(source_powers, image_rows, strict=True):
                    axis_deg = float(image_row["image_edge_axis_deg"])
                    if not np.isfinite(axis_deg):
                        continue
                    if axis_method == "projection":
                        source_axis, counts = projected_image_power(
                            power2d,
                            fx,
                            fy,
                            edges,
                            contour_axis_deg=axis_deg,
                            component=component,
                        )
                    elif axis_method == "wedge":
                        source_axis, counts = directional_image_power(
                            power2d,
                            fx,
                            fy,
                            rr,
                            edges,
                            contour_axis_deg=axis_deg,
                            component=component,
                            wedge_half_width_deg=float(args.wedge_half_width_deg),
                        )
                    else:  # pragma: no cover - guarded by fixed loop.
                        raise ValueError(axis_method)
                    freq_i, q_axis = projected_q_spectrum(
                        selected,
                        sf_centers,
                        contour_axis_deg=axis_deg,
                        component=component,
                        dt_s=float(args.dt_s),
                        chunk_size=int(args.trace_chunk_size),
                    )
                    if freq_hz is None:
                        freq_hz = freq_i
                        acc_q = np.zeros((sf_centers.size, freq_hz.size), dtype=np.float64)
                        acc_mod = np.zeros((sf_centers.size, freq_hz.size), dtype=np.float64)
                    elif not np.allclose(freq_hz, freq_i):
                        raise RuntimeError("Temporal frequency support changed.")
                    assert acc_q is not None and acc_mod is not None
                    acc_source += np.nan_to_num(source_axis, nan=0.0)
                    acc_q += np.asarray(q_axis, dtype=np.float64)
                    acc_mod += np.nan_to_num(source_axis[:, None], nan=0.0) * np.asarray(q_axis, dtype=np.float64)
                    acc_counts += np.asarray(counts, dtype=np.float64)
                    n_image += 1
                if freq_hz is None or acc_q is None or acc_mod is None or n_image == 0:
                    continue
                source_mean = acc_source / float(n_image)
                q_mean = acc_q / float(n_image)
                mod_mean = acc_mod / float(n_image)
                count_mean = acc_counts / float(n_image)
                for sf_i, sf in enumerate(sf_centers):
                    for tf_i, tf in enumerate(freq_hz):
                        rows.append(
                            {
                                "calculation": "factorized_axis",
                                "axis_method": axis_method,
                                "axis_method_label": AXIS_METHOD_LABELS[axis_method],
                                "condition": condition,
                                "condition_label": TRACE_GROUP_LABELS[condition],
                                "component": component,
                                "component_label": COMPONENT_LABELS[component],
                                "spatial_frequency_cpd": float(sf),
                                "temporal_frequency_hz": float(tf),
                                "source_power_mean": float(source_mean[sf_i]),
                                "motion_q_mean": float(q_mean[sf_i, tf_i]),
                                "modulation_power_mean": float(mod_mean[sf_i, tf_i]),
                                "mean_coefficients_per_image": float(count_mean[sf_i]),
                                "n_images": int(n_image),
                                "n_traces": int(selected.shape[0]),
                            }
                        )
    summary = pd.DataFrame(rows)
    bands = band_summary(summary, group_cols=("calculation", "axis_method", "condition", "component"))
    return summary, bands


def motion_q_spectrum_projected_mode(
    traces_xy_deg: np.ndarray,
    fx_cpd: np.ndarray,
    fy_cpd: np.ndarray,
    *,
    mode: str,
    contour_axis_deg: float | None,
    dt_s: float,
    chunk_size: int,
    coeff_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficient-level temporal Q for full or projected-only motion."""
    traces_arr = np.asarray(traces_xy_deg, dtype=np.float32)
    if traces_arr.ndim != 3 or traces_arr.shape[1] < 2 or traces_arr.shape[2] != 2:
        raise ValueError(f"Expected traces with shape (n,T,2); got {traces_arr.shape}")
    flat_fx = np.asarray(fx_cpd, dtype=np.float32).ravel()
    flat_fy = np.asarray(fy_cpd, dtype=np.float32).ravel()
    freq_hz, freq_bins = abs_temporal_frequency_bins(traces_arr.shape[1], float(dt_s))
    q = np.zeros((flat_fx.size, freq_hz.size), dtype=np.float64)
    if mode in {"along_only", "across_only"}:
        if contour_axis_deg is None:
            raise ValueError("contour_axis_deg is required for projected motion modes")
        component = "along" if mode == "along_only" else "across"
        u = _component_vector(float(contour_axis_deg), component).astype(np.float32)
        # The full BackImage phase convention is screen_x=-gaze_x, screen_y=gaze_y.
        # A gaze-coordinate projected vector d*u therefore couples to
        # (-u_x*kx + u_y*ky).  The temporal power is even in sign, but this keeps
        # the direct control aligned with the full 2D kernel.
        flat_keff = (-u[0]) * flat_fx + u[1] * flat_fy
    elif mode == "full_2d":
        flat_keff = None
    else:
        raise ValueError(f"unknown mode {mode!r}")

    n_trace = 0
    for trace_start in range(0, traces_arr.shape[0], int(chunk_size)):
        chunk = traces_arr[trace_start : trace_start + int(chunk_size)].astype(np.float32, copy=True)
        chunk -= np.nanmean(chunk, axis=1, keepdims=True)
        screen_x = -chunk[:, :, 0]
        screen_y = chunk[:, :, 1]
        if mode == "full_2d":
            scalar_disp = None
        else:
            assert contour_axis_deg is not None
            component = "along" if mode == "along_only" else "across"
            u = _component_vector(float(contour_axis_deg), component).astype(np.float32)
            scalar_disp = chunk[:, :, 0] * u[0] + chunk[:, :, 1] * u[1]
        n_trace += int(chunk.shape[0])
        for coeff_start in range(0, flat_fx.size, int(coeff_chunk_size)):
            coeff_stop = min(coeff_start + int(coeff_chunk_size), flat_fx.size)
            if mode == "full_2d":
                phase_arg = (
                    screen_x[:, :, None] * flat_fx[None, None, coeff_start:coeff_stop]
                    + screen_y[:, :, None] * flat_fy[None, None, coeff_start:coeff_stop]
                )
            else:
                assert scalar_disp is not None and flat_keff is not None
                phase_arg = scalar_disp[:, :, None] * flat_keff[None, None, coeff_start:coeff_stop]
            phase = np.exp((-2j * np.pi) * phase_arg)
            phase -= np.mean(phase, axis=1, keepdims=True)
            spec = np.fft.fft(phase, axis=1, norm="ortho")
            power = np.sum(np.abs(spec) ** 2, axis=0)
            for freq_i, bins in enumerate(freq_bins):
                q[coeff_start:coeff_stop, freq_i] += np.sum(power[bins, :], axis=0)
    if n_trace == 0:
        return freq_hz, q.astype(np.float32)
    return freq_hz, (q / float(n_trace)).astype(np.float32)


def bin_coeff_temporal(
    source_power2d: np.ndarray,
    q_flat: np.ndarray,
    bin_values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_source = np.asarray(source_power2d, dtype=np.float64).ravel()
    q = np.asarray(q_flat, dtype=np.float64)
    bins = np.asarray(bin_values, dtype=np.float64).ravel()
    n_tf = q.shape[1]
    source = np.full(edges.size - 1, np.nan, dtype=np.float64)
    q_mean = np.full((edges.size - 1, n_tf), np.nan, dtype=np.float64)
    mod = np.full((edges.size - 1, n_tf), np.nan, dtype=np.float64)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (bins >= float(lo)) & (bins < float(hi))
        counts[i] = int(np.sum(mask))
        if not np.any(mask):
            continue
        source[i] = float(np.nanmean(flat_source[mask]))
        q_mean[i, :] = np.nanmean(q[mask, :], axis=0)
        mod[i, :] = np.nanmean(flat_source[mask, None] * q[mask, :], axis=0)
    return source, q_mean, mod, counts


def compute_direct_projected_summary(
    *,
    source_powers: np.ndarray,
    image_rows: list[dict[str, Any]],
    fx: np.ndarray,
    fy: np.ndarray,
    rr: np.ndarray,
    traces: np.ndarray,
    trace_groups: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bool(args.skip_direct):
        return pd.DataFrame(), pd.DataFrame()
    edges = radial_edges(rr, int(args.n_sf_bins))
    sf_centers = geometric_centers(edges)
    image_count = min(int(args.direct_max_images), int(source_powers.shape[0]), len(image_rows))
    rows: list[dict[str, Any]] = []
    for condition in TRACE_GROUP_ORDER:
        selected = _selected_traces(
            traces,
            trace_groups[condition],
            max_n=int(args.direct_max_traces),
            seed=int(args.seed) + 7019 + TRACE_GROUP_ORDER.index(condition),
        )
        print(f"[direct] {condition}: {selected.shape[0]} traces x {image_count} images", flush=True)
        full_freq, full_q = motion_q_spectrum_projected_mode(
            selected,
            fx,
            fy,
            mode="full_2d",
            contour_axis_deg=None,
            dt_s=float(args.dt_s),
            chunk_size=int(args.trace_chunk_size),
            coeff_chunk_size=int(args.coeff_chunk_size),
        )
        for image_i in range(image_count):
            image_power = source_powers[image_i]
            image_row = image_rows[image_i]
            axis_deg = float(image_row["image_edge_axis_deg"])
            if not np.isfinite(axis_deg):
                continue
            mode_q: dict[str, tuple[np.ndarray, np.ndarray]] = {"full_2d": (full_freq, full_q)}
            for mode in ("along_only", "across_only"):
                freq_i, q_i = motion_q_spectrum_projected_mode(
                    selected,
                    fx,
                    fy,
                    mode=mode,
                    contour_axis_deg=axis_deg,
                    dt_s=float(args.dt_s),
                    chunk_size=int(args.trace_chunk_size),
                    coeff_chunk_size=int(args.coeff_chunk_size),
                )
                mode_q[mode] = (freq_i, q_i)
            for mode, (freq_hz, q_flat) in mode_q.items():
                source_axis, q_mean, mod_mean, counts = bin_coeff_temporal(image_power, q_flat, rr, edges)
                if not np.allclose(freq_hz, full_freq):
                    raise RuntimeError("Temporal frequency support changed.")
                for sf_i, sf in enumerate(sf_centers):
                    for tf_i, tf in enumerate(freq_hz):
                        rows.append(
                            {
                                "calculation": "direct_projected_motion",
                                "condition": condition,
                                "condition_label": TRACE_GROUP_LABELS[condition],
                                "motion_mode": mode,
                                "motion_mode_label": MOTION_MODE_LABELS[mode],
                                "spatial_axis": "radial",
                                "image_pos": int(image_i),
                                "image_index": int(image_row["image_index"]),
                                "spatial_frequency_cpd": float(sf),
                                "temporal_frequency_hz": float(tf),
                                "source_power_mean": float(source_axis[sf_i]),
                                "motion_q_mean": float(q_mean[sf_i, tf_i]),
                                "modulation_power_mean": float(mod_mean[sf_i, tf_i]),
                                "n_coefficients": int(counts[sf_i]),
                                "n_traces": int(selected.shape[0]),
                            }
                        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    summary = (
        detail.groupby(
            [
                "calculation",
                "condition",
                "condition_label",
                "motion_mode",
                "motion_mode_label",
                "spatial_axis",
                "spatial_frequency_cpd",
                "temporal_frequency_hz",
            ],
            sort=False,
        )
        .agg(
            source_power_mean=("source_power_mean", "mean"),
            motion_q_mean=("motion_q_mean", "mean"),
            modulation_power_mean=("modulation_power_mean", "mean"),
            n_images=("image_pos", "nunique"),
            n_traces=("n_traces", "first"),
        )
        .reset_index()
    )
    bands = band_summary(summary, group_cols=("calculation", "condition", "motion_mode"))
    return summary, bands


def band_summary(frame: pd.DataFrame, *, group_cols: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols_list = list(group_cols)
    for keys, sub in frame.groupby(group_cols_list, sort=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        key_payload = dict(zip(group_cols_list, key_tuple, strict=True))
        for band_id, band_label, low_cpd, high_cpd in SPATIAL_BANDS:
            keep = (sub["spatial_frequency_cpd"] >= float(low_cpd)) & (sub["spatial_frequency_cpd"] < float(high_cpd))
            band = sub[keep]
            if band.empty:
                continue
            by_tf = band.groupby("temporal_frequency_hz", sort=True)["modulation_power_mean"].mean()
            freq = by_tf.index.to_numpy(dtype=np.float64)
            power = by_tf.to_numpy(dtype=np.float64)
            total = float(np.nansum(power))
            rows.append(
                {
                    **key_payload,
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
    baseline_cols = [col for col in group_cols_list if col != "condition"]
    base = out[out["condition"].eq("all_real_fem")].copy()
    rename = {"total_nonzero_tf_power": "all_real_total_nonzero_tf_power"}
    base = base[baseline_cols + ["spatial_band", "total_nonzero_tf_power"]].rename(columns=rename)
    out = out.merge(base, on=baseline_cols + ["spatial_band"], how="left")
    out["total_power_over_all_real_same_band"] = out["total_nonzero_tf_power"] / np.maximum(
        out["all_real_total_nonzero_tf_power"], EPS
    )
    return out


def matrix_from_summary(frame: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.asarray(sorted(frame["spatial_frequency_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(frame["temporal_frequency_hz"].unique()), dtype=np.float64)
    mat = np.full((tf.size, sf.size), np.nan, dtype=np.float64)
    sf_index = {float(v): i for i, v in enumerate(sf)}
    tf_index = {float(v): i for i, v in enumerate(tf)}
    for row in frame.itertuples(index=False):
        mat[tf_index[float(row.temporal_frequency_hz)], sf_index[float(row.spatial_frequency_cpd)]] = float(
            getattr(row, value_col)
        )
    return sf, tf, mat


def setup_sftf_axis(ax: plt.Axes, sf: np.ndarray, tf: np.ndarray, *, xlabel: str) -> None:
    sf_edges = spatial_edges_from_centers(sf)
    tf_edges = temporal_edges(tf)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(sf_edges[0]), float(sf_edges[-1]))
    ax.set_ylim(max(1.5, float(tf_edges[0])), float(tf_edges[-1]))
    ax.set_xticks([0.5, 1, 3, 10], ["0.5", "1", "3", "10"])
    ax.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("temporal frequency (Hz)")
    ax.spines[["top", "right"]].set_visible(False)


def plot_factorized_heatmaps(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    axis_method: str,
) -> Path:
    sub_all = summary[summary["axis_method"].eq(axis_method) & summary["condition"].isin(PLOT_TRACE_GROUPS)]
    values = db(sub_all["modulation_power_mean"].to_numpy(dtype=np.float64))
    finite = values[np.isfinite(values)]
    vmin = float(np.percentile(finite, 5.0)) if finite.size else -80.0
    vmax = float(np.percentile(finite, 98.0)) if finite.size else -20.0
    fig, axes = plt.subplots(2, len(PLOT_TRACE_GROUPS), figsize=(14.0, 7.4), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.08, top=0.83, hspace=0.62, wspace=0.24)
    mesh = None
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    panel_i = 0
    for row_i, component in enumerate(("along", "across")):
        for col_i, condition in enumerate(PLOT_TRACE_GROUPS):
            ax = axes[row_i, col_i]
            sub = sub_all[sub_all["condition"].eq(condition) & sub_all["component"].eq(component)]
            sf, tf, mat = matrix_from_summary(sub, "modulation_power_mean")
            mesh = ax.pcolormesh(
                spatial_edges_from_centers(sf),
                temporal_edges(tf),
                db(mat),
                shading="auto",
                cmap="hot",
                vmin=vmin,
                vmax=vmax,
            )
            xlabel = "projected spatial frequency (cycles/deg)" if axis_method == "projection" else "radial spatial frequency (cycles/deg)"
            setup_sftf_axis(ax, sf, tf, xlabel=xlabel)
            if col_i > 0:
                ax.set_ylabel("")
            title = f"{letters[panel_i]}. {TRACE_GROUP_LABELS[condition]}\n{COMPONENT_LABELS[component]}"
            ax.set_title(title, loc="left", fontweight="bold", fontsize=8.8, color=TRACE_GROUP_COLORS[condition])
            panel_i += 1
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.012, shrink=0.86)
        cbar.set_label("10 log10 image-weighted motion power")
    fig.suptitle(
        f"Fast contour-coordinate SF/TF factorization by RMS bin: {AXIS_METHOD_LABELS[axis_method]}",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    stem = f"factorized_{axis_method}_component_heatmaps_by_rms"
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def ratio_frame(summary: pd.DataFrame) -> pd.DataFrame:
    along = summary[summary["component"].eq("along")].rename(
        columns={"modulation_power_mean": "along_modulation_power_mean", "motion_q_mean": "along_motion_q_mean"}
    )
    across = summary[summary["component"].eq("across")].rename(
        columns={"modulation_power_mean": "across_modulation_power_mean", "motion_q_mean": "across_motion_q_mean"}
    )
    merged = across.merge(
        along[
            [
                "axis_method",
                "condition",
                "spatial_frequency_cpd",
                "temporal_frequency_hz",
                "along_modulation_power_mean",
                "along_motion_q_mean",
            ]
        ],
        on=["axis_method", "condition", "spatial_frequency_cpd", "temporal_frequency_hz"],
        how="inner",
    )
    merged["across_minus_along_modulation_db"] = db(
        merged["across_modulation_power_mean"].to_numpy(dtype=np.float64)
    ) - db(merged["along_modulation_power_mean"].to_numpy(dtype=np.float64))
    merged["across_over_along_modulation"] = merged["across_modulation_power_mean"] / np.maximum(
        merged["along_modulation_power_mean"], EPS
    )
    return merged


def plot_factorized_ratio_heatmaps(ratios: pd.DataFrame, out_dir: Path, *, axis_method: str) -> Path:
    sub_all = ratios[ratios["axis_method"].eq(axis_method) & ratios["condition"].isin(PLOT_TRACE_GROUPS)]
    vals = sub_all["across_minus_along_modulation_db"].to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vmax = max(float(np.percentile(np.abs(vals), 98.0)), 1.0) if vals.size else 3.0
    fig, axes = plt.subplots(1, len(PLOT_TRACE_GROUPS), figsize=(14.0, 3.35), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.18, top=0.74, wspace=0.24)
    mesh = None
    for col_i, condition in enumerate(PLOT_TRACE_GROUPS):
        ax = axes[0, col_i]
        sub = sub_all[sub_all["condition"].eq(condition)]
        sf, tf, mat = matrix_from_summary(sub, "across_minus_along_modulation_db")
        mesh = ax.pcolormesh(
            spatial_edges_from_centers(sf),
            temporal_edges(tf),
            mat,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        setup_sftf_axis(
            ax,
            sf,
            tf,
            xlabel="projected spatial frequency (cycles/deg)" if axis_method == "projection" else "radial spatial frequency (cycles/deg)",
        )
        if col_i > 0:
            ax.set_ylabel("")
        ax.set_title(TRACE_GROUP_LABELS[condition], loc="left", fontweight="bold", color=TRACE_GROUP_COLORS[condition])
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.012, shrink=0.9)
        cbar.set_label("normal minus parallel power (dB)")
    fig.suptitle(
        f"Contour-normal minus contour-parallel SF/TF power: {AXIS_METHOD_LABELS[axis_method]}",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    stem = f"factorized_{axis_method}_normal_minus_parallel_by_rms"
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def plot_factorized_band_dose(bands: pd.DataFrame, trace_meta: pd.DataFrame, out_dir: Path) -> Path:
    plot_bands = bands[
        bands["axis_method"].eq("projection") & bands["condition"].isin(("drift_rms_low", "drift_rms_mid", "drift_rms_high"))
    ].copy()
    plot_bands = plot_bands.merge(trace_meta[["condition", "rms_median_arcmin"]], on="condition", how="left")
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.1, top=0.84, hspace=0.38, wspace=0.32)
    for col_i, (band_id, band_label, _lo, _hi) in enumerate(SPATIAL_BANDS):
        ax_total = axes[0, col_i]
        ax_centroid = axes[1, col_i]
        for component, color, marker in (("along", "#2B6CB0", "s"), ("across", "#B24A3B", "o")):
            sub = plot_bands[plot_bands["spatial_band"].eq(band_id) & plot_bands["component"].eq(component)].sort_values(
                "rms_median_arcmin"
            )
            if sub.empty:
                continue
            ax_total.plot(
                sub["rms_median_arcmin"],
                sub["total_power_over_all_real_same_band"],
                marker=marker,
                color=color,
                linewidth=1.9,
                label=COMPONENT_LABELS[component],
            )
            ax_centroid.plot(
                sub["rms_median_arcmin"],
                sub["tf_power_centroid_hz"],
                marker=marker,
                color=color,
                linewidth=1.9,
            )
        ax_total.axhline(1.0, color="0.55", linestyle="--", linewidth=0.8)
        ax_total.set_title(band_label, loc="left", fontweight="bold")
        ax_total.set_yscale("log")
        ax_total.set_ylabel("power / all-real")
        ax_centroid.set_xlabel("trace RMS excursion bin median (arcmin)")
        ax_centroid.set_ylabel("TF centroid (Hz)")
        ax_total.spines[["top", "right"]].set_visible(False)
        ax_centroid.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Projected-axis SF/TF power dose curves across drift RMS bins",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    png = out_dir / "factorized_projection_band_dose_by_rms.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "factorized_projection_band_dose_by_rms.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "factorized_projection_band_dose_by_rms.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def direct_ratio_frame(summary: pd.DataFrame) -> pd.DataFrame:
    along = summary[summary["motion_mode"].eq("along_only")].rename(
        columns={"modulation_power_mean": "along_modulation_power_mean"}
    )
    across = summary[summary["motion_mode"].eq("across_only")].rename(
        columns={"modulation_power_mean": "across_modulation_power_mean"}
    )
    merged = across.merge(
        along[["condition", "spatial_frequency_cpd", "temporal_frequency_hz", "along_modulation_power_mean"]],
        on=["condition", "spatial_frequency_cpd", "temporal_frequency_hz"],
        how="inner",
    )
    merged["across_minus_along_modulation_db"] = db(
        merged["across_modulation_power_mean"].to_numpy(dtype=np.float64)
    ) - db(merged["along_modulation_power_mean"].to_numpy(dtype=np.float64))
    return merged


def plot_direct_heatmaps(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    tuning_surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Path | None:
    if summary.empty:
        return None
    plot_groups = PLOT_TRACE_GROUPS
    modes = ("full_2d", "along_only", "across_only", "ratio")
    values = db(summary[summary["condition"].isin(plot_groups)]["modulation_power_mean"].to_numpy(dtype=np.float64))
    finite = values[np.isfinite(values)]
    vmin = float(np.percentile(finite, 5.0)) if finite.size else -80.0
    vmax = float(np.percentile(finite, 98.0)) if finite.size else -20.0
    ratios = direct_ratio_frame(summary)
    ratio_vals = ratios[ratios["condition"].isin(plot_groups)]["across_minus_along_modulation_db"].to_numpy(dtype=np.float64)
    ratio_vals = ratio_vals[np.isfinite(ratio_vals)]
    ratio_vmax = max(float(np.percentile(np.abs(ratio_vals), 98.0)), 1.0) if ratio_vals.size else 3.0
    fig, axes = plt.subplots(len(plot_groups), len(modes), figsize=(14.0, 11.0), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.065, top=0.88, hspace=0.58, wspace=0.22)
    heat_mesh = None
    ratio_mesh = None
    for row_i, condition in enumerate(plot_groups):
        for col_i, mode in enumerate(modes):
            ax = axes[row_i, col_i]
            if mode == "ratio":
                sub = ratios[ratios["condition"].eq(condition)]
                sf, tf, mat = matrix_from_summary(sub, "across_minus_along_modulation_db")
                ratio_mesh = ax.pcolormesh(
                    spatial_edges_from_centers(sf),
                    temporal_edges(tf),
                    mat,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-ratio_vmax,
                    vmax=ratio_vmax,
                )
                title = "normal - parallel"
            else:
                sub = summary[summary["condition"].eq(condition) & summary["motion_mode"].eq(mode)]
                sf, tf, mat = matrix_from_summary(sub, "modulation_power_mean")
                heat_mesh = ax.pcolormesh(
                    spatial_edges_from_centers(sf),
                    temporal_edges(tf),
                    db(mat),
                    shading="auto",
                    cmap="hot",
                    vmin=vmin,
                    vmax=vmax,
                )
                title = MOTION_MODE_LABELS[mode]
            setup_sftf_axis(ax, sf, tf, xlabel="radial spatial frequency (cycles/deg)")
            if row_i < len(plot_groups) - 1:
                ax.set_xlabel("")
            if col_i > 0:
                ax.set_ylabel("")
            if col_i == 0:
                ax.text(
                    -0.38,
                    0.5,
                    TRACE_GROUP_LABELS[condition],
                    color=TRACE_GROUP_COLORS[condition],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
            if row_i == 0:
                ax.set_title(title, loc="left", fontweight="bold")
            if mode != "ratio":
                overlay_tuning_contours(ax, tuning_surfaces, linewidth=0.9)
                if row_i == 0 and col_i == 0:
                    add_tuning_legend(ax, tuning_surfaces, loc="lower left")
    if heat_mesh is not None:
        cbar = fig.colorbar(heat_mesh, ax=axes[:, :3].ravel().tolist(), pad=0.012, shrink=0.86)
        cbar.set_label("10 log10 image-weighted motion power")
    if ratio_mesh is not None:
        cbar = fig.colorbar(ratio_mesh, ax=axes[:, 3].ravel().tolist(), pad=0.018, shrink=0.86)
        cbar.set_label("normal minus parallel (dB)")
    fig.suptitle(
        "Direct coefficient-level projected-motion controls by RMS bin",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    png = out_dir / "direct_projected_motion_radial_heatmaps_by_rms.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "direct_projected_motion_radial_heatmaps_by_rms.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "direct_projected_motion_radial_heatmaps_by_rms.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def write_readme(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    paths: dict[str, Path | None],
    trace_meta: pd.DataFrame,
    factor_bands: pd.DataFrame,
    direct_bands: pd.DataFrame,
) -> None:
    lines = [
        "# SSI SF/TF Motion Component Options",
        "",
        "Exploratory plots for separating contour-relative image frequency from contour-relative FEM components.",
        "",
        f"- Image bank: `{_relative(Path(args.image_table))}`",
        f"- Trace bank: `{_relative(Path(args.trace_table))}`",
        f"- Trace XY: `{_relative(Path(args.trace_xy))}`",
        f"- Tuning contours: `{_relative(Path(args.tuning_points_csv))}`",
        "",
        "## Figures",
        "",
    ]
    for key, path in paths.items():
        if path is not None:
            lines.append(f"- {key}: `{_relative(Path(path))}`")
    lines.extend(
        [
            "",
            "## Trace RMS Groups",
            "",
        ]
    )
    for row in trace_meta.itertuples(index=False):
        lines.append(
            f"- {row.condition_label}: n={int(row.n_selected_traces)} selected "
            f"(available {int(row.n_available_traces)}), median RMS {float(row.rms_median_arcmin):.3g} arcmin, "
            f"median path {float(row.path_median_arcmin):.3g} arcmin."
        )
    lines.extend(
        [
            "",
            "## What The Views Mean",
            "",
            "- `factorized_projection_*`: bins the image spectrum by the absolute projection of each Fourier wavevector onto the contour-parallel or contour-normal axis, then multiplies by the temporal spectrum of the matching projected trajectory component.",
            f"- `factorized_wedge_*`: uses +/-{float(args.wedge_half_width_deg):.1f} deg Fourier wedges around the contour axis or normal. This is closer to an oriented-slice view.",
            "- `direct_projected_motion_*`: smaller exact control. It keeps every 2D Fourier coefficient of each image, but drives it with full, contour-parallel-only, or contour-normal-only motion before radial binning.",
            "",
            "## Quick Band Checks",
            "",
        ]
    )
    if not factor_bands.empty:
        use = factor_bands[
            factor_bands["axis_method"].eq("projection")
            & factor_bands["condition"].isin(("drift_rms_low", "drift_rms_high"))
            & factor_bands["spatial_band"].eq("high_sf")
        ]
        if not use.empty:
            lines.append("Projection-factorized high-SF band, low-to-high RMS:")
            for component in ("along", "across"):
                sub = use[use["component"].eq(component)].set_index("condition")
                if {"drift_rms_low", "drift_rms_high"}.issubset(sub.index):
                    lines.append(
                        f"- {COMPONENT_LABELS[component]}: "
                        f"{float(sub.loc['drift_rms_low', 'total_power_over_all_real_same_band']):.3g}x -> "
                        f"{float(sub.loc['drift_rms_high', 'total_power_over_all_real_same_band']):.3g}x all-real."
                    )
    if not direct_bands.empty:
        use = direct_bands[
            direct_bands["condition"].isin(("drift_rms_low", "drift_rms_high"))
            & direct_bands["spatial_band"].eq("high_sf")
        ]
        if not use.empty:
            lines.append("")
            lines.append("Direct radial high-SF band, low-to-high RMS:")
            for mode in ("full_2d", "along_only", "across_only"):
                sub = use[use["motion_mode"].eq(mode)].set_index("condition")
                if {"drift_rms_low", "drift_rms_high"}.issubset(sub.index):
                    lines.append(
                        f"- {MOTION_MODE_LABELS[mode]}: "
                        f"{float(sub.loc['drift_rms_low', 'total_power_over_all_real_same_band']):.3g}x -> "
                        f"{float(sub.loc['drift_rms_high', 'total_power_over_all_real_same_band']):.3g}x all-real."
                    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path | None]:
    configure_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_table = pd.read_csv(args.image_table)
    trace_table = pd.read_csv(args.trace_table)
    traces = np.load(args.trace_xy, mmap_mode="r")
    trace_groups, trace_meta = select_rms_trace_groups(
        trace_table,
        max_traces_per_condition=int(args.max_traces_per_condition),
        seed=int(args.seed),
    )
    trace_meta.to_csv(out_dir / "trace_rms_group_summary.csv", index=False)
    source_powers, image_rows, _example_patch = load_image_powers(
        image_table,
        max_images=int(args.max_images),
        patch_size_px=int(args.patch_size_px),
    )
    image_rows_df = pd.DataFrame(image_rows)
    image_rows_df.to_csv(out_dir / "image_sample_summary.csv", index=False)
    ppd = float(np.nanmedian(image_rows_df["ppd"].to_numpy(dtype=np.float64)))
    fx, fy, rr = frequency_grid(int(args.patch_size_px), ppd)

    factor_summary, factor_bands = compute_factorized_summary(
        source_powers=source_powers,
        image_rows=image_rows,
        fx=fx,
        fy=fy,
        rr=rr,
        traces=traces,
        trace_groups=trace_groups,
        args=args,
    )
    factor_summary_csv = out_dir / "factorized_axis_sftf_summary.csv"
    factor_bands_csv = out_dir / "factorized_axis_band_summary.csv"
    factor_summary.to_csv(factor_summary_csv, index=False)
    factor_bands.to_csv(factor_bands_csv, index=False)
    ratios = ratio_frame(factor_summary)
    ratios_csv = out_dir / "factorized_axis_normal_minus_parallel_summary.csv"
    ratios.to_csv(ratios_csv, index=False)

    direct_summary, direct_bands = compute_direct_projected_summary(
        source_powers=source_powers,
        image_rows=image_rows,
        fx=fx,
        fy=fy,
        rr=rr,
        traces=traces,
        trace_groups=trace_groups,
        args=args,
    )
    direct_summary_csv = out_dir / "direct_projected_motion_sftf_summary.csv"
    direct_bands_csv = out_dir / "direct_projected_motion_band_summary.csv"
    direct_summary.to_csv(direct_summary_csv, index=False)
    direct_bands.to_csv(direct_bands_csv, index=False)

    tuning_surfaces = load_tuning_surfaces(Path(args.tuning_points_csv))
    paths: dict[str, Path | None] = {
        "factorized_projection_heatmaps": plot_factorized_heatmaps(factor_summary, out_dir, axis_method="projection"),
        "factorized_projection_ratio": plot_factorized_ratio_heatmaps(ratios, out_dir, axis_method="projection"),
        "factorized_wedge_heatmaps": plot_factorized_heatmaps(factor_summary, out_dir, axis_method="wedge"),
        "factorized_wedge_ratio": plot_factorized_ratio_heatmaps(ratios, out_dir, axis_method="wedge"),
        "factorized_projection_band_dose": plot_factorized_band_dose(factor_bands, trace_meta, out_dir),
        "direct_projected_heatmaps": plot_direct_heatmaps(direct_summary, out_dir, tuning_surfaces=tuning_surfaces),
        "factorized_summary_csv": factor_summary_csv,
        "factorized_band_csv": factor_bands_csv,
        "factorized_ratio_csv": ratios_csv,
        "direct_summary_csv": direct_summary_csv,
        "direct_band_csv": direct_bands_csv,
    }
    write_readme(
        out_dir,
        args=args,
        paths=paths,
        trace_meta=trace_meta,
        factor_bands=factor_bands,
        direct_bands=direct_bands,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_IMAGE_TABLE)
    parser.add_argument("--trace-table", type=Path, default=DEFAULT_TRACE_TABLE)
    parser.add_argument("--trace-xy", type=Path, default=DEFAULT_TRACE_XY)
    parser.add_argument("--tuning-points-csv", type=Path, default=DEFAULT_TUNING_POINTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-images", type=int, default=18)
    parser.add_argument("--max-traces-per-condition", type=int, default=160)
    parser.add_argument("--direct-max-images", type=int, default=6)
    parser.add_argument("--direct-max-traces", type=int, default=64)
    parser.add_argument("--patch-size-px", type=int, default=PATCH_SIZE_PX)
    parser.add_argument("--n-sf-bins", type=int, default=17)
    parser.add_argument("--wedge-half-width-deg", type=float, default=22.5)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--trace-chunk-size", type=int, default=128)
    parser.add_argument("--coeff-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-direct", action="store_true")
    return parser.parse_args()


def main() -> None:
    paths = run(parse_args())
    for key, path in paths.items():
        if path is not None:
            print(f"{key}: {path}")


if __name__ == "__main__":
    main()
