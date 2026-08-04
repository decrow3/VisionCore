#!/usr/bin/env python3
"""Plot joint spatial/temporal retinal power induced by SSI-v3 FEM banks.

This is the spatiotemporal counterpart to
``plot_eye_movement_power_spectrum_shift.py``.  It follows the Rucci-style
factorization used in the reference notebook:

    retinal movie power(k, f_t) ~= static image power(k) * eye-motion Q(k, f_t)

where ``Q`` is the temporal spectrum of the Fourier phase modulation generated
by the fixation trajectory.  The output focuses on non-zero temporal
frequencies, because static translations preserve image Fourier magnitude and
only redistribute power through time.
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

from plot_eye_movement_power_spectrum_shift import (
    COLORS,
    DEFAULT_IMAGE_TABLE,
    DEFAULT_TRACE_TABLE,
    DEFAULT_TRACE_XY,
    DT_S,
    FIT_HIGH_CPD,
    FIT_LOW_CPD,
    LABELS,
    PATCH_SIZE_PX,
    ROOT,
    frequency_grid,
    load_image_powers,
    radial_bin,
    radial_edges,
    sample_indices,
    select_trace_groups,
    write_csv,
    write_json,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "joint_sftf_power"
PLOT_CONDITIONS = ("short_drift", "mid_drift", "long_drift", "microsaccade", "all_real_fem")
REFERENCE_CONDITIONS = ("all_real_fem", "drift_only", "brownian_matched")
EXTRA_LABELS = {
    "drift_only": "drift only",
    "brownian_matched": "BM matched",
}
EXTRA_COLORS = {
    "drift_only": "#2f63c6",
    "brownian_matched": "#333333",
}
SPATIAL_BANDS = (
    ("low_sf", "low SF", 0.5, 2.0),
    ("mid_sf", "mid SF", 2.0, 8.0),
    ("high_sf", "high SF", 8.0, 16.0),
)


def condition_label(condition: str) -> str:
    return LABELS.get(condition, EXTRA_LABELS.get(condition, condition.replace("_", " ")))


def condition_color(condition: str) -> str:
    return COLORS.get(condition, EXTRA_COLORS.get(condition, "black"))
TEMPORAL_BANDS = (
    ("slow_tf", "slow TF", 3.0, 12.0),
    ("mid_tf", "mid TF", 12.0, 30.0),
    ("fast_tf", "fast TF", 30.0, 60.0),
)


def abs_temporal_frequency_bins(n_time: int, dt_s: float) -> tuple[np.ndarray, list[np.ndarray]]:
    freqs = np.fft.fftfreq(int(n_time), d=float(dt_s))
    abs_freqs = np.unique(np.abs(freqs))
    abs_freqs = abs_freqs[abs_freqs > 0]
    bins = [np.where(np.isclose(np.abs(freqs), f))[0] for f in abs_freqs]
    return abs_freqs.astype(np.float32), bins


def temporal_edges(freq_hz: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq_hz, dtype=np.float64)
    if freq.size == 1:
        width = max(float(freq[0]), 1.0)
        return np.asarray([max(0.0, float(freq[0]) - width / 2.0), float(freq[0]) + width / 2.0])
    mid = 0.5 * (freq[:-1] + freq[1:])
    first = max(0.0, float(freq[0]) - float(mid[0] - freq[0]))
    last = float(freq[-1]) + float(freq[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]]).astype(np.float64)


def add_reference_trace_groups(
    trace_groups: dict[str, np.ndarray],
    trace_group_rows: list[dict[str, Any]],
    trace_table: pd.DataFrame,
    traces: np.ndarray,
    *,
    max_traces_per_condition: int,
    seed: int,
) -> tuple[dict[str, np.ndarray | tuple[str, np.ndarray]], list[dict[str, Any]]]:
    out_groups: dict[str, np.ndarray | tuple[str, np.ndarray]] = {key: value for key, value in trace_groups.items()}
    out_rows = list(trace_group_rows)
    table = trace_table.copy()
    table["trace_bank_index"] = pd.to_numeric(table["trace_bank_index"], errors="coerce").astype(int)
    table["path_length_arcmin"] = pd.to_numeric(table["rendered_path_length_arcmin"], errors="coerce")
    table["n_ms"] = pd.to_numeric(table["rendered_n_microsaccade_events"], errors="coerce").fillna(0).astype(int)
    no_ms = table[(table["n_ms"] == 0) & np.isfinite(table["path_length_arcmin"])].copy()
    drift_indices = sample_indices(
        no_ms["trace_bank_index"].to_numpy(dtype=int),
        max_n=int(max_traces_per_condition),
        seed=int(seed) + 9419,
    )
    out_groups["drift_only"] = drift_indices
    selected_frame = table[table["trace_bank_index"].isin(drift_indices)]
    out_rows.append(
        {
            "condition": "drift_only",
            "label": condition_label("drift_only"),
            "n_available_traces": int(no_ms.shape[0]),
            "n_selected_traces": int(drift_indices.size),
            "path_q25_arcmin": float(selected_frame["path_length_arcmin"].quantile(0.25)) if not selected_frame.empty else float("nan"),
            "path_median_arcmin": float(selected_frame["path_length_arcmin"].median()) if not selected_frame.empty else float("nan"),
            "path_q75_arcmin": float(selected_frame["path_length_arcmin"].quantile(0.75)) if not selected_frame.empty else float("nan"),
            "microsaccade_fraction": 0.0,
        }
    )

    all_real = np.asarray(traces[np.asarray(trace_groups["all_real_fem"], dtype=int)], dtype=np.float32)
    brownian = synthetic_brownian_matched(all_real, seed=int(seed) + 17777)
    out_groups["brownian_matched"] = ("array", brownian)
    brownian_path = np.sum(np.linalg.norm(np.diff(brownian, axis=1), axis=2), axis=1) * 60.0
    out_rows.append(
        {
            "condition": "brownian_matched",
            "label": condition_label("brownian_matched"),
            "n_available_traces": int(all_real.shape[0]),
            "n_selected_traces": int(brownian.shape[0]),
            "path_q25_arcmin": float(np.quantile(brownian_path, 0.25)),
            "path_median_arcmin": float(np.median(brownian_path)),
            "path_q75_arcmin": float(np.quantile(brownian_path, 0.75)),
            "microsaccade_fraction": float("nan"),
        }
    )
    return out_groups, out_rows


def synthetic_brownian_matched(traces_xy_deg: np.ndarray, *, seed: int) -> np.ndarray:
    traces = np.asarray(traces_xy_deg, dtype=np.float32)
    steps = np.diff(traces, axis=1)
    flat_steps = steps.reshape(-1, 2)
    cov = np.cov(flat_steps.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        step_std = float(np.nanstd(flat_steps))
        cov = np.eye(2, dtype=np.float64) * max(step_std, 1e-6) ** 2
    cov = np.asarray(cov, dtype=np.float64)
    cov.flat[::3] += 1e-12
    rng = np.random.default_rng(int(seed))
    synth_steps = rng.multivariate_normal(np.zeros(2), cov, size=(traces.shape[0], traces.shape[1] - 1)).astype(np.float32)
    walk = np.concatenate([np.zeros((traces.shape[0], 1, 2), dtype=np.float32), np.cumsum(synth_steps, axis=1)], axis=1)
    walk -= np.mean(walk, axis=1, keepdims=True)
    return walk.astype(np.float32)


def motion_q_spectrum(
    traces_xy_deg: np.ndarray,
    fx_cpd: np.ndarray,
    fy_cpd: np.ndarray,
    *,
    dt_s: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces_xy_deg, dtype=np.float32)
    if traces.ndim != 3 or traces.shape[1] < 2 or traces.shape[2] != 2:
        raise ValueError(f"Expected traces with shape (n,T,2); got {traces.shape}")
    flat_fx = np.asarray(fx_cpd, dtype=np.float32).ravel()
    flat_fy = np.asarray(fy_cpd, dtype=np.float32).ravel()
    freq_hz, freq_bins = abs_temporal_frequency_bins(traces.shape[1], float(dt_s))
    q = np.zeros((flat_fx.size, freq_hz.size), dtype=np.float64)
    n_trace = 0
    for trace_start in range(0, traces.shape[0], int(chunk_size)):
        chunk = traces[trace_start : trace_start + int(chunk_size)].astype(np.float32, copy=True)
        chunk -= np.nanmean(chunk, axis=1, keepdims=True)
        screen_x = -chunk[:, :, 0]
        screen_y = chunk[:, :, 1]
        n_trace += int(chunk.shape[0])
        for coeff_start in range(0, flat_fx.size, 512):
            coeff_stop = min(coeff_start + 512, flat_fx.size)
            phase_arg = (
                screen_x[:, :, None] * flat_fx[None, None, coeff_start:coeff_stop]
                + screen_y[:, :, None] * flat_fy[None, None, coeff_start:coeff_stop]
            )
            phase = np.exp((-2j * np.pi) * phase_arg)
            phase -= np.mean(phase, axis=1, keepdims=True)
            spec = np.fft.fft(phase, axis=1, norm="ortho")
            power = np.sum(np.abs(spec) ** 2, axis=0)
            for freq_i, bins in enumerate(freq_bins):
                q[coeff_start:coeff_stop, freq_i] += np.sum(power[bins, :], axis=0)
    if n_trace == 0:
        return freq_hz, q.astype(np.float32)
    return freq_hz, (q / float(n_trace)).astype(np.float32)


def radial_temporal_summary(
    source_power2d: np.ndarray,
    q_by_condition: dict[str, np.ndarray],
    freq_hz: np.ndarray,
    rr: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    sf_center, source_radial = radial_bin(source_power2d, rr, edges)
    rows: list[dict[str, Any]] = []
    flat_source = np.asarray(source_power2d, dtype=np.float64).ravel()
    flat_rr = np.asarray(rr, dtype=np.float64).ravel()
    for condition, q in q_by_condition.items():
        q_flat = np.asarray(q, dtype=np.float64)
        modulation = flat_source[:, None] * q_flat
        for sf_i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mask = (flat_rr >= float(lo)) & (flat_rr < float(hi))
            if not np.any(mask):
                continue
            for tf_i, tf in enumerate(freq_hz):
                rows.append(
                    {
                        "condition": condition,
                        "label": condition_label(condition),
                        "spatial_frequency_cpd": float(sf_center[sf_i]),
                        "temporal_frequency_hz": float(tf),
                        "source_power_mean": float(source_radial[sf_i]),
                        "motion_q_mean": float(np.mean(q_flat[mask, tf_i])),
                        "modulation_power_mean": float(np.mean(modulation[mask, tf_i])),
                        "n_spatial_coefficients": int(np.sum(mask)),
                        "in_fit_band": bool(FIT_LOW_CPD <= float(sf_center[sf_i]) <= FIT_HIGH_CPD),
                    }
                )
    return rows


def band_summary(source_power2d: np.ndarray, q_by_condition: dict[str, np.ndarray], freq_hz: np.ndarray, rr: np.ndarray) -> list[dict[str, Any]]:
    flat_source = np.asarray(source_power2d, dtype=np.float64).ravel()
    flat_rr = np.asarray(rr, dtype=np.float64).ravel()
    freq = np.asarray(freq_hz, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    totals: dict[tuple[str, str], float] = {}
    for condition, q in q_by_condition.items():
        modulation = flat_source[:, None] * np.asarray(q, dtype=np.float64)
        for band_id, band_label, lo, hi in SPATIAL_BANDS:
            sf_mask = (flat_rr >= float(lo)) & (flat_rr < float(hi))
            if not np.any(sf_mask):
                continue
            power_tf = np.mean(modulation[sf_mask, :], axis=0)
            total = float(np.sum(power_tf))
            totals[(condition, band_id)] = total
            centroid = float(np.sum(freq * power_tf) / max(total, 1e-30))
            row = {
                "condition": condition,
                "label": condition_label(condition),
                "spatial_band": band_id,
                "spatial_band_label": band_label,
                "spatial_low_cpd": float(lo),
                "spatial_high_cpd": float(hi),
                "total_nonzero_tf_power": total,
                "tf_power_centroid_hz": centroid,
            }
            for band_i, (tf_id, tf_label, tf_lo, tf_hi) in enumerate(TEMPORAL_BANDS):
                if band_i == len(TEMPORAL_BANDS) - 1:
                    tf_mask = (freq >= float(tf_lo)) & (freq <= float(tf_hi))
                else:
                    tf_mask = (freq >= float(tf_lo)) & (freq < float(tf_hi))
                band_power = float(np.sum(power_tf[tf_mask]))
                row[f"{tf_id}_power"] = band_power
                row[f"{tf_id}_fraction"] = band_power / total if total > 0 else float("nan")
                row[f"{tf_id}_label"] = tf_label
            rows.append(row)
    baseline = {band_id: totals.get(("all_real_fem", band_id), float("nan")) for band_id, _label, _lo, _hi in SPATIAL_BANDS}
    for row in rows:
        denom = baseline.get(str(row["spatial_band"]), float("nan"))
        row["total_power_over_all_real_same_band"] = float(row["total_nonzero_tf_power"]) / denom if np.isfinite(denom) and denom > 0 else float("nan")
    return rows


def matrix_from_summary(radial_rows: list[dict[str, Any]], condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(radial_rows)
    sub = frame[frame["condition"].eq(condition)].copy()
    sf = np.asarray(sorted(sub["spatial_frequency_cpd"].unique()), dtype=np.float64)
    tf = np.asarray(sorted(sub["temporal_frequency_hz"].unique()), dtype=np.float64)
    mat = np.full((tf.size, sf.size), np.nan, dtype=np.float64)
    sf_index = {float(v): i for i, v in enumerate(sf)}
    tf_index = {float(v): i for i, v in enumerate(tf)}
    for row in sub.itertuples(index=False):
        mat[tf_index[float(row.temporal_frequency_hz)], sf_index[float(row.spatial_frequency_cpd)]] = float(row.modulation_power_mean)
    return sf, tf, mat


def plot_figure(out_dir: Path, radial_rows: list[dict[str, Any]], band_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    radial = pd.DataFrame(radial_rows)
    band = pd.DataFrame(band_rows)
    fig = plt.figure(figsize=(13.0, 7.0))
    gs = fig.add_gridspec(2, 3, wspace=0.38, hspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    heat_conditions = ("short_drift", "mid_drift", "long_drift", "microsaccade", "all_real_fem")
    heat_values = radial[radial["condition"].isin(heat_conditions)]["modulation_power_mean"].to_numpy(dtype=np.float64)
    positive = heat_values[np.isfinite(heat_values) & (heat_values > 0)]
    vmin = float(np.percentile(np.log10(positive), 5)) if positive.size else -12.0
    vmax = float(np.percentile(np.log10(positive), 98)) if positive.size else 0.0

    ax_source = axes[0]
    src = (
        radial[radial["condition"].eq("all_real_fem")]
        .drop_duplicates("spatial_frequency_cpd")
        .sort_values("spatial_frequency_cpd")
    )
    ax_source.plot(src["spatial_frequency_cpd"], src["source_power_mean"], color="#666666", linewidth=2.0)
    ax_source.axvspan(FIT_LOW_CPD, FIT_HIGH_CPD, color="#eeeeee", zorder=-10)
    ax_source.set_xscale("log")
    ax_source.set_yscale("log")
    ax_source.set_title("Static SSI-v3 image spectrum", loc="left", fontsize=10, fontweight="bold")
    ax_source.set_xlabel("spatial frequency (cpd)")
    ax_source.set_ylabel("mean patch power")

    mappable = None
    for ax, condition in zip(axes[1:], heat_conditions, strict=True):
        sf, tf, mat = matrix_from_summary(radial_rows, condition)
        sf_edges = radial_edges(np.asarray([sf[0], sf[-1]], dtype=np.float64), max(len(sf) - 1, 1))
        if sf_edges.size != sf.size + 1:
            mids = np.sqrt(sf[:-1] * sf[1:])
            sf_edges = np.concatenate([[sf[0] ** 2 / mids[0]], mids, [sf[-1] ** 2 / mids[-1]]])
        tf_edges = temporal_edges(tf)
        plot_mat = np.log10(np.maximum(mat, 1e-30))
        mappable = ax.pcolormesh(sf_edges, tf_edges, plot_mat, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.axvspan(FIT_LOW_CPD, FIT_HIGH_CPD, color="white", alpha=0.12, zorder=2)
        ax.set_xscale("log")
        ax.set_ylim(float(tf_edges[0]), min(60.0, float(tf_edges[-1])))
        ax.set_title(condition_label(condition), loc="left", fontsize=10, fontweight="bold", color=condition_color(condition))
        ax.set_xlabel("spatial frequency (cpd)")
        ax.set_ylabel("temporal frequency (Hz)")

        for _, row in band[band["condition"].eq(condition)].iterrows():
            y = float(row["tf_power_centroid_hz"])
            x = math.sqrt(float(row["spatial_low_cpd"]) * float(row["spatial_high_cpd"]))
            ax.scatter([x], [y], s=18, facecolor="none", edgecolor="white", linewidth=0.8)

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes[1:], shrink=0.82, pad=0.015)
        cbar.set_label("log10 modulation power")
    fig.suptitle(
        "SSI-v3 image power times FEM temporal kernel: joint SF-TF retinal modulation",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_dir / "eye_movement_joint_sftf_power.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "eye_movement_joint_sftf_power.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "eye_movement_joint_sftf_power.svg", bbox_inches="tight")
    plt.close(fig)


def db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), 1e-30))


def anchored_db(freq: np.ndarray, values: np.ndarray, anchor_cpd: float = 16.0) -> np.ndarray:
    f = np.asarray(freq, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(f) & np.isfinite(v) & (v > 0)
    if not np.any(keep):
        return db(v)
    idx_candidates = np.where(keep)[0]
    idx = int(idx_candidates[np.argmin(np.abs(f[idx_candidates] - float(anchor_cpd)))])
    return db(v / max(float(v[idx]), 1e-30))


def spatial_edges_from_centers(sf: np.ndarray) -> np.ndarray:
    sf = np.asarray(sf, dtype=np.float64)
    if sf.size == 1:
        return np.asarray([sf[0] / math.sqrt(2.0), sf[0] * math.sqrt(2.0)], dtype=np.float64)
    mids = np.sqrt(sf[:-1] * sf[1:])
    return np.concatenate([[sf[0] ** 2 / mids[0]], mids, [sf[-1] ** 2 / mids[-1]]])


def plot_rucci_style_figure(out_dir: Path, radial_rows: list[dict[str, Any]]) -> None:
    radial = pd.DataFrame(radial_rows)
    sf, tf, all_real = matrix_from_summary(radial_rows, "all_real_fem")
    _sf_bm, _tf_bm, bm = matrix_from_summary(radial_rows, "brownian_matched")
    sf_edges = spatial_edges_from_centers(sf)
    tf_edges = temporal_edges(tf)

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.1))
    ax_a, ax_b, ax_c, ax_d = axs.ravel()

    heat = db(all_real)
    vmin = float(np.nanpercentile(heat, 5))
    vmax = float(np.nanpercentile(heat, 98))
    mesh = ax_a.pcolormesh(sf_edges, tf_edges, heat, shading="auto", cmap="hot", vmin=vmin, vmax=vmax)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_ylim(max(float(tf_edges[0]), 1.5), min(80.0, float(tf_edges[-1])))
    ax_a.set_xlabel("spatial frequency (cycles/deg)")
    ax_a.set_ylabel("temporal frequency (Hz)")
    ax_a.set_title("A", loc="left", fontsize=18, fontweight="bold")
    ax_a.set_xticks([1, 10], ["1", "10"])
    ax_a.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
    cbar = fig.colorbar(mesh, ax=ax_a, shrink=0.78, pad=0.02)
    cbar.set_label("spectral density (dB)")

    sf_targets = (0.5, 3.0, 7.0)
    sf_colors = ("#2d3a9d", "#1e8acb", "#22c8bd")
    for target, color in zip(sf_targets, sf_colors, strict=True):
        idx = int(np.argmin(np.abs(sf - target)))
        ax_b.plot(tf, db(all_real[:, idx]), color=color, linewidth=2.0, label=f"{sf[idx]:.2g} c/d")
        ax_b.plot(tf, db(bm[:, idx]), color=color, linewidth=1.4, linestyle="--", alpha=0.88, label=f"{sf[idx]:.2g} c/d (BM)")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("temporal frequency (Hz)")
    ax_b.set_ylabel("spectral density (dB)")
    ax_b.set_title("B", loc="left", fontsize=18, fontweight="bold")
    ax_b.legend(fontsize=7, frameon=False)

    tf_targets = (4.0, 8.0, 16.0)
    tf_colors = ("#3f4ab8", "#1e8acb", "#22c8bd")
    for target, color in zip(tf_targets, tf_colors, strict=True):
        idx = int(np.argmin(np.abs(tf - target)))
        ax_c.plot(sf, db(all_real[idx, :]), color=color, linewidth=2.0, label=f"{tf[idx]:.0f} Hz")
        ax_c.plot(sf, db(bm[idx, :]), color=color, linewidth=1.4, linestyle="--", alpha=0.88, label=f"{tf[idx]:.0f} Hz (BM)")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("spatial frequency (cycles/deg)")
    ax_c.set_ylabel("spectral density (dB)")
    ax_c.set_title("C", loc="left", fontsize=18, fontweight="bold")
    ax_c.legend(fontsize=7, frameon=False)

    source = (
        radial[radial["condition"].eq("all_real_fem")]
        .drop_duplicates("spatial_frequency_cpd")
        .sort_values("spatial_frequency_cpd")
    )
    ax_d.plot(
        source["spatial_frequency_cpd"],
        anchored_db(
            source["spatial_frequency_cpd"].to_numpy(dtype=np.float64),
            source["source_power_mean"].to_numpy(dtype=np.float64),
        ),
        color="black",
        linewidth=2.2,
        label="natural images",
    )
    for condition, style in (
        ("all_real_fem", "-"),
        ("drift_only", "-"),
        ("brownian_matched", "--"),
    ):
        sub = radial[radial["condition"].eq(condition)]
        grouped = sub.groupby("spatial_frequency_cpd", sort=True)["modulation_power_mean"].sum()
        freq = grouped.index.to_numpy(dtype=np.float64)
        ax_d.plot(
            freq,
            anchored_db(freq, grouped.to_numpy(dtype=np.float64)),
            color=condition_color(condition),
            linestyle=style,
            linewidth=2.0 if style == "-" else 1.7,
            label=condition_label(condition),
        )
    ax_d.set_xscale("log")
    ax_d.set_xlabel("spatial frequency (cycles/deg)")
    ax_d.set_ylabel("relative density (dB, 16 cpd aligned)")
    ax_d.set_title("D", loc="left", fontsize=18, fontweight="bold")
    ax_d.legend(fontsize=7, frameon=False)

    for ax in axs.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Rucci-style SSI-v3 retinal modulation spectra", x=0.02, y=0.99, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(out_dir / "eye_movement_joint_sftf_rucci_style.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "eye_movement_joint_sftf_rucci_style.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "eye_movement_joint_sftf_rucci_style.svg", bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path, manifest: dict[str, Any], band_rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(band_rows)
    lines = [
        "# Eye-Movement Joint SF-TF Power",
        "",
        "This diagnostic starts from the banked SSI figure-v3 image windows and fixation trajectories.",
        "",
        f"- image bank: `{Path(manifest['image_table']).relative_to(ROOT)}`",
        f"- trajectory bank: `{Path(manifest['trace_xy_npy']).relative_to(ROOT)}`",
        f"- trace metadata: `{Path(manifest['trace_table']).relative_to(ROOT)}`",
        "",
        "The calculation uses the Rucci-style factorization `P_movie(k, f_t) ~= P_image(k) Q_trace(k, f_t)`, where `Q_trace` is the temporal spectrum of the Fourier phase modulation induced by each trace. The zero temporal-frequency component is removed before summarizing modulation power.",
        "",
        "## Band Summary",
        "",
    ]
    for band_id, band_label, _lo, _hi in SPATIAL_BANDS:
        lines.append(f"### {band_label}")
        sub = frame[frame["spatial_band"].eq(band_id)].copy()
        for condition in PLOT_CONDITIONS:
            row = sub[sub["condition"].eq(condition)]
            if row.empty:
                continue
            item = row.iloc[0]
            lines.append(
                f"- `{condition_label(condition)}`: total/all-real `{float(item['total_power_over_all_real_same_band']):.3g}`, "
                f"TF centroid `{float(item['tf_power_centroid_hz']):.3g}` Hz, "
                f"fast-TF fraction `{float(item['fast_tf_fraction']):.3g}`."
            )
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_table = pd.read_csv(args.image_table)
    trace_table = pd.read_csv(args.trace_table)
    traces = np.load(args.trace_xy, mmap_mode="r")
    trace_groups, trace_group_rows = select_trace_groups(
        trace_table,
        max_traces_per_condition=int(args.max_traces_per_condition),
        seed=int(args.seed),
    )
    trace_groups, trace_group_rows = add_reference_trace_groups(
        trace_groups,
        trace_group_rows,
        trace_table,
        traces,
        max_traces_per_condition=int(args.max_traces_per_condition),
        seed=int(args.seed),
    )
    source_powers, image_rows, _example_patch = load_image_powers(
        image_table,
        max_images=int(args.max_images),
        patch_size_px=int(args.patch_size_px),
    )
    ppd = float(np.median([row["ppd"] for row in image_rows]))
    fx, fy, rr = frequency_grid(int(args.patch_size_px), ppd)
    edges = radial_edges(rr, int(args.n_radial_bins))
    source_power2d = np.mean(np.asarray(source_powers, dtype=np.float64), axis=0)

    q_by_condition: dict[str, np.ndarray] = {}
    freq_hz: np.ndarray | None = None
    for condition, trace_source in trace_groups.items():
        if isinstance(trace_source, tuple):
            selected_traces = np.asarray(trace_source[1], dtype=np.float32)
        else:
            selected_traces = np.asarray(traces[np.asarray(trace_source, dtype=int)], dtype=np.float32)
        print(f"[q] {condition}: {selected_traces.shape[0]} traces", flush=True)
        cond_freq, q = motion_q_spectrum(
            selected_traces,
            fx,
            fy,
            dt_s=float(args.dt_s),
            chunk_size=int(args.trace_chunk_size),
        )
        if freq_hz is None:
            freq_hz = cond_freq
        elif not np.allclose(freq_hz, cond_freq):
            raise RuntimeError("Temporal frequency support changed across conditions.")
        q_by_condition[condition] = q
    if freq_hz is None:
        raise RuntimeError("No trace groups available.")

    radial_rows = radial_temporal_summary(source_power2d, q_by_condition, freq_hz, rr, edges)
    band_rows = band_summary(source_power2d, q_by_condition, freq_hz, rr)
    write_csv(out_dir / "eye_movement_joint_sftf_radial_temporal_summary.csv", radial_rows)
    write_csv(out_dir / "eye_movement_joint_sftf_population_band_summary.csv", band_rows)
    write_csv(out_dir / "eye_movement_joint_sftf_trace_groups.csv", trace_group_rows)
    write_csv(out_dir / "eye_movement_joint_sftf_image_sample.csv", image_rows)

    manifest = {
        "analysis": "eye_movement_joint_sftf_power",
        "image_table": Path(args.image_table),
        "trace_table": Path(args.trace_table),
        "trace_xy_npy": Path(args.trace_xy),
        "out_dir": out_dir,
        "n_images_available": int(image_table.shape[0]),
        "n_images_selected": int(len(image_rows)),
        "n_traces_available": int(traces.shape[0]),
        "trace_groups": trace_group_rows,
        "patch_size_px": int(args.patch_size_px),
        "ppd_median": ppd,
        "dt_s": float(args.dt_s),
        "temporal_frequencies_hz": freq_hz,
        "spatial_bands": [
            {"id": band_id, "label": label, "low_cpd": lo, "high_cpd": hi}
            for band_id, label, lo, hi in SPATIAL_BANDS
        ],
        "temporal_bands": [
            {"id": band_id, "label": label, "low_hz": lo, "high_hz": hi}
            for band_id, label, lo, hi in TEMPORAL_BANDS
        ],
        "method": "Rucci-style Fourier phase modulation kernel Q(k,ft), multiplied by mean SSI-v3 static image power; nonzero temporal frequencies only.",
    }
    write_json(out_dir / "eye_movement_joint_sftf_manifest.json", manifest)
    write_readme(out_dir, manifest, band_rows)
    plot_figure(out_dir, radial_rows, band_rows)
    plot_rucci_style_figure(out_dir, radial_rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_IMAGE_TABLE)
    parser.add_argument("--trace-table", type=Path, default=DEFAULT_TRACE_TABLE)
    parser.add_argument("--trace-xy", type=Path, default=DEFAULT_TRACE_XY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--max-traces-per-condition", type=int, default=192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patch-size-px", type=int, default=PATCH_SIZE_PX)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--n-radial-bins", type=int, default=17)
    parser.add_argument("--trace-chunk-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(f"Wrote joint SF-TF diagnostic to {manifest['out_dir']}")


if __name__ == "__main__":
    main()
