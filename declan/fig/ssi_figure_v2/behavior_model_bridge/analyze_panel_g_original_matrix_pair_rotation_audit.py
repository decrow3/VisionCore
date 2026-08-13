#!/usr/bin/env python3
"""Map-first readout of the frozen original-matrix Panel G rotation audit."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge
from declan.fig.ssi_figure_v2.behavior_model_bridge.prepare_panel_g_original_matrix_pair_audit import (
    _predict_from_covariance,
)


AUDIT_ROOT = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_original_matrix_pair_rotation_audit_v1"
)
RUN_ROOT = AUDIT_ROOT / "fresh_direct_rotation_n32_gpu0"
BANK_ROOT = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
OUT_ROOT = RUN_ROOT / "checkpoint_results_map_first"
CORRECTED = "high_sf_aligned_corrected_array_axis"
HISTORICAL = "high_sf_aligned_historical_gaze_axis"
MAP_FRAMES = (9, 19, 29, 39)


def _map_ssi(rate_map: np.ndarray) -> float:
    flat = np.maximum(np.asarray(rate_map, dtype=np.float64), 0.0).reshape(-1)
    mean = float(np.mean(flat))
    gain = flat / (mean + 1e-12)
    return float(np.mean(gain * np.log2(gain + 1e-12)))


def _surrogate_curves() -> dict[str, pd.DataFrame]:
    model_values = pd.read_csv(bridge.MODEL_VALUES_CSV)
    return {
        component: bridge._curve_for(
            model_values,
            population_key="high_sf_aligned",
            metric_family="component_rms",
            component=component,
        )
        for component in ("along", "across")
    }


def _surrogate_at_angles(
    row: pd.Series,
    trace: np.ndarray,
    angles_deg: np.ndarray,
    curves: dict[str, pd.DataFrame],
) -> np.ndarray:
    centered = np.asarray(trace, dtype=np.float64)
    centered -= np.mean(centered, axis=0, keepdims=True)
    axis = math.radians(float(row["image_edge_axis_gaze_deg"]))
    parallel = np.asarray([math.cos(axis), math.sin(axis)])
    normal = np.asarray([-math.sin(axis), math.cos(axis)])
    p = centered @ parallel
    n = centered @ normal
    prediction, _parallel, _normal = _predict_from_covariance(
        np.asarray([np.mean(p * p)]),
        np.asarray([np.mean(n * n)]),
        np.asarray([np.mean(p * n)]),
        np.radians(angles_deg),
        curves,
    )
    return prediction[0]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 3 or np.nanstd(x[finite]) <= 0 or np.nanstd(y[finite]) <= 0:
        return np.nan
    return float(spearmanr(x[finite], y[finite]).statistic)


def build_comparison_table(
    selection: pd.DataFrame,
    conditions: pd.DataFrame,
    contrasts: pd.DataFrame,
    traces: np.ndarray,
    surrogate_curves: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for selected in selection.itertuples(index=False):
        index = int(selected.selection_index)
        row = pd.Series(selected._asdict())
        corrected = conditions[
            conditions["selection_index"].astype(int).eq(index)
            & conditions["population"].astype(str).eq(CORRECTED)
        ].sort_values("condition_index")
        historical = conditions[
            conditions["selection_index"].astype(int).eq(index)
            & conditions["population"].astype(str).eq(HISTORICAL)
        ].sort_values("condition_index")
        if len(corrected) != 33 or len(historical) != 33:
            raise RuntimeError(f"Incomplete rotation curve for selection {index}")
        angles = corrected["rotation_angle_deg"].to_numpy(dtype=float, copy=True)
        angles[0] = 0.0
        surrogate = _surrogate_at_angles(row, traces[int(selected.trace_index)], angles, surrogate_curves)
        direct_corrected = corrected["bits_per_spike"].to_numpy(dtype=float)
        direct_historical = historical["bits_per_spike"].to_numpy(dtype=float)
        corrected_contrast = contrasts[
            contrasts["selection_index"].astype(int).eq(index)
            & contrasts["population"].astype(str).eq(CORRECTED)
        ].iloc[0]
        historical_contrast = contrasts[
            contrasts["selection_index"].astype(int).eq(index)
            & contrasts["population"].astype(str).eq(HISTORICAL)
        ].iloc[0]
        antipodal_mae = float(np.mean(np.abs(direct_corrected[1:17] - direct_corrected[17:33])))
        rotation_range = float(np.ptp(direct_corrected[1:]))
        rows.append(
            {
                "selection_index": index,
                "selection_role": str(selected.selection_role),
                "image_index": int(selected.image_index),
                "trace_index": int(selected.trace_index),
                "image_orientation_coherence": float(selected.image_orientation_coherence),
                "surrogate_match_advantage_percent_points": float(selected.surrogate_match_advantage_percent_points),
                "surrogate_both_components_rotation_finite_fraction": float(selected.surrogate_both_components_rotation_finite_fraction),
                "surrogate_support_clean": bool(float(selected.surrogate_both_components_rotation_finite_fraction) >= 1.0 - 1e-12),
                "within_image_calibration_spearman": float(selected.within_image_calibration_spearman),
                "corrected_direct_match_advantage_bits_per_spike": float(corrected_contrast.real_minus_rotation_bits_per_spike),
                "historical_direct_match_advantage_bits_per_spike": float(historical_contrast.real_minus_rotation_bits_per_spike),
                "corrected_direct_information_effect_bits_per_sample": float(corrected_contrast.real_minus_rotation_information_bits_per_sample),
                "corrected_direct_expected_spike_effect_per_sample": float(corrected_contrast.real_minus_rotation_expected_spikes_per_sample),
                "corrected_direct_fraction_rotations_below_real": float(corrected_contrast.fraction_rotations_below_real_bits_per_spike),
                "surrogate_vs_corrected_rotation_spearman": _safe_spearman(surrogate[1:], direct_corrected[1:]),
                "surrogate_vs_historical_rotation_spearman": _safe_spearman(surrogate[1:], direct_historical[1:]),
                "corrected_antipodal_mae_bits_per_spike": antipodal_mae,
                "corrected_rotation_range_bits_per_spike": rotation_range,
                "corrected_antipodal_mae_fraction_of_range": antipodal_mae / max(rotation_range, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def plot_rotation_curves(
    comparison: pd.DataFrame,
    selection: pd.DataFrame,
    conditions: pd.DataFrame,
    traces: np.ndarray,
    surrogate_curves: dict[str, pd.DataFrame],
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15.2, 14.5), constrained_layout=True)
    for ax, selected in zip(axes.ravel(), selection.itertuples(index=False)):
        index = int(selected.selection_index)
        selected_series = pd.Series(selected._asdict())
        corrected = conditions[
            conditions["selection_index"].astype(int).eq(index)
            & conditions["population"].astype(str).eq(CORRECTED)
        ].sort_values("condition_index")
        historical = conditions[
            conditions["selection_index"].astype(int).eq(index)
            & conditions["population"].astype(str).eq(HISTORICAL)
        ].sort_values("condition_index")
        angles = corrected["rotation_angle_deg"].to_numpy(dtype=float, copy=True)
        angles[0] = 0.0
        surrogate = _surrogate_at_angles(
            selected_series, traces[int(selected.trace_index)], angles, surrogate_curves
        )
        surrogate_centered = surrogate - np.nanmean(surrogate[1:])
        corrected_bits = corrected["bits_per_spike"].to_numpy(dtype=float)
        historical_bits = historical["bits_per_spike"].to_numpy(dtype=float)
        corrected_centered = corrected_bits - np.mean(corrected_bits[1:])
        historical_centered = historical_bits - np.mean(historical_bits[1:])
        order = np.argsort(angles)
        ax.plot(angles[order], surrogate_centered[order], color="#b4492d", lw=1.5, label="historical surrogate")
        ax.scatter([0], [surrogate_centered[0]], color="#b4492d", s=22, zorder=4)
        ax.axhline(0, color="0.78", lw=0.7)
        ax.axvline(180, color="0.88", lw=0.6)
        ax.set_xlim(-4, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("trace rotation from recorded (deg)")
        ax.set_ylabel("surrogate − rotation mean (pp)", color="#b4492d")
        ax.tick_params(axis="y", colors="#b4492d")
        twin = ax.twinx()
        twin.plot(angles[order], corrected_centered[order], color="#2878b5", lw=1.45, marker="o", ms=2.2, label="direct corrected mask")
        twin.plot(angles[order], historical_centered[order], color="0.35", lw=1.0, alpha=0.75, label="direct historical mask")
        twin.scatter([0], [corrected_centered[0]], color="#2878b5", s=22, zorder=4)
        twin.set_ylabel("direct SSI − rotation mean (bits/spike)", color="#2878b5")
        twin.tick_params(axis="y", colors="#2878b5")
        result = comparison[comparison["selection_index"].astype(int).eq(index)].iloc[0]
        support = "clean support" if bool(result.surrogate_support_clean) else f"joint support={float(result.surrogate_both_components_rotation_finite_fraction):.2f}"
        ax.set_title(
            f"{index:02d} {str(selected.selection_role).replace('_', ' ')}\n"
            f"coh={float(selected.image_orientation_coherence):.3f}; {support}; "
            f"curve ρ={float(result.surrogate_vs_corrected_rotation_spearman):+.2f}\n"
            f"Δ surrogate={float(selected.surrogate_match_advantage_percent_points):+.2f} pp; "
            f"direct corr={float(result.corrected_direct_match_advantage_bits_per_spike):+.4f}, "
            f"hist={float(result.historical_direct_match_advantage_bits_per_spike):+.4f}",
            fontsize=8.2,
        )
        if index == 0:
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = twin.get_legend_handles_labels()
            ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=7.0, loc="upper right")
    fig.suptitle(
        "Frozen original-matrix pairs: historical marginal surrogate versus fresh direct RR100 rotation curves\n"
        "Every curve is centered on its own 32-rotation mean; recorded trajectory is the point at 0°",
        fontsize=12, weight="bold",
    )
    fig.savefig(OUT_ROOT / "checkpoint_direct_vs_surrogate_rotation_curves_all12.png", dpi=210)
    fig.savefig(OUT_ROOT / "checkpoint_direct_vs_surrogate_rotation_curves_all12.pdf")
    plt.close(fig)


def _plot_input(ax: plt.Axes, selected: pd.Series, trace: np.ndarray) -> None:
    aperture_path = AUDIT_ROOT / "input_aperture_cache" / f"selection_{int(selected.selection_index):02d}.npz"
    with np.load(aperture_path) as z:
        aperture = np.asarray(z["aperture"])
        ppd = float(z["ppd"][0])
    centered = np.asarray(trace, dtype=float) - np.mean(trace, axis=0, keepdims=True)
    center = np.asarray([aperture.shape[1] / 2.0, aperture.shape[0] / 2.0])
    screen = np.column_stack([center[0] + centered[:, 0] * ppd, center[1] - centered[:, 1] * ppd])
    ax.imshow(aperture, cmap="gray", origin="upper")
    ax.plot(screen[:, 0], screen[:, 1], color="#f28e2b", lw=0.9)
    ax.scatter(screen[0, 0], screen[0, 1], color="#2ca02c", s=10)
    angle = math.radians(float(selected.image_edge_axis_array_deg))
    delta = np.asarray([math.cos(angle), math.sin(angle)]) * 0.36 * ppd
    ax.plot(
        [center[0] - delta[0], center[0] + delta[0]],
        [center[1] - delta[1], center[1] + delta[1]],
        color="#00bcd4", lw=1.4,
    )
    ax.set_xlim(0, aperture.shape[1] - 1)
    ax.set_ylim(aperture.shape[0] - 1, 0)
    ax.set_xticks([]); ax.set_yticks([])


def _corrected_map_unit(index: int, map_selection: pd.DataFrame) -> int:
    return int(
        map_selection[
            map_selection["selection_index"].astype(int).eq(index)
            & map_selection["map_selection_role"].astype(str).eq(CORRECTED)
        ].iloc[0]["unit_index"]
    )


def build_map_figure(
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    map_selection: pd.DataFrame,
    traces: np.ndarray,
    frame_index: int,
) -> plt.Figure:
    fig, axes = plt.subplots(len(selection), 4, figsize=(11.8, 2.05 * len(selection)))
    for row_index, selected_tuple in enumerate(selection.itertuples(index=False)):
        selected = pd.Series(selected_tuple._asdict())
        index = int(selected.selection_index)
        cache_path = RUN_ROOT / "cache" / f"selection_{index:02d}.npz"
        with np.load(cache_path) as z:
            map_units = z["selected_map_unit_index"].astype(int)
            unit = _corrected_map_unit(index, map_selection)
            unit_pos = int(np.flatnonzero(map_units == unit)[0])
            maps = z["selected_map"][:, frame_index, unit_pos].astype(float)
            frame_bits = z["selected_unit_frame_bits_per_spike"][:, frame_index, unit_pos].astype(float)
            frame_rates = z["selected_unit_frame_mean_rate"][:, frame_index, unit_pos].astype(float)
        real, rotation_mean_map = maps[0], maps[1]
        difference = real - rotation_mean_map
        absolute = np.concatenate([real.ravel(), rotation_mean_map.ravel()])
        vmin, vmax = np.percentile(absolute, [1.0, 99.0])
        diff_lim = max(float(np.percentile(np.abs(difference), 99.0)), 1e-12)
        _plot_input(axes[row_index, 0], selected, traces[int(selected.trace_index)])
        axes[row_index, 1].imshow(real, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        axes[row_index, 2].imshow(rotation_mean_map, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        axes[row_index, 3].imshow(difference, cmap="coolwarm", origin="lower", vmin=-diff_lim, vmax=diff_lim)
        for column in range(1, 4):
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
        direct = comparison[comparison["selection_index"].astype(int).eq(index)].iloc[0]
        axes[row_index, 0].set_ylabel(
            f"{index:02d}  u{unit:03d}\n"
            f"coh={float(selected.image_orientation_coherence):.2f}\n"
            f"sur={float(selected.surrogate_match_advantage_percent_points):+.2f} pp\n"
            f"direct={float(direct.corrected_direct_match_advantage_bits_per_spike):+.4f}",
            rotation=0, ha="right", va="center", fontsize=7.2, labelpad=8,
        )
        axes[row_index, 1].text(
            0.02, 0.02,
            f"inst SSI={frame_bits[0]:.3f}\nrate={frame_rates[0]:.2f}\nmean-map SSI={_map_ssi(real):.3f}",
            transform=axes[row_index, 1].transAxes, color="white", fontsize=6.0,
            bbox={"facecolor": "black", "alpha": 0.46, "edgecolor": "none", "pad": 1.1},
        )
        axes[row_index, 2].text(
            0.02, 0.02,
            f"mean inst SSI={np.mean(frame_bits[1:]):.3f}\nmean rate={np.mean(frame_rates[1:]):.2f}\nmean-map SSI={_map_ssi(rotation_mean_map):.3f}",
            transform=axes[row_index, 2].transAxes, color="white", fontsize=6.0,
            bbox={"facecolor": "black", "alpha": 0.46, "edgecolor": "none", "pad": 1.1},
        )
        if row_index == 0:
            for column, title in enumerate(("1° input + trace", "recorded response", "rotation-mean response", "recorded − rotation mean")):
                axes[row_index, column].set_title(title, fontsize=8.5, weight="bold")
    fig.suptitle(
        f"Corrected locally aligned high-SF unit maps — fixed frame {frame_index + 1}/40\n"
        "Units selected from tuning metadata before fresh outcomes; response panels share a scale within each row",
        fontsize=11, weight="bold", y=0.998,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.965, bottom=0.015, hspace=0.17, wspace=0.05)
    return fig


def plot_maps(
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    map_selection: pd.DataFrame,
    traces: np.ndarray,
) -> None:
    compact = build_map_figure(selection, comparison, map_selection, traces, 19)
    compact.savefig(OUT_ROOT / "checkpoint_corrected_aligned_maps_all12_frame20.png", dpi=210)
    plt.close(compact)
    with PdfPages(OUT_ROOT / "checkpoint_corrected_aligned_maps_all12_fixed_frames.pdf") as pdf:
        for frame in MAP_FRAMES:
            fig = build_map_figure(selection, comparison, map_selection, traces, frame)
            pdf.savefig(fig)
            plt.close(fig)


def plot_effect_summary(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    clean = comparison["surrogate_support_clean"].astype(bool).to_numpy()
    colors = np.where(clean, "#2878b5", "#e68632")
    for ax, column, title in (
        (axes[0], "corrected_direct_match_advantage_bits_per_spike", "corrected array-axis population"),
        (axes[1], "historical_direct_match_advantage_bits_per_spike", "historical gaze-axis population"),
    ):
        x = comparison["surrogate_match_advantage_percent_points"].to_numpy(dtype=float)
        y = comparison[column].to_numpy(dtype=float)
        ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)
        ax.scatter(x, y, c=colors, s=48, edgecolor="white", linewidth=0.5)
        for row in comparison.itertuples(index=False):
            ax.annotate(str(int(row.selection_index)), (float(row.surrogate_match_advantage_percent_points), float(getattr(row, column))), xytext=(4, 3), textcoords="offset points", fontsize=7)
        rho = _safe_spearman(x, y)
        ax.set_title(f"{title}\nSpearman ρ={rho:+.2f}")
        ax.set_xlabel("historical surrogate match advantage (percentage points)")
        ax.set_ylabel("fresh direct match advantage (bits/spike)")
    fig.suptitle(
        "Targeted frozen examples only — blue: full two-component surrogate support; orange: support switching",
        fontsize=10.5, weight="bold",
    )
    fig.savefig(OUT_ROOT / "checkpoint_surrogate_vs_direct_match_advantage.png", dpi=220)
    fig.savefig(OUT_ROOT / "checkpoint_surrogate_vs_direct_match_advantage.pdf")
    plt.close(fig)


def plot_metric_decomposition(contrasts: pd.DataFrame, comparison: pd.DataFrame) -> None:
    work = contrasts[contrasts["population"].astype(str).eq(CORRECTED)].sort_values("selection_index").copy()
    metrics = (
        ("bits_per_spike", "SSI efficiency", "#2878b5"),
        ("information_bits_per_sample", "information/sample", "#6a4c93"),
        ("expected_spikes_per_sample", "expected spikes/sample", "#e68632"),
    )
    x = np.arange(len(work), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(12.4, 5.2), constrained_layout=True)
    for metric_index, (metric, label, color) in enumerate(metrics):
        real = work[f"real_{metric}"].to_numpy(dtype=float)
        rotation = work[f"rotation_mean_{metric}"].to_numpy(dtype=float)
        percent = 100.0 * (real - rotation) / np.maximum(np.abs(rotation), 1e-12)
        ax.bar(x + (metric_index - 1) * width, percent, width=width, color=color, label=label)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x)
    labels = []
    clean_lookup = comparison.set_index("selection_index")["surrogate_support_clean"].to_dict()
    for index in work["selection_index"].astype(int):
        labels.append(f"{index}\n{'clean' if clean_lookup[index] else 'switch'}")
    ax.set_xticklabels(labels)
    ax.set_xlabel("frozen selection and surrogate-support status")
    ax.set_ylabel("recorded − rotation mean (% of rotation mean)")
    ax.legend(frameon=False, ncol=3)
    ax.set_title(
        "Corrected aligned high-SF population: direct metric decomposition\n"
        "Positive bits/spike can reflect preserved information with fewer expected spikes",
        weight="bold",
    )
    fig.savefig(OUT_ROOT / "checkpoint_corrected_direct_metric_decomposition.png", dpi=220)
    fig.savefig(OUT_ROOT / "checkpoint_corrected_direct_metric_decomposition.pdf")
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(RUN_ROOT / "frozen_pair_selection_used.csv").sort_values("selection_index").reset_index(drop=True)
    conditions = pd.read_csv(RUN_ROOT / "direct_pair_population_rotation_curves.csv")
    contrasts = pd.read_csv(RUN_ROOT / "direct_pair_rotation_contrasts.csv")
    map_selection = pd.read_csv(RUN_ROOT / "direct_selected_map_units.csv")
    traces = np.load(BANK_ROOT / "trace_xy.npy")
    curves = _surrogate_curves()
    comparison = build_comparison_table(selection, conditions, contrasts, traces, curves)
    comparison.to_csv(OUT_ROOT / "direct_surrogate_pair_comparison.csv", index=False)
    plot_rotation_curves(comparison, selection, conditions, traces, curves)
    plot_maps(selection, comparison, map_selection, traces)
    plot_effect_summary(comparison)
    plot_metric_decomposition(contrasts, comparison)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
