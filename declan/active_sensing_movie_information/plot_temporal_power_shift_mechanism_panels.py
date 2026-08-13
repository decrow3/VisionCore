#!/usr/bin/env python3
"""Mechanism-panel draft for the temporal power-shift SSI analysis.

The figure separates the proposed upstream linear power drive from the
downstream nonlinear activation-map effects. It is a map-first checkpoint,
not a final manuscript layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
DEFAULT_SCORECARD_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_05_linear_power_scorecard_v1"
DEFAULT_EXPLANATION_DIR = DEFAULT_RUN_DIR / "sftf_power_explanation_normal_first"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_06_mechanism_panels_v1"
DEFAULT_EXAMPLE = "example_largest_positive_delta"
DEFAULT_UNIT = 8

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "grey": "#777777",
    "black": "#222222",
}
GROUP_ORDER = ["low_sf", "middle_sf", "high_sf"]
GROUP_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
GROUP_COLORS = {"low_sf": OKABE_ITO["blue"], "middle_sf": OKABE_ITO["green"], "high_sf": OKABE_ITO["orange"]}
SF_BANDS = (
    ("image_power_0_2_cpd_fraction", 0.0, 2.0, "0-2 cpd"),
    ("image_power_2_4_cpd_fraction", 2.0, 4.0, "2-4 cpd"),
    ("image_power_4_8_cpd_fraction", 4.0, 8.0, "4-8 cpd"),
    ("image_power_8plus_cpd_fraction", 8.0, np.inf, "8+ cpd"),
)
TF_MATCH_SIGMA_OCTAVES = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--scorecard-dir", type=Path, default=DEFAULT_SCORECARD_DIR)
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--example", type=str, default=DEFAULT_EXAMPLE)
    parser.add_argument("--unit-index", type=int, default=DEFAULT_UNIT)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def sf_band_for_unit(preferred_sf_cpd: float) -> tuple[str, str]:
    for column, lo, hi, label in SF_BANDS:
        if np.isfinite(preferred_sf_cpd) and preferred_sf_cpd >= lo and preferred_sf_cpd < hi:
            return column, label
    return SF_BANDS[-1][0], SF_BANDS[-1][3]


def unit_display_label(unit: pd.Series) -> str:
    return f"{unit['unit_label']} ({GROUP_LABELS.get(str(unit['sf_group']), str(unit['sf_group']))})"


def tf_match(projected_tf_hz: np.ndarray, pref_tf_hz: float) -> np.ndarray:
    projected = np.asarray(projected_tf_hz, dtype=float)
    out = np.zeros_like(projected, dtype=float)
    valid = np.isfinite(projected) & (projected > 0.0) & np.isfinite(pref_tf_hz) & (pref_tf_hz > 0.0)
    log_distance = np.zeros_like(projected, dtype=float)
    log_distance[valid] = np.log2(projected[valid] / pref_tf_hz)
    out[valid] = np.exp(-0.5 * (log_distance[valid] / TF_MATCH_SIGMA_OCTAVES) ** 2)
    return out


def linear_drive_timecourses(framewise: pd.DataFrame, units: pd.DataFrame, image_row: pd.Series) -> pd.DataFrame:
    rows = []
    contrast2 = float(image_row["image_patch_rms_contrast"]) ** 2
    for _, unit in units.iterrows():
        label = str(unit["unit_label"])
        induced_col = f"{label}_tf_landing_hz"
        if induced_col not in framewise:
            continue
        band_col, band_label = sf_band_for_unit(float(unit["preferred_sf_cpd"]))
        sf_power_fraction = float(image_row[band_col])
        sf_power_abs = contrast2 * sf_power_fraction
        induced_tf = framewise[induced_col].to_numpy(dtype=float)
        match = tf_match(induced_tf, float(unit["dense_fit_pref_tf_hz"]))
        drive = sf_power_abs * match
        for frame_idx, time_ms, tf_value, match_value, drive_value in zip(
            framewise["frame_index"],
            framewise["time_ms"],
            induced_tf,
            match,
            drive,
            strict=True,
        ):
            rows.append(
                {
                    "frame_index": int(frame_idx),
                    "time_ms": float(time_ms),
                    "unit_index": int(unit["unit_index"]),
                    "unit_label": label,
                    "sf_group": str(unit["sf_group"]),
                    "preferred_sf_cpd": float(unit["preferred_sf_cpd"]),
                    "preferred_tf_hz": float(unit["dense_fit_pref_tf_hz"]),
                    "sf_power_band": band_label,
                    "sf_power_fraction": sf_power_fraction,
                    "sf_power_abs": sf_power_abs,
                    "motion_induced_tf_hz": float(tf_value),
                    "tf_match": float(match_value),
                    "linear_power_drive": float(drive_value),
                }
            )
    return pd.DataFrame(rows)


def plot_speed_panel(ax: plt.Axes, framewise: pd.DataFrame) -> None:
    add_panel_label(ax, "A")
    ax.plot(framewise["time_ms"], framewise["normal_speed_deg_s"], color=OKABE_ITO["black"], lw=2.0, label="normal motion")
    ax.plot(framewise["time_ms"], framewise["stabilized_speed_deg_s"], color=OKABE_ITO["grey"], lw=2.0, label="stabilized")
    ax.set_title("Retinal motion", fontsize=12)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("speed (deg/s)")
    ax.set_ylim(-0.5, float(framewise["normal_speed_deg_s"].max()) * 1.12)
    ax.grid(True, color="#e8e8e8", lw=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_tf_landing_panel(ax: plt.Axes, framewise: pd.DataFrame, units: pd.DataFrame) -> None:
    add_panel_label(ax, "B")
    for _, unit in units.iterrows():
        label = str(unit["unit_label"])
        group = str(unit["sf_group"])
        color = GROUP_COLORS.get(group, OKABE_ITO["grey"])
        landing_col = f"{label}_tf_landing_hz"
        pref_col = f"{label}_dense_fit_pref_tf_hz"
        if landing_col not in framewise:
            continue
        landing = np.maximum(framewise[landing_col].to_numpy(dtype=float), 0.05)
        ax.plot(framewise["time_ms"], landing, color=color, lw=1.8, label=unit_display_label(unit))
        pref = float(framewise[pref_col].iloc[0]) if pref_col in framewise else float(unit["dense_fit_pref_tf_hz"])
        ax.axhline(pref, color=color, lw=1.1, ls="--", alpha=0.75)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 80.0)
    ax.set_title("Motion-induced TF = speed x preferred SF", fontsize=12)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("temporal frequency (Hz)")
    ax.grid(True, which="both", color="#e8e8e8", lw=0.7)
    ax.text(0.03, 0.07, "dashed = preferred TF", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def plot_tf_match_panel(ax: plt.Axes, drive_rows: pd.DataFrame) -> None:
    add_panel_label(ax, "C")
    plot_floor = 1e-8
    for _, unit_rows in drive_rows.groupby("unit_label", sort=False):
        group = str(unit_rows["sf_group"].iloc[0])
        color = GROUP_COLORS.get(group, OKABE_ITO["grey"])
        label = f"{unit_rows['unit_label'].iloc[0]} ({GROUP_LABELS.get(group, group)})"
        visible_match = np.maximum(unit_rows["tf_match"].to_numpy(dtype=float), plot_floor)
        ax.plot(unit_rows["time_ms"], visible_match, color=color, lw=1.9, label=label)
    ax.set_yscale("log")
    ax.set_ylim(plot_floor, 1.2)
    ax.set_title("TF match")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("match")
    ax.grid(True, which="both", color="#e8e8e8", lw=0.7)
    ax.text(0.03, 0.07, "log scale; floor used only for display", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)


def plot_linear_drive_panel(ax: plt.Axes, drive_rows: pd.DataFrame) -> None:
    add_panel_label(ax, "D")
    max_drive = 0.0
    for _, unit_rows in drive_rows.groupby("unit_label", sort=False):
        group = str(unit_rows["sf_group"].iloc[0])
        color = GROUP_COLORS.get(group, OKABE_ITO["grey"])
        label = f"{unit_rows['unit_label'].iloc[0]} ({GROUP_LABELS.get(group, group)})"
        ax.plot(unit_rows["time_ms"], unit_rows["linear_power_drive"], color=color, lw=1.9, label=label)
        max_drive = max(max_drive, float(unit_rows["linear_power_drive"].max()))
    sf_bands = sorted(set(str(value) for value in drive_rows["sf_power_band"]))
    ax.set_ylim(-0.003, max(0.01, max_drive * 1.18))
    ax.set_title("Linear power drive")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("SF power x TF match")
    ax.grid(True, color="#e8e8e8", lw=0.7)
    ax.text(
        0.03,
        0.93,
        "drive(t) = image SF-band power x TF match(t)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )
    ax.text(
        0.03,
        0.07,
        "SF band in this example: " + ", ".join(sf_bands),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )


def plot_map_panel(fig: plt.Figure, slot: Any, map_npz: Path, unit_metrics: pd.DataFrame, fingerprint_metrics: pd.DataFrame, unit_index: int) -> None:
    subgs = slot.subgridspec(1, 3, wspace=0.07)
    axes = [fig.add_subplot(subgs[0, idx]) for idx in range(3)]
    data = np.load(map_npz, allow_pickle=True)
    units = data["selected_units"].astype(int)
    unit_labels = data["selected_unit_labels"].astype(str)
    unit_pos = int(np.flatnonzero(units == unit_index)[0])
    label = str(unit_labels[unit_pos])
    row = fingerprint_metrics[(fingerprint_metrics["unit_index"].eq(unit_index))].iloc[0]
    frame = int(row["peak_abs_dssi_frame"])
    static_map = data["static_maps"][frame, unit_pos]
    normal_map = data["normal_maps"][frame, unit_pos]
    diff_map = normal_map - static_map
    activation_values = np.concatenate([static_map.ravel(), normal_map.ravel()])
    vmin, vmax = np.nanpercentile(activation_values, [1, 99])
    diff_limit = float(np.nanpercentile(np.abs(diff_map), 99))
    titles = ["stabilized", "normal", "difference"]
    act0 = axes[0].imshow(static_map, cmap="cividis", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].imshow(normal_map, cmap="cividis", vmin=vmin, vmax=vmax, interpolation="nearest")
    diff_im = axes[2].imshow(diff_map, cmap="PuOr_r", vmin=-diff_limit, vmax=diff_limit, interpolation="nearest")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=9)
        clean_axis(ax)
    act_cb = fig.colorbar(act0, ax=axes[:2], fraction=0.022, pad=0.02)
    act_cb.set_label("activation", fontsize=8)
    act_cb.ax.tick_params(labelsize=7)
    diff_cb = fig.colorbar(diff_im, ax=axes[2], fraction=0.046, pad=0.02)
    diff_cb.set_label("normal - stabilized", fontsize=8)
    diff_cb.ax.tick_params(labelsize=7)
    add_panel_label(axes[0], "E")
    static_ssi = float(
        unit_metrics[(unit_metrics["condition"].eq("stabilized")) & (unit_metrics["unit_index"].eq(unit_index))][
            "movie_ssi_bits_per_spike"
        ].iloc[0]
    )
    normal_ssi = float(
        unit_metrics[(unit_metrics["condition"].eq("normal")) & (unit_metrics["unit_index"].eq(unit_index))][
            "movie_ssi_bits_per_spike"
        ].iloc[0]
    )
    axes[1].text(
        0.5,
        -0.12,
        f"{label}, frame {frame} | SSI {static_ssi:.3f} -> {normal_ssi:.3f} | frame dSSI {row['peak_abs_dssi_value']:+.3f}",
        ha="center",
        va="top",
        fontsize=9,
        transform=axes[1].transAxes,
    )


def plot_group_means(ax: plt.Axes, group_table: pd.DataFrame) -> None:
    add_panel_label(ax, "F")
    gx = np.arange(group_table.shape[0])
    ax.errorbar(
        gx - 0.08,
        group_table["observed_mean_delta"],
        yerr=group_table["observed_sem_delta"],
        fmt="o",
        ms=7,
        color=OKABE_ITO["black"],
        ecolor=OKABE_ITO["black"],
        capsize=3,
        label="observed",
    )
    ax.errorbar(
        gx + 0.08,
        group_table["predicted_mean_delta"],
        yerr=group_table["predicted_sem_delta"],
        fmt="s",
        ms=6,
        color=OKABE_ITO["vermillion"],
        ecolor=OKABE_ITO["vermillion"],
        capsize=3,
        label="linear power proxy",
    )
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set_xticks(gx)
    ax.set_xticklabels(group_table["sf_group_label"])
    ax.set_ylim(-0.001, 0.032)
    ax.set_ylabel("SSI change")
    ax.set_title("Group means", fontsize=12)
    ax.grid(True, axis="y", color="#e8e8e8", lw=0.7)
    ax.legend(frameon=False, fontsize=8)


def plot_retiming_context(ax: plt.Axes, condition_summary: pd.DataFrame, scale_table: pd.DataFrame) -> None:
    add_panel_label(ax, "G")
    cond = condition_summary[condition_summary["condition_group"].eq("retiming")].copy()
    x = cond["sftf_matched_power_mean"].to_numpy(dtype=float)
    y = cond["unit_ssi_delta_absolute_mean"].to_numpy(dtype=float)
    context = scale_table[scale_table["analysis_scale"].eq("condition_means")].iloc[0]
    ax.scatter(cond["sftf_matched_power_mean"], cond["unit_ssi_delta_absolute_mean"], s=30, color=OKABE_ITO["blue"], alpha=0.82)
    if np.isfinite(x).sum() >= 3:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(xx, coef[0] * xx + coef[1], color=OKABE_ITO["black"], lw=1.2)
    ax.set_xlabel("matched power")
    ax.set_ylabel("mean SSI change")
    ax.set_title("Retiming conditions", fontsize=12)
    ax.text(
        0.06,
        0.92,
        f"R2={context['r2']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )
    ax.grid(True, color="#e8e8e8", lw=0.7)


def plot_primary_r2(ax: plt.Axes, scale_table: pd.DataFrame) -> None:
    add_panel_label(ax, "H")
    primary = scale_table[scale_table["question"].eq("normal_vs_stabilized")].copy()
    labels = ["raw\nexamples", "within\nunit", "movie\nmean", "unit\nmean"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["orange"]]
    values = primary["percent_variance_explained"].to_numpy(dtype=float)
    x = np.arange(values.size)
    ax.bar(x, values, color=colors, alpha=0.9)
    for xx, value in zip(x, values, strict=True):
        ax.text(xx, value + 0.35, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(14.0, float(np.nanmax(values)) + 2.0))
    ax.set_ylim(0.0, 12.2)
    ax.set_ylabel("variance explained (%)")
    ax.set_title("Explained variance", fontsize=12)
    ax.grid(True, axis="y", color="#e8e8e8", lw=0.7)


def plot_residuals(ax: plt.Axes, unit_means: pd.DataFrame) -> None:
    add_panel_label(ax, "I")
    residual = unit_means.copy()
    residual["residual_delta"] = residual["unit_ssi_delta_absolute"] - residual["linear_power_predicted_delta"]
    rng = np.random.default_rng(0)
    for idx, group_name in enumerate(GROUP_ORDER):
        group = residual[residual["sf_group"].eq(group_name)]
        jitter = rng.uniform(-0.12, 0.12, size=group.shape[0])
        ax.scatter(
            np.full(group.shape[0], idx) + jitter,
            group["residual_delta"],
            s=32,
            alpha=0.78,
            color=GROUP_COLORS[group_name],
            edgecolor="white",
            linewidth=0.35,
        )
        q1, med, q3 = np.nanpercentile(group["residual_delta"], [25, 50, 75])
        ax.plot([idx - 0.18, idx + 0.18], [med, med], color=OKABE_ITO["black"], lw=1.5)
        ax.plot([idx, idx], [q1, q3], color=OKABE_ITO["black"], lw=2.2)
    ax.axhline(0.0, color="#555555", lw=0.9)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[name] for name in GROUP_ORDER])
    ax.set_ylim(-0.12, 0.12)
    ax.set_ylabel("observed - predicted")
    ax.set_title("Residual SSI change", fontsize=12)
    ax.grid(True, axis="y", color="#e8e8e8", lw=0.7)


def save_logic_figure(out_dir: Path, framewise: pd.DataFrame, units: pd.DataFrame, drive_rows: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(17.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(4)]
    plot_speed_panel(axes[0], framewise)
    plot_tf_landing_panel(axes[1], framewise, units)
    plot_tf_match_panel(axes[2], drive_rows)
    plot_linear_drive_panel(axes[3], drive_rows)
    fig.suptitle("Linear power proxy, step by step", fontsize=16)
    path = out_dir / "checkpoint_06a_mechanism_logic_panels.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "checkpoint_06a_mechanism_logic_panels.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def save_evidence_figure(
    out_dir: Path,
    map_npz: Path,
    unit_metrics: pd.DataFrame,
    fingerprint_example: pd.DataFrame,
    unit_index: int,
    group_table: pd.DataFrame,
    condition_summary: pd.DataFrame,
    scale_table: pd.DataFrame,
    unit_means: pd.DataFrame,
) -> Path:
    fig = plt.figure(figsize=(15.5, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 1.05])
    plot_map_panel(fig, gs[0, :], map_npz, unit_metrics, fingerprint_example, unit_index)
    ax_f = fig.add_subplot(gs[1, 0])
    ax_g = fig.add_subplot(gs[1, 1])
    ax_h = fig.add_subplot(gs[1, 2])
    plot_group_means(ax_f, group_table)
    plot_retiming_context(ax_g, condition_summary, scale_table)
    plot_primary_r2(ax_h, scale_table)
    ax_i = fig.add_subplot(gs[2, :])
    plot_residuals(ax_i, unit_means)
    fig.suptitle("Drive, maps, and residuals", fontsize=16)
    path = out_dir / "checkpoint_06b_mechanism_evidence_panels.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "checkpoint_06b_mechanism_evidence_panels.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def write_panel_summary(out_dir: Path, scale_table: pd.DataFrame, group_table: pd.DataFrame, example_row: pd.Series) -> None:
    rows = [
        {
            "panel": "A",
            "role": "manipulation",
            "message": "Normal motion changes retinal speed while stabilized motion has zero speed.",
        },
        {
            "panel": "B",
            "role": "linear_drive",
            "message": "Retinal speed converts each unit's preferred spatial frequency into a motion-induced temporal frequency.",
        },
        {
            "panel": "C",
            "role": "tf_window",
            "message": "Each unit's TF preference defines a bandpass window over the motion-induced temporal frequency.",
        },
        {
            "panel": "D",
            "role": "linear_power_drive",
            "message": "The linear drive proxy multiplies TF match by image power in the unit's SF band.",
        },
        {
            "panel": "E",
            "role": "concrete_map_example",
            "message": (
                f"Example unit u{int(example_row['unit_index']):03d} has mean dSSI "
                f"{float(example_row['mean_delta_ssi_bits_per_spike']):+.4f} and peak-frame dSSI "
                f"{float(example_row['peak_abs_dssi_value']):+.4f}."
            ),
        },
        {
            "panel": "F",
            "role": "group_test",
            "message": "At SF-group scale the proxy captures the positive sign and rough magnitude.",
        },
        {
            "panel": "G",
            "role": "retiming_test",
            "message": (
                "Across broad retiming condition means the matched-power proxy tracks the condition curve "
                f"(R2={float(scale_table[scale_table['analysis_scale'].eq('condition_means')]['r2'].iloc[0]):.3f})."
            ),
        },
        {
            "panel": "H",
            "role": "not_full_encoding_model",
            "message": "For normal-vs-stabilized observations the proxy explains little at raw scale and about 10% after averaging.",
        },
        {
            "panel": "I",
            "role": "residuals",
            "message": "Observed-minus-predicted residuals are the target for nonlinear/map-level follow-up.",
        },
    ]
    pd.DataFrame(rows).to_csv(out_dir / "checkpoint_06_mechanism_panel_summary.csv", index=False)
    group_table.to_csv(out_dir / "checkpoint_06_group_mean_observed_vs_linear_proxy.csv", index=False)
    scale_table.to_csv(out_dir / "checkpoint_06_scale_r2_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    scorecard_dir = Path(args.scorecard_dir)
    explanation_dir = Path(args.explanation_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    example_dir_name = args.example if args.example.startswith("example_") else f"example_{args.example}"
    example_label = example_dir_name.removeprefix("example_")
    input_dir = run_dir / "map_first_power_shift_checkpoint_01_inputs_v1" / example_dir_name
    map_dir = run_dir / "map_first_power_shift_checkpoint_02_activation_maps_v1" / example_dir_name
    fingerprint_dir = run_dir / "map_first_power_shift_checkpoint_04_example_fingerprints_v1"

    framewise = pd.read_csv(input_dir / "checkpoint_01_framewise_speed_tf_landing.csv")
    units = pd.read_csv(input_dir / "checkpoint_01_representative_units.csv")
    example_selection = pd.read_csv(run_dir / "map_first_power_shift_checkpoint_01_inputs_v1" / "checkpoint_01_more_examples_selection.csv")
    selection_row = example_selection[example_selection["selection_role"].eq(example_label)].iloc[0]
    image_features = pd.read_csv(run_dir / "image_feature_table.csv")
    image_row = image_features[image_features["image_index"].eq(int(selection_row["image_position"]))].iloc[0]
    drive_rows = linear_drive_timecourses(framewise, units, image_row)
    drive_rows.to_csv(out_dir / "checkpoint_06_linear_power_drive_timecourses.csv", index=False)
    unit_metrics = pd.read_csv(map_dir / "checkpoint_02_unit_metric_summary.csv")
    fingerprint_metrics = pd.read_csv(fingerprint_dir / "checkpoint_04_example_unit_metrics.csv")
    fingerprint_example = fingerprint_metrics[fingerprint_metrics["example_label"].eq(example_label)].copy()
    unit_example_row = fingerprint_example[fingerprint_example["unit_index"].eq(int(args.unit_index))].iloc[0]
    group_table = pd.read_csv(scorecard_dir / "checkpoint_05_linear_power_unit_group_summary.csv")
    scale_table = pd.read_csv(scorecard_dir / "checkpoint_05_linear_power_scale_summary.csv")
    unit_means = pd.read_csv(scorecard_dir / "checkpoint_05_normal_vs_static_unit_means_with_groups.csv")
    condition_summary = pd.read_csv(explanation_dir / "sftf_power_explanation_condition_summary.csv")

    fig = plt.figure(figsize=(18.0, 13.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.95, 1.25, 1.05], width_ratios=[1, 1, 1, 1])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    plot_speed_panel(ax_a, framewise)
    plot_tf_landing_panel(ax_b, framewise, units)
    plot_tf_match_panel(ax_c, drive_rows)
    plot_linear_drive_panel(ax_d, drive_rows)

    plot_map_panel(
        fig,
        gs[1, 0:2],
        map_dir / "checkpoint_02_selected_activation_maps.npz",
        unit_metrics,
        fingerprint_example,
        int(args.unit_index),
    )
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[1, 3])
    plot_group_means(ax_f, group_table)
    plot_retiming_context(ax_g, condition_summary, scale_table)

    ax_h = fig.add_subplot(gs[2, 0])
    ax_i = fig.add_subplot(gs[2, 1:4])
    plot_primary_r2(ax_h, scale_table)
    plot_residuals(ax_i, unit_means)

    fig.suptitle("Linear power drive and movement SSI", fontsize=18)
    png_path = out_dir / "checkpoint_06_linear_power_mechanism_panels.png"
    pdf_path = out_dir / "checkpoint_06_linear_power_mechanism_panels.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logic_path = save_logic_figure(out_dir, framewise, units, drive_rows)
    evidence_path = save_evidence_figure(
        out_dir,
        map_dir / "checkpoint_02_selected_activation_maps.npz",
        unit_metrics,
        fingerprint_example,
        int(args.unit_index),
        group_table,
        condition_summary,
        scale_table,
        unit_means,
    )

    write_panel_summary(out_dir, scale_table, group_table, unit_example_row)
    metadata = {
        "analysis": "linear_power_mechanism_panel_checkpoint",
        "run_dir": run_dir,
        "scorecard_dir": scorecard_dir,
        "explanation_dir": explanation_dir,
        "out_dir": out_dir,
        "example": example_label,
        "example_dir_name": example_dir_name,
        "image_index": int(selection_row["image_position"]),
        "image_source_row": int(selection_row["image_source_row"]),
        "unit_index": int(args.unit_index),
        "figures": {
            "composite_png": png_path,
            "composite_pdf": pdf_path,
            "logic_png": logic_path,
            "evidence_png": evidence_path,
        },
        "contract": (
            "The linear power-shift proxy is represented as an upstream drive. "
            "It is not expected to fully predict individual downstream activation maps."
        ),
    }
    (out_dir / "checkpoint_06_metadata.json").write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"wrote mechanism panels to {png_path}")
    print(f"wrote logic panels to {logic_path}")
    print(f"wrote evidence panels to {evidence_path}")
    print(f"wrote panel summary to {out_dir / 'checkpoint_06_mechanism_panel_summary.csv'}")


if __name__ == "__main__":
    main()
