#!/usr/bin/env python3
"""Pick low-SF trace examples for the temporal power-shift checkpoint.

This is a map-first checkpoint. It chooses a small set of image x trace
examples where the normal-motion condition gives the low-SF population high
linear-power drive, then plots the concrete retinal traces that produce that
drive. The goal is interpretability, not a population claim.
"""

from __future__ import annotations

import argparse
import json
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

from declan.active_sensing_movie_information.analyze_temporal_remapping_sftf_power_explanation import (
    DEFAULT_DENSE_FIT_CSV,
    DEFAULT_PARAMETRIC_MODEL_CSV,
    load_analysis_table,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    build_trace_bank,
    load_source_rows,
    microsaccade_event_count,
)
from declan.active_sensing_movie_information.run_backimage_temporal_remapping_pilot import row_contour_axis_deg
from declan.active_sensing_movie_information.temporal_remapping import MODEL_RATE_HZ, contour_basis


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_07_low_sf_trace_examples_v1"
DEFAULT_CROSS_SF_UNIT_INDICES = "86,92,8"
DEFAULT_PARAMETRIC_CROSS_SF_UNIT_INDICES = "50,14,6"
EPS = 1e-12
TF_MATCH_SIGMA_OCTAVES = 1.0

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
UNIT_COLORS = {
    "matched_low_tf": OKABE_ITO["blue"],
    "intermediate_tf": OKABE_ITO["blue"],
    "higher_tf": OKABE_ITO["blue"],
    "high_tf_control": OKABE_ITO["blue"],
}
GROUP_ORDER = ["low_sf", "middle_sf", "high_sf"]
GROUP_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
GROUP_COLORS = {"low_sf": OKABE_ITO["blue"], "middle_sf": OKABE_ITO["green"], "high_sf": OKABE_ITO["orange"]}
UNIT_LINESTYLES = {
    "matched_low_tf": "-",
    "intermediate_tf": "--",
    "higher_tf": "-.",
    "high_tf_control": ":",
}
SF_BANDS = (
    ("image_power_0_2_cpd_fraction", 0.0, 2.0, "0-2 cpd"),
    ("image_power_2_4_cpd_fraction", 2.0, 4.0, "2-4 cpd"),
    ("image_power_4_8_cpd_fraction", 4.0, 8.0, "4-8 cpd"),
    ("image_power_8plus_cpd_fraction", 8.0, math.inf, "8+ cpd"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dense-fit-csv", type=Path, default=DEFAULT_DENSE_FIT_CSV)
    parser.add_argument("--parametric-model-csv", type=Path, default=DEFAULT_PARAMETRIC_MODEL_CSV)
    parser.add_argument(
        "--preference-source",
        choices=("legacy", "parametric"),
        default="legacy",
        help="Use legacy retiming/dense preferences or canonical RR100 parametric SF/TF preferences.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument(
        "--image-index",
        type=int,
        default=None,
        help="Image index to hold fixed. Defaults to the image in the top low-SF drive example.",
    )
    parser.add_argument(
        "--max-display-unit-tf-hz",
        type=float,
        default=20.0,
        help=(
            "Upper TF-preference limit for low-SF units shown in the checkpoint figures. "
            "The full low-SF unit ranking is still saved for audit."
        ),
    )
    parser.add_argument(
        "--include-high-tf-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the highest-TF low-SF control in the trace gallery.",
    )
    parser.add_argument(
        "--cross-sf-unit-indices",
        type=str,
        default=DEFAULT_CROSS_SF_UNIT_INDICES,
        help="Comma-separated unit indices for the cross-SF gallery. Default is low u086, middle u092, high u008.",
    )
    parser.add_argument("--force-rebuild-traces", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def clean_path_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def sf_band_for_value(preferred_sf_cpd: float) -> tuple[str, str]:
    for column, lo, hi, label in SF_BANDS:
        if np.isfinite(preferred_sf_cpd) and preferred_sf_cpd >= lo and preferred_sf_cpd < hi:
            return column, label
    return SF_BANDS[-1][0], SF_BANDS[-1][3]


def tf_match(projected_tf_hz: np.ndarray, pref_tf_hz: float) -> np.ndarray:
    projected = np.asarray(projected_tf_hz, dtype=float)
    out = np.zeros_like(projected, dtype=float)
    valid = np.isfinite(projected) & (projected > 0.0) & np.isfinite(pref_tf_hz) & (pref_tf_hz > 0.0)
    log_distance = np.zeros_like(projected, dtype=float)
    log_distance[valid] = np.log2(projected[valid] / pref_tf_hz)
    out[valid] = np.exp(-0.5 * (log_distance[valid] / TF_MATCH_SIGMA_OCTAVES) ** 2)
    return out


def summarize_units(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    if "unit_label" not in work.columns:
        work["unit_label"] = work["unit_index"].map(lambda value: f"u{int(value):03d}")
    return (
        work.groupby(["sf_group", "unit_index"], dropna=False)
        .agg(
            unit_label=("unit_label", "first"),
            preferred_sf_cpd=("preferred_sf_cpd", "first"),
            fit_pref_tf_hz=("fit_pref_tf_hz", "first"),
            fit_pref_sf_cpd=("fit_pref_sf_cpd", "first"),
            observed_peak_tf_hz=("observed_peak_tf_hz", "first"),
            mean_tf_match=("tf_match_fixed", "mean"),
            max_tf_match=("tf_match_fixed", "max"),
            mean_linear_power_drive=("sftf_matched_power", "mean"),
            max_linear_power_drive=("sftf_matched_power", "max"),
            mean_ssi_delta=("unit_ssi_delta_absolute", "mean"),
            n_observations=("sftf_matched_power", "size"),
        )
        .reset_index()
        .sort_values(["mean_linear_power_drive", "max_tf_match"], ascending=False)
    )


def build_candidate_tables(
    run_dir: Path,
    dense_fit_csv: Path,
    *,
    preference_source: str,
    parametric_model_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = load_analysis_table(
        run_dir,
        dense_fit_csv,
        include_tf_edge_fits=True,
        sigma_octaves=TF_MATCH_SIGMA_OCTAVES,
        preference_source=preference_source,
        parametric_model_csv=parametric_model_csv,
    )
    normal = rows[rows["condition_group"].eq("original")].copy()
    if "unit_label" not in normal.columns:
        normal["unit_label"] = normal["unit_index"].map(lambda value: f"u{int(value):03d}")
    low = normal[normal["sf_group"].eq("low_sf")].copy()
    if low.empty:
        raise ValueError("No low-SF normal-motion rows found.")
    all_unit_table = summarize_units(normal)
    unit_table = all_unit_table[all_unit_table["sf_group"].eq("low_sf")].copy()

    movie_table = (
        low.groupby(["image_index", "trace_index"], dropna=False)
        .agg(
            low_sf_mean_linear_power_drive=("sftf_matched_power", "mean"),
            low_sf_max_linear_power_drive=("sftf_matched_power", "max"),
            low_sf_mean_tf_match=("tf_match_fixed", "mean"),
            low_sf_max_tf_match=("tf_match_fixed", "max"),
            low_sf_mean_ssi_delta=("unit_ssi_delta_absolute", "mean"),
            low_sf_characteristic_tf_mean_hz=("characteristic_motion_tf_hz", "mean"),
            low_sf_characteristic_tf_max_hz=("characteristic_motion_tf_hz", "max"),
            rms_across_velocity_deg_s=("rms_across_velocity_deg_s", "first"),
        )
        .reset_index()
    )
    trajectory = pd.read_csv(run_dir / "retimed_trajectory_metrics.csv")
    original_traj = trajectory[trajectory["condition_group"].eq("original")].copy()
    original_traj = original_traj[
        [
            "image_index",
            "trace_index",
            "trace_source_row",
            "rms_speed_deg_s",
            "peak_speed_deg_s",
            "rms_across_velocity_deg_s",
            "peak_across_velocity_deg_s",
        ]
    ].drop_duplicates(["image_index", "trace_index"])
    movie_table = movie_table.merge(original_traj, on=["image_index", "trace_index"], how="left", suffixes=("", "_trajectory"))
    movie_table = movie_table.sort_values(
        ["low_sf_mean_linear_power_drive", "low_sf_max_tf_match"],
        ascending=False,
    ).reset_index(drop=True)
    movie_table.insert(0, "candidate_rank", np.arange(1, movie_table.shape[0] + 1, dtype=int))
    return movie_table, unit_table, all_unit_table


def select_units(unit_table: pd.DataFrame, *, max_display_unit_tf_hz: float, include_high_tf_control: bool) -> pd.DataFrame:
    eligible = unit_table.copy()
    max_tf = float(max_display_unit_tf_hz)
    if math.isfinite(max_tf) and max_tf > 0.0:
        eligible = eligible[eligible["fit_pref_tf_hz"].le(max_tf)].copy()
    if eligible.empty:
        raise ValueError(f"No low-SF units remain after max_display_unit_tf_hz={max_display_unit_tf_hz:g}.")

    selected: list[pd.Series] = []
    roles: list[str] = []

    def add_first(role: str, candidates: pd.DataFrame) -> None:
        if candidates.empty:
            return
        unit_index = int(candidates.iloc[0]["unit_index"])
        if any(int(row["unit_index"]) == unit_index for row in selected):
            return
        selected.append(candidates.iloc[0])
        roles.append(role)

    ranked = eligible.sort_values(["mean_linear_power_drive", "max_tf_match"], ascending=False)
    add_first("matched_low_tf", ranked)
    add_first("intermediate_tf", eligible[eligible["fit_pref_tf_hz"].between(1.0, 3.0)].sort_values("max_tf_match", ascending=False))
    add_first("higher_tf", eligible[eligible["fit_pref_tf_hz"].between(5.0, 12.0)].sort_values("max_tf_match", ascending=False))
    if include_high_tf_control:
        add_first("high_tf_control", unit_table.sort_values("fit_pref_tf_hz", ascending=False))

    out = pd.DataFrame(selected).reset_index(drop=True)
    out.insert(1, "selection_role", roles)
    out["plot_color"] = [UNIT_COLORS.get(role, OKABE_ITO["grey"]) for role in roles]
    return out


def parse_unit_indices(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one unit index.")
    return values


def select_cross_sf_units(all_unit_table: pd.DataFrame, unit_indices: list[int]) -> pd.DataFrame:
    requested = list(dict.fromkeys(int(value) for value in unit_indices))
    table = all_unit_table[all_unit_table["unit_index"].isin(requested)].copy()
    found = set(table["unit_index"].astype(int))
    missing = [value for value in requested if value not in found]
    if missing:
        raise ValueError(f"Requested cross-SF unit indices were not found: {missing}")
    table["selection_role"] = table["sf_group"].map(lambda group: f"{group}_representative")
    table["sf_group_label"] = table["sf_group"].map(lambda group: GROUP_LABELS.get(str(group), str(group)))
    table["plot_color"] = table["sf_group"].map(lambda group: GROUP_COLORS.get(str(group), OKABE_ITO["grey"]))
    table["selection_note"] = "user-requested cross-SF representative unit"
    table["display_order"] = table["sf_group"].map({name: idx for idx, name in enumerate(GROUP_ORDER)})
    table["request_order"] = table["unit_index"].map({unit_index: idx for idx, unit_index in enumerate(requested)})
    table = table.sort_values(["display_order", "request_order"]).drop(columns=["display_order", "request_order"])
    return table.reset_index(drop=True)


def selected_trace_cache_path(out_dir: Path) -> Path:
    return out_dir / "checkpoint_07_reconstructed_selected_traces.npz"


def load_selected_traces(run_dir: Path, out_dir: Path, *, force_rebuild: bool) -> dict[int, dict[str, Any]]:
    cache_path = selected_trace_cache_path(out_dir)
    trace_features = pd.read_csv(run_dir / "trace_feature_table.csv")
    if cache_path.exists() and not force_rebuild:
        data = np.load(cache_path)
        return {
            int(trace_index): {
                "trace_index": int(trace_index),
                "trace_source_row": int(source_row),
                "trace": trace.astype(np.float64),
            }
            for trace_index, source_row, trace in zip(
                data["trace_index"],
                data["trace_source_row"],
                data["trace"],
                strict=True,
            )
        }

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    source_csv = Path(summary["source_csv"])
    n_timepoints = int(summary.get("n_timepoints", 32))
    bin_seconds = float(summary.get("bin_seconds", 1.0 / MODEL_RATE_HZ))
    source_rows = set(pd.to_numeric(trace_features["trace_source_row"], errors="coerce").dropna().astype(int))
    rows = load_source_rows(source_csv)
    trace_bank = build_trace_bank(
        rows,
        n_timepoints=n_timepoints,
        bin_seconds=bin_seconds,
        max_path_arcmin=350.0,
    )
    trace_bank = [item for item in trace_bank if microsaccade_event_count(item) <= 0]
    by_source = {int(item["source_row"]): item for item in trace_bank if int(item["source_row"]) in source_rows}
    missing = sorted(source_rows.difference(by_source))
    if missing:
        raise ValueError(f"Could not reconstruct selected traces for source rows: {missing[:8]}")

    traces: list[np.ndarray] = []
    trace_indices: list[int] = []
    trace_source_rows: list[int] = []
    lookup: dict[int, dict[str, Any]] = {}
    for _, row in trace_features.sort_values("trace_index").iterrows():
        trace_index = int(row["trace_index"])
        source_row = int(row["trace_source_row"])
        trace = np.asarray(by_source[source_row]["trace"], dtype=np.float64)
        traces.append(trace)
        trace_indices.append(trace_index)
        trace_source_rows.append(source_row)
        lookup[trace_index] = {
            "trace_index": trace_index,
            "trace_source_row": source_row,
            "trace": trace,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        trace_index=np.asarray(trace_indices, dtype=int),
        trace_source_row=np.asarray(trace_source_rows, dtype=int),
        trace=np.stack(traces, axis=0),
    )
    return lookup


def framewise_rows_for_trace(
    trace: np.ndarray,
    image_row: pd.Series,
    units: pd.DataFrame,
    *,
    image_index: int,
    trace_index: int,
    trace_source_row: int,
    bin_seconds: float,
) -> pd.DataFrame:
    n_frames = int(trace.shape[0])
    time_ms = np.arange(n_frames, dtype=float) * float(bin_seconds) * 1000.0
    vel = np.diff(trace, axis=0) / float(bin_seconds)
    _along_u, across_u = contour_basis(row_contour_axis_deg(image_row))
    across_speed = np.zeros(n_frames, dtype=float)
    full_speed = np.zeros(n_frames, dtype=float)
    if vel.size:
        across_speed[1:] = np.abs(vel @ across_u)
        full_speed[1:] = np.linalg.norm(vel, axis=1)

    rows: list[dict[str, Any]] = []
    contrast2 = float(image_row["image_patch_rms_contrast"]) ** 2
    for frame_idx in range(n_frames):
        for _, unit in units.iterrows():
            band_col, band_label = sf_band_for_value(float(unit["preferred_sf_cpd"]))
            sf_power_abs = contrast2 * float(image_row[band_col])
            projected_tf = float(across_speed[frame_idx] * float(unit["preferred_sf_cpd"]))
            match = float(tf_match(np.asarray([projected_tf]), float(unit["fit_pref_tf_hz"]))[0])
            rows.append(
                {
                    "image_index": int(image_index),
                    "trace_index": int(trace_index),
                    "trace_source_row": int(trace_source_row),
                    "frame_index": int(frame_idx),
                    "time_ms": float(time_ms[frame_idx]),
                    "full_speed_deg_s": float(full_speed[frame_idx]),
                    "across_contour_speed_deg_s": float(across_speed[frame_idx]),
                    "unit_index": int(unit["unit_index"]),
                    "unit_label": str(unit["unit_label"]),
                    "sf_group": str(unit["sf_group"]),
                    "sf_group_label": GROUP_LABELS.get(str(unit["sf_group"]), str(unit["sf_group"])),
                    "selection_role": str(unit["selection_role"]),
                    "preferred_sf_cpd": float(unit["preferred_sf_cpd"]),
                    "fit_pref_tf_hz": float(unit["fit_pref_tf_hz"]),
                    "sf_power_band": band_label,
                    "sf_power_abs": float(sf_power_abs),
                    "motion_induced_tf_hz": projected_tf,
                    "tf_match": match,
                    "linear_power_drive": float(sf_power_abs * match),
                }
            )
    return pd.DataFrame(rows)


def select_trace_examples(movie_table: pd.DataFrame, *, image_index: int | None, n_examples: int) -> pd.DataFrame:
    if image_index is None:
        image_index = int(movie_table.iloc[0]["image_index"])
    fixed = movie_table[movie_table["image_index"].eq(int(image_index))].copy()
    if fixed.empty:
        raise ValueError(f"No movie rows found for image_index={image_index}.")
    selected = fixed.sort_values(
        ["low_sf_mean_linear_power_drive", "low_sf_max_tf_match"],
        ascending=False,
    ).head(int(n_examples))
    selected = selected.reset_index(drop=True)
    selected.insert(1, "selection_role", [f"fixed_image_top_low_sf_drive_rank_{idx + 1}" for idx in range(selected.shape[0])])
    return selected


def plot_selection_context(out_dir: Path, movie_table: pd.DataFrame, selected: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    sc = ax.scatter(
        movie_table["rms_across_velocity_deg_s"],
        movie_table["low_sf_mean_linear_power_drive"],
        c=movie_table["low_sf_max_tf_match"],
        cmap="cividis",
        s=28,
        alpha=0.72,
        edgecolor="none",
    )
    ax.scatter(
        selected["rms_across_velocity_deg_s"],
        selected["low_sf_mean_linear_power_drive"],
        s=88,
        facecolor="none",
        edgecolor=OKABE_ITO["vermillion"],
        linewidth=1.8,
        label="selected",
    )
    label_offsets = [(7, 0), (8, -9), (8, 8), (8, -9), (8, 8)]
    for label_idx, (_, row) in enumerate(selected.iterrows()):
        ax.annotate(
            f"t{int(row['trace_index'])}",
            xy=(float(row["rms_across_velocity_deg_s"]), float(row["low_sf_mean_linear_power_drive"])),
            xytext=label_offsets[label_idx % len(label_offsets)],
            textcoords="offset points",
            fontsize=8,
            va="center",
            color=OKABE_ITO["black"],
        )
    ax.set_xlabel("RMS across-contour motion (deg/s)")
    ax.set_ylabel("mean low-SF linear drive")
    ax.set_title("Trace candidates for low-SF drive")
    ax.grid(True, color="#e8e8e8", lw=0.7)
    ax.legend(frameon=False, loc="upper left")
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("best low-SF RMS-TF match")
    path = out_dir / "checkpoint_07a_low_sf_trace_selection_context.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "checkpoint_07a_low_sf_trace_selection_context.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trace_gallery(
    out_dir: Path,
    selected: pd.DataFrame,
    units: pd.DataFrame,
    traces: dict[int, dict[str, Any]],
    timecourses: pd.DataFrame,
    *,
    title: str,
    filename_stem: str,
    legend_title: str,
) -> Path:
    n_rows = int(selected.shape[0])
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(16.2, max(3.0, 2.05 * n_rows)),
        constrained_layout=True,
        squeeze=False,
    )
    role_to_color = dict(zip(units["selection_role"], units["plot_color"], strict=True))
    unit_label_for_role = {
        str(unit["selection_role"]): f"{unit['unit_label']} ({float(unit['fit_pref_tf_hz']):.1f} Hz pref)"
        for _, unit in units.iterrows()
    }
    max_drive = max(float(timecourses["linear_power_drive"].max()), 1e-6)
    min_visible_drive = max(max_drive * 1e-8, 1e-12)
    min_visible_tf = 0.01
    max_tf = max(float(timecourses["motion_induced_tf_hz"].max()), float(units["fit_pref_tf_hz"].max()))

    for row_idx, (_, selected_row) in enumerate(selected.iterrows()):
        trace_index = int(selected_row["trace_index"])
        trace_info = traces[trace_index]
        trace = np.asarray(trace_info["trace"], dtype=float)
        row_time = timecourses[timecourses["trace_index"].eq(trace_index)].copy()
        speed = row_time.drop_duplicates("frame_index")

        ax_path, ax_speed, ax_tf, ax_drive = axes[row_idx]

        ax_path.plot(trace[:, 0], trace[:, 1], color=OKABE_ITO["black"], lw=1.5)
        ax_path.scatter(trace[0, 0], trace[0, 1], s=26, color="white", edgecolor=OKABE_ITO["black"], linewidth=1.0, zorder=3)
        ax_path.scatter(trace[-1, 0], trace[-1, 1], s=28, color=OKABE_ITO["black"], zorder=3)
        clean_path_axis(ax_path)
        ax_path.set_aspect("equal", adjustable="box")
        ax_path.set_title(f"trace {trace_index}", fontsize=10)
        ax_path.text(
            0.02,
            0.02,
            f"peak {float(selected_row['peak_across_velocity_deg_s']):.1f} deg/s\nmean drive {float(selected_row['low_sf_mean_linear_power_drive']):.4f}",
            transform=ax_path.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 2.0},
        )

        ax_speed.plot(speed["time_ms"], speed["across_contour_speed_deg_s"], color=OKABE_ITO["black"], lw=1.7)
        ax_speed.set_ylim(0.0, max(1.0, float(timecourses["across_contour_speed_deg_s"].max()) * 1.12))
        ax_speed.grid(True, color="#e8e8e8", lw=0.7)
        if row_idx == 0:
            ax_speed.set_title("Retinal motion", fontsize=10)
        ax_speed.set_ylabel("deg/s")

        for _, unit_rows in row_time.groupby("selection_role", sort=False):
            role = str(unit_rows["selection_role"].iloc[0])
            color = role_to_color.get(role, OKABE_ITO["grey"])
            label = unit_label_for_role.get(role, role)
            linestyle = UNIT_LINESTYLES.get(role, "-")
            tf_values = unit_rows["motion_induced_tf_hz"].to_numpy(dtype=float)
            visible_tf = np.maximum(tf_values, min_visible_tf)
            ax_tf.plot(unit_rows["time_ms"], visible_tf, color=color, lw=1.7, ls=linestyle, label=label)
            at_floor = tf_values <= min_visible_tf
            if np.any(at_floor):
                ax_tf.scatter(
                    unit_rows["time_ms"].to_numpy(dtype=float)[at_floor],
                    visible_tf[at_floor],
                    s=12,
                    color=color,
                    marker="v",
                    alpha=0.65,
                    linewidth=0.0,
                )
            ax_tf.axhline(float(unit_rows["fit_pref_tf_hz"].iloc[0]), color=color, lw=0.9, ls=linestyle, alpha=0.42)
            visible_drive = np.maximum(unit_rows["linear_power_drive"].to_numpy(dtype=float), min_visible_drive)
            ax_drive.plot(unit_rows["time_ms"], visible_drive, color=color, lw=1.5, ls=linestyle, label=label)
        ax_tf.set_yscale("log")
        ax_tf.set_ylim(min_visible_tf, max(8.0, max_tf * 1.6))
        ax_tf.grid(True, which="both", color="#e8e8e8", lw=0.7)
        if row_idx == 0:
            ax_tf.set_title("Motion-induced TF", fontsize=10)
        ax_tf.set_ylabel("Hz")

        ax_drive.set_yscale("log")
        ax_drive.set_ylim(min_visible_drive, max_drive * 1.6)
        ax_drive.grid(True, which="both", color="#e8e8e8", lw=0.7)
        if row_idx == 0:
            ax_drive.set_title("Linear power drive", fontsize=10)
        ax_drive.set_ylabel("power x match")

        if row_idx == n_rows - 1:
            ax_speed.set_xlabel("time (ms)")
            ax_tf.set_xlabel("time (ms)")
            ax_drive.set_xlabel("time (ms)")
        else:
            ax_speed.set_xticklabels([])
            ax_tf.set_xticklabels([])
            ax_drive.set_xticklabels([])

    add_panel_label(axes[0, 0], "A")
    add_panel_label(axes[0, 1], "B")
    add_panel_label(axes[0, 2], "C")
    add_panel_label(axes[0, 3], "D")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.02),
        ncol=3,
        frameon=False,
        fontsize=8,
        title=legend_title,
    )
    legend.get_title().set_fontsize(8)
    fig.suptitle(title, fontsize=15, y=1.045)
    path = out_dir / f"{filename_stem}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_reachable_low_sf_units(
    out_dir: Path,
    unit_table: pd.DataFrame,
    selected_units: pd.DataFrame,
    *,
    max_display_unit_tf_hz: float,
) -> Path:
    display_table = unit_table.copy()
    max_tf = float(max_display_unit_tf_hz)
    if math.isfinite(max_tf) and max_tf > 0.0:
        display_table = display_table[display_table["fit_pref_tf_hz"].le(max_tf)].copy()

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    selected_set = set(selected_units["unit_index"].astype(int))
    ax.scatter(
        display_table["fit_pref_tf_hz"],
        display_table["max_tf_match"],
        s=42,
        color=OKABE_ITO["grey"],
        alpha=0.58,
        edgecolor="none",
    )
    for _, row in selected_units.iterrows():
        color = str(row["plot_color"])
        ax.scatter(
            float(row["fit_pref_tf_hz"]),
            float(row["max_tf_match"]),
            s=88,
            color=color,
            edgecolor=OKABE_ITO["black"],
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            str(row["unit_label"]),
            xy=(float(row["fit_pref_tf_hz"]), max(float(row["max_tf_match"]), 1e-10)),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    finite_tf = pd.to_numeric(display_table["fit_pref_tf_hz"], errors="coerce").to_numpy(dtype=float)
    finite_tf = finite_tf[np.isfinite(finite_tf) & (finite_tf > 0.0)]
    if finite_tf.size:
        ax.set_xlim(float(np.nanmin(finite_tf)) / 1.25, float(np.nanmax(finite_tf)) * 1.55)
    ax.set_ylim(8e-11, 1.8)
    ax.set_xlabel("low-SF unit TF preference (Hz)")
    ax.set_ylabel("best RMS-TF match")
    ax.set_title("Which low-SF units are reachable on average?")
    ax.grid(True, which="both", color="#e8e8e8", lw=0.7)
    ax.text(
        0.03,
        0.05,
        f"selected units: {len(selected_set)} of {display_table.shape[0]} displayed low-SF units",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "pad": 2.0},
    )
    path = out_dir / "checkpoint_07c_low_sf_unit_reachability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "checkpoint_07c_low_sf_unit_reachability.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def build_timecourses_for_units(
    selected_traces: pd.DataFrame,
    trace_lookup: dict[int, dict[str, Any]],
    image_row: pd.Series,
    units: pd.DataFrame,
    *,
    image_index: int,
    bin_seconds: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, row in selected_traces.iterrows():
        trace_index = int(row["trace_index"])
        trace_info = trace_lookup[trace_index]
        parts.append(
            framewise_rows_for_trace(
                np.asarray(trace_info["trace"], dtype=float),
                image_row,
                units,
                image_index=image_index,
                trace_index=trace_index,
                trace_source_row=int(trace_info["trace_source_row"]),
                bin_seconds=bin_seconds,
            )
        )
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    movie_table, unit_table, all_unit_table = build_candidate_tables(
        run_dir,
        Path(args.dense_fit_csv),
        preference_source=str(args.preference_source),
        parametric_model_csv=Path(args.parametric_model_csv),
    )
    selected_traces = select_trace_examples(movie_table, image_index=args.image_index, n_examples=int(args.n_examples))
    selected_units = select_units(
        unit_table,
        max_display_unit_tf_hz=float(args.max_display_unit_tf_hz),
        include_high_tf_control=bool(args.include_high_tf_control),
    )
    cross_sf_unit_text = str(args.cross_sf_unit_indices)
    if str(args.preference_source) == "parametric" and cross_sf_unit_text == DEFAULT_CROSS_SF_UNIT_INDICES:
        cross_sf_unit_text = DEFAULT_PARAMETRIC_CROSS_SF_UNIT_INDICES
    selected_cross_sf_units = select_cross_sf_units(all_unit_table, parse_unit_indices(cross_sf_unit_text))
    excluded_units = unit_table[
        unit_table["fit_pref_tf_hz"].gt(float(args.max_display_unit_tf_hz))
        & ~unit_table["unit_index"].isin(selected_units["unit_index"])
    ].copy()
    if not excluded_units.empty:
        excluded_units.insert(1, "exclusion_reason", f"fit_pref_tf_hz > {float(args.max_display_unit_tf_hz):g} Hz display limit")

    image_table = pd.read_csv(run_dir / "image_feature_table.csv")
    image_index = int(selected_traces["image_index"].iloc[0])
    image_row = image_table[image_table["image_index"].eq(image_index)].iloc[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    bin_seconds = float(summary.get("bin_seconds", 1.0 / MODEL_RATE_HZ))
    trace_lookup = load_selected_traces(run_dir, out_dir, force_rebuild=bool(args.force_rebuild_traces))

    timecourses = build_timecourses_for_units(
        selected_traces,
        trace_lookup,
        image_row,
        selected_units,
        image_index=image_index,
        bin_seconds=bin_seconds,
    )
    cross_sf_timecourses = build_timecourses_for_units(
        selected_traces,
        trace_lookup,
        image_row,
        selected_cross_sf_units,
        image_index=image_index,
        bin_seconds=bin_seconds,
    )

    candidate_path = out_dir / "checkpoint_07_low_sf_candidate_ranking.csv"
    selected_trace_path = out_dir / "checkpoint_07_selected_low_sf_trace_examples.csv"
    selected_unit_path = out_dir / "checkpoint_07_selected_low_sf_units.csv"
    selected_cross_sf_unit_path = out_dir / "checkpoint_07_selected_cross_sf_units.csv"
    excluded_unit_path = out_dir / "checkpoint_07_excluded_low_sf_units.csv"
    timecourse_path = out_dir / "checkpoint_07_low_sf_trace_timecourses.csv"
    cross_sf_timecourse_path = out_dir / "checkpoint_07_cross_sf_trace_timecourses.csv"
    movie_table.to_csv(candidate_path, index=False)
    selected_traces.to_csv(selected_trace_path, index=False)
    unit_table.to_csv(out_dir / "checkpoint_07_low_sf_unit_ranking.csv", index=False)
    all_unit_table.to_csv(out_dir / "checkpoint_07_all_sf_unit_ranking.csv", index=False)
    selected_units.to_csv(selected_unit_path, index=False)
    selected_cross_sf_units.to_csv(selected_cross_sf_unit_path, index=False)
    excluded_units.to_csv(excluded_unit_path, index=False)
    timecourses.to_csv(timecourse_path, index=False)
    cross_sf_timecourses.to_csv(cross_sf_timecourse_path, index=False)

    selection_context = plot_selection_context(out_dir, movie_table, selected_traces)
    trace_gallery = plot_trace_gallery(
        out_dir,
        selected_traces,
        selected_units,
        trace_lookup,
        timecourses,
        title="Low-SF trace examples: highest available motion-induced TF",
        filename_stem="checkpoint_07b_low_sf_trace_gallery",
        legend_title="Low SF units: blue; linestyle distinguishes units",
    )
    cross_sf_trace_gallery = plot_trace_gallery(
        out_dir,
        selected_traces,
        selected_cross_sf_units,
        trace_lookup,
        cross_sf_timecourses,
        title="Same traces, representative SF groups",
        filename_stem="checkpoint_07d_cross_sf_trace_gallery",
        legend_title="SF group colors: Low blue, Middle green, High orange",
    )
    reachability = plot_reachable_low_sf_units(
        out_dir,
        unit_table,
        selected_units,
        max_display_unit_tf_hz=float(args.max_display_unit_tf_hz),
    )

    metadata = {
        "analysis": "map_first_low_sf_trace_examples",
        "run_dir": run_dir,
        "dense_fit_csv": Path(args.dense_fit_csv),
        "parametric_model_csv": Path(args.parametric_model_csv),
        "preference_source": str(args.preference_source),
        "out_dir": out_dir,
        "image_index": image_index,
        "n_examples": int(selected_traces.shape[0]),
        "max_display_unit_tf_hz": float(args.max_display_unit_tf_hz),
        "include_high_tf_control": bool(args.include_high_tf_control),
        "cross_sf_unit_indices": parse_unit_indices(cross_sf_unit_text),
        "sf_group_color_mapping": GROUP_COLORS,
        "selection_rule": (
            "Hold the image fixed at the global top low-SF linear-drive example, then select "
            "the highest ranked normal-motion traces by low-SF mean linear-power drive."
        ),
        "tf_drive_contract": (
            "Framewise TF uses absolute across-contour velocity times each unit's preferred SF, "
            "matching the normal-vs-stabilized linear-power proxy."
        ),
        "tables": {
            "candidate_ranking": candidate_path,
            "selected_traces": selected_trace_path,
            "selected_units": selected_unit_path,
            "selected_cross_sf_units": selected_cross_sf_unit_path,
            "excluded_units": excluded_unit_path,
            "timecourses": timecourse_path,
            "cross_sf_timecourses": cross_sf_timecourse_path,
        },
        "figures": {
            "selection_context": selection_context,
            "trace_gallery": trace_gallery,
            "cross_sf_trace_gallery": cross_sf_trace_gallery,
            "unit_reachability": reachability,
        },
    }
    (out_dir / "checkpoint_07_metadata.json").write_text(json.dumps(json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    print(f"wrote selection context to {selection_context}")
    print(f"wrote trace gallery to {trace_gallery}")
    print(f"wrote cross-SF trace gallery to {cross_sf_trace_gallery}")
    print(f"wrote reachability plot to {reachability}")
    print(f"wrote selected traces to {selected_trace_path}")
    print(f"wrote selected units to {selected_unit_path}")
    print(f"wrote selected cross-SF units to {selected_cross_sf_unit_path}")


if __name__ == "__main__":
    main()
