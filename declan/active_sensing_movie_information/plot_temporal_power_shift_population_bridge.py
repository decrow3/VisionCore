#!/usr/bin/env python3
"""Population bridge from linear power drive to activation and map SSI.

This checkpoint asks the simple stepwise question exposed by the example maps:

1. Does the linear SF/TF power proxy predict more model activation?
2. Does more activation predict larger map SSI change?
3. Does the same proxy predict map SSI change directly?

The figure deliberately keeps the raw unit x image x trace view separate from
unit-mean SF-group summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.analyze_temporal_remapping_sftf_power_explanation import (
    DEFAULT_DENSE_FIT_CSV,
    DEFAULT_PARAMETRIC_MODEL_CSV,
    DEFAULT_RUN_DIR,
    load_analysis_table,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    build_trace_bank,
    load_source_rows,
    microsaccade_event_count,
)
from declan.active_sensing_movie_information.run_backimage_temporal_remapping_pilot import row_contour_axis_deg
from declan.active_sensing_movie_information.temporal_remapping import MODEL_RATE_HZ, contour_basis


DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_09_population_bridge_v1"
DEFAULT_EXAMPLE_IMAGE_INDEX = 9
DEFAULT_EXAMPLE_TRACE_INDICES = "27,28,29,30,31"
DEFAULT_EXAMPLE_UNIT_INDICES = "86,92,8"
DEFAULT_PARAMETRIC_EXAMPLE_UNIT_INDICES = "50,14,6"
DEFAULT_MAX_UNIT_TF_HZ = 20.0
EPS = 1e-12
SF_BANDS = (
    ("image_power_0_2_cpd_fraction", 0.0, 2.0, "0-2 cpd"),
    ("image_power_2_4_cpd_fraction", 2.0, 4.0, "2-4 cpd"),
    ("image_power_4_8_cpd_fraction", 4.0, 8.0, "4-8 cpd"),
    ("image_power_8plus_cpd_fraction", 8.0, math.inf, "8+ cpd"),
)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "grey": "#777777",
    "black": "#222222",
}
GROUP_ORDER = ["low_sf", "middle_sf", "high_sf"]
GROUP_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
GROUP_COLORS = {"low_sf": OKABE_ITO["blue"], "middle_sf": OKABE_ITO["green"], "high_sf": OKABE_ITO["orange"]}


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
    parser.add_argument("--tf-match-sigma-octaves", type=float, default=1.0)
    parser.add_argument("--include-tf-edge-fits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-unit-tf-hz",
        type=float,
        default=DEFAULT_MAX_UNIT_TF_HZ,
        help="Maximum fitted TF preference included in the visual population. Use inf to disable.",
    )
    parser.add_argument("--force-rebuild-traces", action="store_true")
    parser.add_argument("--example-image-index", type=int, default=DEFAULT_EXAMPLE_IMAGE_INDEX)
    parser.add_argument("--example-trace-indices", type=str, default=DEFAULT_EXAMPLE_TRACE_INDICES)
    parser.add_argument("--example-unit-indices", type=str, default=DEFAULT_EXAMPLE_UNIT_INDICES)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def finite_xy(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    return frame.loc[np.isfinite(x) & np.isfinite(y)].copy()


def simple_linear_stats(frame: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    work = finite_xy(frame, x_col, y_col)
    x = work[x_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    if x.size < 3 or float(np.nanstd(x)) <= 0.0 or float(np.nanstd(y)) <= 0.0:
        return {"n": int(x.size), "r": float("nan"), "r2": float("nan"), "slope": float("nan")}
    design = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": int(x.size),
        "r": r,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"),
        "slope": float(coef[1]),
    }


def sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def sf_band_for_value(preferred_sf_cpd: float) -> tuple[str, str]:
    for column, lo, hi, label in SF_BANDS:
        if np.isfinite(preferred_sf_cpd) and preferred_sf_cpd >= lo and preferred_sf_cpd < hi:
            return column, label
    return SF_BANDS[-1][0], SF_BANDS[-1][3]


def tf_match(projected_tf_hz: np.ndarray, pref_tf_hz: float, *, sigma_octaves: float) -> np.ndarray:
    projected = np.asarray(projected_tf_hz, dtype=float)
    out = np.zeros_like(projected, dtype=float)
    valid = np.isfinite(projected) & (projected > 0.0) & np.isfinite(pref_tf_hz) & (pref_tf_hz > 0.0)
    log_distance = np.zeros_like(projected, dtype=float)
    log_distance[valid] = np.log2(projected[valid] / pref_tf_hz)
    out[valid] = np.exp(-0.5 * (log_distance[valid] / float(sigma_octaves)) ** 2)
    return out


def trace_cache_path(out_dir: Path) -> Path:
    return out_dir / "checkpoint_09_reconstructed_traces.npz"


def load_trace_lookup(run_dir: Path, out_dir: Path, *, force_rebuild: bool) -> tuple[dict[int, dict[str, Any]], float]:
    cache_path = trace_cache_path(out_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    bin_seconds = float(summary.get("bin_seconds", 1.0 / MODEL_RATE_HZ))
    if cache_path.exists() and not force_rebuild:
        data = np.load(cache_path)
        return (
            {
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
            },
            bin_seconds,
        )

    trace_features = pd.read_csv(run_dir / "trace_feature_table.csv")
    source_csv = Path(summary["source_csv"])
    n_timepoints = int(summary.get("n_timepoints", 32))
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

    lookup: dict[int, dict[str, Any]] = {}
    trace_indices: list[int] = []
    trace_source_rows: list[int] = []
    traces: list[np.ndarray] = []
    for _, row in trace_features.sort_values("trace_index").iterrows():
        trace_index = int(row["trace_index"])
        source_row = int(row["trace_source_row"])
        trace = np.asarray(by_source[source_row]["trace"], dtype=np.float64)
        lookup[trace_index] = {
            "trace_index": trace_index,
            "trace_source_row": source_row,
            "trace": trace,
        }
        trace_indices.append(trace_index)
        trace_source_rows.append(source_row)
        traces.append(trace)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        trace_index=np.asarray(trace_indices, dtype=int),
        trace_source_row=np.asarray(trace_source_rows, dtype=int),
        trace=np.stack(traces, axis=0),
    )
    return lookup, bin_seconds


def add_framewise_linear_drive(
    rows: pd.DataFrame,
    run_dir: Path,
    out_dir: Path,
    *,
    sigma_octaves: float,
    force_rebuild_traces: bool,
) -> pd.DataFrame:
    out = rows.copy()
    traces, bin_seconds = load_trace_lookup(run_dir, out_dir, force_rebuild=force_rebuild_traces)
    images = pd.read_csv(run_dir / "image_feature_table.csv").set_index("image_index")
    drive = np.full(out.shape[0], np.nan, dtype=float)
    peak_tf = np.full(out.shape[0], np.nan, dtype=float)
    mean_tf = np.full(out.shape[0], np.nan, dtype=float)
    peak_match = np.full(out.shape[0], np.nan, dtype=float)
    mean_match = np.full(out.shape[0], np.nan, dtype=float)

    for (image_index, trace_index), group in out.groupby(["image_index", "trace_index"], sort=False):
        image_row = images.loc[int(image_index)]
        trace = np.asarray(traces[int(trace_index)]["trace"], dtype=float)
        vel = np.diff(trace, axis=0) / float(bin_seconds)
        _along_u, across_u = contour_basis(row_contour_axis_deg(image_row))
        across_speed = np.zeros(trace.shape[0], dtype=float)
        if vel.size:
            across_speed[1:] = np.abs(vel @ across_u)
        contrast2 = float(image_row["image_patch_rms_contrast"]) ** 2
        for row_index, row in group.iterrows():
            band_col, _band_label = sf_band_for_value(float(row["preferred_sf_cpd"]))
            sf_power_abs = contrast2 * float(image_row[band_col])
            projected_tf = across_speed * float(row["preferred_sf_cpd"])
            match = tf_match(projected_tf, float(row["fit_pref_tf_hz"]), sigma_octaves=sigma_octaves)
            drive[int(row_index)] = float(sf_power_abs * np.nanmean(match))
            mean_tf[int(row_index)] = float(np.nanmean(projected_tf))
            peak_tf[int(row_index)] = float(np.nanmax(projected_tf))
            mean_match[int(row_index)] = float(np.nanmean(match))
            peak_match[int(row_index)] = float(np.nanmax(match))

    out["linear_power_drive"] = drive
    out["motion_induced_tf_mean_hz"] = mean_tf
    out["motion_induced_tf_peak_hz"] = peak_tf
    out["framewise_tf_match_mean"] = mean_match
    out["framewise_tf_match_peak"] = peak_match
    return out


def load_normal_rows(
    run_dir: Path,
    dense_fit_csv: Path,
    parametric_model_csv: Path,
    out_dir: Path,
    *,
    preference_source: str,
    include_tf_edge_fits: bool,
    sigma_octaves: float,
    max_unit_tf_hz: float,
    force_rebuild_traces: bool,
) -> pd.DataFrame:
    analysis = load_analysis_table(
        run_dir,
        dense_fit_csv,
        include_tf_edge_fits=include_tf_edge_fits,
        sigma_octaves=sigma_octaves,
        preference_source=preference_source,
        parametric_model_csv=parametric_model_csv,
    )
    normal = analysis[analysis["condition_group"].astype(str).eq("original")].copy()
    normal["rms_linear_power_drive"] = pd.to_numeric(normal["sftf_matched_power"], errors="coerce")
    if math.isfinite(float(max_unit_tf_hz)):
        normal = normal[pd.to_numeric(normal["fit_pref_tf_hz"], errors="coerce").le(float(max_unit_tf_hz))].copy()

    rate_cols = [
        "image_index",
        "trace_index",
        "condition_group",
        "unit_index",
        "unit_label",
        "unit_mean_rate",
        "unit_expected_spikes",
    ]
    rate_rows = pd.read_csv(run_dir / "retiming_unit_observations.csv", usecols=rate_cols)
    key_cols = ["image_index", "trace_index", "unit_index"]
    original = rate_rows[rate_rows["condition_group"].astype(str).eq("original")].copy()
    stabilized = rate_rows[rate_rows["condition_group"].astype(str).eq("stabilized")].copy()
    original = original.rename(
        columns={
            "unit_mean_rate": "mean_activation_normal",
            "unit_expected_spikes": "expected_spikes_normal",
        }
    )
    stabilized = stabilized.rename(
        columns={
            "unit_mean_rate": "mean_activation_stabilized",
            "unit_expected_spikes": "expected_spikes_stabilized",
        }
    )
    paired = original[
        [*key_cols, "unit_label", "mean_activation_normal", "expected_spikes_normal"]
    ].merge(
        stabilized[[*key_cols, "mean_activation_stabilized", "expected_spikes_stabilized"]],
        on=key_cols,
        how="inner",
    )
    paired["delta_mean_activation"] = (
        paired["mean_activation_normal"].to_numpy(dtype=float)
        - paired["mean_activation_stabilized"].to_numpy(dtype=float)
    )
    paired["delta_expected_spikes"] = (
        paired["expected_spikes_normal"].to_numpy(dtype=float)
        - paired["expected_spikes_stabilized"].to_numpy(dtype=float)
    )

    merged = normal.merge(paired, on=key_cols, how="inner")
    missing = normal.shape[0] - merged.shape[0]
    if missing:
        raise ValueError(f"Missing rate pairs for {missing} normal-motion rows")
    merged["sf_group_label"] = merged["sf_group"].map(lambda group: GROUP_LABELS.get(str(group), str(group)))
    merged["plot_color"] = merged["sf_group"].map(lambda group: GROUP_COLORS.get(str(group), OKABE_ITO["grey"]))
    merged = add_framewise_linear_drive(
        merged.reset_index(drop=True),
        run_dir,
        out_dir,
        sigma_octaves=sigma_octaves,
        force_rebuild_traces=force_rebuild_traces,
    )
    return merged.sort_values(["sf_group", "unit_index", "image_index", "trace_index"]).reset_index(drop=True)


def unit_mean_rows(normal: pd.DataFrame) -> pd.DataFrame:
    grouped = normal.groupby("unit_index", dropna=False, sort=False)
    rows = grouped.agg(
        unit_label=("unit_label", "first"),
        sf_group=("sf_group", "first"),
        sf_group_label=("sf_group_label", "first"),
        preferred_sf_cpd=("preferred_sf_cpd", "first"),
        fit_pref_tf_hz=("fit_pref_tf_hz", "first"),
        n_movie_examples=("linear_power_drive", "size"),
        linear_power_drive=("linear_power_drive", "mean"),
        delta_mean_activation=("delta_mean_activation", "mean"),
        map_ssi_change=("unit_ssi_delta_absolute", "mean"),
        tf_match=("tf_match_fixed", "mean"),
        unit_sf_power=("unit_sf_power_abs", "mean"),
        rms_linear_power_drive=("rms_linear_power_drive", "mean"),
    )
    return rows.reset_index()


def group_summary(unit_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        ("linear_power_drive", "Linear power drive"),
        ("delta_mean_activation", "Mean activation change"),
        ("map_ssi_change", "Map SSI change"),
    ]
    for group_name in GROUP_ORDER:
        group = unit_means[unit_means["sf_group"].astype(str).eq(group_name)]
        for column, label in metrics:
            values = pd.to_numeric(group[column], errors="coerce")
            rows.append(
                {
                    "sf_group": group_name,
                    "sf_group_label": GROUP_LABELS[group_name],
                    "metric": column,
                    "metric_label": label,
                    "n_units": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sem": sem(values),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def select_example_rows(
    normal: pd.DataFrame,
    unit_means: pd.DataFrame,
    *,
    image_index: int,
    trace_indices: list[int],
    unit_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = normal[
        normal["image_index"].eq(int(image_index))
        & normal["trace_index"].isin(trace_indices)
        & normal["unit_index"].isin(unit_indices)
    ].copy()
    means = unit_means[unit_means["unit_index"].isin(unit_indices)].copy()
    order = {unit: idx for idx, unit in enumerate(unit_indices)}
    raw["_unit_order"] = raw["unit_index"].map(order).fillna(999).astype(int)
    means["_unit_order"] = means["unit_index"].map(order).fillna(999).astype(int)
    raw = raw.sort_values(["_unit_order", "trace_index"]).drop(columns=["_unit_order"])
    means = means.sort_values(["_unit_order"]).drop(columns=["_unit_order"])
    return raw, means


def apply_robust_limits(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    work = finite_xy(frame, x_col, y_col)
    for getter, setter, col in [(ax.get_xlim, ax.set_xlim, x_col), (ax.get_ylim, ax.set_ylim, y_col)]:
        values = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        if values.size == 0:
            continue
        lo, hi = np.nanpercentile(values, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        pad = 0.06 * (hi - lo)
        setter(float(lo - pad), float(hi + pad))


def plot_population_bridge(
    normal: pd.DataFrame,
    unit_means: pd.DataFrame,
    example_raw: pd.DataFrame,
    example_means: pd.DataFrame,
    stats: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8), constrained_layout=True)
    scatter_specs = [
        (
            "Does power drive activation?",
            "linear_power_drive",
            "delta_mean_activation",
            "linear power drive (a.u.)",
            "mean activation change\nnormal - stabilized",
        ),
        (
            "Does activation predict SSI?",
            "delta_mean_activation",
            "unit_ssi_delta_absolute",
            "mean activation change\nnormal - stabilized",
            "map SSI change\nnormal - stabilized",
        ),
        (
            "Does power drive predict SSI?",
            "linear_power_drive",
            "unit_ssi_delta_absolute",
            "linear power drive (a.u.)",
            "map SSI change\nnormal - stabilized",
        ),
    ]
    for idx, (title, x_col, y_col, xlabel, ylabel) in enumerate(scatter_specs):
        ax = axes[0, idx]
        add_panel_label(ax, chr(ord("A") + idx))
        for group_name in GROUP_ORDER:
            group = normal[normal["sf_group"].astype(str).eq(group_name)]
            ax.scatter(
                group[x_col],
                group[y_col],
                s=8,
                alpha=0.075,
                color=GROUP_COLORS[group_name],
                edgecolors="none",
                rasterized=True,
            )
        if y_col != "linear_power_drive":
            ax.axhline(0.0, color="#555555", lw=0.8)
        if x_col != "linear_power_drive":
            ax.axvline(0.0, color="#555555", lw=0.8)
        for unit_index, rows in example_raw.groupby("unit_index", sort=False):
            group_name = str(rows["sf_group"].iloc[0])
            ax.scatter(
                rows[x_col],
                rows[y_col],
                s=62,
                color=GROUP_COLORS.get(group_name, OKABE_ITO["grey"]),
                edgecolors=OKABE_ITO["black"],
                linewidths=0.9,
                alpha=0.96,
                zorder=5,
            )
        apply_robust_limits(ax, normal, x_col, y_col)
        row = stats[stats["relation"].eq(f"{x_col}_to_{y_col}")].iloc[0]
        ax.text(
            0.04,
            0.96,
            f"raw examples\nR2={float(row['raw_r2']):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
        )
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e8e8e8", lw=0.65)

    metric_specs = [
        ("Population drive", "linear_power_drive", "linear power drive (a.u.)"),
        ("Population activation change", "delta_mean_activation", "mean activation change\nnormal - stabilized"),
        ("Population SSI change", "map_ssi_change", "map SSI change\nnormal - stabilized"),
    ]
    rng = np.random.default_rng(2)
    for panel_offset, (title, column, ylabel) in enumerate(metric_specs):
        ax = axes[1, panel_offset]
        add_panel_label(ax, chr(ord("D") + panel_offset))
        for group_idx, group_name in enumerate(GROUP_ORDER):
            group = unit_means[unit_means["sf_group"].astype(str).eq(group_name)]
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            jitter = rng.uniform(-0.13, 0.13, size=values.size)
            ax.scatter(
                np.full(values.size, group_idx) + jitter,
                values,
                s=33,
                alpha=0.72,
                color=GROUP_COLORS[group_name],
                edgecolors="white",
                linewidths=0.35,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                q25, med, q75 = np.nanpercentile(finite, [25, 50, 75])
                ax.plot([group_idx - 0.18, group_idx + 0.18], [med, med], color=OKABE_ITO["black"], lw=1.5)
                ax.plot([group_idx, group_idx], [q25, q75], color=OKABE_ITO["black"], lw=2.3)
        for _, row in example_means.iterrows():
            group_idx = GROUP_ORDER.index(str(row["sf_group"]))
            ax.scatter(
                group_idx,
                row[column],
                s=105,
                marker="D",
                color=GROUP_COLORS.get(str(row["sf_group"]), OKABE_ITO["grey"]),
                edgecolors=OKABE_ITO["black"],
                linewidths=1.1,
                zorder=6,
            )
        if column != "linear_power_drive":
            ax.axhline(0.0, color="#555555", lw=0.8)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12.5)
        ax.grid(True, axis="y", color="#e8e8e8", lw=0.65)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLORS[group], markeredgecolor="none", markersize=7, label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=OKABE_ITO["black"],
            markersize=7,
            label="highlighted example units",
        )
    )
    axes[0, 2].legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("From linear power drive to activation and map SSI", fontsize=14)

    png = out_dir / "checkpoint_09_population_bridge.png"
    pdf = out_dir / "checkpoint_09_population_bridge.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_indices = parse_int_list(args.example_trace_indices)
    example_unit_text = str(args.example_unit_indices)
    if str(args.preference_source) == "parametric" and example_unit_text == DEFAULT_EXAMPLE_UNIT_INDICES:
        example_unit_text = DEFAULT_PARAMETRIC_EXAMPLE_UNIT_INDICES
    unit_indices = parse_int_list(example_unit_text)
    normal = load_normal_rows(
        Path(args.run_dir),
        Path(args.dense_fit_csv),
        Path(args.parametric_model_csv),
        out_dir,
        preference_source=str(args.preference_source),
        include_tf_edge_fits=bool(args.include_tf_edge_fits),
        sigma_octaves=float(args.tf_match_sigma_octaves),
        max_unit_tf_hz=float(args.max_unit_tf_hz),
        force_rebuild_traces=bool(args.force_rebuild_traces),
    )
    unit_means = unit_mean_rows(normal)
    groups = group_summary(unit_means)
    example_raw, example_means = select_example_rows(
        normal,
        unit_means,
        image_index=int(args.example_image_index),
        trace_indices=trace_indices,
        unit_indices=unit_indices,
    )

    relation_specs = [
        ("linear_power_drive", "delta_mean_activation"),
        ("delta_mean_activation", "unit_ssi_delta_absolute"),
        ("linear_power_drive", "unit_ssi_delta_absolute"),
    ]
    stats_rows = []
    for x_col, y_col in relation_specs:
        raw_stats = simple_linear_stats(normal, x_col, y_col)
        unit_stats = simple_linear_stats(
            unit_means.rename(columns={"map_ssi_change": "unit_ssi_delta_absolute"}),
            x_col,
            y_col if y_col != "unit_ssi_delta_absolute" else "unit_ssi_delta_absolute",
        )
        stats_rows.append(
            {
                "relation": f"{x_col}_to_{y_col}",
                "x": x_col,
                "y": y_col,
                "raw_n": raw_stats["n"],
                "raw_r": raw_stats["r"],
                "raw_r2": raw_stats["r2"],
                "raw_slope": raw_stats["slope"],
                "unit_mean_n": unit_stats["n"],
                "unit_mean_r": unit_stats["r"],
                "unit_mean_r2": unit_stats["r2"],
                "unit_mean_slope": unit_stats["slope"],
            }
        )
    stats = pd.DataFrame(stats_rows)

    raw_cols = [
        "image_index",
        "trace_index",
        "unit_index",
        "unit_label",
        "sf_group",
        "sf_group_label",
        "preferred_sf_cpd",
        "fit_pref_tf_hz",
        "linear_power_drive",
        "rms_linear_power_drive",
        "motion_induced_tf_mean_hz",
        "motion_induced_tf_peak_hz",
        "framewise_tf_match_mean",
        "framewise_tf_match_peak",
        "unit_sf_power_abs",
        "tf_match_fixed",
        "delta_mean_activation",
        "unit_ssi_delta_absolute",
        "mean_activation_normal",
        "mean_activation_stabilized",
        "unit_ssi_bits_per_spike",
    ]
    raw_table_path = out_dir / "checkpoint_09_population_raw_examples.csv"
    unit_table_path = out_dir / "checkpoint_09_population_unit_means.csv"
    group_table_path = out_dir / "checkpoint_09_population_group_summary.csv"
    example_raw_path = out_dir / "checkpoint_09_highlighted_example_raw_rows.csv"
    example_unit_path = out_dir / "checkpoint_09_highlighted_example_unit_means.csv"
    stats_path = out_dir / "checkpoint_09_population_relation_stats.csv"

    normal[raw_cols].to_csv(raw_table_path, index=False)
    unit_means.to_csv(unit_table_path, index=False)
    groups.to_csv(group_table_path, index=False)
    example_raw[raw_cols].to_csv(example_raw_path, index=False)
    example_means.to_csv(example_unit_path, index=False)
    stats.to_csv(stats_path, index=False)

    png, pdf = plot_population_bridge(normal, unit_means, example_raw, example_means, stats, out_dir, dpi=int(args.dpi))
    write_json(
        out_dir / "checkpoint_09_population_bridge_metadata.json",
        {
            "analysis": "population_bridge_linear_power_to_activation_and_map_ssi",
            "run_dir": Path(args.run_dir),
            "dense_fit_csv": Path(args.dense_fit_csv),
            "parametric_model_csv": Path(args.parametric_model_csv),
            "preference_source": str(args.preference_source),
            "include_tf_edge_fits": bool(args.include_tf_edge_fits),
            "max_unit_tf_hz": float(args.max_unit_tf_hz),
            "tf_match_sigma_octaves": float(args.tf_match_sigma_octaves),
            "linear_power_drive_definition": (
                "Mean over the 32-frame normal-motion block of image SF-band power at the unit "
                "preferred SF band times fixed-width TF match between framewise motion-induced TF "
                "and fitted unit TF preference"
            ),
            "rms_linear_power_drive_definition": (
                "Older trace-level proxy from the scorecard: image SF-band power times TF match "
                "at the stored characteristic motion TF"
            ),
            "activation_change_definition": "unit_mean_rate(original_natural_timing) - unit_mean_rate(stabilized_static)",
            "map_ssi_change_definition": "unit_ssi_bits_per_spike(original_natural_timing) - unit_ssi_bits_per_spike(stabilized_static)",
            "highlighted_example": {
                "image_index": int(args.example_image_index),
                "trace_indices": trace_indices,
                "unit_indices": unit_indices,
            },
            "outputs": {
                "figure_png": png,
                "figure_pdf": pdf,
                "raw_examples": raw_table_path,
                "unit_means": unit_table_path,
                "group_summary": group_table_path,
                "highlighted_example_raw_rows": example_raw_path,
                "highlighted_example_unit_means": example_unit_path,
                "relation_stats": stats_path,
            },
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {raw_table_path}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
