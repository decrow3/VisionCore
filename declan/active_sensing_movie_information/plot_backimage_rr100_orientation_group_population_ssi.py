#!/usr/bin/env python3
"""Plot orientation-group spike-weighted population SSI from cached RR100 maps.

This remakes the orientation-group z-score plot as raw population SSI.  The
group curve is

    sum(unit SSI * expected spikes) / sum(expected spikes)

within each orientation-tuning group and condition.  When a run's all-unit CSV
is incomplete, the script reconstructs the all-unit SSI table from the cached
instantaneous activation maps instead of rerunning the twin.
"""

from __future__ import annotations

import argparse
import csv
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPPOSING_AXIS0_RUN = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_opposite_axis0_v1"
)
DEFAULT_OPPOSING_AXIS1_RUN = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1"
)
DEFAULT_OUT_DIR = DEFAULT_OPPOSING_AXIS1_RUN / "population_ssi_summary" / "orientation_group_spike_weighted"
VALUE_COL = "displayed_movie_time_resolved_ssi_bits_per_spike"
RATE_COL = "displayed_movie_mean_rate"
SPIKES_COL = "displayed_movie_expected_spikes_arbitrary_dt"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opposing-axis0-run-dir", type=Path, default=DEFAULT_OPPOSING_AXIS0_RUN)
    parser.add_argument("--opposing-axis1-run-dir", type=Path, default=DEFAULT_OPPOSING_AXIS1_RUN)
    parser.add_argument("--orientation-groups-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--unit-inclusion",
        choices=("all", "zscore_usable"),
        default="all",
        help=(
            "all includes every classified unit in the spike-weighted population. "
            "zscore_usable matches the original z-score figure's subset by requiring "
            "nonzero within-axis SSI variance."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}


def required_ssi_columns() -> set[str]:
    return {
        "unit_index",
        "unit_label",
        "axis_mode",
        "display_scale",
        "condition_index",
        "condition_id",
        "along_scale",
        "across_scale",
        VALUE_COL,
        RATE_COL,
        SPIKES_COL,
    }


def unit_spatial_ssi_for_movie(rate_map: np.ndarray, *, bin_seconds: float = 1.0) -> dict[str, np.ndarray]:
    y = np.asarray(rate_map, dtype=np.float64)
    if y.ndim != 4:
        raise ValueError(f"Expected rate map with shape T x N x H x W, got {y.shape}")
    if np.nanmin(y) < -1e-7:
        raise ValueError(f"rate map contains negative values; min={float(np.nanmin(y)):.6g}")
    y = np.maximum(y, 0.0)
    t_max, n_units, height, width = y.shape
    flat = y.reshape(t_max, n_units, height * width)
    rbar = np.mean(flat, axis=2)
    gain = flat / (rbar[..., None] + EPS)
    unit_bits_t = np.mean(gain * np.log2(gain + EPS), axis=2)
    unit_expected = np.sum(rbar * float(bin_seconds), axis=0)
    unit_bits = np.sum(unit_bits_t * rbar * float(bin_seconds), axis=0) / np.maximum(unit_expected, EPS)
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_expected_spikes": unit_expected.astype(np.float32),
        "unit_mean_rate": np.mean(rbar, axis=0).astype(np.float32),
    }


def load_complete_ssi_table(run_dir: Path, *, out_dir: Path, tag: str) -> tuple[pd.DataFrame, Path, bool]:
    run_dir = Path(run_dir)
    csv_path = run_dir / "displayed_movie_instantaneous_ssi_all_units.csv"
    cache_path = run_dir / "cache" / "backimage_rr100_instantaneous_unit_maps.npz"
    expected_units = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            expected_units = int(payload["maps"].shape[2])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        missing = sorted(required_ssi_columns().difference(df.columns))
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")
        n_units = int(df["unit_index"].nunique())
        if expected_units is None or n_units >= expected_units:
            return normalize_ssi_frame(df), csv_path, False

    if not cache_path.exists():
        raise FileNotFoundError(f"Need either a complete all-unit CSV or map cache: {cache_path}")
    refs_path = run_dir / "condition_display_refs.csv"
    if not refs_path.exists():
        raise FileNotFoundError(refs_path)

    payload = load_npz(cache_path)
    maps = np.asarray(payload["maps"], dtype=np.float32)
    condition_id = np.asarray(payload["condition_id"]).astype(str)
    refs = pd.read_csv(refs_path)
    rows: list[dict[str, Any]] = []
    print(f"Recomputing all-unit SSI from {cache_path}", flush=True)
    for ref in refs.sort_values(["axis_mode", "display_scale"]).to_dict("records"):
        condition_idx = int(ref["condition_index"])
        print(
            f"  condition {condition_idx}: {ref['axis_mode']} scale={float(ref['display_scale']):g}",
            flush=True,
        )
        ssi = unit_spatial_ssi_for_movie(maps[condition_idx], bin_seconds=1.0)
        unit_bits = np.asarray(ssi["unit_bits_per_spike"], dtype=np.float64)
        unit_rate = np.asarray(ssi["unit_mean_rate"], dtype=np.float64)
        unit_spikes = np.asarray(ssi["unit_expected_spikes"], dtype=np.float64)
        for unit in range(unit_bits.shape[0]):
            rows.append(
                {
                    "unit_index": int(unit),
                    "unit_label": f"u{int(unit):03d}",
                    "axis_mode": str(ref["axis_mode"]),
                    "display_scale": float(ref["display_scale"]),
                    "condition_index": condition_idx,
                    "condition_id": str(condition_id[condition_idx]),
                    "along_scale": float(ref["along_scale"]),
                    "across_scale": float(ref["across_scale"]),
                    VALUE_COL: float(unit_bits[unit]),
                    RATE_COL: float(unit_rate[unit]),
                    SPIKES_COL: float(unit_spikes[unit]),
                }
            )
    out_csv = out_dir / f"{tag}_displayed_movie_instantaneous_ssi_all_units_recomputed.csv"
    write_csv(out_csv, rows)
    return normalize_ssi_frame(pd.DataFrame(rows)), out_csv, True


def normalize_ssi_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "unit_index",
        "display_scale",
        "condition_index",
        "along_scale",
        "across_scale",
        VALUE_COL,
        RATE_COL,
        SPIKES_COL,
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["unit_index"] = out["unit_index"].astype(int)
    out["unit_label"] = out["unit_label"].astype(str)
    out["axis_mode"] = out["axis_mode"].astype(str)
    out["condition_id"] = out["condition_id"].astype(str)
    out["information_numerator_bits_arbitrary_dt"] = out[VALUE_COL].astype(float) * out[SPIKES_COL].astype(float)
    return out


def load_orientation_groups(path: Path) -> pd.DataFrame:
    groups = pd.read_csv(path)
    required = {"unit_index", "unit_label", "orientation_group", "orientation_group_label", "orientation_group_rank"}
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    groups["unit_index"] = pd.to_numeric(groups["unit_index"], errors="coerce").astype(int)
    return groups.drop_duplicates("unit_index")


def add_axis_curve_usable_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["run_label", "axis_mode", "unit_index"]
    std = out.groupby(group_cols, sort=False)[VALUE_COL].transform(
        lambda values: float(np.nanstd(values.to_numpy(dtype=float), ddof=0))
    )
    out["ssi_curve_std_axis_mode"] = std.astype(float)
    out["ssi_curve_usable_axis_mode"] = np.isfinite(std.to_numpy(dtype=float)) & (std.to_numpy(dtype=float) > 1e-9)
    return out


def annotate_fixed_axis(df: pd.DataFrame, *, fixed_opposing_axis_scale: float, run_label: str, source_csv: Path) -> pd.DataFrame:
    out = df.copy()
    out["fixed_opposing_axis_scale"] = float(fixed_opposing_axis_scale)
    out["run_label"] = str(run_label)
    out["source_ssi_csv"] = str(source_csv)
    return out


def population_bits(sub: pd.DataFrame) -> float:
    numerator = sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)
    denom = sub[SPIKES_COL].to_numpy(dtype=np.float64)
    return float(np.nansum(numerator) / max(float(np.nansum(denom)), EPS))


def bootstrap_population_ci(
    sub: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    units = np.asarray(sorted(sub["unit_index"].astype(int).unique()), dtype=int)
    if units.size == 0 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    per_unit = (
        sub.groupby("unit_index", sort=True)[["information_numerator_bits_arbitrary_dt", SPIKES_COL]]
        .sum()
        .reindex(units)
    )
    numerators = per_unit["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)
    denominators = per_unit[SPIKES_COL].to_numpy(dtype=np.float64)
    sample_idx = rng.integers(0, units.size, size=(int(n_bootstrap), units.size))
    values = np.nansum(numerators[sample_idx], axis=1) / np.maximum(
        np.nansum(denominators[sample_idx], axis=1),
        EPS,
    )
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def build_group_summary(
    df: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "run_label",
        "fixed_opposing_axis_scale",
        "axis_mode",
        "display_scale",
        "orientation_group",
        "orientation_group_label",
    ]
    for key, sub in df.groupby(group_cols, sort=True):
        record = dict(zip(group_cols, key, strict=True))
        group_rank = int(sub["orientation_group_rank"].iloc[0])
        condition = sub.sort_values("unit_index").iloc[0]
        lo, hi = bootstrap_population_ci(sub, n_bootstrap=n_bootstrap, rng=rng)
        unit_values = sub[VALUE_COL].to_numpy(dtype=np.float64)
        n_finite = int(np.isfinite(unit_values).sum())
        unit_std = float(np.nanstd(unit_values, ddof=1)) if n_finite > 1 else float("nan")
        rows.append(
            {
                **record,
                "orientation_group_rank": group_rank,
                "condition_index": int(condition["condition_index"]),
                "condition_id": str(condition["condition_id"]),
                "along_scale": float(condition["along_scale"]),
                "across_scale": float(condition["across_scale"]),
                "n_units": int(sub["unit_index"].nunique()),
                "n_values": int(sub.shape[0]),
                "population_bits_per_spike_spike_weighted": population_bits(sub),
                "population_bits_arbitrary_dt": float(
                    np.nansum(sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64))
                ),
                "population_expected_spikes_arbitrary_dt": float(np.nansum(sub[SPIKES_COL].to_numpy(dtype=np.float64))),
                "mean_unit_bits_per_spike": float(np.nanmean(unit_values)),
                "sem_unit_bits_per_spike": float(unit_std / math.sqrt(n_finite)) if n_finite > 1 else float("nan"),
                "std_unit_bits_per_spike": unit_std,
                "median_unit_bits_per_spike": float(np.nanmedian(unit_values)),
                "unit_q25_bits_per_spike": float(np.nanquantile(unit_values, 0.25)),
                "unit_q75_bits_per_spike": float(np.nanquantile(unit_values, 0.75)),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            }
        )
    return rows


def panel_title(axis_mode: str, fixed_opposing_axis_scale: float) -> str:
    if axis_mode == "across_sweep":
        return f"scale across; along={float(fixed_opposing_axis_scale):g}"
    if axis_mode == "along_sweep":
        return f"scale along; across={float(fixed_opposing_axis_scale):g}"
    return axis_mode


def padded_limits(values: np.ndarray, *, lower_floor: float = 0.0, min_span: float = 0.01) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return lower_floor, max(lower_floor + min_span, 0.05)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    span = max(hi - lo, float(min_span))
    pad = 0.10 * span
    ymin = max(float(lower_floor), lo - pad)
    ymax = hi + pad
    if ymax <= ymin:
        ymax = ymin + float(min_span)
    return ymin, ymax


def plot_group_summary(
    *,
    out_dir: Path,
    all_units: pd.DataFrame,
    summary: pd.DataFrame,
    dpi: int,
    show_unit_lines: bool = True,
    scale_to_means: bool = False,
    filename_stem: str = "backimage_rr100_orientation_group_spike_weighted_population_ssi",
) -> tuple[Path, Path]:
    colors = {
        "contour_biased": "#18a6b8",
        "across_biased": "#d95f02",
        "off_axis_or_mixed": "#5b8a2f",
    }
    group_order = (
        summary[["orientation_group", "orientation_group_label", "orientation_group_rank"]]
        .drop_duplicates()
        .sort_values("orientation_group_rank")
    )
    panels = [
        ("across_sweep", 0.0),
        ("across_sweep", 1.0),
        ("along_sweep", 0.0),
        ("along_sweep", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), sharey=True, constrained_layout=True)

    y_values = summary["population_bits_per_spike_spike_weighted"].to_numpy(dtype=float)
    if scale_to_means:
        plot_values = np.concatenate(
            [
                y_values,
                summary["bootstrap_ci_low"].to_numpy(dtype=float),
                summary["bootstrap_ci_high"].to_numpy(dtype=float),
            ]
        )
        ymin, ymax = padded_limits(plot_values, lower_floor=0.0, min_span=0.01)
    else:
        unit_values = all_units[VALUE_COL].to_numpy(dtype=float)
        finite_unit_values = unit_values[np.isfinite(unit_values)]
        finite_y_values = y_values[np.isfinite(y_values)]
        ymin = 0.0
        ymax = max(
            float(np.nanmax(finite_unit_values)) if finite_unit_values.size else 0.0,
            float(np.nanmax(finite_y_values)) * 1.25 if finite_y_values.size else 0.0,
        )
        ymax = max(ymax, 0.05)

    for ax, (axis_mode, fixed) in zip(axes.ravel(), panels, strict=True):
        panel_units = all_units[
            (all_units["axis_mode"].astype(str) == axis_mode)
            & np.isclose(all_units["fixed_opposing_axis_scale"].astype(float), fixed)
        ].copy()
        panel_summary = summary[
            (summary["axis_mode"].astype(str) == axis_mode)
            & np.isclose(summary["fixed_opposing_axis_scale"].astype(float), fixed)
        ].copy()
        for group_row in group_order.to_dict("records"):
            group = str(group_row["orientation_group"])
            label = str(group_row["orientation_group_label"])
            color = colors.get(group, "0.35")
            if show_unit_lines:
                unit_sub = panel_units[panel_units["orientation_group"].astype(str) == group]
                for _unit, curve in unit_sub.groupby("unit_index", sort=False):
                    curve = curve.sort_values("display_scale")
                    ax.plot(
                        curve["display_scale"].to_numpy(dtype=float),
                        curve[VALUE_COL].to_numpy(dtype=float),
                        color=color,
                        alpha=0.075,
                        linewidth=0.55,
                        zorder=1,
                    )

            group_sub = panel_summary[panel_summary["orientation_group"].astype(str) == group].sort_values("display_scale")
            if group_sub.empty:
                continue
            x = group_sub["display_scale"].to_numpy(dtype=float)
            y = group_sub["population_bits_per_spike_spike_weighted"].to_numpy(dtype=float)
            lo = group_sub["bootstrap_ci_low"].to_numpy(dtype=float)
            hi = group_sub["bootstrap_ci_high"].to_numpy(dtype=float)
            n_units = int(group_sub["n_units"].iloc[0])
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0.0, zorder=5)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.25,
                markersize=4.5,
                color=color,
                label=f"{label} (n={n_units})",
                zorder=10,
            )
        ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
        ax.grid(True, color="0.9", linewidth=0.75)
        ax.set_title(panel_title(axis_mode, fixed), fontsize=11)
        ax.set_xlabel("scale")
        ax.set_ylim(ymin, ymax * (1.05 if not scale_to_means else 1.0))
    axes[0, 0].set_ylabel("SSI bits/spike")
    axes[1, 0].set_ylabel("SSI bits/spike")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    subtitle = (
        "thick lines are group spike-weighted population SSI; shaded bands are unit-bootstrap 95% CI; y-axis zoomed"
        if not show_unit_lines
        else "thin lines are raw units; thick lines are group spike-weighted population SSI; shaded bands are unit-bootstrap 95% CI"
    )
    fig.suptitle(
        "BackImage RR100 displayed-movie instantaneous SSI grouped by orientation tuning\n"
        f"{subtitle}",
        fontsize=12,
    )
    png = out_dir / f"{filename_stem}.png"
    pdf = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_group_raw_mean_summary(
    *,
    out_dir: Path,
    all_units: pd.DataFrame,
    summary: pd.DataFrame,
    dpi: int,
    show_unit_lines: bool = True,
    scale_to_means: bool = False,
    filename_stem: str = "backimage_rr100_orientation_group_raw_mean_ssi",
) -> tuple[Path, Path]:
    colors = {
        "contour_biased": "#18a6b8",
        "across_biased": "#d95f02",
        "off_axis_or_mixed": "#5b8a2f",
    }
    group_order = (
        summary[["orientation_group", "orientation_group_label", "orientation_group_rank"]]
        .drop_duplicates()
        .sort_values("orientation_group_rank")
    )
    panels = [
        ("across_sweep", 0.0),
        ("across_sweep", 1.0),
        ("along_sweep", 0.0),
        ("along_sweep", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), sharey=True, constrained_layout=True)

    mean_values = summary["mean_unit_bits_per_spike"].to_numpy(dtype=float)
    if scale_to_means:
        sem_values = summary["sem_unit_bits_per_spike"].to_numpy(dtype=float)
        plot_values = np.concatenate([mean_values, mean_values - sem_values, mean_values + sem_values])
        ymin, ymax = padded_limits(plot_values, lower_floor=0.0, min_span=0.01)
    else:
        unit_values = all_units[VALUE_COL].to_numpy(dtype=float)
        finite_unit_values = unit_values[np.isfinite(unit_values)]
        finite_mean_values = mean_values[np.isfinite(mean_values)]
        ymin = 0.0
        ymax = max(
            float(np.nanmax(finite_unit_values)) if finite_unit_values.size else 0.0,
            float(np.nanmax(finite_mean_values)) * 1.25 if finite_mean_values.size else 0.0,
        )
        ymax = max(ymax, 0.05)

    for ax, (axis_mode, fixed) in zip(axes.ravel(), panels, strict=True):
        panel_units = all_units[
            (all_units["axis_mode"].astype(str) == axis_mode)
            & np.isclose(all_units["fixed_opposing_axis_scale"].astype(float), fixed)
        ].copy()
        panel_summary = summary[
            (summary["axis_mode"].astype(str) == axis_mode)
            & np.isclose(summary["fixed_opposing_axis_scale"].astype(float), fixed)
        ].copy()
        for group_row in group_order.to_dict("records"):
            group = str(group_row["orientation_group"])
            label = str(group_row["orientation_group_label"])
            color = colors.get(group, "0.35")
            if show_unit_lines:
                unit_sub = panel_units[panel_units["orientation_group"].astype(str) == group]
                for _unit, curve in unit_sub.groupby("unit_index", sort=False):
                    curve = curve.sort_values("display_scale")
                    ax.plot(
                        curve["display_scale"].to_numpy(dtype=float),
                        curve[VALUE_COL].to_numpy(dtype=float),
                        color=color,
                        alpha=0.075,
                        linewidth=0.55,
                        zorder=1,
                    )

            group_sub = panel_summary[panel_summary["orientation_group"].astype(str) == group].sort_values("display_scale")
            if group_sub.empty:
                continue
            x = group_sub["display_scale"].to_numpy(dtype=float)
            y = group_sub["mean_unit_bits_per_spike"].to_numpy(dtype=float)
            sem = group_sub["sem_unit_bits_per_spike"].to_numpy(dtype=float)
            lo = y - sem
            hi = y + sem
            n_units = int(group_sub["n_units"].iloc[0])
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0.0, zorder=5)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.25,
                markersize=4.5,
                color=color,
                label=f"{label} (n={n_units})",
                zorder=10,
            )
        ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
        ax.grid(True, color="0.9", linewidth=0.75)
        ax.set_title(panel_title(axis_mode, fixed), fontsize=11)
        ax.set_xlabel("scale")
        ax.set_ylim(ymin, ymax * (1.05 if not scale_to_means else 1.0))
    axes[0, 0].set_ylabel("SSI bits/spike")
    axes[1, 0].set_ylabel("SSI bits/spike")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    subtitle = (
        "thick lines are equal-unit raw mean SSI +/- SEM; y-axis zoomed"
        if not show_unit_lines
        else "thin lines are raw units; thick lines are equal-unit raw mean SSI +/- SEM"
    )
    fig.suptitle(
        "BackImage RR100 displayed-movie instantaneous SSI grouped by orientation tuning\n"
        f"{subtitle}",
        fontsize=12,
    )
    png = out_dir / f"{filename_stem}.png"
    pdf = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups_csv = (
        Path(args.orientation_groups_csv)
        if args.orientation_groups_csv is not None
        else Path(args.opposing_axis1_run_dir) / "orientation_tuning_groups.csv"
    )
    groups = load_orientation_groups(groups_csv)
    rng = np.random.default_rng(int(args.bootstrap_seed))

    df0, csv0, recomputed0 = load_complete_ssi_table(
        Path(args.opposing_axis0_run_dir),
        out_dir=out_dir,
        tag="opposing_axis0",
    )
    df1, csv1, recomputed1 = load_complete_ssi_table(
        Path(args.opposing_axis1_run_dir),
        out_dir=out_dir,
        tag="opposing_axis1",
    )
    df0 = annotate_fixed_axis(df0, fixed_opposing_axis_scale=0.0, run_label="opposing_axis0", source_csv=csv0)
    df1 = annotate_fixed_axis(df1, fixed_opposing_axis_scale=1.0, run_label="opposing_axis1", source_csv=csv1)
    all_units = pd.concat([df0, df1], ignore_index=True)
    all_units = all_units.merge(
        groups[
            [
                "unit_index",
                "orientation_group",
                "orientation_group_label",
                "orientation_group_rank",
                "preferred_orientation_deg",
                "orientation_selectivity_index",
                "preferred_delta_from_contour_deg",
                "preferred_delta_from_across_deg",
            ]
        ],
        on="unit_index",
        how="left",
        validate="many_to_one",
    )
    missing_group = all_units["orientation_group"].isna()
    if bool(missing_group.any()):
        count = int(all_units.loc[missing_group, "unit_index"].nunique())
        raise ValueError(f"Missing orientation-group labels for {count} units.")

    all_units = add_axis_curve_usable_flags(all_units)
    all_classified_units_csv = out_dir / "orientation_group_all_classified_unit_ssi_long.csv"
    all_units.to_csv(all_classified_units_csv, index=False)
    included_units = all_units.copy()
    if str(args.unit_inclusion) == "zscore_usable":
        included_units = included_units[included_units["ssi_curve_usable_axis_mode"].astype(bool)].copy()

    all_units_csv = out_dir / "orientation_group_all_unit_ssi_long.csv"
    included_units.to_csv(all_units_csv, index=False)
    summary_rows = build_group_summary(included_units, n_bootstrap=int(args.n_bootstrap), rng=rng)
    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "orientation_group_spike_weighted_population_ssi_summary.csv"
    summary.to_csv(summary_csv, index=False)
    png, pdf = plot_group_summary(out_dir=out_dir, all_units=included_units, summary=summary, dpi=int(args.dpi))
    weighted_means_only_png, weighted_means_only_pdf = plot_group_summary(
        out_dir=out_dir,
        all_units=included_units,
        summary=summary,
        dpi=int(args.dpi),
        show_unit_lines=False,
        scale_to_means=True,
        filename_stem="backimage_rr100_orientation_group_spike_weighted_population_ssi_means_only",
    )
    raw_mean_png, raw_mean_pdf = plot_group_raw_mean_summary(
        out_dir=out_dir,
        all_units=included_units,
        summary=summary,
        dpi=int(args.dpi),
    )
    raw_mean_means_only_png, raw_mean_means_only_pdf = plot_group_raw_mean_summary(
        out_dir=out_dir,
        all_units=included_units,
        summary=summary,
        dpi=int(args.dpi),
        show_unit_lines=False,
        scale_to_means=True,
        filename_stem="backimage_rr100_orientation_group_raw_mean_ssi_means_only",
    )

    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_rr100_orientation_group_spike_weighted_population_ssi",
            "opposing_axis0_run_dir": Path(args.opposing_axis0_run_dir),
            "opposing_axis1_run_dir": Path(args.opposing_axis1_run_dir),
            "orientation_groups_csv": groups_csv,
            "opposing_axis0_ssi_csv": csv0,
            "opposing_axis1_ssi_csv": csv1,
            "opposing_axis0_recomputed_from_map_cache": bool(recomputed0),
            "opposing_axis1_recomputed_from_map_cache": bool(recomputed1),
            "unit_inclusion": str(args.unit_inclusion),
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "contract": {
                "group_population_bits_per_spike": "sum(unit_bits_per_spike * unit_expected_spikes) / sum(unit_expected_spikes) within orientation group",
                "group_raw_mean_bits_per_spike": "equal-unit arithmetic mean of unit SSI bits/spike within orientation group",
                "thin_unit_lines": "raw displayed-movie instantaneous unit SSI bits/spike, not z-scored",
                "unit_inclusion": (
                    "all classified units"
                    if str(args.unit_inclusion) == "all"
                    else "original z-score-plot subset: units with nonzero within-axis SSI variance"
                ),
                "bootstrap_ci": "unit-resampled 95% CI for group spike-weighted population SSI",
            },
            "outputs": {
                "all_classified_unit_long_csv": all_classified_units_csv,
                "all_unit_long_csv": all_units_csv,
                "group_summary_csv": summary_csv,
                "figure_png": png,
                "figure_pdf": pdf,
                "weighted_means_only_figure_png": weighted_means_only_png,
                "weighted_means_only_figure_pdf": weighted_means_only_pdf,
                "raw_mean_figure_png": raw_mean_png,
                "raw_mean_figure_pdf": raw_mean_pdf,
                "raw_mean_means_only_figure_png": raw_mean_means_only_png,
                "raw_mean_means_only_figure_pdf": raw_mean_means_only_pdf,
            },
        },
    )
    print(f"Wrote orientation-group spike-weighted SSI figure: {pdf}")
    print(f"Wrote orientation-group spike-weighted means-only SSI figure: {weighted_means_only_pdf}")
    print(f"Wrote orientation-group raw-mean SSI figure: {raw_mean_pdf}")
    print(f"Wrote orientation-group raw-mean means-only SSI figure: {raw_mean_means_only_pdf}")


if __name__ == "__main__":
    main()
