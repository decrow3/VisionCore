#!/usr/bin/env python3
"""Cache-backed population summaries for BackImage RR100 instantaneous SSI.

This script does not run the digital twin.  It reads the all-unit table written
by ``plot_backimage_rr100_instantaneous_unit_maps.py`` and aggregates SSI in
the same numerator/denominator form used by ``unit_spatial_ssi_for_movie``:

    population bits/spike = sum(unit bits/spike * expected spikes) / sum(expected spikes)

It also writes unit-relative summaries and floor-sensitivity tables so
low-rate or low-baseline units cannot silently dominate qualitative claims.
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
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1"
)
VALUE_COL = "displayed_movie_time_resolved_ssi_bits_per_spike"
RATE_COL = "displayed_movie_mean_rate"
SPIKES_COL = "displayed_movie_expected_spikes_arbitrary_dt"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ssi-csv", type=Path, default=None)
    parser.add_argument("--orientation-groups-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--reference-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-scale", type=float, default=None)
    parser.add_argument("--ssi-floors", type=str, default="0,0.02,0.05,0.1")
    parser.add_argument("--rate-floors", type=str, default="0,0.01,0.05,0.1")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


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


def finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, q)) if arr.size else float("nan")


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else float("nan")


def finite_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def ci(values: np.ndarray, q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(arr, q)
    return float(lo), float(hi)


def bootstrap_unit_stats(delta: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> dict[str, float]:
    arr = np.asarray(delta, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0 or int(n_bootstrap) <= 0:
        return {
            "median_delta_boot_ci_low": float("nan"),
            "median_delta_boot_ci_high": float("nan"),
            "fraction_positive_boot_ci_low": float("nan"),
            "fraction_positive_boot_ci_high": float("nan"),
        }
    med = np.empty(int(n_bootstrap), dtype=np.float64)
    frac = np.empty(int(n_bootstrap), dtype=np.float64)
    for idx in range(int(n_bootstrap)):
        sample = arr[rng.integers(0, n, size=n)]
        med[idx] = np.nanmedian(sample)
        frac[idx] = np.nanmean(sample > 0.0)
    med_lo, med_hi = ci(med)
    frac_lo, frac_hi = ci(frac)
    return {
        "median_delta_boot_ci_low": med_lo,
        "median_delta_boot_ci_high": med_hi,
        "fraction_positive_boot_ci_low": frac_lo,
        "fraction_positive_boot_ci_high": frac_hi,
    }


def required_columns() -> set[str]:
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


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None, Path, Path]:
    run_dir = Path(args.run_dir)
    ssi_csv = Path(args.ssi_csv) if args.ssi_csv is not None else run_dir / "displayed_movie_instantaneous_ssi_all_units.csv"
    if not ssi_csv.exists():
        raise FileNotFoundError(f"Missing all-unit SSI table: {ssi_csv}")
    df = pd.read_csv(ssi_csv)
    missing = sorted(required_columns().difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {ssi_csv}: {missing}")
    for col in ("unit_index", "display_scale", "condition_index", "along_scale", "across_scale", VALUE_COL, RATE_COL, SPIKES_COL):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_index"] = df["unit_index"].astype(int)
    df["unit_label"] = df["unit_label"].astype(str)
    df["axis_mode"] = df["axis_mode"].astype(str)
    df["condition_id"] = df["condition_id"].astype(str)
    df["information_numerator_bits_arbitrary_dt"] = df[VALUE_COL].astype(float) * df[SPIKES_COL].astype(float)

    groups_csv = (
        Path(args.orientation_groups_csv)
        if args.orientation_groups_csv is not None
        else run_dir / "orientation_tuning_groups.csv"
    )
    groups: pd.DataFrame | None = None
    if groups_csv.exists():
        groups = pd.read_csv(groups_csv)
        if "unit_index" in groups.columns:
            groups["unit_index"] = pd.to_numeric(groups["unit_index"], errors="coerce").astype("Int64")
            groups = groups.dropna(subset=["unit_index"]).copy()
            groups["unit_index"] = groups["unit_index"].astype(int)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "population_ssi_summary"
    return df, groups, ssi_csv, out_dir


def population_row(sub: pd.DataFrame, *, prefix: str = "") -> dict[str, Any]:
    y = sub[VALUE_COL].to_numpy(dtype=np.float64)
    w = sub[SPIKES_COL].to_numpy(dtype=np.float64)
    numerator = sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(w) & np.isfinite(numerator) & (w >= 0.0)
    y = y[valid]
    w = w[valid]
    numerator = numerator[valid]
    denom = float(np.nansum(w))
    bits = float(np.nansum(numerator))
    unit_count = int(sub.loc[valid, "unit_index"].nunique()) if valid.size else 0
    return {
        f"{prefix}n_units": unit_count,
        f"{prefix}n_values": int(y.size),
        f"{prefix}population_bits_per_spike_spike_weighted": bits / max(denom, EPS),
        f"{prefix}population_bits_arbitrary_dt": bits,
        f"{prefix}population_expected_spikes_arbitrary_dt": denom,
        f"{prefix}equal_weight_mean_bits_per_spike": finite_mean(y),
        f"{prefix}median_unit_bits_per_spike": finite_median(y),
        f"{prefix}q25_unit_bits_per_spike": finite_quantile(y, 0.25),
        f"{prefix}q75_unit_bits_per_spike": finite_quantile(y, 0.75),
        f"{prefix}mean_unit_rate": finite_mean(sub[RATE_COL].to_numpy(dtype=np.float64)),
    }


def build_population_scale_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (axis_mode, scale), sub in df.groupby(["axis_mode", "display_scale"], sort=True):
        condition = sub.sort_values("unit_index").iloc[0]
        rows.append(
            {
                "axis_mode": str(axis_mode),
                "display_scale": float(scale),
                "condition_index": int(condition["condition_index"]),
                "condition_id": str(condition["condition_id"]),
                "along_scale": float(condition["along_scale"]),
                "across_scale": float(condition["across_scale"]),
                **population_row(sub),
            }
        )
    return rows


def reference_table(df: pd.DataFrame, reference_scale: float) -> pd.DataFrame:
    refs = df[np.isclose(df["display_scale"].astype(float), float(reference_scale))].copy()
    if refs.empty:
        raise ValueError(f"No rows matched --reference-scale={float(reference_scale):g}.")
    duplicated = refs.duplicated(["axis_mode", "unit_index"], keep=False)
    if bool(duplicated.any()):
        dup = refs.loc[duplicated, ["axis_mode", "unit_index", "condition_id"]].head()
        raise ValueError(f"Reference rows are not unique per unit/axis_mode. Examples:\n{dup}")
    cols = [
        "axis_mode",
        "unit_index",
        "condition_id",
        "along_scale",
        "across_scale",
        VALUE_COL,
        RATE_COL,
        SPIKES_COL,
        "information_numerator_bits_arbitrary_dt",
    ]
    out = refs[cols].copy()
    return out.rename(
        columns={
            "condition_id": "reference_condition_id",
            "along_scale": "reference_along_scale",
            "across_scale": "reference_across_scale",
            VALUE_COL: "reference_bits_per_spike",
            RATE_COL: "reference_mean_rate",
            SPIKES_COL: "reference_expected_spikes_arbitrary_dt",
            "information_numerator_bits_arbitrary_dt": "reference_bits_arbitrary_dt",
        }
    )


def merge_reference(df: pd.DataFrame, reference_scale: float) -> pd.DataFrame:
    refs = reference_table(df, reference_scale)
    merged = df.merge(refs, on=["axis_mode", "unit_index"], how="left", validate="many_to_one")
    missing = merged["reference_bits_per_spike"].isna()
    if bool(missing.any()):
        count = int(merged.loc[missing, "unit_index"].nunique())
        raise ValueError(f"Missing reference rows for {count} units.")
    merged["delta_bits_per_spike_vs_reference"] = (
        merged[VALUE_COL].astype(float) - merged["reference_bits_per_spike"].astype(float)
    )
    merged["fractional_delta_bits_per_spike_vs_reference"] = merged["delta_bits_per_spike_vs_reference"] / (
        np.abs(merged["reference_bits_per_spike"].astype(float)) + EPS
    )
    merged["bits_arbitrary_dt_delta_vs_reference"] = (
        merged["information_numerator_bits_arbitrary_dt"].astype(float) - merged["reference_bits_arbitrary_dt"].astype(float)
    )
    merged["mean_rate_delta_vs_reference"] = merged[RATE_COL].astype(float) - merged["reference_mean_rate"].astype(float)
    return merged


def build_unit_delta_rows(merged: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "unit_index",
        "unit_label",
        "axis_mode",
        "display_scale",
        "condition_index",
        "condition_id",
        "along_scale",
        "across_scale",
        VALUE_COL,
        SPIKES_COL,
        RATE_COL,
        "information_numerator_bits_arbitrary_dt",
        "reference_condition_id",
        "reference_along_scale",
        "reference_across_scale",
        "reference_bits_per_spike",
        "reference_mean_rate",
        "reference_expected_spikes_arbitrary_dt",
        "reference_bits_arbitrary_dt",
        "delta_bits_per_spike_vs_reference",
        "fractional_delta_bits_per_spike_vs_reference",
        "bits_arbitrary_dt_delta_vs_reference",
        "mean_rate_delta_vs_reference",
    ]
    extra_cols = [
        col
        for col in (
            "orientation_group",
            "orientation_group_label",
            "preferred_orientation_deg",
            "orientation_selectivity_index",
            "preferred_delta_from_contour_deg",
            "preferred_delta_from_across_deg",
        )
        if col in merged.columns
    ]
    return merged[cols + extra_cols].sort_values(["axis_mode", "unit_index", "display_scale"]).to_dict("records")


def build_unit_delta_summary(
    merged: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (axis_mode, scale), sub in merged.groupby(["axis_mode", "display_scale"], sort=True):
        condition = sub.sort_values("unit_index").iloc[0]
        delta = sub["delta_bits_per_spike_vs_reference"].to_numpy(dtype=np.float64)
        bits_delta = sub["bits_arbitrary_dt_delta_vs_reference"].to_numpy(dtype=np.float64)
        finite = delta[np.isfinite(delta)]
        rows.append(
            {
                "axis_mode": str(axis_mode),
                "display_scale": float(scale),
                "condition_index": int(condition["condition_index"]),
                "condition_id": str(condition["condition_id"]),
                "along_scale": float(condition["along_scale"]),
                "across_scale": float(condition["across_scale"]),
                "reference_condition_id": str(condition["reference_condition_id"]),
                "reference_along_scale": float(condition["reference_along_scale"]),
                "reference_across_scale": float(condition["reference_across_scale"]),
                "n_units": int(sub["unit_index"].nunique()),
                "mean_delta_bits_per_spike": finite_mean(delta),
                "median_delta_bits_per_spike": finite_median(delta),
                "q25_delta_bits_per_spike": finite_quantile(delta, 0.25),
                "q75_delta_bits_per_spike": finite_quantile(delta, 0.75),
                "fraction_positive_delta": float(np.nanmean(finite > 0.0)) if finite.size else float("nan"),
                "mean_bits_arbitrary_dt_delta": finite_mean(bits_delta),
                "sum_bits_arbitrary_dt_delta": float(np.nansum(bits_delta)),
                **bootstrap_unit_stats(delta, n_bootstrap=int(n_bootstrap), rng=rng),
            }
        )
    return rows


def build_floor_sensitivity(
    merged: pd.DataFrame,
    *,
    ssi_floors: list[float],
    rate_floors: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (axis_mode, scale), scale_sub in merged.groupby(["axis_mode", "display_scale"], sort=True):
        condition = scale_sub.sort_values("unit_index").iloc[0]
        for ssi_floor in ssi_floors:
            for rate_floor in rate_floors:
                sub = scale_sub[
                    (scale_sub["reference_bits_per_spike"].astype(float) >= float(ssi_floor))
                    & (scale_sub["reference_mean_rate"].astype(float) >= float(rate_floor))
                ].copy()
                if sub.empty:
                    continue
                ref_denom = float(np.nansum(sub["reference_expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64)))
                ref_bits = float(np.nansum(sub["reference_bits_arbitrary_dt"].to_numpy(dtype=np.float64)))
                pop = population_row(sub, prefix="filtered_")
                delta = sub["delta_bits_per_spike_vs_reference"].to_numpy(dtype=np.float64)
                filtered_pop = float(pop["filtered_population_bits_per_spike_spike_weighted"])
                ref_pop = ref_bits / max(ref_denom, EPS)
                rows.append(
                    {
                        "axis_mode": str(axis_mode),
                        "display_scale": float(scale),
                        "condition_index": int(condition["condition_index"]),
                        "condition_id": str(condition["condition_id"]),
                        "along_scale": float(condition["along_scale"]),
                        "across_scale": float(condition["across_scale"]),
                        "reference_condition_id": str(condition["reference_condition_id"]),
                        "ssi_reference_floor": float(ssi_floor),
                        "mean_rate_reference_floor": float(rate_floor),
                        **pop,
                        "filtered_reference_population_bits_per_spike_spike_weighted": ref_pop,
                        "filtered_population_delta_bits_per_spike_vs_reference": filtered_pop - ref_pop,
                        "filtered_population_bits_arbitrary_dt_delta_vs_reference": float(
                            np.nansum(sub["bits_arbitrary_dt_delta_vs_reference"].to_numpy(dtype=np.float64))
                        ),
                        "median_unit_delta_bits_per_spike": finite_median(delta),
                        "fraction_positive_unit_delta": float(np.nanmean(delta[np.isfinite(delta)] > 0.0))
                        if np.isfinite(delta).any()
                        else float("nan"),
                    }
                )
    return rows


def build_group_summaries(merged: pd.DataFrame) -> list[dict[str, Any]]:
    if "orientation_group_label" not in merged.columns:
        return []
    rows: list[dict[str, Any]] = []
    for (axis_mode, scale, group_label), sub in merged.groupby(
        ["axis_mode", "display_scale", "orientation_group_label"], sort=True, dropna=False
    ):
        condition = sub.sort_values("unit_index").iloc[0]
        delta = sub["delta_bits_per_spike_vs_reference"].to_numpy(dtype=np.float64)
        finite = delta[np.isfinite(delta)]
        rows.append(
            {
                "axis_mode": str(axis_mode),
                "display_scale": float(scale),
                "orientation_group_label": str(group_label),
                "orientation_group": str(condition.get("orientation_group", "")),
                "condition_index": int(condition["condition_index"]),
                "condition_id": str(condition["condition_id"]),
                **population_row(sub, prefix="group_"),
                "group_median_delta_bits_per_spike_vs_reference": finite_median(delta),
                "group_fraction_positive_delta": float(np.nanmean(finite > 0.0)) if finite.size else float("nan"),
            }
        )
    return rows


def add_orientation_groups(df: pd.DataFrame, groups: pd.DataFrame | None) -> pd.DataFrame:
    if groups is None:
        return df
    wanted = [
        col
        for col in (
            "unit_index",
            "orientation_group",
            "orientation_group_label",
            "orientation_group_rank",
            "preferred_orientation_deg",
            "orientation_selectivity_index",
            "preferred_delta_from_contour_deg",
            "preferred_delta_from_across_deg",
        )
        if col in groups.columns
    ]
    if "unit_index" not in wanted:
        return df
    return df.merge(groups[wanted].drop_duplicates("unit_index"), on="unit_index", how="left", validate="many_to_one")


def detect_static_condition(df: pd.DataFrame) -> dict[str, Any]:
    static = df[np.isclose(df["along_scale"].astype(float), 0.0) & np.isclose(df["across_scale"].astype(float), 0.0)]
    if static.empty:
        return {
            "static_condition_available": False,
            "static_condition_note": (
                "No along=0/across=0 condition is present in this table. A display scale of 0 in a one-axis sweep "
                "removes only that axis component while the other axis remains at 1."
            ),
        }
    conds = static[["condition_index", "condition_id", "axis_mode", "display_scale", "along_scale", "across_scale"]]
    return {
        "static_condition_available": True,
        "static_condition_rows": conds.drop_duplicates().to_dict("records"),
    }


def padded_limits(values: np.ndarray, *, lower_floor: float = 0.0, min_span: float = 0.002) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return lower_floor, lower_floor + float(min_span)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    span = max(hi - lo, float(min_span))
    pad = 0.12 * span
    ymin = max(float(lower_floor), lo - pad)
    ymax = hi + pad
    if ymax <= ymin:
        ymax = ymin + float(min_span)
    return ymin, ymax


def plot_population_bits_per_spike_overlay(
    out_dir: Path,
    population_rows: list[dict[str, Any]],
    *,
    dpi: int,
) -> tuple[Path, Path]:
    pop = pd.DataFrame(population_rows)
    axis_modes = [axis for axis in ("across_sweep", "along_sweep") if axis in set(pop["axis_mode"].astype(str))]
    colors = {"across_sweep": "#1f77b4", "along_sweep": "#d95f02"}
    values = pop["population_bits_per_spike_spike_weighted"].to_numpy(dtype=float)
    ymin, ymax = padded_limits(values, lower_floor=0.0, min_span=0.002)

    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    for axis_mode in axis_modes:
        sub = pop[pop["axis_mode"].astype(str) == axis_mode].sort_values("display_scale")
        ax.plot(
            sub["display_scale"].to_numpy(dtype=float),
            sub["population_bits_per_spike_spike_weighted"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=colors.get(axis_mode, "0.35"),
            label=axis_mode.replace("_", " "),
        )
    ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    ax.grid(True, color="0.9")
    ax.set_title("spike-weighted population SSI\nacross vs along; y-axis zoomed")
    ax.set_xlabel("display scale")
    ax.set_ylabel("SSI bits/spike")
    ax.set_ylim(ymin, ymax)
    ax.legend(frameon=False, fontsize=9)

    png = out_dir / "backimage_rr100_population_bits_per_spike_across_vs_along.png"
    pdf = out_dir / "backimage_rr100_population_bits_per_spike_across_vs_along.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_summary(
    out_dir: Path,
    population_rows: list[dict[str, Any]],
    unit_summary_rows: list[dict[str, Any]],
    floor_rows: list[dict[str, Any]],
    *,
    endpoint_scale: float,
    ssi_floors: list[float],
    rate_floors: list[float],
    dpi: int,
) -> tuple[Path, Path]:
    pop = pd.DataFrame(population_rows)
    unit = pd.DataFrame(unit_summary_rows)
    floors = pd.DataFrame(floor_rows)
    axis_modes = [axis for axis in ("across_sweep", "along_sweep") if axis in set(pop["axis_mode"].astype(str))]

    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    for col, axis_mode in enumerate(axis_modes[:2]):
        ax = fig.add_subplot(gs[0, col])
        sub = pop[pop["axis_mode"].astype(str) == axis_mode].sort_values("display_scale")
        x = sub["display_scale"].to_numpy(dtype=float)
        ax.plot(
            x,
            sub["population_bits_per_spike_spike_weighted"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
            label="spike-weighted population",
        )
        ax.plot(
            x,
            sub["median_unit_bits_per_spike"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            color="#d95f02",
            label="median unit",
        )
        ax.fill_between(
            x,
            sub["q25_unit_bits_per_spike"].to_numpy(dtype=float),
            sub["q75_unit_bits_per_spike"].to_numpy(dtype=float),
            color="0.7",
            alpha=0.18,
            linewidth=0.0,
            label="unit IQR",
        )
        ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
        ax.grid(True, color="0.9")
        ax.set_title(axis_mode.replace("_", " "))
        ax.set_xlabel("display scale")
        ax.set_ylabel("SSI bits/spike")
        if col == 0:
            ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    for axis_mode, color in zip(axis_modes, ("#1f77b4", "#d95f02"), strict=False):
        sub = pop[pop["axis_mode"].astype(str) == axis_mode].sort_values("display_scale")
        ax.plot(
            sub["display_scale"].to_numpy(dtype=float),
            sub["population_bits_arbitrary_dt"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            color=color,
            label=axis_mode.replace("_", " "),
        )
    ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    ax.grid(True, color="0.9")
    ax.set_title("population bits")
    ax.set_xlabel("display scale")
    ax.set_ylabel("bits / displayed movie (arbitrary dt)")
    ax.legend(frameon=False, fontsize=8)

    for col, axis_mode in enumerate(axis_modes[:2]):
        ax = fig.add_subplot(gs[1, col])
        sub = floors[
            (floors["axis_mode"].astype(str) == axis_mode)
            & np.isclose(floors["display_scale"].astype(float), float(endpoint_scale))
        ].copy()
        mat = sub.pivot(
            index="ssi_reference_floor",
            columns="mean_rate_reference_floor",
            values="filtered_population_delta_bits_per_spike_vs_reference",
        ).reindex(index=ssi_floors, columns=rate_floors)
        values = mat.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(values, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(rate_floors)), [f"{v:g}" for v in rate_floors])
        ax.set_yticks(range(len(ssi_floors)), [f"{v:g}" for v in ssi_floors])
        ax.set_xlabel("reference mean-rate floor")
        ax.set_ylabel("reference SSI floor")
        ax.set_title(f"{axis_mode.replace('_', ' ')} at {float(endpoint_scale):g}x\nspike-weighted delta vs reference")
        for iy, ssi_floor in enumerate(ssi_floors):
            for ix, rate_floor in enumerate(rate_floors):
                row = sub[
                    np.isclose(sub["ssi_reference_floor"].astype(float), float(ssi_floor))
                    & np.isclose(sub["mean_rate_reference_floor"].astype(float), float(rate_floor))
                ]
                if row.empty:
                    continue
                value = float(row["filtered_population_delta_bits_per_spike_vs_reference"].iloc[0])
                n = int(row["filtered_n_units"].iloc[0])
                ax.text(ix, iy, f"{value:+.3f}\nn={n}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[1, 2])
    for axis_mode, color in zip(axis_modes, ("#1f77b4", "#d95f02"), strict=False):
        sub = unit[unit["axis_mode"].astype(str) == axis_mode].sort_values("display_scale")
        x = sub["display_scale"].to_numpy(dtype=float)
        y = sub["median_delta_bits_per_spike"].to_numpy(dtype=float)
        lo = sub["median_delta_boot_ci_low"].to_numpy(dtype=float)
        hi = sub["median_delta_boot_ci_high"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0.0)
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=axis_mode.replace("_", " "))
    ax.axhline(0.0, color="0.35", linewidth=1.0)
    ax.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
    ax.grid(True, color="0.9")
    ax.set_title("typical-unit change")
    ax.set_xlabel("display scale")
    ax.set_ylabel("median unit delta bits/spike")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "BackImage RR100 displayed-movie instantaneous SSI population summary\n"
        "primary curve aggregates information numerator over expected-spike denominator; no mean-map SSI",
        fontsize=12,
    )
    png = out_dir / "backimage_rr100_population_ssi_principled_summary.png"
    pdf = out_dir / "backimage_rr100_population_ssi_principled_summary.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    df, groups, ssi_csv, out_dir = load_inputs(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_orientation_groups(df, groups)
    scales = sorted(float(v) for v in df["display_scale"].dropna().unique())
    endpoint_scale = float(args.endpoint_scale) if args.endpoint_scale is not None else float(max(scales))
    if not any(np.isclose(scales, endpoint_scale)):
        raise ValueError(f"--endpoint-scale={endpoint_scale:g} is not present in {ssi_csv}. Available: {scales}")
    ssi_floors = parse_float_list(str(args.ssi_floors))
    rate_floors = parse_float_list(str(args.rate_floors))

    merged = merge_reference(df, float(args.reference_scale))
    rng = np.random.default_rng(int(args.bootstrap_seed))

    population_rows = build_population_scale_summary(df)
    unit_delta_rows = build_unit_delta_rows(merged)
    unit_summary_rows = build_unit_delta_summary(
        merged,
        n_bootstrap=int(args.n_bootstrap),
        rng=rng,
    )
    floor_rows = build_floor_sensitivity(merged, ssi_floors=ssi_floors, rate_floors=rate_floors)
    group_rows = build_group_summaries(merged)

    population_csv = out_dir / "population_scale_summary.csv"
    unit_delta_csv = out_dir / "unit_relative_to_reference.csv"
    unit_summary_csv = out_dir / "unit_delta_summary.csv"
    floor_csv = out_dir / "floor_sensitivity_summary.csv"
    group_csv = out_dir / "orientation_group_summary.csv"
    write_csv(population_csv, population_rows)
    write_csv(unit_delta_csv, unit_delta_rows)
    write_csv(unit_summary_csv, unit_summary_rows)
    write_csv(floor_csv, floor_rows)
    if group_rows:
        write_csv(group_csv, group_rows)

    png, pdf = plot_summary(
        out_dir,
        population_rows,
        unit_summary_rows,
        floor_rows,
        endpoint_scale=endpoint_scale,
        ssi_floors=ssi_floors,
        rate_floors=rate_floors,
        dpi=int(args.dpi),
    )
    bits_per_spike_overlay_png, bits_per_spike_overlay_pdf = plot_population_bits_per_spike_overlay(
        out_dir,
        population_rows,
        dpi=int(args.dpi),
    )

    ref_rows = (
        merged[
            np.isclose(merged["display_scale"].astype(float), float(args.reference_scale))
        ][["axis_mode", "condition_id", "along_scale", "across_scale", "reference_condition_id"]]
        .drop_duplicates()
        .to_dict("records")
    )
    summary = {
        "input_ssi_csv": ssi_csv,
        "orientation_groups_csv": Path(args.orientation_groups_csv)
        if args.orientation_groups_csv is not None
        else Path(args.run_dir) / "orientation_tuning_groups.csv",
        "out_dir": out_dir,
        "reference_scale": float(args.reference_scale),
        "reference_rows": ref_rows,
        "endpoint_scale": endpoint_scale,
        "n_units": int(df["unit_index"].nunique()),
        "n_rows": int(df.shape[0]),
        "contracts": {
            "population_bits_per_spike_spike_weighted": (
                "sum(unit_bits_per_spike * unit_expected_spikes) / sum(unit_expected_spikes)"
            ),
            "population_bits_arbitrary_dt": "sum(unit_bits_per_spike * unit_expected_spikes)",
            "unit_delta": "unit_bits_per_spike(condition) - unit_bits_per_spike(reference condition within axis_mode)",
            "floor_sensitivity": "recompute population numerator/denominator after filtering units by reference SSI and rate floors",
        },
        **detect_static_condition(df),
        "outputs": {
            "population_scale_summary_csv": population_csv,
            "unit_relative_to_reference_csv": unit_delta_csv,
            "unit_delta_summary_csv": unit_summary_csv,
            "floor_sensitivity_summary_csv": floor_csv,
            "orientation_group_summary_csv": group_csv if group_rows else None,
            "summary_png": png,
            "summary_pdf": pdf,
            "bits_per_spike_overlay_png": bits_per_spike_overlay_png,
            "bits_per_spike_overlay_pdf": bits_per_spike_overlay_pdf,
        },
    }
    summary_json = out_dir / "summary.json"
    write_json(summary_json, summary)
    print(f"Wrote population SSI summary: {summary_json}")
    print(f"Wrote figure: {pdf}")
    print(f"Wrote bits/spike overlay figure: {bits_per_spike_overlay_pdf}")


if __name__ == "__main__":
    main()
