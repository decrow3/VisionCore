#!/usr/bin/env python3
"""Compare across-vs-along population SSI for opposing-axis 1x and 0x sweeps.

The left panel uses the standard single-axis sweeps where the opposing axis is
held at 1x.  The right panel uses the nulled-opposing-axis cache where the
opposing axis is held at 0x.  In both panels the plotted quantity is the
spike-weighted population SSI:

    sum(unit SSI bits/spike * expected spikes) / sum(expected spikes)
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
DEFAULT_AXIS1_RUN = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1"
)
DEFAULT_AXIS0_RUN = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_opposite_axis0_v1"
)
DEFAULT_OUT_DIR = DEFAULT_AXIS1_RUN / "population_ssi_summary"
VALUE_COL = "displayed_movie_time_resolved_ssi_bits_per_spike"
RATE_COL = "displayed_movie_mean_rate"
SPIKES_COL = "displayed_movie_expected_spikes_arbitrary_dt"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opposing-axis1-population-csv",
        type=Path,
        default=DEFAULT_AXIS1_RUN / "population_ssi_summary" / "population_scale_summary.csv",
    )
    parser.add_argument("--opposing-axis0-run-dir", type=Path, default=DEFAULT_AXIS0_RUN)
    parser.add_argument(
        "--opposing-axis0-ssi-csv",
        type=Path,
        default=DEFAULT_AXIS1_RUN
        / "population_ssi_summary"
        / "orientation_group_spike_weighted"
        / "opposing_axis0_displayed_movie_instantaneous_ssi_all_units_recomputed.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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


def normalize_ssi_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = sorted(required_ssi_columns().difference(out.columns))
    if missing:
        raise ValueError(f"Missing required all-unit SSI columns: {missing}")
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


def load_npz_without_identity(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}


def load_complete_axis0_ssi_table(axis0_run_dir: Path, ssi_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, Path, bool]:
    axis0_run_dir = Path(axis0_run_dir)
    ssi_csv = Path(ssi_csv)
    cache_path = axis0_run_dir / "cache" / "backimage_rr100_instantaneous_unit_maps.npz"
    refs_path = axis0_run_dir / "condition_display_refs.csv"
    expected_units: int | None = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            expected_units = int(payload["maps"].shape[2])

    if ssi_csv.exists():
        df = normalize_ssi_frame(pd.read_csv(ssi_csv))
        if expected_units is None or int(df["unit_index"].nunique()) >= expected_units:
            return df, ssi_csv, False

    fallback_csv = axis0_run_dir / "displayed_movie_instantaneous_ssi_all_units.csv"
    if fallback_csv.exists():
        df = normalize_ssi_frame(pd.read_csv(fallback_csv))
        if expected_units is None or int(df["unit_index"].nunique()) >= expected_units:
            return df, fallback_csv, False

    if not cache_path.exists():
        raise FileNotFoundError(f"Need a complete all-unit CSV or map cache: {cache_path}")
    if not refs_path.exists():
        raise FileNotFoundError(refs_path)

    payload = load_npz_without_identity(cache_path)
    maps = np.asarray(payload["maps"], dtype=np.float32)
    condition_id = np.asarray(payload["condition_id"]).astype(str)
    refs = pd.read_csv(refs_path)
    rows: list[dict[str, Any]] = []
    print(f"Recomputing 0x all-unit SSI from {cache_path}", flush=True)
    for ref in refs.sort_values(["axis_mode", "display_scale"]).to_dict("records"):
        condition_idx = int(ref["condition_index"])
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
    out_csv = out_dir / "opposing_axis0_displayed_movie_instantaneous_ssi_all_units_recomputed.csv"
    write_csv(out_csv, rows)
    return normalize_ssi_frame(pd.DataFrame(rows)), out_csv, True


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else float("nan")


def finite_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, q)) if arr.size else float("nan")


def population_row(sub: pd.DataFrame) -> dict[str, Any]:
    y = sub[VALUE_COL].to_numpy(dtype=np.float64)
    w = sub[SPIKES_COL].to_numpy(dtype=np.float64)
    numerator = sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(w) & np.isfinite(numerator) & (w >= 0.0)
    y = y[valid]
    w = w[valid]
    numerator = numerator[valid]
    denom = float(np.nansum(w))
    bits = float(np.nansum(numerator))
    return {
        "n_units": int(sub.loc[valid, "unit_index"].nunique()) if valid.size else 0,
        "n_values": int(y.size),
        "population_bits_per_spike_spike_weighted": bits / max(denom, EPS),
        "population_bits_arbitrary_dt": bits,
        "population_expected_spikes_arbitrary_dt": denom,
        "equal_weight_mean_bits_per_spike": finite_mean(y),
        "median_unit_bits_per_spike": finite_median(y),
        "q25_unit_bits_per_spike": finite_quantile(y, 0.25),
        "q75_unit_bits_per_spike": finite_quantile(y, 0.75),
        "mean_unit_rate": finite_mean(sub[RATE_COL].to_numpy(dtype=np.float64)),
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


def load_axis1_population_summary(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    pop = pd.read_csv(path)
    required = {"axis_mode", "display_scale", "population_bits_per_spike_spike_weighted"}
    missing = sorted(required.difference(pop.columns))
    if missing:
        raise ValueError(f"Missing required population columns in {path}: {missing}")
    return pop


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


def plot_side_by_side(out_dir: Path, axis1_pop: pd.DataFrame, axis0_pop: pd.DataFrame, *, dpi: int) -> tuple[Path, Path]:
    panels = [
        ("opposing axis fixed at 1x", axis1_pop),
        ("opposing axis fixed at 0x", axis0_pop),
    ]
    colors = {"across_sweep": "#1f77b4", "along_sweep": "#d95f02"}
    all_values = np.concatenate(
        [panel["population_bits_per_spike_spike_weighted"].to_numpy(dtype=float) for _title, panel in panels]
    )
    ymin, ymax = padded_limits(all_values, lower_floor=0.0, min_span=0.002)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True, constrained_layout=True)
    for ax, (title, pop) in zip(axes, panels, strict=True):
        for axis_mode in ("across_sweep", "along_sweep"):
            sub = pop[pop["axis_mode"].astype(str) == axis_mode].sort_values("display_scale")
            if sub.empty:
                continue
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
        ax.set_title(title)
        ax.set_xlabel("display scale")
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("SSI bits/spike")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("BackImage RR100 spike-weighted population SSI\nacross vs along; y-axis shared and zoomed", fontsize=12)

    png = out_dir / "backimage_rr100_population_bits_per_spike_opposing_axis_comparison.png"
    pdf = out_dir / "backimage_rr100_population_bits_per_spike_opposing_axis_comparison.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axis1_pop = load_axis1_population_summary(Path(args.opposing_axis1_population_csv)).copy()
    axis1_pop["opposing_axis_scale"] = 1.0
    axis1_pop["opposing_axis_label"] = "opposing axis fixed at 1x"

    axis0_ssi, axis0_ssi_csv, axis0_recomputed = load_complete_axis0_ssi_table(
        Path(args.opposing_axis0_run_dir),
        Path(args.opposing_axis0_ssi_csv),
        out_dir,
    )
    axis0_rows = build_population_scale_summary(axis0_ssi)
    axis0_pop = pd.DataFrame(axis0_rows)
    axis0_pop["opposing_axis_scale"] = 0.0
    axis0_pop["opposing_axis_label"] = "opposing axis fixed at 0x"

    axis0_summary_csv = out_dir / "population_scale_summary_opposing_axis0.csv"
    axis0_pop.to_csv(axis0_summary_csv, index=False)

    comparison = pd.concat([axis1_pop, axis0_pop], ignore_index=True, sort=False)
    comparison_csv = out_dir / "population_scale_summary_opposing_axis_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)

    png, pdf = plot_side_by_side(out_dir, axis1_pop, axis0_pop, dpi=int(args.dpi))
    summary_json = out_dir / "population_bits_per_spike_opposing_axis_comparison_summary.json"
    write_json(
        summary_json,
        {
            "analysis": "backimage_rr100_population_bits_per_spike_opposing_axis_comparison",
            "contract": {
                "population_bits_per_spike_spike_weighted": (
                    "sum(unit_bits_per_spike * unit_expected_spikes) / sum(unit_expected_spikes)"
                ),
                "left_panel": "single-axis sweep with opposing axis fixed at 1x",
                "right_panel": "single-axis sweep with opposing axis nulled/fixed at 0x",
            },
            "opposing_axis1_population_csv": Path(args.opposing_axis1_population_csv),
            "opposing_axis0_run_dir": Path(args.opposing_axis0_run_dir),
            "opposing_axis0_ssi_csv": axis0_ssi_csv,
            "opposing_axis0_recomputed_from_map_cache": bool(axis0_recomputed),
            "outputs": {
                "opposing_axis0_population_csv": axis0_summary_csv,
                "comparison_csv": comparison_csv,
                "figure_png": png,
                "figure_pdf": pdf,
            },
        },
    )
    print(f"Wrote opposing-axis population comparison: {pdf}")
    print(f"Wrote comparison CSV: {comparison_csv}")


if __name__ == "__main__":
    main()
