#!/usr/bin/env python3
"""Orientation-stratified SF x contour-alignment population SSI plots."""

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
DEFAULT_BASE = ROOT / "outputs/active_sensing_movie_information"
DEFAULT_OUT_DIR = DEFAULT_BASE / (
    "backimage_contour_axis_rr100_sf_contour_alignment_orientation_stratified_"
    "dynamic_log_gaussian_marginal_low0p05_high0p5_v1"
)
EPS = 1e-12


VIEW_SPECS = [
    {
        "view": "across_along0",
        "label": "across scale\nalong=0",
        "dir": "backimage_contour_axis_rr100_sf_contour_alignment_population_ssi_dynamic_log_gaussian_marginal_low0p05_high0p5_across_sweep_along0_v1",
        "x_col": "across_scale",
    },
    {
        "view": "across_along1",
        "label": "across scale\nalong=1",
        "dir": "backimage_contour_axis_rr100_sf_contour_alignment_population_ssi_dynamic_log_gaussian_marginal_low0p05_high0p5_across_sweep_along1_v1",
        "x_col": "across_scale",
    },
    {
        "view": "along_across0",
        "label": "along scale\nacross=0",
        "dir": "backimage_contour_axis_rr100_sf_contour_alignment_population_ssi_dynamic_log_gaussian_marginal_low0p05_high0p5_along_sweep_across0_v1",
        "x_col": "along_scale",
    },
    {
        "view": "along_across1",
        "label": "along scale\nacross=1",
        "dir": "backimage_contour_axis_rr100_sf_contour_alignment_population_ssi_dynamic_log_gaussian_marginal_low0p05_high0p5_along_sweep_across1_v1",
        "x_col": "along_scale",
    },
]


BANDS = [
    {
        "band": "near_horizontal_axis0pm15",
        "label": "near-horizontal contour axes\n0 +/- 15 deg",
        "description": "axial distance to 0 deg <= 15 deg",
    },
    {
        "band": "near_vertical_axis90pm15",
        "label": "near-vertical contour axes\n90 +/- 15 deg",
        "description": "abs(axis - 90 deg) <= 15 deg",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--across-along0-dir", type=Path, default=None)
    parser.add_argument("--across-along1-dir", type=Path, default=None)
    parser.add_argument("--along-across0-dir", type=Path, default=None)
    parser.add_argument("--along-across1-dir", type=Path, default=None)
    parser.add_argument(
        "--band-mode",
        choices=("dominant", "coarse30", "dominant_and_coarse30"),
        default="dominant_and_coarse30",
    )
    parser.add_argument("--coarse-bin-width-deg", type=float, default=30.0)
    parser.add_argument(
        "--min-fixations-per-band",
        type=int,
        default=20,
        help="Skip plotted band figures with fewer unique fixations than this. Summary rows are still written.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def view_specs_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    override_paths = {
        "across_along0": args.across_along0_dir,
        "across_along1": args.across_along1_dir,
        "along_across0": args.along_across0_dir,
        "along_across1": args.along_across1_dir,
    }
    specs: list[dict[str, Any]] = []
    for spec in VIEW_SPECS:
        out = dict(spec)
        override = override_paths[str(spec["view"])]
        out["csv_path"] = (
            Path(override) / "per_fixation_weighted_alignment_population_ssi.csv"
            if override is not None
            else Path(args.base_dir) / str(spec["dir"]) / "per_fixation_weighted_alignment_population_ssi.csv"
        )
        specs.append(out)
    return specs


def build_bands(mode: str, *, coarse_bin_width_deg: float) -> list[dict[str, Any]]:
    bands: list[dict[str, Any]] = []
    if mode in {"dominant", "dominant_and_coarse30"}:
        bands.extend(BANDS)
    if mode in {"coarse30", "dominant_and_coarse30"}:
        width = float(coarse_bin_width_deg)
        if width <= 0.0 or 180.0 % width > 1e-9:
            raise ValueError("--coarse-bin-width-deg must be a positive divisor of 180")
        for start in np.arange(0.0, 180.0, width):
            stop = float(start + width)
            bands.append(
                {
                    "band": f"axis_bin_{int(start):03d}_{int(stop):03d}",
                    "label": f"contour axes {start:g}-{stop:g} deg",
                    "description": f"{start:g} <= contour_axis_image_deg < {stop:g}",
                }
            )
    return bands


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


def sf_label(sf_group: str) -> str:
    return {"low_sf": "low SF", "high_sf": "high SF"}.get(str(sf_group), str(sf_group))


def alignment_label(alignment_group: str) -> str:
    return {
        "contour_aligned": "contour-aligned",
        "contour_orthogonal": "orthogonal",
    }.get(str(alignment_group), str(alignment_group))


def alignment_color(alignment_group: str) -> str:
    return {"contour_aligned": "#168a96", "contour_orthogonal": "#c06b2d"}.get(str(alignment_group), "0.35")


def band_mask(axis_deg: np.ndarray, band: str) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64) % 180.0
    if band == "near_horizontal_axis0pm15":
        return np.minimum(axis, 180.0 - axis) <= 15.0
    if band == "near_vertical_axis90pm15":
        return np.abs(axis - 90.0) <= 15.0
    if str(band).startswith("axis_bin_"):
        parts = str(band).split("_")
        if len(parts) == 4:
            start = float(parts[2])
            stop = float(parts[3])
            if stop >= 180.0:
                return (axis >= start) & (axis <= 180.0)
            return (axis >= start) & (axis < stop)
    raise ValueError(f"Unknown band {band!r}")


def bootstrap_ratio_ci(sub: pd.DataFrame, *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if int(n_bootstrap) <= 0 or sub.empty:
        return float("nan"), float("nan")
    per_movie = (
        sub.groupby("movie_index", sort=True)[
            ["information_numerator_bits_arbitrary_dt", "expected_spikes_arbitrary_dt"]
        ]
        .sum()
        .reset_index()
    )
    numer = per_movie["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=np.float64)
    denom = per_movie["expected_spikes_arbitrary_dt"].to_numpy(dtype=np.float64)
    n = int(numer.size)
    if n <= 1:
        return float("nan"), float("nan")
    sample_idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    values = np.nansum(numer[sample_idx], axis=1) / np.maximum(np.nansum(denom[sample_idx], axis=1), EPS)
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_view(
    frame: pd.DataFrame,
    *,
    view: str,
    view_label: str,
    x_col: str,
    band: str,
    band_label: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    axes = frame[["movie_index", "contour_axis_image_deg"]].drop_duplicates("movie_index")
    keep_movies = set(
        axes.loc[band_mask(axes["contour_axis_image_deg"].to_numpy(dtype=float), band), "movie_index"].astype(int).tolist()
    )
    filtered = frame[frame["movie_index"].astype(int).isin(keep_movies)].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["condition_id", "condition_index", x_col, "sf_group", "alignment_group"]
    for keys, sub in filtered.groupby(group_cols, sort=True):
        condition_id, condition_index, x_value, sf_group, alignment_group = keys
        numerator = float(np.nansum(sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=float)))
        denominator = float(np.nansum(sub["expected_spikes_arbitrary_dt"].to_numpy(dtype=float)))
        ci_lo, ci_hi = bootstrap_ratio_ci(sub, n_bootstrap=n_bootstrap, rng=rng)
        rows.append(
            {
                "band": band,
                "band_label": band_label,
                "view": view,
                "view_label": view_label,
                "condition_id": str(condition_id),
                "condition_index": int(condition_index),
                "x_scale": float(x_value),
                "sf_group": str(sf_group),
                "alignment_group": str(alignment_group),
                "n_fixations": int(sub["movie_index"].nunique()),
                "accumulated_bits_per_spike": numerator / max(denominator, EPS),
                "accumulated_bits_per_spike_boot_ci_low": ci_lo,
                "accumulated_bits_per_spike_boot_ci_high": ci_hi,
                "expected_spikes_sum_arbitrary_dt": denominator,
                "information_numerator_sum_bits_arbitrary_dt": numerator,
            }
        )
    return rows


def plot_band_curves(
    out_dir: Path,
    summary: pd.DataFrame,
    *,
    band: str,
    band_label: str,
    dpi: int,
) -> tuple[Path, Path]:
    sf_groups = ["low_sf", "high_sf"]
    fig, axes = plt.subplots(
        len(sf_groups),
        len(VIEW_SPECS),
        figsize=(14.4, 5.9),
        sharey="row",
        constrained_layout=True,
    )
    for row_idx, sf_group in enumerate(sf_groups):
        for col_idx, spec in enumerate(VIEW_SPECS):
            ax = axes[row_idx, col_idx]
            view_summary = summary[
                (summary["band"].astype(str) == band)
                & (summary["view"].astype(str) == str(spec["view"]))
                & (summary["sf_group"].astype(str) == sf_group)
            ].copy()
            for alignment_group in ["contour_aligned", "contour_orthogonal"]:
                sub = view_summary[view_summary["alignment_group"].astype(str) == alignment_group].sort_values(
                    ["x_scale", "condition_index"]
                )
                if sub.empty:
                    continue
                x = sub["x_scale"].to_numpy(dtype=float)
                y = sub["accumulated_bits_per_spike"].to_numpy(dtype=float)
                lo = sub["accumulated_bits_per_spike_boot_ci_low"].to_numpy(dtype=float)
                hi = sub["accumulated_bits_per_spike_boot_ci_high"].to_numpy(dtype=float)
                color = alignment_color(alignment_group)
                finite = np.isfinite(lo) & np.isfinite(hi)
                if finite.any():
                    ax.fill_between(x[finite], lo[finite], hi[finite], color=color, alpha=0.12, linewidth=0.0)
                support = int(np.nanmedian(sub["n_fixations"].to_numpy(dtype=float)))
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.0,
                    markersize=4.0,
                    color=color,
                    label=f"{alignment_label(alignment_group)} (n={support})",
                )
            ax.axvline(1.0, color="0.65", linestyle=":", linewidth=0.9)
            ax.grid(True, color="0.9", linewidth=0.7)
            if row_idx == 0:
                ax.set_title(str(spec["label"]), fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(f"{sf_label(sf_group)}\nbits/spike")
            if row_idx == len(sf_groups) - 1:
                ax.set_xlabel("scale")
            if row_idx == 0 and col_idx == len(VIEW_SPECS) - 1:
                ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle(
        "BackImage RR100 weighted absolute SSI, restricted to one contour-orientation mode\n"
        f"{band_label}",
        fontsize=12,
    )
    png = out_dir / f"backimage_rr100_orientation_stratified_population_curves_{band}.png"
    pdf = out_dir / f"backimage_rr100_orientation_stratified_population_curves_{band}.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view_specs = view_specs_from_args(args)
    bands = build_bands(str(args.band_mode), coarse_bin_width_deg=float(args.coarse_bin_width_deg))
    all_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.bootstrap_seed))
    for spec in view_specs:
        csv_path = Path(spec["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(csv_path)
        frame = frame[frame["sf_group"].astype(str).isin(["low_sf", "high_sf"])].copy()
        for band_spec in bands:
            all_rows.extend(
                summarize_view(
                    frame,
                    view=str(spec["view"]),
                    view_label=str(spec["label"]),
                    x_col=str(spec["x_col"]),
                    band=str(band_spec["band"]),
                    band_label=str(band_spec["label"]),
                    n_bootstrap=int(args.n_bootstrap),
                    rng=rng,
                )
            )
    summary_csv = out_dir / "orientation_stratified_weighted_population_summary.csv"
    write_csv(summary_csv, all_rows)
    summary = pd.DataFrame(all_rows)
    figure_outputs: dict[str, dict[str, Path]] = {}
    counts = []
    first_csv = Path(view_specs[0]["csv_path"])
    axes = pd.read_csv(first_csv)[["movie_index", "contour_axis_image_deg"]].drop_duplicates("movie_index")
    for band_spec in bands:
        mask = band_mask(axes["contour_axis_image_deg"].to_numpy(dtype=float), str(band_spec["band"]))
        count_payload = {
            "band": str(band_spec["band"]),
            "band_label": str(band_spec["label"]),
            "description": str(band_spec["description"]),
            "n_fixations": int(mask.sum()),
            "fraction_fixations": float(mask.mean()),
        }
        counts.append(count_payload)
        if int(mask.sum()) < int(args.min_fixations_per_band):
            continue
        png, pdf = plot_band_curves(
            out_dir,
            summary,
            band=str(band_spec["band"]),
            band_label=str(band_spec["label"]),
            dpi=int(args.dpi),
        )
        figure_outputs[str(band_spec["band"])] = {"png": png, "pdf": pdf}
    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_rr100_orientation_stratified_population_ssi",
            "summary_csv": summary_csv,
            "bands": counts,
            "view_specs": view_specs,
            "band_mode": str(args.band_mode),
            "min_fixations_per_band": int(args.min_fixations_per_band),
            "figure_outputs": figure_outputs,
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
    )
    print(f"Wrote {summary_csv}")
    for band, outputs in figure_outputs.items():
        print(f"Wrote {outputs['png']}")


if __name__ == "__main__":
    main()
