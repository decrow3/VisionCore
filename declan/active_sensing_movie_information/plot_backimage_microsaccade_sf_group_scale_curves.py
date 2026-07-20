#!/usr/bin/env python3
"""Plot low- vs high-SF unit curves for the microsaccade snippet scale sweep."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_microsaccade_snippet_rr100_spatial_ssi_isotropic_full_patched_v2"
)
DEFAULT_CURVE_CSV = DEFAULT_RUN_DIR / "sf_group_scale_curve_input.csv"
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "sf_group_low_high_comparison_dynamic_log_gaussian_marginal_low0p05_high0p5"
VALUE_COL = "displayed_movie_time_resolved_ssi_bits_per_spike"
GROUP_ORDER = ["low_sf", "high_sf"]
GROUP_COLORS = {"low_sf": "#1f77b4", "high_sf": "#d62728"}
GROUP_LABELS = {"low_sf": "low SF", "high_sf": "high SF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-csv", type=Path, default=DEFAULT_CURVE_CSV)
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--value-col", default=VALUE_COL)
    parser.add_argument("--show-unit-curves", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sem(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def load_curves(curve_csv: Path, sf_groups_csv: Path, value_col: str) -> pd.DataFrame:
    curves = pd.read_csv(curve_csv)
    units = pd.read_csv(sf_groups_csv)
    required_curve = {"unit_index", "unit_label", "display_scale", "axis_mode", value_col}
    required_units = {"unit_index", "unit_label", "sf_group", "sf_group_label", "sf_group_definition"}
    missing_curve = sorted(required_curve.difference(curves.columns))
    missing_units = sorted(required_units.difference(units.columns))
    if missing_curve:
        raise ValueError(f"{curve_csv} is missing columns: {missing_curve}")
    if missing_units:
        raise ValueError(f"{sf_groups_csv} is missing columns: {missing_units}")

    use_units = units[list(required_units)].copy()
    out = curves.merge(use_units, on=["unit_index", "unit_label"], how="inner", validate="many_to_one")
    out = out[out["sf_group"].isin(GROUP_ORDER)].copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out["display_scale"] = pd.to_numeric(out["display_scale"], errors="coerce")
    ref = out[np.isclose(out["display_scale"], 0.0)][["unit_index", "axis_mode", value_col]].rename(
        columns={value_col: "ssi_at_0x"}
    )
    out = out.merge(ref, on=["unit_index", "axis_mode"], how="left", validate="many_to_one")
    out["ssi_delta_vs_0x"] = out[value_col] - out["ssi_at_0x"]
    return out.sort_values(["sf_group", "unit_index", "display_scale"]).reset_index(drop=True)


def summarize(curves: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sf_group, display_scale), sub in curves.groupby(["sf_group", "display_scale"], sort=True):
        for col in [value_col, "ssi_delta_vs_0x", "displayed_movie_mean_rate"]:
            if col not in sub.columns:
                continue
            values = pd.to_numeric(sub[col], errors="coerce")
            rows.append(
                {
                    "sf_group": str(sf_group),
                    "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                    "display_scale": float(display_scale),
                    "value_name": col,
                    "n_units": int(sub["unit_index"].nunique()),
                    "mean": float(np.nanmean(values)),
                    "sem": sem(values),
                    "median": float(np.nanmedian(values)),
                    "n_finite": int(np.isfinite(values.to_numpy(dtype=float)).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    value_col: str,
    out_dir: Path,
    *,
    dpi: int,
    show_unit_curves: bool,
) -> tuple[Path, Path]:
    definition = str(curves["sf_group_definition"].dropna().iloc[0]) if curves["sf_group_definition"].notna().any() else ""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True)
    panels = [
        (axes[0], value_col, "SSI (bits/spike)", "Absolute SSI"),
        (axes[1], "ssi_delta_vs_0x", "SSI minus 0x (bits/spike)", "Scale modulation"),
    ]
    for ax, value_name, ylabel, title in panels:
        for sf_group in GROUP_ORDER:
            unit_sub = curves[curves["sf_group"].eq(sf_group)]
            color = GROUP_COLORS[sf_group]
            if show_unit_curves:
                for _, per_unit in unit_sub.groupby("unit_index", sort=False):
                    per_unit = per_unit.sort_values("display_scale")
                    ax.plot(
                        per_unit["display_scale"].to_numpy(dtype=float),
                        per_unit[value_name].to_numpy(dtype=float),
                        color=color,
                        alpha=0.13,
                        linewidth=0.8,
                        zorder=1,
                    )
            mean_sub = summary[
                summary["sf_group"].eq(sf_group) & summary["value_name"].eq(value_name)
            ].sort_values("display_scale")
            x = mean_sub["display_scale"].to_numpy(dtype=float)
            y = mean_sub["mean"].to_numpy(dtype=float)
            e = mean_sub["sem"].to_numpy(dtype=float)
            label = f"{GROUP_LABELS[sf_group]} (n={int(mean_sub['n_units'].iloc[0])})"
            ax.plot(x, y, marker="o", linewidth=2.2, markersize=4.5, color=color, label=label, zorder=4)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.axvline(1.0, color="0.55", linestyle="--", linewidth=1.0)
        if value_name == "ssi_delta_vs_0x":
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("microsaccade trace scale")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "BackImage RR100 microsaccade snippets: low- vs high-SF units\n"
        f"{definition}; group mean +/- SEM",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    png = out_dir / "backimage_microsaccade_sf_group_low_high_scale_curves.png"
    pdf = out_dir / "backimage_microsaccade_sf_group_low_high_scale_curves.pdf"
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curves(args.curve_csv, args.sf_groups_csv, str(args.value_col))
    summary = summarize(curves, str(args.value_col))
    curves.to_csv(args.out_dir / "microsaccade_sf_group_low_high_curves_long.csv", index=False)
    summary.to_csv(args.out_dir / "microsaccade_sf_group_low_high_summary.csv", index=False)
    png, pdf = plot(
        curves,
        summary,
        str(args.value_col),
        args.out_dir,
        dpi=int(args.dpi),
        show_unit_curves=bool(args.show_unit_curves),
    )
    write_json(
        args.out_dir / "summary.json",
        {
            "analysis": "backimage_microsaccade_sf_group_low_high_scale_curves",
            "curve_csv": args.curve_csv,
            "sf_groups_csv": args.sf_groups_csv,
            "value_col": str(args.value_col),
            "show_unit_curves": bool(args.show_unit_curves),
            "n_units_by_group": curves.groupby("sf_group")["unit_index"].nunique().to_dict(),
            "display_scales": sorted(float(v) for v in curves["display_scale"].dropna().unique()),
            "outputs": {
                "curves_long": args.out_dir / "microsaccade_sf_group_low_high_curves_long.csv",
                "summary": args.out_dir / "microsaccade_sf_group_low_high_summary.csv",
                "png": png,
                "pdf": pdf,
            },
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(summary[summary["value_name"].eq("ssi_delta_vs_0x")].to_string(index=False))


if __name__ == "__main__":
    main()
