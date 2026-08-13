#!/usr/bin/env python3
"""Plot cross-SF map metrics beside the linear-power drive proxy.

This checkpoint summary intentionally stays small: it combines the selected
cross-SF unit examples, their normal-motion linear power drive, and their
normal-minus-stabilized map metrics for the same image/traces.
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
DEFAULT_TRACE_EXAMPLE_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_07_low_sf_trace_examples_v1"
DEFAULT_MAP_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_08_cross_sf_activation_maps_v1"

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
    parser.add_argument("--trace-example-dir", type=Path, default=DEFAULT_TRACE_EXAMPLE_DIR)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--dpi", type=int, default=180)
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


def load_summary_rows(trace_example_dir: Path, map_dir: Path) -> pd.DataFrame:
    map_metrics = pd.read_csv(map_dir / "checkpoint_08_cross_sf_map_metric_summary.csv")
    drive = pd.read_csv(trace_example_dir / "checkpoint_07_cross_sf_trace_timecourses.csv")
    drive_summary = (
        drive.groupby(["trace_index", "unit_index"], dropna=False)
        .agg(
            unit_label=("unit_label", "first"),
            sf_group=("sf_group", "first"),
            sf_group_label=("sf_group_label", "first"),
            preferred_sf_cpd=("preferred_sf_cpd", "first"),
            fit_pref_tf_hz=("fit_pref_tf_hz", "first"),
            linear_power_drive_mean=("linear_power_drive", "mean"),
            linear_power_drive_peak=("linear_power_drive", "max"),
            tf_match_mean=("tf_match", "mean"),
            tf_match_peak=("tf_match", "max"),
            motion_induced_tf_mean_hz=("motion_induced_tf_hz", "mean"),
            motion_induced_tf_peak_hz=("motion_induced_tf_hz", "max"),
        )
        .reset_index()
    )
    rows = map_metrics.merge(
        drive_summary,
        on=["trace_index", "unit_index", "unit_label", "preferred_sf_cpd"],
        how="left",
        suffixes=("", "_drive"),
    )
    missing = rows["linear_power_drive_mean"].isna()
    if bool(missing.any()):
        missing_rows = rows.loc[missing, ["trace_index", "unit_index", "unit_label"]].to_dict("records")
        raise ValueError(f"Missing linear-drive rows for map metrics: {missing_rows}")
    rows["sf_group_label"] = rows["sf_group"].map(lambda group: GROUP_LABELS.get(str(group), str(group)))
    rows["plot_color"] = rows["sf_group"].map(lambda group: GROUP_COLORS.get(str(group), OKABE_ITO["grey"]))
    rows["display_order"] = rows["sf_group"].map({name: idx for idx, name in enumerate(GROUP_ORDER)}).fillna(99).astype(int)
    rows = rows.sort_values(["display_order", "trace_index", "unit_index"]).drop(columns=["display_order"])
    return rows


def plot_summary(rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.35), constrained_layout=True, sharex=True)
    panels = [
        ("Linear power drive", "linear_power_drive_mean", "mean drive\nnormal motion"),
        ("Mean activation change", "delta_mean_rate_normal_minus_stabilized", "mean activation change\nnormal - stabilized"),
        ("Map SSI change", "delta_ssi_normal_minus_stabilized", "map SSI change\nnormal - stabilized"),
    ]
    label_by_unit: dict[int, str] = {}
    for unit_index, unit_rows in rows.groupby("unit_index", sort=False):
        first = unit_rows.iloc[0]
        label_by_unit[int(unit_index)] = f"{first['sf_group_label']} {first['unit_label']}"

    for panel_idx, (title, column, ylabel) in enumerate(panels):
        ax = axes[panel_idx]
        add_panel_label(ax, chr(ord("A") + panel_idx))
        for unit_index, unit_rows in rows.groupby("unit_index", sort=False):
            unit_rows = unit_rows.sort_values("trace_index")
            color = str(unit_rows["plot_color"].iloc[0])
            ax.plot(
                unit_rows["trace_index"],
                unit_rows[column],
                marker="o",
                lw=1.9,
                ms=6.2,
                color=color,
                label=label_by_unit[int(unit_index)],
            )
        if column != "linear_power_drive_mean":
            ax.axhline(0.0, color="#555555", lw=0.85)
        ax.set_title(title)
        ax.set_xlabel("trace index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(rows["trace_index"].unique()))
        ax.grid(True, color="#e8e8e8", lw=0.7)
    axes[2].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Linear power drive beside map-level effects", fontsize=13)
    png = out_dir / "checkpoint_08_cross_sf_map_metric_summary_with_linear_drive.png"
    pdf = out_dir / "checkpoint_08_cross_sf_map_metric_summary_with_linear_drive.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary_rows(Path(args.trace_example_dir), Path(args.map_dir))
    table_path = out_dir / "checkpoint_08_cross_sf_map_metric_summary_with_linear_drive.csv"
    rows.to_csv(table_path, index=False)
    png, pdf = plot_summary(rows, out_dir, dpi=int(args.dpi))
    write_json(
        out_dir / "checkpoint_08_cross_sf_map_metric_summary_with_linear_drive_metadata.json",
        {
            "analysis": "cross_sf_map_metric_summary_with_linear_power_drive",
            "trace_example_dir": Path(args.trace_example_dir),
            "map_dir": Path(args.map_dir),
            "out_dir": out_dir,
            "linear_power_drive_definition": (
                "Mean over the 32-frame normal-motion block of image SF-band power times the unit TF match. "
                "This is an upstream proxy, not a full nonlinear response prediction."
            ),
            "sf_group_color_mapping": GROUP_COLORS,
            "outputs": {
                "summary_table": table_path,
                "summary_png": png,
                "summary_pdf": pdf,
            },
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
