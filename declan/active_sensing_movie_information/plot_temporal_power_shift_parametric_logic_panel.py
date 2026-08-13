#!/usr/bin/env python3
"""Plot the parametric temporal-power logic for one selected trace."""

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
DEFAULT_TRACE_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_07_parametric_trace_examples_v1"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_06_parametric_mechanism_panels_v1"

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
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-index", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=190)
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
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(trace_dir / "checkpoint_07_selected_low_sf_trace_examples.csv")
    trace_index = int(args.trace_index) if args.trace_index is not None else int(selected.iloc[0]["trace_index"])
    rows = pd.read_csv(trace_dir / "checkpoint_07_cross_sf_trace_timecourses.csv")
    rows = rows[rows["trace_index"].eq(trace_index)].copy()
    if rows.empty:
        raise ValueError(f"No cross-SF timecourse rows found for trace_index={trace_index}")
    speed = rows.drop_duplicates("frame_index").sort_values("frame_index")

    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.8), constrained_layout=True)
    add_panel_label(axes[0], "A")
    axes[0].plot(speed["time_ms"], speed["across_contour_speed_deg_s"], color=OKABE_ITO["black"], lw=2.0)
    axes[0].set_title("Retinal motion")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("speed across contour (deg/s)")
    axes[0].grid(True, color="#e8e8e8", lw=0.7)

    add_panel_label(axes[1], "B")
    for _, unit_rows in rows.groupby("unit_index", sort=False):
        unit_rows = unit_rows.sort_values("frame_index")
        group = str(unit_rows["sf_group"].iloc[0])
        color = GROUP_COLORS.get(group, OKABE_ITO["grey"])
        label = f"{unit_rows['unit_label'].iloc[0]} ({GROUP_LABELS.get(group, group)})"
        axes[1].plot(
            unit_rows["time_ms"],
            np.maximum(unit_rows["motion_induced_tf_hz"].to_numpy(float), 0.05),
            color=color,
            lw=1.9,
            label=label,
        )
        axes[1].axhline(float(unit_rows["fit_pref_tf_hz"].iloc[0]), color=color, lw=1.0, ls="--", alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_title("Motion-induced TF")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("Hz")
    axes[1].grid(True, which="both", color="#e8e8e8", lw=0.7)
    axes[1].text(0.03, 0.07, "dashed = unit's preferred TF", transform=axes[1].transAxes, fontsize=8)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")

    add_panel_label(axes[2], "C")
    for _, unit_rows in rows.groupby("unit_index", sort=False):
        unit_rows = unit_rows.sort_values("frame_index")
        group = str(unit_rows["sf_group"].iloc[0])
        color = GROUP_COLORS.get(group, OKABE_ITO["grey"])
        axes[2].plot(unit_rows["time_ms"], unit_rows["tf_match"], color=color, lw=1.9)
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].set_title("TF match")
    axes[2].set_xlabel("time (ms)")
    axes[2].set_ylabel("match to unit")
    axes[2].grid(True, color="#e8e8e8", lw=0.7)

    add_panel_label(axes[3], "D")
    for _, unit_rows in rows.groupby("unit_index", sort=False):
        unit_rows = unit_rows.sort_values("frame_index")
        group = str(unit_rows["sf_group"].iloc[0])
        color = GROUP_COLORS.get(group, OKABE_ITO["grey"])
        axes[3].plot(unit_rows["time_ms"], unit_rows["linear_power_drive"], color=color, lw=1.9)
    axes[3].set_title("Linear power drive")
    axes[3].set_xlabel("time (ms)")
    axes[3].set_ylabel("image power x TF match")
    axes[3].grid(True, color="#e8e8e8", lw=0.7)

    fig.suptitle(f"Parametric linear-power logic for trace {trace_index}", fontsize=14)
    png = out_dir / "checkpoint_06a_parametric_mechanism_logic_panels.png"
    pdf = out_dir / "checkpoint_06a_parametric_mechanism_logic_panels.pdf"
    table = out_dir / "checkpoint_06a_parametric_mechanism_logic_rows.csv"
    rows.to_csv(table, index=False)
    fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    write_json(
        out_dir / "checkpoint_06a_parametric_mechanism_logic_metadata.json",
        {
            "analysis": "parametric_temporal_power_logic_panel",
            "trace_dir": trace_dir,
            "out_dir": out_dir,
            "trace_index": trace_index,
            "outputs": {"figure_png": png, "figure_pdf": pdf, "rows": table},
        },
    )
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
