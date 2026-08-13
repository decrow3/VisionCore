#!/usr/bin/env python3
"""Collect targeted cross-SF activation-map metrics into a checkpoint table."""

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
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--trace-indices", type=str, default="27,28,29,30,31")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
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
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")


def collect_rows(map_dir: Path, trace_indices: list[int]) -> pd.DataFrame:
    rows = []
    for trace_index in trace_indices:
        path = map_dir / f"trace{int(trace_index)}" / "checkpoint_02_unit_metric_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        metrics = pd.read_csv(path)
        static = metrics[metrics["condition"].astype(str).eq("stabilized")].copy()
        normal = metrics[metrics["condition"].astype(str).eq("normal")].copy()
        key_cols = ["unit_index", "unit_label", "selection_role", "preferred_sf_cpd"]
        optional = [col for col in ["fit_pref_tf_hz", "dense_fit_pref_tf_hz", "sf_group", "preference_source"] if col in metrics.columns]
        key_cols = [*key_cols, *optional]
        static = static[key_cols + ["movie_ssi_bits_per_spike", "movie_expected_spikes", "movie_mean_rate"]].rename(
            columns={
                "movie_ssi_bits_per_spike": "ssi_stabilized",
                "movie_expected_spikes": "expected_spikes_stabilized",
                "movie_mean_rate": "mean_rate_stabilized",
            }
        )
        normal = normal[key_cols + ["movie_ssi_bits_per_spike", "movie_expected_spikes", "movie_mean_rate"]].rename(
            columns={
                "movie_ssi_bits_per_spike": "ssi_normal",
                "movie_expected_spikes": "expected_spikes_normal",
                "movie_mean_rate": "mean_rate_normal",
            }
        )
        merged = static.merge(normal, on=key_cols, how="inner", validate="one_to_one")
        merged.insert(0, "trace_index", int(trace_index))
        merged["delta_ssi_normal_minus_stabilized"] = merged["ssi_normal"] - merged["ssi_stabilized"]
        merged["delta_mean_rate_normal_minus_stabilized"] = merged["mean_rate_normal"] - merged["mean_rate_stabilized"]
        merged["delta_expected_spikes_normal_minus_stabilized"] = (
            merged["expected_spikes_normal"] - merged["expected_spikes_stabilized"]
        )
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    if "fit_pref_tf_hz" not in out.columns and "dense_fit_pref_tf_hz" in out.columns:
        out["fit_pref_tf_hz"] = out["dense_fit_pref_tf_hz"]
    if "dense_fit_pref_tf_hz" not in out.columns and "fit_pref_tf_hz" in out.columns:
        out["dense_fit_pref_tf_hz"] = out["fit_pref_tf_hz"]
    return out


def plot_summary(rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    rows = rows.copy()
    rows["sf_group_label"] = rows["sf_group"].map(lambda group: GROUP_LABELS.get(str(group), str(group)))
    rows["plot_color"] = rows["sf_group"].map(lambda group: GROUP_COLORS.get(str(group), OKABE_ITO["grey"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35), constrained_layout=True, sharex=True)
    panels = [
        ("Mean activation change", "delta_mean_rate_normal_minus_stabilized", "mean activation change\nnormal - stabilized"),
        ("Map SSI change", "delta_ssi_normal_minus_stabilized", "map SSI change\nnormal - stabilized"),
    ]
    for panel_idx, (title, column, ylabel) in enumerate(panels):
        ax = axes[panel_idx]
        add_panel_label(ax, chr(ord("A") + panel_idx))
        for unit_index, unit_rows in rows.groupby("unit_index", sort=False):
            unit_rows = unit_rows.sort_values("trace_index")
            group = str(unit_rows["sf_group"].iloc[0])
            label = f"{unit_rows['sf_group_label'].iloc[0]} {unit_rows['unit_label'].iloc[0]}"
            ax.plot(
                unit_rows["trace_index"],
                unit_rows[column],
                marker="o",
                lw=1.9,
                ms=6.2,
                color=GROUP_COLORS.get(group, OKABE_ITO["grey"]),
                label=label,
            )
        ax.axhline(0.0, color="#555555", lw=0.85)
        ax.set_title(title)
        ax.set_xlabel("trace index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(rows["trace_index"].unique()))
        ax.grid(True, color="#e8e8e8", lw=0.7)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Map-level effects for selected parametric units", fontsize=13)
    png = out_dir / "checkpoint_08_cross_sf_map_metric_summary.png"
    pdf = out_dir / "checkpoint_08_cross_sf_map_metric_summary.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    map_dir = Path(args.map_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else map_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_indices = parse_ints(str(args.trace_indices))
    rows = collect_rows(map_dir, trace_indices)
    table_path = out_dir / "checkpoint_08_cross_sf_map_metric_summary.csv"
    rows.to_csv(table_path, index=False)
    png, pdf = plot_summary(rows, out_dir, dpi=int(args.dpi))
    write_json(
        out_dir / "checkpoint_08_cross_sf_map_metric_summary_metadata.json",
        {
            "analysis": "collect_cross_sf_map_metrics",
            "map_dir": map_dir,
            "out_dir": out_dir,
            "trace_indices": trace_indices,
            "outputs": {"summary_table": table_path, "summary_png": png, "summary_pdf": pdf},
        },
    )
    print(f"Wrote {table_path}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
