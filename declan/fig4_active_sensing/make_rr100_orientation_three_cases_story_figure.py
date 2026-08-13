#!/usr/bin/env python3
"""Turn checkpoint-03 orientation maps into a three-case reader-facing figure.

This is a presentation-only reorganization. It uses the saved display maps and
the unchanged unsmoothed overlap scores from checkpoint 03.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.polish_rr100_kuang_unit_overlap_checkpoint import kuang_colormap


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_overlap_checkpoint_03_v1"
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_orientation_three_cases_story_figure_v1"
DISPLAY_FLOOR_DB = -30.0


CASE_DEFINITIONS = {
    3: {
        "case_title": "BROAD SPECTRAL MATCH",
        "interpretation": "FEM power enters the unit's passband\nacross orientations",
        "unit_caption": "RR100 3 · low SF",
        "outcome": "Preferred ≈ 90° rotated",
        "outcome_detail": "preferred and 90°-rotated matches are similar",
    },
    1: {
        "case_title": "ORIENTATION-SPECIFIC MATCH",
        "interpretation": "FEM power preferentially matches\nthe unit's tuned orientation",
        "unit_caption": "RR100 1 · middle SF",
        "outcome": "Preferred orientation gives\n2.4× stronger match",
        "outcome_detail": "orientation strengthens the spectral prediction",
    },
    19: {
        "case_title": "ORIENTATION MISMATCH",
        "interpretation": "SF–TF alone looks strong, but the image\nstructure has the wrong orientation",
        "unit_caption": "RR100 19 · high SF",
        "outcome": "80% of the apparent match\ndisappears",
        "outcome_detail": "the 90°-rotated control contains more matching power",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--models-csv", type=Path, default=MODELS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def edges(values: np.ndarray) -> np.ndarray:
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (middle[0] - values[0]), middle, values[-1] + (values[-1] - middle[-1])]


def to_db(values: np.ndarray, reference: float) -> np.ndarray:
    relative = np.maximum(np.asarray(values, dtype=float) / max(float(reference), 1e-30), 10.0 ** (DISPLAY_FLOOR_DB / 10.0))
    return 10.0 * np.log10(relative)


def setup_frequency_axis(axis: plt.Axes, show_y: bool) -> None:
    sf_ticks = np.asarray([1, 2, 4, 8], dtype=float)
    tf_ticks = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    axis.set_xticks(np.log2(sf_ticks), [f"{value:g}" for value in sf_ticks])
    axis.set_yticks(np.log2(tf_ticks), [f"{value:g}" for value in tf_ticks] if show_y else [])
    axis.set_xlabel("spatial frequency (cpd)", fontsize=8)
    if show_y:
        axis.set_ylabel("temporal frequency (Hz)", fontsize=8)
    axis.tick_params(labelsize=7, length=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.65)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.read_csv(args.source_dir / "selected_unit_orientation_overlap_metrics.csv").set_index("rr100_index")
    models = pd.read_csv(args.models_csv).set_index("rr100_index")
    archive = np.load(args.source_dir / "display_smoothed_orientation_maps.npz")
    sf = archive["spatial_cpd"].astype(float)
    tf = archive["temporal_hz"].astype(float)
    x_edges = edges(np.log2(sf))
    y_edges = edges(np.log2(tf))
    radial_power = archive["radial_power_display"].astype(float)
    common_power_reference = float(radial_power.max())
    cmap = kuang_colormap()
    norm = Normalize(vmin=DISPLAY_FLOOR_DB, vmax=0.0)

    case_rows = []
    for display_order, unit in enumerate((3, 1, 19), start=1):
        source = metrics.loc[unit]
        model = models.loc[unit]
        definition = CASE_DEFINITIONS[unit]
        radial = float(source["radial_sf_tf_overlap"])
        matched = float(source["preferred_orientation_aware_overlap"])
        rotated = float(source["orthogonal_orientation_control_overlap"])
        case_rows.append({
            "display_order": display_order,
            "rr100_index": unit,
            **definition,
            "preferred_sf_cpd": float(model["preferred_sf_cpd"]),
            "preferred_tf_hz": float(model["preferred_tf_hz"]),
            "preferred_orientation_deg": float(model["preferred_orientation_deg"]),
            "sf_tf_only_overlap": radial,
            "full_tuning_overlap": matched,
            "rotated_90deg_overlap": rotated,
            "full_to_rotated_ratio": matched / max(rotated, 1e-30),
            "fraction_of_sf_tf_only_retained": matched / max(radial, 1e-30),
            "fraction_of_sf_tf_only_lost": 1.0 - matched / max(radial, 1e-30),
        })
    story = pd.DataFrame(case_rows)
    story.to_csv(args.out_dir / "three_case_narrative_values.csv", index=False)

    fig = plt.figure(figsize=(16.0, 9.4), constrained_layout=False)
    grid = fig.add_gridspec(
        3, 5, width_ratios=[1.48, 1.0, 1.0, 1.0, 1.28],
        left=0.035, right=0.965, top=0.83, bottom=0.09, hspace=0.48, wspace=0.28,
    )
    axes = np.empty((3, 5), dtype=object)
    for row in range(3):
        for column in range(5):
            axes[row, column] = fig.add_subplot(grid[row, column])

    map_image = None
    bar_colors = ["#8A8A8A", "#A71930", "#174AA5"]
    bar_labels = ["SF–TF only", "preferred", "90° rotated"]
    for row_index, case in story.iterrows():
        unit = int(case["rr100_index"])
        text_axis = axes[row_index, 0]
        text_axis.axis("off")
        text_axis.text(0.04, 0.92, case["case_title"], transform=text_axis.transAxes,
                       fontsize=13, fontweight="bold", va="top")
        text_axis.text(0.04, 0.68, case["interpretation"], transform=text_axis.transAxes,
                       fontsize=10.5, va="top", linespacing=1.25)
        text_axis.text(0.04, 0.34, case["unit_caption"], transform=text_axis.transAxes,
                       fontsize=9.5, color="0.35", va="top")
        text_axis.plot([0.0, 0.0], [0.06, 0.98], color="#A71930", lw=3.0,
                       transform=text_axis.transAxes, clip_on=False)

        gain_power = archive[f"rr100_{unit:03d}_gain_power"].astype(float)
        preferred_power = archive[f"rr100_{unit:03d}_preferred_power"].astype(float)
        matched_power = archive[f"rr100_{unit:03d}_preferred_overlap"].astype(float)
        panels = (
            (gain_power, 1.0, f"prefers {case['preferred_sf_cpd']:.2f} cpd, {case['preferred_tf_hz']:.1f} Hz"),
            (preferred_power, common_power_reference, f"image power at {case['preferred_orientation_deg']:.0f}°"),
            (matched_power, common_power_reference, f"full-tuning score = {case['full_tuning_overlap']:.3f}"),
        )
        for offset, (values, reference, subtitle) in enumerate(panels, start=1):
            axis = axes[row_index, offset]
            map_image = axis.pcolormesh(
                x_edges, y_edges, to_db(values, reference).T,
                shading="flat", cmap=cmap, norm=norm, rasterized=True,
            )
            setup_frequency_axis(axis, show_y=(offset == 1))
            axis.set_title(subtitle, fontsize=9, pad=5)

        bar_axis = axes[row_index, 4]
        bar_values = np.asarray([
            case["sf_tf_only_overlap"], case["full_tuning_overlap"], case["rotated_90deg_overlap"]
        ], dtype=float)
        y = np.asarray([2, 1, 0])
        bars = bar_axis.barh(y, bar_values, color=bar_colors, height=0.56)
        bar_axis.set_yticks(y, bar_labels)
        bar_axis.set_xlim(0.0, 0.39)
        bar_axis.set_xticks([0.0, 0.1, 0.2, 0.3])
        bar_axis.set_xlabel("spectral-overlap score", fontsize=8)
        bar_axis.tick_params(labelsize=8)
        bar_axis.spines[["top", "right"]].set_visible(False)
        bar_axis.grid(axis="x", color="0.90", lw=0.6)
        bar_axis.set_axisbelow(True)
        for bar, value in zip(bars, bar_values):
            bar_axis.text(value + 0.009, bar.get_y() + bar.get_height() / 2.0, f"{value:.2f}",
                          va="center", fontsize=8.5)
        bar_axis.set_title(case["outcome"], fontsize=10.2, fontweight="bold", loc="left", pad=7)

    headers = (
        "THREE MECHANISTIC CASES",
        "A  What frequencies\nthe unit passes",
        "B  × FEM power at\nthe unit's orientation",
        "C  = Power matching\nthe full tuning",
        "D  What orientation\nchanges",
    )
    for column, header in enumerate(headers):
        axes[0, column].text(
            0.0 if column in (0, 4) else 0.5, 1.22, header,
            transform=axes[0, column].transAxes,
            ha="left" if column in (0, 4) else "center", va="bottom",
            fontsize=10.5, fontweight="bold",
        )

    cbar = fig.colorbar(map_image, ax=[axes[row, column] for row in range(3) for column in (1, 2, 3)],
                        orientation="horizontal", fraction=0.035, pad=0.075, aspect=45)
    cbar.set_ticks([-30, -20, -10, 0])
    cbar.set_label(
        "relative level (dB); sensitivity is normalized per unit, power uses a shared FEM reference",
        fontsize=8,
    )
    fig.suptitle(
        "Spectral and orientation tuning determine which V1 channels receive FEM-induced image power",
        fontsize=15.5, y=0.975,
    )
    fig.text(
        0.5, 0.925,
        "SF–TF overlap is necessary but not sufficient: orientation can preserve, strengthen, or overturn the prediction",
        ha="center", fontsize=11, color="0.28",
    )
    fig.text(
        0.035, 0.02,
        "Spectral-overlap proxy from one exact 51×51 retinal movie; not a measured firing-rate response. "
        "Maps are lightly smoothed for display; all scores use unsmoothed Fourier bins.",
        fontsize=8, color="0.32",
    )

    output_base = args.out_dir / "orientation_three_mechanistic_cases"
    fig.savefig(output_base.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "reader-facing three-case reorganization of checkpoint 03",
        "source_checkpoint": str(args.source_dir.resolve()),
        "cases": story.to_dict(orient="records"),
        "omitted_from_main_story": {
            "rr100_index": 81,
            "reason": "low-overlap control remains in checkpoint-03 audit figure but is not one of the three mechanistic cases",
        },
        "quantification_policy": "no scores recomputed or smoothed; values copied from checkpoint-03 unsmoothed metrics",
        "display_policy": "uses checkpoint-03 display-smoothed maps and reader-facing titles",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Orientation-aware overlap: three mechanistic cases\n\n"
        "This presentation figure reorganizes checkpoint 03 around broad spectral match, orientation-specific "
        "match, and orientation mismatch. RR100 81 remains available as the low-overlap control in the source "
        "audit figure. No values were recomputed; all displayed scores are the unsmoothed checkpoint-03 values.\n"
    )
    print(story[["case_title", "rr100_index", "sf_tf_only_overlap", "full_tuning_overlap",
                 "rotated_90deg_overlap", "fraction_of_sf_tf_only_lost"]].to_string(index=False))


if __name__ == "__main__":
    main()
