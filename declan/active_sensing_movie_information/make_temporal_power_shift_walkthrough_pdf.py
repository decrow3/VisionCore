#!/usr/bin/env python3
"""Build a plain-language walkthrough PDF for the temporal-frequency analysis."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_10_plain_language_walkthrough_v1"
DEFAULT_PARAMETRIC_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_checkpoint_10_parametric_plain_language_walkthrough_v1"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "grey": "#777777",
    "black": "#222222",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--figure-set",
        choices=("legacy", "parametric"),
        default="legacy",
        help="Which saved checkpoint folders to use in the walkthrough.",
    )
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
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_wrapped_text(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    width: int = 78,
    fontsize: float = 13,
    color: str = "#222222",
    weight: str = "normal",
    linespacing: float = 1.18,
    ha: str = "left",
    va: str = "top",
) -> None:
    wrapped = "\n".join(textwrap.fill(line, width=width) if line else "" for line in text.split("\n"))
    fig.text(
        x,
        y,
        wrapped,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=linespacing,
    )


def add_bullets(
    fig: plt.Figure,
    x: float,
    y: float,
    bullets: list[str],
    *,
    width: int = 58,
    fontsize: float = 12.3,
    color: str = "#222222",
    bullet_color: str = "#222222",
    line_gap: float = 0.075,
) -> float:
    cursor = y
    for item in bullets:
        fig.text(x, cursor, "-", ha="left", va="top", fontsize=fontsize, color=bullet_color)
        wrapped = textwrap.fill(item, width=width)
        fig.text(x + 0.022, cursor, wrapped, ha="left", va="top", fontsize=fontsize, color=color, linespacing=1.18)
        cursor -= max(line_gap, 0.034 * (wrapped.count("\n") + 1) + 0.021)
    return cursor


def add_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.05, 0.94, title, ha="left", va="top", fontsize=22, fontweight="bold", color="#111111")
    if subtitle:
        fig.text(0.05, 0.885, subtitle, ha="left", va="top", fontsize=13.5, color="#444444")


def add_footer(fig: plt.Figure, page_number: int) -> None:
    fig.text(0.05, 0.035, "Temporal-frequency mechanism analysis", ha="left", va="bottom", fontsize=8.8, color="#666666")
    fig.text(0.955, 0.035, str(page_number), ha="right", va="bottom", fontsize=8.8, color="#666666")


def add_image(fig: plt.Figure, path: Path, rect: tuple[float, float, float, float], *, title: str | None = None) -> None:
    ax = fig.add_axes(rect)
    image = mpimg.imread(path)
    ax.imshow(image)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11.5, pad=8)


def add_box(fig: plt.Figure, xy: tuple[float, float], size: tuple[float, float], text: str, *, face: str, edge: str) -> None:
    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        transform=fig.transFigure,
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    fig.patches.append(patch)
    add_wrapped_text(fig, x + 0.02, y + h - 0.035, text, width=24, fontsize=12.5, color="#111111")


def add_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.5,
        color="#444444",
        transform=fig.transFigure,
    )
    fig.patches.append(arrow)


def new_page(page_number: int) -> plt.Figure:
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    add_footer(fig, page_number)
    return fig


def save_page(pdf: PdfPages, fig: plt.Figure, dpi: int) -> None:
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def paths(run_dir: Path, *, figure_set: str) -> dict[str, Path]:
    if figure_set == "parametric":
        checkpoint_06 = run_dir / "map_first_power_shift_checkpoint_06_parametric_mechanism_panels_v1"
        checkpoint_07 = run_dir / "map_first_power_shift_checkpoint_07_parametric_trace_examples_v1"
        checkpoint_08 = run_dir / "map_first_power_shift_checkpoint_08_parametric_cross_sf_activation_maps_v1"
        checkpoint_09 = run_dir / "map_first_power_shift_checkpoint_09_parametric_population_bridge_v1"
        preference_audit = run_dir / "map_first_power_shift_parametric_preference_audit_v1"
        logic_png = checkpoint_06 / "checkpoint_06a_parametric_mechanism_logic_panels.png"
    else:
        checkpoint_06 = run_dir / "map_first_power_shift_checkpoint_06_mechanism_panels_v1"
        checkpoint_07 = run_dir / "map_first_power_shift_checkpoint_07_low_sf_trace_examples_v1"
        checkpoint_08 = run_dir / "map_first_power_shift_checkpoint_08_cross_sf_activation_maps_v1"
        checkpoint_09 = run_dir / "map_first_power_shift_checkpoint_09_population_bridge_v1"
        preference_audit = None
        logic_png = checkpoint_06 / "checkpoint_06a_mechanism_logic_panels.png"
    out = {
        "logic": logic_png,
        "trace_gallery": checkpoint_07 / "checkpoint_07d_cross_sf_trace_gallery.png",
        "selected_units": checkpoint_07 / "checkpoint_07_selected_cross_sf_units.csv",
        "example_summary": checkpoint_08 / "checkpoint_08_cross_sf_map_metric_summary_with_linear_drive.png",
        "example_table": checkpoint_08 / "checkpoint_08_cross_sf_map_metric_summary_with_linear_drive.csv",
        "activation_maps": checkpoint_08 / "trace31/checkpoint_02_activation_map_tiles.png",
        "map_timecourses": checkpoint_08 / "trace31/checkpoint_02_map_metric_timecourses.png",
        "population_bridge": checkpoint_09 / "checkpoint_09_population_bridge.png",
        "population_stats": checkpoint_09 / "checkpoint_09_population_relation_stats.csv",
        "population_groups": checkpoint_09 / "checkpoint_09_population_group_summary.csv",
    }
    if preference_audit is not None:
        out["preference_audit"] = preference_audit / "rr100_temporal_power_parametric_preference_audit.png"
    return out


def check_inputs(input_paths: dict[str, Path]) -> None:
    missing = [str(path) for path in input_paths.values() if path.suffix and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing walkthrough inputs:\n" + "\n".join(missing))


def selected_unit_lines(selected_units_csv: Path) -> list[str]:
    units = pd.read_csv(selected_units_csv)
    lines = []
    for _, row in units.iterrows():
        label = str(row["sf_group_label"]).replace("SF", "detail")
        unit = str(row["unit_label"])
        pref_tf = float(row["fit_pref_tf_hz"])
        lines.append(f"{label}: {unit}, likes about {pref_tf:.1f} changes per second.")
    return lines


def selected_unit_sentence(selected_units_csv: Path) -> str:
    lines = selected_unit_lines(selected_units_csv)
    short = [
        line.rstrip(".").replace("Low detail: ", "low ").replace("Middle detail: ", "middle ").replace("High detail: ", "high ")
        for line in lines
    ]
    return "Examples: " + "; ".join(short) + "."


def relation_stat_lines(stats_csv: Path) -> list[str]:
    stats = pd.read_csv(stats_csv)
    label = {
        "linear_power_drive_to_delta_mean_activation": "simple drive -> activation",
        "delta_mean_activation_to_unit_ssi_delta_absolute": "activation -> map score",
        "linear_power_drive_to_unit_ssi_delta_absolute": "simple drive -> map score",
    }
    lines = []
    for _, row in stats.iterrows():
        relation = label.get(str(row["relation"]), str(row["relation"]))
        lines.append(f"{relation}: R2 = {float(row['raw_r2']):.4f} across raw examples.")
    return lines


def group_summary_lines(groups_csv: Path) -> list[str]:
    groups = pd.read_csv(groups_csv)
    drive = groups[groups["metric"].eq("linear_power_drive")]
    ssi = groups[groups["metric"].eq("map_ssi_change")]
    drive_order = drive.sort_values("mean")["sf_group_label"].tolist()
    ssi_order = ssi.sort_values("mean")["sf_group_label"].tolist()
    return [
        "Average simple drive is lowest for low-detail units and highest for high-detail units.",
        f"Drive order: {' < '.join(drive_order)}.",
        f"Map-score order: {' < '.join(ssi_order)}.",
    ]


def write_outline(path: Path) -> None:
    outline = """# Temporal-frequency mechanism analysis walkthrough

Plain-language story:

1. Eye movement makes the image change over time on the retina.
2. Different units see that change differently because they prefer different image detail sizes.
3. We call the simplest input estimate "linear power drive": image detail power times the match to the unit's preferred speed of change.
4. We then ask three separate questions: does drive predict activation, does activation predict map information, and does drive predict map information directly?
5. The example units show these links can come apart.
6. The population view says the simple pairwise links are very weak at the raw-example scale.
7. Current interpretation: the simple drive is a useful upstream pressure, but the model's later processing determines the final activation maps.
"""
    path.write_text(outline, encoding="utf-8")


def build_pdf(run_dir: Path, out_dir: Path, *, figure_set: str, dpi: int) -> tuple[Path, dict[str, Path]]:
    input_paths = paths(run_dir, figure_set=figure_set)
    check_inputs(input_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "temporal_frequency_mechanism_plain_language_walkthrough.pdf"

    unit_sentence = selected_unit_sentence(input_paths["selected_units"])
    stat_lines = relation_stat_lines(input_paths["population_stats"])
    group_lines = group_summary_lines(input_paths["population_groups"])

    with PdfPages(pdf_path) as pdf:
        page = 1

        fig = new_page(page)
        add_title(fig, "Temporal Frequency Mechanism Analysis", "A plain-language walkthrough")
        add_wrapped_text(
            fig,
            0.07,
            0.76,
            "Question: when the eye moves across an image, does that movement create input patterns that help explain how the model's maps change?",
            width=74,
            fontsize=16,
        )
        add_wrapped_text(
            fig,
            0.07,
            0.62,
            "Short answer so far: the simple input pattern is real and useful, but it does not by itself explain the final maps.",
            width=74,
            fontsize=16,
            weight="bold",
        )
        add_bullets(
            fig,
            0.09,
            0.45,
            [
                "SF means image detail. Low detail means broad, smooth structure. High detail means fine texture and edges.",
                "SSI is the map information score. Higher means the map points more clearly to a location.",
                "Linear power drive is our simple input estimate before the model adds its later processing.",
            ],
            width=76,
            fontsize=13,
        )
        save_page(pdf, fig, dpi)
        page += 1

        if "preference_audit" in input_paths:
            fig = new_page(page)
            add_title(fig, "First: Use the New RR100 Preferences")
            add_image(fig, input_paths["preference_audit"], (0.05, 0.20, 0.90, 0.62))
            add_bullets(
                fig,
                0.07,
                0.16,
                [
                    "The parametric model changes which units count as low, middle, and high SF for this analysis.",
                    "From here on, SF group, motion-induced TF, and linear drive all use the same parametric preferred SF/TF table.",
                ],
                width=100,
                fontsize=10.7,
                line_gap=0.052,
            )
            save_page(pdf, fig, dpi)
            page += 1

        fig = new_page(page)
        add_title(fig, "The Simple Story We Are Testing")
        add_box(fig, (0.06, 0.52), (0.20, 0.17), "The eye moves across the image.", face="#f6f6f6", edge="#bbbbbb")
        add_box(fig, (0.315, 0.52), (0.20, 0.17), "That movement makes the image flicker over time.", face="#f6f6f6", edge="#bbbbbb")
        add_box(fig, (0.57, 0.52), (0.20, 0.17), "Some units receive a stronger simple drive.", face="#f6f6f6", edge="#bbbbbb")
        add_box(fig, (0.81, 0.52), (0.14, 0.17), "Maps may change.", face="#f6f6f6", edge="#bbbbbb")
        add_arrow(fig, (0.265, 0.605), (0.31, 0.605))
        add_arrow(fig, (0.52, 0.605), (0.565, 0.605))
        add_arrow(fig, (0.775, 0.605), (0.805, 0.605))
        add_wrapped_text(
            fig,
            0.07,
            0.33,
            "This is only the first-pass story. It does not include all the model's later steps. The goal is to ask whether this simple drive is pushing the effect, not whether it fully explains everything.",
            width=78,
            fontsize=15,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Step 1: Movement Creates Different Time Patterns")
        add_image(fig, input_paths["logic"], (0.04, 0.20, 0.92, 0.56))
        add_bullets(
            fig,
            0.07,
            0.17,
            [
                "The black trace is eye movement speed over time.",
                "The same movement can be slow for one unit and fast for another. Drive is strongest when image detail and movement speed match the unit.",
            ],
            width=90,
            fontsize=11.3,
            line_gap=0.065,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Step 2: Follow Three Concrete Units")
        add_image(fig, input_paths["trace_gallery"], (0.05, 0.21, 0.90, 0.60))
        add_bullets(
            fig,
            0.07,
            0.17,
            [
                "We keep the image fixed and compare several movement traces.",
                unit_sentence,
            ],
            width=106,
            fontsize=10.7,
            line_gap=0.043,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Step 3: Separate Drive, Activation, and Map Score")
        add_image(fig, input_paths["example_summary"], (0.06, 0.31, 0.88, 0.42))
        add_bullets(
            fig,
            0.08,
            0.25,
            [
                "The blue unit gets strong simple drive, but its activation and map score go down.",
                "The green unit shows a clear activation increase and a map-score increase.",
                "The orange unit gets more activation, but its map score changes little. This is why we keep the three steps separate.",
            ],
            width=92,
            fontsize=11.8,
            line_gap=0.052,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Step 4: Check the Maps Directly")
        add_image(fig, input_paths["activation_maps"], (0.05, 0.13, 0.90, 0.72))
        add_wrapped_text(
            fig,
            0.07,
            0.095,
            "Each unit has rows for stabilized, moving, and moving-minus-stabilized maps. The important point is visual: the maps do not all change in the same way.",
            width=110,
            fontsize=10.5,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Step 5: Follow the Same Maps Over Time")
        add_image(fig, input_paths["map_timecourses"], (0.09, 0.16, 0.82, 0.65))
        add_bullets(
            fig,
            0.08,
            0.13,
            [
                "Activation and map score can move differently over time.",
                "More activity is not automatically the same as a more informative map.",
            ],
            width=96,
            fontsize=11.2,
            line_gap=0.043,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Step 6: Ask the Same Questions Across the Population")
        add_image(fig, input_paths["population_bridge"], (0.04, 0.20, 0.92, 0.65))
        add_bullets(
            fig,
            0.07,
            0.15,
            [
                "If one simple link controlled everything, the dots would form a clear upward trend.",
                "Near-zero links across raw examples: " + "; ".join(line.replace(" across raw examples.", "") for line in stat_lines) + ".",
            ],
            width=102,
            fontsize=10.4,
            line_gap=0.052,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "What the Population Figure Says")
        add_wrapped_text(
            fig,
            0.07,
            0.78,
            "The simple drive changes across unit groups, but it does not cleanly predict the next two steps.",
            width=82,
            fontsize=16,
            weight="bold",
        )
        add_bullets(
            fig,
            0.09,
            0.61,
            [
                *group_lines,
                "The example units sit inside the population, but they are best used as a descriptive guide.",
                "The main result is not a straight line. It is a chain with extra model processing in the middle.",
            ],
            width=80,
            fontsize=13.2,
            line_gap=0.075,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Takeaway")
        add_bullets(
            fig,
            0.08,
            0.78,
            [
                "Eye movement changes the time pattern of image input.",
                "That simple input drive is a good candidate for the first push in the system.",
                "But the model does more than pass that drive through.",
                "More simple drive does not always mean more activation.",
                "More activation does not always mean a more informative map.",
                "So the next scientific question is: which later model steps turn the simple drive into map changes?",
            ],
            width=82,
            fontsize=14.2,
            line_gap=0.072,
        )
        save_page(pdf, fig, dpi)
        page += 1

        fig = new_page(page)
        add_title(fig, "Audit Trail")
        add_wrapped_text(
            fig,
            0.07,
            0.80,
            "The PDF is a reader-facing guide. The original figures and tables remain saved as separate checkpoint artifacts.",
            width=88,
            fontsize=14,
        )
        audit_lines = [
            f"Full input paths are saved in: {out_dir / 'temporal_frequency_mechanism_plain_language_walkthrough_metadata.json'}",
            f"Mechanism panels: {input_paths['logic'].parent}",
            f"Trace examples: {input_paths['trace_gallery'].parent}",
            f"Map examples and summary: {input_paths['example_summary'].parent}",
            f"Population bridge: {input_paths['population_bridge'].parent}",
        ]
        add_bullets(fig, 0.08, 0.61, audit_lines, width=112, fontsize=8.6, line_gap=0.073)
        save_page(pdf, fig, dpi)

    return pdf_path, input_paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_PARAMETRIC_OUT_DIR if str(args.figure_set) == "parametric" else DEFAULT_OUT_DIR
    )
    pdf_path, input_paths = build_pdf(Path(args.run_dir), out_dir, figure_set=str(args.figure_set), dpi=int(args.dpi))
    outline_path = out_dir / "temporal_frequency_mechanism_plain_language_walkthrough_outline.md"
    write_outline(outline_path)
    write_json(
        out_dir / "temporal_frequency_mechanism_plain_language_walkthrough_metadata.json",
        {
            "analysis": "temporal_frequency_mechanism_plain_language_walkthrough",
            "run_dir": Path(args.run_dir),
            "out_dir": out_dir,
            "figure_set": str(args.figure_set),
            "pdf": pdf_path,
            "outline": outline_path,
            "input_artifacts": input_paths,
            "language_policy": "plain-language reader-facing walkthrough; unavoidable labels SF and SSI are defined on page 1",
        },
    )
    print(f"Wrote {pdf_path}")
    print(f"Wrote {outline_path}")


if __name__ == "__main__":
    main()
