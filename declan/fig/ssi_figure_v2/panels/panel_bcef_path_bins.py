#!/usr/bin/env python3
"""Panels B/C/E/F: trajectory-path bins from the SF0.5/coh0.20/match15 run.

This compact renderer reuses the precomputed output from
``declan/active_sensing_movie_information/make_backimage_panel_b_orientation_match_15deg_sf05.py``.
That upstream wrapper configures the BackImage story-panel code to use
low SF < 0.5, high SF >= 0.5, contour coherence >= 0.20, and a 15 degree
unit-contour match threshold for the aligned panels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information import make_backimage_panel_b_orientation_match_15deg_sf05 as panel_b_sf05  # noqa: E402


panel_b = panel_b_sf05.panel_b
panel_b._configure_story()
story = panel_b.story

OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
PLOT_COLLECTION_DIR = story.OUT_DIR
SOURCE_STEM = panel_b.OUT_STEM
VALUES_CSV = PLOT_COLLECTION_DIR / f"{SOURCE_STEM}_values.csv"
SELECTION_CSV = PLOT_COLLECTION_DIR / f"{SOURCE_STEM}_selection_summary.csv"
SUMMARY_JSON = PLOT_COLLECTION_DIR / f"{SOURCE_STEM}_summary.json"
UPSTREAM_WRAPPER = (
    ROOT / "declan" / "active_sensing_movie_information" / "make_backimage_panel_b_orientation_match_15deg_sf05.py"
)
UPSTREAM_CORE = (
    ROOT / "declan" / "active_sensing_movie_information" / "make_backimage_panel_b_orientation_match_15deg.py"
)
UPSTREAM_STORY = (
    ROOT
    / "declan"
    / "active_sensing_movie_information"
    / "make_backimage_reordered_geometry_story_figure_cell_baseline_sf075_coh020_cde8bins.py"
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#6B6F75"
INK = "#111111"
BROKEN_AXIS_BREAK_LEFT = 0.27
BROKEN_AXIS_BREAK_RIGHT = 0.82
BROKEN_AXIS_BREAK_CENTER = 0.545

PANEL_SPECS = {
    "B": {
        "title": "Low-SF units",
        "sf_group": "low_lt0p5",
        "relation": "strong_contours_no_osi",
        "color": BLUE,
        "ylabel": "SSI change (%)",
    },
    "C": {
        "title": "High-SF units",
        "sf_group": "high_ge0p75",
        "relation": "strong_contours_no_osi",
        "color": ORANGE,
        "ylabel": "SSI change (%)",
    },
    "E": {
        "title": "Low-SF aligned units",
        "sf_group": "low_lt0p5",
        "relation": "contour_matched",
        "color": BLUE,
        "ylabel": "SSI change (%)",
    },
    "F": {
        "title": "High-SF aligned units",
        "sf_group": "high_ge0p75",
        "relation": "contour_matched",
        "color": ORANGE,
        "ylabel": "SSI change (%)",
    },
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_panel_values(values_csv: Path = VALUES_CSV) -> pd.DataFrame:
    if not values_csv.exists():
        raise FileNotFoundError(values_csv)
    values = pd.read_csv(values_csv)
    required = {
        "sf_group",
        "sf_group_label",
        "relation",
        "context",
        "path_median_arcmin",
        "path_bin_order",
        "ssi_percent_vs_cell_baseline",
        "n_selected_units",
        "n_selected_unit_image_pairs",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Missing required columns in {values_csv}: {missing}")
    return values


def shared_ylim(values: pd.DataFrame) -> tuple[float, float]:
    if values.empty:
        return (-20.0, 48.0)
    return story._shared_ylim([values["ssi_percent_vs_cell_baseline"]])


def shared_ylim_for(values: pd.DataFrame, *, relations: tuple[str, ...]) -> tuple[float, float]:
    if values.empty:
        return (-20.0, 48.0)
    frame = values[values["relation"].isin(relations)].copy()
    if frame.empty:
        return shared_ylim(values)
    return story._shared_ylim([frame["ssi_percent_vs_cell_baseline"]])


def path_xlimit_right(values: pd.DataFrame, *, pad: float = 0.22) -> float:
    if values.empty or "path_median_arcmin" not in values:
        return 5.55
    raw = pd.to_numeric(values["path_median_arcmin"], errors="coerce").to_numpy(dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return 5.55
    mapped = story._x_broken_log(raw, min_pos=story.B_MIN_POS, max_pos=story.B_MAX_POS)
    return float(np.nanmax(mapped) + pad)


def _remove_upstream_break_label(ax: plt.Axes) -> None:
    for text in list(ax.texts):
        if text.get_text() == "//":
            text.remove()


def _draw_broken_x_axis(ax: plt.Axes) -> None:
    _remove_upstream_break_label(ax)
    ax.spines["bottom"].set_visible(False)
    trans = ax.get_xaxis_transform()
    x_left, x_right = ax.get_xlim()
    ax.plot(
        [x_left, BROKEN_AXIS_BREAK_LEFT],
        [0.0, 0.0],
        transform=trans,
        color="black",
        lw=0.8,
        clip_on=False,
        zorder=10,
    )
    ax.plot(
        [BROKEN_AXIS_BREAK_RIGHT, x_right],
        [0.0, 0.0],
        transform=trans,
        color="black",
        lw=0.8,
        clip_on=False,
        zorder=10,
    )
    for offset in (-0.040, 0.040):
        ax.plot(
            [
                BROKEN_AXIS_BREAK_CENTER + offset - 0.035,
                BROKEN_AXIS_BREAK_CENTER + offset + 0.035,
            ],
            [-0.033, 0.033],
            transform=trans,
            color="black",
            lw=1.05,
            clip_on=False,
            solid_capstyle="butt",
            zorder=11,
        )


def format_broken_path_axis(ax: plt.Axes, *, xlim_right: float | None = None) -> None:
    story._format_broken_axis(
        ax,
        ticks=story.B_TICKS,
        min_pos=story.B_MIN_POS,
        max_pos=story.B_MAX_POS,
        xlabel="path length (arcmin)",
    )
    if xlim_right is not None:
        ax.set_xlim(-0.12, xlim_right)
    _draw_broken_x_axis(ax)


def load_provenance(
    *,
    values_csv: Path = VALUES_CSV,
    selection_csv: Path = SELECTION_CSV,
    summary_json: Path = SUMMARY_JSON,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    return {
        "panels": ["B", "C", "E", "F"],
        "source_wrapper_script": _relative(UPSTREAM_WRAPPER),
        "source_core_script": _relative(UPSTREAM_CORE),
        "source_story_script": _relative(UPSTREAM_STORY),
        "source_values_csv": _relative(values_csv),
        "source_selection_csv": _relative(selection_csv),
        "source_summary_json": _relative(summary_json),
        "selection": summary.get(
            "selection",
            {
                "sf_metric_col": story.SF_METRIC_COL,
                "low_sf": f"{story.SF_METRIC_COL} < {story.LOW_SF_MAX_CPD}",
                "high_sf": f"{story.SF_METRIC_COL} >= {panel_b.HIGH_SF_MIN_CPD}",
                "contour_coherence_min": story.CONTOUR_COHERENCE_MIN,
                "min_osi": story.MIN_OSI,
                "match_max_deg": panel_b.MATCH_MAX_DEG,
                "orthogonal_min_deg": story.ORTHOGONAL_MIN_DEG,
                "panel_b_drift_bins": story.N_DRIFT_BINS,
                "panel_b_microsaccade_bins": story.N_MICROSACCADE_BINS,
            },
        ),
        "baseline": summary.get(
            "baseline",
            "Each plotted nonzero bin is compared with a cell-matched stabilized baseline weighted by that bin's image composition.",
        ),
    }


def _add_support_note(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    n_units = int(pd.to_numeric(frame["n_selected_units"], errors="coerce").dropna().iloc[0])
    n_pairs = int(pd.to_numeric(frame["n_selected_unit_image_pairs"], errors="coerce").dropna().iloc[0])
    # Upper-left, not upper-right: these series start near SSI=0 at path=0
    # and only grow larger later, so the top-right corner is exactly where
    # the last (often highest) data point and this note used to collide.
    ax.text(
        0.045,
        0.930,
        f"{n_units} units\n{n_pairs} pairs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=GRAY,
        linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
    )


def _draw_two_color_title(ax: plt.Axes, label: str, title: str, *, color: str) -> None:
    """Panel letter in black (matching every other lettered panel in the
    figure) with the descriptive title text in the population color, as two
    separate Text artists -- ax.set_title only takes one color for the
    whole string."""
    y = 1.020
    ax.text(0.0, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(
        0.085, y, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, fontweight="bold", color=color
    )


def _add_microsaccade_legend(ax: plt.Axes, *, color: str) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], color=color, marker="o", markerfacecolor="white", markeredgewidth=1.1, lw=1.4, label="drift only"
        ),
        Line2D([0], [0], color=color, marker="o", markerfacecolor=color, markeredgewidth=1.1, lw=1.4, label="microsaccade"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=5.6,
        loc="lower right",
        handlelength=1.5,
        labelspacing=0.3,
        borderaxespad=0.2,
        handletextpad=0.4,
    )


def draw_panel(
    ax: plt.Axes,
    *,
    label: str,
    title: str,
    sf_group: str,
    relation: str,
    color: str,
    ylabel: str | None,
    values: pd.DataFrame | None = None,
    ylim: tuple[float, float] | None = None,
    show_microsaccade_legend: bool = False,
) -> pd.DataFrame:
    values = load_panel_values() if values is None else values.copy()
    frame = values[values["sf_group"].eq(sf_group) & values["relation"].eq(relation)].copy()
    if frame.empty:
        raise ValueError(f"No values for sf_group={sf_group!r}, relation={relation!r}")

    story._plot_b_series(ax, frame, color=color)
    format_broken_path_axis(ax, xlim_right=path_xlimit_right(frame))
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_ylim(*(shared_ylim(values) if ylim is None else ylim))
    _draw_two_color_title(ax, label, title, color=color)
    if ylabel:
        ax.set_ylabel(ylabel)
    _add_support_note(ax, frame)
    if show_microsaccade_legend:
        _add_microsaccade_legend(ax, color=color)
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    return frame


def build_panel(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_panel_values()
    values.to_csv(out_dir / "panel_bcef_path_bins_values.csv", index=False)
    (out_dir / "panel_bcef_path_bins_provenance.json").write_text(
        json.dumps(load_provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(4.85, 4.50), constrained_layout=True, sharey=False)
    top_ylim = shared_ylim_for(values, relations=("strong_contours_no_osi",))
    bottom_ylim = shared_ylim_for(values, relations=("contour_matched",))
    for ax, label in zip(axes.flat, ["B", "C", "E", "F"], strict=True):
        spec = PANEL_SPECS[label]
        ylim = top_ylim if label in {"B", "C"} else bottom_ylim
        draw_panel(ax, label=label, values=values, ylim=ylim, **spec)
    paths = {
        "png": out_dir / "panel_bcef_path_bins.png",
        "pdf": out_dir / "panel_bcef_path_bins.pdf",
        "svg": out_dir / "panel_bcef_path_bins.svg",
    }
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    paths = build_panel(args.out_dir)
    for key in ("png", "pdf", "svg"):
        print(paths[key])


if __name__ == "__main__":
    main()
