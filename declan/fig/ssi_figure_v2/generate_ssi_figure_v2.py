#!/usr/bin/env python3
"""Generate a draft SSI multipanel figure scaffold.

This version is composition-first: it preserves the layout and storyboard of
the supplied PDF sketch, pulls in existing BackImage story-figure plotting code
where available, and leaves the remaining result/data slots as explicit
placeholders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information import (  # noqa: E402
    make_backimage_reordered_geometry_story_figure_cell_baseline_sf075 as story_panels,
)

try:  # noqa: E402
    from declan.fig_ssi import make_ssi_contour_schematic as ssi_schematic

    SCHEMATIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    ssi_schematic = None
    SCHEMATIC_IMPORT_ERROR = exc

try:  # noqa: E402
    from panels import panel_bcef_path_bins

    PANEL_BCEF_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_bcef_path_bins = None
    PANEL_BCEF_IMPORT_ERROR = exc

try:  # noqa: E402
    from panels import panel_g_matched_bins_bracket

    PANEL_G_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_g_matched_bins_bracket = None
    PANEL_G_IMPORT_ERROR = exc

try:  # noqa: E402
    from panels import panel_h_unwrapped_edge_coherence, panel_i_edge_alignment

    PANEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_h_unwrapped_edge_coherence = None
    panel_i_edge_alignment = None
    PANEL_IMPORT_ERROR = exc

OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2"
PLOT_COLLECTION_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
    / "merged"
    / "phase1_phase2_conditioning_v1"
    / "plot_collections"
)
STORY_STEM = "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075_coh020_cde8bins"
PANEL_BCEF_STEM = "backimage_real_trace_panel_b_cell_baseline_sf05_coh020_match15"
PANEL_B_VALUES_CSV = PLOT_COLLECTION_DIR / f"{PANEL_BCEF_STEM}_values.csv"
COMPONENT_VALUES_CSV = PLOT_COLLECTION_DIR / f"{STORY_STEM}_component_values.csv"

BLUE = "#0072B2"
CYAN = "#00A9C8"
ORANGE = "#D55E00"
RED = "#C9252D"
GRAY = "#6B6F75"
INK = "#111111"
PALE_GRID = "#E7E7E7"
PLACEHOLDER_FILL = "#F6F7F8"
PLACEHOLDER_EDGE = "#B9BFC6"
FIGURE_SIZE_IN = (8.5, 11.0)
MAIN_GRID_KWARGS = {
    "left": 0.075,
    "right": 0.955,
    "top": 0.925,
    "bottom": 0.060,
    "width_ratios": [1.12, 1.12, 0.92],
    "height_ratios": [1.13, 1.05, 1.03],
    "hspace": 0.430,
    "wspace": 0.360,
}
RIGHT_PANEL_HSPACE = 0.54
PANEL_BOX_LABELS = {
    "A": "Motion schematic",
    "B": "Low-SF units",
    "C": "High-SF units",
    "D": "Contour-relative stimulus",
    "E": "Low-SF aligned",
    "F": "High-SF aligned",
    "G": "Aligned high-SF components",
    "H": "FEM spread",
    "I": "Edge alignment",
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


def read_existing_story_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load precomputed BackImage panel values used by the figure."""
    panel_b = pd.read_csv(PANEL_B_VALUES_CSV) if PANEL_B_VALUES_CSV.exists() else pd.DataFrame()
    component = pd.read_csv(COMPONENT_VALUES_CSV) if COMPONENT_VALUES_CSV.exists() else pd.DataFrame()
    return panel_b, component


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_methods_provenance(out_dir: Path = OUT_DIR) -> dict:
    provenance: dict[str, object] = {
        "figure": "ssi_figure_v2",
        "outputs": {
            "png": _relative(out_dir / "ssi_figure_v2.png"),
            "pdf": _relative(out_dir / "ssi_figure_v2.pdf"),
            "svg": _relative(out_dir / "ssi_figure_v2.svg"),
            "panel_boxes_svg": _relative(out_dir / "ssi_figure_v2_panel_boxes.svg"),
        },
        "panels": {},
    }
    panels = provenance["panels"]
    if panel_bcef_path_bins is not None:
        try:
            panels["B_C_E_F"] = panel_bcef_path_bins.load_provenance()
        except Exception as exc:
            panels["B_C_E_F"] = {"status": "provenance load failed", "error": repr(exc)}
    else:
        panels["B_C_E_F"] = {"status": "panel import failed", "error": repr(PANEL_BCEF_IMPORT_ERROR)}

    if panel_g_matched_bins_bracket is not None:
        try:
            panels["G"] = panel_g_matched_bins_bracket.load_provenance()
        except Exception as exc:
            panels["G"] = {"status": "provenance load failed", "error": repr(exc)}
    else:
        panels["G"] = {"status": "panel import failed", "error": repr(PANEL_G_IMPORT_ERROR)}

    panels["H"] = {
        "source_script": _relative(ROOT / "declan" / "fig" / "ssi_figure_v2" / "panels" / "panel_h_unwrapped_edge_coherence.py"),
        "source_profile_csv": _relative(panel_h_unwrapped_edge_coherence.PROFILE_CSV)
        if panel_h_unwrapped_edge_coherence is not None
        else None,
    }
    panels["I"] = {
        "source_script": _relative(ROOT / "declan" / "fig" / "ssi_figure_v2" / "panels" / "panel_i_edge_alignment.py"),
        "source_windows_csv": _relative(panel_i_edge_alignment.WINDOWS_CSV)
        if panel_i_edge_alignment is not None
        else None,
    }
    return provenance


def _svg_number(value: float) -> str:
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def compute_panel_box_layout() -> dict[str, dict[str, float]]:
    """Return current panel axes positions in SVG point coordinates."""
    page_w = FIGURE_SIZE_IN[0] * 72.0
    page_h = FIGURE_SIZE_IN[1] * 72.0
    fig = plt.figure(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    gs = fig.add_gridspec(3, 3, **MAIN_GRID_KWARGS)
    axes: dict[str, plt.Axes] = {}
    axes["A"] = fig.add_subplot(gs[0, :2])

    gs_b_right = gs[0, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    axes["B"] = fig.add_subplot(gs_b_right[0, 0])
    axes["C"] = fig.add_subplot(gs_b_right[1, 0])

    axes["D"] = fig.add_subplot(gs[1, :2])
    gs_a_right = gs[1, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    axes["E"] = fig.add_subplot(gs_a_right[0, 0])
    axes["F"] = fig.add_subplot(gs_a_right[1, 0])

    axes["G"] = fig.add_subplot(gs[2, 0])
    axes["H"] = fig.add_subplot(gs[2, 1])
    axes["I"] = fig.add_subplot(gs[2, 2])
    fig.canvas.draw()

    boxes: dict[str, dict[str, float]] = {}
    for label, ax in axes.items():
        pos = ax.get_position()
        boxes[label] = {
            "x": float(pos.x0 * page_w),
            "y": float((1.0 - pos.y1) * page_h),
            "width": float(pos.width * page_w),
            "height": float(pos.height * page_h),
        }
    plt.close(fig)
    return boxes


def write_panel_boxes_svg(out_dir: Path = OUT_DIR) -> Path:
    """Write an editable empty-box SVG matching the current panel layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_w = FIGURE_SIZE_IN[0] * 72.0
    page_h = FIGURE_SIZE_IN[1] * 72.0
    boxes = compute_panel_box_layout()
    path = out_dir / "ssi_figure_v2_panel_boxes.svg"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_svg_number(page_w)}pt" '
            f'height="{_svg_number(page_h)}pt" viewBox="0 0 {_svg_number(page_w)} {_svg_number(page_h)}">'
        ),
        '  <title>SSI figure v2 editable panel bounding boxes</title>',
        '  <desc>Empty panel boxes generated from the current matplotlib GridSpec layout.</desc>',
        '  <rect id="page" x="0" y="0" width="100%" height="100%" fill="white" stroke="#d0d4d9" stroke-width="0.75"/>',
        '  <g id="panel-bounding-boxes" fill="none" stroke="#0072B2" stroke-width="1.25" stroke-dasharray="5 3">',
    ]
    for label in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        box = boxes[label]
        lines.append(f'    <g id="panel-{label}" data-panel="{label}">')
        lines.append(
            '      <rect '
            f'id="panel-{label}-box" '
            f'x="{_svg_number(box["x"])}" '
            f'y="{_svg_number(box["y"])}" '
            f'width="{_svg_number(box["width"])}" '
            f'height="{_svg_number(box["height"])}" />'
        )
        lines.append(
            '      <text '
            f'id="panel-{label}-label" '
            f'x="{_svg_number(box["x"] + 5.0)}" '
            f'y="{_svg_number(box["y"] + 12.0)}" '
            'font-family="DejaVu Sans, Arial, sans-serif" '
            'font-size="9" '
            'font-weight="700" '
            'fill="#111111" '
            'stroke="none">'
            f'{label}  {PANEL_BOX_LABELS[label]}</text>'
        )
        lines.append("    </g>")
    lines.extend(["  </g>", "</svg>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_schematic_payload() -> dict | None:
    if ssi_schematic is None:
        return None
    try:
        return ssi_schematic.load_real_payload()
    except Exception:
        return None


def schematic_response_maps(payload: dict | None) -> tuple[dict[str, object], tuple[float, float] | None]:
    if ssi_schematic is None or payload is None:
        return {}, None
    maps = payload.get("schematic_rr100_final_maps")
    condition_ids = payload.get("schematic_rr100_final_condition_id")
    unit_row = ssi_schematic.choose_right_panel_real_unit(payload)
    if maps is None or condition_ids is None or unit_row is None:
        return {}, None
    try:
        ids = [str(x) for x in condition_ids]
        real_idx = ids.index("real_trace_final")
        stable_idx = ids.index("endpoint_stabilized_final")
        unit_idx = int(unit_row["unit_index"])
        real_maps = {
            "fem": maps[real_idx, unit_idx],
            "stable": maps[stable_idx, unit_idx],
        }
        return real_maps, ssi_schematic.panel_b_map_pair_limits(real_maps.values())
    except Exception:
        return {}, None


def shared_story_ylim(frame: pd.DataFrame, *, fallback: tuple[float, float]) -> tuple[float, float]:
    if frame.empty or "ssi_percent_vs_cell_baseline" not in frame:
        return fallback
    return story_panels._shared_ylim([frame["ssi_percent_vs_cell_baseline"]])


def first_int(frame: pd.DataFrame, column: str) -> int | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return int(values.iloc[0]) if not values.empty else None


def add_support_note(ax: plt.Axes, frame: pd.DataFrame) -> None:
    n_units = first_int(frame, "n_selected_units")
    n_pairs = first_int(frame, "n_selected_unit_image_pairs")
    if n_units is None or n_pairs is None:
        return
    ax.text(
        0.985,
        0.930,
        f"{n_units} units\n{n_pairs} pairs",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=GRAY,
        linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
    )


def draw_panel_header(ax: plt.Axes, letter: str, title: str, *, y: float = 1.025) -> None:
    ax.text(
        0.000,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.052,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        clip_on=False,
    )


def set_panel_title(
    ax: plt.Axes,
    label: str,
    title: str,
    *,
    color: str = INK,
    fontsize: float = 8.6,
    pad: float = 3.0,
) -> None:
    ax.set_title(
        f"{label}  {title}",
        loc="left",
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        pad=pad,
    )


def placeholder_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    sublabel: str | None = None,
    hatch: str = "///",
    edgecolor: str = PLACEHOLDER_EDGE,
    label_size: float = 7.0,
) -> None:
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=PLACEHOLDER_FILL,
            edgecolor=edgecolor,
            linewidth=0.9,
            hatch=hatch,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2 + (0.012 if sublabel else 0.0),
        label,
        ha="center",
        va="center",
        fontsize=label_size,
        color="#343A40",
        fontweight="bold",
        linespacing=1.1,
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h * 0.25,
            sublabel,
            ha="center",
            va="center",
            fontsize=max(label_size - 1.2, 5.2),
            color=GRAY,
            linespacing=1.08,
        )


def add_flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="data",
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0, mutation_scale=12),
    )


def draw_movie_cube(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    jittered: bool,
) -> None:
    dx = 0.070
    dy = 0.052
    top = patches.Polygon(
        [(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor="#F1F5F6",
        edgecolor=CYAN,
        lw=1.4,
        hatch="//" if jittered else "--",
    )
    side = patches.Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor="#E7ECEE",
        edgecolor=CYAN,
        lw=1.4,
        hatch="//" if jittered else "--",
    )
    ax.add_patch(top)
    ax.add_patch(side)
    placeholder_box(
        ax,
        x,
        y,
        w,
        h,
        "movie\nplaceholder",
        hatch="//" if jittered else "--",
        edgecolor=CYAN,
        label_size=6.3,
    )
    ax.text(x + w / 2, y + h + dy + 0.035, label, ha="center", va="bottom", fontsize=7.0, color=INK)
    ax.annotate(
        "",
        xy=(x + w + dx * 0.72, y - 0.040),
        xytext=(x + 0.020, y - 0.040),
        arrowprops=dict(arrowstyle="-|>", lw=0.85, color=GRAY, mutation_scale=9),
    )
    ax.text(x + 0.5 * w, y - 0.066, "267 ms", ha="center", va="top", fontsize=6.0, color=GRAY)


def draw_schematic_movie_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    schematic_payload: dict | None,
    trace_key: str,
    trace_color: str,
    fallback_jittered: bool,
) -> None:
    if ssi_schematic is None or schematic_payload is None:
        draw_movie_cube(ax, x, y + 0.035, w * 0.82, h * 0.56, label=label, jittered=fallback_jittered)
        return
    try:
        cube_ax = ax.inset_axes([x, y, w, h], transform=ax.transData)
        ssi_schematic.add_visual_model_input_cube(
            cube_ax,
            schematic_payload.get("patch"),
            schematic_payload.get("contour_axis_image_deg", 10.352312),
            show_motion_labels=False,
            show_model_labels=False,
            source_image=schematic_payload.get("stimulus_model_source_patch"),
            trace_xy=schematic_payload.get(trace_key),
            trace_path_color=trace_color,
            show_motion_overlay=False,
        )
        ax.text(x + w / 2, y + h + 0.008, label, ha="center", va="bottom", fontsize=7.0, color=INK)
    except Exception:
        draw_movie_cube(ax, x, y + 0.035, w * 0.82, h * 0.56, label=label, jittered=fallback_jittered)


def draw_model_icon(ax: plt.Axes, x: float, y: float, s: float = 1.0) -> None:
    core = patches.FancyBboxPatch(
        (x, y),
        0.058 * s,
        0.155 * s,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        facecolor="#DDE2E0",
        edgecolor="#AAB1AE",
        lw=0.8,
    )
    ax.add_patch(core)
    ax.text(x + 0.029 * s, y + 0.127 * s, "Core", ha="center", va="center", fontsize=5.6)
    ax.text(x + 0.029 * s, y + 0.105 * s, "Conv", ha="center", va="center", fontsize=4.8)
    for i, color in enumerate(["#E97B68", "#8FC9CF", "#F0C06C", "#E6A34E"]):
        yy = y + 0.029 * s + i * 0.025 * s
        ax.add_patch(
            patches.Rectangle(
                (x + 0.012 * s, yy),
                0.030 * s,
                0.014 * s,
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
            )
        )
    ax.text(x + 0.029 * s, y - 0.014 * s, "model", ha="center", va="top", fontsize=5.2, color=GRAY)

    add_flow_arrow(ax, (x + 0.068 * s, y + 0.077 * s), (x + 0.094 * s, y + 0.077 * s))
    stack_x = x + 0.106 * s
    for j, color in enumerate(["#E66A52", "#56B4AE", "#F0C05A", "#79A96B"]):
        ax.add_patch(
            patches.Rectangle(
                (stack_x + 0.006 * j * s, y + 0.062 * s + 0.004 * j * s),
                0.040 * s,
                0.029 * s,
                facecolor="white",
                edgecolor=color,
                linewidth=0.75,
            )
        )
    add_flow_arrow(ax, (x + 0.160 * s, y + 0.077 * s), (x + 0.189 * s, y + 0.077 * s))

    rf_x = x + 0.204 * s
    for j in range(3):
        ax.add_patch(
            patches.Rectangle(
                (rf_x + 0.016 * j * s, y + 0.037 * s + 0.021 * j * s),
                0.049 * s,
                0.080 * s,
                facecolor="#EFF7EF",
                edgecolor="#49834E",
                linewidth=0.75,
                alpha=0.75,
            )
        )
    ax.plot([rf_x + 0.017 * s, rf_x + 0.083 * s], [y + 0.047 * s, y + 0.123 * s], color="#267335", lw=1.0)
    ax.plot(rf_x + 0.050 * s, y + 0.084 * s, marker="o", ms=2.4, color="#267335")
    ax.text(rf_x + 0.058 * s, y + 0.134 * s, "x,y", fontsize=4.8, color="#267335")
    ax.text(rf_x + 0.040 * s, y + 0.017 * s, "one response\nper position", ha="center", va="top", fontsize=4.7, color=GRAY)


def draw_response_placeholder(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    real_map=None,
    map_vlim: tuple[float, float] | None = None,
) -> None:
    if ssi_schematic is not None and real_map is not None:
        map_ax = ax.inset_axes([x, y, w, h], transform=ax.transData)
        ssi_schematic.add_spatial_activation_map(
            map_ax,
            "fem",
            real_map=real_map,
            map_vlim=map_vlim,
        )
        ax.text(
            x + w / 2,
            y - 0.020,
            label.replace("\n", " "),
            ha="center",
            va="top",
            fontsize=5.7,
            color=GRAY,
        )
        ax.text(x - 0.018, y + 0.012, "map\npixel", ha="right", va="bottom", fontsize=5.3, color=GRAY)
        return
    placeholder_box(
        ax,
        x,
        y,
        w,
        h,
        label,
        sublabel="activation map\nfrom real run",
        hatch="xx",
        label_size=6.7,
    )
    ax.text(x - 0.018, y + 0.012, "map\npixel", ha="right", va="bottom", fontsize=5.3, color=GRAY)


def draw_panel_b(ax: plt.Axes, *, schematic_payload: dict | None = None) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    real_maps, real_map_vlim = schematic_response_maps(schematic_payload)
    draw_panel_header(ax, "A", "Motion sharpens unit activations across space", y=1.010)
    ax.text(0.045, 0.892, "FEM jittered movie", fontsize=11.0, ha="left", va="bottom")
    ax.text(0.045, 0.448, "Stabilized movie", fontsize=11.0, ha="left", va="bottom")
    map_note = "warm/cool = above/below map mean" if real_maps else "reserved for above/below-mean maps"
    ax.text(0.807, 0.886, map_note, fontsize=6.4, color=GRAY, ha="center")

    draw_schematic_movie_block(
        ax,
        0.050,
        0.580,
        0.225,
        0.235,
        label="model input",
        schematic_payload=schematic_payload,
        trace_key="stimulus_real_trace_lag32",
        trace_color=BLUE,
        fallback_jittered=True,
    )
    draw_model_icon(ax, 0.370, 0.610, s=1.02)
    add_flow_arrow(ax, (0.653, 0.688), (0.695, 0.688))
    draw_response_placeholder(
        ax,
        0.720,
        0.585,
        0.250,
        0.215,
        "FEM response\nmap",
        real_map=real_maps.get("fem"),
        map_vlim=real_map_vlim,
    )
    ax.text(0.468, 0.824, "same spatial kernel, shifted center", fontsize=5.4, color=GRAY, ha="center")

    draw_schematic_movie_block(
        ax,
        0.050,
        0.139,
        0.225,
        0.235,
        label="endpoint-stabilized input",
        schematic_payload=schematic_payload,
        trace_key="stimulus_endpoint_stabilized_trace_lag32",
        trace_color=GRAY,
        fallback_jittered=False,
    )
    draw_model_icon(ax, 0.370, 0.168, s=1.02)
    add_flow_arrow(ax, (0.653, 0.246), (0.695, 0.246))
    draw_response_placeholder(
        ax,
        0.720,
        0.143,
        0.250,
        0.215,
        "stabilized response\nmap",
        real_map=real_maps.get("stable"),
        map_vlim=real_map_vlim,
    )


def draw_panel_a(ax: plt.Axes, *, schematic_payload: dict | None = None) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_panel_header(ax, "D", "Contour-relative stimulus and unit responses", y=1.020)

    full_x, full_y, full_w, full_h = 0.035, 0.595, 0.170, 0.235
    crop_x, crop_y, crop_w, crop_h = 0.300, 0.550, 0.235, 0.315
    has_real_schematic = False
    if ssi_schematic is not None and schematic_payload is not None:
        try:
            full_ax = ax.inset_axes([full_x, full_y, full_w, full_h])
            ssi_schematic.add_source_overview(
                full_ax,
                schematic_payload["stimulus_canvas"],
                schematic_payload["stimulus_crop_center_xy"],
                schematic_payload["stimulus_crop_size_px"],
                label=False,
            )
            crop_ax = ax.inset_axes([crop_x, crop_y, crop_w, crop_h])
            synthetic_left = ssi_schematic.make_synthetic_left_side(
                schematic_payload.get("patch"),
                schematic_payload.get("contour_axis_image_deg", 10.352312),
            )
            ssi_schematic.add_stimulus(
                crop_ax,
                schematic_payload.get("patch"),
                schematic_payload.get("contour_axis_image_deg", 10.352312),
                motion_eye=synthetic_left.get("eye"),
            )
            has_real_schematic = True
        except Exception:
            has_real_schematic = False

    if not has_real_schematic:
        placeholder_box(ax, full_x, full_y, full_w, full_h, "full stimulus", sublabel="image asset", hatch="...", label_size=6.5)
        roi = (full_x + 0.068, full_y + 0.091, 0.032, 0.058)
        ax.add_patch(patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, edgecolor=CYAN, linewidth=1.25))
        placeholder_box(ax, crop_x, crop_y, crop_w, crop_h, "model window", sublabel="151 x 151 crop", hatch="///", label_size=6.9)
        for start_y, end_y in [(roi[1] + roi[3], crop_y + crop_h), (roi[1], crop_y)]:
            ax.plot(
                [roi[0] + roi[2], crop_x],
                [start_y, end_y],
                color=CYAN,
                lw=0.8,
                ls=(0, (3, 3)),
                alpha=0.58,
            )
    else:
        ax.plot([full_x + full_w, crop_x], [full_y + full_h * 0.78, crop_y + crop_h], color=CYAN, lw=0.8, ls=(0, (3, 3)), alpha=0.58)
        ax.plot([full_x + full_w, crop_x], [full_y + full_h * 0.22, crop_y], color=CYAN, lw=0.8, ls=(0, (3, 3)), alpha=0.58)
        ax.text(full_x + full_w / 2, full_y - 0.024, "full stimulus", fontsize=6.4, color=GRAY, ha="center", va="top")
        ax.text(crop_x + crop_w / 2, crop_y - 0.024, "151 x 151 crop", fontsize=6.4, color=GRAY, ha="center", va="top")
    ax.plot([crop_x + 0.06 * crop_w, crop_x + 0.92 * crop_w], [crop_y + 0.52 * crop_h, crop_y + 0.39 * crop_h], color=CYAN, lw=1.2, ls=(0, (4, 3)))
    ax.scatter([crop_x + 0.52 * crop_w], [crop_y + 0.50 * crop_h], s=18, facecolor="white", edgecolor=INK, linewidth=0.8)

    ax.text(0.060, 0.398, "real FEM traces", fontsize=7.5, color=GRAY, ha="left")
    trace_y = [0.330, 0.240]
    for y, color, label in [(trace_y[0], RED, "short path"), (trace_y[1], BLUE, "long path")]:
        ax.plot([0.060, 0.410], [y, y], color=color, lw=1.5, ls=(0, (5, 3)), alpha=0.9)
        ax.text(0.423, y, label, fontsize=6.4, color=color, ha="left", va="center")
    ax.annotate("", xy=(0.410, 0.180), xytext=(0.060, 0.180), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=8))
    ax.text(0.235, 0.153, "frame", fontsize=6.8, ha="center")

    ax.text(0.610, 0.840, "Contour-carried signal", fontsize=10.0, fontweight="bold", ha="left", va="top")
    ax.text(
        0.610,
        0.790,
        "Reserved for the intuitive visual linking\n"
        "contour content, SF tuning, and trajectory size.",
        fontsize=7.2,
        color="#333333",
        ha="left",
        va="top",
        linespacing=1.12,
    )

    for y, label, color in [(0.445, "High-SF unit\nRF placeholder", ORANGE), (0.195, "Low-SF unit\nRF placeholder", BLUE)]:
        placeholder_box(ax, 0.610, y, 0.145, 0.165, label, hatch="///", edgecolor=color, label_size=6.3)
        placeholder_box(ax, 0.790, y + 0.006, 0.185, 0.153, "luminance trace\nplaceholder", hatch="--", edgecolor=color, label_size=6.2)
        ax.annotate("", xy=(0.778, y + 0.083), xytext=(0.760, y + 0.083), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=GRAY, mutation_scale=8))


def format_placeholder_plot(
    ax: plt.Axes,
    *,
    label: str,
    title: str,
    color: str,
    ylabel: str | None,
    xlabel: str,
    ylim: tuple[float, float],
) -> None:
    set_panel_title(ax, label, title, color=color)
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_xlim(-0.12, 6.25)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1.0, 3.1, 4.0, 5.0, 6.0])
    ax.set_xticklabels(["0", "90", "105", "120", "150", "175"])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.spines["bottom"].set_visible(False)
    trans = ax.get_xaxis_transform()
    x_left, x_right = ax.get_xlim()
    ax.plot([x_left, 0.27], [0.0, 0.0], transform=trans, color="black", lw=0.8, clip_on=False, zorder=10)
    ax.plot([0.82, x_right], [0.0, 0.0], transform=trans, color="black", lw=0.8, clip_on=False, zorder=10)
    for offset in (-0.040, 0.040):
        ax.plot(
            [0.545 + offset - 0.035, 0.545 + offset + 0.035],
            [-0.033, 0.033],
            transform=trans,
            color="black",
            lw=1.05,
            clip_on=False,
            solid_capstyle="butt",
            zorder=11,
        )
    ax.text(
        0.50,
        0.54,
        "data panel placeholder",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PLACEHOLDER_EDGE, alpha=0.96),
    )
    ax.grid(True, color=PALE_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_story_path_panel(
    ax: plt.Axes,
    panel_b: pd.DataFrame,
    *,
    label: str,
    title: str,
    sf_group: str,
    relation: str,
    color: str,
    ylabel: str | None,
    ylim: tuple[float, float],
) -> None:
    frame = pd.DataFrame()
    if not panel_b.empty:
        frame = panel_b[panel_b["sf_group"].eq(sf_group) & panel_b["relation"].eq(relation)].copy()

    if frame.empty:
        format_placeholder_plot(
            ax,
            label=label,
            title=title,
            color=color,
            ylabel=ylabel,
            xlabel="path length (arcmin)",
            ylim=ylim,
        )
        return

    story_panels._plot_b_series(ax, frame, color=color)
    story_panels._format_broken_axis(
        ax,
        ticks=story_panels.B_TICKS,
        min_pos=story_panels.B_MIN_POS,
        max_pos=story_panels.B_MAX_POS,
        xlabel="path length (arcmin)",
    )
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_ylim(*ylim)
    set_panel_title(ax, label, title, color=color)
    if ylabel:
        ax.set_ylabel(ylabel)
    add_support_note(ax, frame)
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)


def draw_panel_bcef_or_placeholder(
    ax: plt.Axes,
    panel_b: pd.DataFrame,
    *,
    label: str,
    title: str,
    sf_group: str,
    relation: str,
    color: str,
    ylabel: str | None,
    ylim: tuple[float, float],
) -> None:
    if panel_bcef_path_bins is not None and not panel_b.empty:
        try:
            panel_bcef_path_bins.draw_panel(
                ax,
                values=panel_b,
                label=label,
                title=title,
                sf_group=sf_group,
                relation=relation,
                color=color,
                ylabel=ylabel,
                ylim=ylim,
            )
            return
        except Exception:
            ax.clear()
    format_placeholder_plot(
        ax,
        label=label,
        title=title,
        color=color,
        ylabel=ylabel,
        xlabel="path length (arcmin)",
        ylim=ylim,
    )


def draw_story_component_panel(
    ax: plt.Axes,
    component: pd.DataFrame,
    *,
    label: str = "G",
    relation: str = "contour_matched",
    color: str = ORANGE,
    ylim: tuple[float, float] = (-32, 24),
) -> None:
    frame = pd.DataFrame()
    if not component.empty:
        frame = component[component["relation"].eq(relation)].copy()

    if frame.empty:
        format_placeholder_plot(
            ax,
            label=label,
            title="Aligned high-SF components",
            color=INK,
            ylabel="SSI change (%)",
            xlabel="component path (arcmin)",
            ylim=ylim,
        )
        ax.text(
            0.05,
            0.08,
            "reserve: across-contour and\nalong-contour series",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
            color=GRAY,
        )
        return

    story_panels._plot_component_series(ax, frame, color=color)
    story_panels._format_broken_axis(
        ax,
        ticks=story_panels.LOWER_TICKS,
        min_pos=story_panels.LOWER_MIN_POS,
        max_pos=story_panels.LOWER_MAX_POS,
        xlabel="component path (arcmin)",
    )
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_ylim(*ylim)
    set_panel_title(ax, label, "Aligned high-SF components", color=INK, fontsize=8.3)
    ax.set_ylabel("SSI change (%)")
    add_support_note(ax, frame)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        short_labels = ["across" if "across" in item else "along" for item in labels]
        ax.legend(
            handles,
            short_labels,
            frameon=False,
            fontsize=6.1,
            loc="lower left",
            handlelength=1.8,
            borderaxespad=0.2,
        )
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)


def draw_polar_placeholder(ax: plt.Axes) -> None:
    set_panel_title(ax, "H", "FEM paths align with local contours", fontsize=9.0, pad=10)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks([0, 1.5708, 3.1416, 4.7124])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel", "orthogonal"], fontsize=7)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["1", "2"], fontsize=7)
    ax.set_ylim(0.0, 1.12)
    ax.grid(True, color="#D8DEE5", lw=0.8)
    ax.spines["polar"].set_linewidth(1.1)
    ax.text(
        0.5,
        0.5,
        "polar distribution\nplaceholder",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PLACEHOLDER_EDGE, alpha=0.96),
    )


def draw_path_alignment_placeholder(ax: plt.Axes, *, error: Exception | None = None) -> None:
    set_panel_title(ax, "H", "FEM spread follows local contours", fontsize=9.0, pad=4)
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(1.65, 2.55)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
    ax.set_xlabel("angle from local edge")
    ax.set_ylabel("position spread RMS (arcmin)")
    message = "unwrapped profile\nplaceholder"
    if error is not None:
        message = "panel generation\nfailed; placeholder"
    ax.text(
        0.50,
        0.54,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PLACEHOLDER_EDGE, alpha=0.96),
    )
    ax.grid(True, color=PALE_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_edge_alignment_placeholder(ax: plt.Axes, *, error: Exception | None = None) -> None:
    set_panel_title(ax, "I", "Edge alignment", fontsize=9.0, pad=4)
    ax.axhline(0.0, color="0.35", lw=0.85)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 0.38)
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel("edge-following alignment")
    message = "coherence-binned\nresult placeholder"
    if error is not None:
        message = "panel generation\nfailed; placeholder"
    ax.text(
        0.50,
        0.54,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=PLACEHOLDER_EDGE, alpha=0.96),
    )
    ax.grid(True, color=PALE_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_generated_panel_or_placeholder(
    ax: plt.Axes,
    module: object | None,
    fallback,
) -> None:
    if module is None:
        fallback(ax, error=PANEL_IMPORT_ERROR)
        return
    try:
        module.draw_panel(ax)
    except Exception as exc:
        ax.clear()
        fallback(ax, error=exc)


def draw_panel_g_or_fallback(
    ax: plt.Axes,
    component_values: pd.DataFrame,
    *,
    component_ylim: tuple[float, float],
) -> None:
    if panel_g_matched_bins_bracket is not None:
        try:
            panel_g_matched_bins_bracket.draw_panel(ax)
            return
        except Exception:
            ax.clear()
    draw_story_component_panel(ax, pd.DataFrame(), ylim=component_ylim)


def draw_overall_title(fig: plt.Figure, *, has_existing_panels: bool) -> None:
    fig.text(
        0.50,
        0.986,
        "Single-spike information and contour-relative FEM",
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
    )
    subtitle = (
        "Draft composition; B/C/E/F/G/H/I and schematic image/map assets reuse existing BackImage/Figure 4 data"
        if has_existing_panels
        else "Draft composition scaffold; quantitative/result panels are placeholders"
    )
    fig.text(
        0.50,
        0.967,
        subtitle,
        ha="center",
        va="top",
        fontsize=8.5,
        color=GRAY,
    )


def build_figure(out_dir: Path) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_b_values, component_values = read_existing_story_values()
    schematic_payload = read_schematic_payload()
    has_existing_panels = not panel_b_values.empty and panel_g_matched_bins_bracket is not None
    if panel_bcef_path_bins is not None and not panel_b_values.empty:
        bc_ylim = panel_bcef_path_bins.shared_ylim_for(panel_b_values, relations=("strong_contours_no_osi",))
        ef_ylim = panel_bcef_path_bins.shared_ylim_for(panel_b_values, relations=("contour_matched",))
    else:
        bc_ylim = shared_story_ylim(panel_b_values, fallback=(-20, 48))
        ef_ylim = bc_ylim
    aligned_component = (
        component_values[component_values["relation"].eq("contour_matched")].copy()
        if not component_values.empty
        else pd.DataFrame()
    )
    component_ylim = shared_story_ylim(aligned_component, fallback=(-32, 24))

    fig = plt.figure(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    draw_overall_title(fig, has_existing_panels=has_existing_panels)
    gs = fig.add_gridspec(3, 3, **MAIN_GRID_KWARGS)

    ax_b = fig.add_subplot(gs[0, :2])
    draw_panel_b(ax_b, schematic_payload=schematic_payload)

    gs_b_right = gs[0, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    ax_b1 = fig.add_subplot(gs_b_right[0, 0])
    draw_panel_bcef_or_placeholder(
        ax_b1,
        panel_b_values,
        label="B",
        title="Low-SF units",
        sf_group="low_lt0p5",
        relation="strong_contours_no_osi",
        color=BLUE,
        ylabel="SSI change (%)",
        ylim=bc_ylim,
    )
    ax_b2 = fig.add_subplot(gs_b_right[1, 0])
    draw_panel_bcef_or_placeholder(
        ax_b2,
        panel_b_values,
        label="C",
        title="High-SF units",
        sf_group="high_ge0p75",
        relation="strong_contours_no_osi",
        color=ORANGE,
        ylabel=None,
        ylim=bc_ylim,
    )

    ax_a = fig.add_subplot(gs[1, :2])
    draw_panel_a(ax_a, schematic_payload=schematic_payload)

    gs_a_right = gs[1, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    ax_a1 = fig.add_subplot(gs_a_right[0, 0])
    draw_panel_bcef_or_placeholder(
        ax_a1,
        panel_b_values,
        label="E",
        title="Low-SF aligned",
        sf_group="low_lt0p5",
        relation="contour_matched",
        color=BLUE,
        ylabel=None,
        ylim=ef_ylim,
    )
    ax_a2 = fig.add_subplot(gs_a_right[1, 0])
    draw_panel_bcef_or_placeholder(
        ax_a2,
        panel_b_values,
        label="F",
        title="High-SF aligned",
        sf_group="high_ge0p75",
        relation="contour_matched",
        color=ORANGE,
        ylabel=None,
        ylim=ef_ylim,
    )

    ax_c1 = fig.add_subplot(gs[2, 0])
    draw_panel_g_or_fallback(ax_c1, component_values, component_ylim=component_ylim)
    ax_c2 = fig.add_subplot(gs[2, 1])
    draw_generated_panel_or_placeholder(ax_c2, panel_h_unwrapped_edge_coherence, draw_path_alignment_placeholder)
    ax_c3 = fig.add_subplot(gs[2, 2])
    draw_generated_panel_or_placeholder(ax_c3, panel_i_edge_alignment, draw_edge_alignment_placeholder)

    paths = {
        "png": out_dir / "ssi_figure_v2.png",
        "pdf": out_dir / "ssi_figure_v2.pdf",
        "svg": out_dir / "ssi_figure_v2.svg",
    }
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
    paths["panel_boxes_svg"] = write_panel_boxes_svg(out_dir)
    provenance_path = out_dir / "ssi_figure_v2_methods_provenance.json"
    provenance_path.write_text(
        json.dumps(collect_methods_provenance(out_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["methods_provenance_json"] = provenance_path
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the draft SSI figure v2 scaffold.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory for rendered figure files.")
    args = parser.parse_args()
    paths = build_figure(args.out_dir)
    for key in ("png", "pdf", "svg", "panel_boxes_svg", "methods_provenance_json"):
        print(paths[key])


if __name__ == "__main__":
    main()
