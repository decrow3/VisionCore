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
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    from panels import panel_g_rms_excursion

    PANEL_G_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_g_rms_excursion = None
    PANEL_G_IMPORT_ERROR = exc

try:  # noqa: E402
    from panels import panel_h_unwrapped_edge_coherence
    from panels import panel_j_match_advantage as panel_i_match_advantage

    PANEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_h_unwrapped_edge_coherence = None
    panel_i_match_advantage = None
    PANEL_IMPORT_ERROR = exc

try:  # noqa: E402
    from panels import panel_d_coherence_gallery

    PANEL_D_GALLERY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    panel_d_coherence_gallery = None
    PANEL_D_GALLERY_IMPORT_ERROR = exc

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
# Eye-trace color in D/G's crop images -- deliberately not BLUE, which means
# "low-SF population" everywhere else in this figure (B/C/E/F/I/J); a reader
# who's learned that convention shouldn't also read this trace as low-SF.
TRACE_COLOR = "#6A3D9A"
GRAY = "#6B6F75"
INK = "#111111"
PALE_GRID = "#E7E7E7"
PLACEHOLDER_FILL = "#F6F7F8"
PLACEHOLDER_EDGE = "#B9BFC6"
CONTOUR_WINDOW = "#E6A700"
ZOOM_BOX = "#E8118A"  # vivid magenta -- center-zoom marker box, connectors, and the crop border all use this
CROP_BORDER_LW = 2.6
CENTER_ZOOM_HALF_DEG = 0.25
CENTER_ZOOM_TRACE_PAD_DEG = 0.04
D_IMAGE_LABEL_FS = 5.8
D_TRACE_LABEL_FS = 6.2
D_SUBHEAD_FS = 7.0
D_CONNECTOR_LW = 0.65
# E/F now live as insets inside D's own axes (data coords, D's xlim/ylim are
# (0, 1)) rather than as separate gridspec cells -- this is the region that
# used to hold D's unfinished "Contour-carried signal" placeholder, which
# moved out to its own panel (see draw_contour_components_panel).
# Heights/widths at scale 1.0 make E/F's physical footprint (inside D's
# axes) match B/C's exactly -- 0.355/0.4454 of D's data-x/y range ==
# B/C's real gridspec-cell width/height (1.955in / 1.337in), computed
# directly via gridspec instantiation. AXES_SHRINK (see B/C below, applied
# via _shrink_axes_center there) scales both down by the same ~10% here,
# re-centered on the same midpoint each occupied at scale 1.0, to keep B/C
# and E/F matched at the smaller size too.
_EF_FULL_X, _EF_FULL_W = 0.620, 0.355
_EF_FULL_F_Y, _EF_FULL_F_H = 0.020, 0.4454
_EF_FULL_E_Y, _EF_FULL_E_H = 0.660, 0.4454
AXES_SHRINK = 0.90
EF_INSET_W = _EF_FULL_W * AXES_SHRINK
EF_INSET_X = _EF_FULL_X + (_EF_FULL_W - EF_INSET_W) / 2
EF_INSET_F_H = _EF_FULL_F_H * AXES_SHRINK
EF_INSET_F_Y = _EF_FULL_F_Y + (_EF_FULL_F_H - EF_INSET_F_H) / 2
EF_INSET_E_H = _EF_FULL_E_H * AXES_SHRINK
EF_INSET_E_Y = _EF_FULL_E_Y + (_EF_FULL_E_H - EF_INSET_E_H) / 2
# The E/F gap (E_Y - (F_Y + F_H)) has to clear both F's title (~0.047 above
# its own axes box) and E's x-tick labels/xlabel (~0.144 below its own axes
# box) or the two collide -- measured via get_tightbbox, not visually
# obvious from the nominal axes boxes alone; shrinking only widens that
# clearance. D's row has ~0.18 of slack above y=1 and below y=0 (the
# gridspec hspace to the neighboring rows) to park overflow in.
FIGURE_SIZE_IN = (8.5, 11.0)
MAIN_GRID_KWARGS = {
    "left": 0.060,
    "right": 0.982,
    "top": 0.930,
    "bottom": 0.045,
    "width_ratios": [1.18, 1.18, 0.90],
    "height_ratios": [1.24, 1.16, 0.94],
    "hspace": 0.190,
    "wspace": 0.160,
}
RIGHT_PANEL_HSPACE = 0.400
# A and D are self-drawn (axis off) and don't need MAIN_GRID_KWARGS['left'] --
# that margin exists for H's automatic y-tick labels/ylabel, which A/D don't
# have. Give A/D their own tighter left edge instead of the shared gridspec
# column boundary; see _wide_panel_axes.
WIDE_PANEL_LEFT = 0.016


def _wide_panel_axes(fig: plt.Figure, gs, row: int) -> plt.Axes:
    """Add the row-spanning A/D axes shifted flush to WIDE_PANEL_LEFT instead
    of MAIN_GRID_KWARGS['left'], as a rigid translation (same width/height as
    gs[row, :2] would give). A rigid shift, rather than stretching the left
    edge out to WIDE_PANEL_LEFT while keeping the right edge fixed, keeps
    every width-derived constant tuned elsewhere in D (EF_INSET_W matching
    B/C's width, crop/zoom sizing, etc.) numerically exact -- it just opens a
    correspondingly small gap between A/D and the column to their right
    (B/C, G) where MAIN_GRID_KWARGS['left'] used to be closed up."""
    pos = gs[row, :2].get_position(fig)
    return fig.add_axes([WIDE_PANEL_LEFT, pos.y0, pos.width, pos.height])


def _shrink_axes_center(ax: plt.Axes, factor: float = AXES_SHRINK) -> None:
    """Scale an axes' box down by ``factor`` around its own center, keeping
    its aspect ratio (both dimensions scaled equally). Used for B/C so they
    stay matched to E/F, which are shrunk the same way via EF_INSET_*."""
    pos = ax.get_position()
    cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
    new_w, new_h = pos.width * factor, pos.height * factor
    ax.set_position([cx - new_w / 2, cy - new_h / 2, new_w, new_h])
PANEL_BOX_LABELS = {
    "A": "Motion schematic",
    "B": "Low-SF units",
    "C": "High-SF units",
    "D": "Contour-relative stimulus",
    "E": "Low-SF aligned",
    "F": "High-SF aligned",
    "G": "Local contour detail (crop ref. + zoom, from D)",
    "H": "Aligned high-SF RMS excursion",
    "I": "Position spread",
    "J": "Trace-contour match advantage",
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
    panels["D"] = {
        "source_script": _relative(ROOT / "declan" / "fig" / "ssi_figure_v2" / "generate_ssi_figure_v2.py"),
        "standalone_script": _relative(
            ROOT / "declan" / "fig" / "ssi_figure_v2" / "panels" / "panel_d_contour_relative_stimulus.py"
        ),
        "contour_window_radius_source": "stimulus_row.image_patch_radius_px",
        "contour_window_radius_interpretation": (
            "BackImage reviewed windows store image_patch_radius_px=38 px; with MODEL_PPD=37.50476617 "
            "this is 1.01 deg, drawn as a 1 deg local contour-axis/coherence window."
        ),
        "center_zoom_rule": (
            f"minimum +/-{CENTER_ZOOM_HALF_DEG:g} deg, expanded by "
            f"{CENTER_ZOOM_TRACE_PAD_DEG:g} deg beyond the selected trace extent when needed"
        ),
        "coherence_gallery": {
            "status": "real image crops, one per COHERENCE_ORDER bin",
            "source_script": _relative(
                ROOT / "declan" / "fig" / "ssi_figure_v2" / "panels" / "panel_d_coherence_gallery.py"
            ),
            "cache_builder_script": _relative(
                ROOT / "declan" / "fig" / "ssi_figure_v2" / "panels" / "build_coherence_gallery_cache.py"
            ),
            "cache_provenance_json": _relative(
                ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "cache" / "coherence_gallery_provenance.json"
            ),
            "note": (
                "Replaces D's former short/long-path trace legend; the real traces moved to G, where "
                "there's room to show them at a legible scale."
            ),
        },
    }
    if panel_bcef_path_bins is not None:
        try:
            panels["B_C_E_F"] = panel_bcef_path_bins.load_provenance()
        except Exception as exc:
            panels["B_C_E_F"] = {"status": "provenance load failed", "error": repr(exc)}
    else:
        panels["B_C_E_F"] = {"status": "panel import failed", "error": repr(PANEL_BCEF_IMPORT_ERROR)}

    panels["G"] = {
        "status": "real image assets (crop + zoom) and a schematic decomposition, no population-level result",
        "source_script": _relative(ROOT / "declan" / "fig" / "ssi_figure_v2" / "generate_ssi_figure_v2.py"),
        "note": (
            "Reference copy of D's 151x151 crop plus the zoomed local-contour aperture that used to be "
            "drawn as D's third cascaded image; moved here so it isn't squeezed for width against D's "
            "E/F insets. Below that, this same example trace is decomposed into across-/along-contour "
            "components (draw_rms_excursion_explainer) -- the same split H's dose curve reports as "
            "separate lines, using this one trace's real geometry, not H's population statistic."
        ),
    }

    if panel_g_rms_excursion is not None:
        try:
            panels["H"] = panel_g_rms_excursion.load_provenance()
        except Exception as exc:
            panels["H"] = {"status": "provenance load failed", "error": repr(exc)}
    else:
        panels["H"] = {"status": "panel import failed", "error": repr(PANEL_G_IMPORT_ERROR)}

    panels["I"] = {
        "source_script": _relative(ROOT / "declan" / "fig" / "ssi_figure_v2" / "panels" / "panel_h_unwrapped_edge_coherence.py"),
        "source_profile_csv": _relative(panel_h_unwrapped_edge_coherence.PROFILE_CSV)
        if panel_h_unwrapped_edge_coherence is not None
        else None,
        "source_random_orientation_baseline_csv": _relative(panel_h_unwrapped_edge_coherence.BASELINE_CSV)
        if panel_h_unwrapped_edge_coherence is not None
        else None,
        "metric_interpretation": "Position-spread RMS profile from covariance/position cloud, not unsigned path length.",
    }
    panels["J"] = {
        "source_script": _relative(ROOT / "declan" / "fig" / "ssi_figure_v2" / "panels" / "panel_j_match_advantage.py"),
        "source_coherence_summary_csv": _relative(panel_i_match_advantage.COHERENCE_SUMMARY_CSV)
        if panel_i_match_advantage is not None
        else None,
        "metric_interpretation": (
            "Observed-minus-random-rotation model SSI prediction (RMS excursion axis) by local edge "
            "coherence; replaced the descriptive drift-cloud/edge-alignment panel "
            "(panels/panel_i_edge_alignment.py, kept unwired for reference) once the random-rotation "
            "null showed the coherence-dependent correlation is also model-beneficial relative to chance."
        ),
    }
    return provenance


def _svg_number(value: float) -> str:
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def _make_layout_axes(fig: plt.Figure) -> dict[str, plt.Axes]:
    gs = fig.add_gridspec(3, 3, **MAIN_GRID_KWARGS)
    axes: dict[str, plt.Axes] = {}
    axes["A"] = _wide_panel_axes(fig, gs, 0)

    gs_b_right = gs[0, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    axes["B"] = fig.add_subplot(gs_b_right[0, 0])
    _shrink_axes_center(axes["B"])
    axes["C"] = fig.add_subplot(gs_b_right[1, 0])
    _shrink_axes_center(axes["C"])

    axes["D"] = _wide_panel_axes(fig, gs, 1)
    axes["D"].set_xlim(0.0, 1.0)
    axes["D"].set_ylim(0.0, 1.0)
    axes["E"] = axes["D"].inset_axes(
        [EF_INSET_X, EF_INSET_E_Y, EF_INSET_W, EF_INSET_E_H], transform=axes["D"].transData
    )
    axes["F"] = axes["D"].inset_axes(
        [EF_INSET_X, EF_INSET_F_Y, EF_INSET_W, EF_INSET_F_H], transform=axes["D"].transData
    )
    axes["G"] = fig.add_subplot(gs[1, 2])

    axes["H"] = fig.add_subplot(gs[2, 0])
    axes["I"] = fig.add_subplot(gs[2, 1])
    axes["J"] = fig.add_subplot(gs[2, 2])
    return axes


def _figure_bbox_to_svg_box(bbox, *, page_w: float, page_h: float) -> dict[str, float]:
    return {
        "x": float(bbox.x0 * page_w),
        "y": float((1.0 - bbox.y1) * page_h),
        "width": float(bbox.width * page_w),
        "height": float(bbox.height * page_h),
    }


def _axis_position_box(ax: plt.Axes, *, page_w: float, page_h: float) -> dict[str, float]:
    return _figure_bbox_to_svg_box(ax.get_position(), page_w=page_w, page_h=page_h)


def _set_axis_text_extent_style(ax: plt.Axes, label: str) -> None:
    """Add representative axis text, excluding panel titles/headings."""
    if label in {"A", "D", "G"}:
        ax.set_axis_off()
        return
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    if label in {"B", "C", "E", "F"}:
        ax.set_xlim(-0.12, 6.10)
        ax.set_xticks([0.0, 1.16, 2.26, 3.21, 4.80, 5.90])
        ax.set_xticklabels(["0", "90", "105", "120", "150", "175"])
        ax.set_yticks([-20.0, 0.0, 20.0, 40.0])
        ax.set_xlabel("path length (arcmin)")
        if label == "B":
            ax.set_ylabel("SSI change (%)")
    elif label == "H":
        ax.set_xlim(-0.12, 5.73)
        ax.set_xticks([0.0, 0.91, 3.63, 5.23])
        ax.set_xticklabels(["0", "1", "2", "3"])
        ax.set_yticks([-15.0, 0.0, 15.0])
        ax.set_xlabel("component RMS excursion (arcmin)")
        ax.set_ylabel("SSI change (%)")
    elif label == "I":
        ax.set_xlim(0.0, 180.0)
        ax.set_xticks([0.0, 90.0, 180.0])
        ax.set_xticklabels(["parallel", "orthogonal", "parallel"])
        ax.set_yticks([1.7, 1.9, 2.1])
        ax.set_xlabel("angle from local edge")
        ax.set_ylabel("position spread RMS (arcmin)")
    elif label == "J":
        ax.set_xlim(-0.12, 3.12)
        ax.set_xticks([0.0, 1.0, 2.0, 3.0])
        ax.set_xticklabels(["0-.2", ".2-.5", ".5-.8", ".8-1"])
        ax.set_yticks([-0.2, 0.0, 0.2])
        ax.set_xlabel("local edge coherence")
        ax.set_ylabel("observed - random rotated (pp SSI)")


def compute_panel_box_layout() -> dict[str, dict[str, dict[str, float]]]:
    """Return current panel axes and axis-text boxes in SVG point coordinates."""
    page_w = FIGURE_SIZE_IN[0] * 72.0
    page_h = FIGURE_SIZE_IN[1] * 72.0
    fig = plt.figure(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    axes = _make_layout_axes(fig)
    for label, ax in axes.items():
        _set_axis_text_extent_style(ax, label)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axes_boxes: dict[str, dict[str, float]] = {}
    axis_text_boxes: dict[str, dict[str, float]] = {}
    for label, ax in axes.items():
        axes_boxes[label] = _axis_position_box(ax, page_w=page_w, page_h=page_h)
        if label in {"A", "D"}:
            axis_text_boxes[label] = axes_boxes[label]
        else:
            bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            axis_text_boxes[label] = _figure_bbox_to_svg_box(bbox, page_w=page_w, page_h=page_h)
    plt.close(fig)
    return {"axes": axes_boxes, "axis_text": axis_text_boxes}


def write_panel_boxes_svg(out_dir: Path = OUT_DIR) -> Path:
    """Write an editable empty-box SVG matching the current panel layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_w = FIGURE_SIZE_IN[0] * 72.0
    page_h = FIGURE_SIZE_IN[1] * 72.0
    boxes = compute_panel_box_layout()
    axes_boxes = boxes["axes"]
    axis_text_boxes = boxes["axis_text"]
    path = out_dir / "ssi_figure_v2_panel_boxes.svg"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_svg_number(page_w)}pt" '
            f'height="{_svg_number(page_h)}pt" viewBox="0 0 {_svg_number(page_w)} {_svg_number(page_h)}">'
        ),
        '  <title>SSI figure v2 editable panel bounding boxes</title>',
        (
            "  <desc>Blue boxes are raw panel axes; orange boxes include tick labels and axis labels, "
            "but exclude panel headings.</desc>"
        ),
        '  <rect id="page" x="0" y="0" width="100%" height="100%" fill="white" stroke="#d0d4d9" stroke-width="0.75"/>',
        '  <g id="panel-axis-text-boxes" fill="none" stroke="#D55E00" stroke-width="1.0" stroke-dasharray="2.5 2.5">',
    ]
    for label in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        box = axis_text_boxes[label]
        lines.append(f'    <g id="panel-{label}-axis-text" data-panel="{label}" data-box-type="axis-text">')
        lines.append(
            '      <rect '
            f'id="panel-{label}-axis-text-box" '
            f'x="{_svg_number(box["x"])}" '
            f'y="{_svg_number(box["y"])}" '
            f'width="{_svg_number(box["width"])}" '
            f'height="{_svg_number(box["height"])}" />'
        )
        lines.append("    </g>")
    lines.extend(
        [
            "  </g>",
            '  <g id="panel-axes-boxes" fill="none" stroke="#0072B2" stroke-width="1.25" stroke-dasharray="5 3">',
        ]
    )
    for label in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        box = axes_boxes[label]
        lines.append(f'    <g id="panel-{label}" data-panel="{label}" data-box-type="axes">')
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


def _finite_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def contour_window_metadata(payload: dict | None) -> dict[str, float | str]:
    """Return display metadata for the local contour-axis/coherence aperture."""
    default_ppd = 37.50476617
    if ssi_schematic is not None:
        default_ppd = _finite_float(getattr(ssi_schematic, "MODEL_PPD", default_ppd), default_ppd)

    row = {}
    if isinstance(payload, dict) and isinstance(payload.get("stimulus_row"), dict):
        row = payload["stimulus_row"]

    radius_px = _finite_float(row.get("image_patch_radius_px"), default_ppd)
    crop_size_px = _finite_float((payload or {}).get("stimulus_crop_size_px"), 151.0)
    if crop_size_px <= 0:
        crop_size_px = 151.0
    ppd = _finite_float(row.get("ppd"), default_ppd)
    if ppd <= 0:
        ppd = default_ppd
    radius_deg = radius_px / ppd
    zoom_half_px = CENTER_ZOOM_HALF_DEG * ppd
    axis_image_deg = _finite_float((payload or {}).get("contour_axis_image_deg"), 10.352312)
    coherence = _finite_float(row.get("image_orientation_coherence"), float("nan"))
    radius_label = "1 deg radius" if 0.93 <= radius_deg <= 1.08 else f"{radius_deg:.2f} deg radius"
    return {
        "radius_px": radius_px,
        "radius_deg": radius_deg,
        "radius_fraction": min(0.48, max(0.08, radius_px / crop_size_px)),
        "crop_size_px": crop_size_px,
        "ppd": ppd,
        "center_zoom_half_deg": CENTER_ZOOM_HALF_DEG,
        "center_zoom_half_px": zoom_half_px,
        "center_zoom_fraction": min(0.48, max(0.03, zoom_half_px / crop_size_px)),
        "center_zoom_mode": "minimum",
        "axis_image_deg": axis_image_deg,
        "coherence": coherence,
        "radius_label": radius_label,
    }


def trace_fit_center_zoom_metadata(
    metadata: dict[str, float | str],
    motion_eye: object | None,
) -> dict[str, float | str]:
    """Expand the center zoom just enough to contain the displayed traces."""
    if not isinstance(motion_eye, dict):
        return metadata

    trace_extents: list[float] = []
    for key in ("small_xy_px", "large_xy_px"):
        try:
            trace = np.asarray(motion_eye.get(key), dtype=np.float64)
        except Exception:
            continue
        if trace.ndim != 2 or trace.shape[1] != 2 or trace.size == 0:
            continue
        finite = trace[np.isfinite(trace)]
        if finite.size:
            trace_extents.append(float(np.nanmax(np.abs(finite))))
    if not trace_extents:
        return metadata

    updated = dict(metadata)
    ppd = _finite_float(updated.get("ppd"), 37.50476617)
    crop_size_px = _finite_float(updated.get("crop_size_px"), 151.0)
    min_half_px = CENTER_ZOOM_HALF_DEG * ppd
    trace_max_px = max(trace_extents)
    half_px = max(min_half_px, trace_max_px + CENTER_ZOOM_TRACE_PAD_DEG * ppd)
    updated["center_zoom_half_px"] = half_px
    updated["center_zoom_half_deg"] = half_px / ppd if ppd > 0 else CENTER_ZOOM_HALF_DEG
    updated["center_zoom_fraction"] = min(0.48, max(0.03, half_px / crop_size_px))
    updated["center_zoom_trace_max_px"] = trace_max_px
    updated["center_zoom_trace_max_deg"] = trace_max_px / ppd if ppd > 0 else float("nan")
    updated["center_zoom_mode"] = "trace_fit" if half_px > min_half_px else "minimum"
    return updated


def center_zoom_label(metadata: dict[str, float | str]) -> str:
    half_deg = _finite_float(metadata.get("center_zoom_half_deg"), CENTER_ZOOM_HALF_DEG)
    if abs(half_deg - CENTER_ZOOM_HALF_DEG) < 0.005:
        return f"central +/-{CENTER_ZOOM_HALF_DEG:g} deg"
    return f"central +/-{half_deg:.2f} deg"


def data_width_for_physical_aspect(ax: plt.Axes, height: float, width_over_height: float) -> float:
    """Convert a desired physical aspect into parent-axis data coordinates."""
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    axis_w = float(bbox.width) * float(fig_w)
    axis_h = float(bbox.height) * float(fig_h)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = abs(float(x1) - float(x0))
    yspan = abs(float(y1) - float(y0))
    if axis_w <= 0 or axis_h <= 0 or xspan <= 0 or yspan <= 0:
        return float(height) * float(width_over_height)
    return float(width_over_height) * float(height) * (axis_h / axis_w) * (xspan / yspan)


def stimulus_canvas_aspect(payload: dict | None) -> float:
    canvas = (payload or {}).get("stimulus_canvas")
    try:
        height, width = canvas.shape[:2]
        if float(height) > 0:
            return float(width) / float(height)
    except Exception:
        pass
    return 16.0 / 9.0


def _axis_vector_image(axis_image_deg: float) -> tuple[float, float]:
    if ssi_schematic is not None and hasattr(ssi_schematic, "axis_vector_image"):
        try:
            vec = ssi_schematic.axis_vector_image(float(axis_image_deg))
            return float(vec[0]), float(vec[1])
        except Exception:
            pass
    theta = math.radians(float(axis_image_deg))
    return math.cos(theta), math.sin(theta)


def add_contour_window_to_crop_axis(crop_ax: plt.Axes, metadata: dict[str, float | str]) -> None:
    """Draw the real 1-deg contour-analysis aperture in crop pixel coordinates."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    radius = _finite_float(metadata.get("radius_px"), 38.0)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    norm = math.hypot(dx, dy)
    if norm <= 0:
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx / norm, dy / norm

    crop_ax.add_patch(
        patches.Circle(
            (center, center),
            radius,
            fill=False,
            edgecolor=CONTOUR_WINDOW,
            linewidth=1.0,
            linestyle=(0, (3.0, 2.2)),
            zorder=8,
        )
    )
    local_half = radius * 0.84
    crop_ax.plot(
        [center - dx * local_half, center + dx * local_half],
        [center - dy * local_half, center + dy * local_half],
        color=CYAN,
        lw=1.55,
        alpha=0.95,
        solid_capstyle="round",
        zorder=9,
    )
    crop_ax.plot(
        [center, center + radius],
        [center, center],
        color=CONTOUR_WINDOW,
        lw=0.72,
        solid_capstyle="round",
        zorder=10,
    )
    crop_ax.scatter(
        [center],
        [center],
        s=18,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.65,
        zorder=11,
    )


def restore_trace_orientation(motion_eye: dict | None) -> dict | None:
    """Undo the extra display rotation baked into the "large"/long-path
    trace by make_ssi_contour_schematic.py's selected_real_panel_a_trace_pair
    (PANEL_A_LARGE_TRACE_ROTATION_DEG = 90 deg, PANEL_A_SMALL_TRACE_ROTATION_DEG
    = 0 deg) for that module's own Panel A schematic. D/G want the traces at
    their real recorded orientation, not that rotation, so rotate back by the
    same angle in reverse. Only applies when the real trace bank was used
    (large_trace_index >= 0); the synthetic fallback trace was never rotated.
    """
    if not isinstance(motion_eye, dict) or ssi_schematic is None:
        return motion_eye
    if motion_eye.get("large_trace_index", -1) is None or motion_eye.get("large_trace_index", -1) < 0:
        return motion_eye
    rotate = getattr(ssi_schematic, "rotate_trace_xy_px", None)
    if rotate is None:
        return motion_eye
    restored = dict(motion_eye)
    large_rotation_deg = getattr(ssi_schematic, "PANEL_A_LARGE_TRACE_ROTATION_DEG", 0.0)
    if large_rotation_deg and "large_xy_px" in restored:
        restored["large_xy_px"] = rotate(restored["large_xy_px"], -large_rotation_deg)
    small_rotation_deg = getattr(ssi_schematic, "PANEL_A_SMALL_TRACE_ROTATION_DEG", 0.0)
    if small_rotation_deg and "small_xy_px" in restored:
        restored["small_xy_px"] = rotate(restored["small_xy_px"], -small_rotation_deg)
    return restored


def add_center_zoom_box_to_crop_axis(crop_ax: plt.Axes, metadata: dict[str, float | str]) -> None:
    """Mark the tight central zoom window on the 151 px crop."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    crop_ax.add_patch(
        patches.Rectangle(
            (center - half_px, center - half_px),
            2.0 * half_px,
            2.0 * half_px,
            fill=False,
            edgecolor=ZOOM_BOX,
            linewidth=0.85,
            zorder=12,
        )
    )


def draw_plain_crop(
    crop_ax: plt.Axes,
    patch: object,
    *,
    trace_xy_px: object | None = None,
    trace_color: str | None = None,
) -> int:
    """Render a crop image directly instead of via ssi_schematic.add_stimulus.

    add_stimulus always draws a red+blue trace pair (or, with motion_eye=None,
    a red/blue placeholder double-arrow) with colors hardcoded to its own
    module -- there's no way to show only one trace, or to recolor it, through
    that function. This replicates just its image/border rendering, then
    optionally draws a single trace in a caller-chosen color via
    ssi_schematic.add_panel_a_trace_path directly. Returns the crop's pixel
    size (patches are square).
    """
    image = ssi_schematic.normalize_image(patch)
    n = int(image.shape[0])
    crop_ax.imshow(image, cmap="gray", interpolation="bicubic")
    crop_ax.set_aspect("equal", adjustable="box")
    crop_ax.set_axis_off()
    crop_ax.set_xlim(0, n - 1)
    crop_ax.set_ylim(n - 1, 0)
    crop_ax.add_patch(patches.Rectangle((0, 0), n - 1, n - 1, fill=False, lw=1.0, ec=INK))
    if trace_xy_px is not None and trace_color is not None:
        axis_center = np.array([0.5 * (n - 1), 0.5 * (n - 1)], dtype=np.float64)
        ssi_schematic.add_panel_a_trace_path(crop_ax, axis_center, trace_xy_px, trace_color, lw=1.85, zorder=4)
    return n


def add_zoomed_crop_view(
    zoom_ax: plt.Axes,
    schematic_payload: dict,
    motion_eye: dict | None,
    metadata: dict[str, float | str],
    *,
    trace_color: str = TRACE_COLOR,
) -> None:
    """Draw the crop again, zoomed to the central +/-0.25 deg window."""
    if ssi_schematic is None:
        raise RuntimeError("SSI schematic helpers are unavailable")
    trace_xy_px = motion_eye.get("large_xy_px") if isinstance(motion_eye, dict) else None
    draw_plain_crop(zoom_ax, schematic_payload.get("patch"), trace_xy_px=trace_xy_px, trace_color=trace_color)
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    zoom_ax.set_xlim(center - half_px, center + half_px)
    zoom_ax.set_ylim(center + half_px, center - half_px)
    zoom_ax.add_patch(
        patches.Rectangle(
            (center - half_px, center - half_px),
            2.0 * half_px,
            2.0 * half_px,
            fill=False,
            edgecolor=INK,
            linewidth=0.85,
            zorder=20,
        )
    )
    zoom_ax.scatter([center], [center], s=22, facecolor="white", edgecolor=INK, linewidth=0.65, zorder=21)


def add_center_zoom_parent_overlay(
    ax: plt.Axes,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    metadata: dict[str, float | str],
) -> None:
    """Fallback/connector geometry for the tight central zoom window."""
    center_x = crop_x + 0.5 * crop_w
    center_y = crop_y + 0.5 * crop_h
    half_fraction = _finite_float(metadata.get("center_zoom_fraction"), CENTER_ZOOM_HALF_DEG * 37.50476617 / 151.0)
    ax.add_patch(
        patches.Rectangle(
            (center_x - crop_w * half_fraction, center_y - crop_h * half_fraction),
            2.0 * crop_w * half_fraction,
            2.0 * crop_h * half_fraction,
            fill=False,
            edgecolor=ZOOM_BOX,
            linewidth=0.75,
            zorder=24,
        )
    )


def add_zoom_connectors(
    ax: plt.Axes,
    source_box: tuple[float, float, float, float],
    target_box: tuple[float, float, float, float],
    *,
    color: str = CYAN,
) -> None:
    """Connect the visible source zoom box to the next crop stage."""
    source_x, source_y, source_w, source_h = source_box
    target_x, target_y, _target_w, target_h = target_box
    ax.plot(
        [source_x + source_w, target_x],
        [source_y + source_h, target_y + target_h],
        color=color,
        lw=D_CONNECTOR_LW,
        ls=(0, (3, 3)),
        alpha=0.58,
        zorder=6,
    )
    ax.plot(
        [source_x + source_w, target_x],
        [source_y, target_y],
        color=color,
        lw=D_CONNECTOR_LW,
        ls=(0, (3, 3)),
        alpha=0.58,
        zorder=6,
    )


def add_contour_window_parent_overlay(
    ax: plt.Axes,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    metadata: dict[str, float | str],
) -> None:
    """Fallback aperture overlay when the real crop inset is unavailable."""
    center_x = crop_x + 0.5 * crop_w
    center_y = crop_y + 0.5 * crop_h
    radius_fraction = _finite_float(metadata.get("radius_fraction"), 38.0 / 151.0)
    radius_x = crop_w * radius_fraction
    radius_y = crop_h * radius_fraction
    theta = math.radians(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    dx = math.cos(theta)
    dy = -math.sin(theta)
    line_scale = min(0.46 / max(abs(dx), 1e-6), 0.46 / max(abs(dy), 1e-6))
    local_scale = radius_fraction * 0.84

    ax.plot(
        [crop_x + crop_w * (0.5 - dx * line_scale), crop_x + crop_w * (0.5 + dx * line_scale)],
        [crop_y + crop_h * (0.5 - dy * line_scale), crop_y + crop_h * (0.5 + dy * line_scale)],
        color=CYAN,
        lw=0.95,
        ls=(0, (4, 3)),
        zorder=18,
    )
    ax.add_patch(
        patches.Ellipse(
            (center_x, center_y),
            width=2.0 * radius_x,
            height=2.0 * radius_y,
            fill=False,
            edgecolor=CONTOUR_WINDOW,
            linewidth=0.9,
            linestyle=(0, (3.0, 2.0)),
            zorder=19,
        )
    )
    ax.plot(
        [crop_x + crop_w * (0.5 - dx * local_scale), crop_x + crop_w * (0.5 + dx * local_scale)],
        [crop_y + crop_h * (0.5 - dy * local_scale), crop_y + crop_h * (0.5 + dy * local_scale)],
        color=CYAN,
        lw=1.3,
        solid_capstyle="round",
        zorder=20,
    )
    ax.plot([center_x, center_x + radius_x], [center_y, center_y], color=CONTOUR_WINDOW, lw=0.65, zorder=21)
    ax.scatter([center_x], [center_y], s=15, facecolor="white", edgecolor=INK, linewidth=0.65, zorder=22)


def add_contour_window_callout(
    ax: plt.Axes,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    metadata: dict[str, float | str],
) -> None:
    radius_fraction = _finite_float(metadata.get("radius_fraction"), 38.0 / 151.0)
    xy = (crop_x + 0.50 * crop_w, crop_y + (0.50 + radius_fraction * 0.82) * crop_h)
    text_xy = (crop_x + 0.52 * crop_w, crop_y + crop_h + 0.040)
    ax.annotate(
        f"local contour estimate\n{metadata['radius_label']}; axis + coherence",
        xy=xy,
        xytext=text_xy,
        xycoords="data",
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#343A40",
        linespacing=1.08,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.82),
        arrowprops=dict(arrowstyle="-|>", lw=0.75, color=CONTOUR_WINDOW, mutation_scale=7, shrinkA=2, shrinkB=2),
        zorder=30,
    )


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
    label: str | None = None,
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
    if label:
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
    label: str | None = None,
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
        if label:
            ax.text(x + w / 2, y + h + 0.008, label, ha="center", va="bottom", fontsize=7.0, color=INK)
    except Exception:
        draw_movie_cube(ax, x, y + 0.035, w * 0.82, h * 0.56, label=label, jittered=fallback_jittered)


def draw_model_icon(ax: plt.Axes, x: float, y: float, sx: float = 1.0, sy: float | None = None) -> None:
    """Draw the compact core/conv/RF-readout icon.

    ``sx``/``sy`` scale the icon's horizontal and vertical extent
    independently (``sy`` defaults to ``sx``) so it can be squeezed
    horizontally without shrinking its vertical presence next to the taller
    movie-cube and response-map images either side of it.
    """
    sy = sx if sy is None else sy
    # The core box's own width uses a gentler compression (sqrt of sx) than
    # the rest of the icon -- squeezing it by the full sx leaves too little
    # room for the "Core"/"Conv" labels at a legible size.
    core_sx = sx**0.5
    core = patches.FancyBboxPatch(
        (x, y),
        0.058 * core_sx,
        0.155 * sy,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        facecolor="#DDE2E0",
        edgecolor="#AAB1AE",
        lw=0.8,
    )
    ax.add_patch(core)
    ax.text(x + 0.029 * core_sx, y + 0.127 * sy, "Core", ha="center", va="center", fontsize=5.2)
    ax.text(x + 0.029 * core_sx, y + 0.105 * sy, "Conv", ha="center", va="center", fontsize=4.4)
    for i, color in enumerate(["#E97B68", "#8FC9CF", "#F0C06C", "#E6A34E"]):
        yy = y + 0.029 * sy + i * 0.025 * sy
        ax.add_patch(
            patches.Rectangle(
                (x + 0.012 * core_sx, yy),
                0.030 * core_sx,
                0.014 * sy,
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
            )
        )
    ax.text(x + 0.029 * core_sx, y - 0.014 * sy, "model", ha="center", va="top", fontsize=5.2, color=GRAY)

    stack_x = x + 0.106 * sx
    add_flow_arrow(ax, (x + 0.058 * core_sx + 0.006, y + 0.077 * sy), (stack_x - 0.006, y + 0.077 * sy))
    for j, color in enumerate(["#E66A52", "#56B4AE", "#F0C05A", "#79A96B"]):
        ax.add_patch(
            patches.Rectangle(
                (stack_x + 0.006 * j * sx, y + 0.062 * sy + 0.004 * j * sy),
                0.040 * sx,
                0.029 * sy,
                facecolor="white",
                edgecolor=color,
                linewidth=0.75,
            )
        )
    add_flow_arrow(ax, (x + 0.160 * sx, y + 0.077 * sy), (x + 0.189 * sx, y + 0.077 * sy))

    rf_x = x + 0.204 * sx
    for j in range(3):
        ax.add_patch(
            patches.Rectangle(
                (rf_x + 0.016 * j * sx, y + 0.037 * sy + 0.021 * j * sy),
                0.049 * sx,
                0.080 * sy,
                facecolor="#EFF7EF",
                edgecolor="#49834E",
                linewidth=0.75,
                alpha=0.75,
            )
        )
    ax.plot([rf_x + 0.017 * sx, rf_x + 0.083 * sx], [y + 0.047 * sy, y + 0.123 * sy], color="#267335", lw=1.0)
    ax.plot(rf_x + 0.050 * sx, y + 0.084 * sy, marker="o", ms=2.4, color="#267335")
    ax.text(rf_x + 0.058 * sx, y + 0.134 * sy, "x,y", fontsize=4.8, color="#267335")
    ax.text(rf_x + 0.040 * sx, y + 0.017 * sy, "one response\nper position", ha="center", va="top", fontsize=4.7, color=GRAY)


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

    # Movie cubes are the real, primary images here, so they get the bulk of
    # both the width budget (icon compressed further, response map trimmed
    # back a little) and the height budget (rows now span label-to-label
    # with no per-cube caption eating into it -- see draw_schematic_movie_block).
    movie_x, movie_w, movie_h = 0.045, 0.390, 0.360
    top_label_y, bottom_label_y = 0.900, 0.450
    top_movie_y = top_label_y - 0.025 - movie_h
    # The response map's caption sits ~0.02 below its own y -- keep
    # bottom_movie_y off y=0 (rather than flush with it) so that caption
    # stays inside this axes instead of spilling into the row0/row1 gutter,
    # where D's E/F insets are independently reaching for the same space.
    bottom_movie_y = bottom_label_y - 0.025 - movie_h
    icon_x, icon_sx, icon_sy = movie_x + movie_w + 0.035, 0.50, 1.15
    icon_w = 0.285 * icon_sx  # matches draw_model_icon's own right-edge extent (rf box end) at scale sx
    # Wider gap than movie->icon: the icon's own "one response per position"
    # caption and the response map's "map pixel" caption both live in this
    # gap and collide if it's too tight.
    map_x, map_w, map_h = icon_x + icon_w + 0.075, 0.280, movie_h
    icon_y_offset = 0.1275 * movie_h
    top_icon_y, bottom_icon_y = top_movie_y + icon_y_offset, bottom_movie_y + icon_y_offset

    ax.text(movie_x, top_label_y, "FEM jittered movie", fontsize=11.0, ha="left", va="bottom")
    ax.text(movie_x, bottom_label_y, "Stabilized movie", fontsize=11.0, ha="left", va="bottom")
    map_note = "warm/cool = above/below map mean" if real_maps else "reserved for above/below-mean maps"
    ax.text(map_x + map_w / 2, top_label_y - 0.006, map_note, fontsize=6.4, color=GRAY, ha="center")

    draw_schematic_movie_block(
        ax,
        movie_x,
        top_movie_y,
        movie_w,
        movie_h,
        schematic_payload=schematic_payload,
        trace_key="stimulus_real_trace_lag32",
        trace_color=BLUE,
        fallback_jittered=True,
    )
    draw_model_icon(ax, icon_x, top_icon_y, sx=icon_sx, sy=icon_sy)
    add_flow_arrow(ax, (icon_x + icon_w + 0.008, top_icon_y + 0.077 * icon_sy), (map_x - 0.008, top_icon_y + 0.077 * icon_sy))
    draw_response_placeholder(
        ax,
        map_x,
        top_movie_y,
        map_w,
        map_h,
        "FEM response\nmap",
        real_map=real_maps.get("fem"),
        map_vlim=real_map_vlim,
    )
    ax.text(
        icon_x + 0.34 * icon_w,
        top_movie_y + movie_h - 0.048,
        "same spatial kernel, shifted center",
        fontsize=5.4,
        color=GRAY,
        ha="center",
    )

    draw_schematic_movie_block(
        ax,
        movie_x,
        bottom_movie_y,
        movie_w,
        movie_h,
        schematic_payload=schematic_payload,
        trace_key="stimulus_endpoint_stabilized_trace_lag32",
        trace_color=GRAY,
        fallback_jittered=False,
    )
    draw_model_icon(ax, icon_x, bottom_icon_y, sx=icon_sx, sy=icon_sy)
    add_flow_arrow(
        ax,
        (icon_x + icon_w + 0.008, bottom_icon_y + 0.077 * icon_sy),
        (map_x - 0.008, bottom_icon_y + 0.077 * icon_sy),
    )
    draw_response_placeholder(
        ax,
        map_x,
        bottom_movie_y,
        map_w,
        map_h,
        "stabilized response\nmap",
        real_map=real_maps.get("stable"),
        map_vlim=real_map_vlim,
    )


def draw_panel_a(
    ax: plt.Axes,
    *,
    schematic_payload: dict | None = None,
    draw_ef_insets: bool = True,
    panel_b_values: pd.DataFrame | None = None,
    ef_ylim: tuple[float, float] = (-32, 24),
    header_title: str = "Contour-relative stimulus",
    xlim: tuple[float, float] = (0.0, 1.0),
) -> None:
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    draw_panel_header(ax, "D", header_title, y=1.020)

    full_x, full_y, full_h = 0.012, 0.545, 0.420
    full_w = data_width_for_physical_aspect(ax, full_h, stimulus_canvas_aspect(schematic_payload))
    crop_y, crop_h = 0.500, 0.385
    crop_w = data_width_for_physical_aspect(ax, crop_h, 1.0)
    # Small overlap with the full-stimulus image, cascaded-photos style. This
    # used to also have to dodge a third "zoom" image pinned further right
    # (by EF_INSET_X); that zoom moved to panel G (see
    # draw_contour_components_panel), so the dodge is gone.
    crop_x = full_x + full_w - 0.075
    window_metadata = contour_window_metadata(schematic_payload)
    has_real_schematic = False
    if ssi_schematic is not None and schematic_payload is not None:
        try:
            full_ax = ax.inset_axes([full_x, full_y, full_w, full_h], transform=ax.transData)
            ssi_schematic.add_source_overview(
                full_ax,
                schematic_payload["stimulus_canvas"],
                schematic_payload["stimulus_crop_center_xy"],
                schematic_payload["stimulus_crop_size_px"],
                label=False,
            )
            full_ax.set_anchor("NW")
            full_ax.set_zorder(2)
            crop_ax = ax.inset_axes([crop_x, crop_y, crop_w, crop_h], transform=ax.transData)
            crop_ax.set_zorder(4)
            # No trace overlay and no center-zoom marker box here -- both the
            # real FEM traces and the zoomed detail they'd point to now live
            # in panel G, where there's room to actually see them (see
            # draw_contour_components_panel). D just shows the crop itself
            # plus the local-contour aperture (still relevant here since the
            # coherence gallery below draws on the same aperture concept).
            draw_plain_crop(crop_ax, schematic_payload.get("patch"))
            crop_ax.set_anchor("NW")
            crop_box_edge = getattr(ssi_schematic, "FIG3_CYAN", CYAN)
            crop_x1, crop_y1 = crop_ax.get_xlim()[1], crop_ax.get_ylim()[0]
            crop_ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    crop_x1,
                    crop_y1,
                    fill=False,
                    edgecolor=crop_box_edge,
                    linewidth=CROP_BORDER_LW,
                    zorder=14,
                )
            )
            add_contour_window_to_crop_axis(crop_ax, window_metadata)
            has_real_schematic = True
        except Exception:
            has_real_schematic = False

    if not has_real_schematic:
        placeholder_box(ax, full_x, full_y, full_w, full_h, "full stimulus", sublabel="image asset", hatch="...", label_size=6.3)
        roi = (full_x + 0.096, full_y + 0.109, 0.042, 0.068)
        ax.add_patch(patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, edgecolor=CYAN, linewidth=0.95))
        placeholder_box(ax, crop_x, crop_y, crop_w, crop_h, "model window", sublabel="151 x 151 crop", hatch="///", label_size=6.9)
        for start_y, end_y in [(roi[1] + roi[3], crop_y + crop_h), (roi[1], crop_y)]:
            ax.plot(
                [roi[0] + roi[2], crop_x],
                [start_y, end_y],
                color=CYAN,
                lw=D_CONNECTOR_LW,
                ls=(0, (3, 3)),
                alpha=0.58,
            )
        add_contour_window_parent_overlay(ax, crop_x, crop_y, crop_w, crop_h, window_metadata)
    else:
        ax.plot([full_x + full_w, crop_x], [full_y + full_h * 0.74, crop_y + crop_h], color=CYAN, lw=D_CONNECTOR_LW, ls=(0, (3, 3)), alpha=0.58)
        ax.plot([full_x + full_w, crop_x], [full_y + full_h * 0.26, crop_y], color=CYAN, lw=D_CONNECTOR_LW, ls=(0, (3, 3)), alpha=0.58)
        ax.text(full_x + full_w / 2, full_y - 0.022, "full stimulus", fontsize=D_IMAGE_LABEL_FS, color=GRAY, ha="center", va="top")
        ax.text(crop_x + 0.34 * crop_w, crop_y - 0.022, "151 x 151 crop", fontsize=D_IMAGE_LABEL_FS, color=GRAY, ha="center", va="top")

    # D's lower-left used to show the FEM short/long-path trace legend; the
    # real traces now live in G, at a scale where they're actually legible.
    # This space instead demonstrates what different local edge coherence
    # values look like as real image crops -- see
    # panels/panel_d_coherence_gallery.py and
    # panels/build_coherence_gallery_cache.py.
    gallery_kwargs = dict(x0=0.060, y0=0.075, w=0.540, h=0.200, gap=0.028, header_y=0.300)
    drew_gallery = False
    if panel_d_coherence_gallery is not None:
        try:
            drew_gallery = panel_d_coherence_gallery.draw_gallery(ax, **gallery_kwargs)
        except Exception:
            drew_gallery = False
    if not drew_gallery:
        if panel_d_coherence_gallery is not None:
            panel_d_coherence_gallery.draw_gallery_placeholder(ax, **gallery_kwargs)
        else:
            ax.text(0.060, 0.300, "local edge coherence", fontsize=D_SUBHEAD_FS, color=GRAY, ha="left")

    if not draw_ef_insets:
        return

    # E/F: same path-length dose curves as always, now living inside D's
    # axes (where the unfinished "Contour-carried signal" placeholder used
    # to be) rather than a separate gridspec cell -- see EF_INSET_* constants.
    ax_e = ax.inset_axes([EF_INSET_X, EF_INSET_E_Y, EF_INSET_W, EF_INSET_E_H], transform=ax.transData)
    draw_panel_bcef_or_placeholder(
        ax_e,
        panel_b_values if panel_b_values is not None else pd.DataFrame(),
        label="E",
        title="Low-SF aligned units",
        sf_group="low_lt0p5",
        relation="contour_matched",
        color=BLUE,
        ylabel="SSI change (%)",
        ylim=ef_ylim,
    )
    ax_f = ax.inset_axes([EF_INSET_X, EF_INSET_F_Y, EF_INSET_W, EF_INSET_F_H], transform=ax.transData)
    draw_panel_bcef_or_placeholder(
        ax_f,
        panel_b_values if panel_b_values is not None else pd.DataFrame(),
        label="F",
        title="High-SF aligned units",
        sf_group="high_ge0p75",
        relation="contour_matched",
        color=ORANGE,
        ylabel="SSI change (%)",
        ylim=ef_ylim,
    )


def draw_rms_excursion_explainer(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    motion_eye: dict | None,
    axis_image_deg: float,
) -> bool:
    """Decompose G's own example trace into across-/along-contour components,
    the same split H's dose curve reports as separate lines. Real geometry
    from this trace (not H's population statistic) -- illustrative, not a
    stand-in for H's actual numbers.
    """
    if not isinstance(motion_eye, dict) or "large_xy_px" not in motion_eye:
        return False
    diagram_ax = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
    diagram_ax.set_axis_off()

    dx, dy = _axis_vector_image(axis_image_deg)
    norm = math.hypot(dx, dy)
    tangent = np.array([dx, dy]) / norm if norm > 0 else np.array([1.0, 0.0])
    normal = np.array([-tangent[1], tangent[0]])
    trace_px = np.asarray(motion_eye["large_xy_px"], dtype=np.float64)
    along_px = trace_px @ tangent
    across_px = trace_px @ normal
    along_px = along_px - along_px.mean()
    across_px = across_px - across_px.mean()
    ppd = _finite_float(getattr(ssi_schematic, "MODEL_PPD", None), 37.50476617)
    along_arcmin = along_px / ppd * 60.0
    across_arcmin = across_px / ppd * 60.0
    rms_along = float(np.sqrt(np.mean(along_arcmin**2)))
    rms_across = float(np.sqrt(np.mean(across_arcmin**2)))

    half_span = max(float(np.abs(along_arcmin).max()), float(np.abs(across_arcmin).max()), rms_along, rms_across) * 1.5
    diagram_ax.set_xlim(-half_span, half_span)
    diagram_ax.set_ylim(-half_span * 1.18, half_span)
    # Not aspect="equal": the panel is much wider than tall, and forcing
    # equal data/physical scaling squeezes the along-axis (x) bracket down
    # to a sliver -- its arrowheads then overlap into an illegible blob.
    # This is a schematic, not a ruler, so a stretched trace is an
    # acceptable trade for both brackets rendering at a legible size.
    diagram_ax.axhline(0.0, color=CYAN, lw=1.3, ls=(0, (4, 3)), alpha=0.85, zorder=2)
    diagram_ax.plot(along_arcmin, across_arcmin, color=BLUE, lw=1.3, zorder=3)
    diagram_ax.scatter([0], [0], s=14, facecolor="white", edgecolor=INK, linewidth=0.6, zorder=4)

    bracket_y = -half_span * 1.08
    diagram_ax.annotate(
        "",
        xy=(rms_along, bracket_y),
        xytext=(-rms_along, bracket_y),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=INK),
    )
    diagram_ax.text(
        0.0, bracket_y - half_span * 0.10, f"along: {rms_along:.1f}'", ha="center", va="top", fontsize=5.8, color=INK
    )

    bracket_x = half_span * 1.06
    diagram_ax.annotate(
        "",
        xy=(bracket_x, rms_across),
        xytext=(bracket_x, -rms_across),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color=INK),
    )
    diagram_ax.text(
        bracket_x + half_span * 0.10,
        0.0,
        f"across: {rms_across:.1f}'",
        ha="left",
        va="center",
        fontsize=5.8,
        color=INK,
        rotation=90,
    )
    return True


def draw_contour_components_panel(
    ax: plt.Axes,
    *,
    label: str = "G",
    title: str = "Local contour detail",
    schematic_payload: dict | None = None,
) -> None:
    """Reference crop plus the zoomed local-contour aperture, moved from D.

    D's cascade used to be full stimulus -> 151x151 crop -> zoomed detail,
    all three crowded into one row and competing with D's E/F insets for
    width. The zoomed detail now lives here instead; a copy of the 151x151
    crop comes along as a reference (D keeps its own, independent copy) so
    the zoom isn't shown floating with no visual anchor. The crop/zoom
    overlap is kept proportionally the same as it was in D -- as a fraction
    of the crop's own size, since this axes has a different physical aspect
    than D's (tall and narrow here vs. D's short and wide).
    """
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    set_panel_title(ax, label, title, fontsize=9.0, pad=4)

    crop_x, crop_y, crop_h = 0.060, 0.560, 0.320
    crop_w = data_width_for_physical_aspect(ax, crop_h, 1.0)
    zoom_h = 0.857 * crop_h  # matches D's original zoom_h / crop_h ratio (0.330 / 0.385)
    zoom_w = data_width_for_physical_aspect(ax, zoom_h, 1.0)
    zoom_x = crop_x + crop_w * (1.0 - 0.262)  # matches D's original (crop_w - 0.055) / crop_w overlap fraction
    zoom_y = crop_y - 0.117 * crop_h  # matches D's original (crop_y - zoom_y) / crop_h offset fraction

    window_metadata = contour_window_metadata(schematic_payload)
    axis_image_deg = _finite_float((schematic_payload or {}).get("contour_axis_image_deg"), 10.352312)
    motion_eye: dict | None = None
    has_real_schematic = False
    if ssi_schematic is not None and schematic_payload is not None:
        try:
            crop_ax = ax.inset_axes([crop_x, crop_y, crop_w, crop_h], transform=ax.transData)
            crop_ax.set_zorder(4)
            synthetic_left = ssi_schematic.make_synthetic_left_side(
                schematic_payload.get("patch"),
                schematic_payload.get("contour_axis_image_deg", 10.352312),
            )
            motion_eye = restore_trace_orientation(synthetic_left.get("eye"))
            window_metadata = trace_fit_center_zoom_metadata(window_metadata, motion_eye)
            # Only the long-path trace, in TRACE_COLOR (not add_stimulus's
            # hardcoded red+blue pair) -- see draw_plain_crop.
            draw_plain_crop(
                crop_ax,
                schematic_payload.get("patch"),
                trace_xy_px=motion_eye.get("large_xy_px") if isinstance(motion_eye, dict) else None,
                trace_color=TRACE_COLOR,
            )
            crop_ax.set_anchor("NW")
            crop_box_edge = ZOOM_BOX
            crop_x1, crop_y1 = crop_ax.get_xlim()[1], crop_ax.get_ylim()[0]
            crop_ax.add_patch(
                patches.Rectangle(
                    (0, 0), crop_x1, crop_y1, fill=False, edgecolor=crop_box_edge, linewidth=CROP_BORDER_LW, zorder=14
                )
            )
            add_contour_window_to_crop_axis(crop_ax, window_metadata)
            add_center_zoom_box_to_crop_axis(crop_ax, window_metadata)

            zoom_ax = ax.inset_axes([zoom_x, zoom_y, zoom_w, zoom_h], transform=ax.transData)
            zoom_ax.set_zorder(6)
            add_zoomed_crop_view(zoom_ax, schematic_payload, motion_eye, window_metadata)
            zoom_ax.set_anchor("NW")
            has_real_schematic = True
        except Exception:
            has_real_schematic = False

    if not has_real_schematic:
        placeholder_box(
            ax, crop_x, crop_y, crop_w, crop_h, "model window", sublabel="151 x 151 crop\n(reference)", hatch="///", label_size=6.4
        )
        placeholder_box(
            ax, zoom_x, zoom_y, zoom_w, zoom_h, "trace zoom", sublabel=center_zoom_label(window_metadata), hatch="\\\\\\", label_size=6.0
        )
        add_contour_window_parent_overlay(ax, crop_x, crop_y, crop_w, crop_h, window_metadata)
        add_center_zoom_parent_overlay(ax, crop_x, crop_y, crop_w, crop_h, window_metadata)
    else:
        ax.text(
            crop_x + 0.34 * crop_w, crop_y - 0.028, "151 x 151 crop", fontsize=D_IMAGE_LABEL_FS, color=GRAY, ha="center", va="top"
        )
        ax.text(
            zoom_x + zoom_w / 2, zoom_y - 0.030, center_zoom_label(window_metadata), fontsize=D_IMAGE_LABEL_FS, color=GRAY, ha="center", va="top"
        )

    zoom_fraction = _finite_float(window_metadata.get("center_zoom_fraction"), CENTER_ZOOM_HALF_DEG * 37.50476617 / 151.0)
    zoom_source_box = (
        crop_x + crop_w * (0.5 - zoom_fraction),
        crop_y + crop_h * (0.5 - zoom_fraction),
        2.0 * crop_w * zoom_fraction,
        2.0 * crop_h * zoom_fraction,
    )
    add_zoom_connectors(ax, zoom_source_box, (zoom_x, zoom_y, zoom_w, zoom_h), color=ZOOM_BOX)

    ax.text(
        0.02,
        0.40,
        "Reference crop + zoomed aperture from D.",
        fontsize=6.2,
        color="#333333",
        ha="left",
        va="top",
        linespacing=1.15,
    )

    # Lower half: decompose this same trace into the across-/along-contour
    # components H's dose curve reports as separate lines, so a reader sees
    # what "across" and "along" mean geometrically before H uses them.
    ax.text(0.02, 0.335, "RMS excursion, by direction", fontsize=7.0, color=GRAY, ha="left")
    drew_explainer = draw_rms_excursion_explainer(
        ax, x0=0.14, y0=0.145, w=0.72, h=0.180, motion_eye=motion_eye, axis_image_deg=axis_image_deg
    )
    if not drew_explainer:
        ax.text(
            0.50,
            0.18,
            "trace decomposition\nunavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.0,
            color=GRAY,
        )


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
    show_microsaccade_legend: bool = False,
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
                show_microsaccade_legend=show_microsaccade_legend,
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
    label: str = "H",
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
    set_panel_title(ax, "I", "Real FEMs are anisotropic\nnear local contours", fontsize=7.8, pad=5)
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
    set_panel_title(
        ax,
        "J",
        "Real (vs. randomly rotated)\nFEM trajectories preferentially\nbenefit aligned high-SF units",
        fontsize=7.4,
        pad=5,
    )
    ax.axhline(0.0, color="0.35", lw=0.85)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel("observed - random rotated (pp SSI)")
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
    if panel_g_rms_excursion is not None:
        try:
            panel_g_rms_excursion.draw_panel(ax)
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
        "Draft composition; B/C/E/F/H/I/J and schematic image/map assets reuse existing BackImage/Figure 4 data"
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
    has_existing_panels = not panel_b_values.empty and panel_g_rms_excursion is not None
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

    ax_b = _wide_panel_axes(fig, gs, 0)
    draw_panel_b(ax_b, schematic_payload=schematic_payload)

    gs_b_right = gs[0, 2].subgridspec(2, 1, hspace=RIGHT_PANEL_HSPACE)
    ax_b1 = fig.add_subplot(gs_b_right[0, 0])
    _shrink_axes_center(ax_b1)
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
        show_microsaccade_legend=True,
    )
    ax_b2 = fig.add_subplot(gs_b_right[1, 0])
    _shrink_axes_center(ax_b2)
    draw_panel_bcef_or_placeholder(
        ax_b2,
        panel_b_values,
        label="C",
        title="High-SF units",
        sf_group="high_ge0p75",
        relation="strong_contours_no_osi",
        color=ORANGE,
        ylabel="SSI change (%)",
        ylim=bc_ylim,
    )

    ax_a = _wide_panel_axes(fig, gs, 1)
    draw_panel_a(ax_a, schematic_payload=schematic_payload, panel_b_values=panel_b_values, ef_ylim=ef_ylim)

    ax_g_new = fig.add_subplot(gs[1, 2])
    draw_contour_components_panel(ax_g_new, schematic_payload=schematic_payload)

    ax_c1 = fig.add_subplot(gs[2, 0])
    draw_panel_g_or_fallback(ax_c1, component_values, component_ylim=component_ylim)
    ax_c2 = fig.add_subplot(gs[2, 1])
    draw_generated_panel_or_placeholder(ax_c2, panel_h_unwrapped_edge_coherence, draw_path_alignment_placeholder)
    ax_c3 = fig.add_subplot(gs[2, 2])
    draw_generated_panel_or_placeholder(ax_c3, panel_i_match_advantage, draw_edge_alignment_placeholder)

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
