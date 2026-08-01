#!/usr/bin/env python3
"""Editable box template for Panel A's block layout (position/size of each
movie cube, network-icon slot, and response-map slot, plus each row
label's anchor point): export the current geometry as an SVG -- labeled
rects drawn over a raster preview of the actual current render -- so it
can be dragged/resized by hand in Illustrator/Inkscape, then re-imported
as an override that generate_ssi_figure_v2.draw_panel_b reads instead of
its own hardcoded defaults (panel_a_default_layout_boxes()).

Eight boxes total, two per row ("fem"/"stable"): movie_*, icon_*, map_*,
label_*. Each is fully independent once overridden -- e.g. dragging
movie_fem does NOT also drag icon_fem's box along with it, unlike the
*default* geometry, where icon/map positions derive from the movie box via
fixed gaps. If you move a cube and want its icon/map to follow, drag those
boxes too.

This is deliberately scoped to block layout (position/size) -- NOT the
vector content inside those blocks (the cube's own isometric skew, the
network icon's dot lattice), which either already has its own real
vector-art round trip (extract_panel_a_network_icon.py) or isn't
representable as a rectangle at all. Dragging a box here changes the
canvas an element is scaled into, not its internal geometry.

Caveat found while testing this: a map_* box's WIDTH has no visual effect.
draw_response_placeholder derives the response-map image's width purely
from the box's height (locked to the map data's own 1:1 aspect ratio) and
never reads w -- only x/y/h matter for map_fem/map_stable. Resizing a
map box's width will update the number stored in the override but won't
change what's drawn; use height to resize it, x/y to move it.

Round trip:
    uv run python declan/fig/ssi_figure_v2/panels/panel_a_layout_boxes.py export
    # open outputs/fig/ssi_figure_v2/panels/cache/panel_a_layout_boxes.svg,
    # drag/resize any of the 8 labeled boxes, save
    uv run python declan/fig/ssi_figure_v2/panels/panel_a_layout_boxes.py import
    # panel_a_motion_schematic.build_panel() now reflects the edited boxes
"""

from __future__ import annotations

import base64
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[4]
FIGURE_DIR = ROOT / "declan" / "fig" / "ssi_figure_v2"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

import generate_ssi_figure_v2 as figure  # noqa: E402

from panels import panel_a_motion_schematic as panel_a  # noqa: E402
from panels.svg_box_utils import find_rect_bbox  # noqa: E402

CACHE_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "cache"
BOXES_SVG = CACHE_DIR / "panel_a_layout_boxes.svg"
OVERRIDES_JSON = panel_a.LAYOUT_OVERRIDES_JSON

# label_fem/label_stable's real height (0.001) is just a baseline marker --
# too thin to see or grab in an SVG editor -- so the exported template
# draws them taller. Only each box's (x, y) corner is read back on import
# (see panel_a_default_layout_boxes()'s own docstring: label boxes' w/h
# are unused by draw_panel_b), so the taller export height is cosmetic.
LABEL_EXPORT_HEIGHT = 0.035

BOX_IDS = [
    "movie_fem",
    "movie_stable",
    "label_fem",
    "label_stable",
    "icon_fem",
    "icon_stable",
    "map_fem",
    "map_stable",
]
BOX_COLORS = {
    "movie_fem": "#0072B2",
    "movie_stable": "#0072B2",
    "label_fem": "#D55E00",
    "label_stable": "#D55E00",
    "icon_fem": "#009E73",
    "icon_stable": "#009E73",
    "map_fem": "#CC79A7",
    "map_stable": "#CC79A7",
}
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _current_boxes() -> dict[str, tuple[float, float, float, float]]:
    overrides = panel_a.load_layout_overrides()
    return {**figure.panel_a_default_layout_boxes(), **(overrides or {})}


def _box_to_svg_rect(
    box: tuple[float, float, float, float], figsize: tuple[float, float]
) -> tuple[float, float, float, float]:
    """axes-fraction (x, y, w, h), bottom-left origin -> SVG (x, y, w, h),
    top-left origin, in the same point units as the panel's own page."""
    x, y, w, h = box
    x0_pt, y0_pt = panel_a.data_frac_to_page_pt(x, y, figsize)
    x1_pt, y1_pt = panel_a.data_frac_to_page_pt(x + w, y + h, figsize)
    panel_h_pt = figsize[1] * 72.0
    svg_x = min(x0_pt, x1_pt)
    svg_w = abs(x1_pt - x0_pt)
    svg_y = panel_h_pt - max(y0_pt, y1_pt)
    svg_h = abs(y1_pt - y0_pt)
    return svg_x, svg_y, svg_w, svg_h


def _svg_rect_to_box(
    svg_x: float, svg_y: float, svg_w: float, svg_h: float, figsize: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Inverse of _box_to_svg_rect."""
    panel_h_pt = figsize[1] * 72.0
    x0_pt, y_top_bottom_up = svg_x, panel_h_pt - svg_y
    x1_pt, y_bottom_bottom_up = svg_x + svg_w, panel_h_pt - (svg_y + svg_h)
    x0_frac, y0_frac = panel_a.page_pt_to_data_frac(x0_pt, y_bottom_bottom_up, figsize)
    x1_frac, y1_frac = panel_a.page_pt_to_data_frac(x1_pt, y_top_bottom_up, figsize)
    return x0_frac, y0_frac, x1_frac - x0_frac, y1_frac - y0_frac


def export_layout_boxes(out_svg: Path = BOXES_SVG) -> Path:
    figsize = panel_a.DEFAULT_FIGSIZE
    panel_pdf = panel_a.build_panel()  # fresh render, reflects any existing overrides

    doc = fitz.open(str(panel_pdf))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=True)
    png_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    doc.close()

    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0
    boxes = _current_boxes()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" '
            f'width="{panel_w_pt:.2f}pt" height="{panel_h_pt:.2f}pt" '
            f'viewBox="0 0 {panel_w_pt:.2f} {panel_h_pt:.2f}">'
        ),
        "  <title>Panel A editable block-layout boxes</title>",
        (
            "  <desc>Background is the current render, faded. Drag/resize the "
            "labeled rects (movie_fem, movie_stable, label_fem, label_stable), "
            "save, then run: panels/panel_a_layout_boxes.py import</desc>"
        ),
        (
            f'  <image x="0" y="0" width="{panel_w_pt:.2f}" height="{panel_h_pt:.2f}" '
            f'xlink:href="data:image/png;base64,{png_b64}" opacity="0.55"/>'
        ),
    ]
    for box_id in BOX_IDS:
        box = boxes[box_id]
        if box_id.startswith("label_"):
            box = (box[0], box[1], box[2], LABEL_EXPORT_HEIGHT)
        svg_x, svg_y, svg_w, svg_h = _box_to_svg_rect(box, figsize)
        color = BOX_COLORS[box_id]
        lines.append(
            f'  <rect id="{box_id}" x="{svg_x:.2f}" y="{svg_y:.2f}" '
            f'width="{svg_w:.2f}" height="{svg_h:.2f}" '
            f'fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="1.25"/>'
        )
        lines.append(
            f'  <text x="{svg_x + 3:.2f}" y="{svg_y + 11:.2f}" font-family="monospace" '
            f'font-size="8" fill="{color}">{box_id}</text>'
        )
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_svg


def import_layout_boxes(svg_path: Path = BOXES_SVG) -> dict[str, tuple[float, float, float, float]]:
    figsize = panel_a.DEFAULT_FIGSIZE
    root = ET.parse(svg_path).getroot()

    overrides: dict[str, tuple[float, float, float, float]] = {}
    for box_id in BOX_IDS:
        svg_x, svg_y, svg_w, svg_h = find_rect_bbox(root, box_id)
        overrides[box_id] = _svg_rect_to_box(svg_x, svg_y, svg_w, svg_h, figsize)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDES_JSON.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return overrides


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"export", "import"}:
        print("usage: panel_a_layout_boxes.py {export|import}")
        raise SystemExit(2)
    if sys.argv[1] == "export":
        path = export_layout_boxes()
        print(f"wrote {path}")
        print(f"Drag/resize any of: {', '.join(BOX_IDS)}, save, then run:")
        print("  uv run python declan/fig/ssi_figure_v2/panels/panel_a_layout_boxes.py import")
    else:
        overrides = import_layout_boxes()
        print(f"wrote {OVERRIDES_JSON}")
        for name, box in overrides.items():
            print(f"  {name}: {tuple(round(v, 4) for v in box)}")


if __name__ == "__main__":
    main()
