#!/usr/bin/env python3
"""Editable box template for Panel D's block layout (full-stimulus image,
151x151 crop, and local-edge-coherence gallery): export the current
geometry as an SVG -- labeled rects over a raster preview of the actual
current render -- so it can be dragged/resized by hand, then re-imported
as an override that generate_ssi_figure_v2.draw_panel_a reads instead of
its own hardcoded defaults (panel_d_default_layout_boxes()).

Same pattern as panel_a_layout_boxes.py (see that module's docstring for
the general rationale). Two differences worth knowing:

- full_stimulus/crop's widths are derived from height to preserve the real
  image's own aspect ratio (data_width_for_physical_aspect) -- an override
  is used verbatim regardless, so an aspect that doesn't match the source
  image will letterbox/crop within the box rather than stretch it (as
  draw_plain_crop/add_source_overview already lock aspect internally).
- E/F used to live as insets inside this same axes; v3 draws them as their
  own top-level panels instead (draw_ef_insets=False), so they have no
  box here -- see panel_bcef_path_bins.py.

Round trip:
    uv run python declan/fig/ssi_figure_v2/panels/panel_d_layout_boxes.py export
    # open outputs/fig/ssi_figure_v2/panels/cache/panel_d_layout_boxes.svg,
    # drag/resize full_stimulus / crop / gallery, save
    uv run python declan/fig/ssi_figure_v2/panels/panel_d_layout_boxes.py import
    # panel_d_contour_relative_stimulus.build_panel() now reflects the edit
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

from panels import panel_d_contour_relative_stimulus as panel_d  # noqa: E402
from panels.svg_box_utils import find_rect_bbox  # noqa: E402

CACHE_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "cache"
BOXES_SVG = CACHE_DIR / "panel_d_layout_boxes.svg"
OVERRIDES_JSON = panel_d.LAYOUT_OVERRIDES_JSON

BOX_IDS = ["full_stimulus", "crop", "gallery"]
BOX_COLORS = {
    "full_stimulus": "#0072B2",
    "crop": "#009E73",
    "gallery": "#CC79A7",
}
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _box_to_svg_rect(box: tuple[float, float, float, float], figsize: tuple[float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    x0_pt, y0_pt = panel_d.data_frac_to_page_pt(x, y, figsize)
    x1_pt, y1_pt = panel_d.data_frac_to_page_pt(x + w, y + h, figsize)
    panel_h_pt = figsize[1] * 72.0
    svg_x = min(x0_pt, x1_pt)
    svg_w = abs(x1_pt - x0_pt)
    svg_y = panel_h_pt - max(y0_pt, y1_pt)
    svg_h = abs(y1_pt - y0_pt)
    return svg_x, svg_y, svg_w, svg_h


def _svg_rect_to_box(
    svg_x: float, svg_y: float, svg_w: float, svg_h: float, figsize: tuple[float, float]
) -> tuple[float, float, float, float]:
    panel_h_pt = figsize[1] * 72.0
    x0_pt, y_top_bottom_up = svg_x, panel_h_pt - svg_y
    x1_pt, y_bottom_bottom_up = svg_x + svg_w, panel_h_pt - (svg_y + svg_h)
    x0_frac, y0_frac = panel_d.page_pt_to_data_frac(x0_pt, y_bottom_bottom_up, figsize)
    x1_frac, y1_frac = panel_d.page_pt_to_data_frac(x1_pt, y_top_bottom_up, figsize)
    return x0_frac, y0_frac, x1_frac - x0_frac, y1_frac - y0_frac


def export_layout_boxes(out_svg: Path = BOXES_SVG) -> Path:
    figsize = panel_d.DEFAULT_FIGSIZE
    panel_pdf = panel_d.build_panel()  # fresh render, reflects any existing overrides
    boxes = panel_d.compute_current_boxes(figsize)

    doc = fitz.open(str(panel_pdf))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=True)
    png_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    doc.close()

    panel_w_pt, panel_h_pt = figsize[0] * 72.0, figsize[1] * 72.0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" '
            f'width="{panel_w_pt:.2f}pt" height="{panel_h_pt:.2f}pt" '
            f'viewBox="0 0 {panel_w_pt:.2f} {panel_h_pt:.2f}">'
        ),
        "  <title>Panel D editable block-layout boxes</title>",
        (
            "  <desc>Background is the current render, faded. Drag/resize "
            "full_stimulus / crop / gallery, save, then run: "
            "panels/panel_d_layout_boxes.py import</desc>"
        ),
        (
            f'  <image x="0" y="0" width="{panel_w_pt:.2f}" height="{panel_h_pt:.2f}" '
            f'xlink:href="data:image/png;base64,{png_b64}" opacity="0.55"/>'
        ),
    ]
    for box_id in BOX_IDS:
        svg_x, svg_y, svg_w, svg_h = _box_to_svg_rect(boxes[box_id], figsize)
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
    figsize = panel_d.DEFAULT_FIGSIZE
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
        print("usage: panel_d_layout_boxes.py {export|import}")
        raise SystemExit(2)
    if sys.argv[1] == "export":
        path = export_layout_boxes()
        print(f"wrote {path}")
        print(f"Drag/resize any of: {', '.join(BOX_IDS)}, save, then run:")
        print("  uv run python declan/fig/ssi_figure_v2/panels/panel_d_layout_boxes.py import")
    else:
        overrides = import_layout_boxes()
        print(f"wrote {OVERRIDES_JSON}")
        for name, box in overrides.items():
            print(f"  {name}: {tuple(round(v, 4) for v in box)}")


if __name__ == "__main__":
    main()
