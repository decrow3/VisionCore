#!/usr/bin/env python3
"""Editable box template for where each lettered panel (A-J) sits on the
final assembled page: export the current PANEL_BOXES geometry as an SVG --
labeled rects over a raster preview of the actual composited page -- so it
can be dragged/resized by hand, then re-imported to rewrite
panels/reference_layout_v3.py directly (the same module
compose_ssi_figure_v3.py reads PANEL_BOXES from, and the one
extract_reference_layout.py also writes when re-deriving boxes from
ssi_figure_v2_3.pdf from scratch -- this is a second, independent way to
update the same file, by hand instead of by re-measuring the reference PDF).

This is the page-level counterpart to panel_a_layout_boxes.py, which
handles sub-element layout *inside* Panel A. Panels B, C, E, F, H, I, J are
each a single self-contained plot with no internal sub-composition, so
their PANEL_BOXES entry already is their whole layout -- this module is
the complete story for those seven. Panels D and G also have internal
inset_axes sub-elements (not yet exposed as their own editable boxes, the
way Panel A's cube/icon/map are); moving their box here still correctly
repositions/resizes the whole panel, just without independent control over
what's inside it.

PANEL_BOXES is already stored (x_in, y_in, w_in, h_in) in inches, top-left
origin, y from the page's top edge -- the same convention SVG uses -- so
unlike Panel A's internal axes-fraction boxes (which go through AX_BOX
first), this is a plain inches-to-points scale with no coordinate flip.

Round trip:
    uv run python declan/fig/ssi_figure_v2/panels/page_layout_boxes.py export
    # open outputs/fig/ssi_figure_v2/panels/cache/page_layout_boxes.svg,
    # drag/resize any panel's box (A-J), save
    uv run python declan/fig/ssi_figure_v2/panels/page_layout_boxes.py import
    # rewrites panels/reference_layout_v3.py; re-run compose_ssi_figure_v3.py
"""

from __future__ import annotations

import base64
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[4]
FIGURE_DIR = ROOT / "declan" / "fig" / "ssi_figure_v2"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

import compose_ssi_figure_v3  # noqa: E402

from panels import reference_layout_v3 as layout  # noqa: E402
from panels.svg_box_utils import find_rect_bbox  # noqa: E402

CACHE_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "cache"
BOXES_SVG = CACHE_DIR / "page_layout_boxes.svg"
OUT_PY = Path(__file__).resolve().parent / "reference_layout_v3.py"

PANEL_LETTERS = list("ABCDEFGHIJ")
BOX_COLOR = "#0072B2"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def export_layout_boxes(out_svg: Path = BOXES_SVG) -> Path:
    page_pdf = compose_ssi_figure_v3.compose()["pdf"]  # fresh full-page render

    doc = fitz.open(str(page_pdf))
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=True)
    png_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    doc.close()

    page_w_in, page_h_in = layout.PAGE_SIZE_IN
    page_w_pt, page_h_pt = page_w_in * 72.0, page_h_in * 72.0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" '
            f'width="{page_w_pt:.2f}pt" height="{page_h_pt:.2f}pt" '
            f'viewBox="0 0 {page_w_pt:.2f} {page_h_pt:.2f}">'
        ),
        "  <title>ssi_figure_v3 editable page-level panel boxes</title>",
        (
            "  <desc>Background is the current composite, faded. Drag/resize any "
            "panel's box (A-J), save, then run: panels/page_layout_boxes.py import</desc>"
        ),
        (
            f'  <image x="0" y="0" width="{page_w_pt:.2f}" height="{page_h_pt:.2f}" '
            f'xlink:href="data:image/png;base64,{png_b64}" opacity="0.55"/>'
        ),
    ]
    for letter in PANEL_LETTERS:
        x_in, y_in, w_in, h_in = layout.PANEL_BOXES[letter]
        x_pt, y_pt, w_pt, h_pt = x_in * 72.0, y_in * 72.0, w_in * 72.0, h_in * 72.0
        lines.append(
            f'  <rect id="{letter}" x="{x_pt:.2f}" y="{y_pt:.2f}" '
            f'width="{w_pt:.2f}" height="{h_pt:.2f}" '
            f'fill="{BOX_COLOR}" fill-opacity="0.12" stroke="{BOX_COLOR}" stroke-width="1.25"/>'
        )
        lines.append(
            f'  <text x="{x_pt + 3:.2f}" y="{y_pt + 12:.2f}" font-family="monospace" '
            f'font-size="10" font-weight="bold" fill="{BOX_COLOR}">{letter}</text>'
        )
    lines.append("</svg>")
    out_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_svg


def _write_reference_layout_module(
    boxes_in: dict[str, tuple[float, float, float, float]], page_size_in: tuple[float, float]
) -> None:
    page_w_in, page_h_in = page_size_in
    lines = [
        '"""Panel bounding boxes for ssi_figure_v3\'s page layout.',
        "",
        "PANEL_BOXES[letter] = (x_in, y_in, w_in, h_in), inches, y measured from",
        "the page's TOP edge; convert to PDF-native bottom-up points when",
        "compositing: ty = page_h_pt - y_in*72 - h_in*72.",
        "",
        "This file is written by either of two independent tools -- re-deriving",
        "it from the reference PDF discards any hand edits made by the other:",
        "    uv run python declan/fig/ssi_figure_v2/panels/extract_reference_layout.py",
        "        (re-measures every box from ssi_figure_v2_3.pdf from scratch)",
        "    uv run python declan/fig/ssi_figure_v2/panels/page_layout_boxes.py import",
        "        (reads back a hand-dragged/resized SVG export of these same boxes)",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"PAGE_SIZE_IN = ({page_w_in:.4f}, {page_h_in:.4f})",
        "",
        "PANEL_BOXES: dict[str, tuple[float, float, float, float]] = {",
    ]
    for letter in PANEL_LETTERS:
        x_in, y_in, w_in, h_in = boxes_in[letter]
        lines.append(f'    "{letter}": ({x_in:.4f}, {y_in:.4f}, {w_in:.4f}, {h_in:.4f}),')
    lines.append("}")
    lines.append("")
    OUT_PY.write_text("\n".join(lines), encoding="utf-8")


def import_layout_boxes(svg_path: Path = BOXES_SVG) -> dict[str, tuple[float, float, float, float]]:
    root = ET.parse(svg_path).getroot()
    page_w_in, page_h_in = layout.PAGE_SIZE_IN

    boxes_in: dict[str, tuple[float, float, float, float]] = {}
    for letter in PANEL_LETTERS:
        svg_x, svg_y, svg_w, svg_h = find_rect_bbox(root, letter)
        boxes_in[letter] = (svg_x / 72.0, svg_y / 72.0, svg_w / 72.0, svg_h / 72.0)

    _write_reference_layout_module(boxes_in, (page_w_in, page_h_in))
    print(f"wrote {OUT_PY}")
    return boxes_in


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"export", "import"}:
        print("usage: page_layout_boxes.py {export|import}")
        raise SystemExit(2)
    if sys.argv[1] == "export":
        path = export_layout_boxes()
        print(f"wrote {path}")
        print(f"Drag/resize any of: {', '.join(PANEL_LETTERS)}, save, then run:")
        print("  uv run python declan/fig/ssi_figure_v2/panels/page_layout_boxes.py import")
    else:
        boxes_in = import_layout_boxes()
        for letter, box in boxes_in.items():
            print(f"  {letter}: {tuple(round(v, 4) for v in box)} in")


if __name__ == "__main__":
    main()
