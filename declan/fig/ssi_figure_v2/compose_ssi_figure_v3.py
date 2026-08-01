#!/usr/bin/env python3
"""SSI figure v3: real compositing, not one shared matplotlib canvas.

Every panel (A-J) is rendered as its own independent figure, sized from
PANEL_BOXES (panels/reference_layout_v3.py -- measured directly off
ssi_figure_v2_3.pdf, the Illustrator-fixed reference artwork), then placed
onto a blank page at that measured position via pypdf. No panel's drawing
code shares a coordinate tree with any other panel's, so one panel's
opaque axes background can no longer paint over a neighboring panel's text
-- the recurring failure mode in generate_ssi_figure_v2.py's nested
ax.inset_axes() approach.

Run: uv run python declan/fig/ssi_figure_v2/compose_ssi_figure_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = ROOT / "declan" / "fig" / "ssi_figure_v2"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

from panels import reference_layout_v3 as layout  # noqa: E402
from panels import panel_a_motion_schematic  # noqa: E402
from panels import panel_d_contour_relative_stimulus  # noqa: E402
from panels import panel_g_local_contour_detail  # noqa: E402
from panels import panel_bcef_path_bins  # noqa: E402
from panels import panel_g_rms_excursion  # noqa: E402  (module name is historical; displays as "H")
from panels import panel_h_unwrapped_edge_coherence  # noqa: E402  (displays as "I")
from panels import panel_j_match_advantage  # noqa: E402  (displays as "J")

OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2"
PANELS_OUT_DIR = OUT_DIR / "panels_v3"
TITLE_TEXT = "Single-spike information and contour-relative FEM"


def build_title_panel(page_w_in: float, out_dir: Path = PANELS_OUT_DIR) -> Path:
    """The one page-level element that isn't inside any lettered panel --
    built the same way as every panel, so the compositor has exactly one
    code path (no special-casing) for placing it."""
    fig = plt.figure(figsize=(page_w_in, 0.42))
    fig.text(0.5, 0.5, TITLE_TEXT, ha="center", va="center", fontsize=13.5, fontweight="bold")
    out_path = out_dir / "panel_title.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


def build_all_panels(out_dir: Path = PANELS_OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = layout.PANEL_BOXES

    paths: dict[str, Path] = {}
    paths["A"] = panel_a_motion_schematic.build_panel(figsize=boxes["A"][2:4], out_dir=out_dir)
    paths["D"] = panel_d_contour_relative_stimulus.build_panel(figsize=boxes["D"][2:4], out_dir=out_dir)
    paths["G"] = panel_g_local_contour_detail.build_panel(figsize=boxes["G"][2:4], out_dir=out_dir)
    for letter in ["B", "C", "E", "F"]:
        paths[letter] = panel_bcef_path_bins.build_single_panel(letter, figsize=boxes[letter][2:4], out_dir=out_dir)
    paths["H"] = panel_g_rms_excursion.build_panel(out_dir=out_dir, figsize=boxes["H"][2:4])["pdf"]
    paths["I"] = panel_h_unwrapped_edge_coherence.build_panel(out_dir=out_dir, figsize=boxes["I"][2:4])["pdf"]
    paths["J"] = panel_j_match_advantage.build_panel(out_dir=out_dir, figsize=boxes["J"][2:4])["pdf"]
    return paths


def _place(writer: PdfWriter, base_page, source_pdf: Path, x_in: float, y_in_from_top: float, page_h_pt: float) -> None:
    reader = PdfReader(str(source_pdf))
    panel_page = reader.pages[0]
    actual_h_pt = float(panel_page.mediabox.height)
    tx = x_in * 72.0
    ty = page_h_pt - y_in_from_top * 72.0 - actual_h_pt
    base_page.merge_transformed_page(panel_page, Transformation().translate(tx=tx, ty=ty))


def compose(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels_v3"
    page_w_in, page_h_in = layout.PAGE_SIZE_IN
    page_w_pt, page_h_pt = page_w_in * 72.0, page_h_in * 72.0

    panel_paths = build_all_panels(panels_dir)
    title_path = build_title_panel(page_w_in, panels_dir)

    writer = PdfWriter()
    writer.add_blank_page(width=page_w_pt, height=page_h_pt)
    base_page = writer.pages[0]

    _place(writer, base_page, title_path, 0.0, 0.0, page_h_pt)
    for letter, (x_in, y_in, _w_in, _h_in) in layout.PANEL_BOXES.items():
        _place(writer, base_page, panel_paths[letter], x_in, y_in, page_h_pt)

    out_pdf = out_dir / "ssi_figure_v3.pdf"
    with open(out_pdf, "wb") as f:
        writer.write(f)

    provenance = {
        "figure": "ssi_figure_v3",
        "architecture": "composited: each panel is an independently rendered PDF, placed at a "
        "position measured from ssi_figure_v2_3.pdf (see panels/reference_layout_v3.py and "
        "panels/extract_reference_layout.py)",
        "page_size_in": [page_w_in, page_h_in],
        "panels": {
            letter: {
                "source_script": str(panel_paths[letter].name),
                "measured_box_in_x_y_w_h": list(layout.PANEL_BOXES[letter]),
            }
            for letter in layout.PANEL_BOXES
        },
        "output_pdf": str(out_pdf.relative_to(ROOT)),
    }
    provenance_path = out_dir / "ssi_figure_v3_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"pdf": out_pdf, "provenance_json": provenance_path}


def main() -> None:
    paths = compose()
    for key, path in paths.items():
        print(path)


if __name__ == "__main__":
    main()
