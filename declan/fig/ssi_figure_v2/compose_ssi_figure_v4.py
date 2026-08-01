#!/usr/bin/env python3
"""SSI figure v4: real compositing, not one shared matplotlib canvas.

Every displayed panel (A-H) is rendered as its own independent figure, sized from
PANEL_BOXES (panels/reference_layout_v3.py -- measured directly off
ssi_figure_v2_3.pdf, the Illustrator-fixed reference artwork), then placed
onto a blank page at that measured position via pypdf. No panel's drawing
code shares a coordinate tree with any other panel's, so one panel's
opaque axes background can no longer paint over a neighboring panel's text
-- the recurring failure mode in generate_ssi_figure_v2.py's nested
ax.inset_axes() approach.

Run: uv run python declan/fig/ssi_figure_v2/compose_ssi_figure_v4.py
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
from panels import panel_bcef_path_bins  # noqa: E402
from panels import panel_g_rms_excursion  # noqa: E402  (module name is historical; content moved into "G")
from panels import panel_h_unwrapped_edge_coherence  # noqa: E402  (displays as "I")
from panels import panel_header  # noqa: E402
from panels import panel_j_match_advantage  # noqa: E402  (displays as "J")
from panels import panel_k_patch_radius_alignment_slope  # noqa: E402

OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2"
PANELS_OUT_DIR = OUT_DIR / "panels_v4"
TITLE_TEXT = "Single-spike information and contour-relative FEM"
BCEF_PAIR_LABELS = {
    "BC": ("B", "C"),
    "EF": ("E", "F"),
}
V4_LAYOUT_BOXES = {
    # x, y-from-top, width, height in inches. These are hand-tuned for the
    # combined-panel v4 layout rather than inherited directly from the older
    # Illustrator reference boxes.
    "A": (0.0944, 0.1200, 5.6000, 3.9939),
    "BC": (5.7600, 0.1200, 2.5000, 3.9939),
    "D": (0.0944, 3.9250, 2.9000, 3.8800),
    "EF": (3.0444, 3.9250, 2.6000, 3.8800),
    "G": (5.6849, 3.9250, 2.5962, 3.8800),
    "I": (0.0944, 7.8656, 2.6704, 3.0306),
    "J": (2.9148, 7.8656, 2.6704, 3.0306),
    "K": (5.7352, 7.8656, 2.6704, 3.0306),
}
DISPLAY_SPECS = {
    "A": {
        "label": "A",
        "title": "FEMs sharpen spatial coding",
    },
    "BC": {
        "label": "B",
        "title": "Path length separates low- and\nhigh-SF benefit",
    },
    "D": {
        "label": "C",
        "title": "Local contours define the\nrelevant image axis",
    },
    "EF": {
        "label": "D",
        "title": "Contour alignment exposes a\nhigh-SF limit",
        "xlabel": "path length (arcmin; irrespective of\nspatial footprint)",
    },
    "G": {
        "label": "E",
        "title": "Across-contour spread limits\nhigh-SF benefit",
    },
    "I": {
        "label": "F",
        "title": "Real FEM spread is contour-aligned",
    },
    "J": {
        "label": "G",
        "title": "Contour-matched FEMs beat\nrotations for aligned high-SF units",
    },
    "K": {
        "label": "H",
        "title": "Edge following saturates near\nfoveal scale",
    },
}


def _union_box(*boxes: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def bcef_pair_boxes() -> dict[str, tuple[float, float, float, float]]:
    return {pair_key: V4_LAYOUT_BOXES[pair_key] for pair_key in BCEF_PAIR_LABELS}


def v4_placement_boxes() -> dict[str, tuple[float, float, float, float]]:
    """Final page placement boxes.

    The source panel identities still come from the measured reference layout,
    but v4 uses a deliberate grid tuned after B/C and E/F were consolidated.
    """
    return dict(V4_LAYOUT_BOXES)


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
    placement_boxes = v4_placement_boxes()

    paths: dict[str, Path] = {}
    paths["A"] = panel_a_motion_schematic.build_panel(
        figsize=placement_boxes["A"][2:4],
        out_dir=out_dir,
        panel_label=DISPLAY_SPECS["A"]["label"],
        panel_title=DISPLAY_SPECS["A"]["title"],
    )
    paths["D"] = panel_d_contour_relative_stimulus.build_panel(
        figsize=placement_boxes["D"][2:4],
        out_dir=out_dir,
        panel_label=DISPLAY_SPECS["D"]["label"],
        panel_title=DISPLAY_SPECS["D"]["title"],
    )
    for pair_key, letters in BCEF_PAIR_LABELS.items():
        display = DISPLAY_SPECS[pair_key]
        paths[pair_key] = panel_bcef_path_bins.build_pair_panel(
            letters,
            figsize=placement_boxes[pair_key][2:4],
            out_dir=out_dir,
            panel_label=display["label"],
            panel_title=display["title"],
            panel_subtitle=display.get("subtitle"),
            xlabel=display.get("xlabel"),
            ylabel_x=panel_header.MIDDLE_ROW_YLABEL_X if pair_key == "EF" else None,
            axes_box=(
                panel_bcef_path_bins.TOP_ROW_PAIR_AXES_BOX
                if pair_key == "BC"
                else panel_header.MIDDLE_ROW_AXES_BOX
            ),
            ylim_pad_low=0.055 if pair_key == "EF" else 0.12,
            ylim_pad_high=0.055 if pair_key == "EF" else 0.14,
            tight_pad=0.35 if pair_key == "BC" else 0.55,
            separate_header=(pair_key == "BC"),
        )
    display_g = DISPLAY_SPECS["G"]
    paths["G"] = panel_g_rms_excursion.build_panel(
        out_dir=out_dir,
        figsize=placement_boxes["G"][2:4],
        panel_label=display_g["label"],
        panel_title=display_g["title"],
    )["pdf"]
    paths["I"] = panel_h_unwrapped_edge_coherence.build_panel(
        out_dir=out_dir,
        figsize=placement_boxes["I"][2:4],
        label=DISPLAY_SPECS["I"]["label"],
        title=DISPLAY_SPECS["I"]["title"],
    )["pdf"]
    paths["J"] = panel_j_match_advantage.build_panel(
        out_dir=out_dir,
        figsize=placement_boxes["J"][2:4],
        label=DISPLAY_SPECS["J"]["label"],
        title=DISPLAY_SPECS["J"]["title"],
    )["pdf"]
    paths["K"] = panel_k_patch_radius_alignment_slope.build_panel(
        out_dir=out_dir,
        figsize=placement_boxes["K"][2:4],
        label=DISPLAY_SPECS["K"]["label"],
        title=DISPLAY_SPECS["K"]["title"],
    )["pdf"]
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

    writer = PdfWriter()
    writer.add_blank_page(width=page_w_pt, height=page_h_pt)
    base_page = writer.pages[0]

    placement_boxes = v4_placement_boxes()
    for key in ["A", "BC", "D", "EF", "G", "I", "J", "K"]:
        x_in, y_in, _w_in, _h_in = placement_boxes[key]
        _place(writer, base_page, panel_paths[key], x_in, y_in, page_h_pt)

    out_pdf = out_dir / "ssi_figure_v4.pdf"
    with open(out_pdf, "wb") as f:
        writer.write(f)

    provenance = {
        "figure": "ssi_figure_v4",
        "architecture": "composited: each panel is an independently rendered PDF, placed at a "
        "position in the hand-tuned v4 grid; source panel identities retain the measured "
        "ssi_figure_v2_3.pdf reference boxes for provenance (see panels/reference_layout_v3.py "
        "and panels/extract_reference_layout.py)",
        "page_size_in": [page_w_in, page_h_in],
        "display_panels": {
            display["label"]: {
                "title": display["title"],
                "subtitle": display.get("subtitle"),
                "source_key": key,
                "source_script": str(panel_paths[key].name),
                "placement_box_in_x_y_w_h": list(placement_boxes[key]),
                "combined_source_letters": list(BCEF_PAIR_LABELS[key]) if key in BCEF_PAIR_LABELS else None,
                "content_note": "formerly Panel H RMS-excursion dose curve" if key == "G" else None,
            }
            for key, display in DISPLAY_SPECS.items()
        },
        "source_layout_panels": {
            letter: {
                "measured_box_in_x_y_w_h": list(layout.PANEL_BOXES[letter]),
                "display_panel": (
                    "B"
                    if letter in {"B", "C"}
                    else "D"
                    if letter in {"E", "F"}
                    else DISPLAY_SPECS[letter]["label"]
                    if letter in DISPLAY_SPECS
                    else None
                ),
            }
            for letter in layout.PANEL_BOXES
        },
        "omitted_layout_boxes": {
            "H": {
                "measured_box_in_x_y_w_h": list(layout.PANEL_BOXES["H"]),
                "reason": "RMS-excursion content moved to the former G placement box after G's explainer was folded into D.",
            }
        },
        "new_layout_boxes": {
            key: {"placement_box_in_x_y_w_h": list(box)}
            for key, box in placement_boxes.items()
        },
        "combined_axis_groups": {
            pair_key: {
                "letters": list(letters),
                "placement_box_in_x_y_w_h": list(placement_boxes[pair_key]),
                "source_script": str(panel_paths[pair_key].name),
            }
            for pair_key, letters in BCEF_PAIR_LABELS.items()
        },
        "output_pdf": str(out_pdf.relative_to(ROOT)),
    }
    provenance_path = out_dir / "ssi_figure_v4_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"pdf": out_pdf, "provenance_json": provenance_path}


def main() -> None:
    paths = compose()
    for key, path in paths.items():
        print(path)


if __name__ == "__main__":
    main()
