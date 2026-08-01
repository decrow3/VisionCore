#!/usr/bin/env python3
"""One-time (but re-runnable) tool: measure each panel's real bounding box
straight from ssi_figure_v2_3.pdf -- the artwork you hand-fixed in
Illustrator -- instead of re-deriving panel positions from formulas.

ssi_figure_v2_3.pdf has no per-panel grouping left after the Illustrator
round-trip (everything sits in one generic "Layer 1"), so panel membership
has to be inferred. A plain "assign each element to its nearest panel-letter
label" clustering does NOT work here: A and D each span two columns with
their letter+title anchored at the far left, while their own content (icons,
images) reaches out much further right -- often physically closer to the
neighboring narrow panel's anchor than to their own. So instead this uses
the figure's known row/column structure (three rows; row 0 = A | B/C, row 1
= D | E/F | G, row 2 = H | I | J) to build a small grid of regions, with
each boundary computed from the measured anchor positions (not hardcoded
pixels) -- then assigns every element to a panel by containment in that
grid, and takes the padded union of assigned elements as the panel's real
box.

Writes reference_layout_v3.py (PANEL_BOXES: letter -> (x_in, y_in, w_in,
h_in), y measured from the page's TOP edge, matching how this session's
own PNG-crop diagnostics have been read all along).

Run: uv run python declan/fig/ssi_figure_v2/panels/extract_reference_layout.py
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[4]
REFERENCE_PDF = ROOT / "declan" / "fig" / "ssi_figure_v2" / "ssi_figure_v2_3.pdf"
OUT_PY = Path(__file__).resolve().parent / "reference_layout_v3.py"

PANEL_LETTERS = list("ABCDEFGHIJ")
PAGE_TITLE_TEXT = "Single-spike information and contour-relative FEM"
PAD_PT = 3.0
# A/D each span two columns' worth of content (icons/images reach far past
# their own far-left letter anchor); this is the margin subtracted from the
# next column's anchor x0 to keep that content from crossing the boundary.
SPAN_BUFFER_PT = 35.0


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _find_anchor(letter: str, spans: list[dict]) -> dict:
    """The one text span that IS this panel's letter label.

    Two title styles exist in the source figure: letter and title as two
    separate Text artists (A/D/B/C/E/F -- draw_panel_header /
    _draw_two_color_title), or letter+title baked into one combined string
    (G/H/I/J -- set_panel_title). Match both: an exact "L" span, or a span
    starting with "L  " (two spaces, the f"{label}  {title}" convention).
    """
    candidates = []
    for span in spans:
        text = span["text"]
        stripped = text.strip()
        if stripped == letter:
            candidates.append(span)
        elif text.startswith(f"{letter}  ") and span["size"] >= 6.5:
            candidates.append(span)
    if not candidates:
        raise ValueError(f"No anchor span found for panel {letter!r}")
    candidates.sort(key=lambda s: -s["size"])
    return candidates[0]


def _find_gap_split(
    elements: list[dict],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    second_anchor_y: float,
    fallback: float,
) -> float:
    """Where two panels stack vertically in the same column (B/C, E/F), the
    naive midpoint-between-titles split breaks once one panel grows taller
    than the other -- its own bottom content (e.g. an x-axis tick label)
    ends up past the midpoint, on the wrong side.

    Picking the SINGLE BIGGEST empty gap in the column doesn't work either:
    every normal plot has its own internal gap between the y-tick numbers
    and the x-tick/xlabel row below them, and that gap is often bigger than
    the (tightly packed) gap between two neighboring panels. So instead:
    merge all elements in this column into vertical intervals, find which
    merged interval the second panel's own title anchor falls into, and
    return the midpoint of the gap immediately before THAT interval -- the
    boundary right above where the second panel's title starts, regardless
    of whatever bigger gaps exist further up inside the first panel.
    """
    x0, x1 = x_range
    y_lo, y_hi = y_range
    intervals = []
    for el in elements:
        bx0, by0, bx1, by1 = el["bbox"]
        cx = (bx0 + bx1) / 2.0
        if x0 <= cx < x1 and y_lo <= by0 < y_hi:
            intervals.append((by0, by1))
    if not intervals:
        return fallback
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for i, (s, e) in enumerate(merged):
        if s <= second_anchor_y <= e or s > second_anchor_y:
            if i == 0:
                return fallback
            prev_end = merged[i - 1][1]
            return (prev_end + s) / 2.0
    return fallback


def _find_gap_split_x(
    elements: list[dict],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    second_anchor_x: float,
    fallback: float,
) -> float:
    """Column-direction twin of `_find_gap_split`: where two panels sit
    side by side in the same row (H/I/J), a plain midpoint-between-anchors
    split breaks the same way the row splits did -- one panel's title or
    legend can legitimately reach further toward its neighbor than the
    other's letter anchor does, so the naive midpoint cuts through real
    content on one or both sides even though the actual ink never
    collides (they're just at different heights within the row). Merge
    elements into horizontal intervals, find the interval the second
    panel's own anchor x falls into, and split at the gap immediately
    before it.
    """
    x_lo, x_hi = x_range
    y0, y1 = y_range
    intervals = []
    for el in elements:
        bx0, by0, bx1, by1 = el["bbox"]
        cy = (by0 + by1) / 2.0
        if y0 <= cy < y1 and x_lo <= bx0 < x_hi:
            intervals.append((bx0, bx1))
    if not intervals:
        return fallback
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for i, (s, e) in enumerate(merged):
        if s <= second_anchor_x <= e or s > second_anchor_x:
            if i == 0:
                return fallback
            prev_end = merged[i - 1][1]
            return (prev_end + s) / 2.0
    return fallback


def _region_grid(
    anchors: dict[str, dict], elements: list[dict], page_w: float, page_h: float
) -> dict[str, tuple[float, float, float, float]]:
    """Panel -> (x0, y0, x1, y1) assignment region, from the figure's known
    3-row grid (row0: A | B/C: row1: D | E/F | G; row2: H | I | J)."""

    def y0(letter: str) -> float:
        return anchors[letter]["bbox"][1]

    def x0(letter: str) -> float:
        return anchors[letter]["bbox"][0]

    # Row boundaries also use gap-finding, not a plain title-anchor midpoint:
    # C (row 0) or F (row 1) growing taller can push their real bottom
    # content past a fixed midpoint and into the row below's search window,
    # exactly like the B/C same-column case -- search a generously widened
    # window across the full page width so the true empty gap is found
    # regardless of which side grew.
    row0_bottom = _find_gap_split(elements, (0.0, page_w), (y0("C"), y0("D") + 100.0), y0("D"), (y0("C") + y0("D")) / 2.0)
    row1_bottom = _find_gap_split(elements, (0.0, page_w), (y0("F"), y0("H") + 100.0), y0("H"), (y0("F") + y0("H")) / 2.0)

    split_row0 = x0("B") - SPAN_BUFFER_PT
    split_row1a = x0("E") - SPAN_BUFFER_PT
    split_row1b = x0("G") - SPAN_BUFFER_PT
    bc_split = _find_gap_split(elements, (split_row0, page_w), (y0("B"), y0("C") + 100.0), y0("C"), (y0("B") + y0("C")) / 2.0)
    ef_split = _find_gap_split(elements, (split_row1a, split_row1b), (y0("E"), y0("F") + 100.0), y0("F"), (y0("E") + y0("F")) / 2.0)
    split_row2a = _find_gap_split_x(
        elements, (0.0, page_w), (row1_bottom, page_h), x0("I"), (x0("H") + x0("I")) / 2.0
    )
    split_row2b = _find_gap_split_x(
        elements, (split_row2a, page_w), (row1_bottom, page_h), x0("J"), (x0("I") + x0("J")) / 2.0
    )

    return {
        "A": (0.0, 0.0, split_row0, row0_bottom),
        "B": (split_row0, 0.0, page_w, bc_split),
        "C": (split_row0, bc_split, page_w, row0_bottom),
        "D": (0.0, row0_bottom, split_row1a, row1_bottom),
        "E": (split_row1a, row0_bottom, split_row1b, ef_split),
        "F": (split_row1a, ef_split, split_row1b, row1_bottom),
        "G": (split_row1b, row0_bottom, page_w, row1_bottom),
        "H": (0.0, row1_bottom, split_row2a, page_h),
        "I": (split_row2a, row1_bottom, split_row2b, page_h),
        "J": (split_row2b, row1_bottom, page_w, page_h),
    }


def extract_panel_boxes(pdf_path: Path = REFERENCE_PDF) -> tuple[dict[str, tuple[float, float, float, float]], tuple[float, float]]:
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    page_w_pt, page_h_pt = page.rect.width, page.rect.height

    text_spans = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                text_spans.append({"text": span["text"], "bbox": tuple(span["bbox"]), "size": span["size"]})

    image_infos = [{"text": "<image>", "bbox": tuple(im["bbox"])} for im in page.get_image_info(xrefs=True)]

    anchors = {letter: _find_anchor(letter, text_spans) for letter in PANEL_LETTERS}
    elements = [s for s in text_spans if s["text"] != PAGE_TITLE_TEXT] + image_infos
    regions = _region_grid(anchors, elements, page_w_pt, page_h_pt)
    assigned: dict[str, list[tuple[float, float, float, float]]] = {letter: [anchors[letter]["bbox"]] for letter in PANEL_LETTERS}
    for el in elements:
        cx, cy = _bbox_center(el["bbox"])
        for letter, (rx0, ry0, rx1, ry1) in regions.items():
            if rx0 <= cx < rx1 and ry0 <= cy < ry1:
                assigned[letter].append(el["bbox"])
                break
        else:
            # Shouldn't happen (regions tile the full page) -- fall back to
            # nearest region center so nothing silently vanishes.
            def region_center(box):
                return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

            best = min(regions, key=lambda letter: (cx - region_center(regions[letter])[0]) ** 2 + (cy - region_center(regions[letter])[1]) ** 2)
            assigned[best].append(el["bbox"])

    boxes_pt: dict[str, tuple[float, float, float, float]] = {}
    for letter, bboxes in assigned.items():
        rx0, ry0, rx1, ry1 = regions[letter]
        # Clamp to the panel's own assigned region: PAD_PT is meant to give
        # a little breathing room beyond the panel's own extracted content,
        # not to eat into a neighbor's region -- when two panels' real
        # content comes within PAD_PT*2 of each other (e.g. H's title and
        # I's legend, ~2pt apart), padding both sides independently would
        # otherwise create a spurious overlap between two boxes that are
        # each individually rendered as opaque rectangles.
        x0 = max(min(b[0] for b in bboxes) - PAD_PT, rx0)
        y0 = max(min(b[1] for b in bboxes) - PAD_PT, ry0)
        x1 = min(max(b[2] for b in bboxes) + PAD_PT, rx1)
        y1 = min(max(b[3] for b in bboxes) + PAD_PT, ry1)
        boxes_pt[letter] = (x0, y0, x1 - x0, y1 - y0)
        print(f"{letter}: {len(bboxes)} elements, bbox_pt=({x0:.1f},{y0:.1f},{x1-x0:.1f},{y1-y0:.1f})")

    doc.close()
    return boxes_pt, (page_w_pt, page_h_pt)


def apply_manual_aesthetic_adjustments(
    boxes_pt: dict[str, tuple[float, float, float, float]], page_w_pt: float
) -> dict[str, tuple[float, float, float, float]]:
    """Deliberate, hand-tuned deviations from the extracted reference boxes.

    ssi_figure_v2_3.pdf is a hand-edited Illustrator artifact, not a
    from-first-principles design -- matching its measured proportions
    exactly produced a needlessly cramped J (title/legend/y-axis all fought
    for ~1.87in of width, the tightest column on the page) while leaving a
    0.29in strip of unused margin between J's right edge and the page edge.
    Once collisions are fixed, further tuning should target v3's own
    legibility, not bit-for-bit box parity with the reference -- see
    compose_ssi_figure_v3.py's docstring.

    Widens J by (a) reclaiming the unused right-page margin and (b) taking
    a modest, conservative slice off I (which had more headroom -- its
    bbox_inches="tight" build only needed 0.12in beyond its declared box,
    vs J needing 0.20in even after shortening its title/ylabel).
    """
    boxes_pt = dict(boxes_pt)
    h_x, h_y, h_w, h_h = boxes_pt["H"]
    i_x, i_y, i_w, i_h = boxes_pt["I"]
    j_x, j_y, j_w, j_h = boxes_pt["J"]

    i_give_pt = 0.20 * 72.0  # inches -> pt, taken off I's right edge
    right_margin_pt = 0.08 * 72.0  # small breathing room to the page edge

    new_i_w = i_w - i_give_pt
    new_j_x = i_x + new_i_w
    new_j_w = (page_w_pt - right_margin_pt) - new_j_x

    boxes_pt["I"] = (i_x, i_y, new_i_w, i_h)
    boxes_pt["J"] = (new_j_x, j_y, new_j_w, j_h)
    return boxes_pt


def write_layout_module(boxes_pt: dict[str, tuple[float, float, float, float]], page_size_pt: tuple[float, float]) -> None:
    page_w_in = page_size_pt[0] / 72.0
    page_h_in = page_size_pt[1] / 72.0
    lines = [
        '"""Panel bounding boxes measured directly from ssi_figure_v2_3.pdf',
        "(the Illustrator-fixed reference artwork) -- see extract_reference_layout.py.",
        "",
        "PANEL_BOXES[letter] = (x_in, y_in, w_in, h_in), inches, y measured from",
        "the page's TOP edge (matches how this session reads PNG crops; convert to",
        "PDF-native bottom-up points when compositing: ty = page_h_pt - y_in*72 - h_in*72).",
        "",
        "Regenerate with:",
        "    uv run python declan/fig/ssi_figure_v2/panels/extract_reference_layout.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"PAGE_SIZE_IN = ({page_w_in:.4f}, {page_h_in:.4f})",
        "",
        "PANEL_BOXES: dict[str, tuple[float, float, float, float]] = {",
    ]
    for letter in PANEL_LETTERS:
        x_pt, y_pt, w_pt, h_pt = boxes_pt[letter]
        lines.append(
            f'    "{letter}": ({x_pt/72.0:.4f}, {y_pt/72.0:.4f}, {w_pt/72.0:.4f}, {h_pt/72.0:.4f}),'
        )
    lines.append("}")
    lines.append("")
    OUT_PY.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PY}")


def main() -> None:
    boxes_pt, page_size_pt = extract_panel_boxes()
    boxes_pt = apply_manual_aesthetic_adjustments(boxes_pt, page_size_pt[0])
    write_layout_module(boxes_pt, page_size_pt)


if __name__ == "__main__":
    main()
