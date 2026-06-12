"""
Compose figure 1 into a single SVG, then export PDF and PNG via cairosvg.

Layout:
    Row 1 (3 in tall):  A | B gaze distribution + C RF map
    Rows 2-4:           D-E-F population block | G pair above H-I

Only panel A is an external SVG (Illustrator schematic); the remaining panels
(C, D, F-I) are rendered together inside one matplotlib figure with nested
subfigures so spacing and labels stay coherent. Panel B is rendered as a
separate matplotlib SVG and composited as an inset over A.

Usage:
    uv run ryan/fig1/generate_fig1.py [-r] [--recalc-c] [--recalc-d] [--recalc-f]
"""

import argparse
from io import BytesIO
from pathlib import Path
import re
import matplotlib.pyplot as plt
import svgutils.transform as sg
import cairosvg
from PIL import Image, ImageEnhance, ImageOps

from VisionCore.paths import FIGURES_DIR
from generate_fig1b import plot_panel_b
from generate_fig1c import plot_panel_c
from generate_fig1d import (
    plot_panel_d_roi, plot_panel_d_gaze, plot_panel_ef, _add_block_label,
    SUBJECT as EXAMPLE_SUBJECT, DATE as EXAMPLE_DATE, DEFAULT_CELL,
)
from generate_fig1f import plot_panel_f

HERE = Path(__file__).resolve().parent
FIG_DIR = FIGURES_DIR / "fig1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Layout in inches.
ROW_HEIGHT_IN = 3.0
PANEL_C_W_IN = 2.0
PANEL_D_W_IN = 2.0
PANEL_A_W_IN = 1.5 * PANEL_C_W_IN
PANEL_A_BASE_X_IN = 0.20
PANEL_A_X_IN = PANEL_A_BASE_X_IN + 0.12
PANEL_A_Y_IN = 0.06
PAD_IN = 0.25
PANEL_B_INSET_W_IN = 1.25
PANEL_B_INSET_H_IN = 1.25
PANEL_B_INSET_X_IN = PANEL_A_X_IN + PANEL_A_W_IN - PANEL_B_INSET_W_IN + 0.00
PANEL_B_INSET_Y_IN = PANEL_A_Y_IN + ROW_HEIGHT_IN - PANEL_B_INSET_H_IN - 0.27
PANEL_A_DPIEG_W_IN = 0.82
PANEL_A_DPIEG_H_IN = 0.82
PANEL_A_DPIEG_X_IN = PANEL_B_INSET_X_IN
PANEL_A_DPIEG_Y_IN = PANEL_B_INSET_Y_IN + 0.29

# Second + third row block (D-F and G-I, each rendered as its own subfigure).
BLOCK_HEIGHT_IN = 6.0

# Total figure size.
TOTAL_W_IN = PANEL_A_W_IN + PANEL_C_W_IN + PANEL_D_W_IN + 2 * PAD_IN
TOTAL_H_IN = ROW_HEIGHT_IN + BLOCK_HEIGHT_IN
CANVAS_H_IN = TOTAL_H_IN - 1.02

# Matplotlib region spans the full width; the top-left cell is left empty
# so panel A (SVG schematic) can be composited over it.
A_RESERVE_W_IN = PANEL_A_W_IN + 0.34

# 1 inch = 96 SVG user units.
PPI = 96.0
MAIN_PANEL_Y_SHIFT = 15.0 / (TOTAL_H_IN * PPI)
TOP_ROW_EXTRA_Y_SHIFT = 97.0 / (TOTAL_H_IN * PPI)
RIGHT_COLUMN_EXTRA_Y_SHIFT = 12.0 / (TOTAL_H_IN * PPI)
PANEL_B_LABEL_X_IN = 3.45
PANEL_C_LABEL_X_IN = 5.52

PANEL_LABEL_FONTSIZE_PT = 16
# svgutils sizes in SVG user units (px). matplotlib renders 16pt @ 96 DPI as
# 16 * 96/72 ≈ 21.33 px, so match that for the A label.
PANEL_LABEL_FONTSIZE_PX = PANEL_LABEL_FONTSIZE_PT * 96.0 / 72.0
PANEL_A_LABEL_FONTSIZE_PX = 20.0

FIG1_CAPTION = """# Figure 1

Population recordings in the foveal representation in marmoset V1 show strong dependence on gaze during fixation.

(A) Schematic of the experimental paradigm. A head-fixed marmoset was trained to fixate a rapidly updating sequence of flashed images. Electrical activity was recorded using laminar probes from the lateral surface of V1.

(B) The distribution of gaze during an example experiment shows tight oculomotor control by the marmosets. A majority of gaze positions were within 0.5 deg during the experiment.

(C) Receptive field locations for each experiment. Blue represents RFs of units recorded from monkey A, while green represents RFs from monkey L. The bold black contour marks the example unit shown in (G).

(D) Gaze position over time for eight representative trials colored in red and blue. The eye position traces are highly self-similar within groups, but not between groups.

(E) Population rasters for the eight representative trials from (D) and (F). Colored arrows connect the blue and red gaze traces in (D) to their matching raster groups. There is clear similarity between individual trials with similar gaze traces, and large differences at the population level between trials with dissimilar gaze, even though the difference in position is just fractions of a degree.

(F) Spiking activity averaged across all recorded units for the four red trials and four blue trials over the same time course.

(G) Left: A grayscale example STA for a single foveal unit with clear Gabor-like structure. Right: The position of gaze measured across all trials in a 50 ms bin. Points are colored according to their projection onto the line of maximal sensitivity, which is orthogonal to the subunits under a linear model of the unit's response.

(H) Trial rasters sorted by the position of gaze projected onto the line of maximum sensitivity. Sorting occurs on the same bins as in (I). Clear structure emerges when sorting by eye position.

(I) The peristimulus time histogram of responses for the example unit shown in (G). The gray trace depicts the overall PSTH across all trials, while the blue and red lines represent the PSTH of trials with positive or negative projections onto the line of maximum sensitivity. Since eye position is not steady throughout individual trials, the projection is computed on the average gaze position in 50 ms bins.
"""


def _write_caption_files():
    """Write caption/legend text beside the generated figure files."""
    for name in ("fig1_caption.md", "fig1_legend.md"):
        (FIG_DIR / name).write_text(FIG1_CAPTION, encoding="utf-8")


def _translate_svg_axes(svg_path, markers, dy_px):
    """Move selected matplotlib axes groups in the exported SVG."""
    text = Path(svg_path).read_text(encoding="utf-8")
    parts = re.split(r"(?=^[ \t]*<g id=\"axes_\d+\"[^>]*>)", text,
                     flags=re.MULTILINE)
    moved = []
    for part in parts:
        if part.lstrip().startswith('<g id="axes_') and any(m in part for m in markers):
            part = part.replace(
                ">",
                f' transform="translate(0,{dy_px:g})">',
                1,
            )
        moved.append(part)
    Path(svg_path).write_text("".join(moved), encoding="utf-8")


def _render_main_svg(out_path, recalc_c=False, recalc_d=False, recalc_f=False):
    """Render B-I together as a single full-width matplotlib
    figure. The top-left cell is left empty for panel A (composited later)."""
    fig = plt.figure(
        figsize=(TOTAL_W_IN, TOTAL_H_IN),
        layout="constrained",
    )
    fig.get_layout_engine().set(
        w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0,
    )

    top, bottom = fig.subfigures(
        2, 1, height_ratios=[ROW_HEIGHT_IN, BLOCK_HEIGHT_IN], hspace=-0.14,
    )
    _sub_a_blank, sub_top_right = top.subfigures(
        1, 2,
        width_ratios=[A_RESERVE_W_IN, PANEL_C_W_IN + PANEL_D_W_IN],
        wspace=0.0,
    )
    sub_b, sub_c = sub_top_right.subfigures(
        1, 2, width_ratios=[1.0, 1.0], wspace=0.02,
    )
    sub_pop, sub_right = bottom.subfigures(
        1, 2, width_ratios=[3.45, 3.55], wspace=0.0,
    )

    right_gs = sub_right.add_gridspec(
        2, 1, height_ratios=[1.45, 4.55], hspace=-0.02,
    )
    d_gs = right_gs[0].subgridspec(1, 2, wspace=0.01)
    ax_d = sub_right.add_subplot(d_gs[0])
    _, _, roi_extent = plot_panel_d_roi(ax=ax_d, refresh=recalc_d, panel_letter="G")
    ax_d_gaze = sub_right.add_subplot(d_gs[1])
    plot_panel_d_gaze(ax=ax_d_gaze, refresh=recalc_d)

    ax_b = sub_b.add_subplot(1, 1, 1)
    plot_panel_b(ax=ax_b)

    ax_c = sub_c.add_subplot(1, 1, 1)
    highlight_session = f"{EXAMPLE_SUBJECT}_{EXAMPLE_DATE}"
    plot_panel_c(
        ax=ax_c, refresh=recalc_c, roi_extent=roi_extent,
        highlight_session=highlight_session, highlight_cell=DEFAULT_CELL,
    )

    _, pop_axes = plot_panel_f(
        fig=sub_pop, refresh=recalc_f, panel_letters=("D", "F", "E"),
        bottom_pad=0.45,
    )
    ef_subfig = sub_right.add_subfigure(right_gs[1])
    plot_panel_ef(
        fig=ef_subfig, refresh=recalc_d, panel_letters=("H", "I"),
        vertical_pad=(0.05, 0.52), raster_height=2.05,
    )

    # Constrained layout gives the gaze traces a different x-span from the
    # raster/PSTH because their left-side labels differ. Align the shared time
    # axes after layout has solved the panel positions.
    fig.canvas.draw()
    bottom_shift = 0.088
    for axes in sub_pop.axes:
        pos = axes.get_position()
        axes.set_in_layout(False)
        axes.set_position([
            pos.x0, pos.y0 + bottom_shift + MAIN_PANEL_Y_SHIFT,
            pos.width, pos.height,
        ])

    for axes in sub_right.axes:
        pos = axes.get_position()
        axes.set_in_layout(False)
        axes.set_position([
            pos.x0,
            pos.y0 + bottom_shift + MAIN_PANEL_Y_SHIFT + RIGHT_COLUMN_EXTRA_Y_SHIFT,
            pos.width,
            pos.height,
        ])

    for axes in sub_b.axes + sub_c.axes:
        pos = axes.get_position()
        axes.set_in_layout(False)
        axes.set_position([
            pos.x0,
            pos.y0 + MAIN_PANEL_Y_SHIFT + TOP_ROW_EXTRA_Y_SHIFT,
            pos.width,
            pos.height,
        ])

    ref = pop_axes["raster"].get_position()
    gaze_v_pos = pop_axes["gaze_v"].get_position()
    target_gap = 0.068
    y_shift = (ref.y1 + target_gap) - gaze_v_pos.y0
    for ax in (pop_axes["gaze_h"], pop_axes["gaze_v"]):
        pos = ax.get_position()
        ax.set_in_layout(False)
        ax.set_position([ref.x0, pos.y0 + y_shift, ref.width, pos.height])

    fig.savefig(out_path, dpi=400)
    _translate_svg_axes(
        out_path,
        markers=("Trials, by gaze along oriented axis", "Gaze proj.", "<!-- I -->"),
        dy_px=-44.0,
    )
    plt.close(fig)


def _render_panel_b_svg(out_path):
    fig, ax = plt.subplots(
        figsize=(PANEL_B_INSET_W_IN, PANEL_B_INSET_H_IN),
        layout="constrained",
    )
    fig.get_layout_engine().set(
        w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0,
    )
    plot_panel_b(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def _panel_a_legend_overlay():
    """Cover the old panel A legend area while B carries the constraint legend."""
    x_shift = (PANEL_A_X_IN - PANEL_A_BASE_X_IN) * PPI
    y_shift = PANEL_A_Y_IN * PPI
    cover_x = 82 + x_shift
    cover_y = 170 + y_shift
    return sg.fromstring(f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{TOTAL_W_IN * PPI}" height="{CANVAS_H_IN * PPI}"
         viewBox="0 0 {TOTAL_W_IN * PPI} {CANVAS_H_IN * PPI}">
      <g>
        <rect x="{cover_x}" y="{cover_y}" width="185" height="72" fill="#ffffff"/>
      </g>
    </svg>
    """).getroot()


def _panel_a_inset_image():
    placeholder_w = PANEL_A_DPIEG_W_IN * PPI
    placeholder_h = PANEL_A_DPIEG_H_IN * PPI
    with Image.open(HERE / "dpieg.png") as raw:
        im = ImageOps.grayscale(raw)
        im = ImageOps.autocontrast(im, cutoff=0.5)
        im = ImageEnhance.Contrast(im).enhance(1.75)
        im = ImageEnhance.Sharpness(im).enhance(1.15)
        buf = BytesIO()
        im.save(buf, format="PNG")
    buf.seek(0)
    img = sg.ImageElement(buf, placeholder_w, placeholder_h)
    img.moveto(PANEL_A_DPIEG_X_IN * PPI, PANEL_A_DPIEG_Y_IN * PPI)
    return img


def _panel_a_inset_border():
    placeholder_x = PANEL_A_DPIEG_X_IN * PPI
    placeholder_y = PANEL_A_DPIEG_Y_IN * PPI
    placeholder_w = PANEL_A_DPIEG_W_IN * PPI
    placeholder_h = PANEL_A_DPIEG_H_IN * PPI
    return sg.fromstring(f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{TOTAL_W_IN * PPI}" height="{CANVAS_H_IN * PPI}"
         viewBox="0 0 {TOTAL_W_IN * PPI} {CANVAS_H_IN * PPI}">
      <rect x="{placeholder_x}" y="{placeholder_y}"
            width="{placeholder_w}" height="{placeholder_h}"
            fill="none" stroke="#201c1d" stroke-width="1"/>
    </svg>
    """).getroot()


def compose(recalc_c=False, recalc_d=False, recalc_f=False):
    main_svg = FIG_DIR / "_fig1_main.svg"
    _render_main_svg(main_svg, recalc_c=recalc_c, recalc_d=recalc_d, recalc_f=recalc_f)

    panel_a_path = HERE / "fig1a.svg"

    fig = sg.SVGFigure(f"{TOTAL_W_IN}in", f"{CANVAS_H_IN}in")
    fig.root.set("viewBox", f"0 0 {TOTAL_W_IN * PPI} {CANVAS_H_IN * PPI}")

    def _load_and_place(path, x_in, y_in, target_w_in, target_h_in):
        f = sg.fromfile(str(path))
        root = f.getroot()
        vb_w, vb_h = _viewbox_size(f.root)
        sx = (target_w_in * PPI) / vb_w
        sy = (target_h_in * PPI) / vb_h
        scale = min(sx, sy)
        root.moveto(x_in * PPI, y_in * PPI, scale_x=scale)
        return root

    main = _load_and_place(main_svg, 0.0, 0.0, TOTAL_W_IN, TOTAL_H_IN)
    panel_a = _load_and_place(panel_a_path, PANEL_A_X_IN, PANEL_A_Y_IN,
                              PANEL_A_W_IN, ROW_HEIGHT_IN)

    label_a = sg.TextElement(
        0.05 * PPI, 0.25 * PPI, "A",
        size=PANEL_A_LABEL_FONTSIZE_PX, weight="bold", font="DejaVu Sans",
    )
    label_b = sg.TextElement(
        PANEL_B_LABEL_X_IN * PPI, 0.25 * PPI, "B",
        size=PANEL_A_LABEL_FONTSIZE_PX, weight="bold", font="DejaVu Sans",
    )
    label_c = sg.TextElement(
        PANEL_C_LABEL_X_IN * PPI, 0.25 * PPI, "C",
        size=PANEL_A_LABEL_FONTSIZE_PX, weight="bold", font="DejaVu Sans",
    )
    panel_a_inset = _panel_a_inset_image()
    panel_a_legend = _panel_a_legend_overlay()
    panel_a_border = _panel_a_inset_border()

    fig.append([
        main, panel_a, label_a, label_b, label_c,
        panel_a_legend, panel_a_inset, panel_a_border,
    ])

    out_svg = FIG_DIR / "fig1.svg"
    fig.save(str(out_svg))

    out_pdf = FIG_DIR / "fig1.pdf"
    out_png = FIG_DIR / "fig1.png"
    cairosvg.svg2pdf(url=str(out_svg), write_to=str(out_pdf))
    cairosvg.svg2png(url=str(out_svg), write_to=str(out_png), dpi=300)
    _write_caption_files()

    print(f"Saved {out_svg}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"Saved {FIG_DIR / 'fig1_caption.md'}")
    print(f"Saved {FIG_DIR / 'fig1_legend.md'}")


def _viewbox_size(root_element):
    vb = root_element.get("viewBox") or root_element.get("viewbox")
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    return (_to_user_units(root_element.get("width")),
            _to_user_units(root_element.get("height")))


def _to_user_units(value):
    if value is None:
        return 1.0
    s = str(value).strip()
    units = {"in": 96.0, "cm": 96.0 / 2.54, "mm": 96.0 / 25.4, "pt": 96.0 / 72.0,
             "pc": 96.0 / 6.0, "px": 1.0}
    for u, factor in units.items():
        if s.endswith(u):
            return float(s[: -len(u)]) * factor
    return float(s)


def _parse_args():
    p = argparse.ArgumentParser(description="Compose figure 1.")
    p.add_argument("-r", "--recalc", action="store_true",
                   help="Force recalc of all cached panels (C, D, F).")
    p.add_argument("--recalc-c", action="store_true", help="Force recalc of panel C.")
    p.add_argument("--recalc-d", action="store_true", help="Force recalc of panels D-F.")
    p.add_argument("--recalc-f", action="store_true", help="Force recalc of panels G-I.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compose(
        recalc_c=args.recalc or args.recalc_c,
        recalc_d=args.recalc or args.recalc_d,
        recalc_f=args.recalc or args.recalc_f,
    )
