"""
Compose figure 1 into a single SVG, then export PDF and PNG via cairosvg.

Layout:
    Row 1 (3 in tall):  A (+ gaze inset) | B
    Rows 2-4:           C-D-E population block | F pair above G-H

Only panel A is an external SVG (Illustrator schematic); the remaining panels
(C, D, F-I) are rendered together inside one matplotlib figure with nested
subfigures so spacing and labels stay coherent. Panel B is rendered as a
separate matplotlib SVG and composited as an inset over A.

Usage:
    uv run ryan/fig1/generate_fig1.py [-r] [--recalc-c] [--recalc-d] [--recalc-f]
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import svgutils.transform as sg
import cairosvg

from VisionCore.paths import FIGURES_DIR
from generate_fig1b import plot_panel_b
from generate_fig1c import plot_panel_c
from generate_fig1d import (
    plot_panel_d_roi, plot_panel_d_gaze, plot_panel_ef, _add_block_label,
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

# Second + third row block (D-F and G-I, each rendered as its own subfigure).
BLOCK_HEIGHT_IN = 6.0

# Total figure size.
TOTAL_W_IN = PANEL_A_W_IN + PANEL_C_W_IN + PANEL_D_W_IN + 2 * PAD_IN
TOTAL_H_IN = ROW_HEIGHT_IN + BLOCK_HEIGHT_IN
CANVAS_H_IN = TOTAL_H_IN - 0.52

# Matplotlib region spans the full width; the top-left cell is left empty
# so panel A (SVG schematic) can be composited over it.
A_RESERVE_W_IN = PANEL_A_W_IN + 0.70

# 1 inch = 96 SVG user units.
PPI = 96.0

PANEL_LABEL_FONTSIZE_PT = 16
# svgutils sizes in SVG user units (px). matplotlib renders 16pt @ 96 DPI as
# 16 * 96/72 ≈ 21.33 px, so match that for the A label.
PANEL_LABEL_FONTSIZE_PX = PANEL_LABEL_FONTSIZE_PT * 96.0 / 72.0
PANEL_A_LABEL_FONTSIZE_PX = 20.0

FIG1_CAPTION = """# Figure 1

Population recordings in the foveal representation in marmoset V1 show strong dependence on gaze during fixation.

(A) Schematic of the experimental paradigm. A head-fixed marmoset was trained to fixate a rapidly updating sequence of flashed images. Electrical activity was recorded using laminar probes from the lateral surface of V1.

Inset: The distribution of gaze during an example experiment shows tight oculomotor control by the marmosets. A majority of gaze positions were within 0.5 deg during the experiment.

(B) Receptive field locations for each experiment. Blue represents RFs of units recorded from monkey A, while green represents RFs from monkey L.

(C) Gaze position over time for eight representative trials colored in red and blue. The eye position traces are highly self-similar within groups, but not between groups.

(D) Population rasters for the eight representative trials from (C) and (E). There is clear similarity between individual trials with similar gaze traces, and large differences at the population level between trials with dissimilar gaze, even though the difference in position is just fractions of a degree.

(E) Spiking activity averaged across all recorded units for the four red trials and four blue trials over the same time course.

(F) Left: An example STA for a single foveal unit with clear Gabor-like structure. Right: The position of gaze measured across all trials in a 50 ms bin. Points are colored according to their projection onto the line of maximal sensitivity, which is orthogonal to the subunits under a linear model of the unit's response.

(G) Trial rasters sorted by the position of gaze projected onto the line of maximum sensitivity. Sorting occurs on the same bins as in (H). Clear structure emerges when sorting by eye position.

(H) The peristimulus time histogram of responses for the example unit shown in (F). The gray trace depicts the overall PSTH across all trials, while the blue and red lines represent the PSTH of trials with positive or negative projections onto the line of maximum sensitivity. Since eye position is not steady throughout individual trials, the projection is computed on the average gaze position in 50 ms bins.
"""


def _write_caption_files():
    """Write caption/legend text beside the generated figure files."""
    for name in ("fig1_caption.md", "fig1_legend.md"):
        (FIG_DIR / name).write_text(FIG1_CAPTION, encoding="utf-8")


def _render_main_svg(out_path, recalc_c=False, recalc_d=False, recalc_f=False):
    """Render B-H together as a single full-width matplotlib
    figure. The top-left cell is left empty for panel A (composited later)."""
    fig = plt.figure(
        figsize=(TOTAL_W_IN, TOTAL_H_IN),
        layout="constrained",
    )
    fig.get_layout_engine().set(
        w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0,
    )

    top, bottom = fig.subfigures(
        2, 1, height_ratios=[ROW_HEIGHT_IN, BLOCK_HEIGHT_IN], hspace=-0.02,
    )
    _sub_a_blank, sub_c = top.subfigures(
        1, 2,
        width_ratios=[A_RESERVE_W_IN, PANEL_C_W_IN + PANEL_D_W_IN],
        wspace=0.0,
    )
    sub_pop, sub_right = bottom.subfigures(
        1, 2, width_ratios=[3.45, 3.55], wspace=0.0,
    )

    right_gs = sub_right.add_gridspec(
        2, 1, height_ratios=[1.75, 4.25], hspace=0.03,
    )
    d_gs = right_gs[0].subgridspec(1, 2, wspace=0.01)
    ax_d = sub_right.add_subplot(d_gs[0])
    _, _, roi_extent = plot_panel_d_roi(ax=ax_d, refresh=recalc_d, panel_letter="F")
    ax_d_gaze = sub_right.add_subplot(d_gs[1])
    plot_panel_d_gaze(ax=ax_d_gaze, refresh=recalc_d)

    ax_c = sub_c.add_subplot(1, 1, 1)
    plot_panel_c(ax=ax_c, refresh=recalc_c, roi_extent=roi_extent)
    _add_block_label(ax_c, "B")

    plot_panel_f(
        fig=sub_pop, refresh=recalc_f, panel_letters=("C", "E", "D"),
        bottom_pad=0.62,
    )
    ef_subfig = sub_right.add_subfigure(right_gs[1])
    plot_panel_ef(
        fig=ef_subfig, refresh=recalc_d, panel_letters=("G", "H"),
        vertical_pad=(0.15, 0.65),
    )

    fig.savefig(out_path, dpi=400)
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
    """Replace outlined A legend text with live wrapped SVG text."""
    x_shift = (PANEL_A_X_IN - PANEL_A_BASE_X_IN) * PPI
    y_shift = PANEL_A_Y_IN * PPI
    cover_x = 103 + x_shift
    cover_y = 170 + y_shift
    text_x = 110 + x_shift
    gaze_y = 189 + y_shift
    fix_y = 212 + y_shift
    constraint_y = 228 + y_shift
    return sg.fromstring(f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{TOTAL_W_IN * PPI}" height="{CANVAS_H_IN * PPI}"
         viewBox="0 0 {TOTAL_W_IN * PPI} {CANVAS_H_IN * PPI}">
      <g>
        <rect x="{cover_x}" y="{cover_y}" width="90" height="62" fill="#ffffff"/>
        <text x="{text_x}" y="{gaze_y}" font-size="12" font-family="DejaVu Sans"
              fill="#201c1d">Gaze</text>
        <text x="{text_x}" y="{fix_y}" font-size="12" font-family="DejaVu Sans"
              fill="#201c1d">Fixation</text>
        <text x="{text_x}" y="{constraint_y}" font-size="12" font-family="DejaVu Sans"
              fill="#201c1d">constraint</text>
      </g>
    </svg>
    """).getroot()


def compose(recalc_c=False, recalc_d=False, recalc_f=False):
    main_svg = FIG_DIR / "_fig1_main.svg"
    _render_main_svg(main_svg, recalc_c=recalc_c, recalc_d=recalc_d, recalc_f=recalc_f)
    panel_b_svg = FIG_DIR / "_fig1_panel_b_inset.svg"
    _render_panel_b_svg(panel_b_svg)

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
    panel_b = _load_and_place(
        panel_b_svg,
        PANEL_B_INSET_X_IN,
        PANEL_B_INSET_Y_IN,
        PANEL_B_INSET_W_IN,
        PANEL_B_INSET_H_IN,
    )

    label_a = sg.TextElement(
        0.05 * PPI, 0.25 * PPI, "A",
        size=PANEL_A_LABEL_FONTSIZE_PX, weight="bold", font="DejaVu Sans",
    )
    panel_a_legend = _panel_a_legend_overlay()

    fig.append([main, panel_a, label_a, panel_a_legend, panel_b])

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
