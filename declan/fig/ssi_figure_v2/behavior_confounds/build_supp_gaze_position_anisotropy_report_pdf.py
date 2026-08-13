#!/usr/bin/env python3
"""Build a plain-language multipage PDF for the gaze-position FEM analysis."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT1 = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "supp_gaze_position_anisotropy_checkpoint1_v1"
)
CHECKPOINT2 = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "supp_gaze_position_anisotropy_broad_model_checkpoint2_v1"
)
PDF_PATH = CHECKPOINT2 / "supplemental_gaze_position_anisotropy_report_v1.pdf"
MD_PATH = CHECKPOINT2 / "supplemental_gaze_position_anisotropy_report_v1.md"

INK = "#202124"
MUTED = "#62676D"
BLUE = "#3B6FB6"
ORANGE = "#C56A2D"
LIGHT = "#EEF1F4"


FIGURES = [
    {
        "number": "S1",
        "title": "Where gaze-position effects could enter",
        "path": CHECKPOINT1 / "gaze_position_mechanism_maps.png",
        "caption": (
            "Figure S1. Descriptive maps of the reviewed drift windows. The columns separate total "
            "movement size, axis-free anisotropy, screen-horizontal versus vertical spread, and "
            "tangential versus radial spread. The raw increase at eccentric gaze is strongest in "
            "movement size and the screen frame. It is not a consistent tangential effect."
        ),
    },
    {
        "number": "S2a",
        "title": "Raw trends with gaze eccentricity",
        "path": CHECKPOINT1 / "gaze_eccentricity_descriptive_curves.png",
        "caption": (
            "Figure S2a. Raw binned trends for each animal and for all windows. Eccentric gaze is "
            "associated with larger drift clouds and a larger horizontal-minus-vertical difference. "
            "The tangential-minus-radial result differs between animals."
        ),
    },
    {
        "number": "S2b",
        "title": "Raw effect sizes compared with Figure 4F",
        "path": CHECKPOINT1 / "gaze_position_effect_size_comparison.png",
        "caption": (
            "Figure S2b. Raw peripheral-minus-central differences. These values are useful for "
            "scale, but they mix changes in movement size, gaze direction, and animal composition. "
            "The raw screen-horizontal difference is +0.656 arcmin, about 2.93 times the +0.224 "
            "arcmin Figure 4F reference. Total movement size rises by +1.079 arcmin at the same time."
        ),
    },
    {
        "number": "S3",
        "title": "Data support and movement-size adjustment",
        "path": CHECKPOINT2 / "broad_model_design_and_normalization_audit.png",
        "caption": (
            "Figure S3. Every session contains both central and peripheral gaze positions. Movement "
            "size and gaze direction nevertheless change with eccentricity. The lower panels divide "
            "directional differences by total cloud size and display them at one common movement "
            "scale. This prevents larger clouds from automatically appearing more anisotropic."
        ),
    },
    {
        "number": "S4",
        "title": "Adjusted eccentricity curves",
        "path": CHECKPOINT2 / "broad_model_adjusted_eccentricity_curves.png",
        "caption": (
            "Figure S4. Curves from the main adjusted model. The screen-horizontal component rises "
            "with eccentricity in both animals. The gaze-relative component becomes weakly radial, "
            "while axis-free anisotropy stays nearly flat after movement size is held constant. Gray "
            "dashed lines show the descriptive medians."
        ),
    },
    {
        "number": "S5",
        "title": "What survives step-by-step adjustment?",
        "path": CHECKPOINT2 / "broad_model_incremental_specification_effects.png",
        "caption": (
            "Figure S5. Peripheral-minus-central effects as adjustment terms are added. The screen "
            "effect remains in the main additive model and is comparable with Figure 4F. It becomes "
            "smaller and uncertain in the final, more flexible model, which allows the eccentricity "
            "effect to vary with movement size and gaze direction."
        ),
    },
    {
        "number": "S6",
        "title": "What the main model leaves unexplained",
        "path": CHECKPOINT2 / "broad_model_residual_spatial_maps.png",
        "caption": (
            "Figure S6. Median residuals after the main adjustment, shown separately for each animal. "
            "The remaining pattern is patchy rather than a simple central-to-peripheral ring. This is "
            "why the next analysis should show movement-size and gaze-direction strata directly."
        ),
    },
]


SUMMARY_SECTIONS = [
    (
        "Question",
        "Does the position of gaze change the directional shape of fixational eye movements, and is "
        "that effect large enough to matter for the contour-related effect shown in Figure 4F?",
    ),
    (
        "Why we separated reference frames",
        "Work by Otero-Millan and colleagues suggests that small eye movements can carry an observer-"
        "centered directional bias, whereas larger exploratory movements may be more tied to image "
        "structure. A different proposal involves retinal displacement caused by torsion. Our data "
        "contain only two-dimensional eye position, so they cannot measure that torsional component. "
        "We therefore measured screen-horizontal versus vertical spread separately from tangential "
        "versus radial spread around the current gaze position.",
    ),
    (
        "Data",
        "The analysis uses 11,749 reviewed drift windows from 30 sessions and 1,962 session-trials in "
        "two animals. Detected high-speed events were removed before these windows were extracted. "
        "Every session includes both central and peripheral gaze positions.",
    ),
]


KEY_ROWS = [
    ["Screen horizontal − vertical", "+0.399", "+0.261 to +0.538", "1.78×"],
    ["Gaze tangential − radial", "−0.191", "−0.368 to −0.014", "−0.85×"],
    ["Axis-free major − minor", "−0.036", "−0.084 to +0.011", "−0.16×"],
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_footer(fig: plt.Figure, page_number: int) -> None:
    fig.text(
        0.06,
        0.027,
        "Supplemental gaze-position analysis • exploratory candidate",
        fontsize=7.5,
        color=MUTED,
    )
    fig.text(0.94, 0.027, str(page_number), fontsize=7.5, color=MUTED, ha="right")


def draw_wrapped(
    fig: plt.Figure,
    text: str,
    x: float,
    y: float,
    *,
    width: int = 92,
    fontsize: float = 10.5,
    line_height: float = 0.024,
    color: str = INK,
    weight: str = "normal",
) -> float:
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    fig.text(x, y, "\n".join(lines), fontsize=fontsize, color=color, weight=weight, va="top")
    return y - line_height * len(lines)


def save_cover(pdf: PdfPages, page_number: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.07, 0.93, "Gaze position and FEM anisotropy", fontsize=23, weight="bold")
    fig.text(
        0.07,
        0.895,
        "Supplemental analysis of the behavioral effect shown in Figure 4F",
        fontsize=12.5,
        color=MUTED,
    )
    fig.patches.append(
        plt.Rectangle((0.07, 0.795), 0.86, 0.075, transform=fig.transFigure,
                      facecolor=LIGHT, edgecolor="none")
    )
    draw_wrapped(
        fig,
        "Main result: eccentric gaze is linked to a shift toward screen-horizontal FEM spread, not "
        "to a general increase in normalized anisotropy. The estimated size is comparable with the "
        "Figure 4F contour effect, but it depends on how movement size and gaze direction are handled.",
        0.095,
        0.852,
        width=86,
        fontsize=11.3,
        line_height=0.026,
        weight="semibold",
    )

    y = 0.75
    for heading, body in SUMMARY_SECTIONS:
        fig.text(0.07, y, heading, fontsize=12, weight="bold", color=BLUE)
        y = draw_wrapped(fig, body, 0.07, y - 0.026, width=98, fontsize=10.2, line_height=0.022)
        y -= 0.025

    fig.text(0.07, y, "Primary adjusted comparison", fontsize=12, weight="bold", color=BLUE)
    ax = fig.add_axes([0.07, y - 0.20, 0.86, 0.16])
    ax.axis("off")
    table = ax.table(
        cellText=KEY_ROWS,
        colLabels=["Measure", "Effect (arcmin)", "95% interval", "Relative to 4F"],
        loc="center",
        cellLoc="left",
        colWidths=[0.36, 0.18, 0.27, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.45)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(LIGHT if row == 0 else "#FAFBFC")
        if row == 0:
            cell.set_text_props(weight="bold")

    add_footer(fig, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def save_figure_page(
    pdf: PdfPages,
    item: dict,
    page_number: int,
    *,
    caption_height: float = 0.15,
) -> None:
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    fig.text(
        0.045,
        0.955,
        f"Figure {item['number']}. {item['title']}",
        fontsize=15,
        weight="bold",
        va="top",
        zorder=10,
    )
    ax = fig.add_axes([0.045, caption_height + 0.06, 0.91, 0.64])
    image = mpimg.imread(item["path"])
    ax.imshow(image)
    ax.axis("off")
    draw_wrapped(
        fig,
        item["caption"],
        0.055,
        caption_height + 0.025,
        width=145,
        fontsize=9.2,
        line_height=0.021,
        color=INK,
    )
    add_footer(fig, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def save_conclusion(pdf: PdfPages, page_number: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.07, 0.93, "Interpretation and next step", fontsize=21, weight="bold")

    sections = [
        (
            "Most direct reading",
            "The raw FEM cloud becomes larger at eccentric gaze. Once all clouds are put on the same "
            "movement-size scale, there is no clear increase in axis-free elongation. Instead, the "
            "cloud shifts toward the screen-horizontal axis. In the main model this shift is +0.399 "
            "arcmin, compared with +0.224 arcmin for Figure 4F.",
        ),
        (
            "What is reassuring",
            "Both animals show a similar screen-horizontal shift in the main model. The comparison is "
            "made within sessions, and every session contains central and peripheral gaze positions. "
            "The model also accounts for movement size, gaze direction, image orientation, fixation "
            "phase, time since the previous event, and session-to-session differences.",
        ),
        (
            "What remains uncertain",
            "A more flexible model reduces the screen effect to +0.180 arcmin, with an interval from "
            "−0.044 to +0.404. This means the effect may vary with movement size or gaze direction. "
            "The residual maps in Figure S6 also retain local patches. The current result is therefore "
            "a strong candidate effect, not a final single-number correction for Figure 4F.",
        ),
        (
            "What this analysis does not test",
            "These two-dimensional traces cannot measure retinal displacement caused by torsion. The "
            "results also do not show that gaze position causes the FEM change. Detected microsaccades "
            "were excluded from these drift windows and should be studied as a separate event analysis.",
        ),
        (
            "Recommended next supplemental figure",
            "Plot the screen-horizontal eccentricity effect separately for low, middle, and high "
            "movement sizes and for the main gaze-direction sectors, with Allen and Logan shown "
            "separately and the number of supported windows printed in every cell. This is the shortest "
            "route to explaining why the main and flexible models disagree.",
        ),
    ]
    y = 0.865
    for heading, body in sections:
        fig.text(0.07, y, heading, fontsize=12.3, weight="bold", color=BLUE)
        y = draw_wrapped(fig, body, 0.07, y - 0.028, width=98, fontsize=10.4, line_height=0.023)
        y -= 0.035

    fig.patches.append(
        plt.Rectangle((0.07, 0.058), 0.86, 0.078, transform=fig.transFigure,
                      facecolor=LIGHT, edgecolor="none")
    )
    draw_wrapped(
        fig,
        "Provisional supplemental conclusion: gaze position is a plausible, behaviorally meaningful "
        "modifier of FEM direction. Its clearest expression is in screen coordinates and its adjusted "
        "size is comparable with Figure 4F, but the dependence on movement regime still needs to be "
        "shown directly.",
        0.095,
        0.124,
        width=84,
        fontsize=9.8,
        line_height=0.021,
        weight="semibold",
    )
    add_footer(fig, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def markdown_text() -> str:
    lines = [
        "# Gaze position and FEM anisotropy",
        "",
        "Supplemental analysis of the behavioral effect shown in Figure 4F.",
        "",
        "## Main result",
        "",
        "Eccentric gaze is linked to a shift toward screen-horizontal FEM spread, not to a general "
        "increase in normalized anisotropy. The estimated size is comparable with the Figure 4F "
        "contour effect, but it depends on how movement size and gaze direction are handled.",
        "",
    ]
    for heading, body in SUMMARY_SECTIONS:
        lines.extend([f"## {heading}", "", body, ""])
    lines.extend(
        [
            "## Primary adjusted comparison",
            "",
            "| Measure | Effect (arcmin) | 95% interval | Relative to 4F |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in KEY_ROWS:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for item in FIGURES:
        lines.extend(
            [
                f"## Figure {item['number']}. {item['title']}",
                "",
                f"![Figure {item['number']}]({item['path']})",
                "",
                item["caption"],
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The raw FEM cloud becomes larger at eccentric gaze. Once all clouds are put on the same "
            "movement-size scale, there is no clear increase in axis-free elongation. Instead, the "
            "cloud shifts toward the screen-horizontal axis.",
            "",
            "The main model estimates a +0.399 arcmin screen-horizontal shift, with a 95% interval "
            "from +0.261 to +0.538 arcmin. A more flexible model gives +0.180 arcmin, with an interval "
            "from −0.044 to +0.404 arcmin. The effect is therefore promising but sensitive to whether "
            "it is allowed to vary with movement size and gaze direction.",
            "",
            "## Recommended next analysis",
            "",
            "Show the screen-horizontal eccentricity effect across movement-size and gaze-direction "
            "strata, separately for both animals and with support shown in every cell.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    configure_matplotlib()
    missing = [str(item["path"]) for item in FIGURES if not item["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing figure inputs:\n" + "\n".join(missing))
    CHECKPOINT2.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(markdown_text(), encoding="utf-8")

    with PdfPages(PDF_PATH) as pdf:
        info = pdf.infodict()
        info["Title"] = "Gaze position and FEM anisotropy"
        info["Subject"] = "Supplemental analysis related to Figure 4F"
        info["Author"] = "VisionCore analysis"
        save_cover(pdf, 1)
        save_figure_page(pdf, FIGURES[0], 2)
        save_figure_page(pdf, FIGURES[1], 3)
        save_figure_page(pdf, FIGURES[2], 4)
        save_figure_page(pdf, FIGURES[3], 5)
        save_figure_page(pdf, FIGURES[4], 6)
        save_figure_page(pdf, FIGURES[5], 7)
        save_figure_page(pdf, FIGURES[6], 8)
        save_conclusion(pdf, 9)

    print(PDF_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
