#!/usr/bin/env python3
"""Build report v3 with the corrected axial-orientation Figure 4F audit."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fig.ssi_figure_v2.behavior_confounds import (  # noqa: E402
    build_supp_gaze_position_drift_report_v2_pdf as base,
)


CP5 = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_axial_orientation_audit_checkpoint5_v1"
)
OUT = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "supp_gaze_position_drift_report_v3"
)
PDF_PATH = OUT / "supplemental_gaze_position_drift_report_v3.pdf"
MD_PATH = OUT / "supplemental_gaze_position_drift_report_v3.md"


FIGURES = [dict(item) for item in base.FIGURES[:7]]
FIGURES[5] = {
    **FIGURES[5],
    "title": "Trial-held-out prediction modestly favors the additive model",
    "caption": (
        "Five-fold trial-held-out prediction error. The additive model has lower RMSE for all three "
        "outcomes in both animals, but these folds retain other trials from the same sessions in "
        "training. Figure S9 provides the stricter test on entirely unseen sessions."
    ),
}
FIGURES.extend(
    [
        {
            "number": "S8",
            "title": "Axial-orientation validation of Figure 4F",
            "path": CP5 / "panel_f_axial_orientation_audit.png",
            "caption": (
                "Contours near 0 and 180 degrees are the same horizontal axis and are combined in "
                "the canonical wrapped bin. The exact high-coherence reproduction is +0.204 arcmin "
                "(reported Figure 4F: +0.224). Canonical four-bin standardization gives -0.028 "
                "arcmin (95% interval -0.498 to +0.136), and a doubled-angle median model gives "
                "+0.001 arcmin (-0.185 to +0.187). Fully supported bin-count and boundary variants "
                "range from -0.243 to +0.104 arcmin. Mean regressions with added controls are shown "
                "separately because they change the estimand."
            ),
        },
        {
            "number": "S9",
            "title": "The additive model predicts unseen sessions better",
            "path": CP5 / "session_heldout_model_specification_cv.png",
            "caption": (
                "Leave-one-session-out prediction with session fixed effects omitted. The additive "
                "model has lower RMSE for every outcome in both animals, by about 7-15%. This is "
                "stronger support for parsimony than the trial-held-out comparison, while the "
                "interaction estimate remains the appropriate sensitivity bound."
            ),
        },
    ]
)


def save_cover(pdf: PdfPages, page: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.07, 0.935, "Gaze position and drift-cloud anisotropy", fontsize=22, weight="bold")
    fig.text(0.07, 0.902, "Revised report with axial-orientation validation of Figure 4F", fontsize=12.2, color=base.MUTED)

    fig.patches.append(
        plt.Rectangle((0.07, 0.775), 0.86, 0.095, transform=fig.transFigure,
                      facecolor=base.LIGHT, edgecolor="none")
    )
    base.add_paragraph(
        fig,
        "Main result: drift-cloud scale increases strongly with eccentricity. Axis-free shape changes "
        "little, while screen-horizontal allocation probably increases but remains model- and tracker-"
        "sensitive. More consequentially, the pooled Figure 4F contrast is strongly dependent on the "
        "absolute contour-orientation distribution. After axial-orientation control, evidence for "
        "orientation-independent local contour-drift alignment is weak.",
        0.095, 0.851, width=85, fontsize=10.5, line_height=0.0215, weight="semibold",
    )

    y = 0.73
    y = base.add_section(
        fig,
        "Question",
        "Does gaze position change the size or directional shape of fixational drift clouds, and "
        "does Figure 4F retain a contour-alignment contrast after the axial distribution of absolute "
        "contour orientation is standardized?",
        y,
    )
    y = base.add_section(
        fig,
        "Data and hierarchy",
        "The drift analysis contains 11,749 reviewed windows from 1,962 trials, 30 sessions, and two "
        "animals. Detected high-speed events were removed. Windows are nested within trials and "
        "sessions. Models are fit separately by animal; combined summaries give the animals equal "
        "weight. Figure 4F uses the 2,493 high-coherence windows.",
        y,
    )
    y = base.add_section(
        fig,
        "Gaze-position estimate",
        "At a fixed display radius of 2.706 arcmin, the screen-horizontal central-to-peripheral "
        "estimate is +0.368 arcmin in the additive model and +0.174 arcmin in the interaction model; "
        "the latter interval includes zero. Leave-one-session-out prediction favors the additive "
        "model, but the interaction result remains a useful bound.",
        y,
    )
    y = base.add_section(
        fig,
        "Corrected Figure 4F audit",
        "The exact paired outcome reproduces the pooled panel at +0.204 arcmin. Combining 0 and 180 "
        "degrees into one horizontal bin gives -0.028 arcmin after equal axial weighting. A continuous "
        "doubled-angle median model gives +0.001 arcmin. Both intervals include zero.",
        y,
    )

    fig.text(0.07, y, "Primary estimates", fontsize=12.1, weight="bold", color=base.BLUE, va="top")
    ax = fig.add_axes([0.07, y - 0.18, 0.86, 0.145])
    ax.axis("off")
    rows = [
        ["Screen H-V: additive", "+0.368", "+0.244 to +0.493"],
        ["Screen H-V: interaction", "+0.174", "-0.026 to +0.374"],
        ["Figure 4F: canonical axial bins", "-0.028", "-0.498 to +0.136"],
        ["Figure 4F: doubled-angle median", "+0.001", "-0.185 to +0.187"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Estimand", "Effect (arcmin)", "95% interval"],
        loc="center", cellLoc="left", colWidths=[0.48, 0.20, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(base.LIGHT if row == 0 else "#FAFBFC")
        if row == 0:
            cell.set_text_props(weight="bold")
    base.add_footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def save_conclusion(pdf: PdfPages, page: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.07, 0.935, "Interpretation, limits, and figure implications", fontsize=20.5, weight="bold")
    sections = [
        (
            "Drift-cloud result",
            "Gaze eccentricity is associated with larger drift clouds. Once scale and shape are "
            "separated, axis-free elongation changes little. Screen-horizontal allocation probably "
            "increases, but the estimate spans roughly +0.17 to +0.37 arcmin across plausible models. "
            "It remains possible that a position-dependent tracker contribution explains part of it.",
        ),
        (
            "Figure 4F result",
            "The pooled natural-scene contrast is reproducible, but it is strongly dependent on the "
            "absolute contour-orientation distribution. Both corrected axial methods place the "
            "orientation-standardized contrast near zero, and every interval includes zero. Boundary "
            "and bin-count sensitivity is also compatible with weak or reversed effects. The earlier "
            "82% attenuation should not be retained as a fixed result.",
        ),
        (
            "Why the former Panel C and Panel F differed",
            "Empirical bin reweighting changes only the absolute-orientation distribution of the "
            "paired high-coherence outcome. The regression sequence additionally conditions on gaze, "
            "total drift-cloud RMS radius, image variables, phase, and event timing. Total RMS is part "
            "of the behavior being studied, not an automatic nuisance. Those models answer a different "
            "conditional question and are retained only as sensitivity analyses.",
        ),
        (
            "Implication for the main figure",
            "Figure 4F should show both the empirical natural-scene contrast and an axial-orientation-"
            "standardized contrast, or it should be reframed as alignment arising from the combined "
            "anisotropies of natural scenes and eye movements. It should not be presented as strong "
            "evidence for general local contour-following independent of absolute orientation.",
        ),
        (
            "Remaining limits",
            "Only two animals are represented. Tracker calibration residuals are unavailable, and "
            "two-dimensional traces cannot test torsional retinal displacement. Several fine axial "
            "bins lack support in Allen, so unsupported bin/phase configurations are explicitly "
            "excluded rather than averaged over available bins.",
        ),
    ]
    y = 0.87
    for heading, body in sections:
        y = base.add_section(fig, heading, body, y)

    fig.patches.append(
        plt.Rectangle((0.07, 0.06), 0.86, 0.115, transform=fig.transFigure,
                      facecolor=base.LIGHT, edgecolor="none")
    )
    base.add_paragraph(
        fig,
        "Defensible conclusion: gaze eccentricity is associated with increased drift-cloud scale "
        "and possibly increased screen-horizontal allocation, although the latter is model-dependent "
        "and may include tracker effects. More consequentially, the pooled Figure 4F contour contrast "
        "is highly sensitive to absolute contour orientation. After axial-orientation control, "
        "evidence for orientation-independent local contour-drift alignment is weak.",
        0.095, 0.157, width=85, fontsize=10.0, line_height=0.0215, weight="semibold",
    )
    base.add_footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def markdown_text() -> str:
    lines = [
        "# Gaze position and drift-cloud anisotropy",
        "",
        "Revised report with axial-orientation validation of Figure 4F.",
        "",
        "## Main result",
        "",
        "Drift-cloud scale increases strongly with eccentricity. Axis-free shape changes little, while "
        "screen-horizontal allocation probably increases but remains model- and tracker-sensitive. The "
        "pooled Figure 4F contrast is strongly dependent on absolute contour orientation. After axial-"
        "orientation control, evidence for orientation-independent local contour-drift alignment is weak.",
        "",
        "## Primary estimates",
        "",
        "| Estimand | Effect (arcmin) | 95% interval |",
        "|---|---:|---:|",
        "| Screen H-V: additive | +0.368 | +0.244 to +0.493 |",
        "| Screen H-V: interaction | +0.174 | -0.026 to +0.374 |",
        "| Figure 4F: canonical axial bins | -0.028 | -0.498 to +0.136 |",
        "| Figure 4F: doubled-angle median model | +0.001 | -0.185 to +0.187 |",
        "",
    ]
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
            "The earlier 82% attenuation is not retained. The corrected canonical-bin and continuous "
            "median analyses both place the axial-orientation-standardized contrast near zero. Fully "
            "supported bin-count and boundary variants range from -0.243 to +0.104 arcmin.",
            "",
            "Empirical orientation reweighting and covariate-conditioned mean regression are different "
            "estimands. In particular, adjusting total drift-cloud RMS radius conditions on part of the "
            "behavior being studied and is not used as the primary Figure 4F correction.",
            "",
            "Figure 4F should show the empirical and axial-standardized contrasts separately, or be "
            "reframed as alignment arising from natural-scene and oculomotor anisotropies.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    base.configure()
    missing = [str(item["path"]) for item in FIGURES if not item["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing figure inputs:\n" + "\n".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(markdown_text(), encoding="utf-8")
    with PdfPages(PDF_PATH) as pdf:
        info = pdf.infodict()
        info["Title"] = "Gaze position and drift-cloud anisotropy"
        info["Subject"] = "Axial-orientation validation of Figure 4F"
        info["Author"] = "VisionCore analysis"
        save_cover(pdf, 1)
        for page, item in enumerate(FIGURES, start=2):
            base.save_figure_page(pdf, item, page)
        save_conclusion(pdf, len(FIGURES) + 2)
    print(PDF_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
