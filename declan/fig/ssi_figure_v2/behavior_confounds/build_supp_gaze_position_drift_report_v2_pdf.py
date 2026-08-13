#!/usr/bin/env python3
"""Build the revised plain-language gaze-position supplemental report."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
CP1 = BASE / "supp_gaze_position_anisotropy_checkpoint1_v1"
CP3 = BASE / "supp_gaze_position_covariance_checkpoint3_v1"
CP4 = BASE / "panel_f_gaze_attenuation_checkpoint4_v1"
OUT = BASE / "supp_gaze_position_drift_report_v2"
PDF_PATH = OUT / "supplemental_gaze_position_drift_report_v2.pdf"
MD_PATH = OUT / "supplemental_gaze_position_drift_report_v2.md"

INK = "#202124"
MUTED = "#62676D"
BLUE = "#3B6FB6"
ORANGE = "#C56A2D"
LIGHT = "#EEF1F4"


FIGURES = [
    {
        "number": "S1",
        "title": "Where gaze-position effects could enter",
        "path": CP1 / "gaze_position_mechanism_maps.png",
        "caption": (
            "Descriptive maps of reviewed, microsaccade-free drift windows. Total drift-cloud "
            "RMS radius grows at eccentric gaze. The screen-horizontal component also grows, "
            "whereas the gaze-relative pattern is not consistent between animals. These maps "
            "motivate separate tests of scale, shape, and reference frame."
        ),
    },
    {
        "number": "S2",
        "title": "Raw trends with gaze eccentricity",
        "path": CP1 / "gaze_eccentricity_descriptive_curves.png",
        "caption": (
            "Raw binned trends for each animal and all windows. The center of the drift cloud "
            "moves farther from the screen center as gaze eccentricity rises, and its total RMS "
            "radius increases. Directional differences are therefore separated from total scale "
            "in the remaining analyses."
        ),
    },
    {
        "number": "S3",
        "title": "Shape measures that do not divide one RMS outcome by another",
        "path": CP3 / "covariance_contrast_metric_contract.png",
        "caption": (
            "The primary shape outcomes are dimensionless covariance contrasts. Screen shape is "
            "(variance horizontal minus variance vertical) divided by total variance. Gaze-frame "
            "shape uses tangential minus radial variance after rotating the same covariance matrix. "
            "Axis-free shape uses the difference between its two eigenvalues. Values are translated "
            "to arcminutes only at one fixed reference cloud radius for interpretation."
        ),
    },
    {
        "number": "S4",
        "title": "Adjusted covariance-contrast curves",
        "path": CP3 / "covariance_contrast_adjusted_curves.png",
        "caption": (
            "Adjusted eccentricity curves for the exact covariance contrasts. The screen-horizontal "
            "allocation rises in both animals. Axis-free elongation changes little. The gaze-frame "
            "estimate is weaker and animal-dependent, so it should not be treated as a stable radial "
            "or tangential law."
        ),
    },
    {
        "number": "S5",
        "title": "The estimated screen effect depends on model specification",
        "path": CP3 / "covariance_contrast_specification_effects.png",
        "caption": (
            "Central-to-peripheral comparisons under successive specifications. At the fixed 2.706 "
            "arcmin reference radius, the equal-animal screen estimate is +0.368 arcmin in the additive "
            "model and +0.174 arcmin in the interaction model. The latter interval crosses zero. We "
            "therefore report a plausible range rather than one definitive correction."
        ),
    },
    {
        "number": "S6",
        "title": "Held-out prediction modestly favors the additive model",
        "path": CP3 / "model_specification_cross_validation.png",
        "caption": (
            "Five-fold, trial-held-out prediction error. The additive model has lower RMSE for all "
            "three outcomes in both animals, but the advantage is small. This supports using it as "
            "the main summary while retaining the interaction model as a sensitivity bound."
        ),
    },
    {
        "number": "S7",
        "title": "Available tracker checks do not remove the artifact concern",
        "path": CP3 / "tracker_proxy_diagnostics.png",
        "caption": (
            "High-frequency power, one-sample displacement, spatial symmetry, and session slopes are "
            "shown as tracker proxies. Screen-horizontal slopes are usually positive, but the proxy "
            "measures also change with eccentricity and left-right or upper-lower patterns are not "
            "fully symmetric. Without calibration residuals or stationary-target recordings, a "
            "position-dependent tracker contribution remains possible."
        ),
    },
    {
        "number": "S8",
        "title": "Direct audit of the Figure 4F contrast",
        "path": CP4 / "panel_f_gaze_attenuation_overview.png",
        "caption": (
            "The high-coherence covariance reconstruction reproduces Figure 4F (+0.227 versus "
            "+0.224 arcmin). After giving four absolute contour-axis bins equal weight, the contrast "
            "is +0.040 arcmin (95% interval -0.101 to +0.142), an estimated 82% attenuation. The "
            "gaze-by-orientation support tables show why a precise gaze-specific correction is not "
            "available: several peripheral cells contain very few windows."
        ),
    },
]


def configure() -> None:
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


def wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def add_footer(fig: plt.Figure, page: int) -> None:
    fig.text(0.055, 0.025, "Supplemental gaze-position analysis • revised exploratory report", fontsize=7.3, color=MUTED)
    fig.text(0.945, 0.025, str(page), fontsize=7.3, color=MUTED, ha="right")


def add_paragraph(
    fig: plt.Figure,
    text: str,
    x: float,
    y: float,
    *,
    width: int = 95,
    fontsize: float = 10.2,
    line_height: float = 0.0215,
    weight: str = "normal",
    color: str = INK,
) -> float:
    lines = wrapped(text, width).splitlines()
    fig.text(x, y, "\n".join(lines), va="top", fontsize=fontsize, weight=weight, color=color)
    return y - line_height * len(lines)


def add_section(fig: plt.Figure, heading: str, body: str, y: float) -> float:
    fig.text(0.07, y, heading, fontsize=12.1, weight="bold", color=BLUE, va="top")
    y = add_paragraph(fig, body, 0.07, y - 0.029)
    return y - 0.025


def save_cover(pdf: PdfPages, page: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.07, 0.935, "Gaze position and drift-cloud anisotropy", fontsize=22, weight="bold")
    fig.text(0.07, 0.902, "Revised supplemental analysis and direct Figure 4F audit", fontsize=12.2, color=MUTED)

    fig.patches.append(
        plt.Rectangle((0.07, 0.795), 0.86, 0.077, transform=fig.transFigure, facecolor=LIGHT, edgecolor="none")
    )
    add_paragraph(
        fig,
        "Main result: drift clouds become larger at eccentric gaze and their variance shifts toward "
        "the screen-horizontal axis, while axis-free elongation changes little. The screen effect is "
        "plausible but sensitive to model specification. A direct audit shows that absolute contour "
        "orientation—not gaze position alone—accounts for much of the displayed Figure 4F contrast.",
        0.095,
        0.854,
        width=85,
        fontsize=10.8,
        line_height=0.023,
        weight="semibold",
    )

    y = 0.75
    y = add_section(
        fig,
        "Question",
        "Does gaze position change the size or directional shape of fixational drift clouds, and does "
        "that structure measurably attenuate the contour contrast in Figure 4F?",
        y,
    )
    y = add_section(
        fig,
        "Data and hierarchy",
        "The analysis contains 11,749 reviewed drift windows from 1,962 trials, 30 sessions, and two "
        "animals. Detected high-speed events were removed before the windows were extracted. Windows "
        "are nested within trials and sessions. Models are fit separately by animal with session-"
        "clustered intervals; combined summaries give the two animals equal weight. Figure 4F intervals "
        "use a hierarchical resampling of the available data.",
        y,
    )
    y = add_section(
        fig,
        "Primary comparison",
        "The central endpoint is the median prediction below 4 degrees eccentricity (2.741 degrees); "
        "the peripheral endpoint is the median prediction at or above 8 degrees (9.580 degrees). "
        "Arcminute translations hold total drift-cloud RMS radius fixed at 2.706 arcmin.",
        y,
    )

    fig.text(0.07, y, "Central-to-peripheral shape estimates", fontsize=12.1, weight="bold", color=BLUE, va="top")
    ax = fig.add_axes([0.07, y - 0.205, 0.86, 0.17])
    ax.axis("off")
    rows = [
        ["Screen horizontal - vertical", "+0.368 [+0.244, +0.493]", "+0.174 [-0.026, +0.374]"],
        ["Gaze tangential - radial", "-0.174 [-0.334, -0.013]", "-0.069 [-0.268, +0.129]"],
        ["Axis-free major - minor", "-0.022 [-0.067, +0.024]", "-0.037 [-0.143, +0.068]"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Measure", "Additive model (arcmin)", "Interaction model (arcmin)"],
        loc="center",
        cellLoc="left",
        colWidths=[0.36, 0.32, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(LIGHT if row == 0 else "#FAFBFC")
        if row == 0:
            cell.set_text_props(weight="bold")

    add_footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def save_figure_page(pdf: PdfPages, item: dict, page: int) -> None:
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    fig.text(0.045, 0.955, f"Figure {item['number']}. {item['title']}", fontsize=15, weight="bold", va="top")
    ax = fig.add_axes([0.045, 0.195, 0.91, 0.69])
    ax.imshow(mpimg.imread(item["path"]))
    ax.axis("off")
    add_paragraph(fig, item["caption"], 0.055, 0.15, width=145, fontsize=9.2, line_height=0.0205)
    add_footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def save_conclusion(pdf: PdfPages, page: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.07, 0.935, "Interpretation, limits, and next tests", fontsize=21, weight="bold")

    sections = [
        (
            "What is supported",
            "Gaze eccentricity covaries strongly with total drift-cloud RMS radius. After scale is "
            "separated algebraically, axis-free elongation changes little. Variance is redistributed "
            "toward the screen-horizontal axis. The additive screen estimate is positive in both "
            "animals; a flexible interaction model gives a smaller, uncertain estimate. The most "
            "accurate summary is therefore a plausible screen-frame effect that is sensitive to "
            "specification.",
        ),
        (
            "What the direct Figure 4F audit adds",
            "The old comparison of numerical effect sizes did not test whether gaze position biased "
            "Figure 4F. The direct audit reproduces the original high-coherence contrast, then reduces "
            "it from +0.227 to +0.040 arcmin after absolute contour-axis bins receive equal weight. "
            "This points to shared screen-axis marginals as a major contributor. It does not provide "
            "a precise gaze-position correction because peripheral gaze-by-orientation cells are sparse.",
        ),
        (
            "Measurement limits",
            "The tracker proxy checks do not rule out a gaze-position-dependent measurement artifact. "
            "Calibration residuals or stationary-target recordings are needed for that test. Only two "
            "animals are represented, so animal consistency is descriptive rather than a population "
            "estimate. Two-dimensional traces also cannot test torsional retinal displacement. The "
            "weak gaze-frame result should remain cautious and should use the term polar angle, not "
            "gaze direction.",
        ),
        (
            "Shortest useful next tests",
            "First, repeat the Figure 4F contrast in well-supported strata jointly matched for gaze "
            "eccentricity, gaze polar angle, total drift-cloud RMS radius, and absolute contour "
            "orientation. Second, show the screen covariance contrast within matched eccentricity and "
            "scale strata for each animal. Third, run the same spatial diagnostic on calibration "
            "residuals or stationary-target data if available.",
        ),
    ]
    y = 0.87
    for heading, body in sections:
        y = add_section(fig, heading, body, y)

    fig.patches.append(
        plt.Rectangle((0.07, 0.07), 0.86, 0.095, transform=fig.transFigure, facecolor=LIGHT, edgecolor="none")
    )
    add_paragraph(
        fig,
        "Defensible conclusion: gaze position is associated with the scale and screen-frame allocation "
        "of fixational drift, but the magnitude is model-dependent and may include tracker effects. "
        "The direct Figure 4F audit does not establish a gaze-specific bias; it shows that absolute "
        "contour orientation must be controlled before the panel is interpreted as local image-trajectory matching.",
        0.095,
        0.147,
        width=85,
        fontsize=10.1,
        line_height=0.022,
        weight="semibold",
    )
    add_footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def markdown_text() -> str:
    lines = [
        "# Gaze position and drift-cloud anisotropy",
        "",
        "Revised supplemental analysis and direct Figure 4F audit.",
        "",
        "## Main result",
        "",
        "Drift clouds become larger at eccentric gaze and their variance shifts toward the screen-horizontal "
        "axis, while axis-free elongation changes little. The screen effect is plausible but sensitive to model "
        "specification. A direct audit shows that absolute contour orientation—not gaze position alone—accounts "
        "for much of the displayed Figure 4F contrast.",
        "",
        "## Data and hierarchy",
        "",
        "The analysis contains 11,749 reviewed, microsaccade-free drift windows from 1,962 trials, 30 sessions, "
        "and two animals. Windows are nested within trials and sessions. Models are fit separately by animal "
        "with session-clustered intervals; combined summaries give the animals equal weight.",
        "",
        "## Exact shape measures",
        "",
        "- Screen contrast: `(cov_xx - cov_yy) / (cov_xx + cov_yy)`.",
        "- Gaze-frame contrast: `(variance_tangential - variance_radial) / total_variance` after rotating the covariance matrix by gaze polar angle.",
        "- Axis-free contrast: `(eigenvalue_1 - eigenvalue_2) / (eigenvalue_1 + eigenvalue_2)`.",
        "",
        "Arcminute translations hold total drift-cloud RMS radius fixed at 2.706 arcmin. The central endpoint "
        "is the median prediction below 4 degrees (2.741 degrees); the peripheral endpoint is the median "
        "prediction at or above 8 degrees (9.580 degrees).",
        "",
        "## Central-to-peripheral estimates",
        "",
        "| Measure | Additive model (arcmin) | Interaction model (arcmin) |",
        "|---|---:|---:|",
        "| Screen horizontal - vertical | +0.368 [+0.244, +0.493] | +0.174 [-0.026, +0.374] |",
        "| Gaze tangential - radial | -0.174 [-0.334, -0.013] | -0.069 [-0.268, +0.129] |",
        "| Axis-free major - minor | -0.022 [-0.067, +0.024] | -0.037 [-0.143, +0.068] |",
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
            "The old numerical comparison with Figure 4F did not establish that gaze position biased that panel. "
            "The direct audit reproduces the high-coherence contrast (+0.227 versus +0.224 arcmin reported) and "
            "reduces it to +0.040 arcmin [-0.101, +0.142] after equal-weighting absolute contour-axis bins. This is "
            "82% attenuation, but sparse peripheral gaze-by-orientation cells prevent a precise gaze-specific correction.",
            "",
            "The tracker checks do not rule out a position-dependent artifact. Only two animals are represented, and "
            "two-dimensional eye traces cannot test torsional retinal displacement.",
            "",
            "## Next tests",
            "",
            "1. Repeat Figure 4F in supported strata jointly matched for gaze eccentricity, gaze polar angle, total "
            "drift-cloud RMS radius, and absolute contour orientation.",
            "2. Show the screen covariance contrast in matched eccentricity and scale strata for each animal.",
            "3. Apply the spatial diagnostic to calibration residuals or stationary-target data if available.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    configure()
    missing = [str(item["path"]) for item in FIGURES if not item["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing figure inputs:\n" + "\n".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(markdown_text(), encoding="utf-8")
    with PdfPages(PDF_PATH) as pdf:
        info = pdf.infodict()
        info["Title"] = "Gaze position and drift-cloud anisotropy"
        info["Subject"] = "Revised supplemental analysis and direct Figure 4F audit"
        info["Author"] = "VisionCore analysis"
        save_cover(pdf, 1)
        for page, item in enumerate(FIGURES, start=2):
            save_figure_page(pdf, item, page)
        save_conclusion(pdf, len(FIGURES) + 2)
    print(PDF_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
