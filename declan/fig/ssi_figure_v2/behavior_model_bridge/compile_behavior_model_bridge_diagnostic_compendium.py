#!/usr/bin/env python3
"""Compile behavior-model bridge diagnostics into one annotated PDF packet."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge


OUT_DIR = bridge.OUT_DIR
OUT_PDF = OUT_DIR / "behavior_model_bridge_diagnostic_compendium.pdf"
OUT_MANIFEST = OUT_DIR / "behavior_model_bridge_diagnostic_compendium_manifest.json"
TMP_DIR = Path("/tmp") / "behavior_model_bridge_diagnostic_compendium"

PAGE_SIZE = (11.0, 8.5)
INK = "#111111"
MUTED = "#4b5563"
GREEN = "#1b7f5c"
BLUE = "#2f6f9f"
ORANGE = "#b4492d"
PURPLE = "#7251a5"
GRAY = "#6B6F75"
LIGHT_GREEN = "#eef7f1"
LIGHT_BLUE = "#eef4fa"
LIGHT_ORANGE = "#fbf0ec"


@dataclass(frozen=True)
class Section:
    title: str
    subtitle: str
    paragraphs: tuple[str, ...]
    bullets: tuple[str, ...] = ()
    callout: str | None = None
    theme: str = BLUE


@dataclass(frozen=True)
class FigureBlock:
    title: str
    source_pdf: Path
    description: str


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrapped_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False))
    return lines


def _text_page(
    path: Path,
    *,
    section: Section,
    page_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=PAGE_SIZE)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="none", zorder=-2))
        ax.add_patch(plt.Rectangle((0.0, 0.935), 1, 0.065, facecolor=section.theme, alpha=0.95, edgecolor="none"))
        ax.text(0.055, 0.963, page_label, ha="left", va="center", fontsize=9, color="#ffffff", fontweight="bold")
        ax.text(0.94, 0.963, "Behavior-model bridge diagnostics", ha="right", va="center", fontsize=8, color="#ffffff")

        y = 0.875
        ax.text(0.075, y, section.title, ha="left", va="top", fontsize=21, color=INK, fontweight="bold")
        y -= 0.065
        if section.subtitle:
            ax.text(0.075, y, section.subtitle, ha="left", va="top", fontsize=11.5, color=MUTED)
            y -= 0.065

        for paragraph in section.paragraphs:
            for line in _wrapped_lines(paragraph, width=108):
                ax.text(0.075, y, line, ha="left", va="top", fontsize=10.1, color=INK)
                y -= 0.032 if line else 0.018
            y -= 0.022

        if section.bullets:
            ax.text(0.075, y, "Key points", ha="left", va="top", fontsize=11.0, color=section.theme, fontweight="bold")
            y -= 0.04
            for bullet in section.bullets:
                lines = _wrapped_lines(bullet, width=98)
                ax.text(0.09, y, "-", ha="left", va="top", fontsize=10.2, color=section.theme, fontweight="bold")
                for idx, line in enumerate(lines):
                    ax.text(0.112, y, line, ha="left", va="top", fontsize=9.8, color=INK)
                    y -= 0.031 if idx < len(lines) - 1 else 0.034
                y -= 0.008

        if section.callout:
            box_y = max(0.07, y - 0.115)
            ax.add_patch(
                plt.Rectangle(
                    (0.075, box_y),
                    0.85,
                    0.105,
                    facecolor=LIGHT_GREEN if section.theme == GREEN else LIGHT_BLUE if section.theme == BLUE else LIGHT_ORANGE,
                    edgecolor=section.theme,
                    linewidth=1.0,
                    alpha=0.95,
                )
            )
            callout_lines = _wrapped_lines(section.callout, width=104)
            call_y = box_y + 0.078
            for line in callout_lines[:3]:
                ax.text(0.095, call_y, line, ha="left", va="top", fontsize=10.0, color=INK, fontweight="bold")
                call_y -= 0.031

        ax.text(0.075, 0.035, "Generated from local analysis outputs; pp = percentage points of predicted SSI residual.", ha="left", va="center", fontsize=7.6, color=GRAY)
        pdf.savefig(fig)
        plt.close(fig)


def _cover_page(path: Path) -> None:
    section = Section(
        title="Behavior-Model Bridge Diagnostics",
        subtitle="Contour-relative eye movements, model SSI dose curves, and random-rotation matching controls",
        paragraphs=(
            "This packet collects the bridge diagnostics developed after the Panel G component-dose analysis. The goal is to test whether real BackImage drift geometry occupies regions of the model dose-response curves that preserve high-spatial-frequency contour information.",
            "The most useful control is the random-rotation null: keep each real eye trajectory and image fixed, but rotate the trajectory relative to the image contour axis. If observed trace-contour matching predicts higher SSI than random rotations, the behavior contains image-aligned information beyond generic drift statistics.",
        ),
        bullets=(
            "First-pass bridge: empirical behavior distributions do not by themselves show a large absolute SSI rescue on RMS/range curves.",
            "Random-rotation bridge: observed trace-contour matching reliably beats random rotations for high-SF populations on RMS/range, especially aligned high-SF units and coherent contours.",
            "Control pattern: low-SF units often move in the opposite direction, arguing against a generic rotation artifact.",
        ),
        callout="Main interpretation: real drift-contour matching appears selectively beneficial for high-SF contour-relevant coding, but the current prediction is still a marginal 1D reconstruction rather than a joint normal-by-parallel SSI surface.",
        theme=GREEN,
    )
    _text_page(path, section=section, page_label="Overview")


def _figure_intro_page(path: Path, *, block: FigureBlock, page_label: str, theme: str) -> None:
    section = Section(
        title=block.title,
        subtitle="Figure description and interpretation guide",
        paragraphs=(block.description,),
        theme=theme,
    )
    _text_page(path, section=section, page_label=page_label)


def _sections_and_figures() -> tuple[list[Section], list[FigureBlock]]:
    sections = [
        Section(
            title="Methods Motivation",
            subtitle="Why the bridge analysis was needed",
            paragraphs=(
                "The original model result said that increasing full trajectory path length can hurt high-SF contour-aligned units, but the behavior result said real drift clouds become contour-parallel near coherent image structure. Those statements use different motion metrics. Full path length, component path, RMS excursion, and projected range are related but not interchangeable.",
                "The bridge analysis therefore asks whether behavior-weighted model predictions improve when empirical eye movements are projected through the same component-dose curves. A good bridge should distinguish two possibilities: behavior lands in a genuinely beneficial dose region, or behavior simply avoids the damaging high-normal-motion tail.",
            ),
            bullets=(
                "Dose axes: unsigned component path, component RMS excursion, projected range, and path/range.",
                "Behavior snippets: central 40 samples, matching the 0.325 s trace-bank duration used in the Panel G diagnostics.",
                "Population splits: all high-SF, aligned high-SF, oblique high-SF, orthogonal high-SF, and all low-SF.",
            ),
            theme=BLUE,
        ),
        Section(
            title="First-Pass Bridge Result",
            subtitle="Empirical behavior on model curves",
            paragraphs=(
                "The first bridge integrates empirical behavior doses against the one-dimensional model curves. For aligned high-SF units, high coherence does not create a large absolute rescue in RMS or range predictions. The result is closer to a cautious tail-avoidance story than a strong sweet-spot story.",
                "For aligned high-SF units, high coherence minus low coherence is roughly +0.06 pp for normal RMS and +0.02 pp for normal range, both with wide intervals. Component path is more negative for high coherence on the normal axis, reinforcing that accumulated path and excursion/spread are not the same behavioral claim.",
            ),
            bullets=(
                "Distribution overlays show that empirical behavior sits largely in the trace-bank reference region, with rare outliers.",
                "Tail occupancy shows high coherence reduces some extreme final-tail mass on RMS/range, but leaves mass in q75-to-tail regions.",
                "This motivates a more direct orientation-matching control instead of relying only on absolute coherence-bin averages.",
            ),
            theme=ORANGE,
        ),
        Section(
            title="Random-Rotation Null",
            subtitle="A direct test of trace-contour matching",
            paragraphs=(
                "The random-rotation null keeps each real trajectory intact but rotates it relative to the fixed image contour axis. This preserves trace shape, total path, speed, tortuosity, and image composition while breaking the trace-contour relationship.",
                "Positive observed-minus-random values mean the real trace-contour alignment predicts higher SSI than random trace orientations. This is the cleanest test here for whether the animal's drift geometry prioritizes a model population.",
            ),
            bullets=(
                "At contour coherence >=0.2, aligned high-SF component-mean RMS is +0.090 pp, CI [+0.042, +0.140].",
                "Aligned high-SF component-mean range is +0.058 pp, CI [+0.021, +0.096].",
                "Low-SF all RMS is -0.067 pp, CI [-0.118, -0.016], a useful sign-opposed control.",
            ),
            callout="This is telling because the same traces and images are used in both conditions; only relative orientation is randomized.",
            theme=GREEN,
        ),
        Section(
            title="Coherence-Resolved Rotation Bridge",
            subtitle="Where the matching advantage appears",
            paragraphs=(
                "Folding the rotation null back into the behavior coherence bins shows that the aligned high-SF RMS advantage grows as local contour coherence increases. Low-coherence windows are near zero; coherent windows are consistently positive.",
                "For aligned high-SF RMS, the match advantage is +0.019 pp at coherence 0-0.2, +0.089 pp at 0.2-0.5, +0.101 pp at 0.5-0.8, and +0.181 pp at 0.8-1. The last bin is widest because it contains 371 windows, but the direction is consistent.",
            ),
            bullets=(
                "RMS is the cleanest axis: aligned high-SF is significant for all coherence bins >=0.2.",
                "Projected range follows the same direction but is noisier, especially in the highest-coherence bin.",
                "Low-SF units tilt negative for coherent contours, arguing that the effect is population-specific rather than a generic feature of the null.",
            ),
            theme=PURPLE,
        ),
        Section(
            title="Interpretation And Caveats",
            subtitle="What we can and cannot claim yet",
            paragraphs=(
                "The strongest current claim is that real trace-contour matching is model-beneficial for high-SF populations relative to random trajectory orientation. This is stronger than saying behavior simply has smaller drift or smaller normal spread, because the random-rotation control preserves the traces themselves.",
                "The current bridge still uses one-dimensional marginal model curves. Averaging normal and parallel predictions is useful as a smoke test, but it is not a true joint model of SSI as a function of both axes simultaneously. A joint normal-by-parallel surface is the natural next diagnostic.",
            ),
            bullets=(
                "Do not overstate intention: the analysis supports population-specific prioritization by geometry, not a direct causal behavioral strategy claim.",
                "Report pp as percentage points of predicted SSI residual, not percent change.",
                "Next step: use the same observed-vs-random rotation contrast on a 2D normal-by-parallel dose surface.",
            ),
            callout="The story has become more defensible: behavior is not just aligned with contours descriptively; relative to random rotations, that alignment predicts better high-SF contour coding.",
            theme=GREEN,
        ),
    ]

    figure_blocks = [
        FigureBlock(
            title="Model Dose Curves Across Alternative X-Axes",
            source_pdf=ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "panel_g_alternative_x_axes_diagnostic.pdf",
            description=(
                "This diagnostic establishes the model-side dose curves for component path, RMS excursion, projected range, and path/range. The key use here is not the absolute level of each curve, but the population-specific sensitivity to contour-normal versus contour-parallel motion. It provides the dose-response functions used by the bridge analyses."
            ),
        ),
        FigureBlock(
            title="Empirical Behavior Distributions On Model Curves",
            source_pdf=OUT_DIR / "behavior_model_bridge_distribution_on_curves.pdf",
            description=(
                "These pages overlay empirical behavior doses on the model curves for each population. They answer whether the behavior distribution visibly lands in a beneficial region or mainly avoids damaging tails. The answer is mixed: the plot calibrates behavior to the trace bank, but does not alone show a strong high-SF rescue."
            ),
        ),
        FigureBlock(
            title="Behavior-Weighted Prediction By Coherence",
            source_pdf=OUT_DIR / "behavior_model_bridge_predicted_ssi_by_coherence.pdf",
            description=(
                "This first-pass bridge integrates one-dimensional model curves against empirical behavior distributions in each coherence bin. For aligned high-SF units, RMS/range predictions remain near flat with broad intervals, while component path can look worse at high coherence. That motivated the random-rotation control."
            ),
        ),
        FigureBlock(
            title="Tail Region Occupancy For Aligned High-SF Units",
            source_pdf=OUT_DIR / "behavior_model_bridge_tail_region_occupancy_high_sf_aligned.pdf",
            description=(
                "This diagnostic separates beneficial occupancy from damaging-tail avoidance. High coherence reduces some final-tail occupancy on RMS/range, but the behavior-weighted mean is not strongly rescued because substantial mass remains in q75-to-tail regions and some samples fall outside the modeled curve support."
            ),
        ),
        FigureBlock(
            title="Random-Rotation Main-Point Summary",
            source_pdf=OUT_DIR / "behavior_model_bridge_random_rotation_match_null_main_point.pdf",
            description=(
                "This single-panel plot is the compact result. Positive values mean observed trace-contour matching beats random trajectory rotations. High-SF populations are positive on RMS/range, aligned high-SF is strongest, and low-SF all is negative. This is the clearest evidence that the geometry is selectively beneficial for high-SF coding."
            ),
        ),
        FigureBlock(
            title="Random-Rotation Component-Mean Summary",
            source_pdf=OUT_DIR / "behavior_model_bridge_random_rotation_match_null_component_mean_summary.pdf",
            description=(
                "This multipage sheet repeats the observed-minus-random summary for all coherence-threshold subsets and dose metrics. It shows the robustness of the thresholded random-rotation result: RMS and range are the useful axes, component path is mostly flat, and high-SF populations separate from the low-SF control."
            ),
        ),
        FigureBlock(
            title="Random-Rotation Component-Specific Summary",
            source_pdf=OUT_DIR / "behavior_model_bridge_random_rotation_match_null_component_specific_summary.pdf",
            description=(
                "This sheet disambiguates which component drives the thresholded effect. For high-SF populations, the benefit is driven mainly by contour-normal RMS/range. The low-SF population shows a different, often opposite pattern, strengthening the population-specific interpretation."
            ),
        ),
        FigureBlock(
            title="Coherence-Resolved Match Advantage",
            source_pdf=OUT_DIR / "behavior_model_bridge_random_rotation_prediction_by_coherence_match_advantage_main.pdf",
            description=(
                "This plot folds the rotation null back into the original behavior coherence bins. The aligned high-SF RMS advantage is near zero for low-coherence windows and becomes positive for coherent contours. Range follows the same direction with noisier intervals."
            ),
        ),
        FigureBlock(
            title="Observed Versus Random Predictions By Coherence",
            source_pdf=OUT_DIR / "behavior_model_bridge_random_rotation_prediction_by_coherence_component_mean_predictions.pdf",
            description=(
                "These pages show the observed and random-rotated behavior-weighted model predictions directly for each population. They make clear that the rotation null can reveal a relative advantage even when the absolute observed prediction curve is nearly flat."
            ),
        ),
    ]
    return sections, figure_blocks


def build(out_pdf: Path = OUT_PDF, *, tmp_dir: Path = TMP_DIR) -> dict[str, Any]:
    pdfunite = shutil.which("pdfunite")
    if not pdfunite:
        raise RuntimeError("pdfunite is required to merge vector PDF pages")

    sections, figure_blocks = _sections_and_figures()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for old in tmp_dir.glob("*.pdf"):
        old.unlink()

    merge_inputs: list[Path] = []
    cover = tmp_dir / "000_cover.pdf"
    _cover_page(cover)
    merge_inputs.append(cover)

    section_idx = 1
    figure_idx = 1
    for section in sections[:2]:
        path = tmp_dir / f"{section_idx:03d}_section.pdf"
        _text_page(path, section=section, page_label=f"Section {section_idx}")
        merge_inputs.append(path)
        section_idx += 1

    for block_idx, block in enumerate(figure_blocks[:4], start=1):
        intro = tmp_dir / f"{section_idx:03d}_figure_intro_{block_idx:02d}.pdf"
        _figure_intro_page(intro, block=block, page_label=f"Figure {figure_idx}", theme=BLUE if figure_idx <= 3 else ORANGE)
        merge_inputs.extend([intro, block.source_pdf])
        section_idx += 1
        figure_idx += 1

    for section in sections[2:4]:
        path = tmp_dir / f"{section_idx:03d}_section.pdf"
        _text_page(path, section=section, page_label=f"Section {section_idx}")
        merge_inputs.append(path)
        section_idx += 1

    for block in figure_blocks[4:]:
        intro = tmp_dir / f"{section_idx:03d}_figure_intro_{figure_idx:02d}.pdf"
        _figure_intro_page(intro, block=block, page_label=f"Figure {figure_idx}", theme=GREEN if figure_idx <= 7 else PURPLE)
        merge_inputs.extend([intro, block.source_pdf])
        section_idx += 1
        figure_idx += 1

    final_section = tmp_dir / f"{section_idx:03d}_section.pdf"
    _text_page(final_section, section=sections[-1], page_label="Interpretation")
    merge_inputs.append(final_section)

    missing = [path for path in merge_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PDF inputs: {missing}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([pdfunite, *[str(path) for path in merge_inputs], str(out_pdf)], check=True)

    manifest = {
        "output_pdf": out_pdf,
        "merge_inputs": merge_inputs,
        "source_figures": [block.source_pdf for block in figure_blocks],
        "n_source_figures": len(figure_blocks),
        "notes": (
            "Narrative section pages are generated with matplotlib; source figure PDFs are merged "
            "directly with pdfunite to preserve vector content."
        ),
    }
    _write_json(OUT_MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--tmp-dir", type=Path, default=TMP_DIR)
    args = parser.parse_args()
    manifest = build(out_pdf=args.out_pdf, tmp_dir=args.tmp_dir)
    print(manifest["output_pdf"])
    print(OUT_MANIFEST)


if __name__ == "__main__":
    main()
