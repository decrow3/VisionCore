#!/usr/bin/env python3
"""Collect BackImage real-trace SSI figures into multi-page PDFs."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
DEFAULT_OUTPUT_DIR = DEFAULT_MATRIX_DIR / "phase1_phase2_conditioning_v1" / "plot_collections"

KEY_CAPTIONS = {
    "population_ssi_vs_trace_path_length.png": {
        "heading": "Pilot: population SSI by real-trace path length",
        "body": (
            "This is the first pooled view of stimulus-specific information (SSI) as real eye traces get larger. "
            "It establishes the basic movement axis, but it does not separate spatial-frequency tuning, image content, or unit-contour geometry."
        ),
        "caveat": "Treat this as orientation- and image-content pooled; total path length collapses across and along motion components.",
    },
    "sf_group_ssi_vs_trace_path_length.png": {
        "heading": "Pilot: spatial-frequency groups by trace path length",
        "body": (
            "This separates the pooled trace-size effect by spatial-frequency preference. "
            "The groups differ in baseline SSI and in how strongly SSI changes with movement, motivating the later low-, middle-, and high-SF panels."
        ),
        "caveat": "The plot still pools over image content and unit-contour alignment.",
    },
    "population_ssi_distribution_by_microsaccade.png": {
        "heading": "Pilot: SSI distribution split by detected microsaccades",
        "body": (
            "This compares traces with no detected microsaccade to traces with at least one detected microsaccade. "
            "It shows that microsaccade-containing snippets live in a different part of the movement distribution."
        ),
        "caveat": "Microsaccade presence is not isolated from path length, speed, and other trajectory statistics.",
    },
    "unit_by_trace_ssi_heatmap.png": {
        "heading": "Pilot: unit-by-trace SSI heterogeneity",
        "body": (
            "This heatmap shows that SSI is not distributed uniformly across units or traces. "
            "It is a reminder that population summaries can be shaped by a subset of high-information, strongly driven units."
        ),
        "caveat": "Use population summaries and unit-first summaries together when checking for dominance by a few units.",
    },
    "phase1_feature_qc_distributions.png": {
        "heading": "Feature QC for sampled image windows and eye traces",
        "body": (
            "These panels summarize the sampled image and trace features used for conditioning analyses. "
            "They are the main audit that the sampled bank spans useful ranges of contrast, contour strength, path length, and related movement metrics."
        ),
        "caveat": "These are descriptive distributions; they do not by themselves test SSI effects.",
    },
    "phase2_population_by_microsaccade_trace_path.png": {
        "heading": "Population SSI by path length and microsaccade category",
        "body": (
            "This asks whether real trace snippets with larger total path length carry different population SSI, separately for drift-only and microsaccade-containing traces. "
            "It gives the coarse movement-size picture before conditioning on cell tuning and image geometry."
        ),
        "caveat": "Microsaccade categories differ in more than microsaccade presence, so avoid a pure causal microsaccade interpretation.",
    },
    "phase2_sf_groups_by_trace_path.png": {
        "heading": "Spatial-frequency groups by real-trace path length",
        "body": (
            "This figure shows that the effect of retinal image motion depends on unit spatial-frequency preference. "
            "Low- and middle-SF populations tend to gain SSI with larger drift-scale movement, while high-SF units require more conditioning."
        ),
        "caveat": "This is still based on total path length and does not separate local image contour geometry.",
    },
    "phase2_image_contour_class_by_trace_path.png": {
        "heading": "Image contour content conditions the movement effect",
        "body": (
            "This compares image windows with different contour classifications. "
            "It supports the idea that eye movement effects depend on the local image structure being swept across the model's receptive fields."
        ),
        "caveat": "Contour labels summarize local image features; they are not a direct measure of each unit's preferred stimulus.",
    },
    "phase2_unit_image_orientation_match_by_trace_path.png": {
        "heading": "Unit-contour orientation relationship across image windows",
        "body": (
            "This conditions SSI by the angular difference between each unit's preferred orientation and the local image contour axis. "
            "It tests whether the same trace has different information effects depending on tuning-image alignment."
        ),
        "caveat": "The comparison is over unit-image pairs, not over a single fixed set of units or images.",
    },
    "phase2_unit_image_orientation_match_contour_images_by_trace_path.png": {
        "heading": "Unit-contour orientation relationship on contour-rich images",
        "body": (
            "This repeats the orientation-match analysis after emphasizing image windows with stronger contour structure. "
            "It is closer to the mechanistic question of how fixational motion samples a strong local edge or contour."
        ),
        "caveat": "The selected image set is narrower, so baseline SSI and uncertainty can change substantially.",
    },
    "phase2_trace_image_axis_by_trace_path.png": {
        "heading": "Trace direction relative to image contour axis",
        "body": (
            "This asks whether movement aligned with, or across, the local image contour axis changes SSI. "
            "It motivated the later component-path analyses that separate along-contour and across-contour displacement."
        ),
        "caveat": "Total path bins can hide opposing along- and across-contour contributions.",
    },
    "phase2_real_trace_sf_contour_matched_low_high_scale_curves.png": {
        "heading": "Earlier contour-matched SF comparison",
        "body": (
            "This earlier summary compares low- and high-SF units when preferred orientation is matched to the local contour. "
            "It shows why high-SF units needed a geometry-specific treatment rather than a single movement-size curve."
        ),
        "caveat": "This belongs to the earlier conditioning workflow and should be interpreted alongside the corrected trace-bank notes.",
    },
    "phase2_real_trace_sf_contour_orthogonal_low_high_scale_curves.png": {
        "heading": "Earlier contour-orthogonal SF comparison",
        "body": (
            "This earlier summary compares low- and high-SF units when preferred orientation is orthogonal to the local contour. "
            "It provides the counterpart to the contour-matched condition."
        ),
        "caveat": "As above, this is an older summary and is mainly useful as context for the newer component-path results.",
    },
    "phase2_real_trace_sf_contour_aligned_vs_orthogonal_low_high_scale_curves.png": {
        "heading": "Earlier aligned-versus-orthogonal comparison",
        "body": (
            "This overlays the matched and orthogonal conditioning views. "
            "It helped reveal that high-SF SSI cannot be explained by movement size alone."
        ),
        "caveat": "The later across/along decomposition is the cleaner interpretation of the geometry.",
    },
    "phase2_alignment_response_preferred_vs_orthogonal.png": {
        "heading": "Mean response for preferred versus orthogonal contour relationships",
        "body": (
            "This checks response strength as a separate quantity from SSI. "
            "It helps distinguish whether an SSI effect reflects information changes rather than simply larger mean activation."
        ),
        "caveat": "Response magnitude and SSI are related but not interchangeable.",
    },
    "all_images_no_osi_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png": {
        "heading": "Current summary: all images, all units",
        "body": (
            "This is the broad spike-weighted population SSI result with no OSI gate and all image windows. "
            "The smallest real drift bin is above the stabilized baseline for low-, middle-, and high-SF groups."
        ),
        "caveat": "This pooled result mixes image content, unit tuning, and movement direction.",
    },
    "all_images_no_osi_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png": {
        "heading": "Current component view: all images, all units",
        "body": (
            "This separates movement into path components along and across the local image contour axis while keeping all images and all units. "
            "Both components are positive in the pooled population because many tuning-image relationships are mixed together."
        ),
        "caveat": "Along and across are defined relative to the image contour, not each unit's preferred orientation.",
    },
    "strong_contours_no_osi_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png": {
        "heading": "Current summary: strong contour images, all units",
        "body": (
            "This restricts image windows to those flagged as strong contours while retaining all units. "
            "The high-SF population shows a positive movement effect here, even though strictly aligned and orthogonal high-SF subsets do not show the same total-path jump."
        ),
        "caveat": "The pooled high-SF boost includes intermediate orientation relationships and spike-weight changes.",
    },
    "strong_contours_no_osi_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png": {
        "heading": "Current component view: strong contours, all units",
        "body": (
            "This asks whether along-contour and across-contour motion are both beneficial when all units are pooled on strong contour images. "
            "Both components are positive in the pooled high-SF population."
        ),
        "caveat": "Pooling over unit-contour orientation can make both image-axis components look helpful.",
    },
    "contour_matched_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png": {
        "heading": "Current summary: strong contours, orientation-aligned units",
        "body": (
            "This selects strong contour images and units whose preferred orientation is aligned with the contour axis. "
            "Low- and middle-SF groups show positive movement effects; high-SF units are much more sensitive to movement direction."
        ),
        "caveat": "Total path length combines along- and across-contour motion, which can cancel for high-SF units.",
    },
    "contour_matched_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png": {
        "heading": "Current component view: strong contours, orientation-aligned units",
        "body": (
            "For aligned high-SF units, across-contour motion can increase SSI while along-contour motion can reduce it. "
            "This is the clearest evidence that local motion geometry matters for high-SF information."
        ),
        "caveat": "The sign pattern is specific to the aligned unit-contour relationship.",
    },
    "contour_intermediate_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png": {
        "heading": "Current summary: strong contours, orientation-intermediate units",
        "body": (
            "This is the disjoint group with unit-contour mismatch between the aligned and orthogonal gates. "
            "It explains much of the high-SF pooled boost from stabilized to the smallest real-trace bin."
        ),
        "caveat": "This group should not be called screen-oblique; it is intermediate relative to the local contour axis.",
    },
    "contour_intermediate_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png": {
        "heading": "Current component view: strong contours, orientation-intermediate units",
        "body": (
            "For intermediate high-SF unit-contour pairs, the first across-contour component bin is positive, while the first along-contour bin is weaker. "
            "This supports the idea that the pooled high-SF gain is not coming from the strictly aligned or orthogonal groups."
        ),
        "caveat": "The useful motion axis is not a pure along/across case because the unit preference is intermediate.",
    },
    "contour_orthogonal_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png": {
        "heading": "Current summary: strong contours, orientation-orthogonal units",
        "body": (
            "This selects strong contour images and units whose preferred orientation is roughly orthogonal to the contour. "
            "Low- and middle-SF units increase with movement, while high-SF total-path SSI is flat at the first bin and then falls."
        ),
        "caveat": "Again, total path length hides which component is helpful or harmful.",
    },
    "contour_orthogonal_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png": {
        "heading": "Current component view: strong contours, orientation-orthogonal units",
        "body": (
            "For orthogonal high-SF units, the component pattern flips relative to the aligned case. "
            "Across-contour image motion can reduce SSI, while along-contour image motion can initially boost it because it is closer to motion across the unit's preferred orientation."
        ),
        "caveat": "The interpretation depends on translating image-contour axes into the unit's preferred-orientation frame.",
    },
    "all_images_absolute_contour_matched_delta_low_middle_high_sf_six_panel.png": {
        "heading": "Mixed-context comparison panel",
        "body": (
            "This panel places a broad all-image absolute SSI view next to a contour-conditioned movement-modulation view. "
            "It is useful for presentation context, showing both overall information scale and the contour-specific movement effect."
        ),
        "caveat": "The left and right columns use different selection rules, so compare qualitative patterns rather than exact magnitudes.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def unique_existing(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def key_figure_paths(matrix_dir: Path) -> list[Path]:
    phase_dir = matrix_dir / "phase1_phase2_conditioning_v1"
    phase_fig_dir = phase_dir / "figures"
    pilot_fig_dir = matrix_dir / "pilot_figures"
    schematic_fig_dir = (
        phase_dir
        / "schematic_pathlength_summary_v1"
        / "unit_first_and_population_v1"
        / "figures"
    )
    stems = [
        pilot_fig_dir / "population_ssi_vs_trace_path_length.png",
        pilot_fig_dir / "sf_group_ssi_vs_trace_path_length.png",
        pilot_fig_dir / "population_ssi_distribution_by_microsaccade.png",
        pilot_fig_dir / "unit_by_trace_ssi_heatmap.png",
        phase_fig_dir / "phase1_feature_qc_distributions.png",
        phase_fig_dir / "phase2_population_by_microsaccade_trace_path.png",
        phase_fig_dir / "phase2_sf_groups_by_trace_path.png",
        phase_fig_dir / "phase2_image_contour_class_by_trace_path.png",
        phase_fig_dir / "phase2_unit_image_orientation_match_by_trace_path.png",
        phase_fig_dir / "phase2_unit_image_orientation_match_contour_images_by_trace_path.png",
        phase_fig_dir / "phase2_trace_image_axis_by_trace_path.png",
        phase_fig_dir / "phase2_real_trace_sf_contour_matched_low_high_scale_curves.png",
        phase_fig_dir / "phase2_real_trace_sf_contour_orthogonal_low_high_scale_curves.png",
        phase_fig_dir / "phase2_real_trace_sf_contour_aligned_vs_orthogonal_low_high_scale_curves.png",
        phase_dir / "phase2_alignment_response_preferred_vs_orthogonal.png",
        schematic_fig_dir / "all_images_no_osi_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png",
        schematic_fig_dir / "all_images_no_osi_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
        schematic_fig_dir / "strong_contours_no_osi_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png",
        schematic_fig_dir / "strong_contours_no_osi_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
        schematic_fig_dir / "contour_matched_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png",
        schematic_fig_dir / "contour_matched_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
        schematic_fig_dir / "contour_intermediate_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png",
        schematic_fig_dir / "contour_intermediate_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
        schematic_fig_dir / "contour_orthogonal_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel.png",
        schematic_fig_dir / "contour_orthogonal_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel.png",
        schematic_fig_dir / "all_images_absolute_contour_matched_delta_low_middle_high_sf_six_panel.png",
    ]
    return unique_existing(stems)


def appendix_figure_paths(matrix_dir: Path) -> list[Path]:
    phase_dir = matrix_dir / "phase1_phase2_conditioning_v1"
    figure_dirs = [
        matrix_dir / "pilot_figures",
        phase_dir / "figures",
        phase_dir / "schematic_pathlength_summary_v1" / "unit_first_and_population_v1" / "figures",
    ]
    paths: list[Path] = []
    for figure_dir in figure_dirs:
        if figure_dir.exists():
            paths.extend(sorted(figure_dir.glob("*.png")))
    return unique_existing(paths)


def add_text_page(pdf: PdfPages, title: str, lines: list[str], *, dpi: int) -> None:
    fig = plt.figure(figsize=(8.5, 11.0))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", ha="left", va="top")
    y = 0.89
    for line in lines:
        fig.text(0.08, y, line, fontsize=8.5, ha="left", va="top")
        y -= 0.018
        if y < 0.06:
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11.0))
            fig.patch.set_facecolor("white")
            y = 0.94
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def add_image_page(pdf: PdfPages, path: Path, *, label: str, dpi: int) -> None:
    img = mpimg.imread(path)
    height, width = img.shape[:2]
    aspect = float(width) / max(float(height), 1.0)
    if aspect >= 1.0:
        fig_w = 11.0
        fig_h = min(8.5, max(4.5, 10.6 / aspect + 0.55))
    else:
        fig_h = 11.0
        fig_w = min(8.5, max(5.5, 10.5 * aspect))
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.015, 0.075, 0.97, 0.9])
    ax.imshow(img)
    ax.set_axis_off()
    fig.text(0.5, 0.028, label, ha="center", va="center", fontsize=7, color="0.25")
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def caption_lines(path: Path, *, width: int = 138) -> list[str]:
    caption = KEY_CAPTIONS.get(path.name)
    if caption is None:
        return ["No curated caption is available for this figure."]
    lines: list[str] = []
    heading = str(caption["heading"])
    body = str(caption["body"])
    caveat = str(caption["caveat"])
    lines.append(heading)
    for prefix, text in [("Interpretation: ", body), ("Caveat: ", caveat)]:
        wrapped = textwrap.wrap(
            prefix + text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped)
    return lines


def add_interpreted_image_page(pdf: PdfPages, path: Path, *, label: str, dpi: int) -> None:
    img = mpimg.imread(path)
    height, width = img.shape[:2]
    aspect = float(width) / max(float(height), 1.0)
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    caption = caption_lines(path)

    image_bottom = 0.24
    image_top = 0.965
    image_left = 0.035
    image_width = 0.93
    image_height = image_top - image_bottom
    box_aspect = image_width * 11.0 / (image_height * 8.5)
    if aspect > box_aspect:
        used_h = image_width * 11.0 / aspect / 8.5
        used_y = image_bottom + (image_height - used_h) * 0.5
        ax = fig.add_axes([image_left, used_y, image_width, used_h])
    else:
        used_w = image_height * 8.5 * aspect / 11.0
        used_x = image_left + (image_width - used_w) * 0.5
        ax = fig.add_axes([used_x, image_bottom, used_w, image_height])
    ax.imshow(img)
    ax.set_axis_off()

    fig.text(0.04, 0.205, caption[0], ha="left", va="top", fontsize=10.2, weight="bold", color="0.1")
    y = 0.178
    for line in caption[1:]:
        fig.text(0.04, y, line, ha="left", va="top", fontsize=8.1, color="0.18")
        y -= 0.021
    fig.text(0.5, 0.026, label, ha="center", va="center", fontsize=6.6, color="0.35")
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def write_collection(paths: list[Path], out_path: Path, *, title: str, root: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        add_text_page(
            pdf,
            title,
            [
                f"{len(paths)} figures",
                f"Source root: {root}",
                "",
                "Contents:",
                *[f"{idx:03d}. {path.relative_to(root)}" for idx, path in enumerate(paths, start=1)],
            ],
            dpi=dpi,
        )
        for idx, path in enumerate(paths, start=1):
            add_image_page(pdf, path, label=f"{idx:03d}. {path.relative_to(root)}", dpi=dpi)


def write_interpreted_key_collection(
    paths: list[Path],
    out_path: Path,
    *,
    title: str,
    root: Path,
    dpi: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        add_text_page(
            pdf,
            title,
            [
                f"{len(paths)} figures",
                f"Source root: {root}",
                "",
                "Notes for interpretation:",
                "SSI means stimulus-specific information, reported in bits/spike.",
                "Movement modulation means SSI minus the counterfactual stabilized baseline.",
                "Where shown, p-values compare the stabilized baseline with the smallest drift-only trace bin.",
                "Spike-weighted population SSI can change because SSI changes within a selected group, and because movement changes expected-spike weights.",
                "",
                "Contents:",
                *[f"{idx:03d}. {KEY_CAPTIONS.get(path.name, {}).get('heading', path.name)}" for idx, path in enumerate(paths, start=1)],
            ],
            dpi=dpi,
        )
        for idx, path in enumerate(paths, start=1):
            add_interpreted_image_page(pdf, path, label=f"{idx:03d}. {path.relative_to(root)}", dpi=dpi)



def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    key_paths = key_figure_paths(matrix_dir)
    appendix_paths = appendix_figure_paths(matrix_dir)
    key_pdf = out_dir / "backimage_real_trace_key_figures_multipage.pdf"
    appendix_pdf = out_dir / "backimage_real_trace_all_png_figures_appendix_multipage.pdf"
    write_interpreted_key_collection(
        key_paths,
        key_pdf,
        title="BackImage Real-Trace SSI: Key Figures With Interpretation Notes",
        root=matrix_dir,
        dpi=int(args.dpi),
    )
    write_collection(
        appendix_paths,
        appendix_pdf,
        title="BackImage Real-Trace SSI: All PNG Figure Appendix",
        root=matrix_dir,
        dpi=int(args.dpi),
    )
    manifest = {
        "matrix_dir": str(matrix_dir),
        "out_dir": str(out_dir),
        "key_pdf": str(key_pdf),
        "appendix_pdf": str(appendix_pdf),
        "key_pdf_has_interpretation_notes": True,
        "n_key_figures": len(key_paths),
        "n_appendix_figures": len(appendix_paths),
        "key_figures": [str(path) for path in key_paths],
        "appendix_figures": [str(path) for path in appendix_paths],
    }
    manifest_path = out_dir / "backimage_real_trace_plot_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {key_pdf}")
    print(f"Wrote {appendix_pdf}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
