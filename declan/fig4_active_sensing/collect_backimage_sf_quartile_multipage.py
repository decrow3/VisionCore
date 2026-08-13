#!/usr/bin/env python3
"""Collect the updated SF-quartile BackImage sequence into a fresh 27-page PDF."""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from declan.active_sensing_movie_information.collect_backimage_real_trace_plot_pdfs import (
    KEY_CAPTIONS,
    add_text_page,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    DEFAULT_MATRIX_DIR,
    ROOT,
    file_identity,
)


BASE = ROOT / "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1"
DEFAULT_OUT = BASE / "backimage_real_trace_key_figures_sf_quartiles_multipage_v1.pdf"
DEFAULT_MANIFEST = BASE / "backimage_real_trace_key_figures_sf_quartiles_multipage_v1_manifest.json"
DEFAULT_ORIGINAL = Path(
    "/home/declan/.codex/attachments/32cf1ea5-f4e7-4620-9fd1-c60b914176cd/"
    "backimage_real_trace_key_figures_multipage.pdf"
)


@dataclass(frozen=True)
class Page:
    path: Path
    heading: str
    body: str
    caveat: str


def original_page(path: Path) -> Page:
    caption = KEY_CAPTIONS[path.name]
    return Page(path, str(caption["heading"]), str(caption["body"]), str(caption["caveat"]))


def updated_pages(matrix_dir: Path) -> list[Page]:
    phase = matrix_dir / "phase1_phase2_conditioning_v1"
    pilot = matrix_dir / "pilot_figures"
    figures = phase / "figures"
    return [
        original_page(pilot / "population_ssi_vs_trace_path_length.png"),
        Page(
            BASE / "checkpoint_01_pilot_sf_by_trace_path/002_pilot_sf_quartile_ssi_vs_trace_path_length.png",
            "Pilot: new SF quartiles by trace path length",
            "The pilot SF split now uses the four tie-aware quartiles from the new parametric SF fits, separating Q2 and Q3 rather than merging them into a historical middle group.",
            "This pilot still pools over image content and unit-contour geometry.",
        ),
        original_page(pilot / "population_ssi_distribution_by_microsaccade.png"),
        original_page(pilot / "unit_by_trace_ssi_heatmap.png"),
        original_page(figures / "phase1_feature_qc_distributions.png"),
        original_page(figures / "phase2_population_by_microsaccade_trace_path.png"),
        Page(
            BASE / "checkpoint_02_phase2_sf_by_trace_path/007_phase2_sf_quartiles_by_trace_path.png",
            "Spatial-frequency quartiles by real-trace path length",
            "The phase-2 summary uses the new SF quartiles and retains separate drift-only and microsaccade contexts, exposing distinct Q2 and Q3 path-length profiles.",
            "Total path length still combines contour-relative motion components.",
        ),
        original_page(figures / "phase2_image_contour_class_by_trace_path.png"),
        original_page(figures / "phase2_unit_image_orientation_match_by_trace_path.png"),
        original_page(figures / "phase2_unit_image_orientation_match_contour_images_by_trace_path.png"),
        original_page(figures / "phase2_trace_image_axis_by_trace_path.png"),
        Page(
            BASE / "checkpoint_03_contour_matched_sf_quartiles/012_phase2_contour_matched_sf_q1_q4_curves.png",
            "Earlier contour-matched comparison with new SF fits",
            "This reruns the earlier aligned unit-first contrast using the lowest and highest new SF quartiles, with all-quartile and selection audits saved alongside it.",
            "This is an earlier unit-first conditioning view, not the later spike-weighted population estimand.",
        ),
        Page(
            BASE / "checkpoint_04_contour_orthogonal_sf_quartiles/013_phase2_contour_orthogonal_sf_q1_q4_curves.png",
            "Earlier contour-orthogonal comparison with new SF fits",
            "This applies the same new-fit Q1/Q4 contrast to unit-image pairs whose preferred orientation is orthogonal to the local contour.",
            "Interpret together with the aligned view and its unit-selection audit.",
        ),
        Page(
            BASE / "checkpoint_05_aligned_vs_orthogonal_sf_quartiles/014_phase2_contour_aligned_vs_orthogonal_sf_q1_q4_curves.png",
            "Earlier aligned-versus-orthogonal comparison with new SF fits",
            "The aligned and orthogonal curves are overlaid under the updated Q1/Q4 assignments, with paired relation contrasts retained in the checkpoint outputs.",
            "The later component-path panels are cleaner for contour-relative geometry.",
        ),
        Page(
            BASE / "checkpoint_06_alignment_response_sf_quartiles/015_phase2_alignment_response_sf_q1_q4.png",
            "Response-strength control with new SF quartiles",
            "Mean rate, expected spikes, and SSI are compared for preferred/aligned and orthogonal image windows under the updated SF assignments.",
            "Response magnitude and bits per spike are related but not interchangeable.",
        ),
        Page(
            BASE / "checkpoint_07_all_images_population_sf_quartiles/016_all_images_valid_fit_sf_quartiles_population_absolute_delta.png",
            "All images and valid-fit units: SF-quartile population summary",
            "Across all images, Q1 is negative, Q2 is sustained positive, Q3 crosses from positive to negative, and Q4 shows a transient positive response that weakens with path length.",
            "Spike-weighted and equal-unit estimates diverge for several quartiles; see the checkpoint weighting audit.",
        ),
        Page(
            BASE / "checkpoint_08_all_images_component_sf_quartiles/017_all_images_valid_fit_sf_quartiles_across_along_components.png",
            "All images and valid-fit units: component paths",
            "Across- and along-contour component curves are broadly similar when all images and unit-contour relationships are pooled; Q3 shows the clearest early component difference.",
            "Along and across are defined relative to each image contour, not each unit preference.",
        ),
        Page(
            BASE / "checkpoint_09_strong_contours_population_sf_quartiles/018_strong_contours_valid_fit_sf_quartiles_population_absolute_delta.png",
            "Strong-contour images: SF-quartile population summary",
            "Q2 remains robustly positive and Q3 shows a clear crossover. Q4 approaches zero at longer paths rather than becoming cleaner under the strong-contour restriction.",
            "Q4 remains positive in the equal-unit audit, indicating a spike-weighting contribution to its flattening.",
        ),
        Page(
            BASE / "checkpoint_10_strong_contours_component_sf_quartiles/019_strong_contours_valid_fit_sf_quartiles_across_along_components.png",
            "Strong-contour images: component paths",
            "Q1 and Q2 show little across/along separation. Q3 has an early across-contour advantage that disappears with larger components; Q4 remains weak and ambiguous.",
            "Component-wise baseline tests do not directly test the paired across-minus-along contrast.",
        ),
        Page(
            BASE / "checkpoint_11_contour_matched_population_sf_quartiles/020_contour_matched_valid_fit_sf_quartiles_population_absolute_delta.png",
            "Strong contours and orientation-aligned units",
            "Alignment sharply separates the upper quartiles: Q3 becomes increasingly negative, whereas Q4 is robustly and persistently positive.",
            "Q1 and Q2 are more sensitive to spike weighting than Q3 and Q4.",
        ),
        Page(
            BASE / "checkpoint_12_contour_matched_component_sf_quartiles/021_contour_matched_valid_fit_sf_quartiles_across_along_components.png",
            "Orientation-aligned units: component paths",
            "Aligned Q4 is positive for both components, with a larger shortest-drift response across the contour. Q3 is near zero across and negative along at the first bin.",
            "The old categorical claim that across helps while along hurts Q4 is not retained by the new fits.",
        ),
        Page(
            BASE / "checkpoint_13_contour_intermediate_population_sf_quartiles/022_contour_intermediate_valid_fit_sf_quartiles_population_absolute_delta.png",
            "Strong contours and intermediate-orientation units",
            "The strongest short-path positive effect occurs in Q3 rather than Q4. Q3's longer-path population result becomes sensitive to spike weighting.",
            "Intermediate refers to local unit-contour mismatch, not screen-oblique orientation.",
        ),
        Page(
            BASE / "checkpoint_14_contour_intermediate_component_sf_quartiles/023_contour_intermediate_valid_fit_sf_quartiles_across_along_components.png",
            "Intermediate-orientation units: component paths",
            "Q3 is positive in both across and along components, with only a modest early across advantage; Q4 has smaller positive effects.",
            "This favors a broader motion/context effect over a stable single-axis mechanism.",
        ),
        Page(
            BASE / "checkpoint_15_contour_orthogonal_population_sf_quartiles/024_contour_orthogonal_valid_fit_sf_quartiles_population_absolute_delta.png",
            "Strong contours and orientation-orthogonal units",
            "Q2 is robustly positive, Q3 becomes strongly negative with path length, and Q4 is near zero initially before becoming negative at larger paths.",
            "Q3 and Q4 show substantial population-versus-equal-unit weighting differences.",
        ),
        Page(
            BASE / "checkpoint_16_contour_orthogonal_component_sf_quartiles/025_contour_orthogonal_valid_fit_sf_quartiles_across_along_components.png",
            "Orientation-orthogonal units: component paths",
            "Q4 shows the expected descriptive directional flip at the first drift bin: across-contour modulation is negative and along-contour modulation positive. Q2 is positive in both and Q3 negative in both.",
            "Both Q4 first-bin intervals cross zero, so the flip is suggestive rather than conclusive.",
        ),
        Page(
            BASE / "checkpoint_17_mixed_context_sf_quartiles/026_all_images_absolute_contour_matched_delta_valid_fit_sf_quartiles.png",
            "Mixed-context comparison with four SF quartiles",
            "The final presentation panel places broad all-image absolute SSI beside orientation-aligned movement modulation, now retaining Q1 through Q4 separately.",
            "The two columns intentionally use different selection rules; compare shapes and signs, not exact magnitudes.",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--original-pdf", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def add_page(pdf: PdfPages, page: Page, *, index: int, dpi: int) -> None:
    img = mpimg.imread(page.path)
    height, width = img.shape[:2]
    aspect = float(width) / max(float(height), 1.0)
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    image_bottom, image_top = 0.24, 0.965
    image_left, image_width = 0.035, 0.93
    image_height = image_top - image_bottom
    box_aspect = image_width * 11.0 / (image_height * 8.5)
    if aspect > box_aspect:
        used_h = image_width * 11.0 / aspect / 8.5
        ax = fig.add_axes([image_left, image_bottom + (image_height - used_h) * 0.5, image_width, used_h])
    else:
        used_w = image_height * 8.5 * aspect / 11.0
        ax = fig.add_axes([image_left + (image_width - used_w) * 0.5, image_bottom, used_w, image_height])
    ax.imshow(img)
    ax.set_axis_off()
    fig.text(0.04, 0.205, page.heading, ha="left", va="top", fontsize=10.2, weight="bold", color="0.1")
    y = 0.178
    for prefix, text in (("Interpretation: ", page.body), ("Caveat: ", page.caveat)):
        for line in textwrap.wrap(prefix + text, width=100, break_long_words=False, break_on_hyphens=False):
            fig.text(0.04, y, line, ha="left", va="top", fontsize=8.1, color="0.18")
            y -= 0.021
    try:
        display_path = page.path.resolve().relative_to(ROOT)
    except ValueError:
        display_path = page.path.name
    fig.text(0.5, 0.026, f"{index:03d}. {display_path}", ha="center", va="center", fontsize=6.2, color="0.35")
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir).resolve()
    pages = updated_pages(matrix_dir)
    if len(pages) != 26:
        raise ValueError(f"Expected 26 figure pages, found {len(pages)}")
    missing = [str(page.path) for page in pages if not page.path.exists()]
    if missing:
        raise FileNotFoundError("Missing figure pages:\n" + "\n".join(missing))
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        add_text_page(
            pdf,
            "BackImage SSI: New SF-Quartile Checks",
            [
                "26 figures; 27 pages including this contents page",
                "SF assignments: rr100_sf_tf_parametric_model_arrays.npz, tie-aware quartiles",
                "Historical trial-mean-stabilized estimands are preserved in the iterated-check sequence.",
                "Unchanged non-SF pages are retained from the source analysis outputs.",
                "Original reference PDF preserved unchanged; identity and path are recorded in the manifest.",
                "",
                "Contents:",
                *[f"{idx:03d}. {page.heading}" for idx, page in enumerate(pages, start=1)],
            ],
            dpi=int(args.dpi),
        )
        for idx, page in enumerate(pages, start=1):
            add_page(pdf, page, index=idx, dpi=int(args.dpi))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_pdf": file_identity(Path(args.out_pdf)),
        "original_pdf_unchanged": file_identity(Path(args.original_pdf)),
        "n_figure_pages": len(pages),
        "n_total_pages_expected": len(pages) + 1,
        "figures": [
            {"index": idx, "path": str(page.path.resolve()), "heading": page.heading}
            for idx, page in enumerate(pages, start=1)
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_pdf.resolve()}")
    print(f"Wrote {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
