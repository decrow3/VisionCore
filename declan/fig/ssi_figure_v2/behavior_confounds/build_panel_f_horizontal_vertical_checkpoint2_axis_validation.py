#!/usr/bin/env python3
"""Independent image-axis validation for the frozen Figure 4F H/V checkpoint.

This stage does not alter the frozen Sobel/structure-tensor cohort.  It asks
whether a separately computed Fourier-spectrum orientation supports those
labels, and makes weak independent estimates visible rather than silently
turning them into exclusions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint1 import (
    BIN_HALF_WIDTH_DEG,
    COHERENCE_THRESHOLD,
    ROOT,
    SUBJECTS,
    axial_distance_deg,
    axial_signed_deg,
    extract_patch,
    load_values,
)


PARENT_OUT = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_horizontal_vertical_decisive_v1"
)
SELECTION = PARENT_OUT / "checkpoint1_raw/checkpoint1_random_example_selection.csv"
DEFAULT_OUT = PARENT_OUT / "checkpoint2_axis_validation"
SPECTRUM_READABLE_THRESHOLD = 0.2
AGREEMENT_THRESHOLD_DEG = 22.5
COLORS = {"horizontal": "#2878B5", "vertical": "#D55E4B"}


def add_independent_axis(values: pd.DataFrame) -> pd.DataFrame:
    out = values.copy()
    # The stored spectrum orientation is the frequency-domain orientation,
    # which is perpendicular to the corresponding spatial contour axis.
    out["spectrum_contour_axis_deg"] = (
        pd.to_numeric(out["image_spectrum_orientation_deg"], errors="coerce") + 90.0
    ) % 180.0
    out["sobel_spectrum_axis_disagreement_deg"] = axial_distance_deg(
        pd.to_numeric(out["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float),
        out["spectrum_contour_axis_deg"].to_numpy(dtype=float),
    )
    out["spectrum_axis_readable"] = (
        pd.to_numeric(out["image_spectrum_anisotropy"], errors="coerce")
        .ge(SPECTRUM_READABLE_THRESHOLD)
        .fillna(False)
    )
    out["independent_axis_agrees"] = (
        out["sobel_spectrum_axis_disagreement_deg"].le(AGREEMENT_THRESHOLD_DEG)
        & out["spectrum_axis_readable"]
    )
    out["axis_validation_class"] = np.select(
        [
            ~out["spectrum_axis_readable"],
            out["sobel_spectrum_axis_disagreement_deg"].le(AGREEMENT_THRESHOLD_DEG),
        ],
        ["independent axis weak/ambiguous", "independent estimators agree"],
        default="readable estimators disagree",
    )
    return out


def population_summary(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subject in SUBJECTS:
        for condition in ("horizontal", "vertical"):
            block = values[
                values["subject"].astype(str).eq(subject)
                & values["hv_condition"].eq(condition)
            ].copy()
            finite = block[np.isfinite(block["sobel_spectrum_axis_disagreement_deg"])].copy()
            readable = finite[finite["spectrum_axis_readable"]].copy()
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "n_windows": int(len(block)),
                    "n_with_finite_spectrum_axis": int(len(finite)),
                    "median_axis_disagreement_deg_all": float(
                        finite["sobel_spectrum_axis_disagreement_deg"].median()
                    ),
                    "fraction_agreement_le_22p5_all": float(
                        finite["sobel_spectrum_axis_disagreement_deg"].le(AGREEMENT_THRESHOLD_DEG).mean()
                    ),
                    "n_spectrum_anisotropy_ge_0p2": int(len(readable)),
                    "fraction_spectrum_anisotropy_ge_0p2": float(len(readable) / len(finite)),
                    "median_axis_disagreement_deg_readable": float(
                        readable["sobel_spectrum_axis_disagreement_deg"].median()
                    ),
                    "fraction_agreement_le_22p5_readable": float(
                        readable["sobel_spectrum_axis_disagreement_deg"].le(AGREEMENT_THRESHOLD_DEG).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def draw_image_axis(
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    angle_deg: float,
    length: float,
    **kwargs: object,
) -> None:
    theta = np.radians(angle_deg)
    dx, dy = length * np.cos(theta), length * np.sin(theta)
    ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy], **kwargs)


def render_frozen_vertical_examples(selected: pd.DataFrame, out_dir: Path) -> Path:
    vertical = selected[selected["hv_condition"].eq("vertical")].copy()
    fig, axes = plt.subplots(2, 5, figsize=(13.8, 6.3), constrained_layout=True)
    for row_index, subject in enumerate(SUBJECTS):
        block = vertical[vertical["subject"].astype(str).eq(subject)].sort_values("selection_rank")
        for column_index, (_, row) in enumerate(block.iterrows()):
            ax = axes[row_index, column_index]
            patch = extract_patch(row)
            ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
            center_y, center_x = (np.asarray(patch.shape) - 1.0) / 2.0
            length = 0.40 * min(patch.shape)
            draw_image_axis(
                ax, center_x, center_y, float(row.image_edge_axis_deg), length,
                color="#20A464", lw=2.1, label="Sobel contour axis", zorder=5,
            )
            draw_image_axis(
                ax, center_x, center_y, float(row.spectrum_contour_axis_deg), length,
                color="#A33FA3", lw=1.8, ls="--", label="Fourier contour axis", zorder=6,
            )
            ax.scatter([center_x], [center_y], marker="+", s=32, c="#E23D3D", linewidths=1.2, zorder=7)
            title_color = {
                "independent estimators agree": "#176B45",
                "independent axis weak/ambiguous": "#9A6515",
                "readable estimators disagree": "#A12B2B",
            }[str(row.axis_validation_class)]
            ax.set_title(
                f"#{int(row.selection_rank)}  Δaxis {row.sobel_spectrum_axis_disagreement_deg:.1f}°\n"
                f"Sobel coh {row.image_orientation_coherence:.2f}; Fourier aniso {row.image_spectrum_anisotropy:.2f}",
                fontsize=8.2, color=title_color, weight="bold",
            )
            ax.axis("off")
        axes[row_index, 0].set_ylabel(subject, fontsize=10.0, weight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "The same frozen random vertical examples, checked with an independent image-axis estimator\n"
        "green solid = Sobel/structure-tensor axis; magenta dashed = Fourier-spectrum contour axis",
        fontsize=12.0, weight="bold",
    )
    path = out_dir / "checkpoint2_frozen_vertical_examples_with_two_axes.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def render_disagreement_cdfs(values: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True, constrained_layout=True)
    for ax, subject in zip(axes, SUBJECTS, strict=True):
        for condition in ("horizontal", "vertical"):
            block = values[
                values["subject"].astype(str).eq(subject)
                & values["hv_condition"].eq(condition)
                & values["spectrum_axis_readable"]
            ]
            disagreement = np.sort(block["sobel_spectrum_axis_disagreement_deg"].to_numpy(dtype=float))
            cumulative = np.arange(1, len(disagreement) + 1) / len(disagreement)
            agreement = np.mean(disagreement <= AGREEMENT_THRESHOLD_DEG)
            ax.step(
                disagreement, cumulative, where="post", color=COLORS[condition], lw=1.8,
                label=f"{condition}: {agreement:.1%} ≤22.5° (n={len(block)})",
            )
        ax.axvline(AGREEMENT_THRESHOLD_DEG, color="#7F858C", lw=0.9, ls=":")
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1.01)
        ax.set_xticks([0, 15, 22.5, 45, 67.5, 90])
        ax.set_title(subject, weight="bold")
        ax.set_xlabel("absolute Sobel–Fourier contour-axis disagreement (deg)")
        ax.grid(color="#D8DDE3", lw=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=7.6)
    axes[0].set_ylabel("cumulative fraction of readable Fourier axes")
    fig.suptitle(
        "Independent orientation validation in the complete frozen H/V cohort\n"
        "readable Fourier axis = spectrum anisotropy ≥0.2",
        fontsize=11.2, weight="bold",
    )
    path = out_dir / "checkpoint2_axis_disagreement_cdfs.png"
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    values = add_independent_axis(load_values())
    selected = add_independent_axis(pd.read_csv(SELECTION))
    summary = population_summary(values)
    examples_path = render_frozen_vertical_examples(selected, args.out_dir)
    cdf_path = render_disagreement_cdfs(values, args.out_dir)

    summary.to_csv(args.out_dir / "checkpoint2_population_axis_agreement.csv", index=False)
    selected[selected["hv_condition"].eq("vertical")].to_csv(
        args.out_dir / "checkpoint2_frozen_vertical_example_axis_agreement.csv", index=False
    )

    report = [
        "# Horizontal-versus-vertical Figure 4F: independent axis validation",
        "",
        "The primary cohort and its Sobel/structure-tensor orientation bins remain frozen.",
        "The Fourier-spectrum estimate is an independent validation and sensitivity variable, not a",
        "post-hoc replacement label. Its stored frequency orientation was rotated by 90° to obtain",
        "the corresponding spatial contour axis.",
        "",
        "## Complete-cohort agreement",
        "",
    ]
    for row in summary.itertuples(index=False):
        report.append(
            f"- {row.subject}, {row.condition}: median disagreement {row.median_axis_disagreement_deg_all:.2f}°; "
            f"{row.fraction_agreement_le_22p5_all:.1%} within 22.5° across all finite estimates. "
            f"Fourier anisotropy was ≥0.2 in {row.fraction_spectrum_anisotropy_ge_0p2:.1%}; within that "
            f"readable subset, {row.fraction_agreement_le_22p5_readable:.1%} agreed within 22.5° "
            f"(n={row.n_spectrum_anisotropy_ge_0p2})."
        )
    vertical_examples = selected[selected["hv_condition"].eq("vertical")]
    class_counts = vertical_examples["axis_validation_class"].value_counts()
    report.extend(
        [
            "",
            "## Frozen random vertical examples",
            "",
            *[f"- {label}: {int(count)} of 10" for label, count in class_counts.items()],
            "",
            "The raw-sheet concern is therefore not dismissed: weak Fourier anisotropy identifies",
            "examples for which a single independent axis is not visually or numerically reliable.",
            "The next inferential checkpoint should report the frozen primary test and, separately,",
            "a sensitivity restricted to readable, independently agreeing image axes.",
            "",
            f"Examples: `{examples_path.name}`",
            f"Population CDFs: `{cdf_path.name}`",
        ]
    )
    (args.out_dir / "summary_report.md").write_text("\n".join(report) + "\n")
    print(examples_path)
    print(cdf_path)
    print(args.out_dir / "summary_report.md")


if __name__ == "__main__":
    main()
