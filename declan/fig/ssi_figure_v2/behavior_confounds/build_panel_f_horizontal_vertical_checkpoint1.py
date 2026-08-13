#!/usr/bin/env python3
"""Frozen map-first checkpoint for the Figure 4F horizontal/vertical test.

This stage deliberately renders raw, outcome-blind examples and screen-frame
axis distributions before running any new population inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import _window_trace


ROOT = Path(__file__).resolve().parents[4]
INPUT = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_horizontal_vertical_decisive_v1"
    / "checkpoint1_raw"
)
SUBJECTS = ("Allen", "Logan")
CONDITIONS = ("horizontal", "vertical")
CENTERS = {"horizontal": 0.0, "vertical": 90.0}
COLORS = {"horizontal": "#2878B5", "vertical": "#D55E4B"}
COHERENCE_THRESHOLD = 0.3
BIN_HALF_WIDTH_DEG = 22.5
N_EXAMPLES_PER_CELL = 5
SEED = 20260810


def axial_signed_deg(value: np.ndarray | float) -> np.ndarray | float:
    out = (np.asarray(value, dtype=float) + 90.0) % 180.0 - 90.0
    return float(out) if out.ndim == 0 else out


def axial_distance_deg(a: np.ndarray | float, b: float) -> np.ndarray | float:
    return np.abs(axial_signed_deg(np.asarray(a, dtype=float) - b))


def axial_mean_deg(values: np.ndarray) -> float:
    theta = 2.0 * np.radians(np.asarray(values, dtype=float))
    return float(axial_signed_deg(0.5 * np.degrees(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))))


def load_values() -> pd.DataFrame:
    values = pd.read_csv(INPUT).copy()
    values = values[
        values["image_feature_ok"].fillna(False).astype(bool)
        & values["image_orientation_coherence"].ge(COHERENCE_THRESHOLD)
        & values["image_patch_fraction_inside_image"].ge(0.999)
        & values["image_patch_fraction_background"].le(0.05)
    ].copy()
    axis = np.mod(values["image_edge_axis_deg"].to_numpy(dtype=float), 180.0)
    horizontal = axial_distance_deg(axis, 0.0) <= BIN_HALF_WIDTH_DEG
    vertical = axial_distance_deg(axis, 90.0) <= BIN_HALF_WIDTH_DEG
    values["hv_condition"] = np.where(horizontal, "horizontal", np.where(vertical, "vertical", "other"))
    values = values[values["hv_condition"].isin(CONDITIONS)].copy()
    return values.reset_index(drop=True)


def select_random_examples(values: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[pd.DataFrame] = []
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            block = values[
                values["subject"].astype(str).eq(subject)
                & values["hv_condition"].eq(condition)
            ].copy()
            block["random_trial_key"] = rng.random(len(block))
            # Overlapping windows from one trial cannot occupy multiple example slots.
            trial_candidates = (
                block.sort_values("random_trial_key")
                .drop_duplicates(["session", "trial_idx"], keep="first")
                .sort_values("random_trial_key")
            )
            if len(trial_candidates) < N_EXAMPLES_PER_CELL:
                raise RuntimeError(f"Insufficient {subject} {condition} trial support")
            selected = trial_candidates.head(N_EXAMPLES_PER_CELL).copy()
            selected["selection_rank"] = np.arange(1, len(selected) + 1)
            selected["selection_role"] = f"random_{condition}"
            rows.append(selected)
    selected = pd.concat(rows, ignore_index=True)
    selected["selection_seed"] = SEED
    selected["selection_rule"] = (
        "coherence>=0.3; axis within 22.5deg of frozen center; valid patch; "
        "one random window per trial; lowest fixed-seed random keys"
    )
    return selected


def extract_patch(row: pd.Series) -> np.ndarray:
    canvas, _ppd, _shape = _backimage_canvas(str(row.session), int(row.trial_idx))
    cx = int(round(float(row.image_patch_center_x_px)))
    cy = int(round(float(row.image_patch_center_y_px)))
    radius = int(row.image_patch_radius_px)
    return np.asarray(canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1])


def covariance_ellipse(row: pd.Series) -> tuple[float, float, float]:
    covariance = np.asarray(
        [[row.cov_xx_deg2, row.cov_xy_deg2], [row.cov_xy_deg2, row.cov_yy_deg2]],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    major = eigenvectors[:, order[0]]
    angle = float(axial_signed_deg(np.degrees(np.arctan2(major[1], major[0]))))
    return 4.0 * np.sqrt(eigenvalues[0]), 4.0 * np.sqrt(eigenvalues[1]), angle


def draw_axis(ax: plt.Axes, angle_deg: float, length: float, color: str, lw: float) -> None:
    theta = np.radians(angle_deg)
    dx, dy = length * np.cos(theta), length * np.sin(theta)
    ax.plot([-dx, dx], [-dy, dy], color=color, lw=lw, zorder=6)


def render_examples(selected: pd.DataFrame, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    traces: dict[int, np.ndarray] = {}
    patches: dict[int, np.ndarray] = {}
    audit_rows = []
    for index, row in selected.iterrows():
        trace = np.asarray(_window_trace(row), dtype=float)
        patch = extract_patch(row)
        traces[index] = trace
        patches[index] = patch
        centered = trace - np.mean(trace, axis=0)
        width, height, fem_axis = covariance_ellipse(row)
        audit_rows.append(
            {
                "selected_row": index,
                "trace_samples": int(len(trace)),
                "trace_finite": bool(np.isfinite(trace).all()),
                "trace_centered_max_abs_deg": float(np.max(np.abs(centered))),
                "ellipse_width_deg_2sigma": width,
                "ellipse_height_deg_2sigma": height,
                "recomputed_fem_axis_deg": fem_axis,
                "stored_fem_axis_deg": float(row.drift_orientation_deg),
                "axis_recompute_error_deg": float(axial_distance_deg(fem_axis, float(row.drift_orientation_deg))),
            }
        )
    audit = pd.DataFrame(audit_rows)
    common_limit = max(
        0.12,
        max(float(np.max(np.abs(trace - np.mean(trace, axis=0)))) for trace in traces.values()) * 1.08,
    )

    fig, axes = plt.subplots(
        4,
        2 * N_EXAMPLES_PER_CELL,
        figsize=(18.0, 9.0),
        gridspec_kw={"width_ratios": [1.0, 1.0] * N_EXAMPLES_PER_CELL},
    )
    for row_index, (subject, condition) in enumerate(
        [(subject, condition) for subject in SUBJECTS for condition in CONDITIONS]
    ):
        block = selected[
            selected["subject"].astype(str).eq(subject)
            & selected["hv_condition"].eq(condition)
        ].sort_values("selection_rank")
        for example_index, (selected_index, row) in enumerate(block.iterrows()):
            patch_ax = axes[row_index, 2 * example_index]
            trace_ax = axes[row_index, 2 * example_index + 1]

            patch = patches[selected_index]
            patch_ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
            cy, cx = (np.asarray(patch.shape) - 1.0) / 2.0
            length = 0.40 * min(patch.shape)
            theta = np.radians(float(row.image_edge_axis_array_deg))
            dx, dy = length * np.cos(theta), length * np.sin(theta)
            patch_ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="#20A464", lw=1.7)
            patch_ax.scatter([cx], [cy], marker="+", s=24, c="#E23D3D", linewidths=1.0)
            patch_ax.axis("off")
            patch_ax.set_title(
                f"coh {row.image_orientation_coherence:.2f}\nedge {row.image_edge_axis_deg:+.1f}°",
                fontsize=7.0,
            )

            trace = traces[selected_index]
            centered = trace - np.mean(trace, axis=0)
            trace_ax.plot(centered[:, 0], centered[:, 1], color="#3A3D42", lw=0.65)
            trace_ax.scatter(centered[0, 0], centered[0, 1], s=12, c="#2878B5", zorder=5)
            trace_ax.scatter(centered[-1, 0], centered[-1, 1], s=13, c="#D55E4B", marker="s", zorder=5)
            width, height, fem_axis = covariance_ellipse(row)
            trace_ax.add_patch(
                Ellipse((0.0, 0.0), width=width, height=height, angle=fem_axis,
                        facecolor="#2878B5", edgecolor="#174E77", alpha=0.17, lw=0.9)
            )
            draw_axis(trace_ax, float(row.image_edge_axis_deg), 0.72 * common_limit, "#20A464", 1.5)
            draw_axis(trace_ax, fem_axis, 0.62 * common_limit, "#174E77", 1.8)
            trace_ax.set_xlim(-common_limit, common_limit)
            trace_ax.set_ylim(-common_limit, common_limit)
            trace_ax.set_aspect("equal")
            trace_ax.axhline(0, color="#C9CDD2", lw=0.45)
            trace_ax.axvline(0, color="#C9CDD2", lw=0.45)
            trace_ax.tick_params(labelsize=5.5)
            trace_ax.set_title(
                f"FEM {fem_axis:+.1f}°\nΔRMS {row.rms_delta_along_minus_across_arcmin:+.2f}'",
                fontsize=7.0,
            )
            if example_index:
                trace_ax.set_yticklabels([])
            trace_ax.set_xticks([-common_limit, 0.0, common_limit])

        axes[row_index, 0].set_ylabel(
            f"{subject}\n{condition}", rotation=0, ha="right", va="center",
            fontsize=9.0, weight="bold", color=COLORS[condition], labelpad=8,
        )

    fig.suptitle(
        "Frozen horizontal-versus-vertical checkpoint: random raw trials\n"
        "green = local contour axis; blue = FEM covariance major axis; identical screen-coordinate limits",
        fontsize=12.2,
        weight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.89, bottom=0.04, wspace=0.16, hspace=0.52)
    path = out_dir / "checkpoint1_random_horizontal_vertical_examples.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    audit["common_trace_limit_deg"] = common_limit
    return path, audit


def render_axis_distributions(values: pd.DataFrame, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    # Principal-axis angles are unstable for nearly isotropic clouds. This is a
    # visualization only, so show both all windows and the predeclared readable-axis subset.
    readable = values[values["anisotropy"].ge(0.2)].copy()
    bins = np.linspace(-90.0, 90.0, 37)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, constrained_layout=True)
    rows = []
    for ax, subject in zip(axes, SUBJECTS, strict=True):
        for condition in CONDITIONS:
            block = readable[
                readable["subject"].astype(str).eq(subject)
                & readable["hv_condition"].eq(condition)
            ]
            angles = axial_signed_deg(block["drift_orientation_deg"].to_numpy(dtype=float))
            ax.hist(
                angles, bins=bins, density=True, histtype="step", lw=1.8,
                color=COLORS[condition], label=f"{condition} contours (n={len(block)})",
            )
            mean_axis = axial_mean_deg(angles)
            ax.axvline(mean_axis, color=COLORS[condition], lw=1.2, ls="--")
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "n_windows_all": int(len(values[values["subject"].eq(subject) & values["hv_condition"].eq(condition)])),
                    "n_windows_anisotropy_ge_0p2": int(len(block)),
                    "mean_fem_axis_deg": mean_axis,
                    "fem_axis_resultant": float(np.abs(np.mean(np.exp(2j * np.radians(angles))))),
                    "mean_contour_axis_deg": axial_mean_deg(block["image_edge_axis_deg"].to_numpy(dtype=float)),
                }
            )
        ax.axvline(0.0, color="#8B9299", lw=0.65, ls=":")
        ax.set_title(subject, weight="bold")
        ax.set_xlabel("absolute FEM covariance axis (deg)")
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.grid(axis="y", color="#D8DDE3", lw=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("density (visualization: anisotropy ≥0.2)")
    fig.suptitle(
        "Does the absolute FEM axis rotate when the local contour rotates by 90°?\n"
        "Dashed lines are axial means; no outcome-based matching or reweighting",
        fontsize=11.2,
        weight="bold",
    )
    path = out_dir / "checkpoint1_screen_frame_fem_axis_distributions.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    values = load_values()
    selected = select_random_examples(values)
    examples_path, trace_audit = render_examples(selected, args.out_dir)
    distributions_path, distribution_values = render_axis_distributions(values, args.out_dir)

    selected.to_csv(args.out_dir / "checkpoint1_random_example_selection.csv", index=False)
    trace_audit.to_csv(args.out_dir / "checkpoint1_trace_recomputation_audit.csv", index=False)
    distribution_values.to_csv(args.out_dir / "checkpoint1_screen_axis_distribution_values.csv", index=False)
    config = {
        "stage": "map-first horizontal-versus-vertical checkpoint 1",
        "input": str(INPUT.relative_to(ROOT)),
        "coherence_threshold": COHERENCE_THRESHOLD,
        "horizontal_center_deg": 0.0,
        "vertical_center_deg": 90.0,
        "bin_half_width_deg": BIN_HALF_WIDTH_DEG,
        "primary_outcome": "contour-parallel RMS minus contour-orthogonal RMS, arcmin",
        "planned_hierarchy": "window -> trial median -> session median -> equal animal mean",
        "selection_seed": SEED,
        "examples_per_subject_condition": N_EXAMPLES_PER_CELL,
        "selection_is_outcome_blind": True,
    }
    (args.out_dir / "frozen_test_config.json").write_text(json.dumps(config, indent=2) + "\n")

    report = [
        "# Horizontal-versus-vertical Figure 4F: raw checkpoint",
        "",
        "This checkpoint freezes the test and renders raw input examples before new inference.",
        "Examples are selected by a fixed random seed after image-coherence, orientation-bin, and",
        "patch-validity gates only. FEM anisotropy, FEM axis, and Figure 4F outcome do not enter selection.",
        "",
        f"- retained horizontal/vertical windows: {len(values)}",
        f"- random examples: {len(selected)} ({N_EXAMPLES_PER_CELL} per animal × condition)",
        f"- maximum stored-versus-recomputed FEM-axis error: {trace_audit.axis_recompute_error_deg.max():.6g}°",
        "",
        "## Screen-frame distribution values",
        "",
    ]
    for row in distribution_values.itertuples(index=False):
        report.append(
            f"- {row.subject}, {row.condition}: FEM mean axis {row.mean_fem_axis_deg:+.1f}° "
            f"(R={row.fem_axis_resultant:.3f}); contour mean axis {row.mean_contour_axis_deg:+.1f}°; "
            f"n={row.n_windows_anisotropy_ge_0p2} readable-axis windows."
        )
    report.extend(
        [
            "",
            "No population claim is made at this checkpoint. The next gate is visual confirmation that",
            "vertical-labelled patches are genuinely vertical and that their FEM clouds remain primarily",
            "screen-horizontal rather than rotating with the contour.",
            "",
            f"Examples: `{examples_path.name}`",
            f"Axis distributions: `{distributions_path.name}`",
        ]
    )
    (args.out_dir / "summary_report.md").write_text("\n".join(report) + "\n")
    print(examples_path)
    print(distributions_path)
    print(args.out_dir / "summary_report.md")


if __name__ == "__main__":
    main()
