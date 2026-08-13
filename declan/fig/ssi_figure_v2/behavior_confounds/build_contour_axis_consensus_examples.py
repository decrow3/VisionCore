#!/usr/bin/env python3
"""Select and render outcome-blind BackImage patches with consensus contour axes.

This is a targeted map-first visualization checkpoint, not population inference.
Selection uses image-derived validity measures only; FEM orientation, covariance,
and Figure 4F outcomes never enter ranking or inclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy import ndimage
from skimage.feature import canny

from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint1 import (
    ROOT,
    axial_distance_deg,
    axial_signed_deg,
)
from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint2_axis_validation import (
    add_independent_axis,
)
from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint3_edge_detectors import (
    COLORS,
    axial_mean_deg,
    draw_gaze_axis,
    line_segment_axis,
    normalize_image,
    tensor_edge_axis_deg,
)


INPUT = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1/contour_motion_component_windows.csv"
)
TRIAL_AUDIT = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_trial_scale_audit/backimage_trial_scale_audit.csv"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "contour_axis_consensus_examples_v1"
)
SUBJECTS = ("Allen", "Logan")
AXIS_CENTERS = (0.0, 45.0, 90.0, 135.0)
AXIS_LABELS = {0.0: "horizontal", 45.0: "45°", 90.0: "vertical", 135.0: "135°"}
SCALE_FRACTIONS = (0.5, 0.75, 1.0)
N_PREFILTER_PER_CELL = 20
N_DISPLAY_PER_CELL = 2
BACKGROUND_DIR = ROOT.parent / "DataYatesV1/DataYatesV1/exp/SupportData/Backgrounds"


def nearest_axis_bin(angle_deg: float) -> tuple[float, float]:
    distances = np.asarray([axial_distance_deg(angle_deg, center) for center in AXIS_CENTERS], dtype=float)
    index = int(np.argmin(distances))
    return AXIS_CENTERS[index], float(distances[index])


def load_candidates(
    prefilter_per_cell: int = N_PREFILTER_PER_CELL,
    axis_centers: tuple[float, ...] = AXIS_CENTERS,
    unique_trials: bool = True,
    spectrum_anisotropy_min: float = 0.20,
    spectrum_disagreement_max_deg: float = 10.0,
) -> pd.DataFrame:
    values = pd.read_csv(INPUT)
    audit = pd.read_csv(TRIAL_AUDIT, usecols=[
        "session", "trial_idx", "image_file", "dest_x0_px", "dest_y0_px",
        "dest_x1_px", "dest_y1_px", "screen_width_px", "screen_height_px",
    ])
    values = values.merge(audit, on=["session", "trial_idx"], how="left", validate="many_to_one")
    values = values[
        values.image_feature_ok.fillna(False).astype(bool)
        & values.image_orientation_coherence.ge(0.30)
        & values.image_patch_fraction_inside_image.ge(0.999)
        & values.image_patch_fraction_background.le(0.05)
    ].copy()
    values = add_independent_axis(values)
    bins = [nearest_axis_bin(float(axis)) for axis in values.image_edge_axis_deg]
    values["axis_bin_center_deg"] = [item[0] for item in bins]
    values["axis_bin_distance_deg"] = [item[1] for item in bins]
    values = values[
        values.image_spectrum_anisotropy.ge(spectrum_anisotropy_min)
        & values.sobel_spectrum_axis_disagreement_deg.le(spectrum_disagreement_max_deg)
        & values.axis_bin_distance_deg.le(22.5)
    ].copy()
    values["prefilter_score"] = (
        values.image_orientation_coherence
        + values.image_spectrum_anisotropy
        - values.sobel_spectrum_axis_disagreement_deg / 45.0
    )
    rows = []
    for subject in SUBJECTS:
        for center in axis_centers:
            block = values[(values.subject == subject) & (values.axis_bin_center_deg == center)]
            block = block.sort_values("prefilter_score", ascending=False)
            if unique_trials:
                block = block.drop_duplicates(["session", "trial_idx"], keep="first")
            rows.append(block if prefilter_per_cell <= 0 else block.head(prefilter_per_cell))
    candidates = pd.concat(rows, ignore_index=True)
    # The shared reconstruction helper uses this field only as an audit label.
    # It has no role in stimulus reconstruction or candidate selection.
    candidates["selection_rank"] = np.arange(1, len(candidates) + 1)
    return candidates


def analyze_direct(source: pd.Series) -> tuple[dict[str, object], dict[str, np.ndarray | float]]:
    """Reconstruct exactly as DataYatesV1 BackImageTrial.get_image, then crop."""
    image = PILImage.open(BACKGROUND_DIR / str(source.image_file))
    width = int(source.dest_x1_px - source.dest_x0_px)
    height = int(source.dest_y1_px - source.dest_y0_px)
    displayed = np.asarray(image.resize((width, height), resample=2))
    if displayed.ndim == 3:
        displayed = np.mean(displayed, axis=2).astype(np.uint8)
    cx_screen = int(round(float(source.image_patch_center_x_px)))
    cy_screen = int(round(float(source.image_patch_center_y_px)))
    cx = cx_screen - int(source.dest_x0_px)
    cy = cy_screen - int(source.dest_y0_px)
    radius = int(source.image_patch_radius_px)
    patch = np.asarray(displayed[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1], dtype=np.float64)
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        raise ValueError(f"incomplete displayed-image patch {patch.shape}")
    context_radius = 2 * radius
    x0, x1 = max(0, cx - context_radius), min(displayed.shape[1], cx + context_radius + 1)
    y0, y1 = max(0, cy - context_radius), min(displayed.shape[0], cy + context_radius + 1)
    context = np.asarray(displayed[y0:y1, x0:x1], dtype=np.float64)
    context_center = (float(cx - x0), float(cy - y0))

    patch01 = normalize_image(patch)
    yy, xx = np.indices(patch.shape)
    center_y, center_x = (np.asarray(patch.shape, dtype=float) - 1.0) / 2.0
    circle_mask = np.hypot(xx - center_x, yy - center_y) <= float(radius)
    square_mask = np.ones_like(circle_mask, dtype=bool)
    sobel_x = ndimage.sobel(patch, axis=1, mode="nearest")
    sobel_y = ndimage.sobel(patch, axis=0, mode="nearest")
    sobel_square_axis, sobel_square_coherence = tensor_edge_axis_deg(sobel_x, sobel_y, square_mask)
    sobel_circle_axis, sobel_circle_coherence = tensor_edge_axis_deg(sobel_x, sobel_y, circle_mask)
    scharr_x = cv2.Scharr(patch01, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)
    scharr_y = cv2.Scharr(patch01, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REPLICATE)
    scharr_circle_axis, scharr_circle_coherence = tensor_edge_axis_deg(scharr_x, scharr_y, circle_mask)
    canny_edges = canny(patch01, sigma=1.4, mode="nearest") & circle_mask
    canny_circle_axis, canny_circle_coherence = tensor_edge_axis_deg(scharr_x, scharr_y, canny_edges)
    lsd_axis, lsd_resultant, segments, segment_weights = line_segment_axis(patch01, circle_mask)
    base = {
        "subject": str(source.subject), "selection_rank": int(source.selection_rank),
        "session": str(source.session), "trial_idx": int(source.trial_idx), "patch_radius_px": radius,
        "stored_sobel_square_axis_deg": float(axial_signed_deg(source.image_edge_axis_deg)),
        "sobel_square_axis_deg": sobel_square_axis, "sobel_square_coherence": sobel_square_coherence,
        "sobel_square_recompute_error_deg": float(axial_distance_deg(sobel_square_axis, source.image_edge_axis_deg)),
        "sobel_circle_axis_deg": sobel_circle_axis, "sobel_circle_coherence": sobel_circle_coherence,
        "sobel_square_vs_circle_disagreement_deg": float(axial_distance_deg(sobel_square_axis, sobel_circle_axis)),
        "scharr_circle_axis_deg": scharr_circle_axis, "scharr_circle_coherence": scharr_circle_coherence,
        "canny_circle_axis_deg": canny_circle_axis, "canny_circle_coherence": canny_circle_coherence,
        "canny_edge_fraction_in_circle": float(np.sum(canny_edges) / np.sum(circle_mask)),
        "lsd_circle_axis_deg": lsd_axis, "lsd_circle_resultant": lsd_resultant,
        "lsd_n_segments": int(len(segments)),
        "fourier_square_axis_deg": float(axial_signed_deg(source.spectrum_contour_axis_deg)),
        "fourier_square_anisotropy": float(source.image_spectrum_anisotropy),
    }
    arrays = {
        "context": context, "patch": patch, "patch01": patch01, "circle_mask": circle_mask,
        "scharr_magnitude": np.hypot(scharr_x, scharr_y), "canny_edges": canny_edges,
        "segments": segments, "segment_weights": segment_weights,
        "context_center_x": context_center[0], "context_center_y": context_center[1],
    }
    return base, arrays


def _line_validity(
    arrays: dict[str, np.ndarray | float], reference_axis: float, radius: float,
) -> tuple[float, float]:
    segments = np.asarray(arrays["segments"], dtype=float)
    weights = np.asarray(arrays["segment_weights"], dtype=float)
    if not len(segments) or np.sum(weights) <= 0:
        return float("nan"), float("nan")
    angles = -np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]))
    aligned = np.asarray(axial_distance_deg(angles, reference_axis)) <= 10.0
    aligned_fraction = float(np.sum(weights[aligned]) / np.sum(weights))
    if not np.any(aligned):
        return aligned_fraction, float("nan")
    center = np.asarray([radius, radius], dtype=float)
    distances = []
    for x0, y0, x1, y1 in segments[aligned]:
        p0 = np.asarray([x0, y0], dtype=float)
        direction = np.asarray([x1 - x0, y1 - y0], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            continue
        distances.append(abs(np.cross(direction, center - p0)) / norm)
    return aligned_fraction, float(np.min(distances) / radius) if distances else float("nan")


def add_spatial_validations(
    base: dict[str, object], arrays: dict[str, np.ndarray | float], source: pd.Series,
) -> dict[str, object]:
    out = dict(base)
    patch01 = np.asarray(arrays["patch01"], dtype=float)
    radius = float(base["patch_radius_px"])
    yy, xx = np.indices(patch01.shape)
    cy, cx = (np.asarray(patch01.shape, dtype=float) - 1.0) / 2.0
    scharr_x = cv2.Scharr(patch01, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)
    scharr_y = cv2.Scharr(patch01, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REPLICATE)
    reference = float(base["sobel_square_axis_deg"])

    scale_axes = []
    scale_coherences = []
    for fraction in SCALE_FRACTIONS:
        mask = np.hypot(xx - cx, yy - cy) <= radius * fraction
        axis, coherence = tensor_edge_axis_deg(scharr_x, scharr_y, mask)
        scale_axes.append(axis)
        scale_coherences.append(coherence)
        tag = str(fraction).replace(".", "p")
        out[f"scharr_axis_radius_{tag}_deg"] = axis
        out[f"scharr_coherence_radius_{tag}"] = coherence
    out["multiscale_max_disagreement_deg"] = float(np.nanmax(axial_distance_deg(np.asarray(scale_axes), reference)))
    out["multiscale_min_coherence"] = float(np.nanmin(scale_coherences))

    quadrant_axes = []
    quadrant_weights = []
    circle = np.hypot(xx - cx, yy - cy) <= radius
    quadrant_masks = [
        circle & (xx <= cx) & (yy <= cy),
        circle & (xx > cx) & (yy <= cy),
        circle & (xx <= cx) & (yy > cy),
        circle & (xx > cx) & (yy > cy),
    ]
    for index, mask in enumerate(quadrant_masks):
        axis, coherence = tensor_edge_axis_deg(scharr_x, scharr_y, mask)
        out[f"quadrant_{index + 1}_axis_deg"] = axis
        out[f"quadrant_{index + 1}_coherence"] = coherence
        if np.isfinite(axis) and np.isfinite(coherence) and coherence >= 0.15:
            quadrant_axes.append(axis)
            quadrant_weights.append(coherence)
    if quadrant_axes:
        quadrant_mean, quadrant_resultant = axial_mean_deg(np.asarray(quadrant_axes), np.asarray(quadrant_weights))
        quadrant_max = float(np.max(axial_distance_deg(np.asarray(quadrant_axes), reference)))
    else:
        quadrant_mean = quadrant_resultant = quadrant_max = float("nan")
    out["n_readable_quadrants"] = len(quadrant_axes)
    out["quadrant_consensus_axis_deg"] = quadrant_mean
    out["quadrant_axis_resultant"] = quadrant_resultant
    out["quadrant_max_disagreement_deg"] = quadrant_max

    aligned_fraction, center_distance = _line_validity(arrays, reference, radius)
    out["lsd_aligned_weight_fraction"] = aligned_fraction
    out["nearest_aligned_line_distance_radius"] = center_distance

    validator_fields = [
        "sobel_circle_axis_deg", "scharr_circle_axis_deg", "canny_circle_axis_deg",
        "lsd_circle_axis_deg", "fourier_square_axis_deg",
    ]
    validator_angles = np.asarray([float(out[field]) for field in validator_fields] + scale_axes, dtype=float)
    out["all_validator_max_disagreement_deg"] = float(np.nanmax(axial_distance_deg(validator_angles, reference)))

    out.update({
        "image_file": str(source.image_file),
        "phase": str(source.phase),
        "global_start": int(source.global_start),
        "image_edge_axis_deg": float(source.image_edge_axis_deg),
        "image_orientation_coherence": float(source.image_orientation_coherence),
        "image_spectrum_anisotropy": float(source.image_spectrum_anisotropy),
        "axis_bin_center_deg": float(source.axis_bin_center_deg),
        "axis_bin_label": AXIS_LABELS[float(source.axis_bin_center_deg)],
    })

    out["passes_strict_consensus"] = bool(
        float(out["sobel_square_coherence"]) >= 0.35
        and float(out["sobel_circle_coherence"]) >= 0.30
        and float(out["scharr_circle_coherence"]) >= 0.30
        and float(out["canny_circle_coherence"]) >= 0.30
        and float(out["fourier_square_anisotropy"]) >= 0.20
        and float(out["all_validator_max_disagreement_deg"]) <= 10.0
        and float(out["lsd_circle_resultant"]) >= 0.50
        and int(out["lsd_n_segments"]) >= 3
        and float(out["lsd_aligned_weight_fraction"]) >= 0.50
        and float(out["nearest_aligned_line_distance_radius"]) <= 0.30
        and float(out["multiscale_min_coherence"]) >= 0.20
        and int(out["n_readable_quadrants"]) >= 3
        and float(out["quadrant_axis_resultant"]) >= 0.80
        and float(out["quadrant_max_disagreement_deg"]) <= 15.0
    )
    out["consensus_rank_score"] = float(
        min(
            float(out["sobel_square_coherence"]),
            float(out["sobel_circle_coherence"]),
            float(out["scharr_circle_coherence"]),
            float(out["canny_circle_coherence"]),
            float(out["fourier_square_anisotropy"]),
            float(out["quadrant_axis_resultant"]),
            float(out["lsd_circle_resultant"]),
            float(out["lsd_aligned_weight_fraction"]),
        )
        - float(out["all_validator_max_disagreement_deg"]) / 90.0
        - float(out["nearest_aligned_line_distance_radius"]) / 10.0
    )
    return out


def analyze_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray | float]]]:
    rows = []
    arrays_by_index: dict[int, dict[str, np.ndarray | float]] = {}
    for index, source in candidates.iterrows():
        if index % 20 == 0:
            print(f"analyzing candidate {index + 1}/{len(candidates)}", flush=True)
        try:
            base, arrays = analyze_direct(source)
            row = add_spatial_validations(base, arrays, source)
        except (ValueError, cv2.error) as exc:
            rows.append({
                "candidate_index": int(index),
                "subject": str(source.subject),
                "session": str(source.session),
                "trial_idx": int(source.trial_idx),
                "axis_bin_center_deg": float(source.axis_bin_center_deg),
                "axis_bin_label": AXIS_LABELS[float(source.axis_bin_center_deg)],
                "passes_strict_consensus": False,
                "analysis_error": f"{type(exc).__name__}: {exc}",
            })
            continue
        row["candidate_index"] = int(index)
        row["analysis_error"] = ""
        rows.append(row)
        arrays_by_index[int(index)] = arrays
    return pd.DataFrame(rows), arrays_by_index


def select_display(
    audit: pd.DataFrame,
    axis_centers: tuple[float, ...] = AXIS_CENTERS,
    n_display_per_cell: int = N_DISPLAY_PER_CELL,
    spread_scores: bool = False,
) -> pd.DataFrame:
    selected_rows = []
    for subject in SUBJECTS:
        used_images: set[str] = set()
        for center in axis_centers:
            block = audit[
                (audit.subject == subject)
                & (audit.axis_bin_center_deg == center)
                & audit.passes_strict_consensus
            ].sort_values("consensus_rank_score", ascending=False)
            if spread_scores and len(block) > n_display_per_cell:
                ranks = np.unique(np.round(np.linspace(0, len(block) - 1, n_display_per_cell)).astype(int))
                diverse = block.iloc[ranks]
            else:
                diverse = block[~block.image_file.isin(used_images)].head(n_display_per_cell)
                if len(diverse) < n_display_per_cell:
                    diverse = pd.concat([diverse, block[~block.index.isin(diverse.index)]], ignore_index=False).head(n_display_per_cell)
            diverse = diverse.copy()
            diverse["display_rank_within_cell"] = np.arange(1, len(diverse) + 1)
            diverse["selection_role"] = "strict_consensus_" + AXIS_LABELS[center].replace("°", "deg")
            selected_rows.append(diverse)
            used_images.update(diverse.image_file.astype(str))
    if not selected_rows:
        return pd.DataFrame()
    return pd.concat(selected_rows, ignore_index=True)


def _plot_segments(ax: plt.Axes, arrays: dict[str, np.ndarray | float], values: pd.Series) -> None:
    patch = np.asarray(arrays["patch"])
    radius = float(values.patch_radius_px)
    center = radius
    ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
    circle = Circle((center, center), radius, fill=False, edgecolor="white", lw=0.7)
    ax.add_patch(circle)
    segments = np.asarray(arrays["segments"])
    weights = np.asarray(arrays["segment_weights"])
    if len(weights):
        for segment, weight in zip(segments, weights, strict=True):
            angle = -np.degrees(np.arctan2(segment[3] - segment[1], segment[2] - segment[0]))
            agrees = float(axial_distance_deg(angle, values.sobel_square_axis_deg)) <= 10.0
            line = ax.plot([segment[0], segment[2]], [segment[1], segment[3]],
                           color="#56B4E9" if agrees else "#E69F00", lw=0.6 + 1.4 * weight / np.max(weights), alpha=0.8)[0]
            line.set_clip_path(circle)
    draw_gaze_axis(ax, float(values.sobel_square_axis_deg), center, center, 0.82 * radius,
                   color="#20A464", lw=2.1)
    ax.scatter([center], [center], marker="+", s=28, c="#E23D3D", linewidths=1.0)
    ax.axis("off")


def render_subject(
    subject: str,
    selected: pd.DataFrame,
    arrays_by_index: dict[int, dict[str, np.ndarray | float]],
    out_dir: Path,
    title: str = "outcome-blind strict-consensus local contour examples",
) -> Path:
    block = selected[selected.subject == subject].sort_values(["axis_bin_center_deg", "display_rank_within_cell"])
    fig, axes = plt.subplots(len(block), 4, figsize=(12.8, 2.55 * len(block)), constrained_layout=True)
    for row_index, row in enumerate(block.itertuples(index=False)):
        arrays = arrays_by_index[int(row.candidate_index)]
        context = np.asarray(arrays["context"])
        patch = np.asarray(arrays["patch"])
        radius = float(row.patch_radius_px)
        ccx, ccy = float(arrays["context_center_x"]), float(arrays["context_center_y"])
        center = radius

        ax = axes[row_index, 0]
        ax.imshow(context, cmap="gray", origin="upper", interpolation="nearest")
        ax.add_patch(Rectangle((ccx - radius, ccy - radius), 2 * radius, 2 * radius,
                               fill=False, edgecolor="#FFB000", lw=1.5))
        ax.add_patch(Circle((ccx, ccy), radius, fill=False, edgecolor="#E23D3D", lw=1.2, ls="--"))
        ax.scatter([ccx], [ccy], marker="+", s=22, c="#E23D3D")
        ax.axis("off")
        ax.set_ylabel(f"{row.axis_bin_label} #{int(row.display_rank_within_cell)}\n{Path(row.image_file).stem}\n"
                      f"{str(row.session).replace(subject + '_', '')} t{int(row.trial_idx)}",
                      fontsize=8, weight="bold")

        ax = axes[row_index, 1]
        ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
        specs = [
            ("sobel_square_axis_deg", COLORS["sobel_square"], "-"),
            ("scharr_circle_axis_deg", COLORS["scharr_circle"], "-"),
            ("canny_circle_axis_deg", COLORS["canny_circle"], "-"),
            ("lsd_circle_axis_deg", COLORS["lsd_circle"], "-"),
            ("fourier_square_axis_deg", COLORS["fourier_square"], "--"),
        ]
        for field, color, style in specs:
            draw_gaze_axis(ax, float(getattr(row, field)), center, center, 0.82 * radius, color=color, lw=1.45, ls=style)
        ax.scatter([center], [center], marker="+", s=28, c="#E23D3D")
        ax.axis("off")
        ax.set_title(f"5 estimators: max Δ {row.all_validator_max_disagreement_deg:.1f}°", fontsize=8)

        ax = axes[row_index, 2]
        _plot_segments(ax, arrays, pd.Series(row._asdict()))
        ax.set_title(f"line support {row.lsd_aligned_weight_fraction:.2f}; center {row.nearest_aligned_line_distance_radius:.2f}r", fontsize=8)

        ax = axes[row_index, 3]
        labels = ["Sobel", "Scharr", "Canny", "LSD", "Fourier", "r=.5", "r=.75", "r=1"]
        angles = [row.sobel_square_axis_deg, row.scharr_circle_axis_deg, row.canny_circle_axis_deg,
                  row.lsd_circle_axis_deg, row.fourier_square_axis_deg,
                  row.scharr_axis_radius_0p5_deg, row.scharr_axis_radius_0p75_deg, row.scharr_axis_radius_1p0_deg]
        reference = float(row.sobel_square_axis_deg)
        signed_delta = np.asarray([axial_signed_deg(float(angle) - reference) for angle in angles])
        colors = [COLORS["sobel_square"], COLORS["scharr_circle"], COLORS["canny_circle"],
                  COLORS["lsd_circle"], COLORS["fourier_square"], "#777777", "#999999", "#BBBBBB"]
        ax.barh(np.arange(len(labels)), signed_delta, color=colors)
        ax.axvline(0, color="0.35", lw=1)
        ax.set_xlim(-12, 12)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("axis − Sobel (deg)", fontsize=7)
        ax.set_title(f"quadrants R={row.quadrant_axis_resultant:.2f}; {int(row.n_readable_quadrants)}/4 readable", fontsize=8)
        ax.grid(axis="x", alpha=0.2)

    for ax, column_title in zip(axes[0], ["2° context", "axis overlay", "line evidence", "agreement audit"], strict=True):
        ax.text(0.5, 1.20, column_title, transform=ax.transAxes, ha="center", fontsize=9.2, weight="bold")
    fig.suptitle(f"{subject}: {title}\nred + = fixation; no FEM variable entered selection", fontsize=12, weight="bold")
    path = out_dir / f"strict_consensus_examples_{subject.lower()}.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prefilter-per-cell", type=int, default=N_PREFILTER_PER_CELL,
                        help="Unique trials retained per subject/axis; <=0 keeps all")
    parser.add_argument("--axis-centers", default="0,45,90,135")
    parser.add_argument("--display-per-cell", type=int, default=N_DISPLAY_PER_CELL)
    parser.add_argument("--spread-display-scores", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    axis_centers = tuple(float(value) for value in args.axis_centers.split(","))

    candidates = load_candidates(args.prefilter_per_cell, axis_centers)
    audit, arrays_by_index = analyze_candidates(candidates)
    selected = select_display(
        audit, axis_centers, args.display_per_cell, args.spread_display_scores,
    )
    audit.to_csv(args.out_dir / "candidate_validation_audit.csv", index=False)
    selected.to_csv(args.out_dir / "selected_strict_consensus_examples.csv", index=False)
    paths = [render_subject(subject, selected, arrays_by_index, args.out_dir) for subject in SUBJECTS]
    support = audit.groupby(["subject", "axis_bin_label"], sort=False).agg(
        candidates=("candidate_index", "size"), strict_consensus=("passes_strict_consensus", "sum")
    ).reset_index()
    support.to_csv(args.out_dir / "strict_consensus_support.csv", index=False)
    metadata = {
        "stage": "targeted map-first validation render",
        "selection_is_outcome_blind": True,
        "reconstruction": (
            "DataYatesV1 BackImageTrial.get_image equivalent: source image resized to the "
            "trial audit destination rectangle with PIL resample=2, then cropped at the "
            "stored displayed-canvas fixation coordinates"
        ),
        "input": str(INPUT.relative_to(ROOT)),
        "axis_centers": axis_centers,
        "prefilter_per_subject_axis_cell": args.prefilter_per_cell,
        "display_per_subject_axis_cell": args.display_per_cell,
        "display_selection_spans_consensus_score": args.spread_display_scores,
        "n_candidates_analyzed": int(len(audit)),
        "n_strict_consensus": int(audit.passes_strict_consensus.sum()),
        "n_displayed": int(len(selected)),
        "strict_thresholds": {
            "sobel_square_coherence_min": 0.35,
            "sobel_circle_coherence_min": 0.30,
            "scharr_circle_coherence_min": 0.30,
            "canny_circle_coherence_min": 0.30,
            "fourier_anisotropy_min": 0.20,
            "all_validator_max_disagreement_deg": 10.0,
            "lsd_resultant_min": 0.50,
            "lsd_segments_min": 3,
            "lsd_aligned_weight_fraction_min": 0.50,
            "nearest_aligned_line_distance_radius_max": 0.30,
            "multiscale_min_coherence": 0.20,
            "readable_quadrants_min": 3,
            "quadrant_resultant_min": 0.80,
            "quadrant_max_disagreement_deg": 15.0,
        },
        "outputs": [str(path.relative_to(ROOT)) for path in paths],
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(support.to_string(index=False))
    print("strict", int(audit.passes_strict_consensus.sum()), "displayed", len(selected))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
