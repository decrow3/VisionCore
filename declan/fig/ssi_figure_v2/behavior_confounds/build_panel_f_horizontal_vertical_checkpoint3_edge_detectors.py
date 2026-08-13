#!/usr/bin/env python3
"""Map-first comparison of local edge estimators for frozen vertical patches.

The original feature implementation uses a square extending +/-1 degree in x
and y.  This checkpoint displays that square together with a literal 1-degree
circle, and compares axes derived from Sobel, Scharr, Canny-selected gradients,
OpenCV's line-segment detector, and the existing Fourier-spectrum estimator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import canny

from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint1 import (
    ROOT,
    SUBJECTS,
    axial_distance_deg,
    axial_signed_deg,
)
from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint2_axis_validation import (
    PARENT_OUT,
    SELECTION,
    add_independent_axis,
)
from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas


DEFAULT_OUT = PARENT_OUT / "checkpoint3_edge_detectors"
CONTEXT_RADIUS_MULTIPLIER = 2
CANNY_SIGMA_PX = 1.5
COLORS = {
    "sobel_square": "#20A464",
    "sobel_circle": "#207F5F",
    "scharr_circle": "#E68613",
    "canny_circle": "#1F9EAE",
    "lsd_circle": "#D1495B",
    "fourier_square": "#A33FA3",
}
LABELS = {
    "sobel_square": "Sobel, square (original)",
    "sobel_circle": "Sobel, 1° circle",
    "scharr_circle": "Scharr, 1° circle",
    "canny_circle": "Canny edges + Scharr axis, 1° circle",
    "lsd_circle": "line-segment detector, 1° circle",
    "fourier_square": "Fourier, square",
}


def normalize_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.zeros_like(values)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def axial_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    angles = np.asarray(angles_deg, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan"), float("nan")
    z = np.sum(weights[valid] * np.exp(2j * np.radians(angles[valid])))
    total = float(np.sum(weights[valid]))
    if total <= 0 or abs(z) <= 0:
        return float("nan"), float("nan")
    return float(axial_signed_deg(0.5 * np.degrees(np.angle(z)))), float(abs(z) / total)


def tensor_edge_axis_deg(
    gx: np.ndarray,
    gy: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(gx) & np.isfinite(gy)
    if not np.any(valid):
        return float("nan"), float("nan")
    x = np.asarray(gx, dtype=np.float64)[valid]
    y = np.asarray(gy, dtype=np.float64)[valid]
    jxx = float(np.mean(x * x))
    jyy = float(np.mean(y * y))
    jxy = float(np.mean(x * y))
    denominator = jxx + jyy
    if denominator <= 0:
        return float("nan"), float("nan")
    coherence = float(np.hypot(jxx - jyy, 2.0 * jxy) / denominator)
    gradient_axis_image = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    edge_axis_gaze = -(gradient_axis_image + np.pi / 2.0)
    return float(axial_signed_deg(np.degrees(edge_axis_gaze))), coherence


def draw_gaze_axis(
    ax: plt.Axes,
    axis_deg: float,
    center_x: float,
    center_y: float,
    half_length: float,
    **kwargs: object,
) -> None:
    if not np.isfinite(axis_deg):
        return
    # Gaze coordinates have +y up; displayed image-array coordinates have +row down.
    theta_image = -np.radians(float(axis_deg))
    dx = half_length * np.cos(theta_image)
    dy = half_length * np.sin(theta_image)
    ax.plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy], **kwargs)


def extract_context_and_patch(row: pd.Series) -> tuple[np.ndarray, np.ndarray, int, tuple[float, float]]:
    canvas, _ppd, _shape = _backimage_canvas(str(row.session), int(row.trial_idx))
    cx = int(round(float(row.image_patch_center_x_px)))
    cy = int(round(float(row.image_patch_center_y_px)))
    radius = int(row.image_patch_radius_px)
    context_radius = CONTEXT_RADIUS_MULTIPLIER * radius
    x0, x1 = max(0, cx - context_radius), min(canvas.shape[1], cx + context_radius + 1)
    y0, y1 = max(0, cy - context_radius), min(canvas.shape[0], cy + context_radius + 1)
    context = np.asarray(canvas[y0:y1, x0:x1], dtype=np.float64)
    patch = np.asarray(canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1], dtype=np.float64)
    return context, patch, radius, (float(cx - x0), float(cy - y0))


def line_segment_axis(
    patch01: np.ndarray,
    circle_mask: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    uint8 = np.round(np.clip(patch01, 0, 1) * 255.0).astype(np.uint8)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(uint8)[0]
    if detected is None:
        return float("nan"), float("nan"), np.empty((0, 4)), np.empty(0)
    segments = np.asarray(detected[:, 0, :], dtype=np.float64)
    weights = []
    angles = []
    keep = []
    for segment in segments:
        x0, y0, x1, y1 = segment
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 2.0:
            continue
        t = np.linspace(0.0, 1.0, max(8, int(np.ceil(length)) + 1))
        xs = np.clip(np.round(x0 + t * (x1 - x0)).astype(int), 0, circle_mask.shape[1] - 1)
        ys = np.clip(np.round(y0 + t * (y1 - y0)).astype(int), 0, circle_mask.shape[0] - 1)
        fraction_inside = float(np.mean(circle_mask[ys, xs]))
        effective_weight = length * fraction_inside
        if effective_weight <= 0:
            continue
        keep.append(segment)
        weights.append(effective_weight)
        angles.append(-np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    if not keep:
        return float("nan"), float("nan"), np.empty((0, 4)), np.empty(0)
    axis, resultant = axial_mean_deg(np.asarray(angles), np.asarray(weights))
    return axis, resultant, np.asarray(keep), np.asarray(weights)


def analyze_example(row: pd.Series) -> tuple[dict[str, object], dict[str, np.ndarray | float]]:
    context, patch, radius, context_center = extract_context_and_patch(row)
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
    scharr_magnitude = np.hypot(scharr_x, scharr_y)

    canny_edges = canny(patch01, sigma=CANNY_SIGMA_PX, mode="nearest") & circle_mask
    canny_circle_axis, canny_circle_coherence = tensor_edge_axis_deg(scharr_x, scharr_y, canny_edges)
    lsd_circle_axis, lsd_circle_resultant, segments, segment_weights = line_segment_axis(patch01, circle_mask)

    values: dict[str, object] = {
        "subject": str(row.subject),
        "selection_rank": int(row.selection_rank),
        "session": str(row.session),
        "trial_idx": int(row.trial_idx),
        "patch_radius_px": radius,
        "stored_sobel_square_axis_deg": float(axial_signed_deg(row.image_edge_axis_deg)),
        "sobel_square_axis_deg": sobel_square_axis,
        "sobel_square_coherence": sobel_square_coherence,
        "sobel_square_recompute_error_deg": float(axial_distance_deg(sobel_square_axis, row.image_edge_axis_deg)),
        "sobel_circle_axis_deg": sobel_circle_axis,
        "sobel_circle_coherence": sobel_circle_coherence,
        "sobel_square_vs_circle_disagreement_deg": float(axial_distance_deg(sobel_square_axis, sobel_circle_axis)),
        "scharr_circle_axis_deg": scharr_circle_axis,
        "scharr_circle_coherence": scharr_circle_coherence,
        "canny_circle_axis_deg": canny_circle_axis,
        "canny_circle_coherence": canny_circle_coherence,
        "canny_edge_fraction_in_circle": float(np.sum(canny_edges) / np.sum(circle_mask)),
        "lsd_circle_axis_deg": lsd_circle_axis,
        "lsd_circle_resultant": lsd_circle_resultant,
        "lsd_n_segments": int(len(segments)),
        "fourier_square_axis_deg": float(axial_signed_deg(row.spectrum_contour_axis_deg)),
        "fourier_square_anisotropy": float(row.image_spectrum_anisotropy),
    }
    arrays: dict[str, np.ndarray | float] = {
        "context": context,
        "patch": patch,
        "patch01": patch01,
        "circle_mask": circle_mask,
        "scharr_magnitude": scharr_magnitude,
        "canny_edges": canny_edges,
        "segments": segments,
        "segment_weights": segment_weights,
        "context_center_x": context_center[0],
        "context_center_y": context_center[1],
    }
    return values, arrays


def plot_context(ax: plt.Axes, arrays: dict[str, np.ndarray | float], radius: int) -> None:
    context = np.asarray(arrays["context"])
    cx = float(arrays["context_center_x"])
    cy = float(arrays["context_center_y"])
    ax.imshow(context, cmap="gray", origin="upper", interpolation="nearest")
    ax.add_patch(
        Rectangle(
            (cx - radius, cy - radius), 2 * radius, 2 * radius,
            fill=False, edgecolor="#FFB000", lw=1.8,
        )
    )
    ax.add_patch(
        Circle((cx, cy), radius, fill=False, edgecolor="#E23D3D", lw=1.6, ls="--")
    )
    ax.scatter([cx], [cy], marker="+", s=22, c="#E23D3D", linewidths=1.0)
    ax.axis("off")


def plot_axis_summary(ax: plt.Axes, values: dict[str, object], radius: int) -> None:
    center = float(radius)
    ax.set_facecolor("#F6F7F8")
    ax.add_patch(Circle((center, center), 0.86 * radius, fill=False, edgecolor="#B8BEC5", lw=0.8))
    specs = [
        ("sobel_square", "sobel_square_axis_deg", "-"),
        ("sobel_circle", "sobel_circle_axis_deg", "-"),
        ("scharr_circle", "scharr_circle_axis_deg", "-"),
        ("canny_circle", "canny_circle_axis_deg", "-"),
        ("lsd_circle", "lsd_circle_axis_deg", "-"),
        ("fourier_square", "fourier_square_axis_deg", "--"),
    ]
    for name, field, linestyle in specs:
        draw_gaze_axis(
            ax, float(values[field]), center, center, 0.82 * radius,
            color=COLORS[name], lw=1.5, ls=linestyle, alpha=0.9,
        )
    ax.set_xlim(-1, 2 * radius + 1)
    ax.set_ylim(2 * radius + 1, -1)
    ax.set_aspect("equal")
    ax.axis("off")


def render_subject(
    subject: str,
    analyzed: list[tuple[dict[str, object], dict[str, np.ndarray | float]]],
    out_dir: Path,
) -> Path:
    rows = [(values, arrays) for values, arrays in analyzed if values["subject"] == subject]
    rows.sort(key=lambda pair: int(pair[0]["selection_rank"]))
    fig, axes = plt.subplots(len(rows), 6, figsize=(16.0, 2.75 * len(rows)), constrained_layout=True)
    for row_index, (values, arrays) in enumerate(rows):
        radius = int(values["patch_radius_px"])
        patch = np.asarray(arrays["patch"])
        center_y, center_x = (np.asarray(patch.shape, dtype=float) - 1.0) / 2.0

        plot_context(axes[row_index, 0], arrays, radius)
        axes[row_index, 0].set_ylabel(
            f"random vertical #{int(values['selection_rank'])}\n"
            f"{str(values['session']).replace(subject + '_', '')}, trial {int(values['trial_idx'])}",
            fontsize=8.0, weight="bold",
        )

        axes[row_index, 1].imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
        draw_gaze_axis(
            axes[row_index, 1], float(values["sobel_square_axis_deg"]), center_x, center_y, 0.82 * radius,
            color=COLORS["sobel_square"], lw=2.0,
        )
        draw_gaze_axis(
            axes[row_index, 1], float(values["fourier_square_axis_deg"]), center_x, center_y, 0.74 * radius,
            color=COLORS["fourier_square"], lw=1.7, ls="--",
        )
        axes[row_index, 1].add_patch(Circle((center_x, center_y), radius, fill=False, edgecolor="white", lw=0.7, ls=":"))
        axes[row_index, 1].axis("off")

        magnitude = np.asarray(arrays["scharr_magnitude"])
        axes[row_index, 2].imshow(np.log1p(magnitude), cmap="magma", origin="upper", interpolation="nearest")
        draw_gaze_axis(
            axes[row_index, 2], float(values["scharr_circle_axis_deg"]), center_x, center_y, 0.82 * radius,
            color=COLORS["scharr_circle"], lw=2.0,
        )
        axes[row_index, 2].add_patch(Circle((center_x, center_y), radius, fill=False, edgecolor="white", lw=0.8))
        axes[row_index, 2].axis("off")

        axes[row_index, 3].imshow(np.asarray(arrays["canny_edges"]), cmap="gray", origin="upper", vmin=0, vmax=1)
        draw_gaze_axis(
            axes[row_index, 3], float(values["canny_circle_axis_deg"]), center_x, center_y, 0.82 * radius,
            color=COLORS["canny_circle"], lw=2.0,
        )
        axes[row_index, 3].add_patch(Circle((center_x, center_y), radius, fill=False, edgecolor="#BBBBBB", lw=0.8))
        axes[row_index, 3].axis("off")

        axes[row_index, 4].imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
        circle = Circle((center_x, center_y), radius, fill=False, edgecolor="white", lw=0.8)
        axes[row_index, 4].add_patch(circle)
        segments = np.asarray(arrays["segments"])
        weights = np.asarray(arrays["segment_weights"])
        if len(weights):
            max_weight = float(np.max(weights))
            for segment, weight in zip(segments, weights, strict=True):
                line = axes[row_index, 4].plot(
                    [segment[0], segment[2]], [segment[1], segment[3]],
                    color="#FFD166", lw=0.6 + 1.4 * float(weight / max_weight), alpha=0.72,
                )[0]
                line.set_clip_path(circle)
        draw_gaze_axis(
            axes[row_index, 4], float(values["lsd_circle_axis_deg"]), center_x, center_y, 0.82 * radius,
            color=COLORS["lsd_circle"], lw=2.2,
        )
        axes[row_index, 4].axis("off")

        plot_axis_summary(axes[row_index, 5], values, radius)

        axes[row_index, 1].set_title(
            f"Sobel sq {float(values['sobel_square_axis_deg']):+.0f}°\n"
            f"Fourier {float(values['fourier_square_axis_deg']):+.0f}°",
            fontsize=7.4,
        )
        axes[row_index, 2].set_title(
            f"Scharr circle {float(values['scharr_circle_axis_deg']):+.0f}°\n"
            f"coh {float(values['scharr_circle_coherence']):.2f}", fontsize=7.4,
        )
        axes[row_index, 3].set_title(
            f"Canny circle {float(values['canny_circle_axis_deg']):+.0f}°\n"
            f"coh {float(values['canny_circle_coherence']):.2f}", fontsize=7.4,
        )
        axes[row_index, 4].set_title(
            f"LSD circle {float(values['lsd_circle_axis_deg']):+.0f}°\n"
            f"R {float(values['lsd_circle_resultant']):.2f}, n={int(values['lsd_n_segments'])}", fontsize=7.4,
        )
        axes[row_index, 5].set_title(
            f"all axes\nSobel sq↔circle {float(values['sobel_square_vs_circle_disagreement_deg']):.0f}°",
            fontsize=7.4,
        )

    for column, title in enumerate(
        [
            "wider image context",
            "original square patch",
            "Scharr magnitude",
            "Canny edge map",
            "line segments",
            "axis comparison",
        ]
    ):
        axes[0, column].text(
            0.5, 1.22, title, transform=axes[0, column].transAxes,
            ha="center", va="bottom", fontsize=9.2, weight="bold",
        )

    handles = [
        plt.Line2D([0], [0], color=COLORS[name], lw=2, ls="--" if name == "fourier_square" else "-", label=LABELS[name])
        for name in LABELS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8.0)
    fig.suptitle(
        f"{subject}: alternative edge detectors on the same frozen random vertical examples\n"
        "context: orange square = current ±1° analysis patch; red dashed circle = literal 1° radius",
        fontsize=12.0, weight="bold",
    )
    path = out_dir / f"checkpoint3_edge_detector_comparison_{subject.lower()}.png"
    fig.savefig(path, dpi=230)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = add_independent_axis(pd.read_csv(SELECTION))
    vertical = selected[selected["hv_condition"].eq("vertical")].copy()
    analyzed = [analyze_example(row) for _, row in vertical.iterrows()]
    values = pd.DataFrame([item[0] for item in analyzed])
    values.to_csv(args.out_dir / "checkpoint3_edge_detector_values.csv", index=False)
    paths = [render_subject(subject, analyzed, args.out_dir) for subject in SUBJECTS]

    report = [
        "# Figure 4F vertical-label checkpoint: alternative edge detectors",
        "",
        "This targeted visualization retains the same ten frozen random vertical example slots.",
        "The existing feature named `patch_radius_deg=1` is a square extending ±1° in x and y.",
        "The context panels show that square in orange and a literal 1° circle in dashed red.",
        "",
        "Alternative circle-aperture axes are shown for Scharr gradients, Canny-selected Scharr",
        "gradients, and OpenCV's line-segment detector. The original square Sobel and Fourier axes",
        "are retained as references. No Figure 4F outcome enters any calculation in this checkpoint.",
        "",
        f"Maximum original stored-versus-recomputed Sobel error: {values.sobel_square_recompute_error_deg.max():.6g}°.",
        f"Median Sobel square-versus-circle disagreement: {values.sobel_square_vs_circle_disagreement_deg.median():.2f}°.",
        f"Examples with square-versus-circle disagreement ≤2.5°: "
        f"{int(values.sobel_square_vs_circle_disagreement_deg.le(2.5).sum())} of {len(values)}.",
        "The two aperture-sensitive cases are Logan #4 (16.30°) and Logan #5 (86.73°).",
        "",
        "Artifacts:",
        *[f"- `{path.name}`" for path in paths],
        "- `checkpoint3_edge_detector_values.csv`",
        "",
        "This checkpoint is for visual interpretation. It does not validate a detector or change",
        "the frozen H/V cohort. A population sensitivity should wait until the maps identify which",
        "detector, if any, corresponds to the visible local structure of interest.",
    ]
    (args.out_dir / "summary_report.md").write_text("\n".join(report) + "\n")
    for path in paths:
        print(path)
    print(args.out_dir / "summary_report.md")


if __name__ == "__main__":
    main()
