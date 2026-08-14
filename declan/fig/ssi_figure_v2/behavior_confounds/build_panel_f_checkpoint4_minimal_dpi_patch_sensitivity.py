#!/usr/bin/env python3
"""Minimal eyepos-versus-dpi_pix local-patch sensitivity checkpoint.

The same ten frozen random vertical examples and the same square 1-degree
Sobel estimator are retained. Only the patch center changes: mean raw eyepos
versus mean shifter-corrected dpi_pix over the identical source samples.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint1 import (
    ROOT,
    SUBJECTS,
    axial_distance_deg,
    axial_signed_deg,
)
from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_horizontal_vertical_checkpoint2_axis_validation import (
    PARENT_OUT,
    SELECTION,
)
from declan.fixation_statistics_by_stimulus.extraction import _as_numpy, _load_dict_dataset
from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _patch_orientation_features,
)


DEFAULT_OUT = PARENT_OUT / "checkpoint4_minimal_dpi_patch_sensitivity"
BIN_HALF_WIDTH_DEG = 22.5
EYEPOS_COLOR = "#20A464"
DPI_COLOR = "#A33FA3"


@lru_cache(maxsize=64)
def session_arrays(session_name: str) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], float]:
    from DataYatesV1 import get_session

    subject, date = str(session_name).split("_", 1)
    session = get_session(subject, date)
    dset = _load_dict_dataset(Path(session.sess_dir) / "datasets/backimage.dset")
    eyepos = _as_numpy(dset["eyepos"]).astype(np.float64)
    dpi_pix = _as_numpy(dset["dpi_pix"]).astype(np.float64)
    trial_inds = _as_numpy(dset.covariates["trial_inds"]).reshape(-1).astype(int)
    trial_map = {int(trial): np.where(trial_inds == int(trial))[0] for trial in np.unique(trial_inds)}
    ppd = float(dset.metadata["ppd"])
    if eyepos.shape != dpi_pix.shape or eyepos.shape[0] != trial_inds.size:
        raise RuntimeError(f"Misaligned eyepos/dpi_pix arrays for {session_name}")
    return eyepos, dpi_pix, trial_map, ppd


def source_indices(row: pd.Series, trial_map: dict[int, np.ndarray]) -> np.ndarray:
    trial_idx = int(row.trial_idx)
    local_start = int(row.local_start)
    local_stop = int(row.local_stop)
    idx = trial_map.get(trial_idx)
    if idx is not None and idx.size >= local_stop:
        return idx[local_start:local_stop]
    return np.arange(int(row.global_start), int(row.global_stop), dtype=int)


def extract_square(canvas: np.ndarray, center_x: float, center_y: float, radius: int) -> np.ndarray:
    cx = int(round(float(center_x)))
    cy = int(round(float(center_y)))
    patch = np.asarray(canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1])
    expected = (2 * radius + 1, 2 * radius + 1)
    if patch.shape != expected:
        raise RuntimeError(f"Patch at {(center_x, center_y)} has shape {patch.shape}, expected {expected}")
    return patch


def draw_axis(
    ax: plt.Axes,
    axis_deg: float,
    center_x: float,
    center_y: float,
    half_length: float,
    color: str,
    **kwargs: object,
) -> None:
    theta_image = -np.radians(float(axis_deg))
    dx = half_length * np.cos(theta_image)
    dy = half_length * np.sin(theta_image)
    ax.plot(
        [center_x - dx, center_x + dx], [center_y - dy, center_y + dy],
        color=color, **kwargs,
    )


def analyze(row: pd.Series) -> tuple[dict[str, object], dict[str, np.ndarray | float]]:
    eyepos, dpi_pix, trial_map, ppd = session_arrays(str(row.session))
    idx = source_indices(row, trial_map)
    if len(idx) != int(row.local_stop) - int(row.local_start):
        raise RuntimeError(f"Unexpected source length for {row.session} trial {row.trial_idx}")
    raw_trace = eyepos[idx]
    dpi_trace = dpi_pix[idx]
    canvas, canvas_ppd, _ = _backimage_canvas(str(row.session), int(row.trial_idx))
    if not np.isclose(ppd, canvas_ppd, rtol=0, atol=1e-9):
        raise RuntimeError(f"PPD mismatch for {row.session}: {ppd} vs {canvas_ppd}")

    # Original center: physical gaze in x/right, y/up degrees mapped to screen.
    eyepos_mean = np.mean(raw_trace, axis=0)
    eyepos_center_x = canvas.shape[1] / 2.0 + eyepos_mean[0] * ppd
    eyepos_center_y = canvas.shape[0] / 2.0 - eyepos_mean[1] * ppd
    # dpi_pix is already screen [row, column].
    dpi_mean_row_col = np.mean(dpi_trace, axis=0)
    dpi_center_x = float(dpi_mean_row_col[1])
    dpi_center_y = float(dpi_mean_row_col[0])
    radius = int(row.image_patch_radius_px)

    eyepos_patch = extract_square(canvas, eyepos_center_x, eyepos_center_y, radius)
    dpi_patch = extract_square(canvas, dpi_center_x, dpi_center_y, radius)
    eyepos_features = _patch_orientation_features(eyepos_patch)
    dpi_features = _patch_orientation_features(dpi_patch)
    eyepos_axis = float(axial_signed_deg(eyepos_features["image_edge_axis_deg"]))
    dpi_axis = float(axial_signed_deg(dpi_features["image_edge_axis_deg"]))
    shift_px = float(np.hypot(dpi_center_x - eyepos_center_x, dpi_center_y - eyepos_center_y))
    center_audit_px = float(
        np.hypot(
            eyepos_center_x - float(row.image_patch_center_x_px),
            eyepos_center_y - float(row.image_patch_center_y_px),
        )
    )
    stored_axis = float(axial_signed_deg(row.image_edge_axis_deg))
    values: dict[str, object] = {
        "subject": str(row.subject),
        "selection_rank": int(row.selection_rank),
        "session": str(row.session),
        "trial_idx": int(row.trial_idx),
        "source_n_samples": int(len(idx)),
        "ppd": ppd,
        "patch_radius_px": radius,
        "eyepos_center_x_px": eyepos_center_x,
        "eyepos_center_y_px": eyepos_center_y,
        "dpi_center_x_px": dpi_center_x,
        "dpi_center_y_px": dpi_center_y,
        "stored_vs_recomputed_eyepos_center_error_px": center_audit_px,
        "center_shift_px": shift_px,
        "center_shift_arcmin": shift_px / ppd * 60.0,
        "stored_eyepos_axis_deg": stored_axis,
        "recomputed_eyepos_axis_deg": eyepos_axis,
        "stored_vs_recomputed_eyepos_axis_error_deg": float(axial_distance_deg(stored_axis, eyepos_axis)),
        "dpi_axis_deg": dpi_axis,
        "eyepos_vs_dpi_axis_disagreement_deg": float(axial_distance_deg(eyepos_axis, dpi_axis)),
        "eyepos_coherence": float(eyepos_features["image_orientation_coherence"]),
        "dpi_coherence": float(dpi_features["image_orientation_coherence"]),
        "eyepos_is_vertical_frozen_bin": bool(axial_distance_deg(eyepos_axis, 90.0) <= BIN_HALF_WIDTH_DEG),
        "dpi_is_vertical_frozen_bin": bool(axial_distance_deg(dpi_axis, 90.0) <= BIN_HALF_WIDTH_DEG),
    }
    arrays: dict[str, np.ndarray | float] = {
        "canvas": canvas,
        "eyepos_patch": eyepos_patch,
        "dpi_patch": dpi_patch,
    }
    return values, arrays


def context_crop(
    canvas: np.ndarray,
    centers: list[tuple[float, float]],
    radius: int,
) -> tuple[np.ndarray, float, float]:
    xs = np.asarray([center[0] for center in centers])
    ys = np.asarray([center[1] for center in centers])
    margin = 1.55 * radius
    x0 = max(0, int(np.floor(xs.min() - margin)))
    x1 = min(canvas.shape[1], int(np.ceil(xs.max() + margin)) + 1)
    y0 = max(0, int(np.floor(ys.min() - margin)))
    y1 = min(canvas.shape[0], int(np.ceil(ys.max() + margin)) + 1)
    return np.asarray(canvas[y0:y1, x0:x1]), float(x0), float(y0)


def render_subject(
    subject: str,
    analyzed: list[tuple[dict[str, object], dict[str, np.ndarray | float]]],
    out_dir: Path,
) -> Path:
    rows = [(values, arrays) for values, arrays in analyzed if values["subject"] == subject]
    rows.sort(key=lambda item: int(item[0]["selection_rank"]))
    fig, axes = plt.subplots(len(rows), 4, figsize=(10.8, 2.65 * len(rows)), constrained_layout=True)
    for row_index, (values, arrays) in enumerate(rows):
        radius = int(values["patch_radius_px"])
        ex = float(values["eyepos_center_x_px"])
        ey = float(values["eyepos_center_y_px"])
        dx = float(values["dpi_center_x_px"])
        dy = float(values["dpi_center_y_px"])
        canvas = np.asarray(arrays["canvas"])
        context, x0, y0 = context_crop(canvas, [(ex, ey), (dx, dy)], radius)

        ax = axes[row_index, 0]
        ax.imshow(context, cmap="gray", origin="upper", interpolation="nearest")
        for cx, cy, color, linestyle in [
            (ex - x0, ey - y0, EYEPOS_COLOR, "-"),
            (dx - x0, dy - y0, DPI_COLOR, "--"),
        ]:
            ax.add_patch(
                Rectangle((cx - radius, cy - radius), 2 * radius, 2 * radius,
                          fill=False, edgecolor=color, lw=1.4, ls=linestyle)
            )
            ax.scatter([cx], [cy], marker="+", s=22, color=color, linewidths=1.1)
        ax.plot([ex - x0, dx - x0], [ey - y0, dy - y0], color="#F2C14E", lw=1.0)
        ax.axis("off")
        ax.set_ylabel(
            f"#{int(values['selection_rank'])}\nshift {float(values['center_shift_arcmin']):.1f}'",
            fontsize=8.3, weight="bold",
        )

        for column, patch_key, axis_key, coherence_key, color, title in [
            (1, "eyepos_patch", "recomputed_eyepos_axis_deg", "eyepos_coherence", EYEPOS_COLOR, "raw eyepos center"),
            (2, "dpi_patch", "dpi_axis_deg", "dpi_coherence", DPI_COLOR, "dpi_pix center"),
        ]:
            patch = np.asarray(arrays[patch_key])
            center_y, center_x = (np.asarray(patch.shape, dtype=float) - 1.0) / 2.0
            axes[row_index, column].imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
            draw_axis(
                axes[row_index, column], float(values[axis_key]), center_x, center_y,
                0.82 * radius, color, lw=2.1,
            )
            axes[row_index, column].add_patch(
                Circle((center_x, center_y), radius, fill=False, edgecolor="white", lw=0.6, ls=":")
            )
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(
                f"{title}: {float(values[axis_key]):+.1f}°\ncoh {float(values[coherence_key]):.2f}",
                fontsize=7.7,
            )

        glyph = axes[row_index, 3]
        center = float(radius)
        glyph.set_facecolor("#F6F7F8")
        glyph.add_patch(Circle((center, center), 0.86 * radius, fill=False, edgecolor="#B8BEC5", lw=0.8))
        draw_axis(glyph, float(values["recomputed_eyepos_axis_deg"]), center, center, 0.82 * radius, EYEPOS_COLOR, lw=2.0)
        draw_axis(glyph, float(values["dpi_axis_deg"]), center, center, 0.74 * radius, DPI_COLOR, lw=2.0, ls="--")
        glyph.set_xlim(-1, 2 * radius + 1)
        glyph.set_ylim(2 * radius + 1, -1)
        glyph.set_aspect("equal")
        glyph.axis("off")
        glyph.set_title(
            f"Δaxis {float(values['eyepos_vs_dpi_axis_disagreement_deg']):.1f}°\n"
            f"dpi vertical: {'yes' if values['dpi_is_vertical_frozen_bin'] else 'NO'}",
            fontsize=7.7,
            color="#A12B2B" if not values["dpi_is_vertical_frozen_bin"] else "#333333",
            weight="bold",
        )

    for column, title in enumerate(["same image context", "eyepos patch", "dpi_pix patch", "axis change"]):
        axes[0, column].text(
            0.5, 1.22, title, transform=axes[0, column].transAxes,
            ha="center", va="bottom", fontsize=9.1, weight="bold",
        )
    handles = [
        plt.Line2D([0], [0], color=EYEPOS_COLOR, lw=2, label="raw eyepos-centered"),
        plt.Line2D([0], [0], color=DPI_COLOR, lw=2, ls="--", label="dpi_pix-centered"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle(
        f"{subject}: minimal dpi_pix patch-center sensitivity on frozen vertical examples\n"
        "identical samples, image, 1° square aperture, and Sobel estimator",
        fontsize=11.4, weight="bold",
    )
    path = out_dir / f"checkpoint4_minimal_dpi_patch_sensitivity_{subject.lower()}.png"
    fig.savefig(path, dpi=230)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(SELECTION)
    vertical = selected[selected["hv_condition"].eq("vertical")].copy()
    analyzed = [analyze(row) for _, row in vertical.iterrows()]
    values = pd.DataFrame([item[0] for item in analyzed])
    values.to_csv(args.out_dir / "checkpoint4_minimal_dpi_patch_sensitivity_values.csv", index=False)
    paths = [render_subject(subject, analyzed, args.out_dir) for subject in SUBJECTS]

    report = [
        "# Minimal dpi_pix patch-center sensitivity",
        "",
        "Targeted input-level check only: the same ten frozen random vertical examples, source",
        "samples, static image, square 1° aperture, and Sobel estimator were retained. Only the",
        "patch center changed from mean raw eyepos to mean shifter-corrected dpi_pix.",
        "Raw eyepos remains the FEM trajectory in both cases.",
        "",
        f"- median center shift: {values.center_shift_arcmin.median():.2f} arcmin",
        f"- maximum center shift: {values.center_shift_arcmin.max():.2f} arcmin",
        f"- median axis change: {values.eyepos_vs_dpi_axis_disagreement_deg.median():.2f}°",
        f"- maximum axis change: {values.eyepos_vs_dpi_axis_disagreement_deg.max():.2f}°",
        f"- dpi_pix-centered patches retaining the frozen vertical bin: "
        f"{int(values.dpi_is_vertical_frozen_bin.sum())} of {len(values)}",
        f"- maximum stored-versus-recomputed eyepos center error: "
        f"{values.stored_vs_recomputed_eyepos_center_error_px.max():.6g} px",
        f"- maximum stored-versus-recomputed eyepos Sobel-axis error: "
        f"{values.stored_vs_recomputed_eyepos_axis_error_deg.max():.6g}°",
        "",
        "No population effect or revised cohort is computed here.",
        "",
        "Artifacts:",
        *[f"- `{path.name}`" for path in paths],
        "- `checkpoint4_minimal_dpi_patch_sensitivity_values.csv`",
    ]
    (args.out_dir / "summary_report.md").write_text("\n".join(report) + "\n")
    for path in paths:
        print(path)
    print(args.out_dir / "summary_report.md")


if __name__ == "__main__":
    main()
