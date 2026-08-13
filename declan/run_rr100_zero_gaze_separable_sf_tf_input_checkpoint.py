#!/usr/bin/env python3
"""Validate a zero-gaze 51x51 SF/TF probe before native RR100 evaluation.

This is an input-only map-first checkpoint.  It derives the zero-eye retinal
ROI from each session's recorded eye-position/ROI mapping, extends the exact
ForageGrating carrier formula with a signed temporal phase ramp, and validates
requested versus measured spatial and temporal frequencies.  No model or
neural response is evaluated here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DataYatesV1.exp.gratings import GratingsTrial
from DataYatesV1.utils.io import get_session


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_input_checkpoint_v1"
)
DEFAULT_SESSIONS = ("Allen_2022-02-16", "Logan_2020-02-29")
IMAGE_SIZE = 51
FRAME_RATE_HZ = 120.0
HISTORY_FRAMES = 33
CONTRAST_FALLBACK = 0.8
SF_GRID = np.power(2.0, np.arange(0.0, 4.0 + 0.5, 0.5)).astype(np.float64)
TF_GRID = np.concatenate(
    [
        np.asarray([0.0]),
        np.power(2.0, np.arange(-1.0, 5.0 + 0.5, 0.5)),
        np.asarray([32.0 * math.sqrt(2.0)]),
    ]
).astype(np.float64)
REPRESENTATIVE_SFS = np.asarray([1.0, 2.0, 4.0, 8.0, 8.0 * math.sqrt(2.0), 16.0])
REPRESENTATIVE_SIGNED_TFS = np.asarray([0.0, 4.0, -4.0, 16.0, -16.0, 32.0 * math.sqrt(2.0)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--orientation-deg", type=float, default=90.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def derive_zero_gaze_roi(dset, session: str) -> tuple[np.ndarray, dict[str, Any]]:
    eyepos = np.asarray(dset["eyepos"], dtype=np.float64)
    rois = np.asarray(dset["roi"], dtype=np.float64)
    center_xy = np.column_stack(
        [
            0.5 * (rois[:, 1, 0] + rois[:, 1, 1]),
            0.5 * (rois[:, 0, 0] + rois[:, 0, 1]),
        ]
    )
    valid = np.isfinite(eyepos).all(axis=1) & np.isfinite(center_xy).all(axis=1)
    design = np.column_stack([np.ones(int(valid.sum())), eyepos[valid]])
    coefficients, *_ = np.linalg.lstsq(design, center_xy[valid], rcond=None)
    predicted = design @ coefficients
    residual = center_xy[valid] - predicted
    zero_center_xy = coefficients[0]
    x0 = int(np.rint(float(zero_center_xy[0]) - IMAGE_SIZE / 2.0))
    y0 = int(np.rint(float(zero_center_xy[1]) - IMAGE_SIZE / 2.0))
    roi = np.asarray([[y0, y0 + IMAGE_SIZE], [x0, x0 + IMAGE_SIZE]], dtype=np.int64)
    nearest = int(np.nanargmin(np.sum(eyepos * eyepos, axis=1)))
    total_ss = np.sum((center_xy[valid] - np.mean(center_xy[valid], axis=0)) ** 2, axis=0)
    residual_ss = np.sum(residual**2, axis=0)
    r2 = 1.0 - residual_ss / np.maximum(total_ss, 1e-12)
    row = {
        "session": session,
        "n_valid_eye_samples": int(valid.sum()),
        "zero_eye_center_x_px": float(zero_center_xy[0]),
        "zero_eye_center_y_px": float(zero_center_xy[1]),
        "zero_roi_y0": int(roi[0, 0]),
        "zero_roi_y1": int(roi[0, 1]),
        "zero_roi_x0": int(roi[1, 0]),
        "zero_roi_x1": int(roi[1, 1]),
        "eye_x_to_roi_x_px_per_deg": float(coefficients[1, 0]),
        "eye_y_to_roi_x_px_per_deg": float(coefficients[2, 0]),
        "eye_x_to_roi_y_px_per_deg": float(coefficients[1, 1]),
        "eye_y_to_roi_y_px_per_deg": float(coefficients[2, 1]),
        "roi_x_regression_r2": float(r2[0]),
        "roi_y_regression_r2": float(r2[1]),
        "roi_center_residual_rmse_px": float(np.sqrt(np.mean(residual**2))),
        "nearest_recorded_eye_x_deg": float(eyepos[nearest, 0]),
        "nearest_recorded_eye_y_deg": float(eyepos[nearest, 1]),
        "nearest_recorded_roi_center_x_px": float(center_xy[nearest, 0]),
        "nearest_recorded_roi_center_y_px": float(center_xy[nearest, 1]),
    }
    return roi, row


def carrier_coordinates(trial: GratingsTrial, roi: np.ndarray, orientation_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = 2001.0
    x = np.arange(int(roi[1, 0]), int(roi[1, 1]), dtype=np.float64)
    y = np.arange(int(roi[0, 0]), int(roi[0, 1]), dtype=np.float64)
    xx, yy = np.meshgrid(x - res / 2.0, y - res / 2.0)
    ori_rad = math.radians(90.0 - float(orientation_deg))
    normal_px = np.cos(ori_rad) * xx + np.sin(ori_rad) * yy
    return normal_px, x, y


def make_renderer_extended_movie(
    trial: GratingsTrial,
    roi: np.ndarray,
    *,
    spatial_cpd: float,
    signed_temporal_hz: float,
    orientation_deg: float,
    phase_rad: float,
    n_frames: int,
) -> np.ndarray:
    normal_px, x, y = carrier_coordinates(trial, roi, orientation_deg)
    t = np.arange(int(n_frames), dtype=np.float64) / FRAME_RATE_HZ
    spatial_phase = 2.0 * math.pi * float(spatial_cpd) * normal_px / float(trial.ppd)
    phase = spatial_phase[None] - 2.0 * math.pi * float(signed_temporal_hz) * t[:, None, None] + float(phase_rad)
    grating = np.cos(phase) * 0.5 * float(trial.grating_contrast) * 255.0
    x_oob = (x < trial.screen_rect[1, 0]) | (x > trial.screen_rect[1, 1])
    y_oob = (y < trial.screen_rect[0, 0]) | (y > trial.screen_rect[0, 1])
    grating[:, :, x_oob] = 0.0
    grating[:, y_oob, :] = 0.0
    return np.clip(grating + float(trial.bkgnd) + 0.5, 0.0, 255.0).astype(np.uint8)


def renderer_pixel_audit(trial: GratingsTrial, roi: np.ndarray, spatial_cpd: float, orientation_deg: float) -> dict[str, Any]:
    reference_grating = trial.gen_grating(
        float(spatial_cpd), float(orientation_deg), float(trial.grating_contrast), roi
    )
    reference = np.clip(reference_grating + float(trial.bkgnd) + 0.5, 0.0, 255.0).astype(np.uint8)
    extended = make_renderer_extended_movie(
        trial,
        roi,
        spatial_cpd=float(spatial_cpd),
        signed_temporal_hz=0.0,
        orientation_deg=float(orientation_deg),
        phase_rad=0.0,
        n_frames=1,
    )[0]
    diff = np.abs(reference.astype(np.int16) - extended.astype(np.int16))
    return {
        "spatial_cpd": float(spatial_cpd),
        "orientation_deg": float(orientation_deg),
        "max_abs_pixel_difference": int(np.max(diff)),
        "n_different_pixels": int(np.count_nonzero(diff)),
        "exact_match": bool(not np.any(diff)),
    }


def estimate_spatial_frequency(
    frame: np.ndarray,
    trial: GratingsTrial,
    roi: np.ndarray,
    orientation_deg: float,
) -> float:
    normal_px, _, _ = carrier_coordinates(trial, roi, orientation_deg)
    signal = frame.astype(np.float64) - float(trial.bkgnd)
    candidates = np.linspace(0.5, 18.0, 3501)
    scores = np.empty(len(candidates), dtype=np.float64)
    for i, sf in enumerate(candidates):
        phase = 2.0 * math.pi * sf * normal_px / float(trial.ppd)
        coefficient = np.sum(signal * np.exp(-1j * phase))
        scores[i] = float(np.abs(coefficient) ** 2)
    return float(candidates[int(np.argmax(scores))])


def estimate_signed_temporal_frequency(
    movie: np.ndarray,
    trial: GratingsTrial,
    roi: np.ndarray,
    spatial_cpd: float,
    orientation_deg: float,
) -> tuple[float, float]:
    normal_px, _, _ = carrier_coordinates(trial, roi, orientation_deg)
    spatial_phase = 2.0 * math.pi * float(spatial_cpd) * normal_px / float(trial.ppd)
    centered = movie.astype(np.float64) - float(trial.bkgnd)
    coefficients = np.sum(centered * np.exp(-1j * spatial_phase)[None], axis=(1, 2))
    unwrapped = np.unwrap(np.angle(coefficients))
    t = np.arange(len(movie), dtype=np.float64) / FRAME_RATE_HZ
    design = np.column_stack([t, np.ones_like(t)])
    slope, intercept = np.linalg.lstsq(design, unwrapped, rcond=None)[0]
    predicted = slope * t + intercept
    ss_res = float(np.sum((unwrapped - predicted) ** 2))
    ss_tot = float(np.sum((unwrapped - np.mean(unwrapped)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-12 and ss_res <= 1e-12 else 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(-slope / (2.0 * math.pi)), float(r2)


def make_sampling_table(ppd: float) -> pd.DataFrame:
    fov_deg = IMAGE_SIZE / float(ppd)
    history_s = HISTORY_FRAMES / FRAME_RATE_HZ
    rows: list[dict[str, Any]] = []
    for sf in SF_GRID:
        for tf in TF_GRID:
            cycles_crop = float(sf * fov_deg)
            pixels_cycle = float(ppd / sf)
            cycles_history = float(tf * history_s)
            frames_cycle = float("inf") if np.isclose(tf, 0.0) else float(FRAME_RATE_HZ / tf)
            sf_edge = bool(pixels_cycle < 3.0)
            tf_static = bool(np.isclose(tf, 0.0))
            tf_subcycle = bool((not tf_static) and cycles_history < 1.0)
            tf_edge = bool((not tf_static) and frames_cycle < 3.0)
            rows.append(
                {
                    "spatial_cpd": float(sf),
                    "temporal_hz_magnitude": float(tf),
                    "retinal_crop_deg": fov_deg,
                    "cycles_across_51px_crop": cycles_crop,
                    "pixels_per_spatial_cycle": pixels_cycle,
                    "history_frames": HISTORY_FRAMES,
                    "history_duration_s": history_s,
                    "cycles_per_history": cycles_history,
                    "frames_per_temporal_cycle": frames_cycle,
                    "spatial_cycle_valid": bool(cycles_crop >= 1.0),
                    "spatial_edge_control": sf_edge,
                    "temporal_static_control": tf_static,
                    "temporal_subcycle_control": tf_subcycle,
                    "temporal_edge_control": tf_edge,
                    "primary_separable_support": bool(
                        cycles_crop >= 1.0
                        and not sf_edge
                        and not tf_static
                        and not tf_subcycle
                        and not tf_edge
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_frames_and_kymographs(
    trial: GratingsTrial,
    roi: np.ndarray,
    orientation_deg: float,
    n_frames: int,
    out_path: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(14.2, 8.4))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.85], hspace=0.5, wspace=0.22)
    for col, sf in enumerate(REPRESENTATIVE_SFS):
        movie = make_renderer_extended_movie(
            trial,
            roi,
            spatial_cpd=float(sf),
            signed_temporal_hz=0.0,
            orientation_deg=orientation_deg,
            phase_rad=0.0,
            n_frames=1,
        )
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(movie[0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{sf:g} cpd\n{sf * IMAGE_SIZE / trial.ppd:.2f} cycles/crop", fontsize=9)
        ax.axis("off")
    for col, tf in enumerate(REPRESENTATIVE_SIGNED_TFS):
        movie = make_renderer_extended_movie(
            trial,
            roi,
            spatial_cpd=4.0,
            signed_temporal_hz=float(tf),
            orientation_deg=orientation_deg,
            phase_rad=0.0,
            n_frames=n_frames,
        )
        ax = fig.add_subplot(gs[1, col])
        kymograph = movie[:, IMAGE_SIZE // 2, :]
        ax.imshow(kymograph, cmap="gray", vmin=0, vmax=255, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(f"4 cpd, {tf:+g} Hz", fontsize=9)
        ax.set_xlabel("retinal x (px)")
        if col == 0:
            ax.set_ylabel("time (120-Hz frames)")
        else:
            ax.set_yticklabels([])
    history_t = np.arange(HISTORY_FRAMES) / FRAME_RATE_HZ
    ax = fig.add_subplot(gs[2, :3])
    for tf, color in zip([0.5, 2.0, 4.0, 16.0, 32.0 * math.sqrt(2.0)], ["#7A5195", "#5470C6", "#2A9D8F", "#E76F51", "#9D0208"]):
        ax.plot(history_t * 1000.0, np.cos(2.0 * math.pi * tf * history_t), lw=1.8, label=f"{tf:g} Hz ({tf * HISTORY_FRAMES / FRAME_RATE_HZ:.2f} cyc)")
    ax.set(xlabel="time within 33-frame history (ms)", ylabel="carrier at a fixed retinal position", title="History support: low TFs are locally sub-cycle")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, color="0.9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[2, 3:])
    phases = np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)
    colors = plt.cm.twilight(np.linspace(0.05, 0.95, len(phases)))
    x = np.arange(IMAGE_SIZE)
    normal_px, _, _ = carrier_coordinates(trial, roi, orientation_deg)
    row_phase = 2.0 * math.pi * 4.0 * normal_px[IMAGE_SIZE // 2] / trial.ppd
    for phase, color in zip(phases, colors):
        ax.plot(x, np.cos(row_phase + phase), color=color, lw=1.7, label=f"phase={phase / math.pi:g}π")
    ax.set(xlabel="retinal x (px)", ylabel="unit-contrast carrier", title="Four static phases remove absolute-origin privilege")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, color="0.9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Zero-gaze 51×51 separable SF/TF probe: concrete inputs",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.94,
        "Full-field ForageGrating carrier, fixed retinal ROI, 120 Hz. TF sign reverses drift direction; no model evaluated.",
        fontsize=9,
        color="0.32",
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_validation(
    sampling: pd.DataFrame,
    spectral: pd.DataFrame,
    roi_audit: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.1))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.86, hspace=0.38, wspace=0.31)
    ax = axes[0, 0]
    sf = spectral.drop_duplicates("requested_sf_cpd").sort_values("requested_sf_cpd")
    ax.plot(sf["requested_sf_cpd"], sf["measured_sf_cpd"], "o-", color="#0072B2")
    ax.plot([1, 16], [1, 16], "--", color="0.45", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="requested SF (cpd)", ylabel="measured carrier SF (cpd)", title="A. Spatial carrier validation")
    ax.set_xticks(REPRESENTATIVE_SFS, [f"{v:g}" for v in REPRESENTATIVE_SFS])
    ax.set_yticks(REPRESENTATIVE_SFS, [f"{v:g}" for v in REPRESENTATIVE_SFS])
    ax = axes[0, 1]
    tf = spectral.drop_duplicates("requested_signed_tf_hz").sort_values("requested_signed_tf_hz")
    ax.plot(tf["requested_signed_tf_hz"], tf["measured_signed_tf_hz"], "o", color="#D55E00")
    lim = 48
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.45", lw=1)
    ax.set(xlabel="requested signed TF (Hz)", ylabel="phase-slope measured TF (Hz)", title="B. Signed temporal carrier validation")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax = axes[1, 0]
    grid = sampling.drop_duplicates(["spatial_cpd", "temporal_hz_magnitude"])
    family = np.select(
        [
            grid["temporal_static_control"],
            grid["spatial_edge_control"] | grid["temporal_edge_control"],
            grid["temporal_subcycle_control"],
            grid["primary_separable_support"],
        ],
        ["static", "sampling edge", "sub-cycle TF", "primary"],
        default="other",
    )
    palette = {"static": "#6C757D", "sampling edge": "#C1121F", "sub-cycle TF": "#7A5195", "primary": "#2A9D8F", "other": "#B8B8B8"}
    for label in ["primary", "sub-cycle TF", "static", "sampling edge", "other"]:
        use = family == label
        if np.any(use):
            ax.scatter(grid.loc[use, "spatial_cpd"], grid.loc[use, "temporal_hz_magnitude"].replace(0, 0.25), s=25, color=palette[label], label=label, alpha=0.8)
    ax.set(xscale="log", yscale="log", xlabel="SF (cpd)", ylabel="|TF| (Hz; zero drawn at 0.25)", title="C. Predeclared sampling families")
    ax.set_xticks(SF_GRID, [f"{v:g}" for v in SF_GRID], rotation=35)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axes[1, 1]
    x = np.arange(len(roi_audit))
    width = 0.34
    ax.bar(x - width / 2, roi_audit["roi_x_regression_r2"], width, color="#0072B2", label="ROI x")
    ax.bar(x + width / 2, roi_audit["roi_y_regression_r2"], width, color="#D55E00", label="ROI y")
    ax.set_xticks(x, [v.replace("_", "\n") for v in roi_audit["session"]])
    ax.set_ylim(0.995, 1.0001)
    ax.set(ylabel="eye-position → ROI-center $R^2$", title="D. Zero-gaze ROI is auditable per session")
    ax.legend(frameon=False, fontsize=7)
    for ax in axes.ravel():
        ax.grid(True, color="0.92", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Zero-gaze SF/TF input and sampling validation", x=0.02, y=0.97, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.92, "Primary support requires ≥1 spatial cycle/crop, ≥1 temporal cycle/history, and ≥3 samples/cycle in both domains.", fontsize=9, color="0.32")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sessions = [v.strip() for v in str(args.sessions).split(",") if v.strip()]
    n_frames = int(round(float(args.duration_s) * FRAME_RATE_HZ))
    roi_rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    session_objects: list[tuple[str, Any, Any, np.ndarray]] = []
    for session_name in sessions:
        subject, date = session_name.split("_", maxsplit=1)
        sess = get_session(subject, date)
        dset = sess.get_dataset("gratings", strict=True)
        roi, roi_row = derive_zero_gaze_roi(dset, session_name)
        trial_index = int(np.asarray(dset["trial_inds"])[0])
        trial = GratingsTrial(sess.exp["D"][trial_index], sess.exp["S"])
        roi_row.update(
            {
                "pixels_per_degree": float(trial.ppd),
                "background_uint8": float(trial.bkgnd),
                "grating_contrast": float(trial.grating_contrast),
                "source_trial_index": trial_index,
            }
        )
        roi_rows.append(roi_row)
        for sf in REPRESENTATIVE_SFS:
            audit = renderer_pixel_audit(trial, roi, float(sf), float(args.orientation_deg))
            audit["session"] = session_name
            pixel_rows.append(audit)
        session_objects.append((session_name, sess, trial, roi))
        del dset

    roi_audit = pd.DataFrame(roi_rows)
    pixel_audit = pd.DataFrame(pixel_rows)
    if not bool(pixel_audit["exact_match"].all()):
        raise AssertionError("The signed-TF renderer does not reduce exactly to the original phase-zero renderer")
    reference_name, _reference_sess, trial, roi = session_objects[0]
    sampling = make_sampling_table(float(trial.ppd))
    spectral_rows: list[dict[str, Any]] = []
    measured_sf_by_requested: dict[float, float] = {}
    for sf in REPRESENTATIVE_SFS:
        for tf in REPRESENTATIVE_SIGNED_TFS:
            movie = make_renderer_extended_movie(
                trial,
                roi,
                spatial_cpd=float(sf),
                signed_temporal_hz=float(tf),
                orientation_deg=float(args.orientation_deg),
                phase_rad=0.0,
                n_frames=n_frames,
            )
            if float(sf) not in measured_sf_by_requested:
                measured_sf_by_requested[float(sf)] = estimate_spatial_frequency(
                    movie[0], trial, roi, float(args.orientation_deg)
                )
            measured_sf = measured_sf_by_requested[float(sf)]
            measured_tf, phase_r2 = estimate_signed_temporal_frequency(
                movie,
                trial,
                roi,
                float(sf),
                float(args.orientation_deg),
            )
            spectral_rows.append(
                {
                    "session": reference_name,
                    "requested_sf_cpd": float(sf),
                    "measured_sf_cpd": measured_sf,
                    "sf_absolute_error_cpd": abs(measured_sf - float(sf)),
                    "requested_signed_tf_hz": float(tf),
                    "measured_signed_tf_hz": measured_tf,
                    "tf_absolute_error_hz": abs(measured_tf - float(tf)),
                    "temporal_phase_fit_r2": phase_r2,
                    "orientation_deg": float(args.orientation_deg),
                    "n_frames": n_frames,
                }
            )
    spectral = pd.DataFrame(spectral_rows)
    roi_audit.to_csv(args.out_dir / "zero_gaze_roi_audit.csv", index=False)
    pixel_audit.to_csv(args.out_dir / "renderer_extension_pixel_audit.csv", index=False)
    sampling.to_csv(args.out_dir / "planned_sf_tf_sampling_grid.csv", index=False)
    spectral.to_csv(args.out_dir / "requested_vs_measured_frequency_validation.csv", index=False)
    frames_png = args.out_dir / "zero_gaze_sf_tf_frames_and_kymographs.png"
    validation_png = args.out_dir / "zero_gaze_sf_tf_sampling_validation.png"
    plot_frames_and_kymographs(trial, roi, float(args.orientation_deg), n_frames, frames_png, int(args.dpi))
    plot_validation(sampling, spectral, roi_audit, validation_png, int(args.dpi))
    manifest = {
        "analysis": "rr100_zero_gaze_separable_sf_tf_input_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_no_model_evaluation",
        "sessions": sessions,
        "reference_session_for_visuals": reference_name,
        "image_size_px": IMAGE_SIZE,
        "frame_rate_hz": FRAME_RATE_HZ,
        "history_frames": HISTORY_FRAMES,
        "history_duration_s": HISTORY_FRAMES / FRAME_RATE_HZ,
        "spatial_cpd": SF_GRID.tolist(),
        "temporal_hz_magnitude": TF_GRID.tolist(),
        "orientation_deg_for_visuals": float(args.orientation_deg),
        "zero_gaze_rule": "session-specific ROI center at eyepos=(0,0), derived by linear regression from recorded eyepos to ROI center",
        "renderer_rule": "exact ForageGrating full-field cosine carrier plus signed phase(t)=-2*pi*TF*t; no probes/faces",
        "phase_plan": {
            "tf_zero": 4,
            "abs_tf_below_4_hz": 2,
            "abs_tf_at_least_4_hz": 1,
            "signed_directions": True,
        },
        "primary_response_plan": [
            "phase-averaged mean native fitted rate above gray",
            "F1 sinusoidal response amplitude",
            "preferred-orientation and orientation-marginal surfaces",
        ],
        "separability_plan": "rank-one SF x |TF| factorization with raw residual and leading-component variance explained retained per unit",
        "pixel_audit_all_exact": bool(pixel_audit["exact_match"].all()),
        "pixel_audit_max_abs_difference": int(pixel_audit["max_abs_pixel_difference"].max()),
        "maximum_measured_sf_error_cpd": float(spectral["sf_absolute_error_cpd"].max()),
        "maximum_measured_tf_error_hz": float(spectral["tf_absolute_error_hz"].max()),
        "minimum_temporal_phase_fit_r2": float(spectral["temporal_phase_fit_r2"].min()),
        "artifacts": {
            "frames_figure": frames_png.name,
            "sampling_figure": validation_png.name,
            "roi_audit": "zero_gaze_roi_audit.csv",
            "pixel_audit": "renderer_extension_pixel_audit.csv",
            "sampling_grid": "planned_sf_tf_sampling_grid.csv",
            "frequency_validation": "requested_vs_measured_frequency_validation.csv",
        },
        "script": file_identity(Path(__file__)),
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
