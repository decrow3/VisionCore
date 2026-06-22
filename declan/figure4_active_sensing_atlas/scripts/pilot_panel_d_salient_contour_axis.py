#!/usr/bin/env python3
"""Pilot a connected salient-contour axis estimator for Figure 4D."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

try:
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
ATLAS = REPO_ROOT / "declan" / "figure4_active_sensing_atlas"
DEFAULT_INPUT = BASE / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
DEFAULT_MANIFEST = BASE / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1" / "selected_windows.csv"
DEFAULT_WTA_VALUES = BASE / "backimage_wta_orientation_axis_input_v1" / "selected_wta_axis_values.csv"
DEFAULT_OUT_DIR = ATLAS / "figures" / "panel_D" / "diagnostics" / "salient_contour_axis_pilot"

INK = "#20262c"
MUTED = "#68727d"
GRID = "#dfe4e9"
AVG = "#244f7a"
WTA = "#c15b44"
SALIENT = "#2f8f6a"
HOUGH = "#6d5aa8"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _axis_delta_deg(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def _normalize_axis_deg(angle_deg: float) -> float:
    return float(((float(angle_deg) + 90.0) % 180.0) - 90.0)


def _normalize_image(patch: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _uint8_image(patch: np.ndarray) -> np.ndarray:
    return np.asarray(np.round(_normalize_image(patch) * 255.0), dtype=np.uint8)


def _crop_centered(image: np.ndarray, center_xy: tuple[float, float], size: int) -> np.ndarray:
    height, width = image.shape[:2]
    half = int(size) // 2
    cx, cy = center_xy
    x0 = max(0, min(width - int(size), int(round(cx)) - half))
    y0 = max(0, min(height - int(size), int(round(cy)) - half))
    return np.asarray(image[y0 : y0 + int(size), x0 : x0 + int(size)], dtype=np.float64)


def _patches_for_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    canvas, ppd, screen_shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center = gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=float(ppd),
        screen_shape=screen_shape,
    )
    thumbnail = _crop_centered(canvas, (float(center[0]), float(center[1])), 190)
    radius = int(round(float(row["image_patch_radius_px"])))
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))
    analysis = canvas[
        max(0, cy - radius) : min(canvas.shape[0], cy + radius + 1),
        max(0, cx - radius) : min(canvas.shape[1], cx + radius + 1),
    ]
    return np.asarray(thumbnail, dtype=np.float64), np.asarray(analysis, dtype=np.float64)


def _weighted_pca_axis(coords_xy: np.ndarray, weights: np.ndarray) -> dict[str, float | np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    coords = np.asarray(coords_xy, dtype=np.float64)
    wsum = float(np.sum(weights))
    if coords.shape[0] < 3 or wsum <= 0:
        raise ValueError("not enough weighted contour pixels")
    center = np.sum(coords * weights[:, None], axis=0) / wsum
    centered = coords - center[None, :]
    cov = (centered * weights[:, None]).T @ centered / wsum
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis_vec = eigvecs[:, 0]
    if axis_vec[0] < 0:
        axis_vec = -axis_vec
    angle_array = _normalize_axis_deg(np.degrees(np.arctan2(float(axis_vec[1]), float(axis_vec[0]))))
    angle_display = _normalize_axis_deg(-angle_array)
    denom = float(eigvals[0] + eigvals[1] + 1e-12)
    coherence = float(max(0.0, (eigvals[0] - eigvals[1]) / denom))
    proj = centered @ axis_vec
    length = float(np.percentile(proj, 95.0) - np.percentile(proj, 5.0))
    return {
        "center_xy": center,
        "axis_vec_xy": axis_vec,
        "axis_array_deg": angle_array,
        "axis_display_deg": angle_display,
        "coherence": coherence,
        "length_px": max(length, 0.0),
        "proj_min": float(np.min(proj)),
        "proj_max": float(np.max(proj)),
    }


def salient_contour_axis_from_patch(
    patch: np.ndarray,
    *,
    min_component_pixels: int = 12,
    canny_low: int = 50,
    canny_high: int = 145,
) -> dict[str, Any]:
    """Choose the strongest connected contour near the patch center."""
    if min(patch.shape[:2]) < 12:
        return {"salient_ok": False, "salient_error": "patch_too_small"}
    img = _uint8_image(patch)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(blur, int(canny_low), int(canny_high))
    if int(np.count_nonzero(edges)) < int(min_component_pixels):
        edges = cv2.Canny(blur, max(10, int(canny_low * 0.55)), max(30, int(canny_high * 0.65)))
    if int(np.count_nonzero(edges)) < int(min_component_pixels):
        return {"salient_ok": False, "salient_error": "too_few_canny_edges"}

    grad_x = cv2.Sobel(blur.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(grad_x, grad_y).astype(np.float64)
    grad_ref = float(np.nanpercentile(grad[grad > 0], 95.0)) if np.any(grad > 0) else 1.0
    grad_ref = max(grad_ref, 1e-6)

    dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    n_labels, labels = cv2.connectedComponents(dilated, connectivity=8)
    patch_center = np.asarray([(patch.shape[1] - 1.0) / 2.0, (patch.shape[0] - 1.0) / 2.0], dtype=np.float64)
    scale = max(2.0, 0.24 * float(min(patch.shape[:2])))
    components: list[dict[str, Any]] = []

    for label in range(1, int(n_labels)):
        mask = (labels == label) & (edges > 0)
        ys, xs = np.nonzero(mask)
        if xs.size < int(min_component_pixels):
            continue
        coords = np.column_stack([xs, ys]).astype(np.float64)
        weights = grad[ys, xs] + 1e-6
        try:
            axis = _weighted_pca_axis(coords, weights)
        except ValueError:
            continue
        center = np.asarray(axis["center_xy"], dtype=np.float64)
        vec = np.asarray(axis["axis_vec_xy"], dtype=np.float64)
        rel = patch_center - center
        proj_center = float(rel @ vec)
        proj_min = float(axis["proj_min"])
        proj_max = float(axis["proj_max"])
        if proj_min <= proj_center <= proj_max:
            closest = center + proj_center * vec
            distance = float(np.linalg.norm(patch_center - closest))
        else:
            endpoint_a = center + proj_min * vec
            endpoint_b = center + proj_max * vec
            distance = float(min(np.linalg.norm(patch_center - endpoint_a), np.linalg.norm(patch_center - endpoint_b)))
        proximity = float(np.exp(-((distance / scale) ** 2)))
        contrast = float(np.clip(np.mean(weights) / grad_ref, 0.0, 2.0))
        length = float(axis["length_px"])
        coherence = float(axis["coherence"])
        score = float(length * max(coherence, 1e-3) ** 1.35 * max(contrast, 1e-3) * (0.15 + 0.85 * proximity))
        components.append(
            {
                "salient_axis_deg": float(axis["axis_display_deg"]),
                "salient_axis_array_deg": float(axis["axis_array_deg"]),
                "salient_score": score,
                "salient_length_px": length,
                "salient_coherence": coherence,
                "salient_contrast": contrast,
                "salient_center_distance_px": distance,
                "salient_proximity": proximity,
                "salient_n_edge_pixels": int(xs.size),
                "salient_component_label": int(label),
            }
        )
    if not components:
        return {"salient_ok": False, "salient_error": "no_component_survived_filter"}
    best = max(components, key=lambda item: float(item["salient_score"]))
    best["salient_ok"] = True
    best["salient_error"] = ""
    best["salient_n_components"] = int(len(components))
    return best


def _hough_rail_axis_deg(patch: np.ndarray) -> float:
    img = _uint8_image(patch)
    edges = cv2.Canny(img, 80, 180)
    min_line = max(12, int(round(min(patch.shape) * 0.24)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_line, maxLineGap=15)
    if lines is None:
        return float("nan")
    segments = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        axis_array = _normalize_axis_deg(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 20.0 <= axis_array <= 55.0 and length >= min_line:
            segments.append((length, axis_array))
    if not segments:
        return float("nan")
    weights = np.asarray([item[0] for item in segments], dtype=np.float64)
    angles = np.radians([item[1] for item in segments])
    mean_array = 0.5 * np.arctan2(float(np.sum(weights * np.sin(2.0 * angles))), float(np.sum(weights * np.cos(2.0 * angles))))
    return _normalize_axis_deg(-float(np.degrees(mean_array)))


def _axis_vector(axis_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(axis_deg))
    return np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)


def _add_axis(ax: plt.Axes, axis_deg: float, color: str, *, scale: float, lw: float = 2.0) -> None:
    if not np.isfinite(float(axis_deg)):
        return
    vec = _axis_vector(float(axis_deg))
    center = np.asarray([0.5, 0.5])
    ax.add_patch(
        FancyArrowPatch(
            tuple(center - vec * scale),
            tuple(center + vec * scale),
            arrowstyle="<|-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            transform=ax.transAxes,
            alpha=0.95,
        )
    )


def _plot_patch(
    ax: plt.Axes,
    patch: np.ndarray,
    *,
    avg_axis: float,
    wta_axis: float | None,
    salient_axis: float,
    hough_axis: float | None = None,
    title: str,
) -> None:
    ax.imshow(_normalize_image(patch), cmap="gray", vmin=0, vmax=1)
    _add_axis(ax, avg_axis, AVG, scale=0.36, lw=2.0)
    if wta_axis is not None and np.isfinite(float(wta_axis)):
        _add_axis(ax, float(wta_axis), WTA, scale=0.28, lw=2.0)
    _add_axis(ax, salient_axis, SALIENT, scale=0.20, lw=2.3)
    if hough_axis is not None and np.isfinite(float(hough_axis)):
        _add_axis(ax, float(hough_axis), HOUGH, scale=0.13, lw=2.1)
    ax.set_title(title, fontsize=7.5, color=INK, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _load_selected_rows(input_path: Path, manifest_path: Path, wta_values_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(input_path)
    windows["source_row"] = np.arange(windows.shape[0], dtype=int)
    manifest = pd.read_csv(manifest_path)
    selected = windows.set_index("source_row", drop=False).loc[manifest["source_row"].astype(int).to_list()].reset_index(drop=True)
    if wta_values_path.exists():
        wta = pd.read_csv(wta_values_path)[["source_row", "wta_edge_axis_deg", "wta_peak_fraction"]]
        selected = selected.merge(wta, on="source_row", how="left")
    else:
        selected["wta_edge_axis_deg"] = np.nan
        selected["wta_peak_fraction"] = np.nan
    return windows, selected


def _run_selected_pilot(selected: pd.DataFrame, windows: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    thumbnail_cache: dict[int, np.ndarray] = {}
    analysis_cache: dict[int, np.ndarray] = {}
    for idx, row in selected.iterrows():
        thumbnail, analysis = _patches_for_row(row)
        source_row = int(row["source_row"])
        thumbnail_cache[source_row] = thumbnail
        analysis_cache[source_row] = analysis
        salient = salient_contour_axis_from_patch(
            analysis,
            min_component_pixels=int(args.min_component_pixels),
            canny_low=int(args.canny_low),
            canny_high=int(args.canny_high),
        )
        avg_axis = float(row["image_edge_axis_deg"])
        wta_axis = float(row.get("wta_edge_axis_deg", np.nan))
        rec = {
            "source_row": source_row,
            "session": str(row["session"]),
            "trial_idx": int(row["trial_idx"]),
            "image_edge_axis_deg": avg_axis,
            "wta_edge_axis_deg": wta_axis,
            "wta_peak_fraction": float(row.get("wta_peak_fraction", np.nan)),
            **salient,
        }
        rec["salient_average_abs_delta_deg"] = abs(float(_axis_delta_deg(rec.get("salient_axis_deg", np.nan), avg_axis)))
        rec["salient_wta_abs_delta_deg"] = abs(float(_axis_delta_deg(rec.get("salient_axis_deg", np.nan), wta_axis)))
        records.append(rec)
        if idx + 1 == selected.shape[0] or (idx + 1) % int(args.progress_every) == 0:
            print(f"salient contour pilot {idx + 1}/{selected.shape[0]} selected windows", flush=True)
    values = pd.DataFrame(records)

    full = windows.copy()
    for col in [
        "salient_contour_axis_deg",
        "salient_contour_score",
        "salient_contour_length_px",
        "salient_contour_coherence",
        "salient_contour_contrast",
        "salient_contour_proximity",
        "salient_contour_ok",
        "salient_contour_error",
    ]:
        if col not in full:
            full[col] = np.nan if col not in {"salient_contour_ok", "salient_contour_error"} else (False if col == "salient_contour_ok" else "")
    for rec in records:
        mask = full["source_row"].astype(int).eq(int(rec["source_row"]))
        full.loc[mask, "salient_contour_axis_deg"] = rec.get("salient_axis_deg", np.nan)
        full.loc[mask, "salient_contour_score"] = rec.get("salient_score", np.nan)
        full.loc[mask, "salient_contour_length_px"] = rec.get("salient_length_px", np.nan)
        full.loc[mask, "salient_contour_coherence"] = rec.get("salient_coherence", np.nan)
        full.loc[mask, "salient_contour_contrast"] = rec.get("salient_contrast", np.nan)
        full.loc[mask, "salient_contour_proximity"] = rec.get("salient_proximity", np.nan)
        full.loc[mask, "salient_contour_ok"] = bool(rec.get("salient_ok", False))
        full.loc[mask, "salient_contour_error"] = str(rec.get("salient_error", ""))

    values.to_csv(out_dir / "selected_salient_contour_axis_values.csv", index=False)
    full.to_csv(out_dir / "backimage_image_fem_windows_salient_contour_axis.csv", index=False)
    return values, pd.DataFrame(
        [
            {
                "source_row": source_row,
                "thumbnail_patch": thumbnail_cache[source_row],
                "analysis_patch": analysis_cache[source_row],
            }
            for source_row in thumbnail_cache
        ]
    )


def _rail_rows(input_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    windows = pd.read_csv(input_path)
    rows = []
    for raw_index in [17, 18]:
        row = windows.iloc[int(raw_index)]
        thumbnail, analysis = _patches_for_row(row)
        avg_axis = float(row["image_edge_axis_deg"])
        for patch_name, patch in [("thumbnail_190px", thumbnail), ("analysis_patch", analysis)]:
            salient = salient_contour_axis_from_patch(
                patch,
                min_component_pixels=int(args.min_component_pixels),
                canny_low=int(args.canny_low),
                canny_high=int(args.canny_high),
            )
            hough = _hough_rail_axis_deg(patch)
            rows.append(
                {
                    "raw_window_row": int(raw_index),
                    "patch": patch_name,
                    "session": str(row["session"]),
                    "trial_idx": int(row["trial_idx"]),
                    "image_edge_axis_deg": avg_axis,
                    "hough_visible_rail_axis_deg": hough,
                    **salient,
                    "salient_hough_abs_delta_deg": abs(float(_axis_delta_deg(salient.get("salient_axis_deg", np.nan), hough))),
                    "average_hough_abs_delta_deg": abs(float(_axis_delta_deg(avg_axis, hough))),
                }
            )
    return pd.DataFrame(rows)


def _plot_summary_sheet(values: pd.DataFrame, patch_table: pd.DataFrame, out_dir: Path) -> Path:
    ok = values[values["salient_ok"].astype(bool)].copy()
    thumb_rows = ok.sort_values(["salient_average_abs_delta_deg", "salient_score"], ascending=False).head(12)
    fig = plt.figure(figsize=(12.0, 9.4), constrained_layout=False)
    gs = GridSpec(5, 4, figure=fig, height_ratios=[0.82, 0.82, 1.0, 1.0, 1.0], hspace=0.55, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax1.hist(ok["salient_average_abs_delta_deg"], bins=np.linspace(0, 90, 19), color=SALIENT, alpha=0.82)
    ax1.set_title("Salient contour vs average axis")
    ax1.set_xlabel("|salient - average| (deg)")
    ax1.set_ylabel("windows")
    ax2.hist(ok["salient_wta_abs_delta_deg"], bins=np.linspace(0, 90, 19), color=WTA, alpha=0.82)
    ax2.set_title("Salient contour vs current WTA")
    ax2.set_xlabel("|salient - WTA| (deg)")
    ax2.set_ylabel("windows")
    ax3.scatter(ok["salient_score"], ok["salient_average_abs_delta_deg"], s=18, color=INK, alpha=0.65)
    ax3.set_xlabel("salient component score")
    ax3.set_ylabel("|salient - average| (deg)")
    ax3.set_title("Disagreement is mostly estimator choice")
    ax4.scatter(ok["salient_length_px"], ok["salient_coherence"], s=18, color=SALIENT, alpha=0.65)
    ax4.set_xlabel("component length (px)")
    ax4.set_ylabel("component coherence")
    ax4.set_title("Selected contour geometry")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(color=GRID, lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patch_lookup = patch_table.set_index("source_row")
    for i, (_, row) in enumerate(thumb_rows.iterrows()):
        ax = fig.add_subplot(gs[2 + i // 4, i % 4])
        source_row = int(row["source_row"])
        patch = patch_lookup.loc[source_row, "thumbnail_patch"]
        _plot_patch(
            ax,
            patch,
            avg_axis=float(row["image_edge_axis_deg"]),
            wta_axis=float(row.get("wta_edge_axis_deg", np.nan)),
            salient_axis=float(row["salient_axis_deg"]),
            title=(
                f"src {source_row} | Δavg {float(row['salient_average_abs_delta_deg']):.1f}°\n"
                f"avg {float(row['image_edge_axis_deg']):+.0f}, WTA {float(row.get('wta_edge_axis_deg', np.nan)):+.0f}, sal {float(row['salient_axis_deg']):+.0f}"
            ),
        )
    fig.suptitle(
        "Panel 4D salient connected-contour axis pilot",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.946,
        "Blue = patch-average axis. Orange = current orientation-bin WTA. Green = connected salient-contour axis. Thumbnails show largest salient-vs-average disagreements in the paired n=64 D manifest.",
        ha="left",
        va="top",
        fontsize=8.8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.055)
    path = out_dir / "4D_salient_contour_axis_pilot_sheet.png"
    fig.savefig(path, dpi=230, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_rail_sheet(rail: pd.DataFrame, input_path: Path, args: argparse.Namespace, out_dir: Path) -> Path:
    windows = pd.read_csv(input_path)
    fig = plt.figure(figsize=(10.4, 5.0), constrained_layout=False)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.18)
    for col, raw_index in enumerate([17, 18]):
        row = windows.iloc[int(raw_index)]
        thumbnail, analysis = _patches_for_row(row)
        for j, (patch_name, patch) in enumerate([("thumbnail_190px", thumbnail), ("analysis_patch", analysis)]):
            rec = rail[(rail["raw_window_row"].eq(raw_index)) & (rail["patch"].eq(patch_name))].iloc[0]
            ax = fig.add_subplot(gs[col, j * 2 : j * 2 + 1])
            _plot_patch(
                ax,
                patch,
                avg_axis=float(rec["image_edge_axis_deg"]),
                wta_axis=None,
                salient_axis=float(rec["salient_axis_deg"]),
                hough_axis=float(rec["hough_visible_rail_axis_deg"]),
                title=(
                    f"row {raw_index}: {patch_name.replace('_', ' ')}\n"
                    f"avg {float(rec['image_edge_axis_deg']):+.1f}, salient {float(rec['salient_axis_deg']):+.1f}, rail {float(rec['hough_visible_rail_axis_deg']):+.1f}"
                ),
            )
            txt = (
                f"salient-rail error {float(rec['salient_hough_abs_delta_deg']):.2f} deg\n"
                f"average-rail error {float(rec['average_hough_abs_delta_deg']):.2f} deg\n"
                f"score {float(rec['salient_score']):.2f}, coh {float(rec['salient_coherence']):.2f}, prox {float(rec['salient_proximity']):.2f}"
            )
            ax_text = fig.add_subplot(gs[col, j * 2 + 1 : j * 2 + 2])
            ax_text.axis("off")
            ax_text.text(0.0, 0.70, txt, ha="left", va="top", fontsize=8.2, color=INK, transform=ax_text.transAxes)
    fig.suptitle(
        "Salient connected-contour axis on the diagonal rail stress test",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.93,
        "Blue = patch-average. Green = salient connected contour. Purple = visible rail/Hough fit.",
        ha="left",
        va="top",
        fontsize=8.7,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.84, bottom=0.06)
    path = out_dir / "4D_salient_contour_axis_rail_test.png"
    fig.savefig(path, dpi=230, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--wta-values", type=Path, default=DEFAULT_WTA_VALUES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=145)
    parser.add_argument("--progress-every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows, selected = _load_selected_rows(Path(args.input), Path(args.manifest), Path(args.wta_values))
    values, patch_table = _run_selected_pilot(selected, windows, args, out_dir)
    rail = _rail_rows(Path(args.input), args)
    rail.to_csv(out_dir / "rail_salient_contour_axis_values.csv", index=False)
    sheet = _plot_summary_sheet(values, patch_table, out_dir)
    rail_sheet = _plot_rail_sheet(rail, Path(args.input), args, out_dir)
    metadata = {
        "input": Path(args.input),
        "manifest": Path(args.manifest),
        "wta_values": Path(args.wta_values),
        "out_dir": out_dir,
        "n_selected": int(selected.shape[0]),
        "n_salient_ok": int(values["salient_ok"].astype(bool).sum()),
        "elapsed_s": float(time.time() - start),
        "outputs": [
            sheet,
            sheet.with_suffix(".pdf"),
            rail_sheet,
            rail_sheet.with_suffix(".pdf"),
            out_dir / "selected_salient_contour_axis_values.csv",
            out_dir / "rail_salient_contour_axis_values.csv",
            out_dir / "backimage_image_fem_windows_salient_contour_axis.csv",
        ],
        "parameters": {
            "min_component_pixels": int(args.min_component_pixels),
            "canny_low": int(args.canny_low),
            "canny_high": int(args.canny_high),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in metadata["outputs"]:
        print(path)


if __name__ == "__main__":
    main()
