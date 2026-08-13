#!/usr/bin/env python3
"""Checkpoint 4: separate movement, image-analysis, and gaze scales.

The drift sheet uses an auditable 2x2 movement-RMS by gaze-eccentricity
selection within one session per animal, holding fixation phase fixed.  Each
selected trace and gaze center is then re-expressed at three image patch radii.
Detected small and large high-speed events are selected and rendered in a
separate sheet; they are not mixed into the clean-window factorial.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint1_reference_frame_examples as cp1,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint2_pairing_null_examples as cp2,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint2b_behavior_absolute_rms as cp2b_absolute,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint3_behavior_object_multilag as cp3,
)
from declan.fixation_statistics_by_stimulus.image_features import (
    _backimage_canvas,
    backimage_trial_geometry,
    gaze_deg_to_screen_px,
    image_axis_rad_to_gaze_axis_rad,
)
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _window_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = cp1.DEFAULT_OUT_DIR / "checkpoint4_scale_eccentricity_v1"
WINDOW_FEATURES = (
    REPO_ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv"
)
RADIUS_SWEEP = (
    REPO_ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_patch_radius_sensitivity_v1/patch_radius_alignment_sweep_windows.csv"
)
EVENT_FEATURES = (
    REPO_ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/saccade_event_features.csv"
)

RADII_DEG = (0.5, 1.25, 2.0)
N_ROTATIONS = 256
PHASE = "mid_fixation"
SUBJECTS = ("Allen", "Logan")
CELL_ORDER = (
    "low_movement_central",
    "low_movement_eccentric",
    "high_movement_central",
    "high_movement_eccentric",
)
CELL_LABELS = {
    "low_movement_central": "Low movement / lower-eccentricity gaze",
    "low_movement_eccentric": "Low movement / higher-eccentricity gaze",
    "high_movement_central": "High movement / lower-eccentricity gaze",
    "high_movement_eccentric": "High movement / higher-eccentricity gaze",
}
EVENT_CLASSES = (
    ("small_event", 0.2, 1.0, 0.5),
    ("large_event", 4.0, 6.0, 5.0),
)

PARALLEL_COLOR = "#1b7f5c"
NORMAL_COLOR = "#7a3b9a"
DELTA_COLOR = "#20262c"
COHERENCE_COLOR = "#8a929a"
ALIGNMENT_COLOR = "#276b9b"
EVENT_COLOR = "#c45632"
SUBJECT_COLORS = {"Allen": "#245c8a", "Logan": "#b26b22"}


def _radius_key(radius: float) -> str:
    return f"r{str(float(radius)).replace('.', 'p')}"


def _prepare_complete_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(WINDOW_FEATURES)
    windows["source_window_index"] = windows.index.astype(int)
    windows = windows[
        windows["stimulus"].astype(str).eq("backimage")
        & windows["phase"].astype(str).eq(PHASE)
    ].copy()
    windows["subject"] = windows["session"].astype(str).str.split("_", n=1).str[0]

    sweep = pd.read_csv(RADIUS_SWEEP)
    sweep = sweep[sweep["patch_radius_deg"].astype(float).isin(RADII_DEG)].copy()
    counts = sweep.groupby("source_window_index")["patch_radius_deg"].nunique()
    complete = counts[counts.eq(len(RADII_DEG))].index
    windows = windows[
        windows["source_window_index"].isin(complete)
        & windows["subject"].isin(SUBJECTS)
        & windows["anisotropy"].astype(float).ge(0.30)
        & windows["rms_radius_deg"].astype(float).between(0.015, 0.30)
        & windows["abs_mean_radius_deg"].astype(float).between(0.25, 10.0)
    ].copy()
    return windows, sweep


def _cell_mask(
    table: pd.DataFrame,
    cell: str,
    *,
    rms_q25: float,
    rms_q75: float,
    ecc_q25: float,
    ecc_q75: float,
) -> pd.Series:
    movement_low = cell.startswith("low_movement")
    gaze_central = cell.endswith("central")
    movement = (
        table["rms_radius_deg"].astype(float).le(rms_q25)
        if movement_low
        else table["rms_radius_deg"].astype(float).ge(rms_q75)
    )
    gaze = (
        table["abs_mean_radius_deg"].astype(float).le(ecc_q25)
        if gaze_central
        else table["abs_mean_radius_deg"].astype(float).ge(ecc_q75)
    )
    return movement & gaze


def _session_support(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, block in table.groupby("session", sort=True):
        rms_q25, rms_q75 = block["rms_radius_deg"].quantile([0.25, 0.75])
        ecc_q25, ecc_q75 = block["abs_mean_radius_deg"].quantile([0.25, 0.75])
        counts = {
            cell: int(
                _cell_mask(
                    block,
                    cell,
                    rms_q25=float(rms_q25),
                    rms_q75=float(rms_q75),
                    ecc_q25=float(ecc_q25),
                    ecc_q75=float(ecc_q75),
                ).sum()
            )
            for cell in CELL_ORDER
        }
        rows.append(
            {
                "session": session,
                "subject": str(session).split("_", 1)[0],
                "n_eligible_session_windows": int(len(block)),
                "rms_q25_deg": float(rms_q25),
                "rms_q75_deg": float(rms_q75),
                "ecc_q25_deg": float(ecc_q25),
                "ecc_q75_deg": float(ecc_q75),
                "minimum_cell_support": int(min(counts.values())),
                **{f"n_{cell}": count for cell, count in counts.items()},
            }
        )
    return pd.DataFrame(rows)


def _select_drift_examples(
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = _session_support(windows)
    selected_rows: list[pd.Series] = []
    for subject in SUBJECTS:
        candidates = support[
            support["subject"].astype(str).eq(subject)
            & support["minimum_cell_support"].astype(int).gt(0)
        ].sort_values(
            ["minimum_cell_support", "n_eligible_session_windows", "session"],
            ascending=[False, False, True],
            kind="stable",
        )
        if candidates.empty:
            raise RuntimeError(f"No session supports the complete factorial for {subject}")
        chosen_support = candidates.iloc[0]
        session = str(chosen_support["session"])
        block = windows[windows["session"].astype(str).eq(session)].copy()
        rms_q25 = float(chosen_support["rms_q25_deg"])
        rms_q75 = float(chosen_support["rms_q75_deg"])
        ecc_q25 = float(chosen_support["ecc_q25_deg"])
        ecc_q75 = float(chosen_support["ecc_q75_deg"])
        rms_targets = block["rms_radius_deg"].quantile([0.10, 0.90]).to_dict()
        ecc_targets = block["abs_mean_radius_deg"].quantile([0.10, 0.90]).to_dict()
        since_median = float(block["samples_since_event"].median())
        since_iqr = float(
            block["samples_since_event"].quantile(0.75)
            - block["samples_since_event"].quantile(0.25)
        )
        used_trials: set[int] = set()

        for cell in CELL_ORDER:
            pool = block[
                _cell_mask(
                    block,
                    cell,
                    rms_q25=rms_q25,
                    rms_q75=rms_q75,
                    ecc_q25=ecc_q25,
                    ecc_q75=ecc_q75,
                )
            ].copy()
            movement_low = cell.startswith("low_movement")
            gaze_central = cell.endswith("central")
            rms_target = float(rms_targets[0.10 if movement_low else 0.90])
            ecc_target = float(ecc_targets[0.10 if gaze_central else 0.90])
            pool["selection_score"] = (
                np.abs(np.log(pool["rms_radius_deg"].astype(float) / rms_target))
                + np.abs(pool["abs_mean_radius_deg"].astype(float) - ecc_target)
                / max(ecc_q75 - ecc_q25, 0.25)
                + 0.20 * np.abs(pool["anisotropy"].astype(float) - 0.60)
                + 0.10
                * np.abs(pool["samples_since_event"].astype(float) - since_median)
                / max(since_iqr, 1.0)
            )
            pool = pool.sort_values(
                ["selection_score", "trial_idx", "local_start"], kind="stable"
            )
            unused = pool[~pool["trial_idx"].astype(int).isin(used_trials)]
            if unused.empty:
                unused = pool
            if unused.empty:
                raise RuntimeError(f"No selectable {cell} row for {session}")
            row = unused.iloc[0].copy()
            row["factorial_cell"] = cell
            row["selection_rule"] = (
                f"{PHASE}; anisotropy>=0.30; complete radii {RADII_DEG}; "
                "session selected by maximum minimum 2x2 cell support; window selected "
                "without using edge alignment or image coherence"
            )
            row["session_rms_q25_deg"] = rms_q25
            row["session_rms_q75_deg"] = rms_q75
            row["session_ecc_q25_deg"] = ecc_q25
            row["session_ecc_q75_deg"] = ecc_q75
            row["eligible_cell_count"] = int(len(pool))
            row["selected_session_minimum_cell_support"] = int(
                chosen_support["minimum_cell_support"]
            )
            selected_rows.append(row)
            used_trials.add(int(row["trial_idx"]))
    return pd.DataFrame(selected_rows).reset_index(drop=True), support


@lru_cache(maxsize=64)
def _trial_image_fields(
    session: str, trial_idx: int
) -> tuple[np.ndarray, float, tuple[int, int], dict[str, Any], np.ndarray, np.ndarray]:
    canvas, ppd, shape = _backimage_canvas(session, int(trial_idx))
    geometry = backimage_trial_geometry(session, int(trial_idx))
    arr = np.asarray(canvas, dtype=np.float64)
    gx = ndimage.sobel(arr, axis=1, mode="nearest")
    gy = ndimage.sobel(arr, axis=0, mode="nearest")
    return arr, float(ppd), shape, geometry, gx, gy


def _feature_at_gaze(
    session: str, trial_idx: int, gaze_xy_deg: np.ndarray, radius_deg: float
) -> dict[str, Any]:
    canvas, ppd, (height, width), geometry, gx, gy = _trial_image_fields(
        session, int(trial_idx)
    )
    cx, cy = gaze_deg_to_screen_px(
        np.asarray(gaze_xy_deg, dtype=np.float64), ppd=ppd, screen_shape=(height, width)
    )
    rad = max(2, int(round(float(radius_deg) * ppd)))
    x0, x1 = max(0, int(round(cx)) - rad), min(width, int(round(cx)) + rad + 1)
    y0, y1 = max(0, int(round(cy)) - rad), min(height, int(round(cy)) + rad + 1)
    patch = np.asarray(canvas[y0:y1, x0:x1], dtype=np.float64)
    gx_patch, gy_patch = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
    dest_x0, dest_y0, dest_x1, dest_y1 = geometry["dest_rect"]
    yy, xx = np.indices(patch.shape)
    screen_x, screen_y = xx + x0, yy + y0
    inside = (
        (screen_x >= dest_x0)
        & (screen_x < dest_x1)
        & (screen_y >= dest_y0)
        & (screen_y < dest_y1)
    )
    background = np.isclose(patch, float(geometry["background"]), atol=1e-6)
    jxx = float(np.mean(gx_patch * gx_patch))
    jyy = float(np.mean(gy_patch * gy_patch))
    jxy = float(np.mean(gx_patch * gy_patch))
    denom = jxx + jyy
    coherence = (
        float(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2) / denom)
        if denom > 0
        else float("nan")
    )
    gradient_array = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    edge_array = gradient_array + np.pi / 2.0
    edge_gaze = image_axis_rad_to_gaze_axis_rad(edge_array)
    inside_fraction = float(np.mean(inside))
    background_fraction = float(np.mean(background))
    return {
        "patch": patch,
        "ppd": ppd,
        "center_x_px": float(cx),
        "center_y_px": float(cy),
        "radius_px": int(rad),
        "fraction_inside_image": inside_fraction,
        "fraction_background": background_fraction,
        "passes_patch_qc": bool(
            inside_fraction >= 0.98 and background_fraction <= 0.05
        ),
        "image_orientation_coherence": coherence,
        "image_edge_axis_deg": float(np.degrees(edge_gaze)),
        "image_edge_axis_array_deg": float(np.degrees(edge_array)),
    }


def _window_radius_values(
    selected: pd.DataFrame, sweep: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, int, float], np.ndarray]]:
    lookup = sweep.set_index(["source_window_index", "patch_radius_deg"])
    rotation_angles = np.pi * (np.arange(N_ROTATIONS, dtype=float) + 0.5) / N_ROTATIONS
    rows: list[dict[str, Any]] = []
    traces: dict[tuple[str, int, float], np.ndarray] = {}
    for row in selected.itertuples(index=False):
        series = pd.Series(row._asdict())
        trace = np.asarray(_window_trace(series), dtype=np.float64)
        if trace.shape != (128, 2) or not np.isfinite(trace).all():
            raise RuntimeError(f"Bad drift trace for {row.session} trial {row.trial_idx}")
        traces[(str(row.subject), int(row.source_window_index), 0.0)] = trace
        rotated = [cp2.rotate_trace(trace, float(angle)) for angle in rotation_angles]
        for radius in RADII_DEG:
            radius_row = lookup.loc[(int(row.source_window_index), float(radius))]
            edge_axis = float(radius_row["image_edge_axis_deg"])
            real = cp2b_absolute._rms_components(trace, edge_axis)
            null_delta = np.asarray(
                [
                    cp2b_absolute._rms_components(item, edge_axis)[
                        "parallel_minus_normal_rms_arcmin"
                    ]
                    for item in rotated
                ],
                dtype=float,
            )
            rows.append(
                {
                    "factorial_cell": str(row.factorial_cell),
                    "subject": str(row.subject),
                    "session": str(row.session),
                    "trial_idx": int(row.trial_idx),
                    "source_window_index": int(row.source_window_index),
                    "global_start": int(row.global_start),
                    "global_stop": int(row.global_stop),
                    "movement_rms_deg": float(row.rms_radius_deg),
                    "gaze_eccentricity_deg": float(row.abs_mean_radius_deg),
                    "screen_anisotropy": float(row.anisotropy),
                    "mean_x_deg": float(row.mean_x_deg),
                    "mean_y_deg": float(row.mean_y_deg),
                    "phase": str(row.phase),
                    "samples_since_event": float(row.samples_since_event),
                    "patch_radius_deg": float(radius),
                    "image_edge_axis_deg": edge_axis,
                    "image_orientation_coherence": float(
                        radius_row["image_orientation_coherence"]
                    ),
                    "drift_edge_cos2": float(radius_row["drift_edge_cos2"]),
                    "image_patch_fraction_inside_image": float(
                        radius_row["image_patch_fraction_inside_image"]
                    ),
                    "image_patch_fraction_background": float(
                        radius_row["image_patch_fraction_background"]
                    ),
                    **real,
                    "rotation_mean_parallel_minus_normal_rms_arcmin": float(
                        np.mean(null_delta)
                    ),
                    "real_minus_rotation_mean_rms_advantage_arcmin": float(
                        real["parallel_minus_normal_rms_arcmin"] - np.mean(null_delta)
                    ),
                    "n_rotation_draws": int(N_ROTATIONS),
                }
            )
    return pd.DataFrame(rows), traces


def _crop_window_patch(row: pd.Series, radius: float) -> np.ndarray:
    canvas, ppd, shape, _geometry, _gx, _gy = _trial_image_fields(
        str(row["session"]), int(row["trial_idx"])
    )
    cx, cy = gaze_deg_to_screen_px(
        np.asarray([row["mean_x_deg"], row["mean_y_deg"]], dtype=float),
        ppd=ppd,
        screen_shape=shape,
    )
    rad = max(2, int(round(float(radius) * ppd)))
    return np.asarray(
        canvas[
            int(round(cy)) - rad : int(round(cy)) + rad + 1,
            int(round(cx)) - rad : int(round(cx)) + rad + 1,
        ]
    )


def _plot_screen_and_zoom(ax: plt.Axes, row: pd.Series, trace: np.ndarray) -> None:
    ax.scatter([0], [0], marker="+", s=35, c="#8f969d")
    ax.plot(trace[:, 0], trace[:, 1], color=SUBJECT_COLORS[str(row["subject"])], lw=0.9)
    ax.scatter(trace[0, 0], trace[0, 1], s=13, c="#36a2eb", zorder=4)
    ax.scatter(trace[-1, 0], trace[-1, 1], s=14, c="#df4c4c", marker="s", zorder=4)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.16, lw=0.5)
    inset = ax.inset_axes([0.52, 0.52, 0.45, 0.45])
    centered = (trace - np.mean(trace, axis=0, keepdims=True)) * 60.0
    inset.plot(centered[:, 0], centered[:, 1], color="#33383e", lw=0.6)
    limit = max(float(np.max(np.abs(centered))) * 1.1, 1.0)
    inset.set_xlim(-limit, limit)
    inset.set_ylim(-limit, limit)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    ax.set_title(
        f"screen gaze + path zoom\nRMS={float(row['rms_radius_deg']):.3f}°; ecc={float(row['abs_mean_radius_deg']):.2f}°",
        fontsize=7.2,
        weight="bold",
    )


def _plot_window_patch(
    ax: plt.Axes,
    row: pd.Series,
    trace: np.ndarray,
    radius_row: pd.Series,
    radius: float,
) -> None:
    patch = _crop_window_patch(row, radius)
    ax.imshow(
        patch,
        cmap="gray",
        origin="upper",
        extent=(-radius, radius, -radius, radius),
        interpolation="nearest",
    )
    centered = trace - np.mean(trace, axis=0, keepdims=True)
    ax.plot(centered[:, 0], centered[:, 1], color="#f2ca52", lw=0.75)
    direction = cp1.axis_vector(float(radius_row["image_edge_axis_deg"])) * (0.72 * radius)
    ax.plot(
        [-direction[0], direction[0]],
        [-direction[1], direction[1]],
        color=PARALLEL_COLOR,
        lw=1.7,
    )
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"r={radius:g}° patch\ncoh={float(radius_row['image_orientation_coherence']):.2f}; axis={float(radius_row['image_edge_axis_deg']):+.0f}°",
        fontsize=7.0,
        weight="bold",
    )


def _plot_window_curve(ax: plt.Axes, values: pd.DataFrame) -> None:
    values = values.sort_values("patch_radius_deg")
    x = values["patch_radius_deg"].to_numpy(dtype=float)
    ax.axhline(0, color="#9aa1a8", lw=0.6, ls=":")
    ax.plot(x, values["parallel_rms_arcmin"], color=PARALLEL_COLOR, marker="o", ms=3, lw=1.0, label="RMS ∥")
    ax.plot(x, values["normal_rms_arcmin"], color=NORMAL_COLOR, marker="o", ms=3, lw=1.0, label="RMS ⟂")
    ax.plot(
        x,
        values["real_minus_rotation_mean_rms_advantage_arcmin"],
        color=DELTA_COLOR,
        marker="*",
        ms=5,
        lw=1.1,
        label="D",
    )
    ax.set_xticks(x)
    ax.set_xlabel("patch radius (deg)", fontsize=6.2)
    ax.set_ylabel("absolute spread (arcmin)", fontsize=6.2)
    ax.tick_params(labelsize=5.5)
    ax.grid(axis="y", alpha=0.16, lw=0.5)
    right = ax.twinx()
    right.plot(x, values["drift_edge_cos2"], color=ALIGNMENT_COLOR, marker="s", ms=2.5, ls="--", lw=0.8)
    right.plot(x, values["image_orientation_coherence"], color=COHERENCE_COLOR, marker=".", ms=4, ls=":", lw=0.9)
    right.set_ylim(-1.05, 1.05)
    right.set_yticks([-1, 0, 1])
    right.set_ylabel("cos2 blue / coherence gray", fontsize=5.6)
    right.tick_params(labelsize=5.2)
    ax.set_title("same path; radius-only remeasurement", fontsize=7.2, weight="bold")


def _render_drift_sheet(
    selected: pd.DataFrame,
    values: pd.DataFrame,
    traces: dict[tuple[str, int, float], np.ndarray],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(CELL_ORDER),
        10,
        figsize=(25.0, 11.5),
        gridspec_kw={"width_ratios": [1.05, 1, 1, 1, 1.25] * 2},
    )
    for row_index, cell in enumerate(CELL_ORDER):
        for subject_index, subject in enumerate(SUBJECTS):
            col0 = subject_index * 5
            row = selected[
                selected["factorial_cell"].astype(str).eq(cell)
                & selected["subject"].astype(str).eq(subject)
            ].iloc[0]
            trace = traces[(subject, int(row["source_window_index"]), 0.0)]
            subset = values[
                values["factorial_cell"].astype(str).eq(cell)
                & values["subject"].astype(str).eq(subject)
            ]
            _plot_screen_and_zoom(axes[row_index, col0], row, trace)
            for radius_index, radius in enumerate(RADII_DEG):
                radius_row = subset[
                    np.isclose(subset["patch_radius_deg"].astype(float), radius)
                ].iloc[0]
                _plot_window_patch(
                    axes[row_index, col0 + 1 + radius_index],
                    row,
                    trace,
                    radius_row,
                    radius,
                )
            _plot_window_curve(axes[row_index, col0 + 4], subset)
            if subject_index == 0:
                axes[row_index, col0].set_ylabel(
                    CELL_LABELS[cell],
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=8.2,
                    weight="bold",
                )

    for subject_index, subject in enumerate(SUBJECTS):
        left = axes[0, subject_index * 5].get_position().x0
        right = axes[0, subject_index * 5 + 4].get_position().x1
        session = str(selected[selected["subject"].eq(subject)]["session"].iloc[0])
        fig.text(
            0.5 * (left + right),
            0.951,
            f"{subject.upper()} — {session}",
            ha="center",
            fontsize=10.2,
            weight="bold",
            color=SUBJECT_COLORS[subject],
        )
    fig.suptitle(
        "Figure 4 behavior audit — Checkpoint 4A: movement amplitude, gaze eccentricity, and image radius are different variables\n"
        "Within each animal/session: session-relative eccentricity quartiles selected without image alignment; each row keeps its gaze and trace fixed across radii",
        y=0.997,
        fontsize=12.5,
        weight="bold",
    )
    fig.subplots_adjust(left=0.145, right=0.994, top=0.915, bottom=0.06, hspace=0.52, wspace=0.48)
    boundary = 0.5 * (axes[0, 4].get_position().x1 + axes[0, 5].get_position().x0)
    fig.add_artist(plt.Line2D([boundary, boundary], [0.045, 0.955], transform=fig.transFigure, color="#a6adb5", lw=1.0))
    fig.text(
        0.99,
        0.018,
        "D = measured (RMS∥−RMS⟂) minus the mean of 256 common axial rotations. Blue cos2 is the raw Panel-H-style axis score; gray is image-axis coherence.",
        ha="right",
        fontsize=6.8,
        color="#4a4f55",
    )
    fig.savefig(out_dir / "checkpoint4_scale_eccentricity_examples.png", dpi=220)
    fig.savefig(out_dir / "checkpoint4_scale_eccentricity_examples.pdf")
    plt.close(fig)


def _select_events(sessions: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENT_FEATURES)
    events = events[
        events["stimulus"].astype(str).eq("backimage")
        & events["session"].isin(sessions.values())
        & events["event_duration_s"].astype(float).le(0.12)
    ].copy()
    selected: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        session = sessions[subject]
        used_trials: set[int] = set()
        session_events = events[events["session"].astype(str).eq(session)].copy()
        for event_class, lo, hi, target_amp in EVENT_CLASSES:
            pool = session_events[
                session_events["event_amplitude_deg"].astype(float).between(lo, hi)
            ].copy()
            pool["selection_score"] = np.abs(
                np.log(pool["event_amplitude_deg"].astype(float) / target_amp)
            )
            pool = pool.sort_values(
                ["selection_score", "trial_idx", "event_onset_sample"], kind="stable"
            )
            chosen_payload: dict[str, Any] | None = None
            chosen_radius_rows: list[dict[str, Any]] = []
            for candidate in pool.itertuples(index=False):
                if int(candidate.trial_idx) in used_trials:
                    continue
                trial_trace, valid = cp3._trial_trace_and_valid(session, int(candidate.trial_idx))
                onset, offset = int(candidate.event_onset_sample), int(candidate.event_offset_sample)
                if onset < 0 or offset >= len(trial_trace) or not np.all(valid[onset : offset + 1]):
                    continue
                event_trace = trial_trace[onset : offset + 1]
                center_gaze = np.mean(event_trace, axis=0)
                radius_rows = []
                for radius in RADII_DEG:
                    feature = _feature_at_gaze(session, int(candidate.trial_idx), center_gaze, radius)
                    if not bool(feature["passes_patch_qc"]):
                        radius_rows = []
                        break
                    axis_delta = float(
                        cp1.axial_distance_deg(
                            float(candidate.event_direction_deg),
                            float(feature["image_edge_axis_deg"]),
                        )
                    )
                    radius_rows.append(
                        {
                            "event_class": event_class,
                            "subject": subject,
                            "session": session,
                            "trial_idx": int(candidate.trial_idx),
                            "event_onset_sample": onset,
                            "event_offset_sample": offset,
                            "event_amplitude_deg": float(candidate.event_amplitude_deg),
                            "event_direction_deg": float(candidate.event_direction_deg),
                            "event_peak_speed_deg_s": float(candidate.event_peak_speed_deg_s),
                            "event_center_x_deg": float(center_gaze[0]),
                            "event_center_y_deg": float(center_gaze[1]),
                            "event_center_eccentricity_deg": float(np.linalg.norm(center_gaze)),
                            "patch_radius_deg": float(radius),
                            "image_edge_axis_deg": float(feature["image_edge_axis_deg"]),
                            "image_orientation_coherence": float(feature["image_orientation_coherence"]),
                            "event_edge_axis_delta_deg": axis_delta,
                            "event_edge_cos2": float(np.cos(2.0 * np.radians(axis_delta))),
                            "image_patch_fraction_inside_image": float(feature["fraction_inside_image"]),
                            "image_patch_fraction_background": float(feature["fraction_background"]),
                        }
                    )
                if radius_rows:
                    chosen_payload = {
                        "event_class": event_class,
                        "subject": subject,
                        "session": session,
                        "trial_idx": int(candidate.trial_idx),
                        "event_onset_sample": onset,
                        "event_offset_sample": offset,
                        "event_amplitude_deg": float(candidate.event_amplitude_deg),
                        "event_direction_deg": float(candidate.event_direction_deg),
                        "event_peak_speed_deg_s": float(candidate.event_peak_speed_deg_s),
                        "selection_score": float(candidate.selection_score),
                        "eligible_amplitude_candidates": int(len(pool)),
                        "selection_rule": (
                            f"{lo:g}<=amplitude<={hi:g} deg; duration<=0.12 s; "
                            f"target amplitude={target_amp:g} deg; complete patch QC at {RADII_DEG}; "
                            "no image alignment/coherence criterion"
                        ),
                    }
                    chosen_radius_rows = radius_rows
                    break
            if chosen_payload is None:
                raise RuntimeError(f"No valid {event_class} for {session}")
            selected.append(chosen_payload)
            values.extend(chosen_radius_rows)
            used_trials.add(int(chosen_payload["trial_idx"]))
    return pd.DataFrame(selected), pd.DataFrame(values)


def _event_segment(row: pd.Series) -> tuple[np.ndarray, np.ndarray, int, int]:
    trial_trace, _valid = cp3._trial_trace_and_valid(str(row["session"]), int(row["trial_idx"]))
    onset, offset = int(row["event_onset_sample"]), int(row["event_offset_sample"])
    start, stop = max(0, onset - 12), min(len(trial_trace), offset + 13)
    return trial_trace[start:stop], trial_trace[onset : offset + 1], onset - start, offset - start


def _plot_event_path(ax: plt.Axes, selection: pd.Series) -> None:
    segment, event_trace, onset_rel, offset_rel = _event_segment(selection)
    ax.plot(segment[:, 0], segment[:, 1], color="#8e969e", lw=0.7)
    ax.plot(
        segment[onset_rel : offset_rel + 1, 0],
        segment[onset_rel : offset_rel + 1, 1],
        color=EVENT_COLOR,
        lw=2.0,
    )
    ax.scatter(event_trace[0, 0], event_trace[0, 1], s=15, c="#36a2eb", zorder=4)
    ax.scatter(event_trace[-1, 0], event_trace[-1, 1], s=16, c="#df4c4c", marker="s", zorder=4)
    span = max(float(np.max(np.ptp(segment, axis=0))) * 0.62, 0.4)
    center = np.mean(event_trace, axis=0)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.16, lw=0.5)
    ax.set_title(
        f"detected event + context\namp={float(selection['event_amplitude_deg']):.2f}°; dir={float(selection['event_direction_deg']):+.0f}°",
        fontsize=7.2,
        weight="bold",
    )


def _plot_event_patch(
    ax: plt.Axes, selection: pd.Series, radius_row: pd.Series, radius: float
) -> None:
    _segment, event_trace, _onset_rel, _offset_rel = _event_segment(selection)
    center = np.mean(event_trace, axis=0)
    feature = _feature_at_gaze(
        str(selection["session"]), int(selection["trial_idx"]), center, radius
    )
    ax.imshow(
        feature["patch"],
        cmap="gray",
        origin="upper",
        extent=(-radius, radius, -radius, radius),
        interpolation="nearest",
    )
    centered_event = event_trace - center
    ax.plot(centered_event[:, 0], centered_event[:, 1], color=EVENT_COLOR, lw=1.5)
    direction = cp1.axis_vector(float(radius_row["image_edge_axis_deg"])) * (0.72 * radius)
    ax.plot([-direction[0], direction[0]], [-direction[1], direction[1]], color=PARALLEL_COLOR, lw=1.7)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"r={radius:g}° at event-mean gaze\ncoh={float(radius_row['image_orientation_coherence']):.2f}; Δaxis={float(radius_row['event_edge_axis_delta_deg']):.0f}°",
        fontsize=7.0,
        weight="bold",
    )


def _plot_event_curve(ax: plt.Axes, values: pd.DataFrame) -> None:
    values = values.sort_values("patch_radius_deg")
    x = values["patch_radius_deg"].to_numpy(dtype=float)
    ax.plot(x, values["event_edge_axis_delta_deg"], color=EVENT_COLOR, marker="o", ms=3.5, lw=1.1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 45, 90])
    ax.set_xticks(x)
    ax.set_xlabel("patch radius (deg)", fontsize=6.2)
    ax.set_ylabel("event-edge axial Δ (deg)", fontsize=6.2)
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.16, lw=0.5)
    right = ax.twinx()
    right.plot(x, values["image_orientation_coherence"], color=COHERENCE_COLOR, marker="s", ms=2.5, ls=":", lw=0.9)
    right.set_ylim(0, 1)
    right.set_yticks([0, 0.5, 1])
    right.set_ylabel("coherence", fontsize=5.6)
    right.tick_params(labelsize=5.2)
    ax.set_title("same event direction; radius-only axis", fontsize=7.2, weight="bold")


def _render_event_sheet(
    selected: pd.DataFrame, values: pd.DataFrame, out_dir: Path
) -> None:
    fig, axes = plt.subplots(
        len(EVENT_CLASSES),
        10,
        figsize=(25.0, 6.3),
        gridspec_kw={"width_ratios": [1.05, 1, 1, 1, 1.25] * 2},
    )
    for row_index, (event_class, _lo, _hi, _target) in enumerate(EVENT_CLASSES):
        for subject_index, subject in enumerate(SUBJECTS):
            col0 = subject_index * 5
            selection = selected[
                selected["event_class"].eq(event_class)
                & selected["subject"].eq(subject)
            ].iloc[0]
            subset = values[
                values["event_class"].eq(event_class)
                & values["subject"].eq(subject)
            ]
            _plot_event_path(axes[row_index, col0], selection)
            for radius_index, radius in enumerate(RADII_DEG):
                radius_row = subset[np.isclose(subset["patch_radius_deg"], radius)].iloc[0]
                _plot_event_patch(
                    axes[row_index, col0 + 1 + radius_index], selection, radius_row, radius
                )
            _plot_event_curve(axes[row_index, col0 + 4], subset)
            if subject_index == 0:
                axes[row_index, col0].set_ylabel(
                    "Small detected event" if event_class == "small_event" else "Large detected event",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                    fontsize=8.2,
                    weight="bold",
                )
    for subject_index, subject in enumerate(SUBJECTS):
        left = axes[0, subject_index * 5].get_position().x0
        right = axes[0, subject_index * 5 + 4].get_position().x1
        session = str(selected[selected["subject"].eq(subject)]["session"].iloc[0])
        fig.text(
            0.5 * (left + right),
            0.922,
            f"{subject.upper()} — {session}",
            ha="center",
            fontsize=10.2,
            weight="bold",
            color=SUBJECT_COLORS[subject],
        )
    fig.suptitle(
        "Checkpoint 4B: detected events are a separate movement regime\n"
        "Event amplitude is held fixed within each row while only the image-analysis radius changes",
        y=0.995,
        fontsize=12.3,
        weight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.994, top=0.86, bottom=0.10, hspace=0.60, wspace=0.48)
    boundary = 0.5 * (axes[0, 4].get_position().x1 + axes[0, 5].get_position().x0)
    fig.add_artist(plt.Line2D([boundary, boundary], [0.07, 0.93], transform=fig.transFigure, color="#a6adb5", lw=1.0))
    fig.text(
        0.99,
        0.025,
        "Event patches are centered at the event-mean gaze only for this scale diagnostic; temporal patch-center directionality is reserved for Checkpoint 5.",
        ha="right",
        fontsize=6.8,
        color="#4a4f55",
    )
    fig.savefig(out_dir / "checkpoint4_detected_event_examples.png", dpi=220)
    fig.savefig(out_dir / "checkpoint4_detected_event_examples.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[checkpoint4] loading complete-radius drift candidates", flush=True)
    windows, sweep = _prepare_complete_windows()
    selected, session_support = _select_drift_examples(windows)
    print(
        "[checkpoint4] selected sessions: "
        + ", ".join(
            f"{subject}={selected[selected['subject'].eq(subject)]['session'].iloc[0]}"
            for subject in SUBJECTS
        ),
        flush=True,
    )
    drift_values, traces = _window_radius_values(selected, sweep)

    sessions = {
        subject: str(selected[selected["subject"].eq(subject)]["session"].iloc[0])
        for subject in SUBJECTS
    }
    print("[checkpoint4] selecting separate detected events", flush=True)
    selected_events, event_values = _select_events(sessions)

    selected.to_csv(out_dir / "checkpoint4_scale_eccentricity_selected_windows.csv", index=False)
    session_support.to_csv(out_dir / "checkpoint4_scale_eccentricity_session_support.csv", index=False)
    drift_values.to_csv(out_dir / "checkpoint4_scale_eccentricity_example_values.csv", index=False)
    selected_events.to_csv(out_dir / "checkpoint4_selected_events.csv", index=False)
    event_values.to_csv(out_dir / "checkpoint4_detected_event_values.csv", index=False)
    _render_drift_sheet(selected, drift_values, traces, out_dir)
    _render_event_sheet(selected_events, event_values, out_dir)

    metadata = {
        "artifact_type": "map_first_targeted_visualization_checkpoint",
        "checkpoint": "4A drift factorial plus 4B separate detected events",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_inference_performed": False,
        "window_features": WINDOW_FEATURES,
        "patch_radius_sweep": RADIUS_SWEEP,
        "event_features": EVENT_FEATURES,
        "subjects": SUBJECTS,
        "phase": PHASE,
        "radii_deg": RADII_DEG,
        "factorial_cells": CELL_ORDER,
        "selection_contract": (
            "one session per subject chosen by maximum minimum support over session-specific "
            "RMS/eccentricity quartile cells; one mid-fixation window per cell chosen using "
            "movement, eccentricity, anisotropy, and time-since-event only; no image alignment "
            "or coherence term"
        ),
        "behavior_rotation_contract": (
            "256 deterministic midpoint axial rotations in [0,180 deg), common across radii "
            "within each selected drift window"
        ),
        "event_contract": (
            "detected BackImage events selected separately at 0.2-1.0 and 4.0-6.0 deg; "
            "duration<=0.12 s; patches centered at event-mean gaze for this diagnostic"
        ),
        "git_revision": cp3._git_value("rev-parse", "HEAD"),
        "git_dirty": bool(cp3._git_value("status", "--porcelain")),
        "outputs": [
            "checkpoint4_scale_eccentricity_examples.png",
            "checkpoint4_scale_eccentricity_examples.pdf",
            "checkpoint4_scale_eccentricity_example_values.csv",
            "checkpoint4_scale_eccentricity_selected_windows.csv",
            "checkpoint4_scale_eccentricity_session_support.csv",
            "checkpoint4_detected_event_examples.png",
            "checkpoint4_detected_event_examples.pdf",
            "checkpoint4_detected_event_values.csv",
            "checkpoint4_selected_events.csv",
            "checkpoint4_run_metadata.json",
        ],
    }
    cp3._write_json(out_dir / "checkpoint4_run_metadata.json", metadata)
    print(f"[checkpoint4] wrote artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
