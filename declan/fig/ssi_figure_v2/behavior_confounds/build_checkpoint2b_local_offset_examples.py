#!/usr/bin/env python3
"""Build Figure 4 Panel-G Checkpoint 2B: local versus 5-deg offsets.

This is a targeted map-first diagnostic for the six windows selected at
Checkpoint 1.  It does not run the neural model.  The Panel-G values shown in
the final column are explicitly the existing one-dimensional RMS-dose
interpolation, evaluated with identical rotation draws at the local and offset
patches.  The purpose is to inspect the image-side locality control before an
expensive direct counterfactual model render.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint1_reference_frame_examples as cp1,
)
from declan.fig.ssi_figure_v2.behavior_confounds import (
    build_checkpoint2_pairing_null_examples as cp2,
)
from declan.fixation_statistics_by_stimulus.image_features import (
    _backimage_canvas,
    gaze_deg_to_screen_px,
    local_backimage_features,
)
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _window_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
SOURCE_WINDOWS = cp1.DEFAULT_INPUT
SELECTED_WINDOWS = OUT_DIR / "checkpoint1_selected_windows.csv"
EXTENDED_MODEL_VALUES = cp2.EXTENDED_MODEL_VALUES
FALLBACK_MODEL_VALUES = cp2.FALLBACK_MODEL_VALUES

OFFSET_DISTANCE_DEG = 5.0
OFFSET_ANGLES_DEG = tuple(float(v) for v in range(0, 360, 45))
PATCH_RADIUS_DEG = 1.0
MIN_PATCH_FRACTION_INSIDE = 0.98
MAX_PATCH_FRACTION_BACKGROUND = 0.05
N_ROTATIONS = 256
SEED = 2310
MODEL_POPULATION = "high_sf_aligned"
MODEL_METRIC = "component_rms"

KEYS = ("session", "trial_idx", "global_start", "global_stop")
PROXY_COLUMNS = (
    "ssi_percent_vs_cell_baseline",
    "moving_population_ssi_bits_per_spike",
    "moving_information_bits_per_sample",
    "moving_expected_spikes_per_sample",
)

LOCAL_COLOR = "#245c8a"
PRESERVE_COLOR = "#1b7f5c"
CHANGE_COLOR = "#b4492d"
OFFSET_COLOR = "#9aa2aa"
ROTATION_COLOR = "#7a3b9a"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _find_target(source: pd.DataFrame, selected: pd.Series) -> pd.Series:
    mask = np.ones(len(source), dtype=bool)
    for key in KEYS:
        mask &= source[key].astype(str).eq(str(selected[key])).to_numpy()
    matches = source[mask]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source row for {selected['example_role']}, got {len(matches)}")
    out = matches.iloc[0].copy()
    out["example_role"] = str(selected["example_role"])
    out["subject"] = str(out["session"]).split("_", 1)[0]
    return out


def _offset_gaze(local_gaze: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    return np.asarray(local_gaze, dtype=np.float64) + OFFSET_DISTANCE_DEG * np.asarray(
        [math.cos(theta), math.sin(theta)], dtype=np.float64
    )


def _location_valid(features: dict[str, Any]) -> bool:
    return bool(features.get("image_feature_ok", False)) and (
        float(features.get("image_patch_fraction_inside_image", np.nan))
        >= MIN_PATCH_FRACTION_INSIDE
    ) and (
        float(features.get("image_patch_fraction_background", np.nan))
        <= MAX_PATCH_FRACTION_BACKGROUND
    )


def _location_rows(target: pd.Series) -> list[dict[str, Any]]:
    local_gaze = np.asarray([target["mean_x_deg"], target["mean_y_deg"]], dtype=np.float64)
    specs: list[tuple[str, float | None, np.ndarray]] = [("local", None, local_gaze)]
    specs.extend(
        ("offset", angle, _offset_gaze(local_gaze, angle)) for angle in OFFSET_ANGLES_DEG
    )
    rows: list[dict[str, Any]] = []
    local_axis = float(target["image_edge_axis_deg"])
    for location_index, (kind, angle, gaze) in enumerate(specs):
        features = local_backimage_features(
            session_name=str(target["session"]),
            trial_idx=int(target["trial_idx"]),
            gaze_xy_deg=gaze,
            patch_radius_deg=PATCH_RADIUS_DEG,
        )
        axis = float(features.get("image_edge_axis_deg", np.nan))
        delta = float(cp1.axial_distance_deg(axis, local_axis)) if math.isfinite(axis) else float("nan")
        if kind == "local":
            relationship = "local"
        elif math.isfinite(delta) and delta <= 10.0:
            relationship = "orientation_preserving"
        elif math.isfinite(delta) and delta >= 30.0:
            relationship = "orientation_changing"
        else:
            relationship = "intermediate"
        rows.append(
            {
                "example_role": str(target["example_role"]),
                "session": str(target["session"]),
                "subject": str(target["subject"]),
                "trial_idx": int(target["trial_idx"]),
                "global_start": int(target["global_start"]),
                "global_stop": int(target["global_stop"]),
                "location_index": int(location_index),
                "location_id": "local" if kind == "local" else f"offset_{int(float(angle)):03d}",
                "location_kind": kind,
                "offset_distance_deg": 0.0 if kind == "local" else OFFSET_DISTANCE_DEG,
                "offset_angle_gaze_deg": np.nan if angle is None else float(angle),
                "gaze_x_deg": float(gaze[0]),
                "gaze_y_deg": float(gaze[1]),
                "local_edge_axis_deg": local_axis,
                "local_offset_axis_delta_deg": delta,
                "axis_relationship": relationship,
                "passes_primary_patch_qc": _location_valid(features),
                **features,
            }
        )
    return rows


def _component_mean_proxy(trace: np.ndarray, edge_axis_deg: float, curves: dict[str, pd.DataFrame]) -> dict[str, Any]:
    doses = cp2.component_doses(trace, edge_axis_deg)
    component_payloads: dict[str, dict[str, Any]] = {}
    for component in ("along", "across"):
        dose = float(doses[(MODEL_METRIC, component)])
        payload = cp2.interpolate_curve_row(dose, curves[component])
        component_payloads[component] = {"dose": dose, **payload}

    common_support = all(
        not bool(component_payloads[component]["outside_model_range"])
        and math.isfinite(float(component_payloads[component]["ssi_percent_vs_cell_baseline"]))
        for component in ("along", "across")
    )
    out: dict[str, Any] = {
        "along_rms_arcmin": component_payloads["along"]["dose"],
        "across_rms_arcmin": component_payloads["across"]["dose"],
        "both_components_in_model_support": bool(common_support),
    }
    for column in PROXY_COLUMNS:
        values = np.asarray(
            [component_payloads[component].get(column, np.nan) for component in ("along", "across")],
            dtype=float,
        )
        out[f"component_mean_{column}"] = float(np.mean(values)) if common_support and np.isfinite(values).all() else float("nan")
    return out


def _score_locations(
    target: pd.Series,
    locations: pd.DataFrame,
    trace: np.ndarray,
    rotation_angles: np.ndarray,
    curves: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cloud_axis = float(target["drift_orientation_deg"])
    rotated_traces = [cp2.rotate_trace(trace, float(angle)) for angle in rotation_angles]
    for location in locations.itertuples(index=False):
        edge_axis = float(location.image_edge_axis_deg)
        valid = bool(location.passes_primary_patch_qc) and math.isfinite(edge_axis)
        real_proxy = _component_mean_proxy(trace, edge_axis, curves) if valid else {}
        rotated_proxy = [
            _component_mean_proxy(rotated, edge_axis, curves) for rotated in rotated_traces
        ] if valid else []
        real_score = float(real_proxy.get("component_mean_ssi_percent_vs_cell_baseline", np.nan))
        rotated_scores = np.asarray(
            [item.get("component_mean_ssi_percent_vs_cell_baseline", np.nan) for item in rotated_proxy],
            dtype=float,
        )
        finite_rotated = np.isfinite(rotated_scores)
        rotated_mean = float(np.mean(rotated_scores[finite_rotated])) if np.any(finite_rotated) else float("nan")
        proxy_advantage = real_score - rotated_mean if math.isfinite(real_score) and math.isfinite(rotated_mean) else float("nan")

        real_alignment = float(np.cos(2.0 * np.radians(cloud_axis - edge_axis))) if valid else float("nan")
        rotated_alignment = np.cos(
            2.0 * np.radians(cloud_axis + np.degrees(rotation_angles) - edge_axis)
        ) if valid else np.asarray([], dtype=float)
        rotation_alignment_mean = float(np.mean(rotated_alignment)) if rotated_alignment.size else float("nan")

        out: dict[str, Any] = {
            "example_role": str(target["example_role"]),
            "location_id": str(location.location_id),
            "location_kind": str(location.location_kind),
            "offset_angle_gaze_deg": float(location.offset_angle_gaze_deg),
            "axis_relationship": str(location.axis_relationship),
            "passes_primary_patch_qc": bool(location.passes_primary_patch_qc),
            "image_edge_axis_deg": edge_axis,
            "image_orientation_coherence": float(location.image_orientation_coherence),
            "local_offset_axis_delta_deg": float(location.local_offset_axis_delta_deg),
            "behavior_cloud_edge_cos2": real_alignment,
            "behavior_rotation_mean_cos2": rotation_alignment_mean,
            "behavior_alignment_advantage": real_alignment - rotation_alignment_mean
            if math.isfinite(real_alignment) and math.isfinite(rotation_alignment_mean)
            else float("nan"),
            "proxy_real_ssi_residual_percent": real_score,
            "proxy_rotation_mean_ssi_residual_percent": rotated_mean,
            "proxy_rotation_ci95_low": float(np.percentile(rotated_scores[finite_rotated], 2.5))
            if np.any(finite_rotated)
            else float("nan"),
            "proxy_rotation_ci95_high": float(np.percentile(rotated_scores[finite_rotated], 97.5))
            if np.any(finite_rotated)
            else float("nan"),
            "proxy_real_minus_rotation_pp": proxy_advantage,
            "proxy_rotation_common_support_fraction": float(np.mean(finite_rotated))
            if rotated_scores.size
            else float("nan"),
            "n_rotation_draws": int(rotation_angles.size),
        }
        for key, value in real_proxy.items():
            out[f"proxy_real_{key}"] = value
        rows.append(out)
    values = pd.DataFrame(rows)
    local = values[values["location_kind"].astype(str).eq("local")]
    offsets = values[
        values["location_kind"].astype(str).eq("offset")
        & values["passes_primary_patch_qc"].astype(bool)
    ]
    for prefix, column in (
        ("behavior", "behavior_alignment_advantage"),
        ("proxy", "proxy_real_minus_rotation_pp"),
    ):
        local_value = float(local[column].iloc[0]) if len(local) else float("nan")
        offset_values = pd.to_numeric(offsets[column], errors="coerce").to_numpy(dtype=float)
        offset_mean = float(np.nanmean(offset_values)) if np.isfinite(offset_values).any() else float("nan")
        locality = local_value - offset_mean if math.isfinite(local_value) and math.isfinite(offset_mean) else float("nan")
        values[f"{prefix}_D_local"] = local_value
        values[f"{prefix}_D_offset_mean"] = offset_mean
        values[f"{prefix}_D_locality"] = locality
    values["n_valid_offsets"] = int(len(offsets))
    return values


def _crop_feature_patch(location: pd.Series) -> tuple[np.ndarray, float]:
    canvas, ppd, _shape = _backimage_canvas(str(location["session"]), int(location["trial_idx"]))
    cx = int(round(float(location["image_patch_center_x_px"])))
    cy = int(round(float(location["image_patch_center_y_px"])))
    radius = int(round(float(location["image_patch_radius_px"])))
    return np.asarray(canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]), float(ppd)


def _plot_patch_with_trace(
    ax: plt.Axes,
    location: pd.Series,
    trace: np.ndarray,
    *,
    color: str,
    title: str,
) -> None:
    patch, ppd = _crop_feature_patch(location)
    ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
    center = (np.asarray(patch.shape[::-1], dtype=float) - 1.0) / 2.0
    centered = np.asarray(trace, dtype=float) - np.mean(trace, axis=0, keepdims=True)
    ax.plot(center[0] + centered[:, 0] * ppd, center[1] - centered[:, 1] * ppd, color="#f2ca52", lw=0.8)
    ax.scatter(center[0] + centered[0, 0] * ppd, center[1] - centered[0, 1] * ppd, s=14, c="#36a2eb", zorder=4)
    ax.scatter(center[0] + centered[-1, 0] * ppd, center[1] - centered[-1, 1] * ppd, s=15, c="#df4c4c", marker="s", zorder=4)
    theta = math.radians(float(location["image_edge_axis_array_deg"]))
    length = 0.40 * min(patch.shape)
    dx, dy = length * math.cos(theta), length * math.sin(theta)
    ax.plot([center[0] - dx, center[0] + dx], [center[1] - dy, center[1] + dy], color=color, lw=2.0)
    ax.set_title(title, fontsize=8.2, color=color, weight="bold")
    ax.text(
        0.02,
        0.02,
        f"coh={float(location['image_orientation_coherence']):.2f}\naxis={float(location['image_edge_axis_deg']):+.1f}°",
        transform=ax.transAxes,
        va="bottom",
        fontsize=6.6,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.56, "pad": 1.5, "edgecolor": "none"},
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _representative_offsets(locations: pd.DataFrame) -> tuple[pd.Series, pd.Series, str, str]:
    valid = locations[
        locations["location_kind"].astype(str).eq("offset")
        & locations["passes_primary_patch_qc"].astype(bool)
        & np.isfinite(pd.to_numeric(locations["local_offset_axis_delta_deg"], errors="coerce"))
    ].copy()
    if valid.empty:
        raise RuntimeError(f"No valid 5-deg offsets for {locations['example_role'].iloc[0]}")
    preserving = valid[valid["local_offset_axis_delta_deg"].astype(float) <= 10.0]
    changing = valid[valid["local_offset_axis_delta_deg"].astype(float) >= 30.0]
    preserve_label = "orientation-preserving offset"
    change_label = "orientation-changing offset"
    if preserving.empty:
        preserving = valid.nsmallest(1, "local_offset_axis_delta_deg")
        preserve_label = "closest-axis offset (no ≤10° case)"
    else:
        preserving = preserving.nsmallest(1, "local_offset_axis_delta_deg")
    if changing.empty:
        changing = valid.nlargest(1, "local_offset_axis_delta_deg")
        change_label = "farthest-axis offset (no ≥30° case)"
    else:
        changing = changing.nlargest(1, "local_offset_axis_delta_deg")
    return preserving.iloc[0], changing.iloc[0], preserve_label, change_label


def _plot_overview(ax: plt.Axes, locations: pd.DataFrame) -> None:
    first = locations.iloc[0]
    canvas, _ppd, _shape = _backimage_canvas(str(first["session"]), int(first["trial_idx"]))
    centers = locations[["image_patch_center_x_px", "image_patch_center_y_px"]].to_numpy(dtype=float)
    radius = float(locations["image_patch_radius_px"].max())
    x0 = max(0, int(np.floor(np.min(centers[:, 0]) - 1.25 * radius)))
    x1 = min(canvas.shape[1], int(np.ceil(np.max(centers[:, 0]) + 1.25 * radius)))
    y0 = max(0, int(np.floor(np.min(centers[:, 1]) - 1.25 * radius)))
    y1 = min(canvas.shape[0], int(np.ceil(np.max(centers[:, 1]) + 1.25 * radius)))
    ax.imshow(canvas[y0:y1, x0:x1], cmap="gray", origin="upper", interpolation="nearest")
    for row in locations.itertuples(index=False):
        x = float(row.image_patch_center_x_px) - x0
        y = float(row.image_patch_center_y_px) - y0
        if str(row.location_kind) == "local":
            color, marker, label = LOCAL_COLOR, "*", "L"
        elif bool(row.passes_primary_patch_qc):
            color, marker, label = OFFSET_COLOR, "o", str(int(float(row.offset_angle_gaze_deg) / 45.0))
        else:
            color, marker, label = "#d0d0d0", "x", "×"
        ax.scatter([x], [y], s=32 if label == "L" else 18, c=color, marker=marker, zorder=4)
        ax.text(x + 3, y - 3, label, color=color, fontsize=6.2, weight="bold")
    ax.set_title("same image: local + 5° annulus", fontsize=8.2, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_behavior(ax: plt.Axes, values: pd.DataFrame) -> None:
    offsets = values[values["location_kind"].astype(str).eq("offset") & values["passes_primary_patch_qc"].astype(bool)]
    local = values[values["location_kind"].astype(str).eq("local")].iloc[0]
    ax.axhline(0.0, color="#8b9299", lw=0.7, ls=":")
    ax.scatter(offsets["offset_angle_gaze_deg"], offsets["behavior_alignment_advantage"], color=OFFSET_COLOR, s=22)
    ax.scatter([-35], [local["behavior_alignment_advantage"]], color=LOCAL_COLOR, marker="*", s=60)
    ax.set_xlim(-50, 365)
    ax.set_xticks([-35, 0, 90, 180, 270, 360], ["local", "0", "90", "180", "270", "360"])
    ax.set_ylim(-1.12, 1.12)
    ax.set_xlabel("offset direction (deg)", fontsize=6.8)
    ax.set_ylabel("cos2 alignment advantage", fontsize=6.8)
    ax.tick_params(labelsize=6.1)
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    ax.set_title(
        f"behavior locality={float(local['behavior_D_locality']):+.2f}", fontsize=8.2, weight="bold"
    )


def _plot_proxy(ax: plt.Axes, values: pd.DataFrame) -> None:
    offsets = values[values["location_kind"].astype(str).eq("offset") & values["passes_primary_patch_qc"].astype(bool)]
    local = values[values["location_kind"].astype(str).eq("local")].iloc[0]
    ax.axhline(0.0, color="#8b9299", lw=0.7, ls=":")
    finite = np.isfinite(pd.to_numeric(offsets["proxy_real_minus_rotation_pp"], errors="coerce"))
    ax.scatter(
        offsets.loc[finite, "offset_angle_gaze_deg"],
        offsets.loc[finite, "proxy_real_minus_rotation_pp"],
        color=OFFSET_COLOR,
        s=22,
    )
    ax.scatter([-35], [local["proxy_real_minus_rotation_pp"]], color=LOCAL_COLOR, marker="*", s=60)
    ax.set_xlim(-50, 365)
    ax.set_xticks([-35, 0, 90, 180, 270, 360], ["local", "0", "90", "180", "270", "360"])
    ax.set_xlabel("offset direction (deg)", fontsize=6.8)
    ax.set_ylabel("predicted SSI advantage (pp)", fontsize=6.8)
    ax.tick_params(labelsize=6.1)
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    locality = float(local["proxy_D_locality"])
    support = float(local["proxy_rotation_common_support_fraction"])
    ax.set_title(f"surrogate locality={locality:+.3f} pp\nlocal rotation support={support:.0%}", fontsize=8.2, weight="bold")


def _render(
    targets: list[pd.Series],
    location_tables: list[pd.DataFrame],
    value_tables: list[pd.DataFrame],
    traces: list[np.ndarray],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(targets),
        6,
        figsize=(17.2, 15.7),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0, 1.12, 1.18]},
    )
    for row_index, (target, locations, values, trace) in enumerate(
        zip(targets, location_tables, value_tables, traces, strict=True)
    ):
        local = locations[locations["location_kind"].astype(str).eq("local")].iloc[0]
        preserving, changing, preserve_label, change_label = _representative_offsets(locations)
        _plot_overview(axes[row_index, 0], locations)
        axes[row_index, 0].set_ylabel(
            f"{row_index + 1}. {cp1.ROLE_LABEL[str(target['example_role'])]}\n{target['subject']} | tr {int(target['trial_idx'])}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
            fontsize=8.0,
            weight="bold",
        )
        _plot_patch_with_trace(axes[row_index, 1], local, trace, color=LOCAL_COLOR, title="actual local patch")
        _plot_patch_with_trace(
            axes[row_index, 2], preserving, trace, color=PRESERVE_COLOR,
            title=f"{preserve_label}\nΔaxis={float(preserving['local_offset_axis_delta_deg']):.1f}°",
        )
        _plot_patch_with_trace(
            axes[row_index, 3], changing, trace, color=CHANGE_COLOR,
            title=f"{change_label}\nΔaxis={float(changing['local_offset_axis_delta_deg']):.1f}°",
        )
        _plot_behavior(axes[row_index, 4], values)
        _plot_proxy(axes[row_index, 5], values)

    fig.suptitle(
        "Figure 4 Panel G — Checkpoint 2B: does the recorded trajectory advantage localize to the true image patch?\n"
        "Each row keeps one measured trace fixed; all valid same-image 5° offsets and identical 256 axial rotations are retained",
        y=0.997,
        fontsize=12.8,
        weight="bold",
    )
    fig.text(0.35, 0.952, "OBSERVED IMAGE CONTENT + MEASURED TRACE", ha="center", fontsize=10.2, weight="bold")
    fig.text(0.765, 0.952, "BEHAVIOR", ha="center", fontsize=10.2, weight="bold")
    fig.text(0.925, 0.952, "CURRENT PANEL-G SURROGATE", ha="center", fontsize=10.2, weight="bold")
    fig.subplots_adjust(left=0.13, right=0.992, top=0.925, bottom=0.045, hspace=0.52, wspace=0.38)
    boundary = 0.5 * (axes[0, 4].get_position().x1 + axes[0, 5].get_position().x0)
    fig.add_artist(plt.Line2D([boundary, boundary], [0.035, 0.95], transform=fig.transFigure, color="#a6adb5", lw=0.9))
    fig.text(
        0.99,
        0.012,
        "Surrogate = average of two 1-D RMS-dose interpolations, not a direct model evaluation. "
        "Offsets are selected for display using image-axis criteria only; all valid offsets remain in the CSV and primary mean.",
        ha="right",
        fontsize=6.9,
        color="#4a4f55",
    )
    fig.savefig(out_dir / "checkpoint2b_local_vs_offset_patch_examples.png", dpi=220)
    fig.savefig(out_dir / "checkpoint2b_local_vs_offset_patch_examples.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-rotations", type=int, default=N_ROTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[checkpoint2b] loading tables", flush=True)
    source = pd.read_csv(SOURCE_WINDOWS)
    selected = pd.read_csv(SELECTED_WINDOWS)
    model_values_path = EXTENDED_MODEL_VALUES if EXTENDED_MODEL_VALUES.exists() else FALLBACK_MODEL_VALUES
    model_values = pd.read_csv(model_values_path)
    curves = {
        component: cp2.model_curve(model_values, MODEL_METRIC, component)
        for component in ("along", "across")
    }

    rng = np.random.default_rng(int(args.seed))
    targets: list[pd.Series] = []
    location_tables: list[pd.DataFrame] = []
    value_tables: list[pd.DataFrame] = []
    traces: list[np.ndarray] = []
    manifest_frames: list[pd.DataFrame] = []
    relationship_frames: list[pd.DataFrame] = []

    for selected_row in selected.itertuples(index=False):
        target = _find_target(source, pd.Series(selected_row._asdict()))
        print(
            f"[checkpoint2b] extracting {target['example_role']} "
            f"({target['session']} trial {int(target['trial_idx'])})",
            flush=True,
        )
        trace = np.asarray(_window_trace(target), dtype=np.float64)
        if trace.shape != (128, 2) or not np.isfinite(trace).all():
            raise RuntimeError(f"Expected one native 128x2 trace for {target['example_role']}, got {trace.shape}")
        locations = pd.DataFrame(_location_rows(target))
        angles = rng.uniform(0.0, np.pi, size=int(args.n_rotations))
        values = _score_locations(target, locations, trace, angles, curves)
        print(
            f"[checkpoint2b] scored {target['example_role']}: "
            f"valid offsets={int(values['n_valid_offsets'].iloc[0])}",
            flush=True,
        )

        manifest = locations.copy()
        manifest["rotation_seed"] = int(args.seed)
        manifest["n_rotation_draws"] = int(args.n_rotations)
        manifest["rotation_angles_hash"] = hashlib.sha256(
            np.asarray(angles, dtype=np.float64).tobytes()
        ).hexdigest()[:20]
        manifest_frames.append(manifest)
        relationship_frames.append(
            locations[
                [
                    "example_role", "location_id", "location_kind", "offset_angle_gaze_deg",
                    "passes_primary_patch_qc", "local_edge_axis_deg", "image_edge_axis_deg",
                    "local_offset_axis_delta_deg", "axis_relationship", "image_orientation_coherence",
                    "image_patch_rms_contrast", "image_gradient_energy", "image_patch_fraction_inside_image",
                    "image_patch_fraction_background", "image_patch_distance_to_image_border_px",
                ]
            ].copy()
        )
        targets.append(target)
        location_tables.append(locations)
        value_tables.append(values)
        traces.append(trace)

    print("[checkpoint2b] writing tables and rendering sheet", flush=True)
    manifest_df = pd.concat(manifest_frames, ignore_index=True)
    relationships_df = pd.concat(relationship_frames, ignore_index=True)
    values_df = pd.concat(value_tables, ignore_index=True)
    manifest_df.to_csv(out_dir / "checkpoint2b_offset_patch_manifest.csv", index=False)
    relationships_df.to_csv(out_dir / "checkpoint2b_local_offset_axis_relationship.csv", index=False)
    values_df.to_csv(out_dir / "checkpoint2b_local_vs_offset_patch_values.csv", index=False)
    _render(targets, location_tables, value_tables, traces, out_dir)

    summary = values_df[values_df["location_kind"].astype(str).eq("local")][
        [
            "example_role", "n_valid_offsets", "behavior_D_local", "behavior_D_offset_mean",
            "behavior_D_locality", "proxy_D_local", "proxy_D_offset_mean", "proxy_D_locality",
            "proxy_rotation_common_support_fraction",
        ]
    ].copy()
    summary.to_csv(out_dir / "checkpoint2b_locality_summary.csv", index=False)

    metadata = {
        "artifact_type": "map_first_targeted_visualization_checkpoint",
        "checkpoint": "2B",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_inference_performed": False,
        "focus": "Figure 4 displayed Panel G only",
        "input_windows": SOURCE_WINDOWS,
        "checkpoint1_selection": SELECTED_WINDOWS,
        "model_values": model_values_path,
        "model_population": MODEL_POPULATION,
        "patch_contract": {
            "offset_distance_deg": OFFSET_DISTANCE_DEG,
            "offset_angles_gaze_deg": OFFSET_ANGLES_DEG,
            "feature_patch_radius_deg": PATCH_RADIUS_DEG,
            "min_patch_fraction_inside_image": MIN_PATCH_FRACTION_INSIDE,
            "max_patch_fraction_background": MAX_PATCH_FRACTION_BACKGROUND,
            "primary_offset_summary": "unweighted mean over every valid pre-specified offset; orientation-preserving/changing roles are diagnostics only",
        },
        "trace_contract": "native reviewed 128-sample trace; current Panel-G proxy uses its central 40 native samples without temporal compression",
        "rotation_contract": "identical per-window uniform axial rotation draws in [0,180 deg) reused for local and all offsets",
        "surrogate_contract": "component-mean of contour-parallel and contour-normal one-dimensional RMS-dose interpolations, requiring both components inside model support; no direct activation maps or model responses",
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "outputs": [
            "checkpoint2b_local_vs_offset_patch_examples.png",
            "checkpoint2b_local_vs_offset_patch_examples.pdf",
            "checkpoint2b_offset_patch_manifest.csv",
            "checkpoint2b_local_vs_offset_patch_values.csv",
            "checkpoint2b_local_offset_axis_relationship.csv",
            "checkpoint2b_locality_summary.csv",
            "checkpoint2b_run_metadata.json",
        ],
    }
    _write_json(out_dir / "checkpoint2b_run_metadata.json", metadata)
    print(summary.to_string(index=False))
    print(f"Wrote Panel-G Checkpoint 2B artifacts to {out_dir}")


if __name__ == "__main__":
    main()
