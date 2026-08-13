#!/usr/bin/env python3
"""Build Checkpoint 1: concrete examples for the absolute-reference-frame confound.

This deliberately stops at an example sheet and auditable selection/value tables.
It does not estimate a population effect or run a pairing null.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.image_features import (
    _backimage_canvas,
    gaze_deg_to_screen_px,
)
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _window_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
)
DT_S = 1.0 / 120.0
LAGS = (1, 3, 6, 12, 30)
ROLE_ORDER = (
    "shared-horizontal-positive",
    "oblique-local-positive",
    "motor-prior-dissociation",
    "image-dominant-dissociation",
    "low-coherence-control",
    "low-anisotropy-control",
)
TARGET_SUBJECT = {
    "shared-horizontal-positive": "Logan",
    "oblique-local-positive": "Allen",
    "motor-prior-dissociation": "Logan",
    "image-dominant-dissociation": "Allen",
    "low-coherence-control": "Allen",
    "low-anisotropy-control": "Logan",
}
ROLE_LABEL = {
    "shared-horizontal-positive": "Shared horizontal\npositive",
    "oblique-local-positive": "Oblique local\npositive",
    "motor-prior-dissociation": "Motor-prior\ndissociation",
    "image-dominant-dissociation": "Image-dominant\ndissociation",
    "low-coherence-control": "Low-coherence\ncontrol",
    "low-anisotropy-control": "Low-anisotropy\ncontrol",
}


def axial_signed_deg(angle_deg: Any) -> Any:
    """Signed axial angle in [-90, 90)."""
    arr = (np.asarray(angle_deg, dtype=float) + 90.0) % 180.0 - 90.0
    return float(arr) if arr.ndim == 0 else arr


def axial_distance_deg(a_deg: Any, b_deg: Any) -> Any:
    return np.abs(axial_signed_deg(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float)))


def axial_mean_deg(values_deg: pd.Series) -> float:
    theta = np.radians(2.0 * pd.to_numeric(values_deg, errors="coerce").dropna().to_numpy())
    return float(axial_signed_deg(0.5 * np.degrees(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))))


def subject_from_session(session: pd.Series) -> pd.Series:
    return session.astype(str).str.split("_", n=1).str[0]


def prepare_candidates(source: pd.DataFrame) -> pd.DataFrame:
    df = source.copy()
    df["subject"] = subject_from_session(df["session"])
    priors = df.groupby("session", sort=False)["drift_orientation_deg"].apply(axial_mean_deg)
    df["session_motor_prior_axis_deg"] = df["session"].map(priors)
    df["edge_horizontal_delta_deg"] = axial_distance_deg(df["image_edge_axis_deg"], 0.0)
    df["cloud_horizontal_delta_deg"] = axial_distance_deg(df["drift_orientation_deg"], 0.0)
    df["cloud_edge_delta_deg"] = axial_distance_deg(df["drift_orientation_deg"], df["image_edge_axis_deg"])
    df["cloud_prior_delta_deg"] = axial_distance_deg(df["drift_orientation_deg"], df["session_motor_prior_axis_deg"])
    df["edge_prior_delta_deg"] = axial_distance_deg(df["image_edge_axis_deg"], df["session_motor_prior_axis_deg"])
    return df


def role_mask(df: pd.DataFrame, role: str) -> pd.Series:
    coherent = df["image_orientation_coherence"] >= 0.50
    anisotropic = df["anisotropy"] >= 0.50
    base = (
        df["image_feature_ok"].fillna(False).astype(bool)
        & df["rms_radius_deg"].between(0.025, 0.25)
        & df["abs_mean_radius_deg"].between(1.0, 8.0)
    )
    if role == "shared-horizontal-positive":
        return base & coherent & anisotropic & (df["edge_horizontal_delta_deg"] <= 10) & (df["cloud_horizontal_delta_deg"] <= 10) & (df["cloud_edge_delta_deg"] <= 10)
    if role == "oblique-local-positive":
        return base & coherent & anisotropic & (df["edge_horizontal_delta_deg"] >= 30) & (df["cloud_edge_delta_deg"] <= 10)
    if role == "motor-prior-dissociation":
        return base & coherent & anisotropic & (df["edge_prior_delta_deg"] >= 30) & (df["cloud_prior_delta_deg"] <= 10) & (df["cloud_edge_delta_deg"] >= 30)
    if role == "image-dominant-dissociation":
        return base & coherent & anisotropic & (df["edge_prior_delta_deg"] >= 30) & (df["cloud_prior_delta_deg"] >= 30) & (df["cloud_edge_delta_deg"] <= 10)
    if role == "low-coherence-control":
        return base & (df["image_orientation_coherence"] <= 0.10) & anisotropic
    if role == "low-anisotropy-control":
        return base & coherent & (df["anisotropy"] <= 0.10)
    raise ValueError(f"Unknown role: {role}")


def selection_score(df: pd.DataFrame, role: str) -> pd.Series:
    nuisance = (
        np.abs(np.log(df["rms_radius_deg"] / 0.10))
        + 0.25 * np.abs(df["abs_mean_radius_deg"] - 4.0) / 4.0
        + 0.10 * np.abs(df["samples_since_event"] - 49.0) / 49.0
    )
    if role in {"shared-horizontal-positive", "oblique-local-positive", "image-dominant-dissociation"}:
        return nuisance + df["cloud_edge_delta_deg"] / 20.0 - (df["image_orientation_coherence"] - 0.5) - 0.25 * (df["anisotropy"] - 0.5)
    if role == "motor-prior-dissociation":
        return nuisance + df["cloud_prior_delta_deg"] / 20.0 - (df["image_orientation_coherence"] - 0.5) - 0.25 * (df["anisotropy"] - 0.5)
    if role == "low-coherence-control":
        return nuisance + df["image_orientation_coherence"] - 0.10 * (df["anisotropy"] - 0.5)
    if role == "low-anisotropy-control":
        return nuisance + df["anisotropy"] - 0.10 * (df["image_orientation_coherence"] - 0.5)
    raise ValueError(f"Unknown role: {role}")


def overlaps_selected(row: pd.Series, selected: list[pd.Series]) -> bool:
    for prev in selected:
        if row["session"] != prev["session"] or int(row["trial_idx"]) != int(prev["trial_idx"]):
            continue
        if int(row["global_start"]) < int(prev["global_stop"]) and int(prev["global_start"]) < int(row["global_stop"]):
            return True
    return False


def role_rule_text(role: str) -> str:
    common = "image_ok; 0.025<=RMS<=0.25 deg; 1<=gaze_ecc<=8 deg"
    rules = {
        "shared-horizontal-positive": "coh>=0.5; aniso>=0.5; edge-horizontal<=10; cloud-horizontal<=10; cloud-edge<=10 deg",
        "oblique-local-positive": "coh>=0.5; aniso>=0.5; edge-horizontal>=30; cloud-edge<=10 deg",
        "motor-prior-dissociation": "coh>=0.5; aniso>=0.5; edge-prior>=30; cloud-prior<=10; cloud-edge>=30 deg",
        "image-dominant-dissociation": "coh>=0.5; aniso>=0.5; edge-prior>=30; cloud-prior>=30; cloud-edge<=10 deg",
        "low-coherence-control": "coh<=0.1; aniso>=0.5",
        "low-anisotropy-control": "coh>=0.5; aniso<=0.1",
    }
    return f"{common}; {rules[role]}"


def select_examples(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    selected: list[pd.Series] = []
    candidate_counts: dict[str, int] = {}
    for role in ROLE_ORDER:
        target = TARGET_SUBJECT[role]
        candidates = df[role_mask(df, role) & (df["subject"] == target)].copy()
        candidate_counts[role] = int(len(candidates))
        candidates["criterion_score"] = selection_score(candidates, role)
        candidates = candidates.sort_values(
            ["criterion_score", "session", "trial_idx", "global_start"], kind="stable"
        )
        candidates = candidates[~candidates.apply(lambda row: overlaps_selected(row, selected), axis=1)]
        if candidates.empty:
            raise RuntimeError(f"No non-overlapping {target} candidate for {role}")
        chosen = candidates.iloc[0].copy()
        chosen["example_role"] = role
        chosen["selection_rule"] = role_rule_text(role)
        chosen["target_subject"] = target
        chosen["eligible_candidates_target_subject"] = candidate_counts[role]
        selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True), candidate_counts


def axis_vector(angle_deg: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    return np.array([math.cos(theta), math.sin(theta)])


def displacement_features(trace: np.ndarray, edge_axis_deg: float, lag: int) -> dict[str, float]:
    disp = trace[lag:] - trace[:-lag]
    angle = np.degrees(np.arctan2(disp[:, 1], disp[:, 0]))
    z = np.exp(2j * np.radians(angle))
    mean_z = np.mean(z)
    preferred = axial_signed_deg(0.5 * np.degrees(np.angle(mean_z)))
    relative = axial_signed_deg(preferred - edge_axis_deg)
    relative_samples = axial_signed_deg(angle - edge_axis_deg)
    return {
        f"lag{lag}_ms": 1000.0 * lag * DT_S,
        f"lag{lag}_mean_displacement_deg": float(np.mean(np.linalg.norm(disp, axis=1))),
        f"lag{lag}_axial_resultant_r": float(np.abs(mean_z)),
        f"lag{lag}_preferred_axis_deg": float(preferred),
        f"lag{lag}_preferred_relative_to_edge_deg": float(relative),
        f"lag{lag}_mean_cos2_relative_to_edge": float(np.mean(np.cos(2.0 * np.radians(relative_samples)))),
    }


def crop_patch(row: pd.Series) -> tuple[np.ndarray, float]:
    canvas, ppd, shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center = gaze_deg_to_screen_px(
        np.array([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=ppd,
        screen_shape=shape,
    )
    radius = int(round(float(row["image_patch_radius_px"])))
    cx, cy = np.round(center).astype(int)
    patch = np.asarray(canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1])
    if patch.size == 0:
        raise RuntimeError(f"Empty image patch for {row['session']} trial {row['trial_idx']}")
    return patch, ppd


def plot_axis(ax: plt.Axes, angle_deg: float, length: float, *, color: str, lw: float, label: str | None = None, zorder: int = 5) -> None:
    u = axis_vector(angle_deg) * length
    ax.plot([-u[0], u[0]], [-u[1], u[1]], color=color, lw=lw, label=label, zorder=zorder)


def add_covariance_ellipse(ax: plt.Axes, row: pd.Series) -> None:
    cov = np.array(
        [[float(row["cov_xx_deg2"]), float(row["cov_xy_deg2"])], [float(row["cov_xy_deg2"]), float(row["cov_yy_deg2"])]],
        dtype=float,
    )
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ellipse = Ellipse(
        (0, 0),
        width=4.0 * np.sqrt(max(values[0], 0.0)),
        height=4.0 * np.sqrt(max(values[1], 0.0)),
        angle=angle,
        facecolor="none",
        edgecolor="#245c8a",
        lw=1.2,
        zorder=4,
    )
    ax.add_patch(ellipse)


def render_sheet(selected: pd.DataFrame, traces: list[np.ndarray], patches: list[np.ndarray], out_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(selected), 5, figsize=(15.2, 16.2),
        gridspec_kw={"width_ratios": [1.0, 1.12, 1.2, 1.12, 1.42]},
    )
    obs_bg, derived_bg = "#fbfaf6", "#f5f8fb"
    for ax in axes[:, :2].flat:
        ax.set_facecolor(obs_bg)
    for ax in axes[:, 2:].flat:
        ax.set_facecolor(derived_bg)
    fig.text(0.29, 0.951, "OBSERVED", ha="center", va="top", fontsize=11, weight="bold", color="#333333")
    fig.text(0.705, 0.951, "DERIVED / RE-EXPRESSED", ha="center", va="top", fontsize=11, weight="bold", color="#333333")

    lag_colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(LAGS)))
    for i, (idx, row) in enumerate(selected.iterrows()):
        trace = traces[idx]
        centered = trace - np.mean(trace, axis=0)
        edge = float(row["image_edge_axis_deg"])
        cloud = float(row["drift_orientation_deg"])
        prior = float(row["session_motor_prior_axis_deg"])

        ax = axes[i, 0]
        patch = patches[idx]
        ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
        cy, cx = (np.asarray(patch.shape) - 1.0) / 2.0
        length = 0.42 * min(patch.shape)
        theta_array = math.radians(float(row["image_edge_axis_array_deg"]))
        dx, dy = length * math.cos(theta_array), length * math.sin(theta_array)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="#1b7f5c", lw=2.0)
        ax.scatter([cx], [cy], s=18, c="#cc3d3d", marker="+", linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(
            f"{i + 1}. {ROLE_LABEL[row['example_role']]}\n{row['subject']} {row['session'].split('_', 1)[1]} | tr {int(row['trial_idx'])}",
            fontsize=9.1, weight="bold", rotation=0, ha="right", va="center", labelpad=13,
        )
        if i == 0:
            ax.set_title("Gaze-centered image\npatch + Sobel edge axis", fontsize=10)

        ax = axes[i, 1]
        ax.plot(trace[:, 0], trace[:, 1], color="#30343b", lw=0.8, alpha=0.9)
        ax.scatter(trace[0, 0], trace[0, 1], s=22, c="#2a72b5", label="start", zorder=4)
        ax.scatter(trace[-1, 0], trace[-1, 1], s=24, c="#c54a3d", marker="s", label="end", zorder=4)
        ax.scatter([np.mean(trace[:, 0])], [np.mean(trace[:, 1])], s=18, c="black", marker="+")
        span = max(0.18, float(np.max(np.abs(centered))) * 1.12)
        mx, my = np.mean(trace, axis=0)
        ax.set_xlim(mx - span, mx + span); ax.set_ylim(my - span, my + span)
        ax.set_aspect("equal"); ax.grid(alpha=0.17, lw=0.5)
        ax.set_xlabel("screen x (deg)", fontsize=8); ax.set_ylabel("screen y (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.98, f"mean=({mx:.2f}, {my:.2f})°", transform=ax.transAxes, va="top", fontsize=7.5)
        if i == 0:
            ax.set_title("Raw screen-space eye path\n(start blue, end red)", fontsize=10)

        ax = axes[i, 2]
        ax.plot(centered[:, 0], centered[:, 1], color="#767c84", lw=0.7, alpha=0.85)
        ax.scatter(centered[:, 0], centered[:, 1], s=4, c=np.arange(len(centered)), cmap="Greys", alpha=0.55)
        add_covariance_ellipse(ax, row)
        plot_axis(ax, cloud, 0.19, color="#245c8a", lw=2.3, label="cloud")
        plot_axis(ax, edge, 0.17, color="#1b7f5c", lw=2.0, label="edge")
        plot_axis(ax, prior, 0.15, color="#d08022", lw=1.8, label="session prior")
        limit = max(0.22, float(np.max(np.abs(centered))) * 1.12)
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_aspect("equal")
        ax.axhline(0, color="#b6bbc1", lw=0.5); ax.axvline(0, color="#b6bbc1", lw=0.5)
        ax.set_xlabel("centered x (deg)", fontsize=8); ax.set_ylabel("centered y (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(
            0.02, 0.98,
            f"edge {edge:+.1f}° | cloud {cloud:+.1f}°\nprior {prior:+.1f}° | Δ={float(row['cloud_edge_delta_deg']):.1f}°",
            transform=ax.transAxes, va="top", fontsize=7.2,
        )
        if i == 0:
            ax.set_title("Centered cloud + 2σ ellipse\nabsolute screen/head axes", fontsize=10)
            ax.legend(loc="lower right", fontsize=6.8, frameon=False)

        ax = axes[i, 3]
        along = axis_vector(edge)
        across = np.array([-along[1], along[0]])
        contour = np.column_stack([centered @ along, centered @ across])
        ax.plot(contour[:, 0], contour[:, 1], color="#30343b", lw=0.8)
        ax.scatter(contour[0, 0], contour[0, 1], s=20, c="#2a72b5", zorder=4)
        ax.scatter(contour[-1, 0], contour[-1, 1], s=22, c="#c54a3d", marker="s", zorder=4)
        ax.axhline(0, color="#1b7f5c", lw=0.8, alpha=0.7); ax.axvline(0, color="#b6bbc1", lw=0.5)
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_aspect("equal")
        ax.set_xlabel("along edge (deg)", fontsize=8); ax.set_ylabel("across edge (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title("Same path in local\ncontour coordinates", fontsize=10)

        ax = axes[i, 4]
        bins = np.linspace(-90, 90, 25)
        centers = 0.5 * (bins[:-1] + bins[1:])
        for lag, color in zip(LAGS, lag_colors, strict=True):
            disp = trace[lag:] - trace[:-lag]
            angle = np.degrees(np.arctan2(disp[:, 1], disp[:, 0]))
            relative = axial_signed_deg(angle - edge)
            density, _ = np.histogram(relative, bins=bins, density=True)
            ax.plot(centers, density, color=color, lw=1.3, label=f"{lag * 1000 * DT_S:.0f} ms")
        ax.axvline(0, color="#1b7f5c", lw=1.0, ls="--")
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xlabel("displacement axis − edge axis (deg)", fontsize=8)
        ax.set_ylabel("density", fontsize=8); ax.tick_params(labelsize=7)
        ax.text(
            0.98, 0.96,
            f"coh {float(row['image_orientation_coherence']):.2f}\naniso {float(row['anisotropy']):.2f}\nRMS {float(row['rms_radius_deg']):.3f}°\necc {float(row['abs_mean_radius_deg']):.2f}°\nafter event {float(row['samples_since_event']) * DT_S * 1000:.0f} ms\nthr {float(row['event_threshold_deg_s']):.1f}°/s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
        )
        if i == 0:
            ax.set_title("Displacement-axis distributions\nrelative to local edge", fontsize=10)
            ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2)

    fig.suptitle(
        "Checkpoint 1 — Is apparent FEM–contour alignment an absolute-reference-frame coincidence?\n"
        "Six rule-selected windows; no population inference",
        y=0.998, fontsize=13.5, weight="bold",
    )
    fig.subplots_adjust(left=0.155, right=0.988, top=0.918, bottom=0.04, hspace=0.42, wspace=0.34)
    divider_x = 0.5 * (axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0)
    fig.add_artist(plt.Line2D([divider_x, divider_x], [0.035, 0.951], transform=fig.transFigure, color="#a6adb5", lw=1.0))
    fig.savefig(out_dir / "checkpoint1_reference_frame_examples.png", dpi=210)
    fig.savefig(out_dir / "checkpoint1_reference_frame_examples.pdf")
    plt.close(fig)


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_path)
    candidates = prepare_candidates(source)
    selected, candidate_counts = select_examples(candidates)

    traces: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    value_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        trace = np.asarray(_window_trace(row), dtype=float)
        if trace.shape != (int(row["n_samples"]), 2) or not np.isfinite(trace).all():
            raise RuntimeError(f"Unexpected trace for {row['session']} trial {row['trial_idx']}: {trace.shape}")
        trace_mean = np.mean(trace, axis=0)
        trace_centered = trace - trace_mean
        trace_rms = float(np.sqrt(np.mean(np.sum(trace_centered * trace_centered, axis=1))))
        trace_mean_error = float(
            np.linalg.norm(trace_mean - np.array([float(row["mean_x_deg"]), float(row["mean_y_deg"])]))
        )
        if trace_mean_error > 1e-6 or abs(trace_rms - float(row["rms_radius_deg"])) > 1e-6:
            raise RuntimeError(
                f"Native trace does not reproduce reviewed window metrics for {row['session']} "
                f"trial {row['trial_idx']}: mean error={trace_mean_error:g}, RMS={trace_rms:g}"
            )
        patch, ppd = crop_patch(row)
        traces.append(trace)
        patches.append(patch)
        values: dict[str, Any] = {
            "example_role": row["example_role"],
            "session": row["session"],
            "subject": row["subject"],
            "trial_idx": int(row["trial_idx"]),
            "global_start": int(row["global_start"]),
            "global_stop": int(row["global_stop"]),
            "n_trace_samples": int(len(trace)),
            "trace_duration_s": float(len(trace) * DT_S),
            "trace_mean_reproduction_error_deg": trace_mean_error,
            "trace_rms_radius_recomputed_deg": trace_rms,
            "ppd": float(ppd),
            "image_edge_axis_deg": float(row["image_edge_axis_deg"]),
            "drift_cloud_axis_deg": float(row["drift_orientation_deg"]),
            "session_motor_prior_axis_deg": float(row["session_motor_prior_axis_deg"]),
            "cloud_edge_delta_deg": float(row["cloud_edge_delta_deg"]),
            "cloud_prior_delta_deg": float(row["cloud_prior_delta_deg"]),
            "edge_prior_delta_deg": float(row["edge_prior_delta_deg"]),
            "drift_edge_cos2": float(row["drift_edge_cos2"]),
            "image_orientation_coherence": float(row["image_orientation_coherence"]),
            "cloud_anisotropy": float(row["anisotropy"]),
            "rms_radius_deg": float(row["rms_radius_deg"]),
            "gaze_eccentricity_deg": float(row["abs_mean_radius_deg"]),
            "phase": row["phase"],
            "time_since_event_ms": float(row["samples_since_event"] * DT_S * 1000.0),
            "event_threshold_deg_s": float(row["event_threshold_deg_s"]),
            "image_patch_radius_px": int(round(float(row["image_patch_radius_px"]))),
            "image_patch_radius_deg": float(row["image_patch_radius_px"] / ppd),
        }
        for lag in LAGS:
            values.update(displacement_features(trace, float(row["image_edge_axis_deg"]), lag))
        value_rows.append(values)

    selection_columns = [
        "example_role", "selection_rule", "criterion_score", "target_subject",
        "eligible_candidates_target_subject", "session", "subject", "trial_idx",
        "global_start", "global_stop", "local_start", "local_stop", "n_samples",
        "phase", "samples_since_event", "event_threshold_deg_s", "mean_x_deg",
        "mean_y_deg", "abs_mean_radius_deg", "rms_radius_deg", "anisotropy",
        "image_orientation_coherence", "image_edge_axis_deg", "drift_orientation_deg",
        "session_motor_prior_axis_deg", "edge_horizontal_delta_deg",
        "cloud_horizontal_delta_deg", "cloud_edge_delta_deg", "cloud_prior_delta_deg",
        "edge_prior_delta_deg", "drift_edge_cos2", "image_patch_center_x_px",
        "image_patch_center_y_px", "image_patch_radius_px",
    ]
    selected[selection_columns].to_csv(out_dir / "checkpoint1_selected_windows.csv", index=False)
    pd.DataFrame(value_rows).to_csv(out_dir / "checkpoint1_reference_frame_example_values.csv", index=False)
    render_sheet(selected, traces, patches, out_dir)

    source_metadata_path = input_path.parent / "run_metadata.json"
    source_metadata = None
    if source_metadata_path.exists():
        with source_metadata_path.open() as handle:
            source_metadata = json.load(handle)
    metadata = {
        "artifact_type": "map_first_targeted_visualization_checkpoint",
        "checkpoint": 1,
        "question": "Could local FEM-contour alignment be an absolute screen/head-frame coincidence?",
        "population_inference_performed": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input_windows": str(input_path),
        "source_run_metadata": str(source_metadata_path) if source_metadata_path.exists() else None,
        "source_run_config": source_metadata,
        "n_source_windows": int(len(source)),
        "n_selected_windows": int(len(selected)),
        "role_order": list(ROLE_ORDER),
        "target_subject_by_role": TARGET_SUBJECT,
        "eligible_candidate_counts_target_subject": candidate_counts,
        "selection_overlap_policy": "exclude intervals overlapping an earlier selected interval within the same session and trial",
        "selection_score": "nuisance distance to RMS=.10deg, eccentricity=4deg, time-since-event=49 samples, plus role-specific contrast quality; minimum wins",
        "dt_s": DT_S,
        "displacement_lags_samples": list(LAGS),
        "displacement_lags_ms": [lag * DT_S * 1000.0 for lag in LAGS],
        "trace_provenance": "native reviewed BackImage window loaded from session eyepos using trial-local start/stop, with global-index fallback",
        "coordinate_contract": {
            "gaze": "+x screen right, +y screen up, axial angles in degrees",
            "image_array": "+column right, +row down; plotted edge uses image_edge_axis_array_deg",
            "contour_relative": "+x along local Sobel edge axis, +y orthogonal to it",
        },
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "outputs": [
            "checkpoint1_reference_frame_examples.png",
            "checkpoint1_reference_frame_examples.pdf",
            "checkpoint1_reference_frame_example_values.csv",
            "checkpoint1_selected_windows.csv",
            "checkpoint1_run_metadata.json",
        ],
    }
    with (out_dir / "checkpoint1_run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(selected[["example_role", "session", "trial_idx", "global_start", "criterion_score"]].to_string(index=False))
    print(f"Wrote Checkpoint 1 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
