#!/usr/bin/env python3
"""Build Checkpoint 2: concrete marginal-preserving pairing-null examples."""

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
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds import build_checkpoint1_reference_frame_examples as cp1
from declan.fig.ssi_figure_v2.behavior_model_bridge import run_behavior_model_bridge as bridge
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import _window_trace


REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
SOURCE_WINDOWS = cp1.DEFAULT_INPUT
SELECTED_WINDOWS = OUT_DIR / "checkpoint1_selected_windows.csv"
SNIPPET_METRICS = REPO_ROOT / "outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_behavior_snippet_metrics.csv"
EXTENDED_MODEL_VALUES = OUT_DIR / "checkpoint2_model_curves_extended/panel_g_alternative_x_axes_diagnostic_values.csv"
FALLBACK_MODEL_VALUES = bridge.MODEL_VALUES_CSV
N_ROTATIONS = 256
SEED = 2309
PROFILE_ANGLES_DEG = np.linspace(-90.0, 90.0, 49)
MODEL_POPULATION = "high_sf_aligned"
MODEL_METRICS = ("component_rms", "component_range")
KEYS = ["session", "trial_idx", "global_start", "global_stop", "local_start", "local_stop"]
MATCH_FEATURES = (
    "image_orientation_coherence",
    "rms_radius_deg",
    "snippet_extent_arcmin",
    "anisotropy",
    "abs_mean_radius_deg",
    "samples_since_event",
)
MATCH_FEATURE_LABELS = {
    "image_orientation_coherence": "coherence",
    "rms_radius_deg": "FEM RMS",
    "snippet_extent_arcmin": "FEM range",
    "anisotropy": "FEM anisotropy",
    "abs_mean_radius_deg": "gaze eccentricity",
    "samples_since_event": "time since event",
}
CONSTRUCTION_LABELS = {
    "real_local_pair": "Real local pair",
    "uniform_rotation": "Uniform rotation",
    "trajectory_reassigned": "Matched trajectory\nreassigned",
    "image_axis_reassigned": "Matched image axis\nreassigned",
}
CONSTRUCTION_COLORS = {
    "real_local_pair": "#245c8a",
    "uniform_rotation": "#7a3b9a",
    "trajectory_reassigned": "#d08022",
    "image_axis_reassigned": "#1b7f5c",
}


def load_analysis_table() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_WINDOWS)
    cached = pd.read_csv(SNIPPET_METRICS)
    extra = cached[
        KEYS
        + [
            "coherence_bin",
            "along_snippet_range_arcmin",
            "across_snippet_range_arcmin",
        ]
    ].copy()
    out = source.merge(extra, on=KEYS, how="left", validate="one_to_one")
    out["snippet_extent_arcmin"] = np.hypot(
        out["along_snippet_range_arcmin"], out["across_snippet_range_arcmin"]
    )
    out["subject"] = out["session"].astype(str).str.split("_", n=1).str[0]
    return out


def find_target(table: pd.DataFrame, selected_row: pd.Series) -> pd.Series:
    mask = np.ones(len(table), dtype=bool)
    for key in ("session", "trial_idx", "global_start", "global_stop"):
        mask &= table[key].astype(str).eq(str(selected_row[key])).to_numpy()
    matches = table[mask]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source row for {selected_row['example_role']}, got {len(matches)}")
    target = matches.iloc[0].copy()
    target["example_role"] = selected_row["example_role"]
    return target


def select_partner(table: pd.DataFrame, target: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    session = table[table["session"].astype(str).eq(str(target["session"]))].copy()
    pool = session[
        session["coherence_bin"].astype(str).eq(str(target["coherence_bin"]))
        & session["trial_idx"].astype(int).ne(int(target["trial_idx"]))
        & session["image_feature_ok"].fillna(False).astype(bool)
        & session["n_samples"].astype(int).eq(128)
    ].dropna(subset=list(MATCH_FEATURES) + ["image_edge_axis_deg"])
    if pool.empty:
        raise RuntimeError(f"No matched partner pool for {target['example_role']}")

    session_finite = session.dropna(subset=list(MATCH_FEATURES))
    scales = session_finite[list(MATCH_FEATURES)].astype(float).std(ddof=1).replace(0.0, 1.0)
    target_values = target[list(MATCH_FEATURES)].astype(float).to_numpy()
    zdelta = (
        pool[list(MATCH_FEATURES)].astype(float).to_numpy() - target_values[None, :]
    ) / scales.to_numpy()[None, :]
    pool = pool.copy()
    pool["continuous_match_distance"] = np.sqrt(np.sum(zdelta * zdelta, axis=1))
    pool["phase_mismatch_penalty"] = 0.25 * pool["phase"].astype(str).ne(str(target["phase"])).astype(float)
    pool["match_distance"] = pool["continuous_match_distance"] + pool["phase_mismatch_penalty"]
    pool = pool.sort_values(["match_distance", "trial_idx", "global_start"], kind="stable")
    partner = pool.iloc[0].copy()
    axis_delta = float(cp1.axial_distance_deg(partner["image_edge_axis_deg"], target["image_edge_axis_deg"]))
    diagnostics: dict[str, Any] = {
        "partner_pool_n": int(len(pool)),
        "match_distance": float(partner["match_distance"]),
        "continuous_match_distance": float(partner["continuous_match_distance"]),
        "phase_mismatch_penalty": float(partner["phase_mismatch_penalty"]),
        "phase_exact_match": bool(str(partner["phase"]) == str(target["phase"])),
        "coherence_bin_exact_match": bool(str(partner["coherence_bin"]) == str(target["coherence_bin"])),
        "local_partner_edge_axis_delta_deg": axis_delta,
    }
    for feature in MATCH_FEATURES:
        label = MATCH_FEATURE_LABELS[feature].replace(" ", "_").lower()
        diagnostics[f"target_{label}"] = float(target[feature])
        diagnostics[f"partner_{label}"] = float(partner[feature])
        diagnostics[f"partner_minus_target_{label}"] = float(partner[feature] - target[feature])
        diagnostics[f"standardized_delta_{label}"] = float((partner[feature] - target[feature]) / scales[feature])
    return partner, diagnostics


def rotate_trace(trace: np.ndarray, angle_rad: float) -> np.ndarray:
    centered = np.asarray(trace, dtype=float) - np.mean(trace, axis=0, keepdims=True)
    c, s = math.cos(float(angle_rad)), math.sin(float(angle_rad))
    rotation = np.array([[c, -s], [s, c]])
    return centered @ rotation.T


def contour_coordinates(trace: np.ndarray, edge_axis_deg: float) -> np.ndarray:
    centered = np.asarray(trace, dtype=float) - np.mean(trace, axis=0, keepdims=True)
    along = cp1.axis_vector(edge_axis_deg)
    across = np.array([-along[1], along[0]])
    return np.column_stack([centered @ along, centered @ across])


def spread_profile(trace: np.ndarray, edge_axis_deg: float) -> np.ndarray:
    centered = np.asarray(trace, dtype=float) - np.mean(trace, axis=0, keepdims=True)
    absolute_angles = np.radians(PROFILE_ANGLES_DEG + float(edge_axis_deg))
    axes = np.column_stack([np.cos(absolute_angles), np.sin(absolute_angles)])
    projections = centered @ axes.T
    return np.sqrt(np.mean(projections * projections, axis=0)) * 60.0


def component_doses(trace: np.ndarray, edge_axis_deg: float) -> dict[tuple[str, str], float]:
    snippet = bridge._central_snippet(np.asarray(trace, dtype=float))
    if snippet.shape != (bridge.PANEL_G_SNIPPET_N_SAMPLES, 2):
        raise RuntimeError(f"Unexpected Panel-G snippet shape {snippet.shape}")
    xy = contour_coordinates(snippet, edge_axis_deg)
    steps = np.diff(xy, axis=0)
    along_pos, across_pos = xy[:, 0], xy[:, 1]
    along_range = (np.max(along_pos) - np.min(along_pos)) * 60.0
    across_range = (np.max(across_pos) - np.min(across_pos)) * 60.0
    return {
        ("component_rms", "along"): float(np.sqrt(np.mean(along_pos * along_pos)) * 60.0),
        ("component_rms", "across"): float(np.sqrt(np.mean(across_pos * across_pos)) * 60.0),
        ("component_range", "along"): float(along_range),
        ("component_range", "across"): float(across_range),
    }


def model_curve(model_values: pd.DataFrame, metric: str, component: str) -> pd.DataFrame:
    curve = model_values[
        model_values["population_key"].astype(str).eq(MODEL_POPULATION)
        & model_values["metric_family"].astype(str).eq(metric)
        & model_values["component"].astype(str).eq(component)
    ].copy()
    return curve.sort_values("plot_median").drop_duplicates("plot_median", keep="last")


def interpolate_curve_row(dose: float, curve: pd.DataFrame) -> dict[str, Any]:
    x = curve["plot_median"].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    outside = not np.isfinite(dose) or dose < x[0] or dose > x[-1]
    columns = (
        "ssi_percent_vs_cell_baseline",
        "moving_population_ssi_bits_per_spike",
        "cell_baseline_population_ssi_bits_per_spike",
        "moving_information_bits_per_sample",
        "cell_baseline_information_bits_per_sample",
        "moving_expected_spikes_per_sample",
        "cell_baseline_expected_spikes_per_sample",
    )
    out: dict[str, Any] = {"outside_model_range": bool(outside)}
    for column in columns:
        if outside or column not in curve.columns:
            out[column] = float("nan")
            continue
        y = curve[column].to_numpy(dtype=float)[order]
        ok = np.isfinite(x) & np.isfinite(y)
        out[column] = float(np.interp(dose, x[ok], y[ok])) if np.count_nonzero(ok) >= 2 else float("nan")
    return out


def crop_pair(target: pd.Series, partner: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return cp1.crop_patch(target)[0], cp1.crop_patch(partner)[0]


def plot_patch_pair(ax: plt.Axes, target_patch: np.ndarray, partner_patch: np.ndarray, target: pd.Series, partner: pd.Series) -> None:
    gap = 5
    height = max(target_patch.shape[0], partner_patch.shape[0])
    width = target_patch.shape[1] + gap + partner_patch.shape[1]
    canvas = np.full((height, width), np.nan)
    canvas[: target_patch.shape[0], : target_patch.shape[1]] = target_patch
    partner_x0 = target_patch.shape[1] + gap
    canvas[: partner_patch.shape[0], partner_x0:] = partner_patch
    ax.imshow(canvas, cmap="gray", origin="upper", interpolation="nearest")
    for patch, x0, row, color in (
        (target_patch, 0, target, "#245c8a"),
        (partner_patch, partner_x0, partner, "#d08022"),
    ):
        cy, cx = (np.asarray(patch.shape) - 1.0) / 2.0
        cx += x0
        length = 0.39 * min(patch.shape)
        theta = math.radians(float(row["image_edge_axis_array_deg"]))
        dx, dy = length * math.cos(theta), length * math.sin(theta)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=2.0)
        ax.scatter([cx], [cy], c="#cc3d3d", marker="+", s=14, linewidths=1.0)
    ax.text(0.23, 0.02, "local", transform=ax.transAxes, ha="center", va="bottom", color="#245c8a", fontsize=7.5, weight="bold")
    ax.text(0.77, 0.02, "partner", transform=ax.transAxes, ha="center", va="bottom", color="#d08022", fontsize=7.5, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def plot_local_path(ax: plt.Axes, trace: np.ndarray, edge_axis_deg: float, limit: float, title: str | None = None) -> None:
    xy = contour_coordinates(trace, edge_axis_deg)
    ax.plot(xy[:, 0], xy[:, 1], color="#33383e", lw=0.7)
    ax.scatter(xy[0, 0], xy[0, 1], s=18, c="#2a72b5", zorder=4)
    ax.scatter(xy[-1, 0], xy[-1, 1], s=20, c="#c54a3d", marker="s", zorder=4)
    ax.axhline(0, color="#1b7f5c", lw=0.7); ax.axvline(0, color="#b8bdc3", lw=0.5)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_aspect("equal")
    ax.tick_params(labelsize=6.3)
    ax.set_xlabel("along edge (deg)", fontsize=7); ax.set_ylabel("normal (deg)", fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)


def score_frame_for_plot(values: pd.DataFrame) -> pd.DataFrame:
    group = values.groupby(["example_role", "construction", "draw_index", "metric_family"], sort=False)["ssi_percent_vs_cell_baseline"]
    scores = group.mean().rename("component_mean_predicted_ssi_residual_percent").reset_index()
    return scores


def render(
    targets: list[pd.Series],
    partners: list[pd.Series],
    traces: list[dict[str, np.ndarray]],
    patches: list[tuple[np.ndarray, np.ndarray]],
    rotation_angles: list[np.ndarray],
    values: pd.DataFrame,
    profiles: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(targets), 7, figsize=(18.8, 16.0),
        gridspec_kw={"width_ratios": [1.40, 1, 1, 1, 1, 1.32, 1.28]},
    )
    scores = score_frame_for_plot(values)
    for row_idx, (target, partner, trace_set, patch_set, angles) in enumerate(
        zip(targets, partners, traces, patches, rotation_angles, strict=True)
    ):
        role = str(target["example_role"])
        target_edge = float(target["image_edge_axis_deg"])
        partner_edge = float(partner["image_edge_axis_deg"])
        representative_rotation = float(angles[0])
        rotated_display = rotate_trace(trace_set["target"], representative_rotation)

        ax = axes[row_idx, 0]
        plot_patch_pair(ax, patch_set[0], patch_set[1], target, partner)
        ax.set_ylabel(
            f"{row_idx + 1}. {cp1.ROLE_LABEL[role]}\n{target['subject']} {str(target['session']).split('_', 1)[1]} | tr {int(target['trial_idx'])}",
            rotation=0, ha="right", va="center", labelpad=12, fontsize=8.3, weight="bold",
        )
        ax.text(
            0.5, -0.08,
            f"edge Δ={float(cp1.axial_distance_deg(target_edge, partner_edge)):.1f}° | match d={float(target['match_distance']):.2f}",
            transform=ax.transAxes, ha="center", fontsize=7,
        )
        if row_idx == 0:
            ax.set_title("Observed image patches\n+ measured edge axes", fontsize=9.5)

        construction_paths = (
            ("real_local_pair", trace_set["target"], target_edge),
            ("uniform_rotation", rotated_display, target_edge),
            ("trajectory_reassigned", trace_set["partner"], target_edge),
            ("image_axis_reassigned", trace_set["target"], partner_edge),
        )
        all_xy = [contour_coordinates(trace, edge) for _, trace, edge in construction_paths]
        limit = max(0.16, max(float(np.max(np.abs(xy))) for xy in all_xy) * 1.08)
        for col_idx, (construction, trace, edge) in enumerate(construction_paths, start=1):
            title = CONSTRUCTION_LABELS[construction] if row_idx == 0 else None
            plot_local_path(axes[row_idx, col_idx], trace, edge, limit, title)
            if construction == "uniform_rotation":
                axes[row_idx, col_idx].text(
                    0.03, 0.97, f"display draw\n{math.degrees(representative_rotation):.1f}°",
                    transform=axes[row_idx, col_idx].transAxes, va="top", fontsize=6.6,
                )

        ax = axes[row_idx, 5]
        sub = profiles[profiles["example_role"].astype(str).eq(role)]
        for construction in ("real_local_pair", "trajectory_reassigned", "image_axis_reassigned"):
            line = sub[sub["construction"].astype(str).eq(construction)].sort_values("relative_axis_deg")
            ax.plot(line["relative_axis_deg"], line["rms_arcmin"], color=CONSTRUCTION_COLORS[construction], lw=1.5, label=CONSTRUCTION_LABELS[construction].replace("\n", " "))
        rot = sub[sub["construction"].astype(str).eq("uniform_rotation")].sort_values("relative_axis_deg")
        ax.plot(rot["relative_axis_deg"], rot["rms_arcmin"], color=CONSTRUCTION_COLORS["uniform_rotation"], lw=1.5, label="rotation mean")
        ax.fill_between(rot["relative_axis_deg"], rot["rms_ci95_low"], rot["rms_ci95_high"], color=CONSTRUCTION_COLORS["uniform_rotation"], alpha=0.14, lw=0)
        ax.axvline(0, color="#1b7f5c", lw=0.7, ls="--"); ax.axvline(90, color="#7a3b9a", lw=0.7, ls="--")
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xlabel("projection axis − paired edge (deg)", fontsize=7); ax.set_ylabel("position RMS (arcmin)", fontsize=7)
        ax.tick_params(labelsize=6.3); ax.grid(alpha=0.18, lw=0.5)
        if row_idx == 0:
            ax.set_title("Panel F object\nfull spread profile", fontsize=9.5)
            ax.legend(fontsize=5.8, frameon=False, loc="best")

        ax = axes[row_idx, 6]
        role_scores = scores[scores["example_role"].astype(str).eq(role)]
        x = np.arange(4, dtype=float)
        offsets = {"component_rms": -0.10, "component_range": 0.10}
        markers = {"component_rms": "o", "component_range": "s"}
        for metric in MODEL_METRICS:
            means, lows, highs = [], [], []
            for construction in CONSTRUCTION_LABELS:
                vals = role_scores[
                    role_scores["construction"].astype(str).eq(construction)
                    & role_scores["metric_family"].astype(str).eq(metric)
                ]["component_mean_predicted_ssi_residual_percent"].dropna().to_numpy(dtype=float)
                means.append(float(np.mean(vals)) if vals.size else np.nan)
                lows.append(float(np.percentile(vals, 2.5)) if construction == "uniform_rotation" and vals.size else np.nan)
                highs.append(float(np.percentile(vals, 97.5)) if construction == "uniform_rotation" and vals.size else np.nan)
            means_arr = np.asarray(means)
            yerr = np.vstack([means_arr - np.asarray(lows), np.asarray(highs) - means_arr])
            yerr[:, ~np.isfinite(yerr).all(axis=0)] = 0.0
            ax.errorbar(
                x + offsets[metric], means_arr, yerr=yerr, marker=markers[metric], lw=1.0,
                ms=4, capsize=2, label="RMS dose" if metric == "component_rms" else "range dose",
            )
        ax.axhline(0, color="#80858b", lw=0.7, ls=":")
        ax.set_xticks(x, ["real", "rotate", "traj", "image"], rotation=22, ha="right")
        ax.set_ylabel("predicted SSI residual (%)", fontsize=7); ax.tick_params(labelsize=6.3)
        ax.grid(axis="y", alpha=0.18, lw=0.5)
        if row_idx == 0:
            ax.set_title("Panel G dose-curve interpolation\naligned high-SF units", fontsize=9.5)
            ax.legend(fontsize=6, frameon=False)

    for ax in axes[:, 0].flat:
        ax.set_facecolor("#fbfaf6")
    for ax in axes[:, 1:6].flat:
        ax.set_facecolor("#f5f8fb")
    for ax in axes[:, 6].flat:
        ax.set_facecolor("#faf7fb")
    fig.suptitle(
        "Checkpoint 2 — Does the specific local image/trajectory pairing survive marginal-preserving nulls?\n"
        "Paths show concrete constructions; rotation profiles and model scores summarize 256 uniform draws",
        y=0.998, fontsize=13.2, weight="bold",
    )
    fig.text(0.20, 0.954, "OBSERVED INPUTS", ha="center", fontsize=10.5, weight="bold")
    fig.text(0.53, 0.954, "PAIRING CONSTRUCTIONS / BEHAVIOR", ha="center", fontsize=10.5, weight="bold")
    fig.text(0.925, 0.954, "MODEL PROXY", ha="center", fontsize=10.5, weight="bold")
    fig.subplots_adjust(left=0.135, right=0.992, top=0.925, bottom=0.045, hspace=0.43, wspace=0.42)
    boundary1 = 0.5 * (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0)
    boundary2 = 0.5 * (axes[0, 5].get_position().x1 + axes[0, 6].get_position().x0)
    for boundary in (boundary1, boundary2):
        fig.add_artist(plt.Line2D([boundary, boundary], [0.035, 0.95], transform=fig.transFigure, color="#a6adb5", lw=0.9))
    fig.text(
        0.99, 0.012,
        "Model proxy uses central 40 samples (0.325 s), one-dimensional marginal curves. Absolute SSI, information/sample, and expected spikes/sample are saved in the values table.",
        ha="right", fontsize=7, color="#4a4f55",
    )
    fig.savefig(out_dir / "checkpoint2_pairing_null_examples.png", dpi=210)
    fig.savefig(out_dir / "checkpoint2_pairing_null_examples.pdf")
    plt.close(fig)


def git_value(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-rotations", type=int, default=N_ROTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_values_path = EXTENDED_MODEL_VALUES if EXTENDED_MODEL_VALUES.exists() else FALLBACK_MODEL_VALUES
    model_values = pd.read_csv(model_values_path)
    table = load_analysis_table()
    selected = pd.read_csv(SELECTED_WINDOWS)

    targets: list[pd.Series] = []
    partners: list[pd.Series] = []
    trace_sets: list[dict[str, np.ndarray]] = []
    patch_sets: list[tuple[np.ndarray, np.ndarray]] = []
    rotation_angles: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))

    curves = {
        (metric, component): model_curve(model_values, metric, component)
        for metric in MODEL_METRICS for component in ("along", "across")
    }
    for selected_row in selected.itertuples(index=False):
        target = find_target(table, pd.Series(selected_row._asdict()))
        partner, diagnostics = select_partner(table, target)
        target["match_distance"] = diagnostics["match_distance"]
        target_trace = np.asarray(_window_trace(target), dtype=float)
        partner_trace = np.asarray(_window_trace(partner), dtype=float)
        if target_trace.shape != (128, 2) or partner_trace.shape != (128, 2):
            raise RuntimeError(f"Checkpoint 2 requires native 128x2 traces for {target['example_role']}")
        target_patch, partner_patch = crop_pair(target, partner)
        angles = rng.uniform(0.0, np.pi, size=int(args.n_rotations))

        targets.append(target); partners.append(partner)
        trace_sets.append({"target": target_trace, "partner": partner_trace})
        patch_sets.append((target_patch, partner_patch)); rotation_angles.append(angles)
        manifest: dict[str, Any] = {
            "example_role": target["example_role"],
            **{f"target_{key}": target[key] for key in KEYS},
            **{f"partner_{key}": partner[key] for key in KEYS},
            "target_phase": target["phase"],
            "partner_phase": partner["phase"],
            "target_coherence_bin": target["coherence_bin"],
            "partner_coherence_bin": partner["coherence_bin"],
            "target_edge_axis_deg": float(target["image_edge_axis_deg"]),
            "partner_edge_axis_deg": float(partner["image_edge_axis_deg"]),
            "uniform_rotation_seed": int(args.seed),
            "n_uniform_rotation_draws": int(args.n_rotations),
            "display_rotation_draw_index": 0,
            "display_rotation_angle_deg": float(np.degrees(angles[0])),
            "matching_contract": "same subject+session and coherence bin; different trial; minimize session-standardized distance over coherence, FEM RMS, snippet extent, anisotropy, gaze eccentricity, time since event; +0.25 phase mismatch penalty",
            **diagnostics,
        }
        manifest_rows.append(manifest)

        exact_constructions = (
            ("real_local_pair", target_trace, float(target["image_edge_axis_deg"]), target, target),
            ("trajectory_reassigned", partner_trace, float(target["image_edge_axis_deg"]), partner, target),
            ("image_axis_reassigned", target_trace, float(partner["image_edge_axis_deg"]), target, partner),
        )
        for construction, trace, edge_axis, trajectory_source, image_source in exact_constructions:
            profile = spread_profile(trace, edge_axis)
            for angle_deg, rms in zip(PROFILE_ANGLES_DEG, profile, strict=True):
                profile_rows.append({
                    "example_role": target["example_role"], "construction": construction,
                    "relative_axis_deg": angle_deg, "rms_arcmin": rms,
                    "rms_ci95_low": np.nan, "rms_ci95_high": np.nan,
                })
            doses = component_doses(trace, edge_axis)
            for (metric, component), dose in doses.items():
                value_rows.append({
                    "example_role": target["example_role"], "construction": construction,
                    "draw_index": -1, "rotation_angle_deg": 0.0,
                    "paired_edge_axis_deg": edge_axis,
                    "trajectory_session": trajectory_source["session"],
                    "trajectory_trial_idx": int(trajectory_source["trial_idx"]),
                    "trajectory_global_start": int(trajectory_source["global_start"]),
                    "image_session": image_source["session"], "image_trial_idx": int(image_source["trial_idx"]),
                    "image_global_start": int(image_source["global_start"]),
                    "metric_family": metric, "component": component, "behavior_dose": dose,
                    **interpolate_curve_row(dose, curves[(metric, component)]),
                })

        rotation_profiles = []
        for draw_index, angle in enumerate(angles):
            rotated = rotate_trace(target_trace, float(angle))
            rotation_profiles.append(spread_profile(rotated, float(target["image_edge_axis_deg"])))
            doses = component_doses(rotated, float(target["image_edge_axis_deg"]))
            for (metric, component), dose in doses.items():
                value_rows.append({
                    "example_role": target["example_role"], "construction": "uniform_rotation",
                    "draw_index": int(draw_index), "rotation_angle_deg": float(np.degrees(angle)),
                    "paired_edge_axis_deg": float(target["image_edge_axis_deg"]),
                    "trajectory_session": target["session"], "trajectory_trial_idx": int(target["trial_idx"]),
                    "trajectory_global_start": int(target["global_start"]),
                    "image_session": target["session"], "image_trial_idx": int(target["trial_idx"]),
                    "image_global_start": int(target["global_start"]),
                    "metric_family": metric, "component": component, "behavior_dose": dose,
                    **interpolate_curve_row(dose, curves[(metric, component)]),
                })
        rotation_profiles_arr = np.vstack(rotation_profiles)
        mean = np.mean(rotation_profiles_arr, axis=0)
        low, high = np.percentile(rotation_profiles_arr, [2.5, 97.5], axis=0)
        for angle_deg, rms, lo, hi in zip(PROFILE_ANGLES_DEG, mean, low, high, strict=True):
            profile_rows.append({
                "example_role": target["example_role"], "construction": "uniform_rotation",
                "relative_axis_deg": angle_deg, "rms_arcmin": rms,
                "rms_ci95_low": lo, "rms_ci95_high": hi,
            })

    manifest_df = pd.DataFrame(manifest_rows)
    values_df = pd.DataFrame(value_rows)
    profiles_df = pd.DataFrame(profile_rows)
    manifest_df.to_csv(out_dir / "checkpoint2_pairing_manifest.csv", index=False)
    values_df.to_csv(out_dir / "checkpoint2_pairing_null_example_values.csv", index=False)
    profiles_df.to_csv(out_dir / "checkpoint2_pairing_null_spread_profiles.csv", index=False)
    render(targets, partners, trace_sets, patch_sets, rotation_angles, values_df, profiles_df, out_dir)

    raw_columns = [
        "moving_information_bits_per_sample", "cell_baseline_information_bits_per_sample",
        "moving_expected_spikes_per_sample", "cell_baseline_expected_spikes_per_sample",
    ]
    metadata = {
        "artifact_type": "map_first_targeted_visualization_checkpoint",
        "checkpoint": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_inference_performed": False,
        "input_windows": str(SOURCE_WINDOWS),
        "checkpoint1_selection": str(SELECTED_WINDOWS),
        "snippet_metrics": str(SNIPPET_METRICS),
        "model_values": str(model_values_path),
        "model_population": MODEL_POPULATION,
        "model_prediction_contract": "piecewise-linear interpolation of one-dimensional component RMS/range marginal curves; component-mean summaries average available parallel and normal predictions; not a 2D response surface",
        "model_raw_components_available": all(column in model_values.columns for column in raw_columns),
        "n_uniform_rotations": int(args.n_rotations),
        "seed": int(args.seed),
        "trace_contract": "native reviewed 128-sample traces; Panel-F spread uses all 128 samples; Panel-G dose uses central 40 samples (0.325 s)",
        "rotation_contract": "independent uniform axial rotation in [0,180 deg); paths display draw 0, profiles and model scores summarize all draws",
        "pairing_contract": "target image+target trajectory; target image+rotated target trajectory; target image+matched partner trajectory; matched partner image axis+target trajectory",
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "outputs": [
            "checkpoint2_pairing_null_examples.png", "checkpoint2_pairing_null_examples.pdf",
            "checkpoint2_pairing_null_example_values.csv", "checkpoint2_pairing_manifest.csv",
            "checkpoint2_pairing_null_spread_profiles.csv", "checkpoint2_run_metadata.json",
        ],
    }
    with (out_dir / "checkpoint2_run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(manifest_df[["example_role", "partner_trial_idx", "match_distance", "phase_exact_match", "local_partner_edge_axis_delta_deg"]].to_string(index=False))
    print(f"Wrote Checkpoint 2 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
