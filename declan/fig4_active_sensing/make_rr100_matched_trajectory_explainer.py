#!/usr/bin/env python3
"""Checkpoint 14: reader-facing figures for the matched-trajectory analysis.

The figures unpack the exact cached comparison at two levels:

1. one image patch with its own trace versus eight RMS-matched traces from
   other trials, followed by the operational response decomposition;
2. four auditable examples selected before visualization;
3. the population bridge from concrete examples to the grouped rank result.

Important provenance: the production cache predates the native-timing fix and
used ``resample_full_window`` to construct 48-frame paths.  The paths are
reconstructed exactly and labeled as resampled cache paths.  These figures do
not support calibrated native-speed or temporal-frequency claims.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
    _build_trace_bank,
    _scale_family_raw_trace,
    _session_dataset_cache,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _clip_patch


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_local_pairing_Iz_conditional_n384_k8_rel1_seed23_v1"
)
SOURCE = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
MAP_CHECKPOINT = ROOT / "outputs/fig4_active_sensing/rr100_fullbank_map_checkpoint_13a_v1"
RANK_CHECKPOINT = ROOT / "outputs/fig4_active_sensing/rr100_fullbank_rank_checkpoint_13b_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_matched_trajectory_explainer_checkpoint_14_v1"

OWN = "#d1495b"
MATCHED = "#2f6690"
MATCHED_LIGHT = "#a9c7dc"
RESIDUAL = "#6a4c93"
INK = "#242424"
GRID = "#ececec"
GOLD = "#e9c46a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--source-csv", type=Path, default=SOURCE)
    parser.add_argument("--map-checkpoint", type=Path, default=MAP_CHECKPOINT)
    parser.add_argument("--rank-checkpoint", type=Path, default=RANK_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def reconstruct_examples(
    cache_dir: Path,
    source_csv: Path,
    selected: pd.DataFrame,
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame, dict[str, float]]:
    source = load_source_rows(source_csv)
    motion = pd.read_csv(cache_dir / "local_pairing_motion_metadata.csv")
    chosen = motion.loc[
        motion["image_index"].isin(selected["image_index"])
        & motion["family"].isin(["actual_paired_empirical", "matched_unpaired_empirical"])
    ].copy()
    source_rows = sorted(chosen["trace_source_row"].astype(int).unique())
    work = source.loc[source["source_row"].isin(source_rows)].copy()
    if len(work) != len(source_rows):
        raise ValueError(f"Only found {len(work)}/{len(source_rows)} required trace source rows")
    eye_positions = _session_dataset_cache(work["session"].astype(str).tolist())
    trace_bank = _build_trace_bank(
        work,
        eye_positions,
        48,
        microsaccade_speed_threshold_dps=None,
        microsaccade_threshold_z=6.0,
        microsaccade_pad_frames=1,
        trace_window_policy="resample_full_window",
    )
    by_source = {int(item["source_row"]): item for item in trace_bank}

    examples: dict[int, dict[str, np.ndarray]] = {}
    trace_rows = []
    rms_errors = []
    path_errors = []
    for selected_row in selected.itertuples():
        image_index = int(selected_row.image_index)
        source_row = int(selected_row.source_row)
        image_source = source.loc[source["source_row"].eq(source_row)].iloc[0]
        canvas, ppd, screen_shape = _backimage_canvas(str(image_source["session"]), int(image_source["trial_idx"]))
        center = gaze_deg_to_screen_px(
            np.asarray([float(image_source["mean_x_deg"]), float(image_source["mean_y_deg"])]),
            ppd=ppd,
            screen_shape=screen_shape,
        )
        patch = _clip_patch(canvas, (float(center[0]), float(center[1])), 540).astype(np.float32)

        rows = chosen.loc[chosen["image_index"].eq(image_index)].sort_values(
            ["family", "sample_index"],
            key=lambda col: col.map({"actual_paired_empirical": 0, "matched_unpaired_empirical": 1})
            if col.name == "family" else col,
        )
        own = None
        matched = []
        for row in rows.itertuples():
            raw = by_source[int(row.trace_source_row)]["trace"]
            trace, trace_meta = _scale_family_raw_trace(
                raw,
                float(row.requested_rms_deg),
                max_rms_deg=0.12,
            )
            rms_errors.append(abs(float(trace_meta["effective_rms_deg"]) - float(row.effective_rms_deg)))
            path_errors.append(abs(float(trace_meta["path_length_deg"]) - float(row.path_length_deg)))
            if row.family == "actual_paired_empirical":
                own = trace
                display_sample = -1
            else:
                matched.append(trace)
                display_sample = int(row.sample_index)
            for frame, xy in enumerate(trace):
                trace_rows.append(
                    {
                        "selection_role": selected_row.selection_role,
                        "image_index": image_index,
                        "source_row": source_row,
                        "family": row.family,
                        "sample_index": display_sample,
                        "trace_source_row": int(row.trace_source_row),
                        "frame": frame,
                        "x_deg": float(xy[0]),
                        "y_deg": float(xy[1]),
                        "requested_rms_deg": float(row.requested_rms_deg),
                        "path_length_deg": float(row.path_length_deg),
                        "trace_window_policy": "resample_full_window",
                    }
                )
        if own is None or len(matched) != 8:
            raise ValueError(f"Example {image_index} has own={own is not None}, matched={len(matched)}")
        examples[image_index] = {
            "patch": patch,
            "own_trace": np.asarray(own, np.float32),
            "matched_traces": np.stack(matched).astype(np.float32),
            "ppd": np.asarray(ppd, np.float32),
        }
    reconstruction = {
        "maximum_effective_rms_reconstruction_error_deg": float(max(rms_errors)),
        "maximum_path_length_reconstruction_error_deg": float(max(path_errors)),
    }
    return examples, pd.DataFrame(trace_rows), reconstruction


def load_responses(map_checkpoint: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(map_checkpoint / "fullbank_rr100_effect_matrices.npz")
    paired = archive["paired_four_dct_rms_hz"].astype(float)
    samples = archive["matched_unpaired_sample_four_dct_rms_hz"].astype(float)
    matched = samples.mean(axis=1)
    return paired, matched, samples


def path_limits(own: np.ndarray, matched: np.ndarray, pad: float = 0.015) -> tuple[float, float, float, float]:
    all_paths = np.concatenate([own[None], matched], axis=0)
    xmin, ymin = all_paths.min(axis=(0, 1))
    xmax, ymax = all_paths.max(axis=(0, 1))
    span = max(xmax - xmin, ymax - ymin, 0.02)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * span + pad
    return cx - half, cx + half, cy - half, cy + half


def draw_path(ax: plt.Axes, trace: np.ndarray, color: str, *, alpha: float = 1.0, lw: float = 1.2) -> None:
    ax.plot(trace[:, 0], trace[:, 1], color=color, alpha=alpha, lw=lw)
    ax.scatter(trace[0, 0], trace[0, 1], s=18, facecolor="white", edgecolor=color, linewidth=0.8, zorder=3)
    ax.scatter(trace[-1, 0], trace[-1, 1], s=20, marker="X", color=color, alpha=alpha, zorder=3)


def style_path_axis(ax: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, lw=0.7, zorder=0)
    ax.axvline(0, color=GRID, lw=0.7, zorder=0)
    ax.set_xlabel("horizontal displacement (deg)", fontsize=8)
    ax.set_ylabel("vertical displacement (deg)", fontsize=8)
    ax.tick_params(labelsize=7)


def make_construction_figure(
    selected: pd.DataFrame,
    examples: dict[int, dict[str, np.ndarray]],
    paired: np.ndarray,
    matched: np.ndarray,
    samples: np.ndarray,
    out: Path,
    dpi: int,
) -> None:
    chosen = selected.loc[selected["selection_role"].eq("paired_enhanced")].iloc[0]
    index = int(chosen["image_index"])
    example = examples[index]
    own_trace = example["own_trace"]
    matched_traces = example["matched_traces"]
    limits = path_limits(own_trace, matched_traces)
    residual = paired[index] - matched[index]

    fig = plt.figure(figsize=(17.0, 8.7))
    gs = fig.add_gridspec(
        2, 5,
        height_ratios=[1.05, 0.78],
        width_ratios=[1.05, 1.0, 1.08, 1.55, 1.25],
        left=0.045, right=0.98, bottom=0.10, top=0.87, wspace=0.32, hspace=0.40,
    )

    ax = fig.add_subplot(gs[0, 0])
    lo, hi = np.percentile(example["patch"], [1, 99])
    ax.imshow(example["patch"], cmap="gray", vmin=lo, vmax=hi)
    ax.set_title("A  Hold the image fixed", loc="left", fontweight="bold")
    ax.text(0.02, -0.08, f"window {index} · 540-pixel scored patch", transform=ax.transAxes, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    draw_path(ax, own_trace, OWN, lw=1.8)
    style_path_axis(ax, limits)
    ax.set_title("B  Use its own path once", loc="left", fontweight="bold")
    ax.text(0.02, 0.98, "open circle: start\n×: end", transform=ax.transAxes, va="top", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    for trace in matched_traces:
        draw_path(ax, trace, MATCHED, alpha=0.58, lw=0.95)
    style_path_axis(ax, limits)
    ax.set_title("C  Resample 8 other paths", loc="left", fontweight="bold")
    ax.text(0.02, 0.98, "different trials\nsame RMS amplitude", transform=ax.transAxes, va="top", fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    units = np.arange(100)
    for sample in range(8):
        ax.plot(units, samples[index, sample], color=MATCHED_LIGHT, alpha=0.75, lw=0.75)
    ax.plot(units, matched[index], color=MATCHED, lw=2.1, label="mean of 8 matched paths")
    ax.plot(units, paired[index], color=OWN, lw=2.0, label="own path")
    ax.set_xlabel("RR100 unit")
    ax.set_ylabel("low-frequency modulation RMS (Hz)")
    ax.set_title("D  Score every movie", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID)

    ax = fig.add_subplot(gs[0, 4])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.fill_between(units, 0, residual, where=residual >= 0, color=RESIDUAL, alpha=0.65)
    ax.fill_between(units, 0, residual, where=residual < 0, color="white", edgecolor=RESIDUAL, linewidth=0.8)
    ax.plot(units, residual, color=RESIDUAL, lw=1.0)
    ax.set_xlabel("RR100 unit")
    ax.set_ylabel("own − matched mean (Hz)")
    ax.set_title("E  Keep the pairing residual", loc="left", fontweight="bold")
    ax.grid(color=GRID)

    equation = fig.add_subplot(gs[1, :])
    equation.axis("off")
    equation.add_patch(plt.Rectangle((0.03, 0.20), 0.26, 0.55, transform=equation.transAxes, facecolor="#eef5fa", edgecolor=MATCHED, lw=1.4))
    equation.add_patch(plt.Rectangle((0.38, 0.20), 0.24, 0.55, transform=equation.transAxes, facecolor="#f4eef8", edgecolor=RESIDUAL, lw=1.4))
    equation.add_patch(plt.Rectangle((0.71, 0.20), 0.26, 0.55, transform=equation.transAxes, facecolor="#faeeee", edgecolor=OWN, lw=1.4))
    equation.text(0.16, 0.57, "Trajectory-averaged\nimage susceptibility", ha="center", va="center", fontsize=13, fontweight="bold", color=MATCHED)
    equation.text(0.16, 0.31, "mean modulation under 8\nmatched other-trial paths", ha="center", va="center", fontsize=9, color=INK)
    equation.text(0.335, 0.48, "+", ha="center", va="center", fontsize=28, color=INK)
    equation.text(0.50, 0.57, "Pairing-specific\nresidual", ha="center", va="center", fontsize=13, fontweight="bold", color=RESIDUAL)
    equation.text(0.50, 0.31, "own path minus the\nmatched-path average", ha="center", va="center", fontsize=9, color=INK)
    equation.text(0.665, 0.48, "=", ha="center", va="center", fontsize=28, color=INK)
    equation.text(0.84, 0.57, "Observed response under\nthe image's own path", ha="center", va="center", fontsize=13, fontweight="bold", color=OWN)
    equation.text(0.84, 0.31, "an operational decomposition,\nnot yet a causal model", ha="center", va="center", fontsize=9, color=INK)

    fig.suptitle("Matched-trajectory construction for one fixed image", fontsize=16, fontweight="bold", y=0.97)
    fig.text(
        0.5, 0.025,
        "Trajectory panels show the exact 48-frame resampled paths used by this legacy cache; they are RMS-matched but not native-time calibrated.",
        ha="center", fontsize=9, color="#6b4f00",
        bbox={"facecolor": "#fff6d8", "edgecolor": GOLD, "pad": 5},
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_four_case_figure(
    selected: pd.DataFrame,
    examples: dict[int, dict[str, np.ndarray]],
    paired: np.ndarray,
    matched: np.ndarray,
    samples: np.ndarray,
    out: Path,
    dpi: int,
) -> None:
    roles = ["shared_high", "shared_low", "paired_enhanced", "paired_suppressed"]
    role_titles = {
        "shared_high": "Shared high: this image is strong across trajectories",
        "shared_low": "Shared low: this image is weak across trajectories",
        "paired_enhanced": "Pairing enhanced: its own path drives more modulation",
        "paired_suppressed": "Pairing suppressed: its own path drives less modulation",
    }
    fig = plt.figure(figsize=(18.0, 15.2))
    gs = fig.add_gridspec(
        4, 5,
        width_ratios=[0.92, 0.9, 1.0, 1.7, 1.28],
        left=0.04, right=0.985, bottom=0.055, top=0.91, hspace=0.42, wspace=0.30,
    )
    for row_number, role in enumerate(roles):
        selected_row = selected.loc[selected["selection_role"].eq(role)].iloc[0]
        index = int(selected_row["image_index"])
        example = examples[index]
        own_trace = example["own_trace"]
        matched_traces = example["matched_traces"]
        limits = path_limits(own_trace, matched_traces)
        units = np.arange(100)

        ax = fig.add_subplot(gs[row_number, 0])
        lo, hi = np.percentile(example["patch"], [1, 99])
        ax.imshow(example["patch"], cmap="gray", vmin=lo, vmax=hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{role_titles[role]}\nwindow {index} · image fixed",
            loc="left", fontsize=10, fontweight="bold", pad=8,
        )

        ax = fig.add_subplot(gs[row_number, 1])
        draw_path(ax, own_trace, OWN, lw=1.7)
        style_path_axis(ax, limits)
        ax.set_title("own path", fontsize=9, color=OWN, fontweight="bold")

        ax = fig.add_subplot(gs[row_number, 2])
        for trace in matched_traces:
            draw_path(ax, trace, MATCHED, alpha=0.55, lw=0.9)
        style_path_axis(ax, limits)
        ax.set_title("8 matched paths", fontsize=9, color=MATCHED, fontweight="bold")

        ax = fig.add_subplot(gs[row_number, 3])
        for sample in range(8):
            ax.plot(units, samples[index, sample], color=MATCHED_LIGHT, alpha=0.7, lw=0.65)
        ax.plot(units, matched[index], color=MATCHED, lw=1.9, label="matched mean")
        ax.plot(units, paired[index], color=OWN, lw=1.8, label="own")
        ax.set_xlabel("RR100 unit", fontsize=8)
        ax.set_ylabel("modulation RMS (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color=GRID)
        ax.set_title(
            f"own mean {paired[index].mean():.3f} Hz · matched mean {matched[index].mean():.3f} Hz",
            fontsize=9,
        )
        if row_number == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = fig.add_subplot(gs[row_number, 4])
        residual = paired[index] - matched[index]
        ax.axhline(0, color="0.4", lw=0.8)
        ax.fill_between(units, 0, residual, where=residual >= 0, color=RESIDUAL, alpha=0.65)
        ax.fill_between(units, 0, residual, where=residual < 0, facecolor="white", edgecolor=RESIDUAL, linewidth=0.65)
        ax.plot(units, residual, color=RESIDUAL, lw=0.9)
        ax.set_xlabel("RR100 unit", fontsize=8)
        ax.set_ylabel("own − matched (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color=GRID)
        ax.set_title(f"population-mean residual {residual.mean():+.3f} Hz", fontsize=9)

    fig.suptitle(
        "Four preselected matched-trajectory cases: shared image susceptibility and pairing exceptions",
        fontsize=16, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.018,
        "Exact cached comparison. Paths are legacy 48-frame full-window resamples with matched RMS amplitude; use for internal contrasts, not native-time calibration.",
        ha="center", fontsize=9, color="#6b4f00",
        bbox={"facecolor": "#fff6d8", "edgecolor": GOLD, "pad": 5},
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_population_figure(
    selected: pd.DataFrame,
    rank_checkpoint: Path,
    paired: np.ndarray,
    matched: np.ndarray,
    out: Path,
    dpi: int,
) -> None:
    arrays = np.load(rank_checkpoint / "rank_validation_arrays.npz")
    rank = pd.read_csv(rank_checkpoint / "rank_summary_by_condition_level.csv")
    validation = pd.read_csv(rank_checkpoint / "grouped_validation_summary.csv")
    alignment = pd.read_csv(rank_checkpoint / "paired_matched_alignment.csv")
    reliability = pd.read_csv(rank_checkpoint / "matched_trace_split_half_reliability.csv")
    influence = pd.read_csv(rank_checkpoint / "leave_one_session_out_influence.csv")
    own_score = arrays["paired_trial_score"]
    matched_score = arrays["matched_trial_score"]
    own_loading = arrays["paired_trial_loading"]
    matched_loading = arrays["matched_trial_loading"]

    fig, axes = plt.subplots(2, 3, figsize=(16.7, 9.6), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(matched_score, own_score, s=20, alpha=0.58, color=MATCHED, edgecolors="none")
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    rho = alignment.set_index("level").loc["trial", "pc1_score_spearman"]
    ax.set(xlabel="matched-path trial score", ylabel="own-path trial score")
    ax.set_title(f"A  Trial ordering generalizes across paths\nSpearman ρ = {rho:.2f}", loc="left", fontweight="bold")

    ax = axes[0, 1]
    ax.scatter(matched_loading, own_loading, s=26, color="#7aa6c2", edgecolor="white", linewidth=0.35)
    lo = min(matched_loading.min(), own_loading.min()); hi = max(matched_loading.max(), own_loading.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.4", lw=1)
    r = alignment.set_index("level").loc["trial", "pc1_loading_pearson"]
    ax.set(xlabel="matched-path PC1 loading", ylabel="own-path PC1 loading")
    ax.set_title(f"B  The same RR100 units carry the shared pattern\nPearson r = {r:.2f}", loc="left", fontweight="bold")

    ax = axes[0, 2]
    level_labels = [("trial", "trial means"), ("trial_within_session", "session means removed")]
    x = np.arange(3)
    for offset, (level, label) in zip((-0.16, 0.16), level_labels):
        vals = [
            rank.loc[rank["matrix"].eq(name) & rank["level"].eq(level), "centered_per_unit_zscored_pc1_variance_fraction"].iloc[0]
            for name in ("paired", "matched", "pairing_residual")
        ]
        ax.bar(x + offset, vals, width=0.3, color=([OWN, MATCHED, RESIDUAL] if offset < 0 else ["white"] * 3), edgecolor=[OWN, MATCHED, RESIDUAL], linewidth=1.5, label=label)
    ax.set_xticks(x, ["own", "matched", "own − matched"])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("PC1 variance fraction")
    ax.set_title("C  Pairing residual is structured, not noise", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    pop_matched = matched.mean(axis=1)
    pop_own = paired.mean(axis=1)
    ax.scatter(pop_matched, pop_own, s=17, color="0.55", alpha=0.45, edgecolors="none")
    lim = max(pop_matched.max(), pop_own.max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="0.45", lw=1)
    role_colors = {"shared_high": OWN, "shared_low": MATCHED, "paired_enhanced": "#e07a1f", "paired_suppressed": RESIDUAL}
    for row in selected.itertuples():
        index = int(row.image_index)
        ax.scatter(pop_matched[index], pop_own[index], s=90, color=role_colors[row.selection_role], edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(row.selection_role.replace("_", " "), (pop_matched[index], pop_own[index]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set(xlabel="matched-path population mean (Hz)", ylabel="own-path population mean (Hz)")
    ax.set_title("D  Shared tendency plus bidirectional exceptions", loc="left", fontweight="bold")

    ax = axes[1, 1]
    ax.hist(reliability["window_population_mean_pearson"], bins=12, color=MATCHED_LIGHT, edgecolor=MATCHED, linewidth=0.8)
    median = reliability["window_population_mean_pearson"].median()
    ax.axvline(median, color=MATCHED, lw=2)
    ax.set(xlabel="correlation between two independent 4-path means", ylabel="4-vs-4 split count")
    ax.set_title(f"E  Eight matched paths give a stable image score\nmedian r = {median:.2f}", loc="left", fontweight="bold")

    ax = axes[1, 2]
    colors = {"paired": OWN, "matched": MATCHED, "pairing_residual": RESIDUAL}
    labels = {"paired": "own", "matched": "matched", "pairing_residual": "own − matched"}
    for name in colors:
        block = influence.loc[influence["matrix"].eq(name)].sort_values("leave_session_out_pc1_fraction")
        ax.plot(np.arange(len(block)), block["leave_session_out_pc1_fraction"], marker="o", ms=3, lw=1, color=colors[name], label=labels[name])
    ax.set(xlabel="sessions ordered separately", ylabel="PC1 fraction after leaving out one session")
    ax.set_title("F  No single session creates the decomposition", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(color=GRID, zorder=0)
    fig.suptitle(
        "Matched-trajectory population decomposition",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.map_checkpoint / "selected_windows.csv")
    examples, trace_table, reconstruction = reconstruct_examples(args.cache_dir, args.source_csv, selected)
    paired, matched, samples = load_responses(args.map_checkpoint)

    trace_table.to_csv(args.out_dir / "selected_example_reconstructed_traces.csv", index=False)
    selected.to_csv(args.out_dir / "selected_examples_source_table.csv", index=False)
    np.savez_compressed(
        args.out_dir / "selected_example_images_and_paths.npz",
        **{
            f"image_{index}_{name}": value
            for index, payload in examples.items()
            for name, value in payload.items()
        },
    )
    make_construction_figure(
        selected, examples, paired, matched, samples,
        args.out_dir / "matched_trajectory_construction", args.dpi,
    )
    make_four_case_figure(
        selected, examples, paired, matched, samples,
        args.out_dir / "matched_trajectory_four_auditable_cases", args.dpi,
    )
    make_population_figure(
        selected, args.rank_checkpoint, paired, matched,
        args.out_dir / "matched_trajectory_population_decomposition", args.dpi,
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "reader-facing unpacking of the full-bank matched-trajectory response decomposition",
        "cache_dir": str(args.cache_dir.resolve()),
        "map_checkpoint": str(args.map_checkpoint.resolve()),
        "rank_checkpoint": str(args.rank_checkpoint.resolve()),
        "selection_contract": "uses the four roles selected algorithmically in checkpoint 13A before these figures were designed",
        "trajectory_reconstruction_policy": "resample_full_window",
        "trajectory_timing_warning": (
            "This production cache predates the native center-crop fix. Full source windows were resampled to 48 frames. "
            "Figures support internal own-versus-matched contrasts, not calibrated native FEM speed or temporal-frequency claims."
        ),
        "reconstruction_checks": reconstruction,
        "response_metric": "RMS in first four zero-mean temporal DCT components of FEM-minus-static response",
        "n_selected_examples": 4,
        "n_matched_trajectories_per_image": 8,
        "figures": [
            "matched_trajectory_construction.png",
            "matched_trajectory_four_auditable_cases.png",
            "matched_trajectory_population_decomposition.png",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Matched-trajectory explainer checkpoint 14\n\n"
        "These figures unpack the cached matched-trajectory comparison from construction, through four preselected "
        "examples, to the grouped population result. `matched_trajectory_construction` is the conceptual entry point; "
        "`matched_trajectory_four_auditable_cases` shows positive, negative, and dissociation cases; and "
        "`matched_trajectory_population_decomposition` bridges to checkpoint 13B.\n\n"
        "## Timing provenance warning\n\n"
        "This 384-window production cache predates the native-time trace fix and used `resample_full_window`: each full "
        "source window was resampled into 48 model frames. The reconstructed paths match cached RMS and path length "
        "exactly and are labeled accordingly. The figures support internal own-versus-matched comparisons, but the paths "
        "must not be interpreted as calibrated native-speed FEM trajectories or used for temporal-frequency calibration.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
