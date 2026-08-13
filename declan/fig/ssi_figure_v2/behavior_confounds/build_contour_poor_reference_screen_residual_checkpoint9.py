#!/usr/bin/env python3
"""Checkpoint 9: contour-poor reference and session screen-prior residual.

The first figure compares absolute screen-frame drift-cloud profiles for
low-coherence BackImage, high-coherence BackImage, and FixRSVP in the sessions
shared by all three conditions.  The second estimates a screen-frame baseline
from each session's low-coherence BackImage trials and subtracts its prediction
from high-coherence contour-relative spread.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
SOURCE = SOURCE_ROOT / "backimage_contour_motion_component_plots_v1" / "fixrsvp_backimage_contour_motion_windows.csv"
WINDOW_FEATURES = SOURCE_ROOT / "window_features.csv"
OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "contour_poor_reference_screen_residual_checkpoint9_v1"
)

SUBJECTS = ("Allen", "Logan")
PHASES = ("mid_fixation", "late_fixation")
LOW_MAX = 0.05
HIGH_MIN = 0.30
ANGLES_DEG = np.arange(0.0, 181.0, 1.0)
BIN_CENTERS = np.asarray([0.0, 45.0, 90.0, 135.0])
BIN_LABELS = ("horizontal", "45°", "vertical", "135°")
CONDITIONS = ("low_coherence", "high_coherence", "fixrsvp")
CONDITION_LABELS = {
    "low_coherence": "BackImage\ncoherence ≤0.05",
    "high_coherence": "BackImage\ncoherence >0.30",
    "fixrsvp": "FixRSVP",
}
CONDITION_COLORS = {
    "low_coherence": "#7A5195",
    "high_coherence": "#1B7F5C",
    "fixrsvp": "#8B9299",
}
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
INK = "#202428"
GRID = "#D8DDE3"
N_BOOTSTRAP = 1500
SEED = 20260810
PRIMARY_MIN_LOW_WINDOWS = 5
PRIMARY_MIN_LOW_TRIALS = 2
SENSITIVITY_MIN_LOW_WINDOWS = 10
SENSITIVITY_MIN_LOW_TRIALS = 3


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_values() -> pd.DataFrame:
    values = pd.read_csv(SOURCE)
    gaze_columns = [
        "session", "stimulus", "trial_idx", "global_start", "global_stop",
        "local_start", "local_stop", "mean_x_deg", "mean_y_deg", "abs_mean_radius_deg",
    ]
    gaze = pd.read_csv(WINDOW_FEATURES, usecols=gaze_columns)
    keys = ["session", "stimulus", "trial_idx", "global_start", "global_stop", "local_start", "local_stop"]
    values = values.merge(gaze, on=keys, how="left", validate="one_to_one")
    values = values[
        values["subject"].isin(SUBJECTS)
        & values["phase"].isin(PHASES)
        & values["abs_mean_radius_deg"].notna()
    ].copy()
    return values.reset_index(drop=True)


def condition_blocks(values: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[str]]:
    blocks = {
        "low_coherence": values[
            values["stimulus"].eq("backimage")
            & values["image_orientation_coherence"].le(LOW_MAX)
        ].copy(),
        "high_coherence": values[
            values["stimulus"].eq("backimage")
            & values["image_orientation_coherence"].gt(HIGH_MIN)
        ].copy(),
        "fixrsvp": values[values["stimulus"].eq("fixrsvp")].copy(),
    }
    common = sorted(set.intersection(*(set(block["session"]) for block in blocks.values())))
    return (
        {
            key: block[block["session"].isin(common)].copy().reset_index(drop=True)
            for key, block in blocks.items()
        },
        common,
    )


def projected_rms(values: pd.DataFrame) -> np.ndarray:
    theta = np.radians(ANGLES_DEG)[None, :]
    ux, uy = np.cos(theta), np.sin(theta)
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)[:, None]
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)[:, None]
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)[:, None]
    variance = ux * ux * cxx + 2 * ux * uy * cxy + uy * uy * cyy
    return 60.0 * np.sqrt(np.maximum(variance, 0.0))


def trial_profile_matrices(values: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    profiles = projected_rms(values)
    matrices: dict[tuple[str, str], list[np.ndarray]] = {}
    for (subject, session, _trial), indices in values.groupby(
        ["subject", "session", "trial_idx"], sort=True
    ).groups.items():
        matrices.setdefault((str(subject), str(session)), []).append(
            np.median(profiles[np.asarray(indices, dtype=int)], axis=0)
        )
    return {key: np.stack(rows) for key, rows in matrices.items()}


def reference_profiles(
    blocks: dict[str, pd.DataFrame], common: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrices = {condition: trial_profile_matrices(block) for condition, block in blocks.items()}
    rng = np.random.default_rng(SEED)
    subject_sessions = {
        subject: [session for session in common if session.startswith(subject + "_")]
        for subject in SUBJECTS
    }
    points: dict[tuple[str, str], np.ndarray] = {}
    draws: dict[tuple[str, str], np.ndarray] = {}
    for subject in SUBJECTS:
        sessions = subject_sessions[subject]
        for condition in CONDITIONS:
            points[(subject, condition)] = np.median(
                np.stack([np.median(matrices[condition][(subject, session)], axis=0) for session in sessions]),
                axis=0,
            )
            draws[(subject, condition)] = np.empty((N_BOOTSTRAP, len(ANGLES_DEG)))
        for draw_index in range(N_BOOTSTRAP):
            chosen_sessions = rng.integers(0, len(sessions), size=len(sessions))
            for condition in CONDITIONS:
                session_profiles = []
                for chosen in chosen_sessions:
                    matrix = matrices[condition][(subject, sessions[int(chosen)])]
                    chosen_trials = rng.integers(0, len(matrix), size=len(matrix))
                    session_profiles.append(np.median(matrix[chosen_trials], axis=0))
                draws[(subject, condition)][draw_index] = np.median(np.stack(session_profiles), axis=0)

    rows = []
    summary_rows = []
    for condition in CONDITIONS:
        grand = np.mean(np.stack([points[(subject, condition)] for subject in SUBJECTS]), axis=0)
        grand_draws = np.mean(np.stack([draws[(subject, condition)] for subject in SUBJECTS]), axis=0)
        for subject in (*SUBJECTS, "equal-animal"):
            point = grand if subject == "equal-animal" else points[(subject, condition)]
            sample_draws = grand_draws if subject == "equal-animal" else draws[(subject, condition)]
            low, high = np.quantile(sample_draws, [0.025, 0.975], axis=0)
            rows.extend(
                {
                    "condition": condition,
                    "subject": subject,
                    "absolute_screen_axis_deg": angle,
                    "rms_arcmin": value,
                    "ci95_low": lower,
                    "ci95_high": upper,
                }
                for angle, value, lower, upper in zip(ANGLES_DEG, point, low, high, strict=True)
            )
            hmv = point[0] - point[90]
            hmv_draws = sample_draws[:, 0] - sample_draws[:, 90]
            preferred = int(np.argmax(point[:-1]))
            summary_rows.append(
                {
                    "condition": condition,
                    "subject": subject,
                    "preferred_screen_axis_deg": float(ANGLES_DEG[preferred]),
                    "horizontal_minus_vertical_arcmin": float(hmv),
                    "horizontal_minus_vertical_ci95_low": float(np.quantile(hmv_draws, 0.025)),
                    "horizontal_minus_vertical_ci95_high": float(np.quantile(hmv_draws, 0.975)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def full_axial(block: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = block.sort_values("absolute_screen_axis_deg")
    angle = ordered["absolute_screen_axis_deg"].to_numpy(dtype=float)
    radius = ordered[column].to_numpy(dtype=float)
    return np.radians(np.concatenate([angle, angle[1:] + 180])), np.concatenate([radius, radius[1:]])


def plot_reference(
    blocks: dict[str, pd.DataFrame], profiles: pd.DataFrame, summary: pd.DataFrame
) -> plt.Figure:
    fig = plt.figure(figsize=(11.2, 7.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.28, 0.72], hspace=0.38, wspace=0.25)
    radial_high = float(profiles["ci95_high"].max()) * 1.04
    for column, condition in enumerate(CONDITIONS):
        ax = fig.add_subplot(grid[0, column], projection="polar")
        equal = profiles[
            profiles["condition"].eq(condition) & profiles["subject"].eq("equal-animal")
        ]
        theta, radius = full_axial(equal, "rms_arcmin")
        _, low = full_axial(equal, "ci95_low")
        _, high = full_axial(equal, "ci95_high")
        color = CONDITION_COLORS[condition]
        ax.fill_between(theta, low, high, color=color, alpha=0.16, lw=0)
        ax.plot(theta, radius, color=color, lw=2.2, label="equal-animal")
        for subject in SUBJECTS:
            subject_block = profiles[
                profiles["condition"].eq(condition) & profiles["subject"].eq(subject)
            ]
            subject_theta, subject_radius = full_axial(subject_block, "rms_arcmin")
            ax.plot(subject_theta, subject_radius, color=SUBJECT_COLORS[subject], lw=0.9, alpha=0.75)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetagrids(
            [0, 45, 90, 135, 180, 225, 270, 315],
            ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
            fontsize=6.3,
        )
        ax.set_ylim(0, radial_high)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(CONDITION_LABELS[condition], pad=8, weight="semibold")
        row = summary[
            summary["condition"].eq(condition) & summary["subject"].eq("equal-animal")
        ].iloc[0]
        ax.text(
            0.5, -0.13,
            f"preferred={row.preferred_screen_axis_deg:.0f}°; H−V={row.horizontal_minus_vertical_arcmin:+.2f} arcmin",
            transform=ax.transAxes, ha="center", fontsize=7.2,
        )

    gaze_ax = fig.add_subplot(grid[1, :2])
    rng = np.random.default_rng(SEED + 1)
    gaze_rows = []
    for condition in CONDITIONS:
        block = blocks[condition]
        medians = (
            block.groupby(["subject", "session"], as_index=False)["abs_mean_radius_deg"]
            .median()
            .assign(condition=condition)
        )
        gaze_rows.append(medians)
    gaze = pd.concat(gaze_rows, ignore_index=True)
    positions = np.arange(len(CONDITIONS))
    arrays = [gaze[gaze["condition"].eq(condition)]["abs_mean_radius_deg"].to_numpy() for condition in CONDITIONS]
    boxes = gaze_ax.boxplot(arrays, positions=positions, widths=0.52, patch_artist=True, showfliers=False)
    for box, condition in zip(boxes["boxes"], CONDITIONS, strict=True):
        box.set(facecolor=CONDITION_COLORS[condition], alpha=0.16, edgecolor=CONDITION_COLORS[condition])
    for median in boxes["medians"]:
        median.set(color=INK, lw=1.2)
    for condition_index, condition in enumerate(CONDITIONS):
        block = gaze[gaze["condition"].eq(condition)]
        for subject in SUBJECTS:
            subject_values = block[block["subject"].eq(subject)]["abs_mean_radius_deg"].to_numpy()
            jitter = rng.uniform(-0.10, 0.10, size=len(subject_values))
            gaze_ax.scatter(
                condition_index + jitter, subject_values, s=15,
                color=SUBJECT_COLORS[subject], alpha=0.70, edgecolor="none",
            )
    gaze_ax.set_xticks(positions, [CONDITION_LABELS[c].replace("\n", " ") for c in CONDITIONS])
    gaze_ax.set_ylabel("session median gaze eccentricity (deg)")
    gaze_ax.set_title("D  Gaze support in the same sessions", loc="left", weight="semibold")
    gaze_ax.grid(axis="y", color=GRID, lw=0.6)
    gaze_ax.spines[["top", "right"]].set_visible(False)

    support_ax = fig.add_subplot(grid[1, 2])
    support_ax.axis("off")
    lines = ["E  Matched support", "", "All conditions: 25 sessions", "Allen 11; Logan 14", ""]
    for condition in CONDITIONS:
        block = blocks[condition]
        lines.extend(
            [
                CONDITION_LABELS[condition].replace("\n", " "),
                f"  {len(block):,} windows; {block.groupby(['session', 'trial_idx']).ngroups:,} trials",
            ]
        )
    low_counts = blocks["low_coherence"].groupby("session").size()
    lines.extend(
        [
            "",
            "Low-coherence per session",
            f"  median {low_counts.median():.0f} windows",
            f"  range {low_counts.min():.0f}–{low_counts.max():.0f}",
        ]
    )
    support_ax.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=8.1, color=INK)
    fig.suptitle(
        "Checkpoint 9A: absolute drift-cloud reference across matched sessions\n"
        "Radial origin is zero; thin lines show animals and shaded profiles show equal-animal 95% CIs",
        y=0.99, fontsize=12.0, weight="bold",
    )
    return fig


def axial_bin_index(axis_deg: np.ndarray) -> np.ndarray:
    return np.floor((np.mod(axis_deg, 180.0) + 22.5) / 45.0).astype(int) % 4


def observed_parallel_minus_orthogonal(values: pd.DataFrame) -> np.ndarray:
    theta = np.radians(np.mod(values["image_edge_axis_deg"].to_numpy(dtype=float), 180.0))
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)
    along_var = ux * ux * cxx + 2 * ux * uy * cxy + uy * uy * cyy
    across_var = vx * vx * cxx + 2 * vx * vy * cxy + vy * vy * cyy
    return 60.0 * (np.sqrt(np.maximum(along_var, 0)) - np.sqrt(np.maximum(across_var, 0)))


def interpolate_profile(profile: np.ndarray, axis_deg: np.ndarray) -> np.ndarray:
    return np.interp(np.mod(axis_deg, 180.0), ANGLES_DEG, profile)


def eligible_residual_sessions(
    low: pd.DataFrame, high: pd.DataFrame, min_windows: int, min_trials: int
) -> list[tuple[str, str]]:
    support = low.groupby(["subject", "session"]).agg(
        n_windows=("trial_idx", "size"), n_trials=("trial_idx", "nunique")
    )
    support = support[(support["n_windows"] >= min_windows) & (support["n_trials"] >= min_trials)]
    high_keys = set(zip(high["subject"].astype(str), high["session"].astype(str), strict=True))
    return [tuple(key) for key in support.index if tuple(key) in high_keys]


def residual_structures(
    values: pd.DataFrame, min_windows: int, min_trials: int
) -> tuple[dict[tuple[str, str], dict[str, object]], pd.DataFrame, pd.DataFrame]:
    low = values[
        values["stimulus"].eq("backimage")
        & values["image_orientation_coherence"].le(LOW_MAX)
    ].copy()
    high = values[
        values["stimulus"].eq("backimage")
        & values["image_orientation_coherence"].gt(HIGH_MIN)
    ].copy()
    eligible = eligible_residual_sessions(low, high, min_windows, min_trials)
    structures: dict[tuple[str, str], dict[str, object]] = {}
    support_rows = []
    window_rows = []
    for subject, session in eligible:
        low_session = low[low["session"].eq(session)].reset_index(drop=True)
        low_profiles = projected_rms(low_session)
        low_trial_profiles = []
        for _trial, indices in low_session.groupby("trial_idx", sort=True).groups.items():
            low_trial_profiles.append(np.median(low_profiles[np.asarray(indices, dtype=int)], axis=0))
        low_trial_profiles = np.stack(low_trial_profiles)
        baseline = np.median(low_trial_profiles, axis=0)

        high_session = high[high["session"].eq(session)].copy().reset_index(drop=True)
        axis = np.mod(high_session["image_edge_axis_deg"].to_numpy(dtype=float), 180.0)
        observed = observed_parallel_minus_orthogonal(high_session)
        predicted = interpolate_profile(baseline, axis) - interpolate_profile(baseline, axis + 90.0)
        high_session["absolute_axis_deg"] = axis
        high_session["canonical_bin"] = axial_bin_index(axis)
        high_session["observed_delta_arcmin"] = observed
        high_session["predicted_baseline_delta_arcmin"] = predicted
        high_session["residual_delta_arcmin"] = observed - predicted
        structures[(subject, session)] = {
            "low_trial_profiles": low_trial_profiles,
            "high": high_session,
            "high_axis": axis,
            "high_observed": observed,
            "high_bins": high_session["canonical_bin"].to_numpy(dtype=int),
            "high_trials": high_session["trial_idx"].to_numpy(dtype=int),
        }
        support_rows.append(
            {
                "subject": subject,
                "session": session,
                "n_low_windows": len(low_session),
                "n_low_trials": low_session["trial_idx"].nunique(),
                "n_high_windows": len(high_session),
                "n_high_trials": high_session["trial_idx"].nunique(),
            }
        )
        window_rows.append(
            high_session[
                [
                    "subject", "session", "trial_idx", "global_start", "global_stop",
                    "image_orientation_coherence", "absolute_axis_deg", "canonical_bin",
                    "observed_delta_arcmin", "predicted_baseline_delta_arcmin", "residual_delta_arcmin",
                ]
            ]
        )
    return structures, pd.DataFrame(support_rows), pd.concat(window_rows, ignore_index=True)


def aggregate_session(
    item: dict[str, object], baseline: np.ndarray, chosen_trials: np.ndarray | None = None
) -> np.ndarray:
    axis = np.asarray(item["high_axis"], dtype=float)
    predicted = interpolate_profile(baseline, axis) - interpolate_profile(baseline, axis + 90.0)
    observed = np.asarray(item["high_observed"], dtype=float)
    bins = np.asarray(item["high_bins"], dtype=int)
    trials = np.asarray(item["high_trials"], dtype=int)
    if chosen_trials is None:
        chosen_trials = np.unique(trials)
    trial_values = np.full((len(chosen_trials), 3, 4), np.nan)
    for trial_index, trial in enumerate(chosen_trials):
        trial_mask = trials == int(trial)
        for bin_index in range(4):
            mask = trial_mask & (bins == bin_index)
            if not np.any(mask):
                continue
            trial_values[trial_index, 0, bin_index] = np.median(observed[mask])
            trial_values[trial_index, 1, bin_index] = np.median(predicted[mask])
            trial_values[trial_index, 2, bin_index] = np.median(observed[mask] - predicted[mask])
    output = np.full((3, 4), np.nan)
    for bin_index in range(4):
        for metric_index in range(3):
            finite = trial_values[:, metric_index, bin_index]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                output[metric_index, bin_index] = np.median(finite)
    return output


def bootstrap_residual(
    structures: dict[tuple[str, str], dict[str, object]], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("observed", "predicted", "residual")
    point_subject: dict[str, np.ndarray] = {}
    draw_subject: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for subject in SUBJECTS:
        sessions = sorted(session for key_subject, session in structures if key_subject == subject)
        point_sessions = []
        for session in sessions:
            item = structures[(subject, session)]
            baseline = np.median(item["low_trial_profiles"], axis=0)
            point_sessions.append(aggregate_session(item, baseline))
        point_subject[subject] = np.nanmedian(np.stack(point_sessions), axis=0)
        draws = np.full((N_BOOTSTRAP, 3, 4), np.nan)
        for draw_index in range(N_BOOTSTRAP):
            chosen_sessions = rng.integers(0, len(sessions), size=len(sessions))
            sampled_session_values = []
            for chosen in chosen_sessions:
                item = structures[(subject, sessions[int(chosen)])]
                low_profiles = item["low_trial_profiles"]
                chosen_low = rng.integers(0, len(low_profiles), size=len(low_profiles))
                baseline = np.median(low_profiles[chosen_low], axis=0)
                trial_ids = np.unique(np.asarray(item["high_trials"], dtype=int))
                chosen_trials = rng.choice(trial_ids, size=len(trial_ids), replace=True)
                sampled_session_values.append(aggregate_session(item, baseline, chosen_trials))
            draws[draw_index] = np.nanmedian(np.stack(sampled_session_values), axis=0)
        draw_subject[subject] = draws

    rows = []
    draws_rows = []
    grand_point = np.nanmean(np.stack([point_subject[s] for s in SUBJECTS]), axis=0)
    grand_draws = np.nanmean(np.stack([draw_subject[s] for s in SUBJECTS]), axis=0)
    for subject in (*SUBJECTS, "equal-animal"):
        point = grand_point if subject == "equal-animal" else point_subject[subject]
        draws = grand_draws if subject == "equal-animal" else draw_subject[subject]
        for metric_index, metric in enumerate(metrics):
            for bin_index, label in enumerate(BIN_LABELS):
                finite = draws[:, metric_index, bin_index]
                finite = finite[np.isfinite(finite)]
                rows.append(
                    {
                        "subject": subject,
                        "metric": metric,
                        "canonical_bin": bin_index,
                        "canonical_label": label,
                        "estimate_arcmin": point[metric_index, bin_index],
                        "ci95_low": np.quantile(finite, 0.025),
                        "ci95_high": np.quantile(finite, 0.975),
                        "n_finite_bootstrap": len(finite),
                    }
                )
                draws_rows.extend(
                    {
                        "bootstrap_index": draw_index,
                        "subject": subject,
                        "metric": metric,
                        "canonical_bin": bin_index,
                        "value_arcmin": value,
                    }
                    for draw_index, value in enumerate(draws[:, metric_index, bin_index])
                )
    return pd.DataFrame(rows), pd.DataFrame(draws_rows)


def plot_residual(summary: pd.DataFrame, support: pd.DataFrame, sensitivity: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 4.3), sharey=True)
    x = np.arange(4)
    metric_titles = {
        "observed": "A  Observed high-coherence spread",
        "predicted": "B  Predicted from session baseline",
        "residual": "C  Observed minus prediction",
    }
    global_low = float(summary["ci95_low"].min())
    global_high = float(summary["ci95_high"].max())
    pad = 0.10 * (global_high - global_low)
    for ax, metric in zip(axes, ("observed", "predicted", "residual"), strict=True):
        ax.axhline(0, color="#8E959C", lw=0.8, ls=":")
        for subject in SUBJECTS:
            block = summary[summary["subject"].eq(subject) & summary["metric"].eq(metric)].sort_values("canonical_bin")
            ax.plot(x, block["estimate_arcmin"], "o-", color=SUBJECT_COLORS[subject], lw=1.0, ms=3.5, alpha=0.78)
        equal = summary[summary["subject"].eq("equal-animal") & summary["metric"].eq(metric)].sort_values("canonical_bin")
        estimate = equal["estimate_arcmin"].to_numpy()
        low = equal["ci95_low"].to_numpy()
        high = equal["ci95_high"].to_numpy()
        ax.errorbar(x, estimate, yerr=[estimate - low, high - estimate], color=INK, marker="o", ms=5, lw=1.5, capsize=2.5, label="equal-animal")
        if metric == "residual":
            strict = sensitivity[
                sensitivity["subject"].eq("equal-animal") & sensitivity["metric"].eq("residual")
            ].sort_values("canonical_bin")
            ax.plot(x, strict["estimate_arcmin"], "s--", color="#7B8187", ms=3.2, lw=0.9, label="stricter baseline support")
        ax.set_xticks(x, BIN_LABELS)
        ax.set_ylim(global_low - pad, global_high + pad)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(metric_titles[metric], loc="left", weight="semibold")
        ax.set_xlabel("absolute local contour orientation")
    axes[0].set_ylabel("parallel minus orthogonal RMS (arcmin)")
    axes[2].legend(frameon=False, fontsize=6.8, loc="best")
    counts = support.groupby("subject")["session"].nunique().to_dict()
    fig.suptitle(
        "Checkpoint 9B: subtracting the session's low-coherence screen-frame prediction\n"
        f"Primary baseline: ≥{PRIMARY_MIN_LOW_WINDOWS} windows from ≥{PRIMARY_MIN_LOW_TRIALS} trials "
        f"({counts.get('Allen', 0)} Allen, {counts.get('Logan', 0)} Logan sessions); hierarchical 95% CIs",
        y=0.995, fontsize=11.6, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, **kwargs)
        outputs[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return outputs


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = load_values()
    blocks, common = condition_blocks(values)
    reference, reference_summary = reference_profiles(blocks, common)
    reference_output = save_figure(
        plot_reference(blocks, reference, reference_summary), "contour_poor_reference_profiles"
    )
    reference.to_csv(OUT_DIR / "reference_profile_values.csv", index=False)
    reference_summary.to_csv(OUT_DIR / "reference_profile_summary.csv", index=False)

    primary_structures, primary_support, primary_windows = residual_structures(
        values, PRIMARY_MIN_LOW_WINDOWS, PRIMARY_MIN_LOW_TRIALS
    )
    primary_summary, primary_draws = bootstrap_residual(primary_structures, SEED + 20)
    strict_structures, strict_support, strict_windows = residual_structures(
        values, SENSITIVITY_MIN_LOW_WINDOWS, SENSITIVITY_MIN_LOW_TRIALS
    )
    strict_summary, strict_draws = bootstrap_residual(strict_structures, SEED + 40)
    residual_output = save_figure(
        plot_residual(primary_summary, primary_support, strict_summary),
        "screen_prior_residual_by_absolute_orientation",
    )
    primary_support.to_csv(OUT_DIR / "primary_session_baseline_support.csv", index=False)
    primary_windows.to_csv(OUT_DIR / "primary_window_residual_values.csv", index=False)
    primary_summary.to_csv(OUT_DIR / "primary_residual_summary.csv", index=False)
    primary_draws.to_csv(OUT_DIR / "primary_residual_bootstrap_draws.csv", index=False)
    strict_support.to_csv(OUT_DIR / "strict_session_baseline_support.csv", index=False)
    strict_windows.to_csv(OUT_DIR / "strict_window_residual_values.csv", index=False)
    strict_summary.to_csv(OUT_DIR / "strict_residual_summary.csv", index=False)
    strict_draws.to_csv(OUT_DIR / "strict_residual_bootstrap_draws.csv", index=False)

    equal_reference = reference_summary[reference_summary["subject"].eq("equal-animal")]
    equal_primary = primary_summary[
        primary_summary["subject"].eq("equal-animal") & primary_summary["metric"].eq("residual")
    ].sort_values("canonical_bin")
    report = [
        "# Contour-poor reference and screen-prior residual: checkpoint 9",
        "",
        "## Absolute reference",
        "",
        "All three profiles use the 25 sessions represented in low-coherence BackImage,",
        "high-coherence BackImage, and FixRSVP during the same mid/late fixation phases.",
        "FixRSVP gaze remains much nearer screen center, so it is a same-session task reference",
        "rather than a gaze-matched causal control.",
        "",
    ]
    for row in equal_reference.itertuples(index=False):
        report.append(
            f"- {CONDITION_LABELS[row.condition].replace(chr(10), ' ')}: preferred "
            f"{row.preferred_screen_axis_deg:.0f} deg; H-V {row.horizontal_minus_vertical_arcmin:+.3f} "
            f"arcmin [{row.horizontal_minus_vertical_ci95_low:+.3f}, {row.horizontal_minus_vertical_ci95_high:+.3f}]."
        )
    report.extend(
        [
            "",
            "## Session-baseline residual",
            "",
            "The baseline is the median absolute-screen RMS profile across low-coherence trials",
            "within each session. For each high-coherence window it predicts parallel-minus-orthogonal",
            "spread solely from the contour's absolute orientation. The residual is observed minus predicted.",
            "Baseline uncertainty and high-coherence trial sampling are both included in the hierarchy.",
            "",
        ]
    )
    for row in equal_primary.itertuples(index=False):
        report.append(
            f"- {row.canonical_label}: residual {row.estimate_arcmin:+.3f} arcmin "
            f"[{row.ci95_low:+.3f}, {row.ci95_high:+.3f}]."
        )
    report.extend(
        [
            "",
            "This is a diagnostic subtraction, not a causal estimate. Low-coherence and",
            "high-coherence windows can differ in gaze position, image energy, and time within fixation.",
            "The stricter baseline-support result is shown in the residual panel and saved separately.",
            "",
        ]
    )
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 9; matched absolute reference and screen-prior residual",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "window_features": str(WINDOW_FEATURES.relative_to(ROOT)),
        "window_features_sha256": sha256(WINDOW_FEATURES),
        "phases": list(PHASES),
        "low_coherence_max": LOW_MAX,
        "high_coherence_min": HIGH_MIN,
        "common_sessions": common,
        "n_bootstrap": N_BOOTSTRAP,
        "primary_baseline_support": {
            "min_windows": PRIMARY_MIN_LOW_WINDOWS,
            "min_trials": PRIMARY_MIN_LOW_TRIALS,
        },
        "strict_baseline_support": {
            "min_windows": SENSITIVITY_MIN_LOW_WINDOWS,
            "min_trials": SENSITIVITY_MIN_LOW_TRIALS,
        },
        "outputs": {"reference": reference_output, "residual": residual_output},
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / reference_output["png"])
    print(ROOT / residual_output["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()
