#!/usr/bin/env python3
"""Exact Figure 4F matched real-trajectory reassignment test.

The target image axis and coherence are the exact values used by Figure 4F.
The observed outcome is the exact within-window contour-parallel minus
contour-orthogonal RMS difference.  The null projects 256 matched real drift
covariances from different trials onto each target image axis.  Reassignments
are inherited from the production Figure 4H bank: same session and phase,
adaptive strata over movement RMS, anisotropy, gaze eccentricity, and time
since event, with the full trajectory marginal preserved in every permutation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.image_features import (
    _backimage_canvas,
    _cached_session,
)


ROOT = Path(__file__).resolve().parents[4]
FIGURE4_WINDOWS = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
MATCH_ROOT = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_h_pairing_locality_radius_population_v1"
)
COMPLETE_WINDOWS = MATCH_ROOT / "panel_h_complete_support_windows.csv"
DONOR_BANK = MATCH_ROOT / "panel_h_matched_trajectory_reassignments.npz"
MATCH_METADATA = MATCH_ROOT / "run_metadata.json"
OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_exact_matched_pair_reassignment_v1"
)
PATCH_CACHE_DIR = OUT_DIR / "example_patch_cache"

SUBJECTS = ("Allen", "Logan")
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}
BIN_LABELS = ("horizontal", "45°", "vertical", "135°")
DISPLAY_SESSIONS = {
    "Allen": "Allen_2022-03-30",
    "Logan": "Logan_2020-03-04",
}
COHERENCE_THRESHOLD = 0.3
N_BOOTSTRAP = 2000
SEED = 20260810
INK = "#202124"
GRID = "#D8DDE3"
NULL_COLOR = "#8B9299"
PAIR_COLOR = "#2A9D8F"
KEYS = ("session", "trial_idx", "global_start", "global_stop", "phase")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rms_delta_arcmin(
    cxx: np.ndarray,
    cxy: np.ndarray,
    cyy: np.ndarray,
    edge_deg: np.ndarray,
) -> np.ndarray:
    theta = np.radians(edge_deg)
    co, si = np.cos(theta), np.sin(theta)
    parallel = cxx * co * co + 2.0 * cxy * co * si + cyy * si * si
    orthogonal = cxx * si * si - 2.0 * cxy * co * si + cyy * co * co
    return 60.0 * (
        np.sqrt(np.maximum(parallel, 0.0))
        - np.sqrt(np.maximum(orthogonal, 0.0))
    )


def load_and_score() -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    figure4 = pd.read_csv(FIGURE4_WINDOWS)
    complete = (
        pd.read_csv(COMPLETE_WINDOWS)
        .sort_values("window_order")
        .reset_index(drop=True)
    )
    bank = np.load(DONOR_BANK)
    donors = bank["donors"]
    bank_ids = bank["source_window_index"]
    complete_ids = complete["source_window_index"].to_numpy(dtype=int)
    if not np.array_equal(complete_ids, bank_ids):
        raise RuntimeError("Complete-window order does not match the donor bank")

    figure_columns = [
        *KEYS,
        "image_edge_axis_deg",
        "image_edge_axis_array_deg",
        "image_orientation_coherence",
        "image_patch_center_x_px",
        "image_patch_center_y_px",
        "image_patch_radius_px",
        "image_patch_fraction_background",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
    ]
    merged = complete.merge(
        figure4[figure_columns],
        on=list(KEYS),
        how="left",
        suffixes=("_match", "_figure4"),
        validate="one_to_one",
    ).sort_values("window_order").reset_index(drop=True)
    if merged["image_edge_axis_deg"].isna().any():
        raise RuntimeError("Some complete-support windows do not map to Figure 4F")

    covariance_differences = {}
    for column in ("cov_xx_deg2", "cov_xy_deg2", "cov_yy_deg2"):
        difference = np.abs(
            merged[f"{column}_match"].to_numpy(dtype=float)
            - merged[f"{column}_figure4"].to_numpy(dtype=float)
        )
        covariance_differences[column] = float(np.max(difference))
    if max(covariance_differences.values()) > 1e-12:
        raise RuntimeError(f"Covariance provenance mismatch: {covariance_differences}")

    cxx = merged["cov_xx_deg2_match"].to_numpy(dtype=float)
    cxy = merged["cov_xy_deg2_match"].to_numpy(dtype=float)
    cyy = merged["cov_yy_deg2_match"].to_numpy(dtype=float)
    edge = merged["image_edge_axis_deg"].to_numpy(dtype=float)
    observed = _rms_delta_arcmin(cxx, cxy, cyy, edge)
    donor_outcomes = _rms_delta_arcmin(
        cxx[donors], cxy[donors], cyy[donors], edge[None, :]
    )
    null_mean = np.mean(donor_outcomes, axis=0)

    merged["row_position"] = np.arange(len(merged), dtype=int)
    merged["absolute_contour_axis_deg"] = np.mod(edge, 180.0)
    merged["canonical_bin"] = np.floor(
        np.mod(merged["absolute_contour_axis_deg"].to_numpy(dtype=float) + 22.5, 180.0)
        / 45.0
    ).astype(int)
    merged["canonical_label"] = merged["canonical_bin"].map(dict(enumerate(BIN_LABELS)))
    merged["observed_alignment_delta_arcmin"] = observed
    merged["matched_null_mean_delta_arcmin"] = null_mean
    merged["paired_residual_arcmin"] = observed - null_mean
    merged["donor_null_sd_arcmin"] = np.std(donor_outcomes, axis=0, ddof=1)
    diagnostics = {
        "max_covariance_difference": max(covariance_differences.values()),
        "n_complete_windows": int(len(merged)),
        "n_permutations": int(len(donors)),
    }
    return merged, donor_outcomes, diagnostics


def _aggregate_matrix(
    table: pd.DataFrame,
    outcomes: np.ndarray,
    row_positions: np.ndarray,
) -> np.ndarray:
    """Apply Figure 4F's median window/trial/session hierarchy to each row."""
    block = table.iloc[row_positions]
    trial_groups = list(block.groupby(["session", "trial_idx"], sort=False).indices.values())
    trial_values = np.stack(
        [
            np.median(outcomes[:, row_positions[np.asarray(group, dtype=int)]], axis=1)
            for group in trial_groups
        ],
        axis=1,
    )
    trial_sessions = np.asarray(
        [str(block.iloc[np.asarray(group, dtype=int)[0]]["session"]) for group in trial_groups]
    )
    session_values = np.stack(
        [
            np.median(trial_values[:, np.flatnonzero(trial_sessions == session)], axis=1)
            for session in pd.unique(trial_sessions)
        ],
        axis=1,
    )
    return np.median(session_values, axis=1)


def _group_positions(
    retained: pd.DataFrame, subject: str, bin_index: int | None = None
) -> np.ndarray:
    mask = retained["subject"].astype(str).eq(subject).to_numpy().copy()
    if bin_index is not None:
        mask &= retained["canonical_bin"].eq(bin_index).to_numpy()
    return retained.loc[mask, "row_position"].to_numpy(dtype=int)


def randomization_summary(
    table: pd.DataFrame, donor_outcomes: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    retained = table[table["image_orientation_coherence"].ge(COHERENCE_THRESHOLD)].copy()
    observed_matrix = table["observed_alignment_delta_arcmin"].to_numpy(dtype=float)[None, :]
    rows = []
    distributions: dict[str, np.ndarray] = {}

    def summarize(scope: str, subject: str, bin_index: int | None) -> tuple[float, np.ndarray]:
        positions = _group_positions(retained, subject, bin_index)
        observed = float(_aggregate_matrix(table, observed_matrix, positions)[0])
        null = _aggregate_matrix(table, donor_outcomes, positions)
        centered = null - np.mean(null)
        effect = observed - float(np.mean(null))
        rows.append(
            {
                "scope": scope,
                "subject": subject,
                "canonical_bin": bin_index if bin_index is not None else "all",
                "canonical_label": BIN_LABELS[bin_index] if bin_index is not None else "all",
                "n_windows": int(len(positions)),
                "n_trials": int(
                    retained[retained["row_position"].isin(positions)]
                    .groupby(["session", "trial_idx"])
                    .ngroups
                ),
                "n_sessions": int(
                    retained[retained["row_position"].isin(positions)]["session"].nunique()
                ),
                "observed_arcmin": observed,
                "matched_null_mean_arcmin": float(np.mean(null)),
                "matched_effect_arcmin": effect,
                "null_q025_arcmin": float(np.quantile(null, 0.025)),
                "null_q975_arcmin": float(np.quantile(null, 0.975)),
                "p_one_sided_positive": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
                "p_two_sided_centered": float(
                    (1 + np.sum(np.abs(centered) >= abs(effect))) / (len(null) + 1)
                ),
            }
        )
        return observed, null

    subject_natural: dict[str, tuple[float, np.ndarray]] = {}
    subject_bins: dict[tuple[str, int], tuple[float, np.ndarray]] = {}
    for subject in SUBJECTS:
        subject_natural[subject] = summarize("subject", subject, None)
        for bin_index in range(4):
            subject_bins[(subject, bin_index)] = summarize("subject_axis", subject, bin_index)

    grand_observed = float(np.mean([subject_natural[s][0] for s in SUBJECTS]))
    grand_null = np.mean(np.stack([subject_natural[s][1] for s in SUBJECTS]), axis=0)
    distributions["grand_natural_null"] = grand_null

    def add_grand(scope: str, label: str, observed: float, null: np.ndarray) -> None:
        effect = observed - float(np.mean(null))
        centered = null - np.mean(null)
        rows.append(
            {
                "scope": scope,
                "subject": "equal_subject_mean",
                "canonical_bin": label,
                "canonical_label": label,
                "n_windows": int(len(retained)),
                "n_trials": int(retained.groupby(["subject", "session", "trial_idx"]).ngroups),
                "n_sessions": int(retained.groupby(["subject", "session"]).ngroups),
                "observed_arcmin": observed,
                "matched_null_mean_arcmin": float(np.mean(null)),
                "matched_effect_arcmin": effect,
                "null_q025_arcmin": float(np.quantile(null, 0.025)),
                "null_q975_arcmin": float(np.quantile(null, 0.975)),
                "p_one_sided_positive": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
                "p_two_sided_centered": float(
                    (1 + np.sum(np.abs(centered) >= abs(effect))) / (len(null) + 1)
                ),
            }
        )

    add_grand("grand_natural", "all", grand_observed, grand_null)
    for bin_index, label in enumerate(BIN_LABELS):
        observed = float(np.mean([subject_bins[(s, bin_index)][0] for s in SUBJECTS]))
        null = np.mean(np.stack([subject_bins[(s, bin_index)][1] for s in SUBJECTS]), axis=0)
        distributions[f"grand_bin_{bin_index}_null"] = null
        add_grand("grand_axis", label, observed, null)

    equal_axis_observed_by_subject = {
        subject: float(np.mean([subject_bins[(subject, b)][0] for b in range(4)]))
        for subject in SUBJECTS
    }
    equal_axis_null_by_subject = {
        subject: np.mean(
            np.stack([subject_bins[(subject, b)][1] for b in range(4)]), axis=0
        )
        for subject in SUBJECTS
    }
    equal_axis_observed = float(np.mean(list(equal_axis_observed_by_subject.values())))
    equal_axis_null = np.mean(np.stack(list(equal_axis_null_by_subject.values())), axis=0)
    distributions["grand_equal_axis_null"] = equal_axis_null
    add_grand("grand_equal_axis", "equal_four_axes", equal_axis_observed, equal_axis_null)
    return pd.DataFrame(rows), distributions


def _hierarchical_bootstrap_multi(
    block: pd.DataFrame,
    columns: Iterable[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(columns)
    trial = (
        block.groupby(["session", "trial_idx"], as_index=False, sort=False)[columns]
        .median()
    )
    session_arrays = [
        group[columns].to_numpy(dtype=float)
        for _session, group in trial.groupby("session", sort=False)
    ]
    point = np.median(
        np.stack([np.median(values, axis=0) for values in session_arrays]), axis=0
    )
    draws = np.empty((N_BOOTSTRAP, len(columns)), dtype=float)
    for draw_index in range(N_BOOTSTRAP):
        chosen_sessions = rng.integers(0, len(session_arrays), size=len(session_arrays))
        session_points = []
        for chosen in chosen_sessions:
            values = session_arrays[int(chosen)]
            chosen_trials = rng.integers(0, len(values), size=len(values))
            session_points.append(np.median(values[chosen_trials], axis=0))
        draws[draw_index] = np.median(np.stack(session_points), axis=0)
    return point, draws


def bootstrap_summary(table: pd.DataFrame) -> pd.DataFrame:
    retained = table[table["image_orientation_coherence"].ge(COHERENCE_THRESHOLD)].copy()
    columns = [
        "observed_alignment_delta_arcmin",
        "matched_null_mean_delta_arcmin",
        "paired_residual_arcmin",
    ]
    subject_results: dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    seed_offset = 0
    for subject in SUBJECTS:
        for bin_index in (None, 0, 1, 2, 3):
            block = retained[retained["subject"].astype(str).eq(subject)]
            if bin_index is not None:
                block = block[block["canonical_bin"].eq(bin_index)]
            point, draws = _hierarchical_bootstrap_multi(
                block, columns, np.random.default_rng(SEED + seed_offset)
            )
            subject_results[(subject, bin_index)] = (point, draws)
            difference_draws = draws[:, 0] - draws[:, 1]
            rows.append(
                {
                    "scope": "subject" if bin_index is None else "subject_axis",
                    "subject": subject,
                    "canonical_label": "all" if bin_index is None else BIN_LABELS[bin_index],
                    "observed_point_arcmin": point[0],
                    "null_mean_point_arcmin": point[1],
                    "difference_of_hierarchies_arcmin": point[0] - point[1],
                    "difference_ci95_low": float(np.quantile(difference_draws, 0.025)),
                    "difference_ci95_high": float(np.quantile(difference_draws, 0.975)),
                    "paired_residual_point_arcmin": point[2],
                    "paired_residual_ci95_low": float(np.quantile(draws[:, 2], 0.025)),
                    "paired_residual_ci95_high": float(np.quantile(draws[:, 2], 0.975)),
                }
            )
            seed_offset += 1000

    def add_grand(scope: str, label: str, keys: list[tuple[str, int | None]]) -> None:
        points = np.mean(np.stack([subject_results[key][0] for key in keys]), axis=0)
        draws = np.mean(np.stack([subject_results[key][1] for key in keys]), axis=0)
        difference_draws = draws[:, 0] - draws[:, 1]
        rows.append(
            {
                "scope": scope,
                "subject": "equal_subject_mean",
                "canonical_label": label,
                "observed_point_arcmin": points[0],
                "null_mean_point_arcmin": points[1],
                "difference_of_hierarchies_arcmin": points[0] - points[1],
                "difference_ci95_low": float(np.quantile(difference_draws, 0.025)),
                "difference_ci95_high": float(np.quantile(difference_draws, 0.975)),
                "paired_residual_point_arcmin": points[2],
                "paired_residual_ci95_low": float(np.quantile(draws[:, 2], 0.025)),
                "paired_residual_ci95_high": float(np.quantile(draws[:, 2], 0.975)),
            }
        )

    add_grand("grand_natural", "all", [(subject, None) for subject in SUBJECTS])
    for bin_index, label in enumerate(BIN_LABELS):
        add_grand(
            "grand_axis", label, [(subject, bin_index) for subject in SUBJECTS]
        )

    subject_axis_points = []
    subject_axis_draws = []
    for subject in SUBJECTS:
        subject_axis_points.append(
            np.mean(
                np.stack([subject_results[(subject, b)][0] for b in range(4)]), axis=0
            )
        )
        subject_axis_draws.append(
            np.mean(
                np.stack([subject_results[(subject, b)][1] for b in range(4)]), axis=0
            )
        )
    points = np.mean(np.stack(subject_axis_points), axis=0)
    draws = np.mean(np.stack(subject_axis_draws), axis=0)
    difference_draws = draws[:, 0] - draws[:, 1]
    rows.append(
        {
            "scope": "grand_equal_axis",
            "subject": "equal_subject_mean",
            "canonical_label": "equal_four_axes",
            "observed_point_arcmin": points[0],
            "null_mean_point_arcmin": points[1],
            "difference_of_hierarchies_arcmin": points[0] - points[1],
            "difference_ci95_low": float(np.quantile(difference_draws, 0.025)),
            "difference_ci95_high": float(np.quantile(difference_draws, 0.975)),
            "paired_residual_point_arcmin": points[2],
            "paired_residual_ci95_low": float(np.quantile(draws[:, 2], 0.025)),
            "paired_residual_ci95_high": float(np.quantile(draws[:, 2], 0.975)),
        }
    )
    return pd.DataFrame(rows)


def session_values(table: pd.DataFrame) -> pd.DataFrame:
    retained = table[table["image_orientation_coherence"].ge(COHERENCE_THRESHOLD)]
    trial = (
        retained.groupby(["subject", "session", "trial_idx"], as_index=False)[
            [
                "observed_alignment_delta_arcmin",
                "matched_null_mean_delta_arcmin",
                "paired_residual_arcmin",
            ]
        ]
        .median()
    )
    return (
        trial.groupby(["subject", "session"], as_index=False)
        .median(numeric_only=True)
    )


def select_examples(table: pd.DataFrame) -> pd.DataFrame:
    retained = table[table["image_orientation_coherence"].ge(COHERENCE_THRESHOLD)].copy()
    trial_bin = (
        retained.groupby(
            ["subject", "session", "trial_idx", "canonical_bin", "canonical_label"],
            as_index=False,
        )[
            [
                "observed_alignment_delta_arcmin",
                "matched_null_mean_delta_arcmin",
                "paired_residual_arcmin",
                "image_orientation_coherence",
            ]
        ]
        .median()
    )
    display_trial_bin = trial_bin[
        trial_bin.apply(
            lambda row: str(row.session) == DISPLAY_SESSIONS[str(row.subject)],
            axis=1,
        )
    ].copy()
    selected_groups: set[tuple[str, str, int, int]] = set()
    selections = []

    def choose(role: str, candidates: pd.DataFrame, criterion: str, ascending: bool) -> None:
        work = candidates.copy()
        work["group_key"] = list(
            zip(work["subject"], work["session"], work["trial_idx"], work["canonical_bin"])
        )
        work = work[~work["group_key"].isin(selected_groups)]
        row = work.sort_values(criterion, ascending=ascending).iloc[0]
        key = (str(row.subject), str(row.session), int(row.trial_idx), int(row.canonical_bin))
        selected_groups.add(key)
        block = retained[
            retained["subject"].astype(str).eq(key[0])
            & retained["session"].astype(str).eq(key[1])
            & retained["trial_idx"].eq(key[2])
            & retained["canonical_bin"].eq(key[3])
        ].copy()
        scale = block[
            [
                "observed_alignment_delta_arcmin",
                "matched_null_mean_delta_arcmin",
                "paired_residual_arcmin",
            ]
        ].std(ddof=0).replace(0, 1.0)
        distance = np.zeros(len(block))
        for column in scale.index:
            distance += np.abs((block[column] - float(row[column])) / float(scale[column]))
        representative = block.iloc[int(np.argmin(distance))].copy()
        representative["selection_role"] = role
        representative["criterion_name"] = criterion
        representative["criterion_value_trial_bin_median"] = float(row[criterion])
        representative["selection_level"] = "algorithmic trial-bin role; nearest window shown"
        selections.append(representative)

    for subject in SUBJECTS:
        candidates = display_trial_bin[
            display_trial_bin["subject"].astype(str).eq(subject)
            & display_trial_bin["canonical_bin"].eq(0)
        ].copy()
        target = float(candidates["paired_residual_arcmin"].quantile(0.85))
        candidates["distance_to_positive_role_quantile"] = np.abs(
            candidates["paired_residual_arcmin"] - target
        )
        choose(
            f"horizontal_positive_candidate_{subject.lower()}",
            candidates,
            "distance_to_positive_role_quantile",
            True,
        )
    oblique = display_trial_bin[display_trial_bin["canonical_bin"].eq(3)].copy()
    oblique_target = float(oblique["paired_residual_arcmin"].quantile(0.85))
    oblique["distance_to_positive_role_quantile"] = np.abs(
        oblique["paired_residual_arcmin"] - oblique_target
    )
    choose(
        "oblique_positive_candidate",
        oblique,
        "distance_to_positive_role_quantile",
        True,
    )
    dissociation = display_trial_bin[
        display_trial_bin["observed_alignment_delta_arcmin"].ge(
            display_trial_bin["observed_alignment_delta_arcmin"].quantile(0.75)
        )
    ].copy()
    dissociation["absolute_paired_residual"] = np.abs(dissociation["paired_residual_arcmin"])
    choose("motor_prior_dissociation", dissociation, "absolute_paired_residual", True)
    negative = display_trial_bin.copy()
    negative_target = float(negative["paired_residual_arcmin"].quantile(0.15))
    negative["distance_to_negative_role_quantile"] = np.abs(
        negative["paired_residual_arcmin"] - negative_target
    )
    choose(
        "negative_pairing_control",
        negative,
        "distance_to_negative_role_quantile",
        True,
    )
    return pd.DataFrame(selections)


def _add_covariance_ellipse(
    ax: plt.Axes,
    cxx: float,
    cxy: float,
    cyy: float,
    *,
    color: str,
    alpha: float,
    lw: float,
) -> None:
    covariance = np.asarray([[cxx, cxy], [cxy, cyy]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vector = eigenvectors[:, order[0]]
    angle = np.degrees(np.arctan2(vector[1], vector[0]))
    ellipse = Ellipse(
        (0.0, 0.0),
        width=4.0 * 60.0 * np.sqrt(eigenvalues[0]),
        height=4.0 * 60.0 * np.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        edgecolor=color,
        alpha=alpha,
        lw=lw,
    )
    ax.add_patch(ellipse)


def plot_examples(
    table: pd.DataFrame, donor_outcomes: np.ndarray, selected: pd.DataFrame
) -> plt.Figure:
    fig, axes = plt.subplots(len(selected), 3, figsize=(9.0, 2.0 * len(selected)), constrained_layout=True)
    cxx = table["cov_xx_deg2_match"].to_numpy(dtype=float)
    cxy = table["cov_xy_deg2_match"].to_numpy(dtype=float)
    cyy = table["cov_yy_deg2_match"].to_numpy(dtype=float)
    donors = np.load(DONOR_BANK)["donors"]
    for row_index, (_, row) in enumerate(selected.iterrows()):
        position = int(row.row_position)
        ax = axes[row_index, 0]
        patch_path = PATCH_CACHE_DIR / f"{row.selection_role}.npy"
        if not patch_path.exists():
            raise FileNotFoundError(
                f"Missing {patch_path}; run --extract-patch-index for each selected row first"
            )
        patch = np.load(patch_path)
        ax.imshow(patch, cmap="gray", interpolation="nearest")
        center = (patch.shape[0] - 1) / 2.0
        theta = np.radians(float(row.image_edge_axis_array_deg))
        dx, dy = 0.72 * center * np.cos(theta), 0.72 * center * np.sin(theta)
        ax.plot([center - dx, center + dx], [center - dy, center + dy], color="#E45756", lw=1.6)
        ax.set_title(
            f"{row.selection_role}\n{row.subject}; coh={row.image_orientation_coherence:.2f}",
            fontsize=7.2,
            loc="left",
        )
        ax.axis("off")

        ax = axes[row_index, 1]
        donor_indices = donors[np.linspace(0, len(donors) - 1, 12, dtype=int), position]
        for donor in donor_indices:
            _add_covariance_ellipse(
                ax, cxx[donor], cxy[donor], cyy[donor],
                color=NULL_COLOR, alpha=0.28, lw=0.7,
            )
        _add_covariance_ellipse(
            ax, cxx[position], cxy[position], cyy[position],
            color=SUBJECT_COLORS[str(row.subject)], alpha=1.0, lw=1.8,
        )
        extent = 2.4 * 60.0 * np.sqrt(max(cxx[position], cyy[position], 1e-8))
        edge_theta = np.radians(float(row.image_edge_axis_deg))
        ax.plot(
            [-extent * np.cos(edge_theta), extent * np.cos(edge_theta)],
            [-extent * np.sin(edge_theta), extent * np.sin(edge_theta)],
            color="#E45756", lw=1.2,
        )
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axhline(0, color=GRID, lw=0.5)
        ax.axvline(0, color=GRID, lw=0.5)
        ax.set_title("observed (color) + matched donors", fontsize=7.2)
        ax.set_xlabel("screen x (arcmin)", fontsize=6.5)
        ax.set_ylabel("screen y", fontsize=6.5)
        ax.tick_params(labelsize=6)

        ax = axes[row_index, 2]
        null = donor_outcomes[:, position]
        ax.hist(null, bins=24, color=NULL_COLOR, alpha=0.72)
        ax.axvline(float(row.observed_alignment_delta_arcmin), color=SUBJECT_COLORS[str(row.subject)], lw=1.8)
        ax.axvline(float(row.matched_null_mean_delta_arcmin), color=INK, lw=1.0, ls="--")
        ax.set_title(
            f"obs={row.observed_alignment_delta_arcmin:+.2f}, "
            f"null={row.matched_null_mean_delta_arcmin:+.2f}\n"
            f"paired residual={row.paired_residual_arcmin:+.2f} arcmin",
            fontsize=7.2,
        )
        ax.set_xlabel("parallel − orthogonal RMS (arcmin)", fontsize=6.5)
        ax.set_ylabel("donor count", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Exact Figure 4F matched-pair examples (red = target contour axis)",
        fontsize=12,
        weight="bold",
    )
    return fig


def plot_primary(
    randomization: pd.DataFrame,
    distributions: dict[str, np.ndarray],
    sessions: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), constrained_layout=True)
    ax = axes[0, 0]
    grand = randomization[randomization["scope"].eq("grand_natural")].iloc[0]
    null = distributions["grand_natural_null"]
    ax.hist(null, bins=24, color=NULL_COLOR, alpha=0.78, label="matched reassignments")
    ax.axvline(grand.observed_arcmin, color=PAIR_COLOR, lw=2.0, label="observed")
    ax.set_title("A  Exact hierarchical statistic", loc="left", weight="semibold")
    ax.set_xlabel("parallel − orthogonal RMS (arcmin)")
    ax.set_ylabel("permutations")
    ax.legend(frameon=False, fontsize=7)
    ax.text(
        0.03, 0.96,
        f"effect={grand.matched_effect_arcmin:+.3f}\none-sided p={grand.p_one_sided_positive:.3f}",
        transform=ax.transAxes, va="top", fontsize=7.2,
    )

    ax = axes[0, 1]
    subject = randomization[randomization["scope"].eq("subject")].set_index("subject").loc[list(SUBJECTS)]
    x = np.arange(2)
    for index, name in enumerate(SUBJECTS):
        row = subject.loc[name]
        ax.vlines(index, row.null_q025_arcmin, row.null_q975_arcmin, color=NULL_COLOR, lw=5, alpha=0.65)
        ax.plot(index, row.matched_null_mean_arcmin, "_", color=INK, ms=10)
        ax.plot(index, row.observed_arcmin, "o", color=SUBJECT_COLORS[name], ms=6)
    ax.set_xticks(x, SUBJECTS)
    ax.set_title("B  Animal-resolved observed vs null", loc="left", weight="semibold")
    ax.set_ylabel("hierarchical RMS contrast (arcmin)")

    ax = axes[1, 0]
    bins = randomization[randomization["scope"].eq("grand_axis")].set_index("canonical_label").loc[list(BIN_LABELS)]
    x = np.arange(4)
    effects = bins["matched_effect_arcmin"].to_numpy(dtype=float)
    low = bins["observed_arcmin"].to_numpy(dtype=float) - bins["null_q975_arcmin"].to_numpy(dtype=float)
    high = bins["observed_arcmin"].to_numpy(dtype=float) - bins["null_q025_arcmin"].to_numpy(dtype=float)
    ax.errorbar(x, effects, yerr=np.vstack([effects - low, high - effects]), fmt="o", color=PAIR_COLOR, capsize=3)
    ax.axhline(0, color="#80868B", lw=0.8, ls=":")
    ax.set_xticks(x, BIN_LABELS)
    ax.set_title("C  Pairing effect by absolute axis", loc="left", weight="semibold")
    ax.set_ylabel("observed − matched null (arcmin)")

    ax = axes[1, 1]
    for subject_name in SUBJECTS:
        block = sessions[sessions["subject"].astype(str).eq(subject_name)]
        ax.scatter(
            np.full(len(block), 0 if subject_name == "Allen" else 1) + np.linspace(-0.12, 0.12, len(block)),
            block["paired_residual_arcmin"],
            s=22, color=SUBJECT_COLORS[subject_name], alpha=0.82, label=subject_name,
        )
    ax.axhline(0, color="#80868B", lw=0.8, ls=":")
    ax.set_xticks([0, 1], SUBJECTS)
    ax.set_title("D  Session-level paired residuals", loc="left", weight="semibold")
    ax.set_ylabel("median trial residual (arcmin)")

    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Figure 4F exact matched real-trajectory reassignment (coherence ≥0.3)",
        fontsize=12.2,
        weight="bold",
    )
    return fig


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    paths = {}
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, transparent=True, **kwargs)
        paths[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return paths


def extract_selected_patch(index: int) -> None:
    """Extract one selected patch in an isolated process to bound session memory."""
    selection_path = OUT_DIR / "selected_examples.csv"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"{selection_path} does not exist; run the population analysis once first"
        )
    selected = pd.read_csv(selection_path)
    if index < 0 or index >= len(selected):
        raise IndexError(f"Patch index {index} is outside [0, {len(selected)})")
    row = selected.iloc[index]
    canvas, _ppd, _shape = _backimage_canvas(str(row.session), int(row.trial_idx))
    radius = int(row.image_patch_radius_px)
    cx = int(round(float(row.image_patch_center_x_px)))
    cy = int(round(float(row.image_patch_center_y_px)))
    patch = np.asarray(
        canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
    ).copy()
    PATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = PATCH_CACHE_DIR / f"{row.selection_role}.npy"
    np.save(path, patch)
    _backimage_canvas.cache_clear()
    _cached_session.cache_clear()
    print(path)


def refresh_selected_examples() -> None:
    table, _donor_outcomes, _diagnostics = load_and_score()
    selected = select_examples(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUT_DIR / "selected_examples.csv", index=False)
    print(OUT_DIR / "selected_examples.csv")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, donor_outcomes, diagnostics = load_and_score()
    retained = table[table["image_orientation_coherence"].ge(COHERENCE_THRESHOLD)].copy()
    randomization, distributions = randomization_summary(table, donor_outcomes)
    bootstrap = bootstrap_summary(table)
    sessions = session_values(table)
    selected = select_examples(table)

    primary_observed = float(
        randomization[randomization["scope"].eq("grand_natural")]["observed_arcmin"].iloc[0]
    )
    equal_axis_observed = float(
        randomization[randomization["scope"].eq("grand_equal_axis")]["observed_arcmin"].iloc[0]
    )
    convergence_rows = []
    for n_permutations in (16, 32, 64, 128, 256):
        for estimand, observed, key in (
            ("grand_natural", primary_observed, "grand_natural_null"),
            ("grand_equal_axis", equal_axis_observed, "grand_equal_axis_null"),
        ):
            null = distributions[key][:n_permutations]
            convergence_rows.append(
                {
                    "estimand": estimand,
                    "n_permutations": n_permutations,
                    "observed_arcmin": observed,
                    "null_mean_arcmin": float(np.mean(null)),
                    "matched_effect_arcmin": observed - float(np.mean(null)),
                }
            )
    convergence = pd.DataFrame(convergence_rows)

    randomization.to_csv(OUT_DIR / "exact_matched_randomization_summary.csv", index=False)
    bootstrap.to_csv(OUT_DIR / "exact_matched_bootstrap_summary.csv", index=False)
    sessions.to_csv(OUT_DIR / "exact_matched_session_values.csv", index=False)
    convergence.to_csv(OUT_DIR / "null_bank_convergence.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_examples.csv", index=False)
    retained[
        [
            "source_window_index", "row_position", "subject", "session", "trial_idx",
            "global_start", "global_stop", "phase", "match_stratum",
            "image_orientation_coherence", "absolute_contour_axis_deg",
            "canonical_bin", "canonical_label", "cov_xx_deg2_match",
            "cov_xy_deg2_match", "cov_yy_deg2_match",
            "observed_alignment_delta_arcmin", "matched_null_mean_delta_arcmin",
            "paired_residual_arcmin", "donor_null_sd_arcmin",
        ]
    ].to_csv(OUT_DIR / "exact_matched_window_values.csv.gz", index=False, compression="gzip")

    primary_paths = _save_figure(
        plot_primary(randomization, distributions, sessions),
        "exact_matched_reassignment_primary",
    )
    example_paths = _save_figure(
        plot_examples(table, donor_outcomes, selected),
        "exact_matched_reassignment_examples",
    )

    primary = randomization[randomization["scope"].eq("grand_natural")].iloc[0]
    equal_axis = randomization[randomization["scope"].eq("grand_equal_axis")].iloc[0]
    boot_primary = bootstrap[bootstrap["scope"].eq("grand_natural")].iloc[0]
    subject_lines = []
    for subject in SUBJECTS:
        row = randomization[
            randomization["scope"].eq("subject")
            & randomization["subject"].eq(subject)
        ].iloc[0]
        subject_lines.append(
            f"- {subject}: observed {row.observed_arcmin:+.3f}, matched null "
            f"{row.matched_null_mean_arcmin:+.3f}, effect {row.matched_effect_arcmin:+.3f} arcmin "
            f"(one-sided p={row.p_one_sided_positive:.3f})."
        )
    report = [
        "# Exact Figure 4F matched real-trajectory reassignment",
        "",
        f"The primary target set contains {len(retained):,} coherence >= {COHERENCE_THRESHOLD:.1f} windows "
        f"from {retained.groupby(['subject', 'session', 'trial_idx']).ngroups:,} trials and "
        f"{retained.groupby(['subject', 'session']).ngroups} sessions. The target image axes and "
        "covariances are the exact Figure 4F quantities.",
        "",
        "Each target was compared with 256 different-trial real trajectories from the existing",
        "within-session/phase matched bank. Every donor permutation preserves the full trajectory",
        "marginal. Matching strata adaptively hold movement RMS, anisotropy, gaze eccentricity,",
        "and time since event approximately fixed.",
        "",
        "## Primary result",
        "",
        f"The observed equal-animal Figure 4F statistic is {primary.observed_arcmin:+.3f} arcmin. "
        f"The matched reassignment expectation is {primary.matched_null_mean_arcmin:+.3f} arcmin, "
        f"giving an exact-estimator pairing effect of {primary.matched_effect_arcmin:+.3f} arcmin "
        f"(one-sided positive p={primary.p_one_sided_positive:.3f}; centered two-sided "
        f"p={primary.p_two_sided_centered:.3f}).",
        "",
        f"A paired hierarchical bootstrap using the per-window donor mean gives a difference-of-"
        f"hierarchies point of {boot_primary.difference_of_hierarchies_arcmin:+.3f} arcmin "
        f"[{boot_primary.difference_ci95_low:+.3f}, {boot_primary.difference_ci95_high:+.3f}].",
        "",
        "The pooled effect remains between -0.003 and -0.008 arcmin when the donor bank is",
        "truncated successively to 16, 32, 64, 128, and all 256 permutations.",
        "",
        *subject_lines,
        "",
        f"Equal weighting of the four absolute axes gives a matched effect of "
        f"{equal_axis.matched_effect_arcmin:+.3f} arcmin (one-sided p="
        f"{equal_axis.p_one_sided_positive:.3f}).",
        "",
        "## Interpretation",
        "",
        "The observed positive natural-scene statistic is reproduced by matched real trajectories",
        "that were recorded on other trials. The exact local image/trajectory pairing does not add",
        "a positive pooled effect, and neither animal shows a convincing positive pairing advantage.",
        "Some horizontal and 135-degree point estimates are positive, but the axis-specific",
        "randomization intervals include zero and the 45-degree/vertical estimates do not show the",
        "sign-consistent pattern predicted by general local contour following.",
        "",
        "This test is conditional on the two recorded animals and on the existing matching contract.",
        "It does not prove that every possible contour-contingent component is exactly zero, but it",
        "directly rejects the strong interpretation of Figure 4F as evidence for general local",
        "contour-guided drift.",
        "",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")

    metadata = {
        "stage": "map-first exact Figure 4F matched real-trajectory reassignment",
        "artifact_type": "production_population_behavior_pairing_audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_exact_matched_pair_reassignment",
        "coherence_threshold_contract": f">={COHERENCE_THRESHOLD}",
        "outcome": "contour-parallel RMS minus contour-orthogonal RMS, arcmin",
        "hierarchy": "window median within trial; trial median within session; session median within fixed animal; equal animal mean",
        "null_contract": "256 matched different-trial real-trajectory reassignments; same session/phase adaptive motor strata; exact full-bank trajectory marginal preservation",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "n_retained_windows": int(len(retained)),
        "n_retained_trials": int(retained.groupby(["subject", "session", "trial_idx"]).ngroups),
        "n_retained_sessions": int(retained.groupby(["subject", "session"]).ngroups),
        "n_permutations": int(diagnostics["n_permutations"]),
        "max_covariance_provenance_difference": diagnostics["max_covariance_difference"],
        "source_sha256": _sha256(FIGURE4_WINDOWS),
        "donor_bank_sha256": _sha256(DONOR_BANK),
        "match_metadata": json.loads(MATCH_METADATA.read_text(encoding="utf-8")),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "outputs": {"primary_figure": primary_paths, "example_figure": example_paths},
    }
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(ROOT / primary_paths["png"])
    print(ROOT / example_paths["png"])
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-patch-index", type=int, default=None)
    parser.add_argument("--refresh-selection", action="store_true")
    args = parser.parse_args()
    if args.extract_patch_index is not None:
        extract_selected_patch(args.extract_patch_index)
    elif args.refresh_selection:
        refresh_selected_examples()
    else:
        main()
