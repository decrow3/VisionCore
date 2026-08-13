#!/usr/bin/env python3
"""Simple marginal-preserving pairing test for current Figure 4F.

Keep every recorded FEM covariance and every coherent local contour axis.
For each FEM window, draw a local contour axis from a different trial in the
same session and fixation phase. This samples the directly observed marginal
contour distribution and changes only the local pairing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
INPUT = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_f_simple_pairing_permutation_v1"
)
SUBJECTS = ("Allen", "Logan")


def rms_delta_arcmin(
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


def load_windows(threshold: float) -> pd.DataFrame:
    columns = [
        "subject",
        "session",
        "trial_idx",
        "phase",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
    ]
    values = pd.read_csv(INPUT, usecols=columns)
    numeric = [
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
    ]
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna(subset=columns).copy()
    values = values[values["image_orientation_coherence"].ge(threshold)].copy()
    return values.sort_values(["subject", "session", "trial_idx", "phase"]).reset_index(drop=True)


def hierarchy_groups(values: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    trial_groups = [
        np.asarray(index, dtype=int)
        for index in values.groupby(["session", "trial_idx"], sort=False).indices.values()
    ]
    trial_sessions = np.asarray(
        [str(values.iloc[index[0]]["session"]) for index in trial_groups], dtype=object
    )
    sessions = pd.unique(trial_sessions)
    session_groups = [np.flatnonzero(trial_sessions == session) for session in sessions]
    return trial_groups, session_groups, sessions


def aggregate(outcomes: np.ndarray, trial_groups: list[np.ndarray], session_groups: list[np.ndarray]) -> np.ndarray:
    if outcomes.ndim == 1:
        outcomes = outcomes[None, :]
    trial_values = np.stack(
        [np.median(outcomes[:, index], axis=1) for index in trial_groups], axis=1
    )
    session_values = np.stack(
        [np.median(trial_values[:, index], axis=1) for index in session_groups], axis=1
    )
    return np.median(session_values, axis=1)


def run_test(
    values: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cxx = values["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = values["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = values["cov_yy_deg2"].to_numpy(dtype=float)
    edge = values["image_edge_axis_deg"].to_numpy(dtype=float)
    trial = values["trial_idx"].to_numpy(dtype=int)
    blocks = [
        np.asarray(index, dtype=int)
        for index in values.groupby(["session", "phase"], sort=False).indices.values()
    ]

    subject_results: dict[str, tuple[float, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        subject_positions = np.flatnonzero(values["subject"].astype(str).eq(subject).to_numpy())
        subject_values = values.iloc[subject_positions].reset_index(drop=True)
        trial_groups, session_groups, sessions = hierarchy_groups(subject_values)
        observed_window = rms_delta_arcmin(
            cxx[subject_positions], cxy[subject_positions], cyy[subject_positions], edge[subject_positions]
        )
        observed = float(aggregate(observed_window, trial_groups, session_groups)[0])

        local_lookup = np.full(len(values), -1, dtype=int)
        local_lookup[subject_positions] = np.arange(len(subject_positions), dtype=int)
        subject_blocks = [block for block in blocks if values.iloc[block[0]]["subject"] == subject]
        null = np.empty(n_permutations, dtype=float)
        for permutation_index in range(n_permutations):
            shuffled_edge = edge[subject_positions].copy()
            for block in subject_blocks:
                local = local_lookup[block]
                block_trials = trial[block]
                for target_trial in np.unique(block_trials):
                    target_local = local[block_trials == target_trial]
                    donor_pool = block[block_trials != target_trial]
                    if donor_pool.size == 0:
                        raise RuntimeError("A session/phase block has only one trial")
                    donors = rng.choice(donor_pool, size=len(target_local), replace=True)
                    shuffled_edge[target_local] = edge[donors]
            shuffled_window = rms_delta_arcmin(
                cxx[subject_positions], cxy[subject_positions], cyy[subject_positions], shuffled_edge
            )
            null[permutation_index] = aggregate(shuffled_window, trial_groups, session_groups)[0]

        subject_results[subject] = (observed, null)
        effect = observed - float(np.mean(null))
        centered = null - float(np.mean(null))
        rows.append(
            {
                "scope": "subject",
                "subject": subject,
                "n_windows": int(len(subject_positions)),
                "n_trials": int(subject_values.groupby(["session", "trial_idx"]).ngroups),
                "n_sessions": int(len(sessions)),
                "observed_arcmin": observed,
                "permuted_mean_arcmin": float(np.mean(null)),
                "pairing_excess_arcmin": effect,
                "permuted_q025_arcmin": float(np.quantile(null, 0.025)),
                "permuted_q975_arcmin": float(np.quantile(null, 0.975)),
                "p_one_sided_positive": float((1 + np.sum(null >= observed)) / (n_permutations + 1)),
                "p_two_sided_centered": float(
                    (1 + np.sum(np.abs(centered) >= abs(effect))) / (n_permutations + 1)
                ),
            }
        )

    observed = float(np.mean([subject_results[subject][0] for subject in SUBJECTS]))
    null = np.mean(np.stack([subject_results[subject][1] for subject in SUBJECTS]), axis=0)
    effect = observed - float(np.mean(null))
    centered = null - float(np.mean(null))
    rows.append(
        {
            "scope": "equal_animal_mean",
            "subject": "Allen+Logan",
            "n_windows": int(len(values)),
            "n_trials": int(values.groupby(["subject", "session", "trial_idx"]).ngroups),
            "n_sessions": int(values.groupby(["subject", "session"]).ngroups),
            "observed_arcmin": observed,
            "permuted_mean_arcmin": float(np.mean(null)),
            "pairing_excess_arcmin": effect,
            "permuted_q025_arcmin": float(np.quantile(null, 0.025)),
            "permuted_q975_arcmin": float(np.quantile(null, 0.975)),
            "p_one_sided_positive": float((1 + np.sum(null >= observed)) / (n_permutations + 1)),
            "p_two_sided_centered": float(
                (1 + np.sum(np.abs(centered) >= abs(effect))) / (n_permutations + 1)
            ),
        }
    )
    distribution = pd.DataFrame({"permutation": np.arange(n_permutations), "equal_animal_null_arcmin": null})
    return pd.DataFrame(rows), distribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coherence-threshold", type=float, default=0.3)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    values = load_windows(args.coherence_threshold)
    summary, distribution = run_test(
        values, n_permutations=args.n_permutations, seed=args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "simple_pairing_permutation_summary.csv", index=False)
    distribution.to_csv(args.out_dir / "simple_pairing_permutation_null.csv", index=False)
    row = summary[summary["scope"].eq("equal_animal_mean")].iloc[0]
    report = "\n".join(
        [
            "# Simple Figure 4F empirical-marginal pairing test",
            "",
            f"Coherence threshold: {args.coherence_threshold:g}. Windows: {int(row.n_windows)}; "
            f"trials: {int(row.n_trials)}; sessions: {int(row.n_sessions)}.",
            "",
            "The only manipulation is replacement of each local contour axis by an axis sampled "
            "from a different trial in the same session and phase. Recorded FEM covariances are "
            "unchanged, and replacement axes come directly from the empirical coherent-contour marginal.",
            "",
            f"Observed Figure 4F statistic: {row.observed_arcmin:+.4f} arcmin.",
            f"Independent-marginal expectation: {row.permuted_mean_arcmin:+.4f} arcmin.",
            f"Pairing-specific excess: {row.pairing_excess_arcmin:+.4f} arcmin "
            f"(one-sided p={row.p_one_sided_positive:.4f}; centered two-sided p={row.p_two_sided_centered:.4f}).",
            "",
            "This test asks only whether the real local pair is more aligned than expected from "
            "the observed incidence of contour orientations and FEM orientations in the same session and phase.",
        ]
    )
    (args.out_dir / "summary_report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
