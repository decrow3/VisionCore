#!/usr/bin/env python3
"""Behavior of all fixation windows passing refined vertical-contour purity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_panel_f_exact_matched_pair_reassignment import (
    ROOT,
    SUBJECTS,
    SUBJECT_COLORS,
    _aggregate_matrix,
    load_and_score,
)


DEFAULT_COHORT = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_purity_all_fixations_v1/all_vertical_fixation_purity_audit.csv.gz"
)
OUT_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_purity_behavior_checkpoint_v1"
)
KEYS = ["session", "trial_idx", "global_start", "phase"]
N_BOOTSTRAP = 10000
SEED = 20260810


def hierarchical_bootstrap(block: pd.DataFrame, column: str, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    sessions = []
    for _session, session in block.groupby("session", sort=False):
        trials = [trial[column].to_numpy(float) for _, trial in session.groupby("trial_idx", sort=False)]
        sessions.append(trials)
    point = float(np.median([np.median([np.median(windows) for windows in trials]) for trials in sessions]))
    draws = np.empty(N_BOOTSTRAP)
    for draw in range(N_BOOTSTRAP):
        chosen_sessions = rng.integers(0, len(sessions), len(sessions))
        session_values = []
        for session_index in chosen_sessions:
            trials = sessions[int(session_index)]
            chosen_trials = rng.integers(0, len(trials), len(trials))
            trial_values = []
            for trial_index in chosen_trials:
                windows = trials[int(trial_index)]
                chosen_windows = rng.integers(0, len(windows), len(windows))
                trial_values.append(np.median(windows[chosen_windows]))
            session_values.append(np.median(trial_values))
        draws[draw] = np.median(session_values)
    return point, draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--filter-column", default="passes_refined_vertical_purity",
                        help="Boolean inclusion column, or 'none' when every input row is included")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    purity = pd.read_csv(args.cohort_csv)
    if args.filter_column != "none":
        purity = purity[purity[args.filter_column].fillna(False)].copy()
    full, donors, donor_diagnostics = load_and_score()
    validated = full.merge(
        purity[KEYS + ["orthogonal_cardinal_energy_fraction", "sobel_square_coherence"]],
        on=KEYS, how="inner", validate="one_to_one",
    )
    if len(validated) != len(purity):
        raise RuntimeError("Not every refined-purity window maps to the donor bank")
    validated["screen_horizontal_minus_vertical_arcmin"] = 60.0 * (
        np.sqrt(np.maximum(validated.cov_xx_deg2_match, 0.0))
        - np.sqrt(np.maximum(validated.cov_yy_deg2_match, 0.0))
    )
    validated.to_csv(args.out_dir / "validated_vertical_windows_with_behavior.csv", index=False)

    observed_matrix = full.observed_alignment_delta_arcmin.to_numpy(float)[None, :]
    subject_results = {}
    rows = []
    bootstrap_rows = []
    bootstrap_draws = {}
    rng = np.random.default_rng(SEED)
    for subject in SUBJECTS:
        block = validated[validated.subject.astype(str).eq(subject)]
        positions = block.row_position.to_numpy(int)
        observed = float(_aggregate_matrix(full, observed_matrix, positions)[0])
        null = _aggregate_matrix(full, donors, positions)
        horizontal_bias, draws = hierarchical_bootstrap(block, "screen_horizontal_minus_vertical_arcmin", rng)
        bootstrap_draws[subject] = draws
        subject_results[subject] = (observed, null, horizontal_bias)
        rows.append(summary_row(subject, block, observed, null, horizontal_bias))
        bootstrap_rows.append({
            "subject": subject, "screen_horizontal_minus_vertical_arcmin": horizontal_bias,
            "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
        })

    observed = float(np.mean([subject_results[s][0] for s in SUBJECTS]))
    null = np.mean(np.stack([subject_results[s][1] for s in SUBJECTS]), axis=0)
    horizontal_bias = float(np.mean([subject_results[s][2] for s in SUBJECTS]))
    grand_draws = np.mean(np.stack([bootstrap_draws[s] for s in SUBJECTS]), axis=0)
    rows.append(summary_row("equal_subject_mean", validated, observed, null, horizontal_bias))
    bootstrap_rows.append({
        "subject": "equal_subject_mean", "screen_horizontal_minus_vertical_arcmin": horizontal_bias,
        "ci95_low": float(np.quantile(grand_draws, .025)), "ci95_high": float(np.quantile(grand_draws, .975)),
    })
    summary = pd.DataFrame(rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary.to_csv(args.out_dir / "vertical_behavior_summary.csv", index=False)
    bootstrap.to_csv(args.out_dir / "horizontal_bias_bootstrap.csv", index=False)
    np.savez_compressed(
        args.out_dir / "matched_null_distributions.npz",
        **{s: subject_results[s][1] for s in SUBJECTS}, equal_subject_mean=null,
    )
    figure = render(validated, summary, bootstrap, null, args.out_dir)
    metadata = {
        "stage": "post-visual-approval behavioral summary",
        "cohort_is_image_only_and_frozen": True,
        "cohort_csv": str(args.cohort_csv),
        "filter_column": args.filter_column,
        "n_windows": int(len(validated)),
        "n_trials": int(validated.groupby(["session", "trial_idx"]).ngroups),
        "n_sessions": int(validated.groupby(["subject", "session"]).ngroups),
        "aggregation": "window median -> trial median -> session median; equal mean across subjects",
        "matched_null": "256 exact same-session/phase matched different-trial real-trajectory reassignments per target",
        "donor_diagnostics": donor_diagnostics,
        "figure": str(figure.relative_to(ROOT)),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(figure)


def summary_row(subject: str, block: pd.DataFrame, observed: float, null: np.ndarray, horizontal_bias: float) -> dict[str, object]:
    null_mean = float(np.mean(null))
    effect = observed - null_mean
    centered = null - null_mean
    return {
        "subject": subject,
        "n_windows": int(len(block)),
        "n_trials": int(block.groupby(["session", "trial_idx"]).ngroups),
        "n_sessions": int(block.groupby(["subject", "session"]).ngroups),
        "screen_horizontal_minus_vertical_arcmin": horizontal_bias,
        "observed_contour_parallel_minus_orthogonal_arcmin": observed,
        "matched_null_mean_arcmin": null_mean,
        "observed_minus_matched_arcmin": effect,
        "matched_null_q025_arcmin": float(np.quantile(null, .025)),
        "matched_null_q975_arcmin": float(np.quantile(null, .975)),
        "p_one_sided_contour_following": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "p_two_sided_pairing": float((1 + np.sum(np.abs(centered) >= abs(effect))) / (len(null) + 1)),
    }


def render(
    validated: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    grand_null: np.ndarray,
    out_dir: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    ax = axes[0]
    for i, subject in enumerate(SUBJECTS):
        values = validated.loc[validated.subject.astype(str).eq(subject), "screen_horizontal_minus_vertical_arcmin"]
        jitter = np.linspace(-.10, .10, len(values)) if len(values) > 1 else np.zeros(len(values))
        ax.scatter(i + jitter, values, color=SUBJECT_COLORS[subject], alpha=.72, s=26)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(range(2), SUBJECTS)
    ax.set(ylabel="screen horizontal − vertical RMS (arcmin)", title="A  Individual validated windows")

    ax = axes[1]
    order = [*SUBJECTS, "equal_subject_mean"]
    b = bootstrap.set_index("subject").loc[order]
    y = np.arange(3)
    point = b.screen_horizontal_minus_vertical_arcmin.to_numpy(float)
    ax.errorbar(point, y, xerr=np.vstack([point - b.ci95_low, b.ci95_high - point]), fmt="o",
                color="#2A9D8F", ecolor="#777777", capsize=3)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_yticks(y, ["Allen", "Logan", "equal-subject mean"])
    ax.invert_yaxis()
    ax.set(xlabel="horizontal − vertical RMS (arcmin)", title="B  Hierarchical horizontal bias\n95% bootstrap interval")

    ax = axes[2]
    grand = summary[summary.subject.eq("equal_subject_mean")].iloc[0]
    ax.hist(grand_null, bins=24, color="#8B9299", alpha=.8)
    ax.axvline(grand.observed_contour_parallel_minus_orthogonal_arcmin, color="#D1495B", lw=2,
               label=f"observed {grand.observed_contour_parallel_minus_orthogonal_arcmin:+.3f}")
    ax.axvline(grand.matched_null_mean_arcmin, color="#202124", ls="--", lw=1.5,
               label=f"matched mean {grand.matched_null_mean_arcmin:+.3f}")
    ax.set(xlabel="contour-parallel − orthogonal RMS (arcmin)", ylabel="matched reassignments",
           title="C  Does the correct pairing add alignment?")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Behavior in all refined-purity vertical-contour fixation windows", weight="bold")
    path = out_dir / "vertical_purity_behavior_checkpoint.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
