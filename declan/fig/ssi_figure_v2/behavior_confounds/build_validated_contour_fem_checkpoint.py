#!/usr/bin/env python3
"""Test FEM-contour alignment only in visually approved consensus patches."""

from __future__ import annotations

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


CONSENSUS_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "contour_axis_consensus_examples_v1"
)
CONSENSUS_AUDIT = CONSENSUS_DIR / "candidate_validation_audit.csv"
OUT_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "validated_contour_fem_checkpoint_v1"
)
KEYS = ["session", "trial_idx", "global_start", "phase"]
N_BOOTSTRAP = 5000
SEED = 20260810


def load_validated() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, float]]:
    full, donor_outcomes, diagnostics = load_and_score()
    audit = pd.read_csv(CONSENSUS_AUDIT)
    consensus = audit[audit.passes_strict_consensus.fillna(False)].copy()
    columns = KEYS + [
        "all_validator_max_disagreement_deg", "axis_bin_label",
        "consensus_rank_score", "sobel_square_axis_deg",
    ]
    validated = full.merge(consensus[columns], on=KEYS, how="inner", validate="one_to_one")
    validated["consensus_axis_recompute_error_deg"] = np.abs(
        ((validated.sobel_square_axis_deg - validated.image_edge_axis_deg + 90.0) % 180.0) - 90.0
    )
    return full, validated, donor_outcomes, diagnostics


def aggregate(full: pd.DataFrame, outcomes: np.ndarray, block: pd.DataFrame) -> np.ndarray:
    return _aggregate_matrix(full, outcomes, block.row_position.to_numpy(dtype=int))


def randomization_summary(
    full: pd.DataFrame, validated: pd.DataFrame, donor_outcomes: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    observed_matrix = full.observed_alignment_delta_arcmin.to_numpy(dtype=float)[None, :]
    rows = []
    nulls: dict[str, np.ndarray] = {}
    subject_values = {}
    for subject in SUBJECTS:
        block = validated[validated.subject.astype(str).eq(subject)]
        observed = float(aggregate(full, observed_matrix, block)[0])
        null = aggregate(full, donor_outcomes, block)
        subject_values[subject] = (observed, null)
        nulls[subject] = null
        rows.append(_summary_row(subject, block, observed, null))
    observed = float(np.mean([subject_values[s][0] for s in SUBJECTS]))
    null = np.mean(np.stack([subject_values[s][1] for s in SUBJECTS]), axis=0)
    nulls["equal_subject_mean"] = null
    rows.append(_summary_row("equal_subject_mean", validated, observed, null))
    return pd.DataFrame(rows), nulls


def _summary_row(subject: str, block: pd.DataFrame, observed: float, null: np.ndarray) -> dict[str, object]:
    null_mean = float(np.mean(null))
    effect = observed - null_mean
    centered = null - null_mean
    return {
        "subject": subject,
        "n_windows": int(len(block)),
        "n_trials": int(block.groupby(["session", "trial_idx"]).ngroups),
        "n_sessions": int(block.session.nunique()) if subject != "equal_subject_mean" else int(block.groupby(["subject", "session"]).ngroups),
        "observed_arcmin": observed,
        "matched_null_mean_arcmin": null_mean,
        "observed_minus_matched_arcmin": effect,
        "null_q025_arcmin": float(np.quantile(null, 0.025)),
        "null_q975_arcmin": float(np.quantile(null, 0.975)),
        "p_one_sided_positive": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "p_two_sided_centered": float((1 + np.sum(np.abs(centered) >= abs(effect))) / (len(null) + 1)),
    }


def hierarchical_bootstrap(validated: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    subject_draws = {}
    rows = []
    for subject in SUBJECTS:
        block = validated[validated.subject.astype(str).eq(subject)]
        sessions = [g.paired_residual_arcmin.to_numpy(float) for _, g in block.groupby("session", sort=False)]
        point = float(np.median([np.median(x) for x in sessions]))
        draws = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            chosen = rng.integers(0, len(sessions), len(sessions))
            session_points = []
            for j in chosen:
                values = sessions[int(j)]
                session_points.append(np.median(values[rng.integers(0, len(values), len(values))]))
            draws[i] = np.median(session_points)
        subject_draws[subject] = draws
        rows.append({"subject": subject, "paired_residual_arcmin": point,
                     "ci95_low": np.quantile(draws, .025), "ci95_high": np.quantile(draws, .975)})
    grand_draws = np.mean(np.stack([subject_draws[s] for s in SUBJECTS]), axis=0)
    rows.append({"subject": "equal_subject_mean",
                 "paired_residual_arcmin": np.mean([r["paired_residual_arcmin"] for r in rows]),
                 "ci95_low": np.quantile(grand_draws, .025), "ci95_high": np.quantile(grand_draws, .975)})
    return pd.DataFrame(rows)


def render(validated: pd.DataFrame, summary: pd.DataFrame, nulls: dict[str, np.ndarray]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    ax = axes[0]
    for subject in SUBJECTS:
        b = validated[validated.subject.astype(str).eq(subject)].sort_values("absolute_contour_axis_deg")
        ax.scatter(b.absolute_contour_axis_deg, b.observed_alignment_delta_arcmin, s=25,
                   color=SUBJECT_COLORS[subject], alpha=.75, label=f"{subject} observed")
        ax.scatter(b.absolute_contour_axis_deg, b.matched_null_mean_delta_arcmin, s=18,
                   facecolors="none", edgecolors=SUBJECT_COLORS[subject], alpha=.65)
    ax.axhline(0, color="0.35", lw=1)
    ax.set(xlabel="validated local contour axis (deg)", ylabel="parallel − orthogonal RMS (arcmin)",
           title="A  Validated windows\nfilled observed; open matched expectation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    null = nulls["equal_subject_mean"]
    observed = float(summary.loc[summary.subject.eq("equal_subject_mean"), "observed_arcmin"].iloc[0])
    ax.hist(null, bins=24, color="#8B9299", alpha=.8)
    ax.axvline(observed, color="#D1495B", lw=2.2, label=f"observed {observed:+.3f}")
    ax.axvline(np.mean(null), color="#202124", lw=1.5, ls="--", label=f"null mean {np.mean(null):+.3f}")
    ax.set(xlabel="hierarchical statistic (arcmin)", ylabel="matched reassignments",
           title="B  Exact matched real-trajectory null")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    order = [*SUBJECTS, "equal_subject_mean"]
    s = summary.set_index("subject").loc[order]
    effects = s.observed_minus_matched_arcmin.to_numpy(float)
    centered_low = s.null_q025_arcmin.to_numpy(float) - s.matched_null_mean_arcmin.to_numpy(float)
    centered_high = s.null_q975_arcmin.to_numpy(float) - s.matched_null_mean_arcmin.to_numpy(float)
    y = np.arange(len(order))
    ax.errorbar(effects, y, xerr=np.vstack([-centered_low, centered_high]), fmt="o",
                color="#2A9D8F", ecolor="#777777", capsize=3)
    ax.axvline(0, color="0.35", lw=1)
    ax.set_yticks(y, ["Allen", "Logan", "equal-subject mean"])
    ax.invert_yaxis()
    ax.set(xlabel="observed − matched expectation (arcmin)", title="C  Pairing-specific effect")
    ax.grid(axis="x", alpha=.2)
    fig.suptitle("FEM alignment in visually approved strict-consensus contour patches", weight="bold")
    path = OUT_DIR / "validated_contour_fem_checkpoint.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full, validated, donors, diagnostics = load_validated()
    summary, nulls = randomization_summary(full, validated, donors)
    bootstrap = hierarchical_bootstrap(validated)
    validated.to_csv(OUT_DIR / "validated_windows_with_fem.csv", index=False)
    summary.to_csv(OUT_DIR / "matched_reassignment_summary.csv", index=False)
    bootstrap.to_csv(OUT_DIR / "paired_residual_bootstrap.csv", index=False)
    np.savez_compressed(OUT_DIR / "matched_null_distributions.npz", **nulls)
    figure = render(validated, summary, nulls)
    metadata = {
        "stage": "post-visual-checkpoint targeted behavioral test",
        "selection_is_outcome_blind": True,
        "consensus_audit": str(CONSENSUS_AUDIT.relative_to(ROOT)),
        "n_strict_consensus": 51,
        "n_matched_to_existing_donor_bank": int(len(validated)),
        "n_unmatched_excluded": 2,
        "donor_contract": "existing exact Figure 4F bank; 256 same-session/phase matched different-trial real trajectories per target",
        "aggregation": "window median -> trial median -> session median; equal mean of subject statistics",
        "diagnostics": diagnostics,
        "max_consensus_axis_recompute_error_deg": float(validated.consensus_axis_recompute_error_deg.max()),
        "figure": str(figure.relative_to(ROOT)),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(figure)


if __name__ == "__main__":
    main()
