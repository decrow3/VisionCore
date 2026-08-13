#!/usr/bin/env python3
"""Render every fixation newly admitted by a relaxed, purity-preserving gate."""

from __future__ import annotations

import json

import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_contour_axis_consensus_examples import (
    ROOT,
    analyze_candidates,
    load_candidates,
    render_subject,
)
from declan.fig.ssi_figure_v2.behavior_confounds.build_vertical_purity_refinement_checkpoint import (
    orthogonal_fraction,
)


SOURCE_AUDIT = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_purity_all_fixations_v1/all_vertical_fixation_purity_audit.csv.gz"
)
OUT_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_relaxed_purity_checkpoint_v2"
)
KEYS = ["subject", "session", "trial_idx", "global_start", "phase"]
COHERENCE_MIN = 0.55
ORTHOGONAL_FRACTION_MAX = 0.10
ALL_VALIDATOR_DISAGREEMENT_MAX_DEG = 15.0
NEAREST_LINE_DISTANCE_MAX_RADIUS = 0.50
LSD_ALIGNED_WEIGHT_FRACTION_MIN = 0.50


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_AUDIT)
    source["passes_relaxed_purity"] = (
        source.analysis_error.fillna("").eq("")
        & source.sobel_square_coherence.ge(COHERENCE_MIN)
        & source.orthogonal_cardinal_energy_fraction.le(ORTHOGONAL_FRACTION_MAX)
        & source.all_validator_max_disagreement_deg.le(ALL_VALIDATOR_DISAGREEMENT_MAX_DEG)
        & source.nearest_aligned_line_distance_radius.le(NEAREST_LINE_DISTANCE_MAX_RADIUS)
        & source.lsd_aligned_weight_fraction.ge(LSD_ALIGNED_WEIGHT_FRACTION_MIN)
    )
    chosen = source[source.passes_relaxed_purity].copy()
    candidates = load_candidates(0, (90.0,), unique_trials=False)
    candidates = chosen[KEYS + ["passes_refined_vertical_purity"]].merge(
        candidates, on=KEYS, validate="one_to_one",
    )
    audit, arrays = analyze_candidates(candidates)
    audit["orthogonal_cardinal_energy_fraction"] = [
        orthogonal_fraction(arrays[int(row.candidate_index)], float(row.sobel_square_axis_deg))
        for row in audit.itertuples(index=False)
    ]
    audit["passes_relaxed_purity"] = (
        audit.sobel_square_coherence.ge(COHERENCE_MIN)
        & audit.orthogonal_cardinal_energy_fraction.le(ORTHOGONAL_FRACTION_MAX)
        & audit.all_validator_max_disagreement_deg.le(ALL_VALIDATOR_DISAGREEMENT_MAX_DEG)
        & audit.nearest_aligned_line_distance_radius.le(NEAREST_LINE_DISTANCE_MAX_RADIUS)
        & audit.lsd_aligned_weight_fraction.ge(LSD_ALIGNED_WEIGHT_FRACTION_MIN)
    )
    if not audit.passes_relaxed_purity.all():
        raise RuntimeError("Recomputed relaxed cohort does not reproduce source audit")
    new = audit[~candidates.passes_refined_vertical_purity.to_numpy(bool)].copy()
    new = new.sort_values(["subject", "consensus_rank_score"], ascending=[True, False])
    new["display_rank_within_cell"] = new.groupby("subject").cumcount() + 1
    new["selection_role"] = "all_newly_admitted_by_relaxation"
    audit.to_csv(OUT_DIR / "all_relaxed_purity_windows.csv", index=False)
    new.to_csv(OUT_DIR / "all_newly_admitted_windows.csv", index=False)
    paths = [
        render_subject(
            subject, new, arrays, OUT_DIR,
            title="all windows newly admitted by purity-preserving relaxation",
        )
        for subject in ("Allen", "Logan")
    ]
    support = pd.DataFrame([
        {
            "subject": subject,
            "previous_refined_strict": int(
                source.subject.astype(str).eq(subject).to_numpy()
                @ source.passes_refined_vertical_purity.fillna(False).astype(int).to_numpy()
            ),
            "relaxed_purity": int((chosen.subject.astype(str) == subject).sum()),
            "newly_admitted": int((new.subject.astype(str) == subject).sum()),
        }
        for subject in ("Allen", "Logan")
    ])
    support.to_csv(OUT_DIR / "relaxed_purity_support.csv", index=False)
    metadata = {
        "stage": "targeted visual checkpoint of threshold relaxation",
        "selection_is_outcome_blind": True,
        "all_newly_admitted_windows_are_displayed": True,
        "frozen_purity": {
            "sobel_square_coherence_min": COHERENCE_MIN,
            "orthogonal_cardinal_energy_fraction_max": ORTHOGONAL_FRACTION_MAX,
        },
        "retained_secondary_gates": {
            "all_validator_max_disagreement_deg": ALL_VALIDATOR_DISAGREEMENT_MAX_DEG,
            "nearest_aligned_line_distance_radius_max": NEAREST_LINE_DISTANCE_MAX_RADIUS,
            "lsd_aligned_weight_fraction_min": LSD_ALIGNED_WEIGHT_FRACTION_MIN,
        },
        "removed_secondary_gates": [
            "individual secondary estimator coherence minima",
            "LSD resultant/count minima",
            "multiscale minimum coherence",
            "quadrant readability/resultant/max-disagreement gates",
        ],
        "n_previous": int(source.passes_refined_vertical_purity.fillna(False).sum()),
        "n_relaxed": int(len(audit)),
        "n_newly_admitted": int(len(new)),
        "figures": [str(path.relative_to(ROOT)) for path in paths],
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(support.to_string(index=False))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
