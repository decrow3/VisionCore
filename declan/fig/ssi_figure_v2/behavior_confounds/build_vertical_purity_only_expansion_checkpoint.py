#!/usr/bin/env python3
"""Render every window added by retaining purity but dropping secondary gates."""

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
    / "vertical_purity_only_expansion_checkpoint_v1"
)
KEYS = ["subject", "session", "trial_idx", "global_start", "phase"]
COHERENCE_MIN = 0.55
ORTHOGONAL_FRACTION_MAX = 0.10
ROWS_PER_PAGE = 10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_AUDIT)
    pure = (
        source.analysis_error.fillna("").eq("")
        & source.sobel_square_coherence.ge(COHERENCE_MIN)
        & source.orthogonal_cardinal_energy_fraction.le(ORTHOGONAL_FRACTION_MAX)
    )
    prior = (
        pure
        & source.all_validator_max_disagreement_deg.le(15.0)
        & source.nearest_aligned_line_distance_radius.le(0.50)
        & source.lsd_aligned_weight_fraction.ge(0.50)
    )
    chosen = source[pure].copy()
    chosen["was_in_60_window_tier"] = prior[pure].to_numpy(bool)
    candidates = load_candidates(0, (90.0,), unique_trials=False)
    candidates = chosen[KEYS + ["was_in_60_window_tier"]].merge(
        candidates, on=KEYS, validate="one_to_one",
    )
    audit, arrays = analyze_candidates(candidates)
    audit["orthogonal_cardinal_energy_fraction"] = [
        orthogonal_fraction(arrays[int(row.candidate_index)], float(row.sobel_square_axis_deg))
        for row in audit.itertuples(index=False)
    ]
    audit["passes_purity_only"] = (
        audit.sobel_square_coherence.ge(COHERENCE_MIN)
        & audit.orthogonal_cardinal_energy_fraction.le(ORTHOGONAL_FRACTION_MAX)
    )
    if not audit.passes_purity_only.all():
        raise RuntimeError("Recomputed purity-only cohort does not reproduce source audit")
    new = audit[~candidates.was_in_60_window_tier.to_numpy(bool)].copy()
    new = new.sort_values(["subject", "consensus_rank_score"], ascending=[True, False])
    new["display_rank_within_cell"] = new.groupby("subject").cumcount() + 1
    new["selection_role"] = "all_newly_admitted_by_purity_only_tier"
    audit.to_csv(OUT_DIR / "all_95_purity_only_windows.csv", index=False)
    new.to_csv(OUT_DIR / "all_35_newly_admitted_windows.csv", index=False)

    paths = []
    for subject in ("Allen", "Logan"):
        block = new[new.subject.astype(str).eq(subject)]
        for page_index, start in enumerate(range(0, len(block), ROWS_PER_PAGE), start=1):
            page_dir = OUT_DIR / f"{subject.lower()}_page_{page_index}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page = block.iloc[start:start + ROWS_PER_PAGE]
            paths.append(render_subject(
                subject, page, arrays, page_dir,
                title=f"purity-only additions, page {page_index}",
            ))

    support = pd.DataFrame([
        {
            "subject": subject,
            "prior_60_window_tier": int((source.subject.astype(str).eq(subject) & prior).sum()),
            "purity_only": int((source.subject.astype(str).eq(subject) & pure).sum()),
            "newly_admitted": int((new.subject.astype(str) == subject).sum()),
            "new_unique_trials": int(
                new[new.subject.astype(str).eq(subject)].groupby(["session", "trial_idx"]).ngroups
            ),
        }
        for subject in ("Allen", "Logan")
    ])
    support.to_csv(OUT_DIR / "purity_only_support.csv", index=False)
    metadata = {
        "stage": "targeted visual checkpoint of maximum frozen-purity cohort",
        "selection_is_outcome_blind": True,
        "all_newly_admitted_windows_are_displayed": True,
        "retained_gates": {
            "sobel_square_coherence_min": COHERENCE_MIN,
            "orthogonal_cardinal_energy_fraction_max": ORTHOGONAL_FRACTION_MAX,
            "initial_sobel_fourier_prefilter_inherited": True,
        },
        "secondary_gates_removed": "all post-purity estimator-agreement, line-support, line-distance, multiscale, and quadrant gates",
        "n_prior": int(prior.sum()),
        "n_purity_only": int(pure.sum()),
        "n_new": int(len(new)),
        "figures": [str(path.relative_to(ROOT)) for path in paths],
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(support.to_string(index=False))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
