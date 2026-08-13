#!/usr/bin/env python3
"""Render additions from relaxing orthogonal-energy purity from 10% to 12%."""

from __future__ import annotations

import json

import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_contour_axis_consensus_examples import (
    ROOT,
    analyze_candidates,
    load_candidates,
    render_subject,
)
from declan.fig.ssi_figure_v2.behavior_confounds.build_vertical_purity_refinement_checkpoint import orthogonal_fraction


SOURCE_AUDIT = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_relaxed_fourier_prefilter_checkpoint_v1/expanded_prefilter_validation_audit.csv.gz"
)
PRIOR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_relaxed_purity_checkpoint_v2/all_relaxed_purity_windows.csv"
)
OUT_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_purity12_checkpoint_v1"
)
KEYS = ["subject", "session", "trial_idx", "global_start", "phase"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_AUDIT)
    keep = (
        source.analysis_error.fillna("").eq("")
        & source.sobel_square_coherence.ge(0.55)
        & source.orthogonal_cardinal_energy_fraction.le(0.12)
        & source.all_validator_max_disagreement_deg.le(15.0)
        & source.nearest_aligned_line_distance_radius.le(0.50)
        & source.lsd_aligned_weight_fraction.ge(0.50)
    )
    selected_source = source[keep]
    candidates = load_candidates(
        0, (90.0,), unique_trials=False,
        spectrum_anisotropy_min=0.10, spectrum_disagreement_max_deg=15.0,
    )
    candidates = selected_source[KEYS].merge(candidates, on=KEYS, validate="one_to_one")
    audit, arrays = analyze_candidates(candidates)
    audit["orthogonal_cardinal_energy_fraction"] = [
        orthogonal_fraction(arrays[int(row.candidate_index)], float(row.sobel_square_axis_deg))
        for row in audit.itertuples(index=False)
    ]
    prior = pd.read_csv(PRIOR)
    selected = audit.merge(prior[KEYS].assign(was_in_prior=True), on=KEYS, how="left", validate="one_to_one")
    selected["was_in_prior"] = selected.was_in_prior.fillna(False).astype(bool)
    new = selected[~selected.was_in_prior].copy()
    new = new.sort_values(["subject", "consensus_rank_score"], ascending=[True, False])
    new["display_rank_within_cell"] = new.groupby("subject").cumcount() + 1
    new["selection_role"] = "new_from_orthogonal_energy_10_to_12_percent"
    selected.to_csv(OUT_DIR / "all_72_purity12_windows.csv", index=False)
    new.to_csv(OUT_DIR / "all_12_newly_admitted_windows.csv", index=False)
    paths = [
        render_subject(subject, new, arrays, OUT_DIR, title="new windows from 12% orthogonal-energy purity gate")
        for subject in ("Allen", "Logan") if (new.subject.astype(str) == subject).any()
    ]
    support = pd.DataFrame([
        {"subject": subject, "prior": int((prior.subject.astype(str) == subject).sum()),
         "expanded": int((selected.subject.astype(str) == subject).sum()),
         "new": int((new.subject.astype(str) == subject).sum())}
        for subject in ("Allen", "Logan")
    ])
    support.to_csv(OUT_DIR / "purity12_support.csv", index=False)
    metadata = {
        "stage": "targeted visual checkpoint of modest purity-threshold relaxation",
        "selection_is_outcome_blind": True,
        "coherence_min_unchanged": 0.55,
        "orthogonal_cardinal_energy_fraction_old": 0.10,
        "orthogonal_cardinal_energy_fraction_new": 0.12,
        "all_other_60_window_tier_gates_unchanged": True,
        "known_user_flagged_grid_orthogonal_fractions": [0.150, 0.128, 0.171],
        "n_selected": int(len(selected)), "n_new": int(len(new)),
        "figures": [str(path.relative_to(ROOT)) for path in paths],
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(support.to_string(index=False))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
