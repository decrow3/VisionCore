#!/usr/bin/env python3
"""Apply and render the provisional single-axis purity gate for vertical patches."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_contour_axis_consensus_examples import (
    ROOT,
    analyze_candidates,
    axial_distance_deg,
    load_candidates,
    render_subject,
)


SOURCE_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "contour_axis_consensus_vertical_expansion_v1"
)
OUT_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_purity_refinement_checkpoint_v1"
)
SUBJECTS = ("Allen", "Logan")
COHERENCE_MIN = 0.55
ORTHOGONAL_CARDINAL_FRACTION_MAX = 0.10


def orthogonal_fraction(arrays: dict[str, object], reference_axis_deg: float) -> float:
    image = np.asarray(arrays["patch01"], dtype=float)
    mask = np.asarray(arrays["circle_mask"], dtype=bool)
    gx = cv2.Scharr(image, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Scharr(image, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REPLICATE)
    contour = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    energy = gx * gx + gy * gy
    parallel = float(np.sum(energy[mask & (np.asarray(axial_distance_deg(contour, reference_axis_deg)) <= 15.0)]))
    orthogonal = float(np.sum(energy[mask & (np.asarray(axial_distance_deg(contour, reference_axis_deg + 90.0)) <= 15.0)]))
    return orthogonal / max(parallel + orthogonal, 1e-12)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = pd.read_csv(SOURCE_DIR / "candidate_validation_audit.csv")
    prior = prior[prior.passes_strict_consensus.fillna(False)]
    candidates = load_candidates(0, (90.0,))
    keys = ["subject", "session", "trial_idx"]
    candidates = prior[keys].merge(candidates, on=keys, validate="one_to_one")
    audit, arrays = analyze_candidates(candidates)
    audit["orthogonal_cardinal_energy_fraction"] = [
        orthogonal_fraction(arrays[int(row.candidate_index)], float(row.sobel_square_axis_deg))
        for row in audit.itertuples(index=False)
    ]
    audit["passes_refined_vertical_purity"] = (
        audit.passes_strict_consensus.fillna(False)
        & audit.sobel_square_coherence.ge(COHERENCE_MIN)
        & audit.orthogonal_cardinal_energy_fraction.le(ORTHOGONAL_CARDINAL_FRACTION_MAX)
    )
    selected = audit[audit.passes_refined_vertical_purity].copy()
    selected = selected.sort_values(["subject", "consensus_rank_score"], ascending=[True, False])
    selected["display_rank_within_cell"] = selected.groupby("subject").cumcount() + 1
    selected["selection_role"] = "all_refined_vertical_purity_survivors"
    audit.to_csv(OUT_DIR / "vertical_purity_audit.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_all_purity_survivors.csv", index=False)
    paths = [
        render_subject(
            subject, selected, arrays, OUT_DIR,
            title="all strict-consensus + single-axis-purity vertical survivors",
        )
        for subject in SUBJECTS
    ]
    support = audit.groupby("subject", as_index=False).agg(
        prior_strict=("passes_strict_consensus", "sum"),
        refined_purity=("passes_refined_vertical_purity", "sum"),
    )
    support.to_csv(OUT_DIR / "vertical_purity_support.csv", index=False)
    metadata = {
        "stage": "targeted post-grid-failure visual checkpoint",
        "selection_is_outcome_blind": True,
        "all_survivors_are_displayed": True,
        "prior_strict_source": str((SOURCE_DIR / "candidate_validation_audit.csv").relative_to(ROOT)),
        "provisional_purity_gate": {
            "sobel_square_coherence_min": COHERENCE_MIN,
            "orthogonal_energy_definition": "Scharr energy within +/-15 deg of orthogonal axis divided by energy within +/-15 deg of parallel plus orthogonal axes",
            "orthogonal_cardinal_energy_fraction_max": ORTHOGONAL_CARDINAL_FRACTION_MAX,
        },
        "n_prior_strict": int(audit.passes_strict_consensus.sum()),
        "n_refined": int(audit.passes_refined_vertical_purity.sum()),
        "figures": [str(path.relative_to(ROOT)) for path in paths],
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(support.to_string(index=False))
    print("displayed", len(selected))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
