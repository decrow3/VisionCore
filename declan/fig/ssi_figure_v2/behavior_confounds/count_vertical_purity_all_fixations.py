#!/usr/bin/env python3
"""Count refined-purity vertical contours across every fixation window."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_contour_axis_consensus_examples import (
    INPUT,
    ROOT,
    analyze_candidates,
    axial_distance_deg,
    load_candidates,
    nearest_axis_bin,
)


OUT_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_purity_all_fixations_v1"
)
COHERENCE_MIN = 0.55
ORTHOGONAL_FRACTION_MAX = 0.10


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
    raw = pd.read_csv(INPUT)
    basic = raw[
        raw.image_feature_ok.fillna(False).astype(bool)
        & raw.image_orientation_coherence.ge(0.30)
        & raw.image_patch_fraction_inside_image.ge(0.999)
        & raw.image_patch_fraction_background.le(0.05)
    ].copy()
    basic["axis_bin_center_deg"] = [nearest_axis_bin(float(x))[0] for x in basic.image_edge_axis_deg]
    basic_vertical = basic[basic.axis_bin_center_deg.eq(90.0)]

    candidates = load_candidates(0, (90.0,), unique_trials=False)
    audit, arrays = analyze_candidates(candidates)
    valid = audit.analysis_error.fillna("").eq("")
    audit["orthogonal_cardinal_energy_fraction"] = np.nan
    for row in audit[valid].itertuples(index=False):
        audit.loc[audit.candidate_index.eq(row.candidate_index), "orthogonal_cardinal_energy_fraction"] = orthogonal_fraction(
            arrays[int(row.candidate_index)], float(row.sobel_square_axis_deg)
        )
    audit["passes_refined_vertical_purity"] = (
        audit.passes_strict_consensus.fillna(False)
        & audit.sobel_square_coherence.ge(COHERENCE_MIN)
        & audit.orthogonal_cardinal_energy_fraction.le(ORTHOGONAL_FRACTION_MAX)
    )
    audit.to_csv(OUT_DIR / "all_vertical_fixation_purity_audit.csv.gz", index=False, compression="gzip")
    rows = []
    for subject in ("Allen", "Logan", "all"):
        raw_s = raw if subject == "all" else raw[raw.subject.astype(str).eq(subject)]
        basic_s = basic if subject == "all" else basic[basic.subject.astype(str).eq(subject)]
        bv_s = basic_vertical if subject == "all" else basic_vertical[basic_vertical.subject.astype(str).eq(subject)]
        cand_s = audit if subject == "all" else audit[audit.subject.astype(str).eq(subject)]
        rows.append({
            "subject": subject,
            "all_fixation_windows": len(raw_s),
            "basic_image_valid": len(basic_s),
            "basic_valid_vertical_axis": len(bv_s),
            "vertical_after_sobel_fourier_prefilter": len(cand_s),
            "complete_reconstruction": int(cand_s.analysis_error.fillna("").eq("").sum()),
            "prior_strict_consensus": int(cand_s.passes_strict_consensus.fillna(False).sum()),
            "refined_vertical_purity": int(cand_s.passes_refined_vertical_purity.sum()),
        })
    support = pd.DataFrame(rows)
    support.to_csv(OUT_DIR / "all_fixation_vertical_purity_support.csv", index=False)
    metadata = {
        "unit": "fixation window; repeated windows/trials retained",
        "selection_is_outcome_blind": True,
        "coherence_min": COHERENCE_MIN,
        "orthogonal_cardinal_energy_fraction_max": ORTHOGONAL_FRACTION_MAX,
        "all_previous_strict_consensus_gates_required": True,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(support.to_string(index=False))


if __name__ == "__main__":
    main()
