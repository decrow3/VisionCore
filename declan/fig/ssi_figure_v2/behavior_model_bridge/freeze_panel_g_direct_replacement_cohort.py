#!/usr/bin/env python3
"""Freeze the outcome-blind native-pair cohort for a replacement Panel G."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT_ROOT = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_direct_replacement_strong_contour_v1/input_checkpoint"
)
PRIMARY_MANIFEST = CHECKPOINT_ROOT / "primary_strong_contour_native_pair_manifest.csv"
ALL_MANIFEST = CHECKPOINT_ROOT / "all_native_pairs_input_qc_manifest.csv"
PRIOR_SELECTION = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_original_matrix_pair_rotation_audit_v1/frozen_pair_selection.csv"
)
OUT_ROOT = CHECKPOINT_ROOT.parent
OUT_MANIFEST = OUT_ROOT / "frozen_confirmation_cohort.csv"

# These windows were visibly more complex than the typical retained window,
# but none showed an unmistakable axis error on the outcome-blind contact sheet.
# They remain in the cohort under the predeclared conservative rejection rule.
RETAINED_REVIEW_NOTES = {
    199: "competing local structure; retained because the tensor axis remains plausible",
    456: "peripheral diagonal structure; retained because the central dominant axis remains plausible",
    541: "complex edge structure; retained because there is no unmistakable axis mismatch",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    primary = pd.read_csv(PRIMARY_MANIFEST)
    all_pairs = pd.read_csv(ALL_MANIFEST)
    prior = pd.read_csv(PRIOR_SELECTION)

    prior_source_rows = set(prior["image_source_row"].astype(int))
    prior_hashes = set(
        all_pairs.loc[
            all_pairs["source_row"].astype(int).isin(prior_source_rows),
            "stimulus_image_sha256",
        ].dropna()
    )
    if not prior_hashes:
        raise RuntimeError("Could not resolve any prior diagnostic stimulus-image hashes")

    frozen = primary.copy()
    frozen["blind_visual_axis_qc"] = "pass"
    frozen["blind_visual_contour_strength_qc"] = "pass"
    frozen["blind_visual_review_note"] = frozen["pair_index"].astype(int).map(
        RETAINED_REVIEW_NOTES
    ).fillna("")
    frozen["prior_targeted_diagnostic_image_overlap"] = frozen[
        "stimulus_image_sha256"
    ].isin(prior_hashes)
    frozen["confirmation_eligible"] = ~frozen[
        "prior_targeted_diagnostic_image_overlap"
    ]
    frozen["confirmation_exclusion_reason"] = np.where(
        frozen["prior_targeted_diagnostic_image_overlap"],
        "underlying stimulus image appeared in the prior targeted diagnostic",
        "",
    )
    frozen["selection_uses_historical_surrogate"] = False
    frozen["selection_uses_fresh_model_outcome"] = False
    frozen["selection_frozen_before_model_evaluation"] = True

    frozen = frozen.sort_values(
        ["stimulus_image_sha256", "local_window_cluster_id", "session", "trial_idx", "pair_index"],
        kind="stable",
    ).reset_index(drop=True)
    frozen.insert(0, "frozen_cohort_index", np.arange(len(frozen), dtype=int))
    frozen.to_csv(OUT_MANIFEST, index=False)

    kept = frozen[frozen["confirmation_eligible"]]
    metadata = {
        "analysis": "panel_g_direct_replacement_frozen_confirmation_cohort",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_stage": "outcome_blind_input_freeze",
        "contract": {
            "primary_automatic_gate": "all input_checkpoint primary_eligible pairs (coherence >= 0.60)",
            "blind_visual_rejection_rule": "reject only an unmistakable contour-axis or contour-strength failure",
            "n_blind_visual_rejections": 0,
            "prior_diagnostic_image_excluded_from_confirmation": True,
            "historical_surrogate_used": False,
            "fresh_model_outcome_used": False,
        },
        "counts": {
            "automatic_primary_pairs": int(len(frozen)),
            "confirmation_pairs": int(len(kept)),
            "confirmation_stimulus_images": int(kept["stimulus_image_sha256"].nunique()),
            "confirmation_display_trials": int(kept["display_trial_key"].nunique()),
            "confirmation_local_window_clusters": int(kept["local_window_cluster_id"].nunique()),
            "confirmation_sessions": int(kept["session"].nunique()),
            "confirmation_subjects": int(kept["subject"].nunique()),
            "prior_diagnostic_underlying_images": int(len(prior_hashes)),
            "excluded_prior_image_pairs": int((~frozen["confirmation_eligible"]).sum()),
        },
        "retained_review_notes": {str(key): value for key, value in RETAINED_REVIEW_NOTES.items()},
        "inputs": {
            "primary_manifest": str(PRIMARY_MANIFEST),
            "primary_manifest_sha256": _file_sha256(PRIMARY_MANIFEST),
            "all_manifest": str(ALL_MANIFEST),
            "all_manifest_sha256": _file_sha256(ALL_MANIFEST),
            "prior_targeted_selection": str(PRIOR_SELECTION),
            "prior_targeted_selection_sha256": _file_sha256(PRIOR_SELECTION),
        },
        "output_manifest": str(OUT_MANIFEST),
    }
    (OUT_ROOT / "frozen_confirmation_cohort_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "[panel-g-freeze] "
        f"kept {len(kept)}/{len(frozen)} pairs; "
        f"{kept['stimulus_image_sha256'].nunique()} images, "
        f"{kept['local_window_cluster_id'].nunique()} local-window clusters",
        flush=True,
    )
    print(f"[panel-g-freeze] wrote {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
