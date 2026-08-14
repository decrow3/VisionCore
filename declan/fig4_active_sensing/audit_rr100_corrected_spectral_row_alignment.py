#!/usr/bin/env python3
"""Audit row alignment in the corrected three-round retinal spectral cache."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_002_n003/condition_index.csv"
)
CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected_three_round_spectral_cache_v1/condition_spectra.npz"
PHASE = ROOT / "outputs/fig4_active_sensing/rr100_phase_surrogate_input_checkpoint_40_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_spectral_row_alignment_audit_v1"


def identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.resolve().read_bytes()).hexdigest()
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": digest,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {OUT}")
    OUT.mkdir(parents=True)
    conditions = pd.read_csv(CONDITIONS).sort_values("matrix_row_index").reset_index(drop=True)
    expected_rows = np.arange(len(conditions), dtype=int)
    if not np.array_equal(conditions.matrix_row_index.to_numpy(int), expected_rows):
        raise ValueError("Condition table is not matrix-row aligned")

    # Replay the loop order in build_rr100_corrected_three_round_spectral_cache.py:
    # groupby image_index, then sort each image group by matrix_row_index.
    append_order = (
        conditions.sort_values(["image_index", "matrix_row_index"])
        .matrix_row_index.to_numpy(int)
    )
    inverse_append_position = np.empty(len(append_order), dtype=int)
    inverse_append_position[append_order] = np.arange(len(append_order), dtype=int)
    aligned = append_order == expected_rows
    row_audit = pd.DataFrame(
        {
            "stored_array_position": expected_rows,
            "matrix_row_actually_appended_at_position": append_order,
            "stored_identity_matrix_row": expected_rows,
            "position_matches_stored_identity": aligned,
        }
    )
    actual = conditions.iloc[append_order].reset_index(drop=True)
    stored = conditions.reset_index(drop=True)
    for column in ("round_index", "image_index", "trace_index"):
        row_audit[f"actual_{column}"] = actual[column].to_numpy()
        row_audit[f"stored_{column}"] = stored[column].to_numpy()
    row_audit.to_csv(OUT / "spectral_cache_row_alignment_audit.csv", index=False)

    selected_rows: list[dict[str, object]] = []
    with np.load(CACHE, allow_pickle=False) as cache:
        radial = np.asarray(cache["radial_power"], dtype=np.float64)
        for path in sorted(PHASE.glob("condition_*_movies_and_power.npz")):
            condition_row = int(path.name.split("_")[1])
            with np.load(path, allow_pickle=False) as generated:
                rerendered = np.asarray(generated["radial_power_intact"], dtype=np.float64)
            claimed = radial[condition_row]
            corrected_position = int(inverse_append_position[condition_row])
            corrected = radial[corrected_position]
            denominator = max(float(np.linalg.norm(rerendered)), np.finfo(float).tiny)
            selected_rows.append(
                {
                    "matrix_row_index": condition_row,
                    "image_index": int(conditions.iloc[condition_row].image_index),
                    "trace_index": int(conditions.iloc[condition_row].trace_index),
                    "claimed_cache_position": condition_row,
                    "corrected_cache_position_from_replayed_append_order": corrected_position,
                    "claimed_position_relative_l2_error": float(
                        np.linalg.norm(rerendered - claimed) / denominator
                    ),
                    "corrected_position_relative_l2_error": float(
                        np.linalg.norm(rerendered - corrected) / denominator
                    ),
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUT / "selected_condition_rerender_validation.csv", index=False)
    corrected_pass = bool(
        len(selected) > 0
        and np.all(selected.corrected_position_relative_l2_error.to_numpy(float) < 1e-7)
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "spectral_cache_condition_alignment_invalid",
        "cause": (
            "spectra were appended in image-grouped order while identity arrays were saved in matrix-row order"
        ),
        "scope": {
            "rows": int(len(row_audit)),
            "aligned_rows": int(aligned.sum()),
            "misaligned_rows": int((~aligned).sum()),
            "misaligned_fraction": float(np.mean(~aligned)),
        },
        "independent_rerender_gate": {
            "selected_conditions": int(len(selected)),
            "corrected_lookup_matches_rerender_below_1e-7": corrected_pass,
            "maximum_corrected_lookup_relative_l2_error": (
                float(selected.corrected_position_relative_l2_error.max())
                if len(selected)
                else None
            ),
        },
        "implication": (
            "Any condition-level routing, response comparison, example selection, or cross-validation "
            "built from this cache must be regenerated after restoring matrix-row order."
        ),
        "sources": {
            "conditions": identity(CONDITIONS),
            "spectral_cache": identity(CACHE),
            "phase_checkpoint_manifest": identity(PHASE / "manifest.json"),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
