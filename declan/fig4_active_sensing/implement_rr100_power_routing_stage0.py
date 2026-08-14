#!/usr/bin/env python3
"""Implement Stage 0 quarantine and export clean grating-only tuning."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from declan.fig4_active_sensing.spectral_cache_contract import (
    SpectralCacheContractError,
    SupersededSpectralCacheError,
    sha256,
    validate_artifact_not_superseded,
    validate_grating_only_tuning,
    validate_spectral_cache,
)


ROOT = Path(__file__).resolve().parents[2]
INVALID = ROOT / "outputs/fig4_active_sensing/rr100_corrected_three_round_spectral_cache_v1"
MIXED_TUNING = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_stage0_quarantine_v1"
CLEAN = ROOT / "outputs/fig4_active_sensing/rr100_grating_only_orientation_tuning_v1"
INVALID_NAME = INVALID.name
ALLOWED_AUDIT = ROOT / "declan/fig4_active_sensing/audit_rr100_corrected_spectral_row_alignment.py"
CODE_SUFFIXES = {".py", ".md", ".sh", ".ipynb"}
TEXT_SUFFIXES = {".json", ".md", ".csv", ".txt", ".yaml", ".yml"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def text_mentions(path: Path, needles: tuple[str, ...]) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(needle in text for needle in needles)


def inventory_code() -> pd.DataFrame:
    known_consumers = (
        "declan/fig4_active_sensing/prepare_rr100_power_routing_data.py",
        "declan/fig4_active_sensing/make_rr100_power_routing_figure01_input.py",
        "declan/fig4_active_sensing/make_rr100_orientation_routing_input_checkpoint.py",
        "declan/fig4_active_sensing/make_rr100_orientation_aware_f0_map_checkpoint.py",
        "declan/fig4_active_sensing/analyze_rr100_orientation_aware_population_checkpoint.py",
        "declan/fig4_active_sensing/make_rr100_phase_surrogate_input_checkpoint.py",
        "declan/fig4_active_sensing/make_rr100_natural_image_rf_local_oriented_power_checkpoint.py",
        "declan/fig4_active_sensing/analyze_rr100_natural_image_rf_local_oriented_power_expanded_clean_history.py",
        "declan/fig4_active_sensing/analyze_rr100_natural_image_rf_local_oriented_power_population_clean_history.py",
    )
    rows = [{
        "path": relative,
        "kind": "analysis_consumer",
        "permitted_use": "must_use_validated_replacement_cache",
    } for relative in known_consumers]
    rows.append({
        "path": str(ALLOWED_AUDIT.relative_to(ROOT)),
        "kind": "historical_alignment_audit",
        "permitted_use": "read_only_alignment_audit",
    })
    return pd.DataFrame(rows)


def inventory_artifacts() -> pd.DataFrame:
    needles = (INVALID_NAME, sha256(INVALID / "condition_spectra.npz"))
    rows = []
    for path in sorted((ROOT / "outputs/fig4_active_sensing").rglob("*")):
        if OUT in path.parents or CLEAN in path.parents or path.name == "SUPERSEDED.json":
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES and text_mentions(path, needles):
            artifact_dir = path.parent
            rows.append({
                "artifact_dir": str(artifact_dir.relative_to(ROOT)),
                "evidence_file": str(path.relative_to(ROOT)),
                "kind": "invalid_source_cache" if artifact_dir == INVALID else "derived_or_audit_artifact",
            })
    known_derived = (
        "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1",
        "outputs/fig4_active_sensing/rr100_orientation_aware_routing_input_checkpoint_v1",
        "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1",
        "outputs/fig4_active_sensing/rr100_orientation_aware_population_checkpoint_v1",
        "outputs/fig4_active_sensing/rr100_phase_surrogate_input_checkpoint_40_v1",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_input_checkpoint_v1",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_input_checkpoint_v2",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_input_checkpoint_v3_clean_history",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_input_checkpoint_v3_clean_history_original_reference",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v1",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v2_clean_history",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v2_clean_history_original_reference",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_expanded_n100_clean_history_v1",
        "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_population_n100x61_clean_history_v1",
    )
    existing = {row["artifact_dir"] for row in rows}
    for relative in known_derived:
        if (ROOT / relative).is_dir() and relative not in existing:
            rows.append({
                "artifact_dir": relative,
                "evidence_file": "known_consumer_output_contract",
                "kind": "derived_or_audit_artifact",
            })
    return pd.DataFrame(rows).drop_duplicates("artifact_dir").sort_values("artifact_dir").reset_index(drop=True)


def marker_payload(permitted_use: str) -> dict[str, object]:
    return {
        "created_utc": utc_now(),
        "status": "superseded_do_not_use_for_scientific_inference",
        "reason": (
            "condition spectra were appended in image-grouped order while saved identity arrays were in "
            "matrix-row order; 2,980 of 3,000 rows were misidentified"
        ),
        "permitted_use": permitted_use,
        "replacement": "pending frozen replacement-cohort 100-round corrected spectral cache",
        "authoritative_audit": (
            "outputs/fig4_active_sensing/rr100_corrected_spectral_row_alignment_audit_v1/manifest.json"
        ),
    }


def export_clean_tuning() -> dict[str, object]:
    CLEAN.mkdir(parents=True, exist_ok=True)
    source_npz = MIXED_TUNING / "orientation_aware_f0_tuning_and_routing.npz"
    destination = CLEAN / "grating_only_orientation_tuning.npz"
    clean_keys = (
        "rr100_index", "measured_sf_cpd", "measured_tf_hz",
        "measured_grating_orientation_deg", "measured_signed_f0_hz",
        "measured_positive_f0_hz", "heldout_harmonic_prediction_f0_hz",
        "heldout_separable_prediction_f0_hz", "heldout_chosen_prediction_f0_hz",
    )
    with np.load(source_npz, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in clean_keys}
        source_array_hashes = {
            key: hashlib.sha256(np.ascontiguousarray(arrays[key]).view(np.uint8)).hexdigest()
            for key in clean_keys
        }
    np.savez_compressed(destination, **arrays)
    array_audit_rows = []
    with np.load(destination, allow_pickle=False) as exported:
        for key, source_array in arrays.items():
            exported_array = np.asarray(exported[key])
            exact = np.array_equal(source_array, exported_array, equal_nan=True)
            array_audit_rows.append({
                "array": key,
                "shape": "x".join(map(str, source_array.shape)),
                "dtype": str(source_array.dtype),
                "exact_source_match": exact,
            })
    array_audit = pd.DataFrame(array_audit_rows)
    array_audit.to_csv(CLEAN / "grating_only_array_audit.csv", index=False)
    if not bool(array_audit.exact_source_match.all()):
        raise RuntimeError("Clean tuning export differs from its grating-only source arrays")

    fit_columns = (
        "rr100_index", "harmonic_cv_r2", "separable_cv_r2",
        "harmonic_minus_separable_cv_r2", "harmonic_gamma", "harmonic_alpha",
        "separable_gamma", "separable_alpha", "chosen_orientation_model",
        "chosen_orientation_model_cv_r2", "recorded_validation_pass",
        "recorded_sf_curve_r_full_support",
    )
    fit = pd.read_csv(MIXED_TUNING / "orientation_tuning_fit_quality_and_movie_overlap.csv")
    fit.loc[:, [column for column in fit_columns if column in fit]].to_csv(
        CLEAN / "grating_tuning_fit_and_recorded_validation.csv", index=False
    )
    validation = validate_grating_only_tuning(destination)
    manifest = {
        "created_utc": utc_now(),
        "status": "validated_grating_only_tuning_complete",
        "scope": validation,
        "provenance": {
            "primary_fixed_retina_grating_sources": [
                "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_native_production_v1/native_condition_unit_summary.csv",
                "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_native_extended_tf_32_60_v1/native_condition_unit_summary.csv",
            ],
            "recorded_sf_validation": "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/sf_half_recorded_validated_unit_assignments.csv",
            "extraction_source": str(source_npz.relative_to(ROOT)),
            "extraction_source_sha256": sha256(source_npz),
            "source_array_sha256": source_array_hashes,
            "extraction_note": (
                "Only arrays computed from fixed-retina grating responses before the creator opens the "
                "natural-movie spectral cache are copied."
            ),
        },
        "excluded_mixed_fields": [
            "movie_sf_cpd", "movie_tf_hz", "movie_fourier_orientation_deg", "movie_power",
            "smoothed_radial_f0_weight", "normalized_orientation_factor",
            "orientation_aware_f0_weight", "chosen_orientation_model",
            "radial_accepted_power_map", "orientation_aware_accepted_power_map",
        ],
        "artifacts": {
            "arrays": str(destination.relative_to(ROOT)),
            "fit_and_recorded_validation": str(
                (CLEAN / "grating_tuning_fit_and_recorded_validation.csv").relative_to(ROOT)
            ),
            "exact_array_audit": str((CLEAN / "grating_only_array_audit.csv").relative_to(ROOT)),
        },
    }
    write_json(CLEAN / "manifest.json", manifest)
    return manifest


def smoke_test_consumers() -> pd.DataFrame:
    """Execute every consumer against the quarantined cache and require rejection."""
    modules = (
        ("prepare_rr100_power_routing_data", "environment"),
        ("make_rr100_power_routing_figure01_input", "environment"),
        ("make_rr100_orientation_routing_input_checkpoint", "environment"),
        ("make_rr100_orientation_aware_f0_map_checkpoint", "environment"),
        ("analyze_rr100_orientation_aware_population_checkpoint", "environment"),
        ("make_rr100_phase_surrogate_input_checkpoint", "--spectral-cache"),
        ("make_rr100_natural_image_rf_local_oriented_power_checkpoint", "--spectral-dir"),
        ("analyze_rr100_natural_image_rf_local_oriented_power_expanded_clean_history", "--spectral-dir"),
        ("analyze_rr100_natural_image_rf_local_oriented_power_population_clean_history", "--spectral-dir"),
    )
    rows = []
    for module, mode in modules:
        command = [sys.executable, "-m", f"declan.fig4_active_sensing.{module}"]
        environment = os.environ.copy()
        if mode == "environment":
            environment["RR100_CORRECTED_SPECTRAL_CACHE"] = str(INVALID)
        else:
            command.extend([mode, str(INVALID)])
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        rejected = completed.returncode != 0 and "Refusing superseded spectral cache" in combined
        rows.append({
            "consumer": f"declan/fig4_active_sensing/{module}.py",
            "configuration_mode": mode,
            "exit_code": completed.returncode,
            "superseded_cache_rejected": rejected,
            "last_output_line": combined.strip().splitlines()[-1] if combined.strip() else "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    code = inventory_code()
    artifacts = inventory_artifacts()
    code.to_csv(OUT / "code_consumer_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    artifacts.to_csv(OUT / "derived_artifact_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    write_json(INVALID / "SUPERSEDED.json", marker_payload("read_only_alignment_audit"))
    for row in artifacts.itertuples(index=False):
        artifact_dir = ROOT / row.artifact_dir
        if artifact_dir == INVALID or "alignment_audit" in str(artifact_dir):
            continue
        write_json(artifact_dir / "SUPERSEDED.json", marker_payload("historical_display_only_not_inference"))

    clean_manifest = export_clean_tuning()
    validate_artifact_not_superseded(CLEAN, label="clean grating-only tuning")
    mixed_tuning_rejected = False
    try:
        validate_artifact_not_superseded(MIXED_TUNING, label="mixed movie-routing tuning")
    except SpectralCacheContractError:
        mixed_tuning_rejected = True
    smoke = smoke_test_consumers()
    smoke.to_csv(OUT / "consumer_fail_fast_smoke_test.csv", index=False)
    fail_fast = False
    failure_message = ""
    try:
        validate_spectral_cache(INVALID)
    except SupersededSpectralCacheError as error:
        fail_fast = True
        failure_message = str(error)
    historical = validate_spectral_cache(INVALID, allow_superseded_for_audit=True)
    consumers = code[code.kind.eq("analysis_consumer")]
    validator_names = ("validate_spectral_cache", "validated_spectral_cache_from_environment")
    source_contract_present = consumers.path.map(
        lambda relative: any(
            name in (ROOT / relative).read_text(encoding="utf-8", errors="ignore")
            for name in validator_names
        )
    )
    gate = {
        "created_utc": utc_now(),
        "status": "stage0_complete" if fail_fast and mixed_tuning_rejected and bool(source_contract_present.all()) and bool(smoke.superseded_cache_rejected.all()) else "stage0_incomplete",
        "code_consumers": int(len(consumers)),
        "all_consumers_call_shared_validator": bool(source_contract_present.all()),
        "all_consumers_reject_superseded_cache_in_smoke_test": bool(smoke.superseded_cache_rejected.all()),
        "affected_artifact_directories": int(len(artifacts)),
        "invalid_cache_rejected": fail_fast,
        "mixed_movie_routing_tuning_rejected": mixed_tuning_rejected,
        "rejection_message": failure_message,
        "historical_audit_access": historical,
        "grating_only_tuning": clean_manifest["scope"],
    }
    write_json(OUT / "stage0_gate.json", gate)
    print(json.dumps(gate, indent=2))
    if gate["status"] != "stage0_complete":
        raise RuntimeError("Stage 0 gate is incomplete; inspect stage0_gate.json")


if __name__ == "__main__":
    main()
