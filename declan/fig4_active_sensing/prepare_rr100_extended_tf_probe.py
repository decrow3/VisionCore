#!/usr/bin/env python3
"""Freeze the executable RR100 fixed-retina high-TF F0 probe request."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_extended_tf_probe_manifest_checkpoint_33_v6"
PROBE_OUT = ROOT / "outputs/active_sensing_movie_information/backimage_rr100_extended_tf_f0_probe_v1"
RUNNER = ROOT / "declan/active_sensing_movie_information/run_backimage_rr100_dense_sf_tf_grating_probe.py"
SOURCE_DIR = ROOT / "outputs/active_sensing_movie_information/backimage_rr100_instantaneous_unit_maps_latest_v1"
CURRENT_FITS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/manifest.json"
RR100_VERSION = "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid"

SPATIAL_CPD = (1.0, 1.41421356, 2.0, 2.82842712, 4.0, 5.65685425, 8.0, 11.3137085)
TEMPORAL_HZ = (32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 60.0)
ORIENTATION_DEG = (0.0, 45.0, 90.0, 135.0)
DIRECTION_SIGNS = (-1, 1)
N_PHASES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--probe-out-dir", type=Path, default=PROBE_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def csv_list(values: tuple[float, ...] | tuple[int, ...]) -> str:
    return ",".join(f"{float(value):.10g}" for value in values)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen TF request: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    rows = []
    pair_id = 0
    for sf in SPATIAL_CPD:
        for tf in TEMPORAL_HZ:
            rows.append(
                {
                    "pair_id": pair_id,
                    "spatial_cpd": sf,
                    "temporal_hz": tf,
                    "is_extended_tf_core": bool(32.0 < tf <= 56.0),
                    "is_overlap_anchor_32hz": bool(np.isclose(tf, 32.0)),
                    "is_nyquist_edge_control": bool(np.isclose(tf, 60.0)),
                    "derived_speed_dps": tf / sf,
                }
            )
            pair_id += 1
    pairs = pd.DataFrame(rows)
    pair_path = args.out_dir / "extended_tf_pair_table.csv"
    pairs.to_csv(pair_path, index=False)

    dry_run_pair_path = args.probe_out_dir / "dense_sf_tf_pair_table.csv"
    dry_run_validation: dict[str, object] = {
        "status": "not_yet_run",
        "pair_table_match": False,
    }
    if dry_run_pair_path.exists():
        actual = pd.read_csv(dry_run_pair_path)
        columns = [
            "pair_id",
            "spatial_cpd",
            "temporal_hz",
            "is_extended_tf_core",
            "is_nyquist_edge_control",
        ]
        if list(actual[columns].columns) != columns or len(actual) != len(pairs):
            raise RuntimeError("Extended-TF dry-run pair table has the wrong columns or row count")
        numeric_match = np.array_equal(
            actual[["pair_id", "spatial_cpd", "temporal_hz"]].to_numpy(),
            pairs[["pair_id", "spatial_cpd", "temporal_hz"]].to_numpy(),
        )
        flag_match = actual[["is_extended_tf_core", "is_nyquist_edge_control"]].equals(
            pairs[["is_extended_tf_core", "is_nyquist_edge_control"]]
        )
        if not (numeric_match and flag_match):
            raise RuntimeError("Extended-TF dry-run pair table does not match the frozen request")
        dry_run_validation = {
            "status": "passed",
            "pair_table_match": True,
            "actual_pair_table": file_identity(dry_run_pair_path),
        }

    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "declan.active_sensing_movie_information.run_backimage_rr100_dense_sf_tf_grating_probe",
        "--source-dir",
        str(SOURCE_DIR),
        "--out-dir",
        str(args.probe_out_dir),
        "--rr100-version",
        RR100_VERSION,
        "--orientation-deg",
        csv_list(ORIENTATION_DEG),
        "--cycle-valid-spatial-cpds",
        csv_list(SPATIAL_CPD),
        "--subcycle-control-spatial-cpds",
        "",
        "--no-include-subcycle-controls",
        "--temporal-hz",
        csv_list(TEMPORAL_HZ),
        "--max-temporal-hz",
        "60",
        f"--temporal-direction-signs={csv_list(DIRECTION_SIGNS)}",
        "--n-phases",
        str(N_PHASES),
        "--include-blank-reference",
        "--scalar-readout",
        "center_pixel",
        "--duration-s",
        "3",
        "--frame-rate-hz",
        "120",
        "--n-lags",
        "32",
        "--discard-frames",
        "32",
        "--image-size",
        "101",
        "--ppd",
        "37.50476617",
        "--contrast",
        "0.8",
        "--window-sigma-frac",
        "0.28",
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.batch_size)),
    ]
    dry_run_command = [*command, "--dry-run"]
    expected_stimuli = len(SPATIAL_CPD) * len(TEMPORAL_HZ) * len(ORIENTATION_DEG) * len(DIRECTION_SIGNS) * N_PHASES
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "fixed_retina_extended_tf_probe_dry_run_validated_not_launched"
            if bool(dry_run_validation["pair_table_match"])
            else "fixed_retina_extended_tf_probe_executable_not_launched"
        ),
        "scientific_contract": {
            "primary_response": "positive F0 in Hz: phase-averaged center-pixel mean rate minus matched mean-gray blank, clipped at zero",
            "secondary_responses": ["signed blank-relative F0", "dynamic F1 amplitude"],
            "spatial_cpd": list(SPATIAL_CPD),
            "temporal_hz": list(TEMPORAL_HZ),
            "overlap_anchor_hz": 32.0,
            "extended_core_hz": [34.0, 56.0],
            "nyquist_edge_control_hz": 60.0,
            "orientations_deg": list(ORIENTATION_DEG),
            "temporal_direction_signs": list(DIRECTION_SIGNS),
            "direction_folding": True,
            "phase_radians": [float(2.0 * np.pi * index / N_PHASES) for index in range(N_PHASES)],
            "phase_reliability_outputs": [
                "phase_signed_f0_sd_hz",
                "phase_signed_f0_range_hz",
                "phase_signed_f0_cv_abs",
            ],
            "duration_s": 3.0,
            "frame_rate_hz": 120.0,
            "n_lags": 32,
            "discard_frames": 32,
            "image_size_px": 101,
            "ppd": 37.50476617,
            "contrast": 0.8,
            "window_sigma_frac": 0.28,
            "stimulus_normalization": "standardize_uint_like_then_minus_127_div_255",
        },
        "expected_counts": {
            "sf_tf_pairs": int(len(pairs)),
            "grating_movies": int(expected_stimuli),
            "blank_movies": 1,
            "raw_unit_rows": int(expected_stimuli * 100),
            "direction_folded_surface_rows": int(len(pairs) * 100),
        },
        "sources": {
            "runner": file_identity(RUNNER),
            "current_through_32hz_fit_manifest": file_identity(CURRENT_FITS),
            "source_orientation_dir": str(SOURCE_DIR.resolve()),
        },
        "outputs": {
            "probe_directory": str(args.probe_out_dir.resolve()),
            "frozen_pair_table": file_identity(pair_path),
        },
        "commands": {
            "dry_run": shlex.join(dry_run_command),
            "gpu_run": shlex.join(command),
        },
        "gates": [
            "dry-run pair table exactly matches the frozen 112-row pair table",
            "inspect raw F0 maps/traces for selected low-, mid-, and high-TF units before refitting",
            "require finite blank-relative F0 and acceptable phase reliability",
            "keep 60 Hz labeled as a phase-degenerate Nyquist-edge control",
            "write updated fits to a new path without overwriting <=32 Hz models",
        ],
        "dry_run_validation": dry_run_validation,
        "gpu_scoring": "prepared but not authorized or launched by this manifest",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "run_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    (args.out_dir / "dry_run_command.txt").write_text(shlex.join(dry_run_command) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
