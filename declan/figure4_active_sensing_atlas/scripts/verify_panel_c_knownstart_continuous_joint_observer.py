"""Verify the Figure 4C known-start continuous-joint candidate artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.run_panel_c_knownstart_continuous_joint_observer import (
    DEFAULT_CALIBRATED_OUT_DIR,
)
from declan.figure4_active_sensing_atlas.scripts.verify_panel_c_promoted_continuous_joint_observer import (
    EXPECTED_BASIS_BY_SCALE,
    EXPECTED_RIDGE_BY_SCALE,
    EXPECTED_TEMPERATURE_BY_SCALE,
    _assert_float_maps_close,
    _json_ready,
    _normal_float_map,
    _normal_int_map,
    _score_run,
)


EXPECTED_FULL = {
    "n": 768,
    "image_accuracy": 0.70703125,
    "mean_feature_cosine": 0.937435564942103,
    "mean_true_mass": 0.5966121174243362,
}
HELDOUT_SELECTION_GATE = {
    "mean_feature_cosine": 0.936075,
    "image_accuracy": 0.70703125,
}
DEFAULT_SUMMARY_CSV = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
    / "continuous_joint_knownstart_calibrated_full_summary.csv"
)
DEFAULT_MANIFEST_JSON = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
    / "continuous_joint_knownstart_observer_manifest.json"
)


def verify(args: argparse.Namespace) -> pd.DataFrame:
    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "continuous_joint_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata JSON: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    basis_by_scale = _normal_int_map(metadata.get("basis_max_dim_by_scale", {}))
    ridge_by_scale = _normal_float_map(metadata.get("ridge_by_scale", {}))
    temp_by_scale = _normal_float_map(metadata.get("continuous_posterior_temperature_by_scale", {}))
    if basis_by_scale != EXPECTED_BASIS_BY_SCALE:
        raise AssertionError(f"basis_max_dim_by_scale differs: actual={basis_by_scale}, expected={EXPECTED_BASIS_BY_SCALE}")
    _assert_float_maps_close("ridge_by_scale", ridge_by_scale, EXPECTED_RIDGE_BY_SCALE, float(args.atol))
    _assert_float_maps_close(
        "continuous_posterior_temperature_by_scale",
        temp_by_scale,
        EXPECTED_TEMPERATURE_BY_SCALE,
        float(args.atol),
    )
    if str(metadata.get("continuous_score_mode")) != "quadratic_poisson_profile":
        raise AssertionError(f"unexpected continuous_score_mode: {metadata.get('continuous_score_mode')}")
    if str(metadata.get("trajectory_initial_position")) != "known_start":
        raise AssertionError(f"unexpected trajectory_initial_position: {metadata.get('trajectory_initial_position')}")
    if int(metadata.get("quadratic_optimizer_max_iter", -1)) != 80:
        raise AssertionError(f"unexpected quadratic_optimizer_max_iter: {metadata.get('quadratic_optimizer_max_iter')}")

    posterior, scored = _score_run(run_dir)
    continuous = posterior[posterior["observer_mode"].eq("continuous_joint")].copy()
    temp_by_scale = {
        f"{float(scale):g}": float(group["posterior_temperature"].iloc[0])
        for scale, group in continuous.groupby("prior_scale", sort=True)
    }
    summary_rows = []
    for label, group in [("all", scored), *[(f"{scale:g}", g) for scale, g in scored.groupby("prior_scale", sort=True)]]:
        temp = "mixed" if label == "all" else f"{temp_by_scale[label]:g}"
        summary_rows.append(
            {
                "prior_scale": label,
                "n": int(group.shape[0]),
                "posterior_temperature": temp,
                "image_accuracy": float(group["image_correct"].mean()),
                "mean_feature_cosine": float(group["feature_cosine"].mean()),
                "mean_true_mass": float(group["candidate_posterior_true_mass"].mean()),
                "median_neff_fraction": float(group["candidate_posterior_N_eff_fraction"].median()),
                "max_effective_score_error": float(group["max_effective_score_error"].max()),
                "max_raw_temperature_feature_mismatch": float(group["max_raw_temperature_feature_mismatch"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    all_row = summary[summary["prior_scale"].eq("all")].iloc[0]
    if args.expect_full:
        for key, expected_value in EXPECTED_FULL.items():
            actual = float(all_row[key])
            if not np.isclose(actual, float(expected_value), rtol=0.0, atol=float(args.metric_atol)):
                raise AssertionError(f"{key}={actual} differs from expected full-cache value {expected_value}")
        if int(metadata.get("n_response_tables", -1)) != int(EXPECTED_FULL["n"]):
            raise AssertionError(f"metadata n_response_tables={metadata.get('n_response_tables')} is not 768")
    if args.summary_csv is not None:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
    if args.manifest_json is not None:
        manifest = {
            "observer_slug": "noanchor_quadratic_scale_conditioned_knownstart_calibrated",
            "observer_label": "No-anchor quadratic known-start scale-calibrated",
            "status": "candidate_feature_recovery_readout",
            "artifact": {
                "run_dir": run_dir,
                "metadata_json": metadata_path,
                "feature_posterior_csv": run_dir / "continuous_joint_feature_posterior.csv",
                "summary_csv": args.summary_csv,
                "n_response_tables": int(metadata.get("n_response_tables", -1)),
            },
            "recipe": {
                "continuous_score_mode": str(metadata.get("continuous_score_mode")),
                "basis_max_dim_by_scale": basis_by_scale,
                "ridge_by_scale": ridge_by_scale,
                "continuous_posterior_temperature_by_scale": temp_by_scale,
                "quadratic_optimizer_max_iter": int(metadata.get("quadratic_optimizer_max_iter", -1)),
                "trajectory_initial_position": str(metadata.get("trajectory_initial_position")),
                "trajectory_initial_position_var": float(metadata.get("trajectory_initial_position_var", np.nan)),
            },
            "metrics": summary,
            "expected_full_metrics": EXPECTED_FULL if args.expect_full else {},
            "heldout_selection_gate": HELDOUT_SELECTION_GATE,
            "verification": {
                "passed": True,
                "expect_full": bool(args.expect_full),
                "metric_atol": float(args.metric_atol),
                "atol": float(args.atol),
                "posterior_rows": int(posterior.shape[0]),
            },
            "interpretation": (
                "Known-start improves posterior feature recovery and trajectory correlation, "
                "but uses the first measured eye-position sample, so it is a less strict "
                "no-anchor candidate than the promoted inferred-start endpoint."
            ),
        }
        Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_json).write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"verified {posterior.shape[0]} posterior rows in {run_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_CALIBRATED_OUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--expect-full", action="store_true", help="Enforce full-cache known-start metrics.")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    return parser


def main() -> None:
    verify(build_parser().parse_args())


if __name__ == "__main__":
    main()
