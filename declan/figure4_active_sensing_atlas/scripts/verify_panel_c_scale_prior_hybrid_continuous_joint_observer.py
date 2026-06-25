"""Verify the Figure 4C predeclared scale-prior hybrid candidate artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.run_panel_c_scale_prior_hybrid_continuous_joint_observer import (
    DEFAULT_OUT_DIR,
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


EXPECTED_PROCESS_BY_SCALE = {"0.5": "ar1", "1": "ar1", "2": "matched_brownian"}
EXPECTED_BROWNIAN_SCALE_BY_SCALE = {"2": 8.0}
EXPECTED_FULL = {
    "n": 768,
    "image_accuracy": 0.70703125,
    "mean_feature_cosine": 0.937743445510825,
    "mean_true_mass": 0.5974108121109268,
}
HELDOUT_SELECTION_GATE = {
    "mean_feature_cosine": 0.936666,
    "image_accuracy": 0.70703125,
}
DEFAULT_SUMMARY_CSV = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
    / "continuous_joint_scale_prior_hybrid_predeclared_full_summary.csv"
)
DEFAULT_MANIFEST_JSON = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
    / "continuous_joint_scale_prior_hybrid_observer_manifest.json"
)


def _normal_str_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {f"{float(key):g}": str(item) for key, item in value.items()}


def verify(args: argparse.Namespace) -> pd.DataFrame:
    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "continuous_joint_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata JSON: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    basis_by_scale = _normal_int_map(metadata.get("basis_max_dim_by_scale", {}))
    ridge_by_scale = _normal_float_map(metadata.get("ridge_by_scale", {}))
    temp_by_scale = _normal_float_map(metadata.get("continuous_posterior_temperature_by_scale", {}))
    process_by_scale = _normal_str_map(metadata.get("trajectory_process_model_by_scale", {}))
    brownian_by_scale = _normal_float_map(metadata.get("brownian_cov_scale_by_scale", {}))
    if basis_by_scale != EXPECTED_BASIS_BY_SCALE:
        raise AssertionError(f"basis_max_dim_by_scale differs: actual={basis_by_scale}, expected={EXPECTED_BASIS_BY_SCALE}")
    _assert_float_maps_close("ridge_by_scale", ridge_by_scale, EXPECTED_RIDGE_BY_SCALE, float(args.atol))
    _assert_float_maps_close(
        "continuous_posterior_temperature_by_scale",
        temp_by_scale,
        EXPECTED_TEMPERATURE_BY_SCALE,
        float(args.atol),
    )
    if process_by_scale != EXPECTED_PROCESS_BY_SCALE:
        raise AssertionError(f"trajectory_process_model_by_scale differs: actual={process_by_scale}")
    _assert_float_maps_close("brownian_cov_scale_by_scale", brownian_by_scale, EXPECTED_BROWNIAN_SCALE_BY_SCALE, float(args.atol))
    if str(metadata.get("continuous_score_mode")) != "quadratic_poisson_profile":
        raise AssertionError(f"unexpected continuous_score_mode: {metadata.get('continuous_score_mode')}")
    if str(metadata.get("trajectory_initial_position")) != "known_start":
        raise AssertionError(f"unexpected trajectory_initial_position: {metadata.get('trajectory_initial_position')}")
    if int(metadata.get("quadratic_optimizer_max_iter", -1)) != 80:
        raise AssertionError(f"unexpected quadratic_optimizer_max_iter: {metadata.get('quadratic_optimizer_max_iter')}")

    posterior, scored = _score_run(run_dir)
    continuous = posterior[posterior["observer_mode"].eq("continuous_joint")].copy()
    process_rows = (
        continuous[["prior_scale", "trajectory_process_model", "brownian_cov_scale"]]
        .drop_duplicates()
        .sort_values("prior_scale")
    )
    expected_rows = {
        0.5: ("ar1", 1.0),
        1.0: ("ar1", 1.0),
        2.0: ("matched_brownian", 8.0),
    }
    for scale, (process_model, brownian_scale) in expected_rows.items():
        row = process_rows[np.isclose(process_rows["prior_scale"].astype(float), scale)]
        if row.empty:
            raise AssertionError(f"missing posterior rows for scale {scale:g}")
        item = row.iloc[0]
        if str(item["trajectory_process_model"]) != process_model:
            raise AssertionError(f"scale {scale:g} process model={item['trajectory_process_model']}, expected {process_model}")
        if not np.isclose(float(item["brownian_cov_scale"]), brownian_scale, rtol=0.0, atol=float(args.atol)):
            raise AssertionError(f"scale {scale:g} brownian_cov_scale={item['brownian_cov_scale']}, expected {brownian_scale}")

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
            "observer_slug": "noanchor_quadratic_scale_prior_hybrid_predeclared",
            "observer_label": "No-anchor quadratic known-start scale-prior hybrid",
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
                "trajectory_initial_position": str(metadata.get("trajectory_initial_position")),
                "trajectory_process_model_by_scale": process_by_scale,
                "brownian_cov_scale_by_scale": brownian_by_scale,
                "quadratic_optimizer_max_iter": int(metadata.get("quadratic_optimizer_max_iter", -1)),
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
                "Predeclared source rerun of the scale-specific trajectory prior: "
                "known-start AR(1) for 0.5x/1.0x and matched-Brownian scale 8 for 2.0x. "
                "It is the leading less-strict feature-primary candidate; the strict "
                "inferred-start observer remains the no-start endpoint."
            ),
        }
        Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_json).write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"verified {posterior.shape[0]} posterior rows in {run_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--expect-full", action="store_true", help="Enforce full-cache scale-prior hybrid metrics.")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    return parser


def main() -> None:
    verify(build_parser().parse_args())


if __name__ == "__main__":
    main()
