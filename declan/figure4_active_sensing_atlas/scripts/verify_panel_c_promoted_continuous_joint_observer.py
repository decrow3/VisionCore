"""Verify the promoted Figure 4C continuous-joint observer artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_joint_feature_recovery import (
    PRIMARY_LATENT,
    _load_feature_tables,
    _vectorized_mode_rows,
)
from declan.figure4_active_sensing_atlas.scripts.run_panel_c_promoted_continuous_joint_observer import (
    DEFAULT_OUT_DIR,
)


EXPECTED_BASIS_BY_SCALE = {"0.5": 10, "1": 20, "2": 20}
EXPECTED_RIDGE_BY_SCALE = {"0.5": 0.01, "1": 0.1, "2": 0.1}
EXPECTED_TEMPERATURE_BY_SCALE = {"0.5": 0.125, "1": 0.125, "2": 0.5}
EXPECTED_PROCESS_BY_SCALE = {"0.5": "ar1", "1": "ar1", "2": "matched_brownian"}
EXPECTED_BROWNIAN_SCALE_BY_SCALE = {"2": 8.0}
EXPECTED_FULL = {
    "n": 768,
    "image_accuracy": 0.7083333333333334,
    "mean_feature_cosine": 0.9378186805592742,
    "mean_true_mass": 0.5961611820418192,
}
HELDOUT_SELECTION_GATE = {
    "mean_feature_cosine": 0.937074,
    "image_accuracy": 0.7083333333333334,
}
DEFAULT_MANIFEST_JSON = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "panel_C"
    / "diagnostics"
    / "continuous_joint"
    / "continuous_joint_promoted_observer_manifest.json"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return [_json_ready(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _normal_float_map(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, item in dict(value).items():
        out[f"{float(key):g}"] = float(item)
    return out


def _normal_int_map(value: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, item in dict(value).items():
        out[f"{float(key):g}"] = int(item)
    return out


def _normal_str_map(value: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, item in dict(value).items():
        out[f"{float(key):g}"] = str(item)
    return out


def _assert_float_maps_close(name: str, actual: dict[str, float], expected: dict[str, float], atol: float) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"{name} keys differ: actual={actual}, expected={expected}")
    for key, expected_value in expected.items():
        if not np.isclose(float(actual[key]), float(expected_value), rtol=0.0, atol=atol):
            raise AssertionError(f"{name}[{key}]={actual[key]} != {expected_value}")


def _score_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    posterior_path = run_dir / "continuous_joint_feature_posterior.csv"
    if not posterior_path.exists():
        raise FileNotFoundError(f"missing posterior CSV: {posterior_path}")
    posterior = pd.read_csv(posterior_path)
    continuous = posterior[posterior["observer_mode"].eq("continuous_joint")].copy()
    if continuous.empty:
        raise AssertionError("no continuous_joint posterior rows found")
    required = {"candidate_score", "candidate_score_raw", "posterior_temperature", "candidate_posterior"}
    missing = required.difference(continuous.columns)
    if missing:
        raise AssertionError(f"posterior rows missing required columns: {sorted(missing)}")
    max_score_error = float(
        np.max(
            np.abs(
                continuous["candidate_score"].to_numpy(dtype=float)
                - continuous["candidate_score_raw"].to_numpy(dtype=float)
                / continuous["posterior_temperature"].to_numpy(dtype=float)
            )
        )
    )
    posterior_sums = continuous.groupby("table_index")["candidate_posterior"].sum()
    if not np.allclose(posterior_sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-9):
        raise AssertionError("continuous_joint posterior mass does not sum to one per table")

    features = _load_feature_tables()[PRIMARY_LATENT]
    scored = _vectorized_mode_rows(
        rows=continuous,
        latent=PRIMARY_LATENT,
        feature_table=features,
        score_column="candidate_score",
    )
    raw_parts = []
    for temp, block in continuous.groupby("posterior_temperature", sort=True):
        raw_parts.append(
            _vectorized_mode_rows(
                rows=block,
                latent=PRIMARY_LATENT,
                feature_table=features,
                posterior_temperature=float(temp),
                score_column="candidate_score_raw",
            )
        )
    raw_scored = pd.concat(raw_parts, ignore_index=True).sort_values("table_index").reset_index(drop=True)
    scored = scored.sort_values("table_index").reset_index(drop=True)
    max_feature_mismatch = float(
        np.max(np.abs(scored["feature_cosine"].to_numpy(dtype=float) - raw_scored["feature_cosine"].to_numpy(dtype=float)))
    )
    if max_feature_mismatch > 1e-9:
        raise AssertionError(f"effective score and raw+temperature feature scoring differ by {max_feature_mismatch:g}")
    scored["max_effective_score_error"] = max_score_error
    scored["max_raw_temperature_feature_mismatch"] = max_feature_mismatch
    return posterior, scored


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
        raise AssertionError(f"trajectory_process_model_by_scale differs: actual={process_by_scale}, expected={EXPECTED_PROCESS_BY_SCALE}")
    _assert_float_maps_close("brownian_cov_scale_by_scale", brownian_by_scale, EXPECTED_BROWNIAN_SCALE_BY_SCALE, float(args.atol))
    if str(metadata.get("continuous_score_mode")) != "quadratic_poisson_profile":
        raise AssertionError(f"unexpected continuous_score_mode: {metadata.get('continuous_score_mode')}")
    if str(metadata.get("trajectory_initial_position")) != "inferred":
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
            "observer_slug": "noanchor_quadratic_strict_scale_prior_predeclared",
            "observer_label": "No-anchor quadratic strict scale-prior predeclared",
            "status": "promoted_feature_recovery_readout",
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
                "trajectory_process_model_by_scale": process_by_scale,
                "brownian_cov_scale_by_scale": brownian_by_scale,
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
                "Predeclared strict inferred-start endpoint: AR(1) at 0.5x/1.0x and "
                "matched-Brownian scale 8 at 2.0x. Feature-recovery readout is "
                "posterior-calibrated; hard-negative image accuracy is the MAP endpoint."
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
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--expect-full", action="store_true", help="Enforce full-cache promoted metrics.")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--metric-atol", type=float, default=1e-6)
    return parser


def main() -> None:
    verify(build_parser().parse_args())


if __name__ == "__main__":
    main()
