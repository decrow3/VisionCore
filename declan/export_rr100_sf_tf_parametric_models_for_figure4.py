#!/usr/bin/env python3
"""Export the canonical RR100 parametric SF/TF models for Figure 4 analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/redundancy_resolved_v1_twin"
    / "rr100_joint_f0_parametric_recorded_validation_v2"
    / "rr100_parametric_recorded_validation_by_unit.csv"
)
DEFAULT_MAPPING = (
    ROOT
    / "outputs/redundancy_resolved_v1_twin"
    / "rr100_recorded_twin_gratings_check_v1"
    / "rr100_unit_mapping.csv"
)
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
SF_FIT_SUPPORT = (1.0, 11.313708498984761)
TF_FIT_SUPPORT = (0.5, 32.0)
PARAMETER_NAMES = np.asarray(["baseline", "amplitude", "center_log2", "sigma_octaves"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def log_gaussian(frequency: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    baseline, amplitude, center_log2, sigma_octaves = np.asarray(parameters, dtype=float)
    x = np.log2(np.asarray(frequency, dtype=float))
    return baseline + amplitude * np.exp(-0.5 * ((x - center_log2) / sigma_octaves) ** 2)


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def build_export(source: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    if source["rr100_index"].duplicated().any() or mapping["rr100_index"].duplicated().any():
        raise ValueError("RR100 indices must be unique in source and mapping")
    merged = mapping.merge(source, on="rr100_index", suffixes=("_mapping", "_fit"), validate="one_to_one")
    merged = merged.sort_values("rr100_index").reset_index(drop=True)
    if merged["rr100_index"].tolist() != list(range(100)):
        raise ValueError("Export requires exactly RR100 indices 0 through 99")
    if not merged["session_mapping"].eq(merged["session_fit"]).all():
        raise ValueError("Session identity disagrees between RR100 mapping and fit table")

    valid = (
        merged["responsive_positive_f0_flag"].astype(bool)
        & merged["sf_parametric_fit_ok"].astype(bool)
        & merged["tf_parametric_fit_ok"].astype(bool)
    )
    sf_params = merged[["sf_baseline", "sf_amplitude", "sf_center_log2", "sf_sigma_octaves"]].to_numpy(float)
    tf_params = merged[["tf_baseline", "tf_amplitude", "tf_center_log2", "tf_sigma_octaves"]].to_numpy(float)
    if not np.isfinite(sf_params[valid]).all() or not np.isfinite(tf_params[valid]).all():
        raise ValueError("A valid unit has non-finite model parameters")

    sf_pref = merged["sf_preferred_within_support"].to_numpy(float)
    tf_pref = merged["tf_preferred_within_support"].to_numpy(float)
    sf_norm = np.full(100, np.nan)
    tf_norm = np.full(100, np.nan)
    for unit in np.flatnonzero(valid.to_numpy()):
        sf_norm[unit] = float(log_gaussian(np.asarray([sf_pref[unit]]), sf_params[unit])[0])
        tf_norm[unit] = float(log_gaussian(np.asarray([tf_pref[unit]]), tf_params[unit])[0])
    if np.any(sf_norm[valid] <= 0) or np.any(tf_norm[valid] <= 0):
        raise ValueError("Factor normalization must be positive for every valid unit")

    table = pd.DataFrame(
        {
            "rr100_index": merged["rr100_index"].astype(int),
            "canonical_channel": merged["canonical_channel"].astype(int),
            "session": merged["session_mapping"],
            "source_unit_index": merged["source_unit_index"].astype(int),
            "ccnorm": merged["ccnorm"],
            "group_kind": merged["group_kind"],
            "group_size": merged["group_size"].astype(int),
            "model_valid": valid,
            "model_status": np.where(valid, "valid_positive_dynamic_f0", "no_positive_dynamic_f0_fit"),
            "response_statistic": "positive_dynamic_f0_hz",
            "preferred_orientation_deg": merged["preferred_orientation_deg"],
            "preferred_sf_cpd": np.where(valid, sf_pref, np.nan),
            "preferred_tf_hz": np.where(valid, tf_pref, np.nan),
            "sf_center_cpd_unbounded": merged["sf_center_frequency"],
            "tf_center_hz_unbounded": merged["tf_center_frequency"],
            "sf_sampled_preferred_cpd": merged["sf_sampled_preferred"],
            "tf_sampled_preferred_hz": merged["tf_sampled_preferred"],
            "sf_fit_support_min_cpd": SF_FIT_SUPPORT[0],
            "sf_fit_support_max_cpd": SF_FIT_SUPPORT[1],
            "tf_fit_support_min_hz": TF_FIT_SUPPORT[0],
            "tf_fit_support_max_hz": TF_FIT_SUPPORT[1],
            "sf_baseline": merged["sf_baseline"],
            "sf_amplitude": merged["sf_amplitude"],
            "sf_center_log2_cpd": merged["sf_center_log2"],
            "sf_sigma_octaves": merged["sf_sigma_octaves"],
            "sf_fwhm_octaves": merged["sf_fwhm_octaves"],
            "sf_normalization_on_fit_support": sf_norm,
            "tf_baseline": merged["tf_baseline"],
            "tf_amplitude": merged["tf_amplitude"],
            "tf_center_log2_hz": merged["tf_center_log2"],
            "tf_sigma_octaves": merged["tf_sigma_octaves"],
            "tf_fwhm_octaves": merged["tf_fwhm_octaves"],
            "tf_normalization_on_fit_support": tf_norm,
            "joint_rank1_gain_f0_hz": merged["rank1_gain_f0_hz"],
            "sf_fit_r2": merged["sf_fit_r2"],
            "sf_fit_rmse_normalized_factor": merged["sf_fit_rmse"],
            "tf_fit_r2": merged["tf_fit_r2"],
            "tf_fit_rmse_normalized_factor": merged["tf_fit_rmse"],
            "joint_parametric_surface_r2": merged["parametric_surface_centered_r2"],
            "joint_parametric_surface_rmse_hz": merged["parametric_surface_rmse_hz"],
            "rank1_energy_fraction": merged["rank1_energy_fraction"],
            "rank1_centered_r2": merged["rank1_centered_r2"],
            "rank1_relative_rmse": merged["rank1_relative_rmse"],
            "recorded_sf_curve_r_full_support": merged["recorded_curve_pearson_r_parametric"],
            "recorded_sf_curve_nrmse_full_support": merged["recorded_curve_normalized_rmse_parametric"],
            "recorded_sf_curve_r_in_fit_support": merged["recorded_curve_pearson_r_parametric_in_fit_support"],
            "recorded_sf_curve_nrmse_in_fit_support": merged["recorded_curve_normalized_rmse_parametric_in_fit_support"],
            "recorded_positive_sf_support_cpd": merged["recorded_positive_sf_support_cpd"],
            "recorded_sf_peak_cpd": merged["recorded_peak_sf_on_predicted_orientation_recorded_support"],
            "predicted_sf_peak_on_recorded_support_cpd": merged["predicted_peak_sf_on_recorded_support"],
            "recorded_vs_real_stimulus_twin_sf_curve_r": merged["existing_heldout_twin_vs_recorded_sf_curve_r"],
        }
    )

    sf_grid = np.geomspace(1.0, 16.0, 257)
    tf_grid = np.geomspace(TF_FIT_SUPPORT[0], TF_FIT_SUPPORT[1], 257)
    sf_curves = np.full((100, len(sf_grid)), np.nan)
    tf_curves = np.full((100, len(tf_grid)), np.nan)
    for unit in np.flatnonzero(valid.to_numpy()):
        sf_curves[unit] = log_gaussian(sf_grid, sf_params[unit]) / sf_norm[unit]
        tf_curves[unit] = log_gaussian(tf_grid, tf_params[unit]) / tf_norm[unit]

    models: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        unit = int(row["rr100_index"])
        is_valid = bool(row["model_valid"])
        models.append(
            {
                "rr100_index": unit,
                "identity": {
                    key: finite_or_none(row[key])
                    for key in ("canonical_channel", "session", "source_unit_index", "ccnorm", "group_kind", "group_size")
                },
                "valid": is_valid,
                "status": row["model_status"],
                "response_statistic": row["response_statistic"],
                "preferred": {
                    "orientation_deg": finite_or_none(row["preferred_orientation_deg"]),
                    "spatial_cpd": finite_or_none(row["preferred_sf_cpd"]),
                    "temporal_hz_magnitude": finite_or_none(row["preferred_tf_hz"]),
                },
                "spatial_factor": {
                    "support_cpd": list(SF_FIT_SUPPORT),
                    "parameters": {
                        "baseline": finite_or_none(row["sf_baseline"]),
                        "amplitude": finite_or_none(row["sf_amplitude"]),
                        "center_log2_cpd": finite_or_none(row["sf_center_log2_cpd"]),
                        "sigma_octaves": finite_or_none(row["sf_sigma_octaves"]),
                    },
                    "normalization_on_fit_support": finite_or_none(row["sf_normalization_on_fit_support"]),
                },
                "temporal_factor": {
                    "support_hz_magnitude": list(TF_FIT_SUPPORT),
                    "parameters": {
                        "baseline": finite_or_none(row["tf_baseline"]),
                        "amplitude": finite_or_none(row["tf_amplitude"]),
                        "center_log2_hz": finite_or_none(row["tf_center_log2_hz"]),
                        "sigma_octaves": finite_or_none(row["tf_sigma_octaves"]),
                    },
                    "normalization_on_fit_support": finite_or_none(row["tf_normalization_on_fit_support"]),
                },
                "joint_gain_f0_hz": finite_or_none(row["joint_rank1_gain_f0_hz"]),
                "goodness_of_fit": {
                    key: finite_or_none(row[key])
                    for key in (
                        "sf_fit_r2", "sf_fit_rmse_normalized_factor", "tf_fit_r2",
                        "tf_fit_rmse_normalized_factor", "joint_parametric_surface_r2",
                        "joint_parametric_surface_rmse_hz", "rank1_energy_fraction",
                        "rank1_centered_r2", "rank1_relative_rmse",
                    )
                },
                "recorded_biological_validation": {
                    key: finite_or_none(row[key])
                    for key in (
                        "recorded_sf_curve_r_full_support", "recorded_sf_curve_nrmse_full_support",
                        "recorded_sf_curve_r_in_fit_support", "recorded_sf_curve_nrmse_in_fit_support",
                        "recorded_positive_sf_support_cpd", "recorded_sf_peak_cpd",
                        "predicted_sf_peak_on_recorded_support_cpd",
                        "recorded_vs_real_stimulus_twin_sf_curve_r",
                    )
                },
            }
        )

    bundle = {
        "schema_version": "rr100_sf_tf_parametric_models_v1",
        "response_definition": "positive dynamic F0, direction-collapsed to |TF|",
        "axis_formula": "factor(f) = baseline + amplitude * exp(-0.5 * ((log2(f) - center_log2) / sigma_octaves)^2)",
        "joint_formula": "F0_hz(SF,TF) = joint_gain_f0_hz * spatial_factor(SF)/sf_normalization_on_fit_support * temporal_factor(abs(TF))/tf_normalization_on_fit_support",
        "n_units": 100,
        "n_valid_models": int(valid.sum()),
        "n_invalid_models": int((~valid).sum()),
        "models": models,
    }
    arrays = {
        "rr100_index": np.arange(100, dtype=np.int64),
        "model_valid": valid.to_numpy(bool),
        "parameter_names": PARAMETER_NAMES,
        "sf_parameters": sf_params,
        "tf_parameters": tf_params,
        "preferred_sf_cpd": np.where(valid, sf_pref, np.nan),
        "preferred_tf_hz": np.where(valid, tf_pref, np.nan),
        "preferred_orientation_deg": merged["preferred_orientation_deg"].to_numpy(float),
        "joint_rank1_gain_f0_hz": merged["rank1_gain_f0_hz"].to_numpy(float),
        "sf_normalization_on_fit_support": sf_norm,
        "tf_normalization_on_fit_support": tf_norm,
        "sf_fit_r2": merged["sf_fit_r2"].to_numpy(float),
        "tf_fit_r2": merged["tf_fit_r2"].to_numpy(float),
        "joint_parametric_surface_r2": merged["parametric_surface_centered_r2"].to_numpy(float),
        "recorded_sf_curve_r_full_support": merged["recorded_curve_pearson_r_parametric"].to_numpy(float),
        "sf_evaluation_grid_cpd": sf_grid,
        "tf_evaluation_grid_hz": tf_grid,
        "sf_factor_normalized_curves": sf_curves,
        "tf_factor_normalized_curves": tf_curves,
    }
    return table, bundle, arrays


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source)
    mapping = pd.read_csv(args.mapping)
    table, bundle, arrays = build_export(source, mapping)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "rr100_sf_tf_parametric_models.csv"
    json_path = args.out_dir / "rr100_sf_tf_parametric_models.json"
    npz_path = args.out_dir / "rr100_sf_tf_parametric_model_arrays.npz"
    table.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, allow_nan=False)
    np.savez_compressed(npz_path, **arrays)

    readme = """# RR100 parametric SF/TF models for Figure 4

This is the canonical portable export of the current separable RR100 tuning models.

- `rr100_sf_tf_parametric_models.csv`: one row per RR100 unit, including identity, preferred SF/TF, parameters, fit quality, and recorded-biological validation.
- `rr100_sf_tf_parametric_models.json`: fully specified reconstructable models with formula and nested provenance-friendly fields.
- `rr100_sf_tf_parametric_model_arrays.npz`: numeric arrays and pre-evaluated normalized factor curves for plotting/analysis.

The response is positive dynamic F0. Temporal frequency is represented by magnitude, `abs(TF)`. For a valid unit:

```
factor(f) = baseline + amplitude * exp(-0.5 * ((log2(f) - center_log2) / sigma_octaves)^2)
F0_hz(SF,TF) = gain * SF_factor(SF)/SF_norm * TF_factor(abs(TF))/TF_norm
```

The normalization constants are saved explicitly. SF parameters were fit over 1–11.313708 cpd and TF parameters over 0.5–32 Hz. Evaluation at 16 cpd is an extrapolation. Rows marked `model_valid = false` are retained to preserve the complete RR100 index but must not be imputed as tuned units.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": bundle["schema_version"],
        "status": "complete",
        "n_units": 100,
        "n_valid_models": int(table["model_valid"].sum()),
        "n_invalid_models": int((~table["model_valid"]).sum()),
        "source": file_identity(args.source),
        "mapping": file_identity(args.mapping),
        "artifacts": {
            "table_csv": csv_path.name,
            "models_json": json_path.name,
            "arrays_npz": npz_path.name,
            "readme": "README.md",
        },
        "contract": {
            "response": bundle["response_definition"],
            "axis_formula": bundle["axis_formula"],
            "joint_formula": bundle["joint_formula"],
            "sf_fit_support_cpd": list(SF_FIT_SUPPORT),
            "tf_fit_support_hz_magnitude": list(TF_FIT_SUPPORT),
            "invalid_unit_policy": "retain row and RR100 identity; parameters/preferences are null or NaN; do not impute",
        },
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    print(f"Wrote {args.out_dir.resolve()}")
    print(table[["model_valid", "sf_fit_r2", "tf_fit_r2", "joint_parametric_surface_r2"]].describe(include="all").to_string())


if __name__ == "__main__":
    main()
