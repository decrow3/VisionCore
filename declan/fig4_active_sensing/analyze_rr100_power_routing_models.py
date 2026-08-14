#!/usr/bin/env python3
"""Cross-validated tests of global power, SFxTF routing, and gain form.

The test set in each of 25 crossed folds contains one image fifth crossed
with one trace fifth.  Training excludes *both* the held-out image fifth and
the held-out trace fifth, so every prediction is for unseen images and unseen
traces.  The 25 test intersections partition the 3,000-condition bank.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/data"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/model_tests"


def design(*columns: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(columns[0]))] + [np.asarray(c, float) for c in columns])


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(train)
    scale = np.nanstd(train)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (train - center) / scale, (test - center) / scale


def crossed_predictions(
    y: np.ndarray,
    image: np.ndarray,
    trace: np.ndarray,
    feature_sets: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    predictions = {name: np.full_like(y, np.nan, dtype=float) for name in feature_sets}
    image_fold = image % 5
    trace_fold = trace % 5
    for i_fold in range(5):
        for t_fold in range(5):
            test = (image_fold == i_fold) & (trace_fold == t_fold)
            train = (image_fold != i_fold) & (trace_fold != t_fold)
            for name, columns in feature_sets.items():
                train_columns: list[np.ndarray] = []
                test_columns: list[np.ndarray] = []
                for column in columns:
                    z_train, z_test = standardize_train_test(column[train], column[test])
                    train_columns.append(z_train)
                    test_columns.append(z_test)
                x_train = design(*train_columns)
                x_test = design(*test_columns)
                valid = np.isfinite(y[train]) & np.all(np.isfinite(x_train), axis=1)
                if valid.sum() <= x_train.shape[1] + 2:
                    continue
                beta = np.linalg.lstsq(x_train[valid], y[train][valid], rcond=None)[0]
                predictions[name][test] = x_test @ beta
    return predictions


def score(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(y) & np.isfinite(prediction)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    residual = float(np.sum((y[valid] - prediction[valid]) ** 2))
    total = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    r2 = 1.0 - residual / total if total > 0 else np.nan
    correlation = float(np.corrcoef(y[valid], prediction[valid])[0, 1]) if np.std(y[valid]) > 0 and np.std(prediction[valid]) > 0 else np.nan
    mae = float(np.mean(np.abs(y[valid] - prediction[valid])))
    return r2, correlation, mae


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "power_routing_joined_arrays.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    units = arrays["rr100_index"].astype(int)
    image = arrays["image_index"].astype(int)
    trace = arrays["trace_index"].astype(int)
    global_power = arrays["global_power_amplitude"].astype(float)
    routing = arrays["routed_amplitude"].astype(float)

    outcomes = {
        "activation_rms_hz": arrays["temporal_rms_delta_from_stabilized_hz"].astype(float),
        "delta_mean_rate_hz": arrays["delta_mean_rate_hz"].astype(float),
        "delta_ssi_bits_per_spike": arrays["delta_ssi_bits_per_spike"].astype(float),
        "delta_information_bits_spikes": arrays["delta_information_numerator_bits_spikes"].astype(float),
    }
    predictions_to_save: dict[str, np.ndarray] = {}
    score_rows: list[dict[str, object]] = []
    for outcome_name, outcome in outcomes.items():
        for unit_position, unit in enumerate(units):
            y = outcome[:, unit_position]
            predictions = crossed_predictions(
                y,
                image,
                trace,
                {
                    "global": [global_power],
                    "routing": [routing[:, unit_position]],
                    "hybrid": [global_power, routing[:, unit_position]],
                },
            )
            for model_name, prediction in predictions.items():
                predictions_to_save[f"{outcome_name}__{model_name}__unit_{unit}"] = prediction.astype(np.float32)
                r2, correlation, mae = score(y, prediction)
                score_rows.append(
                    {
                        "rr100_index": unit,
                        "outcome": outcome_name,
                        "model": model_name,
                        "crossed_cv_r2": r2,
                        "crossed_cv_correlation": correlation,
                        "crossed_cv_mae": mae,
                        "n_predictions": int(np.isfinite(prediction).sum()),
                    }
                )

    # Moving-rate models test whether FEM behaves as an additive term, a
    # baseline-scaled term, or requires both. SSI is deliberately excluded:
    # bits/spike is not a response gain and should not be described that way.
    moving = arrays["moving_mean_rate_hz"].astype(float)
    baseline = arrays["stabilized_mean_rate_hz"].astype(float)
    gain_rows: list[dict[str, object]] = []
    for unit_position, unit in enumerate(units):
        y = moving[:, unit_position]
        b = baseline[:, unit_position]
        r = routing[:, unit_position]
        interaction_global = b * global_power
        interaction_routing = b * r
        feature_sets = {
            "baseline_only": [b],
            "global_additive": [b, global_power],
            "global_multiplicative": [b, interaction_global],
            "global_additive_plus_multiplicative": [b, global_power, interaction_global],
            "routing_additive": [b, r],
            "routing_multiplicative": [b, interaction_routing],
            "routing_additive_plus_multiplicative": [b, r, interaction_routing],
        }
        predictions = crossed_predictions(y, image, trace, feature_sets)
        baseline_r2 = score(y, predictions["baseline_only"])[0]
        for model_name, prediction in predictions.items():
            predictions_to_save[f"moving_mean_rate_hz__{model_name}__unit_{unit}"] = prediction.astype(np.float32)
            r2, correlation, mae = score(y, prediction)
            gain_rows.append(
                {
                    "rr100_index": unit,
                    "model": model_name,
                    "crossed_cv_r2": r2,
                    "delta_r2_over_baseline": r2 - baseline_r2,
                    "crossed_cv_correlation": correlation,
                    "crossed_cv_mae": mae,
                    "n_predictions": int(np.isfinite(prediction).sum()),
                }
            )

    score_table = pd.DataFrame(score_rows)
    gain_table = pd.DataFrame(gain_rows)
    score_table.to_csv(OUT / "unit_level_global_routing_hybrid_cv.csv", index=False)
    gain_table.to_csv(OUT / "unit_level_additive_multiplicative_cv.csv", index=False)
    np.savez_compressed(OUT / "crossed_cv_predictions.npz", **predictions_to_save)

    population = (
        score_table.groupby(["outcome", "model"], as_index=False)
        .agg(
            n_units=("rr100_index", "size"),
            median_cv_r2=("crossed_cv_r2", "median"),
            median_cv_correlation=("crossed_cv_correlation", "median"),
            median_cv_mae=("crossed_cv_mae", "median"),
        )
    )
    population.to_csv(OUT / "population_global_routing_hybrid_summary.csv", index=False)
    gain_population = (
        gain_table.groupby("model", as_index=False)
        .agg(
            n_units=("rr100_index", "size"),
            median_cv_r2=("crossed_cv_r2", "median"),
            median_delta_r2_over_baseline=("delta_r2_over_baseline", "median"),
            median_cv_correlation=("crossed_cv_correlation", "median"),
        )
    )
    gain_population.to_csv(OUT / "population_additive_multiplicative_summary.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "crossed_cv_model_tests_complete",
        "n_conditions": int(len(image)),
        "n_units": int(len(units)),
        "cross_validation": "25 test intersections of image_id modulo 5 x trace_id modulo 5; training excludes both held-out groups",
        "global_routing_models": ["global", "routing", "hybrid"],
        "activation_primary": "temporal RMS of moving-minus-stabilized rate timecourses",
        "gain_guardrail": "additive/multiplicative language applies only to moving mean rate; SSI is analyzed as a signed change",
        "gain_weight_guardrail": "native F0 gain is not compared in per-unit free-slope regressions because it is algebraically absorbed by the fitted slope",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
