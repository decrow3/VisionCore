#!/usr/bin/env python3
"""Join corrected retinal power, native F0 passbands, and RR100 outcomes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from declan.fig4_active_sensing.spectral_cache_contract import validated_spectral_cache_from_environment


ROOT = Path(__file__).resolve().parents[2]
RESPONSES = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/assembled/rounds_000_002_n003"
TUNING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_native_extended_tf_f0_analysis_v1"
ASSIGNMENTS = ROOT / "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/sf_half_recorded_validated_unit_assignments.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/data"


def log_interp(frequency: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.interp(np.log2(target), np.log2(frequency), values)
    return result / max(float(np.nanmax(result)), 1e-15)


def main() -> None:
    spectral = validated_spectral_cache_from_environment()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(spectral / "condition_spectra.npz", allow_pickle=False) as data:
        condition_rows = np.asarray(data["matrix_row_index"], dtype=int)
        image_ids = np.asarray(data["image_index"], dtype=int)
        trace_ids = np.asarray(data["trace_index"], dtype=int)
        round_ids = np.asarray(data["round_index"], dtype=int)
        radial = np.asarray(data["radial_power"], dtype=np.float64)
        sf_edges = np.asarray(data["sf_edges_cpd"], dtype=float)
        tf_all = np.asarray(data["tf_hz"], dtype=float)
    if not np.array_equal(condition_rows, np.arange(3000)):
        raise ValueError("Spectral cache is not response-row aligned")
    sf_all = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    sf_mask = (sf_all >= 1.0) & (sf_all <= 11.3137085)
    tf_mask = (tf_all > 0) & (tf_all <= 56.0)
    sf = sf_all[sf_mask]
    tf = tf_all[tf_mask]
    power = radial[:, tf_mask][:, :, sf_mask]
    global_amplitude = np.sqrt(np.maximum(power.sum(axis=(1, 2)), 0.0))

    factors = pd.read_csv(TUNING / "extended_f0_factor_points.csv")
    fit_summary = pd.read_csv(TUNING / "extended_f0_fit_unit_summary.csv")
    assignments = pd.read_csv(ASSIGNMENTS)
    unit_table = fit_summary.merge(
        assignments[["rr100_index", "recorded_validation_pass", "sf_outer_third", "preferred_sf_cpd"]],
        on="rr100_index",
        how="left",
        validate="one_to_one",
    )
    unit_table["routing_quality_pass"] = (
        unit_table.responsive_positive_f0_flag.astype(bool)
        & unit_table.extended_sf_parametric_fit_ok.astype(bool)
        & unit_table.extended_tf_parametric_fit_ok.astype(bool)
        & unit_table.recorded_validation_pass.fillna(False).astype(bool)
    )
    units = unit_table.loc[unit_table.routing_quality_pass, "rr100_index"].astype(int).to_numpy()
    if len(units) < 3:
        raise RuntimeError("Too few completed, recorded-validated extended tuning fits")

    h_shape = np.empty((len(units), len(tf), len(sf)), dtype=float)
    gains = np.empty(len(units), dtype=float)
    for position, unit in enumerate(units):
        unit_factors = factors[factors.rr100_index.eq(unit)]
        sf_factor = unit_factors[unit_factors.axis.eq("spatial_frequency")].sort_values("frequency")
        tf_factor = unit_factors[unit_factors.axis.eq("temporal_frequency")].sort_values("frequency")
        sf_curve = log_interp(sf_factor.frequency.to_numpy(float), sf_factor.parametric_prediction.to_numpy(float), sf)
        tf_curve = log_interp(tf_factor.frequency.to_numpy(float), tf_factor.parametric_prediction.to_numpy(float), tf)
        h_shape[position] = np.outer(tf_curve, sf_curve)
        gains[position] = float(unit_table.loc[unit_table.rr100_index.eq(unit), "extended_rank1_gain_f0_hz"].iloc[0])

    routed_power = np.einsum("ctf,utf->cu", power, h_shape**2)
    routed_amplitude = np.sqrt(np.maximum(routed_power, 0.0))
    gain_weighted_amplitude = routed_amplitude * gains[None, :]
    band_edges = [(0.0, 32.0), (32.0, 45.0), (45.0, 56.0)]
    band_power = np.empty((len(power), len(units), len(band_edges)), dtype=float)
    for band, (low, high) in enumerate(band_edges):
        mask = (tf > low) & (tf <= high)
        band_power[:, :, band] = np.einsum("ctf,utf->cu", power[:, mask], h_shape[:, mask] ** 2)

    moving_info = np.load(RESPONSES / "moving_information_numerator_bits_spikes.npy")
    moving_spikes = np.load(RESPONSES / "moving_expected_spikes.npy")
    moving_rate = np.load(RESPONSES / "moving_mean_rate_hz.npy")
    moving_ssi = np.load(RESPONSES / "moving_movie_ssi_bits_per_spike.npy")
    moving_sd = np.load(RESPONSES / "moving_temporal_sd_rate_hz.npy")
    moving_rms = np.load(RESPONSES / "moving_temporal_rms_delta_from_stabilized_hz.npy")
    baseline = np.load(RESPONSES / "stabilized_by_image_sufficient_statistics.npz")
    baseline_info = baseline["information_numerator_bits_spikes"][image_ids]
    baseline_spikes = baseline["expected_spikes"][image_ids]
    baseline_rate = baseline["mean_rate_hz"][image_ids]
    baseline_ssi = baseline["movie_ssi_bits_per_spike"][image_ids]
    baseline_sd = baseline["temporal_sd_rate_hz"][image_ids]

    outcome_arrays = {
        "moving_mean_rate_hz": moving_rate[:, units],
        "stabilized_mean_rate_hz": baseline_rate[:, units],
        "delta_mean_rate_hz": moving_rate[:, units] - baseline_rate[:, units],
        "moving_ssi_bits_per_spike": moving_ssi[:, units],
        "stabilized_ssi_bits_per_spike": baseline_ssi[:, units],
        "delta_ssi_bits_per_spike": moving_ssi[:, units] - baseline_ssi[:, units],
        "delta_information_numerator_bits_spikes": moving_info[:, units] - baseline_info[:, units],
        "delta_expected_spikes": moving_spikes[:, units] - baseline_spikes[:, units],
        "moving_temporal_sd_rate_hz": moving_sd[:, units],
        "delta_temporal_sd_rate_hz": moving_sd[:, units] - baseline_sd[:, units],
        "temporal_rms_delta_from_stabilized_hz": moving_rms[:, units],
    }
    np.savez_compressed(
        OUT / "power_routing_joined_arrays.npz",
        matrix_row_index=condition_rows,
        image_index=image_ids,
        trace_index=trace_ids,
        round_index=round_ids,
        rr100_index=units,
        sf_centers_cpd=sf,
        tf_hz=tf,
        supported_retinal_power=power.astype(np.float32),
        normalized_unit_sensitivity=h_shape.astype(np.float32),
        unit_f0_gain_hz=gains,
        global_power_amplitude=global_amplitude,
        routed_power=routed_power,
        routed_amplitude=routed_amplitude,
        gain_weighted_routed_amplitude=gain_weighted_amplitude,
        routed_band_power=band_power,
        routed_band_edges_hz=np.asarray(band_edges),
        **{key: np.asarray(value, dtype=np.float32) for key, value in outcome_arrays.items()},
    )
    unit_table.to_csv(OUT / "routing_unit_cohort.csv", index=False)
    condition_table = pd.DataFrame(
        {
            "matrix_row_index": condition_rows,
            "image_index": image_ids,
            "trace_index": trace_ids,
            "round_index": round_ids,
            "global_power_amplitude": global_amplitude,
            "supported_power_fraction_of_all_positive_tf": power.sum(axis=(1, 2)) / np.maximum(radial.sum(axis=(1, 2)), np.finfo(float).tiny),
        }
    )
    condition_table.to_csv(OUT / "routing_condition_table.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "power_routing_join_complete",
        "scope": {"conditions": 3000, "units_with_routing_quality_pass": int(len(units)), "images": 100, "traces": 1000},
        "support": {"sf_cpd": [float(sf.min()), float(sf.max())], "tf_hz": [float(tf.min()), float(tf.max())], "tf_bins": tf.tolist()},
        "predictor_contracts": {
            "global": "sqrt(sum P) over identical supported SFxTF bins",
            "routing": "sqrt(sum P * normalized H_u^2)",
            "gain_weighted": "native-F0 rank1 gain multiplied by routing amplitude",
            "bands": "un-squared routed variance contributions in <=32, 32–45, and 45–56 Hz bands",
        },
        "guardrail": "unit filtering uses only completed native-readout extended fits passing recorded-SF validation",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
