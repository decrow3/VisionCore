#!/usr/bin/env python3
"""Compare full twin, spectral-power, and recorded grating responses.

The checkpoint uses exact held-out gaze-cropped grating movies.  For every
40-frame retinal window, it computes RF-local unit-specific routed SFxTF power, aligns
the response window by the unit's independently selected recorded peak lag,
and distinguishes three objects:

1. the cached full digital-twin prediction (rhat),
2. a trial-held-out linear rate prediction from RF-local routed power, and
3. the recorded spike rate (robs).

This is a one-session map-first checkpoint, not a population claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.stats import pearsonr, spearmanr

from declan.fig4_active_sensing.make_rr100_recorded_grating_retinal_power_input_checkpoint import (
    candidate_windows,
    load_heldout_grating_dataset,
)
from declan.fig4_active_sensing.run_interim_input_spectral_cache import FRAME_RATE_HZ, N_SCORE
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    SF_EDGES_CPD,
    spatial_lookup,
)
from scripts.utils import get_model_and_dataset_configs


ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = ROOT / "experiments/dataset_configs/multi_basic_120_long_legacy.yaml"
CACHE = ROOT / "outputs/artifacts/mcfarland/mcfarland_outputs_mono.pkl"
MAPPING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
GRATING_METRICS = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_grating_tuning_metrics.csv"
ALIGNMENT = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_cache_alignment.csv"
ROUTING_DATA = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/data"
MODEL_CHECKPOINT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints/"
    "learned_resnet_none_convgru_gaussian_ddp_bs128_ds30_lr1e-3_wd1e-4_corelrscale.5_warmup5/"
    "epoch=147-val_bps_overall=0.5702.ckpt"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_response_rf_local_v2"
SEED = 1731


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG)
    parser.add_argument("--response-cache", type=Path, default=CACHE)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING)
    parser.add_argument("--grating-metrics-csv", type=Path, default=GRATING_METRICS)
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT)
    parser.add_argument("--routing-data-dir", type=Path, default=ROUTING_DATA)
    parser.add_argument("--session", default="Logan_2020-02-29")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--stride", type=int, default=N_SCORE)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
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


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3 or np.std(x[valid]) <= 1e-12 or np.std(y[valid]) <= 1e-12:
        return float("nan")
    return float(pearsonr(x[valid], y[valid]).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3 or np.std(x[valid]) <= 1e-12 or np.std(y[valid]) <= 1e-12:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def trial_demean(values: np.ndarray, trials: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for trial in np.unique(trials):
        mask = trials == trial
        result[mask] -= np.nanmean(result[mask])
    return result


def assign_trial_folds(trials: np.ndarray, n_folds: int) -> tuple[np.ndarray, pd.DataFrame]:
    unique = np.unique(np.asarray(trials, dtype=int))
    if n_folds < 2 or len(unique) < n_folds:
        raise ValueError(f"Need at least {n_folds} trials for {n_folds}-fold CV; found {len(unique)}")
    rng = np.random.default_rng(SEED)
    permuted = rng.permutation(unique)
    fold_by_trial = {int(trial): int(position % n_folds) for position, trial in enumerate(permuted)}
    folds = np.asarray([fold_by_trial[int(trial)] for trial in trials], dtype=int)
    table = pd.DataFrame(
        {"trial_index": unique, "fold": [fold_by_trial[int(trial)] for trial in unique]}
    ).sort_values("trial_index")
    return folds, table


def fit_line(x: np.ndarray, y: np.ndarray, *, nonnegative: bool) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denominator = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / max(denominator, 1e-30))
    if nonnegative:
        slope = max(slope, 0.0)
    return y_mean - slope * x_mean, slope


def grouped_cv_line(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, *, nonnegative: bool
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    prediction = np.full(len(x), np.nan, dtype=float)
    baseline = np.full(len(x), np.nan, dtype=float)
    rows = []
    for fold in np.unique(folds):
        train = folds != fold
        test = folds == fold
        intercept, slope = fit_line(x[train], y[train], nonnegative=nonnegative)
        prediction[test] = intercept + slope * x[test]
        baseline[test] = float(np.mean(y[train]))
        rows.append(
            {
                "fold": int(fold),
                "n_train": int(np.count_nonzero(train)),
                "n_test": int(np.count_nonzero(test)),
                "intercept_hz": intercept,
                "slope": slope,
            }
        )
    return prediction, baseline, pd.DataFrame(rows)


def cv_r2(observed: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> float:
    numerator = float(np.sum((observed - prediction) ** 2))
    denominator = float(np.sum((observed - baseline) ** 2))
    return 1.0 - numerator / denominator if denominator > 1e-15 else float("nan")


def load_session_cache(path: Path, session: str) -> dict:
    with path.open("rb") as handle:
        outputs = dill.load(handle)
    matches = [row for row in outputs if str(row["sess"]) == str(session)]
    if len(matches) != 1:
        raise ValueError(f"Expected one cached result for {session}; found {len(matches)}")
    return matches[0]["bps_results"]["gratings"]


def backproject_readout_footprint(readout_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a 14x14 Gaussian readout footprint onto the 51x51 retinal input.

    The transpose-support calculation follows the feedforward spatial path:
    valid 5x5 convolution, 2x2 stride-2 pooling, valid 9x9 convolution, and
    valid 7x7 stem convolution.  It deliberately does not expand the aperture
    through recurrent ConvGRU state propagation: this is a localized input
    spectrum centered on the fitted RF, not a claim about the full nonlinear
    effective receptive field.
    """
    mask = np.asarray(readout_mask, dtype=np.float64)
    if mask.shape != (14, 14):
        raise ValueError(f"Expected 14x14 fitted readout mask, got {mask.shape}")
    footprint = convolve2d(mask, np.ones((5, 5), dtype=float), mode="full")
    prepool = np.zeros((37, 37), dtype=np.float64)
    prepool[:36, :36] = np.repeat(np.repeat(footprint, 2, axis=0), 2, axis=1)
    footprint = convolve2d(prepool, np.ones((9, 9), dtype=float), mode="full")
    footprint = convolve2d(footprint, np.ones((7, 7), dtype=float), mode="full")
    if footprint.shape != (51, 51):
        raise ValueError(f"Backprojected RF footprint has shape {footprint.shape}")
    footprint = np.maximum(footprint, 0.0)
    footprint /= max(float(footprint.sum()), 1e-30)
    # Fourier power is quadratic in the aperture.  sqrt(footprint) therefore
    # makes its squared spatial weight equal the normalized RF footprint.
    aperture = np.sqrt(footprint)
    aperture /= max(float(np.sqrt(np.sum(aperture**2))), 1e-30)
    return footprint.astype(np.float32), aperture.astype(np.float32)


def footprint_metadata(footprint: np.ndarray) -> dict[str, float]:
    weights = np.asarray(footprint, dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(51, dtype=float), np.arange(51, dtype=float), indexing="ij")
    mass = max(float(weights.sum()), 1e-30)
    x = float(np.sum(weights * xx) / mass)
    y = float(np.sum(weights * yy) / mass)
    radius = np.hypot(xx - x, yy - y)
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(weights.ravel()[order]) / mass
    radius95 = float(radius.ravel()[order][np.searchsorted(cumulative, 0.95, side="left")])
    return {
        "rf_center_x_pixel": x,
        "rf_center_y_pixel": y,
        "rf_rms_radius_pixel": float(np.sqrt(np.sum(weights * radius**2) / mass)),
        "rf_radius95_pixel": radius95,
    }


def localized_radial_spectrum(
    movie: np.ndarray, *, ppd: float, spatial_aperture: np.ndarray
) -> np.ndarray:
    arr = np.asarray(movie, dtype=np.float64)
    aperture = np.asarray(spatial_aperture, dtype=np.float64)
    if arr.shape != (N_SCORE, 51, 51) or aperture.shape != (51, 51):
        raise ValueError(f"Unexpected movie/aperture shapes {arr.shape}/{aperture.shape}")
    residual = arr - arr.mean(axis=0, keepdims=True)
    temporal_window = np.hanning(N_SCORE)[:, None, None]
    temporal_fft = np.fft.rfft(residual * temporal_window * aperture[None, :, :], axis=0)
    spectrum = np.fft.fftshift(np.fft.fft2(temporal_fft, axes=(1, 2)), axes=(1, 2))
    power = np.abs(spectrum) ** 2
    temporal_weights = np.ones(power.shape[0], dtype=np.float64)
    temporal_weights[1:-1] = 2.0
    power *= temporal_weights[:, None, None]
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)
    positive_power = power[tf_hz > 0].reshape(np.count_nonzero(tf_hz > 0), -1)
    sf_bin, _, _ = spatial_lookup(float(ppd))
    radial = np.empty((positive_power.shape[0], len(SF_EDGES_CPD) - 1), dtype=np.float32)
    for index, values in enumerate(positive_power):
        radial[index] = np.bincount(
            sf_bin, weights=values, minlength=len(SF_EDGES_CPD) - 1
        ).astype(np.float32)
    return radial


def unit_table(
    args: argparse.Namespace,
    *,
    fitted_model=None,
) -> tuple[
    pd.DataFrame,
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    mapping = pd.read_csv(args.mapping_csv)
    tuning = pd.read_csv(args.routing_data_dir / "routing_unit_cohort.csv")
    grating = pd.read_csv(args.grating_metrics_csv)
    units = (
        tuning.loc[tuning["routing_quality_pass"].astype(bool)]
        .merge(mapping, on=["rr100_index", "session"], how="inner", validate="one_to_one")
        .merge(
            grating[["rr100_index", "peak_lag_bins", "peak_lag_ms", "map_correlation", "sf_curve_correlation"]],
            on="rr100_index",
            how="left",
            validate="one_to_one",
        )
    )
    units = units.loc[units["session"].eq(args.session)].copy().sort_values("rr100_index")
    if units.empty:
        raise RuntimeError(f"No routing-qualified RR100 units belong to {args.session}")
    with np.load(args.routing_data_dir / "power_routing_joined_arrays.npz") as archive:
        archive_units = archive["rr100_index"].astype(int)
        sf = archive["sf_centers_cpd"].astype(float)
        tf = archive["tf_hz"].astype(float)
        sensitivity = archive["normalized_unit_sensitivity"].astype(float)
    lookup = {int(unit): sensitivity[position] for position, unit in enumerate(archive_units)}
    missing = [int(unit) for unit in units.rr100_index if int(unit) not in lookup]
    if missing:
        raise ValueError(f"Missing routing sensitivities for units {missing}")
    model = fitted_model
    if model is None:
        model, _ = get_model_and_dataset_configs(mode="standard")
    if args.session not in model.names:
        raise ValueError(f"Session {args.session} is absent from the fitted model")
    dataset_index = list(model.names).index(args.session)
    readout = model.model.readouts[dataset_index]
    readout_masks = readout.compute_gaussian_mask(14, 14, model.device).detach().cpu().numpy()
    footprints: dict[int, np.ndarray] = {}
    apertures: dict[int, np.ndarray] = {}
    rf_rows = []
    for unit in units.itertuples(index=False):
        rr100_index = int(unit.rr100_index)
        source = int(unit.source_unit_index)
        footprint, aperture = backproject_readout_footprint(readout_masks[source])
        footprints[rr100_index] = footprint
        apertures[rr100_index] = aperture
        rf_rows.append(
            {
                "rr100_index": rr100_index,
                "source_unit_index": source,
                "readout_mean_y_feature_pixel": float(readout.mean[source, 0].detach().cpu()),
                "readout_mean_x_feature_pixel": float(readout.mean[source, 1].detach().cpu()),
                "readout_std_y_feature_pixel": float(readout.std[source, 0].detach().cpu()),
                "readout_std_x_feature_pixel": float(readout.std[source, 1].detach().cpu()),
                "readout_theta_radian": float(readout.theta[source].detach().cpu()),
                **footprint_metadata(footprint),
            }
        )
    units = units.merge(pd.DataFrame(rf_rows), on=["rr100_index", "source_unit_index"], validate="one_to_one")
    return units, lookup, sf, tf, footprints, apertures


def indices_for_support(values: np.ndarray, reference: np.ndarray, label: str) -> np.ndarray:
    indices = []
    for target in reference:
        matches = np.flatnonzero(np.isclose(values, target, rtol=1e-7, atol=1e-8))
        if matches.size != 1:
            raise ValueError(f"Could not match {label}={target:g} in {values.tolist()}")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def build_window_unit_table(
    metrics: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    units: pd.DataFrame,
    sensitivity: dict[int, np.ndarray],
    power_sf: np.ndarray,
    power_tf: np.ndarray,
    candidate_sf: np.ndarray,
    candidate_tf: np.ndarray,
    ppd: float,
    rf_apertures: dict[int, np.ndarray],
    dset,
    local: np.ndarray,
    cached: dict,
    alignment_basis: str,
) -> pd.DataFrame:
    local = np.asarray(local, dtype=int)
    if not np.all(np.diff(local) > 0):
        raise ValueError("Held-out local indices are not strictly increasing")
    robs = cached["robs"].detach().cpu().numpy().astype(float)
    rhat = cached["rhat"].detach().cpu().numpy().astype(float)
    dfs = cached["dfs"].detach().cpu().numpy().astype(float)
    if len(local) != len(robs) or robs.shape != rhat.shape:
        raise ValueError(f"Cached response shapes do not match held-out indices: {len(local)}, {robs.shape}, {rhat.shape}")
    selected_columns = units["source_unit_index"].astype(int).to_numpy()
    if np.any(selected_columns >= robs.shape[1]):
        raise ValueError(f"RR100 source columns exceed cached response width {robs.shape[1]}")
    if alignment_basis == "sample_for_sample_exact_robs":
        dset_robs = dset["robs"][local][:, selected_columns].detach().cpu().numpy().astype(float)
        if not np.array_equal(dset_robs, robs[:, selected_columns]):
            maximum = float(np.max(np.abs(dset_robs - robs[:, selected_columns])))
            raise ValueError(f"Recorded response cache failed declared exact alignment; maximum difference {maximum:g}")
    elif alignment_basis != "deterministic_validation_split_and_equal_time_length":
        raise ValueError(f"Unsupported recorded-response alignment basis: {alignment_basis}")

    local_to_cache = np.full(len(dset), -1, dtype=int)
    local_to_cache[local] = np.arange(len(local), dtype=int)
    trial = dset["trial_inds"].detach().cpu().numpy().astype(int)
    sf_index = indices_for_support(candidate_sf, power_sf, "SF")
    tf_index = indices_for_support(candidate_tf, power_tf, "TF")
    dt = 1.0 / FRAME_RATE_HZ
    rows = []
    for window in metrics.itertuples(index=False):
        item = payload[int(window.window_index)]
        whole_crop_power = item["radial_power_tf_sf"][np.ix_(tf_index, sf_index)].astype(float)
        whole_crop_amplitude = float(np.sqrt(max(float(whole_crop_power.sum()), 0.0)))
        movie = (item["movie_uint8"].astype(np.float32) - 127.0) / 255.0
        for unit in units.itertuples(index=False):
            rr100_index = int(unit.rr100_index)
            lag = int(unit.peak_lag_bins)
            response_local = np.arange(
                int(window.start_index_120hz) + lag,
                int(window.stop_index_120hz_exclusive) + lag,
                dtype=int,
            )
            if response_local[-1] >= len(local_to_cache):
                continue
            response_cache = local_to_cache[response_local]
            if np.any(response_cache < 0):
                continue
            if not np.all(trial[response_local] == int(window.trial_index)):
                continue
            column = int(unit.source_unit_index)
            if not np.all(dfs[response_cache, column] > 0):
                continue
            local_power_full = localized_radial_spectrum(
                movie, ppd=float(ppd), spatial_aperture=rf_apertures[rr100_index]
            )
            local_power = local_power_full[np.ix_(tf_index, sf_index)].astype(float)
            h = sensitivity[rr100_index]
            local_routed_power = float(np.sum(local_power * h**2))
            local_routed_amplitude = float(np.sqrt(max(local_routed_power, 0.0)))
            gain_weighted = local_routed_amplitude * float(unit.extended_rank1_gain_f0_hz)
            whole_crop_routed_power = float(np.sum(whole_crop_power * h**2))
            whole_crop_routed_amplitude = float(np.sqrt(max(whole_crop_routed_power, 0.0)))
            recorded_rate = float(np.mean(robs[response_cache, column]) / dt)
            twin_rate = float(np.mean(rhat[response_cache, column]) / dt)
            rows.append(
                {
                    "window_index": int(window.window_index),
                    "trial_index": int(window.trial_index),
                    "start_index_120hz": int(window.start_index_120hz),
                    "rr100_index": rr100_index,
                    "source_unit_index": column,
                    "peak_lag_bins": lag,
                    "peak_lag_ms": float(unit.peak_lag_ms),
                    "whole_crop_power_amplitude": whole_crop_amplitude,
                    "whole_crop_routed_power": whole_crop_routed_power,
                    "whole_crop_routed_amplitude": whole_crop_routed_amplitude,
                    "local_routed_power": local_routed_power,
                    "local_routed_amplitude": local_routed_amplitude,
                    "gain_weighted_local_routed_amplitude": gain_weighted,
                    "rf_center_x_pixel": float(unit.rf_center_x_pixel),
                    "rf_center_y_pixel": float(unit.rf_center_y_pixel),
                    "rf_rms_radius_pixel": float(unit.rf_rms_radius_pixel),
                    "rf_radius95_pixel": float(unit.rf_radius95_pixel),
                    "recorded_mean_rate_hz": recorded_rate,
                    "full_twin_mean_rate_hz": twin_rate,
                    "recorded_spike_count": float(np.sum(robs[response_cache, column])),
                    "full_twin_expected_spikes": float(np.sum(rhat[response_cache, column])),
                }
            )
    return pd.DataFrame(rows)


def fit_and_score(
    rows: pd.DataFrame, n_folds: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    augmented = []
    metrics = []
    fold_rows = []
    trial_rows = []
    for unit, frame in rows.groupby("rr100_index", sort=True):
        frame = frame.sort_values("start_index_120hz").reset_index(drop=True).copy()
        trial = frame["trial_index"].to_numpy(int)
        folds, trial_table = assign_trial_folds(trial, n_folds)
        observed = frame["recorded_mean_rate_hz"].to_numpy(float)
        twin = frame["full_twin_mean_rate_hz"].to_numpy(float)
        routed = frame["gain_weighted_local_routed_amplitude"].to_numpy(float)
        whole_crop_routed = frame["whole_crop_routed_amplitude"].to_numpy(float)
        power_prediction, power_baseline, fit_table = grouped_cv_line(
            routed, observed, folds, nonnegative=True
        )
        signed_prediction, _, signed_fit = grouped_cv_line(
            routed, observed, folds, nonnegative=False
        )
        global_prediction, global_baseline, global_fit = grouped_cv_line(
            whole_crop_routed, observed, folds, nonnegative=True
        )
        frame["fold"] = folds
        frame["power_predicted_rate_hz"] = power_prediction
        frame["power_increment_hz"] = power_prediction - power_baseline
        frame["power_signal_rate_hz"] = float(np.mean(observed)) + frame["power_increment_hz"]
        frame["signed_power_predicted_rate_hz"] = signed_prediction
        frame["global_power_predicted_rate_hz"] = global_prediction
        frame["global_power_increment_hz"] = global_prediction - global_baseline
        frame["global_power_signal_rate_hz"] = float(np.mean(observed)) + frame["global_power_increment_hz"]
        frame["cv_training_mean_rate_hz"] = power_baseline
        augmented.append(frame)
        for predictor, table in (
            ("rf_local_routed_nonnegative", fit_table),
            ("rf_local_routed_signed_diagnostic", signed_fit),
            ("whole_crop_routed_nonnegative_control", global_fit),
        ):
            table = table.copy()
            table.insert(0, "rr100_index", int(unit))
            table.insert(1, "predictor", predictor)
            fold_rows.append(table)
        within_observed = trial_demean(observed, trial)
        within_twin = trial_demean(twin, trial)
        within_power = trial_demean(power_prediction, trial)
        within_global = trial_demean(global_prediction, trial)
        unit_trial_rows = []
        for trial_index in np.unique(trial):
            mask = trial == trial_index
            unit_trial_rows.append(
                {
                    "rr100_index": int(unit),
                    "trial_index": int(trial_index),
                    "n_windows": int(np.count_nonzero(mask)),
                    "raw_local_routed_power_vs_recorded_r": safe_pearson(routed[mask], observed[mask]),
                    "full_twin_vs_recorded_r": safe_pearson(twin[mask], observed[mask]),
                }
            )
        trial_rows.extend(unit_trial_rows)
        trial_power_r = np.asarray(
            [row["raw_local_routed_power_vs_recorded_r"] for row in unit_trial_rows], dtype=float
        )
        finite_trial_power = np.isfinite(trial_power_r)
        power_slopes = fit_table["slope"].to_numpy(float)
        metrics.append(
            {
                "rr100_index": int(unit),
                "n_windows": len(frame),
                "n_trials": int(np.unique(trial).size),
                "recorded_mean_rate_hz": float(np.mean(observed)),
                "recorded_rate_sd_hz": float(np.std(observed)),
                "full_twin_mean_rate_hz": float(np.mean(twin)),
                "full_twin_vs_recorded_pearson_r": safe_pearson(twin, observed),
                "full_twin_vs_recorded_spearman_rho": safe_spearman(twin, observed),
                "full_twin_vs_recorded_within_trial_r": safe_pearson(within_twin, within_observed),
                "power_vs_recorded_pearson_r": safe_pearson(
                    frame["power_signal_rate_hz"].to_numpy(float), observed
                ),
                "power_vs_recorded_spearman_rho": safe_spearman(
                    frame["power_signal_rate_hz"].to_numpy(float), observed
                ),
                "power_full_cv_prediction_vs_recorded_r": safe_pearson(power_prediction, observed),
                "power_vs_recorded_within_trial_r": safe_pearson(within_power, within_observed),
                "power_vs_recorded_cv_r2": cv_r2(observed, power_prediction, power_baseline),
                "signed_power_vs_recorded_pearson_r": safe_pearson(signed_prediction, observed),
                "whole_crop_power_vs_recorded_pearson_r": safe_pearson(
                    frame["global_power_signal_rate_hz"].to_numpy(float), observed
                ),
                "whole_crop_power_vs_recorded_within_trial_r": safe_pearson(within_global, within_observed),
                "whole_crop_power_vs_recorded_cv_r2": cv_r2(observed, global_prediction, global_baseline),
                "raw_local_routed_power_vs_recorded_r": safe_pearson(routed, observed),
                "raw_local_routed_power_vs_full_twin_r": safe_pearson(routed, twin),
                "power_prediction_vs_full_twin_r": safe_pearson(power_prediction, twin),
                "median_nonnegative_power_slope": float(fit_table["slope"].median()),
                "fraction_positive_nonnegative_slopes": float(np.mean(power_slopes > 0)),
                "fraction_positive_signed_slopes": float(np.mean(signed_fit["slope"] > 0)),
                "median_within_trial_raw_power_vs_recorded_r": (
                    float(np.median(trial_power_r[finite_trial_power]))
                    if np.any(finite_trial_power)
                    else float("nan")
                ),
                "fraction_trials_positive_raw_power_vs_recorded_r": (
                    float(np.mean(trial_power_r[finite_trial_power] > 0))
                    if np.any(finite_trial_power)
                    else float("nan")
                ),
            }
        )
        trial_table.insert(0, "rr100_index", int(unit))
    return (
        pd.concat(augmented, ignore_index=True),
        pd.DataFrame(metrics),
        pd.concat(fold_rows, ignore_index=True),
        pd.DataFrame(trial_rows),
    )


def select_units(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = []
    used: set[int] = set()

    def add(role: str, score: pd.Series, criterion: str, direction: str) -> None:
        available = metrics.loc[~metrics.rr100_index.isin(used)].copy()
        if available.empty:
            return
        available_score = score.reindex(available.index)
        finite = np.isfinite(available_score.to_numpy(float))
        available = available.loc[finite]
        available_score = available_score.loc[finite]
        if available.empty:
            return
        index = available_score.idxmax() if direction == "max" else available_score.idxmin()
        row = metrics.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_value"] = float(score.loc[index])
        selected.append(row)
        used.add(int(row.rr100_index))

    add(
        "strongest power-to-recorded tracking",
        metrics.power_vs_recorded_within_trial_r,
        "maximum within-trial Pearson r between held-out routed-power prediction and recorded rate",
        "max",
    )
    add(
        "full twin succeeds more than power",
        metrics.full_twin_vs_recorded_within_trial_r - metrics.power_vs_recorded_within_trial_r,
        "maximum full-twin minus power within-trial recorded-response correlation",
        "max",
    )
    add(
        "power tracks twin more than recording",
        metrics.raw_local_routed_power_vs_full_twin_r
        - metrics.power_vs_recorded_within_trial_r.fillna(0.0),
        "maximum raw-power/full-twin r minus within-trial power/recorded r (undefined set to zero)",
        "max",
    )
    add(
        "weak tracking control",
        metrics[["full_twin_vs_recorded_within_trial_r", "power_vs_recorded_within_trial_r"]].max(axis=1),
        "minimum of the better within-trial tracking correlation",
        "min",
    )
    return pd.DataFrame(selected)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.mean(values)) / max(float(np.std(values)), 1e-12)


def rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(window, center=True, min_periods=1).mean().to_numpy()


def metric_text(value: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:+.{digits}f}"


def plot_checkpoint(
    selected: pd.DataFrame,
    predictions: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    sensitivity: dict[int, np.ndarray],
    rf_footprints: dict[int, np.ndarray],
    sf: np.ndarray,
    tf: np.ndarray,
    out: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(len(selected), 6, figsize=(24, 3.4 * len(selected)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_number, selection in enumerate(selected.itertuples(index=False)):
        unit = int(selection.rr100_index)
        frame = predictions.loc[predictions.rr100_index.eq(unit)].sort_values("start_index_120hz")
        observed = frame.recorded_mean_rate_hz.to_numpy(float)
        twin = frame.full_twin_mean_rate_hz.to_numpy(float)
        power = frame.power_signal_rate_hz.to_numpy(float)
        routed = frame.gain_weighted_local_routed_amplitude.to_numpy(float)
        representative = frame.iloc[int(np.argmin(np.abs(routed - np.median(routed))))]
        movie = payload[int(representative.window_index)]["movie_uint8"]
        footprint = rf_footprints[unit]

        axes[row_number, 0].imshow(movie[len(movie) // 2], cmap="gray", vmin=0, vmax=255)
        axes[row_number, 0].imshow(footprint, cmap="viridis", alpha=0.52 * footprint / footprint.max())
        axes[row_number, 0].contour(
            footprint,
            levels=[0.1 * footprint.max(), 0.5 * footprint.max()],
            colors=["white", "#D55E00"],
            linewidths=[1.0, 1.4],
        )
        axes[row_number, 0].plot(
            selection.rf_center_x_pixel,
            selection.rf_center_y_pixel,
            "+",
            color="#D55E00",
            ms=9,
            mew=1.8,
        )
        axes[row_number, 0].set_title(
            f"unit-specific input aperture\ncenter "
            f"({selection.rf_center_x_pixel:.1f}, {selection.rf_center_y_pixel:.1f}) px; "
            f"r95={selection.rf_radius95_pixel:.1f} px"
        )
        axes[row_number, 0].axis("off")

        image = axes[row_number, 1].imshow(
            sensitivity[unit] ** 2, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1
        )
        sf_ticks = np.arange(len(sf))
        tf_tick_indices = np.asarray([0, 3, 7, 11, 15, 17], dtype=int)
        axes[row_number, 1].set_xticks(sf_ticks, [f"{value:.2g}" for value in sf], rotation=45)
        axes[row_number, 1].set_yticks(tf_tick_indices, [f"{tf[i]:g}" for i in tf_tick_indices])
        axes[row_number, 1].set(
            xlabel="SF (cpd)", ylabel="TF (Hz)", title=f"{selection.selection_role}\nRR100 {unit} fitted passband²"
        )
        figure.colorbar(image, ax=axes[row_number, 1], label="normalized sensitivity²", fraction=0.046)

        x = np.arange(len(frame))
        axes[row_number, 2].scatter(x, zscore(observed), s=8, color="0.2", alpha=0.22, label="recorded raw")
        axes[row_number, 2].plot(x, rolling(zscore(observed)), color="black", lw=1.5, label="recorded")
        axes[row_number, 2].plot(x, rolling(zscore(twin)), color="#0072B2", lw=1.5, label="full twin")
        axes[row_number, 2].plot(
            x, rolling(zscore(power)), color="#E69F00", lw=1.5, label="power-dependent signal"
        )
        axes[row_number, 2].set(
            xlabel="chronological nonoverlapping window",
            ylabel="within-unit z-score",
            title=f"window-to-window changes\n{len(frame)} windows, {frame.trial_index.nunique()} trials",
        )
        axes[row_number, 2].legend(frameon=False, fontsize=7, ncol=2)

        axes[row_number, 3].scatter(twin, observed, s=18, alpha=0.55, color="#0072B2")
        axes[row_number, 3].set(
            xlabel="full twin mean rate (Hz)",
            ylabel="recorded mean rate (Hz)",
            title=(
                f"full model → recording\nr={selection.full_twin_vs_recorded_pearson_r:+.2f}; "
                f"within trial={selection.full_twin_vs_recorded_within_trial_r:+.2f}"
            ),
        )

        axes[row_number, 4].scatter(power, observed, s=18, alpha=0.55, color="#E69F00")
        axes[row_number, 4].set(
            xlabel="power-dependent rate signal (Hz; common mean)",
            ylabel="recorded mean rate (Hz)",
            title=(
                f"power → recording\nwithin trial="
                f"{metric_text(selection.power_vs_recorded_within_trial_r)}; "
                f"trial median={metric_text(selection.median_within_trial_raw_power_vs_recorded_r)}\n"
                f"held-out ΔR²={metric_text(selection.power_vs_recorded_cv_r2, 3)}"
            ),
        )

        axes[row_number, 5].scatter(routed, twin, s=18, alpha=0.55, color="#009E73")
        axes[row_number, 5].set(
            xlabel="RF-local gain-weighted routed amplitude (a.u.)",
            ylabel="full twin mean rate (Hz)",
            title=f"RF-local power → full model\nr={selection.raw_local_routed_power_vs_full_twin_r:+.2f}",
        )
        for axis in axes[row_number, 3:]:
            axis.grid(alpha=0.2)

    figure.suptitle(
        "Held-out recorded gratings: full twin, routed-power prediction, and actual unit response\n"
        "RF-local Fourier power · exact 40-frame gaze-cropped movies · unit-specific lag · whole trials held out",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dset, local, _ = load_heldout_grating_dataset(args.dataset_config, args.session)
    window_metrics, payload, candidate_sf, candidate_tf, ppd = candidate_windows(
        dset, local, int(args.stride), 0, session=args.session
    )
    units, sensitivity, power_sf, power_tf, rf_footprints, rf_apertures = unit_table(args)
    cached = load_session_cache(args.response_cache, args.session)
    alignment = pd.read_csv(args.alignment_csv)
    alignment_row = alignment.loc[alignment.session.eq(args.session)]
    if len(alignment_row) != 1 or not bool(alignment_row.validation_length_alignment.iloc[0]):
        raise ValueError(f"Missing passing response-cache timeline alignment for {args.session}")
    alignment_basis = str(alignment_row.alignment_basis.iloc[0])
    rows = build_window_unit_table(
        window_metrics,
        payload,
        units,
        sensitivity,
        power_sf,
        power_tf,
        candidate_sf,
        candidate_tf,
        ppd,
        rf_apertures,
        dset,
        local,
        cached,
        alignment_basis,
    )
    predictions, metrics, fold_fits, trial_tracking = fit_and_score(rows, int(args.n_folds))
    metrics = metrics.merge(
        units[
            [
                "rr100_index",
                "session",
                "source_unit_index",
                "peak_lag_bins",
                "peak_lag_ms",
                "extended_rank1_centered_r2",
                "extended_sf_fit_r2",
                "extended_tf_fit_r2",
                "preferred_sf_cpd",
                "extended_tf_center_frequency",
                "readout_mean_y_feature_pixel",
                "readout_mean_x_feature_pixel",
                "readout_std_y_feature_pixel",
                "readout_std_x_feature_pixel",
                "readout_theta_radian",
                "rf_center_x_pixel",
                "rf_center_y_pixel",
                "rf_rms_radius_pixel",
                "rf_radius95_pixel",
            ]
        ],
        on="rr100_index",
        how="left",
        validate="one_to_one",
    )
    selected = select_units(metrics)
    predictions.to_csv(args.out_dir / "window_unit_three_way_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "unit_three_way_metrics.csv", index=False)
    fold_fits.to_csv(args.out_dir / "power_rate_cv_fold_fits.csv", index=False)
    trial_tracking.to_csv(args.out_dir / "unit_trial_tracking.csv", index=False)
    units.to_csv(args.out_dir / "unit_rf_metadata.csv", index=False)
    ordered_rf_units = units.rr100_index.to_numpy(dtype=np.int64)
    np.savez_compressed(
        args.out_dir / "unit_rf_apertures.npz",
        rr100_index=ordered_rf_units,
        rf_footprint=np.stack([rf_footprints[int(unit)] for unit in ordered_rf_units]),
        spectral_aperture=np.stack([rf_apertures[int(unit)] for unit in ordered_rf_units]),
    )
    selected.to_csv(args.out_dir / "selected_units.csv", index=False)
    figure_base = args.out_dir / "recorded_grating_three_way_response_checkpoint"
    plot_checkpoint(
        selected,
        predictions,
        payload,
        sensitivity,
        rf_footprints,
        power_sf,
        power_tf,
        figure_base,
        int(args.dpi),
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_recorded_grating_three_way_response_rf_local_checkpoint",
        "status": "one_session_map_first_checkpoint_complete",
        "session": args.session,
        "scope": {
            "n_routing_qualified_units": int(len(metrics)),
            "n_input_windows": int(window_metrics.window_index.nunique()),
            "window_frames": N_SCORE,
            "window_duration_s": N_SCORE / FRAME_RATE_HZ,
            "n_trial_folds": int(args.n_folds),
        },
        "three_distinct_response_objects": {
            "full_twin": "cached rhat from the full fitted twin on the exact held-out grating input and behavior",
            "power_prediction": "nonnegative linear mapping from fixed unit-specific routed SFxTF amplitude to recorded mean rate, calibrated only on other trials",
            "recorded": "recorded spike counts (robs) converted to mean Hz over the lag-aligned window",
        },
        "contracts": {
            "input": "exact stored gaze-cropped and shifter-corrected 51x51 retinal frames",
            "power": "40-frame positive-TF SF-bin spectrum restricted to the fitted unit's RF aperture; same 3-54 Hz by fitted-SF support as the corrected natural-image routing analysis",
            "rf_localization": "fitted 14x14 Gaussian readout mask back-projected through the feedforward 5x5, 2x2 stride-2 pool, 9x9, and 7x7 spatial support onto the 51x51 input; sqrt of normalized footprint is the Fourier aperture",
            "whole_crop_control": "the former whole-51x51 routed spectrum is retained only as a diagnostic control and never supplies the primary power prediction",
            "unit_filter": "recorded-validated, routing-qualified extended native F0 passband",
            "response_alignment": f"{alignment_basis}; each input window shifted by the unit-specific recorded grating peak lag; target rhat and robs use identical cached bins",
            "cross_validation": "entire held-out experimental trials assigned to one of five deterministic folds",
            "primary_tracking_metric": "Pearson correlation after demeaning recorded and predicted window rates within trial",
            "power_change_visualization": "power-dependent increment above each training-fold baseline, recentered to one common observed mean; fold baselines are excluded from change correlations and plots",
        },
        "caveats": [
            "This checkpoint covers one session and is not a population conclusion.",
            "Recorded 333-ms window rates contain substantial finite-spike noise.",
            "The grating movie rapidly changes nominal SF and orientation, so retinal power contains display-sequence and gaze contributions.",
            "The primary power mapping is linear and nonnegative; nonlinear mappings are not yet adjudicated.",
            "The RF aperture back-projects the feedforward fitted readout footprint and intentionally does not expand through recurrent ConvGRU state propagation.",
        ],
        "inputs": {
            "dataset_config": file_identity(args.dataset_config),
            "response_cache": file_identity(args.response_cache),
            "mapping": file_identity(args.mapping_csv),
            "grating_metrics": file_identity(args.grating_metrics_csv),
            "cache_alignment": file_identity(args.alignment_csv),
            "routing_arrays": file_identity(args.routing_data_dir / "power_routing_joined_arrays.npz"),
            "fitted_model_checkpoint": file_identity(MODEL_CHECKPOINT),
        },
        "artifacts": {
            "window_predictions": "window_unit_three_way_predictions.csv",
            "unit_metrics": "unit_three_way_metrics.csv",
            "fold_fits": "power_rate_cv_fold_fits.csv",
            "trial_tracking": "unit_trial_tracking.csv",
            "unit_rf_metadata": "unit_rf_metadata.csv",
            "unit_rf_apertures": "unit_rf_apertures.npz",
            "selected_units": "selected_units.csv",
            "figure_png": "recorded_grating_three_way_response_checkpoint.png",
            "figure_pdf": "recorded_grating_three_way_response_checkpoint.pdf",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        metrics[
            [
                "rr100_index",
                "n_windows",
                "full_twin_vs_recorded_within_trial_r",
                "power_vs_recorded_within_trial_r",
                "power_vs_recorded_cv_r2",
                "raw_local_routed_power_vs_full_twin_r",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote three-way checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
