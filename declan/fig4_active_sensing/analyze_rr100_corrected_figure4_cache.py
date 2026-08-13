#!/usr/bin/env python3
"""Audit and analyze the corrected native-time Figure-4 RR100 SSI cache.

The primary contrast is the exact original-natural-motion movie versus the
stabilized movie for every crossed image x trace pair (16 x 32 = 512).  The
script keeps mean-rate and spatial-SSI effects separate, summarizes their
shared population structure, and repeats the retinal-power predictor test
with image- and trace-disjoint cross-validation.
"""

from __future__ import annotations

import hashlib
import gc
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import (
    FRAME_RATE_HZ,
    SF_FIT_MAX_CPD,
    SF_FIT_MIN_CPD,
    TF_FIT_MAX_HZ,
    TF_FIT_MIN_HZ,
    radialize_power,
    spectral_decomposition,
)
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.make_rr100_kuang_unit_overlap_checkpoint import surface
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
TRACE_CACHE = RUN / (
    "map_first_power_shift_checkpoint_09_parametric_population_bridge_v1/"
    "checkpoint_09_reconstructed_traces.npz"
)
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
SURFACE = ROOT / (
    "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1/"
    "f0_surface_fit_and_residual_points.csv"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_figure4_cache_checkpoint_16_v1"
EPS = 1e-15
SEED = 20260812
N_SPLITS = 20


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def json_ready(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def quality_mask(models: pd.DataFrame) -> pd.Series:
    return (
        models["model_valid"].astype(bool)
        & models["sf_fit_r2"].ge(0.70)
        & models["tf_fit_r2"].ge(0.70)
        & models["joint_parametric_surface_r2"].ge(0.50)
    )


def cache_audit() -> tuple[dict, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    summary = json.loads((RUN / "summary.json").read_text())
    observations = pd.read_csv(RUN / "retiming_population_observations.csv")
    conditions = pd.read_csv(RUN / "condition_table.csv")
    archive = np.load(RUN / "retiming_ssi.npz")

    expected_rows = 16 * 32 * 32
    key_cols = ["image_index", "trace_index", "condition_index"]
    if len(observations) != expected_rows:
        raise ValueError(f"Expected {expected_rows} observations; found {len(observations)}")
    if observations.duplicated(key_cols).any():
        raise ValueError("Duplicate image x trace x condition keys in corrected cache")
    if not np.array_equal(observations["observation_index"].to_numpy(int), np.arange(expected_rows)):
        raise ValueError("Observation rows are not aligned to NPZ rows")
    if sorted(observations["frame_rate_hz"].unique()) != [120.0]:
        raise ValueError("Corrected cache is not uniformly sampled at 120 Hz")
    if sorted(observations["total_frames"].unique()) != [32]:
        raise ValueError("Corrected cache does not contain uniformly 32-frame movies")
    if conditions["condition_index"].tolist() != list(range(32)):
        raise ValueError("Condition table indices are incomplete or unordered")

    arrays = {
        key: np.asarray(archive[key], dtype=float)
        for key in (
            "unit_bits_per_movie",
            "unit_expected_spikes_per_movie",
            "unit_mean_rate_per_movie",
            "population_bits_per_movie",
        )
    }
    for key, values in arrays.items():
        if values.shape[0] != expected_rows or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid corrected-cache array {key}: {values.shape}")

    row_grid = np.full((32, 16, 32), -1, dtype=int)
    for row in observations.itertuples():
        row_grid[int(row.condition_index), int(row.image_index), int(row.trace_index)] = int(row.observation_index)
    if np.any(row_grid < 0):
        raise ValueError("Incomplete corrected-cache condition grid")

    static_rows = row_grid[0]
    static_ranges = {}
    for key in ("unit_bits_per_movie", "unit_expected_spikes_per_movie", "unit_mean_rate_per_movie"):
        values = arrays[key][static_rows]
        static_ranges[key] = float(np.ptp(values, axis=1).max())
        if static_ranges[key] != 0.0:
            raise ValueError(f"Static baseline unexpectedly varies over trace index for {key}")

    rate_spike_error = float(
        np.max(
            np.abs(
                arrays["unit_expected_spikes_per_movie"]
                - arrays["unit_mean_rate_per_movie"] * float(summary["n_timepoints"]) * float(summary["bin_seconds"])
            )
        )
    )
    audit = {
        "status": "pass",
        "intended_grain": "one 32-frame movie per image x trace x timing condition",
        "n_rows": int(len(observations)),
        "n_images": int(observations["image_index"].nunique()),
        "n_traces": int(observations["trace_index"].nunique()),
        "n_conditions": int(observations["condition_index"].nunique()),
        "n_units": int(arrays["unit_mean_rate_per_movie"].shape[1]),
        "duplicate_primary_keys": 0,
        "missing_grid_cells": 0,
        "frame_rate_hz": 120.0,
        "n_frames": 32,
        "movie_duration_ms": 1000.0 * 32.0 / 120.0,
        "original_condition": "original_natural_timing",
        "baseline_condition": "stabilized_static",
        "static_max_trace_range": static_ranges,
        "max_expected_spike_rate_identity_error": rate_spike_error,
        "metric_contracts": {
            "mean_rate": "spatially averaged unit rate, averaged across all 32 movie frames",
            "ssi": "expected-spike-weighted instantaneous spatial information, bits/spike, across the 32-frame movie",
            "effect": "original_natural_timing minus stabilized_static for the identical image",
        },
        "source_identity": {
            "summary": file_identity(RUN / "summary.json"),
            "observations": file_identity(RUN / "retiming_population_observations.csv"),
            "responses": file_identity(RUN / "retiming_ssi.npz"),
            "traces": file_identity(TRACE_CACHE),
        },
    }
    return audit, observations, arrays, row_grid


def pca_result(
    matrix: np.ndarray, *, compute_pairwise: bool = True, min_effect_sd: float = EPS
) -> dict[str, np.ndarray | float | int]:
    x = np.asarray(matrix, dtype=float)
    scale = x.std(axis=0, ddof=0)
    good = scale > float(min_effect_sd)
    z = (x[:, good] - x[:, good].mean(axis=0, keepdims=True)) / scale[good]
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    score = u[:, 0] * s[0]
    loading = vt[0].copy()
    if np.corrcoef(score, z.mean(axis=1))[0, 1] < 0:
        score *= -1
        loading *= -1
    if compute_pairwise:
        corr = stats.spearmanr(z, axis=0).statistic
        median_pairwise = float(np.nanmedian(corr[np.triu_indices_from(corr, 1)]))
    else:
        median_pairwise = float("nan")
    return {
        "fraction": float(s[0] ** 2 / np.sum(s**2)),
        "score": score,
        "loading": loading,
        "z": z,
        "good": good,
        "median_pairwise_spearman": median_pairwise,
        "n_units": int(np.sum(good)),
    }


def two_way_decomposition(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    grand = float(x.mean())
    image = x.mean(axis=1) - grand
    trace = x.mean(axis=0) - grand
    residual = x - grand - image[:, None] - trace[None, :]
    ss = np.array(
        [x.shape[1] * np.sum(image**2), x.shape[0] * np.sum(trace**2), np.sum(residual**2)],
        dtype=float,
    )
    fractions = ss / max(float(ss.sum()), EPS)
    return {
        "image_fraction": float(fractions[0]),
        "trace_fraction": float(fractions[1]),
        "interaction_fraction": float(fractions[2]),
    }


def split_reliability(cube: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    result = pca_result(cube.reshape(-1, cube.shape[-1]))
    good_cube = cube[:, :, np.asarray(result["good"], dtype=bool)]
    z = np.asarray(result["z"], dtype=float)
    rows = []
    for split_index in range(N_SPLITS):
        units = rng.permutation(good_cube.shape[-1])
        a = z[:, units[: len(units) // 2]].mean(axis=1)
        b = z[:, units[len(units) // 2 :]].mean(axis=1)
        rows.append({"split_type": "unit_half_scores", "split_index": split_index, "pearson_r": stats.pearsonr(a, b).statistic})

        images = rng.permutation(cube.shape[0])
        la = np.asarray(pca_result(good_cube[images[:8]].reshape(-1, good_cube.shape[-1]), compute_pairwise=False)["loading"])
        lb = np.asarray(pca_result(good_cube[images[8:]].reshape(-1, good_cube.shape[-1]), compute_pairwise=False)["loading"])
        rows.append({"split_type": "image_half_loadings", "split_index": split_index, "pearson_r": abs(stats.pearsonr(la, lb).statistic)})

        traces = rng.permutation(cube.shape[1])
        la = np.asarray(pca_result(good_cube[:, traces[:16]].reshape(-1, good_cube.shape[-1]), compute_pairwise=False)["loading"])
        lb = np.asarray(pca_result(good_cube[:, traces[16:]].reshape(-1, good_cube.shape[-1]), compute_pairwise=False)["loading"])
        rows.append({"split_type": "trace_half_loadings", "split_index": split_index, "pearson_r": abs(stats.pearsonr(la, lb).statistic)})
    return pd.DataFrame(rows)


def rank_analysis(arrays: dict[str, np.ndarray], row_grid: np.ndarray) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static = row_grid[0]
    original = row_grid[1]
    cubes = {
        "mean_rate_signed": arrays["unit_mean_rate_per_movie"][original] - arrays["unit_mean_rate_per_movie"][static],
        "mean_rate_magnitude": np.abs(arrays["unit_mean_rate_per_movie"][original] - arrays["unit_mean_rate_per_movie"][static]),
        "ssi_signed": arrays["unit_bits_per_movie"][original] - arrays["unit_bits_per_movie"][static],
        "ssi_magnitude": np.abs(arrays["unit_bits_per_movie"][original] - arrays["unit_bits_per_movie"][static]),
    }
    rng = np.random.default_rng(SEED)
    summary_rows = []
    reliability_rows = []
    results = {}
    for name, cube in cubes.items():
        response_floor = 1e-4 if name.startswith("mean_rate") else 1e-6
        result = pca_result(cube.reshape(-1, cube.shape[-1]), min_effect_sd=response_floor)
        results[name] = result
        decomposition = two_way_decomposition(np.asarray(result["score"]).reshape(16, 32))
        population_decomposition = two_way_decomposition(cube.mean(axis=2))
        summary_rows.append(
            {
                "metric": name,
                "n_variable_units": result["n_units"],
                "pc1_variance_fraction": result["fraction"],
                "median_pairwise_unit_profile_spearman": result["median_pairwise_spearman"],
                **{f"pc1_score_{k}": v for k, v in decomposition.items()},
                **{f"population_mean_{k}": v for k, v in population_decomposition.items()},
            }
        )
        rel = split_reliability(cube, rng)
        rel.insert(0, "metric", name)
        reliability_rows.append(rel)

    rate_score = np.asarray(results["mean_rate_signed"]["score"])
    ssi_score = np.asarray(results["ssi_signed"]["score"])
    rate_delta = cubes["mean_rate_signed"].reshape(512, 100)
    ssi_delta = cubes["ssi_signed"].reshape(512, 100)
    unit_rows = []
    for unit in range(100):
        rate_rho = stats.spearmanr(rate_delta[:, unit], rate_score).statistic if np.ptp(rate_delta[:, unit]) > EPS else np.nan
        ssi_rho = stats.spearmanr(ssi_delta[:, unit], ssi_score).statistic if np.ptp(ssi_delta[:, unit]) > EPS else np.nan
        unit_rows.append(
            {
                "rr100_index": unit,
                "mean_rate_delta_mean_hz": float(rate_delta[:, unit].mean()),
                "mean_rate_delta_sd_hz": float(rate_delta[:, unit].std()),
                "ssi_delta_mean_bits_per_spike": float(ssi_delta[:, unit].mean()),
                "ssi_delta_sd_bits_per_spike": float(ssi_delta[:, unit].std()),
                "rate_profile_vs_rate_pc1_spearman": rate_rho,
                "ssi_profile_vs_ssi_pc1_spearman": ssi_rho,
            }
        )
    units = pd.DataFrame(unit_rows)
    nondegenerate = units.loc[
        units["mean_rate_delta_sd_hz"].ge(1e-4)
        & units["ssi_delta_sd_bits_per_spike"].ge(1e-4)
    ]
    roles = [
        ("shared_rate_exemplar", int(units["rate_profile_vs_rate_pc1_spearman"].idxmax()), "largest rate-profile correlation with the common rate score"),
        ("shared_ssi_exemplar", int(units["ssi_profile_vs_ssi_pc1_spearman"].idxmax()), "largest SSI-profile correlation with the common SSI score"),
        ("rate_common_ssi_opposite", int((units["rate_profile_vs_rate_pc1_spearman"] - units["ssi_profile_vs_ssi_pc1_spearman"]).idxmax()), "largest rate-common minus SSI-common correlation"),
        ("rate_opposite_ssi_common", int((nondegenerate["ssi_profile_vs_ssi_pc1_spearman"] - nondegenerate["rate_profile_vs_rate_pc1_spearman"]).idxmax()), "largest SSI-common minus rate-common correlation among nondegenerate effects"),
    ]
    selected = []
    for role, row_index, criterion in roles:
        row = units.loc[row_index].to_dict()
        row.update({"selection_role": role, "selection_criterion": criterion})
        selected.append(row)
    return cubes, pd.DataFrame(summary_rows), pd.concat(reliability_rows, ignore_index=True), pd.DataFrame(selected)


def interpolate_empirical(frame: pd.DataFrame, sf: np.ndarray, tf: np.ndarray) -> np.ndarray:
    pivot = frame.pivot(index="spatial_cpd", columns="temporal_hz", values="observed_positive_f0_hz").sort_index().sort_index(axis=1)
    source_sf = pivot.index.to_numpy(float)
    source_tf = pivot.columns.to_numpy(float)
    if sf.min() < source_sf.min() or sf.max() > source_sf.max() or tf.min() < source_tf.min() or tf.max() > source_tf.max():
        raise ValueError("Corrected-cache power support extends beyond the measured grating surface")
    interp = RegularGridInterpolator((np.log2(source_sf), np.log2(source_tf)), pivot.to_numpy(float), bounds_error=True)
    sm, tm = np.meshgrid(np.log2(sf), np.log2(tf), indexing="ij")
    values = np.maximum(interp(np.column_stack([sm.ravel(), tm.ravel()])).reshape(len(sf), len(tf)), 0.0)
    return values / max(float(values.max()), EPS)


def render_with_common(patch: np.ndarray, trace_xy: np.ndarray, *, ppd: float, common) -> np.ndarray:
    """Exact canonical renderer with the helper module loaded only once."""
    import torch

    image = _standardize_uint_like(patch)
    trace = np.asarray(trace_xy, dtype=np.float32)
    full_stack = np.broadcast_to(
        image[None, :, :], (trace.shape[0] + int(common.N_LAGS) + 1, *image.shape)
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    stim = common.make_counterfactual_stim(
        full_stack,
        eye,
        ppd=float(ppd),
        scale_factor=1.0,
        n_lags=int(common.N_LAGS),
        out_size=(51, 51),
    )
    result = stim.detach().cpu().numpy()[:, 0, 0]
    if result.shape[0] >= trace.shape[0] + 1:
        result = result[1 : trace.shape[0] + 1]
    else:
        result = result[: trace.shape[0]]
    if result.shape != (trace.shape[0], 51, 51):
        raise ValueError(f"Unexpected retinal movie shape {result.shape}")
    return result.astype(np.float32, copy=False)


def build_power_and_predictors(observations: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cached_npz = OUT / "corrected_power_cache.npz"
    cached_csv = OUT / "corrected_power_table.csv"
    if cached_npz.exists() and cached_csv.exists():
        cached = np.load(cached_npz)
        return (
            pd.read_csv(cached_csv),
            cached["power"].astype(float),
            cached["sf"].astype(float),
            cached["tf"].astype(float),
        )
    summary = json.loads((RUN / "summary.json").read_text())
    images = pd.read_csv(RUN / "image_feature_table.csv").sort_values("image_index")
    trace_data = np.load(TRACE_CACHE)
    traces = trace_data["trace"].astype(np.float32)
    if not np.array_equal(trace_data["trace_index"], np.arange(32)) or traces.shape != (32, 32, 2):
        raise ValueError("Reconstructed trace cache does not match the corrected 32 x 32 contract")
    source_rows = load_source_rows(Path(summary["source_csv"]))
    ppd_values = observations["patch_patch_ppd"].to_numpy(float)
    if np.ptp(ppd_values) > 1e-9:
        raise ValueError("Corrected cache has inconsistent pixels/degree")
    ppd = float(ppd_values[0])
    common = _load_twin_common()

    powers = []
    power_rows = []
    sf_grid = tf_grid = None
    for image in images.itertuples():
        source = source_row_by_id(source_rows, int(image.source_row))
        patch, _ = _extract_patch(source, canvas_cache={}, patch_size_px=540)
        for trace_index, trace in enumerate(traces):
            movie = render_with_common(np.asarray(patch, np.float32), trace, ppd=ppd, common=common)
            decomp = spectral_decomposition(movie, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
            radial = radialize_power(decomp, ppd=ppd, frame_size=movie.shape[-1])
            sf_all = radial["sf_centers_cpd"].astype(float)
            tf_all = decomp["temporal_frequency_hz"].astype(float)
            annular = radial["dynamic_radial_power"].astype(float) * radial["spatial_mode_count"][:, None]
            sf_mask = (sf_all >= SF_FIT_MIN_CPD) & (sf_all <= SF_FIT_MAX_CPD)
            tf_mask = (tf_all >= TF_FIT_MIN_HZ) & (tf_all <= TF_FIT_MAX_HZ)
            sf = sf_all[sf_mask]
            tf = tf_all[tf_mask]
            power = annular[np.ix_(sf_mask, tf_mask)]
            if sf_grid is None:
                sf_grid, tf_grid = sf, tf
            elif not np.array_equal(sf_grid, sf) or not np.array_equal(tf_grid, tf):
                raise ValueError("Power grids differ across corrected-cache movies")
            powers.append(power)
            power_rows.append(
                {
                    "image_index": int(image.image_index),
                    "trace_index": trace_index,
                    "total_supported_dynamic_power": float(power.sum()),
                    "total_supported_dynamic_power_amplitude": float(np.sqrt(max(power.sum(), 0.0))),
                }
            )
            del movie, decomp, radial, annular, power
        gc.collect()
        print(f"reconstructed corrected retinal power: image {int(image.image_index) + 1}/16", flush=True)
    return pd.DataFrame(power_rows), np.stack(powers), np.asarray(sf_grid), np.asarray(tf_grid)


def crossed_folds() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    image_fold = np.empty(16, dtype=int)
    image_fold[rng.permutation(16)] = np.tile(np.arange(4), 4)
    trace_features = pd.read_csv(RUN / "trace_feature_table.csv").sort_values("rendered_path_length_arcmin")
    trace_fold = np.empty(32, dtype=int)
    trace_fold[trace_features["trace_index"].to_numpy(int)] = np.arange(32) % 4
    return image_fold, trace_fold


def fit_line(x: np.ndarray, y: np.ndarray, nonnegative: bool) -> tuple[float, float]:
    xm, ym = float(x.mean()), float(y.mean())
    denom = float(np.sum((x - xm) ** 2))
    slope = float(np.sum((x - xm) * (y - ym)) / max(denom, EPS))
    if nonnegative:
        slope = max(slope, 0.0)
    return ym - slope * xm, slope


def crossed_cv(x: np.ndarray, y: np.ndarray, nonnegative: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_fold, trace_fold = crossed_folds()
    pred = np.full((16, 32), np.nan)
    baseline = np.full((16, 32), np.nan)
    slopes = np.full((4, 4), np.nan)
    for fi in range(4):
        for ft in range(4):
            train = (image_fold[:, None] != fi) & (trace_fold[None, :] != ft)
            test = (image_fold[:, None] == fi) & (trace_fold[None, :] == ft)
            intercept, slope = fit_line(x[train], y[train], nonnegative=nonnegative)
            pred[test] = intercept + slope * x[test]
            baseline[test] = float(y[train].mean())
            slopes[fi, ft] = slope
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(baseline)):
        raise ValueError("Crossed CV did not predict every image x trace pair")
    return pred, baseline, slopes


def predictor_analysis(
    cubes: dict[str, np.ndarray], power_table: pd.DataFrame, power: np.ndarray, sf: np.ndarray, tf: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(MODELS).sort_values("rr100_index").reset_index(drop=True)
    empirical = pd.read_csv(SURFACE)
    if not np.array_equal(models["rr100_index"].to_numpy(int), np.arange(100)):
        raise ValueError("Parametric model table is not aligned to RR100 indices")
    total = np.sqrt(np.maximum(power.sum(axis=(1, 2)), 0.0)).reshape(16, 32)
    predictor_rows = []
    metric_rows = []
    prediction_rows = []
    for model in models.itertuples():
        unit = int(model.rr100_index)
        if not bool(model.model_valid):
            continue
        parametric_gain = surface(pd.Series(model._asdict()), sf, tf)
        empirical_gain = interpolate_empirical(empirical.loc[empirical["rr100_index"].eq(unit)], sf, tf)
        variants = {
            "total_power": total,
            "parametric_sftf": np.sqrt(np.maximum(np.sum(power * parametric_gain[None, :, :] ** 2, axis=(1, 2)), 0.0)).reshape(16, 32),
            "measured_2d_sftf": np.sqrt(np.maximum(np.sum(power * empirical_gain[None, :, :] ** 2, axis=(1, 2)), 0.0)).reshape(16, 32),
        }
        for image_index in range(16):
            for trace_index in range(32):
                predictor_rows.append(
                    {
                        "image_index": image_index,
                        "trace_index": trace_index,
                        "rr100_index": unit,
                        **{name: float(values[image_index, trace_index]) for name, values in variants.items()},
                        "mean_rate_delta_hz": float(cubes["mean_rate_signed"][image_index, trace_index, unit]),
                        "ssi_delta_bits_per_spike": float(cubes["ssi_signed"][image_index, trace_index, unit]),
                    }
                )
        for outcome_name, outcome_cube in (
            ("mean_rate_delta_hz", cubes["mean_rate_signed"]),
            ("ssi_delta_bits_per_spike", cubes["ssi_signed"]),
        ):
            y = outcome_cube[:, :, unit]
            for variant, x in variants.items():
                pred, baseline, slopes = crossed_cv(x, y, nonnegative=True)
                sse = float(np.sum((y - pred) ** 2))
                base_sse = float(np.sum((y - baseline) ** 2))
                metric_rows.append(
                    {
                        "rr100_index": unit,
                        "outcome": outcome_name,
                        "predictor_variant": variant,
                        "n_image_trace_pairs": 512,
                        "cv_scheme": "4x4 crossed blocks; held-out images and held-out traces are both absent from training",
                        "cv_r2_vs_training_mean": 1.0 - sse / max(base_sse, EPS),
                        "oof_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                        "oof_mae": float(np.mean(np.abs(y - pred))),
                        "oof_pearson_r": float(stats.pearsonr(y.ravel(), pred.ravel()).statistic),
                        "oof_spearman_rho": float(stats.spearmanr(y.ravel(), pred.ravel()).statistic),
                        "positive_slope_folds": int(np.sum(slopes > 0)),
                        "quality_cohort": bool(quality_mask(models).iloc[unit]),
                    }
                )
                for image_index in range(16):
                    for trace_index in range(32):
                        prediction_rows.append(
                            {
                                "image_index": image_index,
                                "trace_index": trace_index,
                                "rr100_index": unit,
                                "outcome": outcome_name,
                                "predictor_variant": variant,
                                "observed": float(y[image_index, trace_index]),
                                "predicted": float(pred[image_index, trace_index]),
                                "training_mean_baseline": float(baseline[image_index, trace_index]),
                            }
                        )
    return pd.DataFrame(predictor_rows), pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def make_example_figure(cubes: dict[str, np.ndarray], selected: pd.DataFrame, audit: dict) -> None:
    fig = plt.figure(figsize=(14.8, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[0.50, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(0.01, 0.82, "Corrected Figure-4 cache contract", fontsize=14, weight="bold", transform=ax.transAxes)
    ax.text(
        0.01,
        0.25,
        "16 images × 32 independently selected eye traces × 32 timing conditions · 32 frames at 120 Hz\n"
        "Panels below use only original natural timing − stabilized static for all 512 crossed image–trace pairs.",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(0.99, 0.52, "No 48-frame resampling\nNo four-DCT response proxy", ha="right", va="center", fontsize=11, color="#136f63", weight="bold", transform=ax.transAxes)

    rate = cubes["mean_rate_signed"]
    ssi = cubes["ssi_signed"]
    for row_idx, row in enumerate(selected.itertuples()):
        unit = int(row.rr100_index)
        for col, (cube, label, cmap) in enumerate(
            [(rate, "Mean-rate change (Hz)", "coolwarm"), (ssi, "Spatial SSI change (bits/spike)", "coolwarm")]
        ):
            ax = fig.add_subplot(gs[row_idx + 1, col])
            values = cube[:, :, unit]
            limit = max(float(np.quantile(np.abs(values), 0.99)), EPS)
            im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
            ax.set_xlabel("Eye-trace index")
            ax.set_ylabel("Image index")
            ax.set_title(f"RR100 {unit} · {str(row.selection_role).replace('_', ' ')}\n{label}", loc="left", fontsize=10.5, weight="bold")
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Checkpoint 16A: concrete effects in the corrected native-time SSI cache", fontsize=16, weight="bold")
    fig.savefig(OUT / "checkpoint_16a_corrected_cache_examples.png", dpi=190, facecolor="white")
    fig.savefig(OUT / "checkpoint_16a_corrected_cache_examples.pdf", facecolor="white")
    plt.close(fig)


def make_population_figure(rank: pd.DataFrame, reliability: pd.DataFrame, cubes: dict[str, np.ndarray], metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0), constrained_layout=True)
    colors = {"mean_rate_signed": "#d65238", "mean_rate_magnitude": "#ef9b69", "ssi_signed": "#356fa3", "ssi_magnitude": "#8ab6d6"}
    labels = {"mean_rate_signed": "Rate\nsigned", "mean_rate_magnitude": "Rate\nmagnitude", "ssi_signed": "SSI\nsigned", "ssi_magnitude": "SSI\nmagnitude"}
    order = list(labels)

    ax = axes[0, 0]
    ax.bar(np.arange(4), 100 * rank.set_index("metric").loc[order, "pc1_variance_fraction"], color=[colors[x] for x in order])
    ax.set_xticks(np.arange(4), [labels[x] for x in order])
    ax.set_ylabel("PC1 variance (%)")
    ax.set_title("A  Shared structure is stronger for mean rate", loc="left", weight="bold")

    ax = axes[0, 1]
    ax.bar(np.arange(4), rank.set_index("metric").loc[order, "median_pairwise_unit_profile_spearman"], color=[colors[x] for x in order])
    ax.set_xticks(np.arange(4), [labels[x] for x in order])
    ax.set_ylabel("Median pairwise unit-profile Spearman ρ")
    ax.set_title("B  Unit ordering is positive but metric-dependent", loc="left", weight="bold")

    ax = axes[0, 2]
    positions = []
    data = []
    tick_labels = []
    pos = 0
    for metric in ("mean_rate_signed", "ssi_signed"):
        for split_type, short in (("unit_half_scores", "units"), ("image_half_loadings", "images"), ("trace_half_loadings", "traces")):
            data.append(reliability.loc[(reliability.metric == metric) & (reliability.split_type == split_type), "pearson_r"].to_numpy())
            positions.append(pos)
            tick_labels.append(f"{labels[metric].splitlines()[0]}\n{short}")
            pos += 1
        pos += 0.5
    parts = ax.violinplot(data, positions=positions, widths=0.72, showextrema=False)
    for body, color in zip(parts["bodies"], [colors["mean_rate_signed"]] * 3 + [colors["ssi_signed"]] * 3):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.32)
    for p, values in zip(positions, data):
        ax.plot([p - 0.20, p + 0.20], [np.median(values)] * 2, color="#172029", lw=2)
    ax.set_xticks(positions, tick_labels)
    ax.set_ylim(-0.05, 1.03)
    ax.set_ylabel("Split-half Pearson r")
    ax.set_title("C  Shared axes replicate in held-out subsets", loc="left", weight="bold")

    rank_idx = rank.set_index("metric")
    ax = axes[1, 0]
    names = ["Image", "Eye trace", "Interaction"]
    x = np.arange(3)
    width = 0.35
    rate_vals = [rank_idx.loc["mean_rate_signed", f"population_mean_{n.lower().replace('eye trace','trace')}_fraction"] for n in names]
    ssi_vals = [rank_idx.loc["ssi_signed", f"population_mean_{n.lower().replace('eye trace','trace')}_fraction"] for n in names]
    ax.bar(x - width / 2, 100 * np.asarray(rate_vals), width, color=colors["mean_rate_signed"], label="mean-rate change")
    ax.bar(x + width / 2, 100 * np.asarray(ssi_vals), width, color=colors["ssi_signed"], label="SSI change")
    ax.set_xticks(x, names)
    ax.set_ylabel("Two-way variance fraction (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("D  Rate is trace-led; SSI is image-led", loc="left", weight="bold")

    ax = axes[1, 1]
    pop_rate = cubes["mean_rate_signed"].mean(axis=2)
    im = ax.imshow(pop_rate, aspect="auto", origin="lower", cmap="magma")
    ax.set(xlabel="Eye-trace index", ylabel="Image index")
    ax.set_title("E  Population mean rate change", loc="left", weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="Hz")

    ax = axes[1, 2]
    quality = metrics.loc[metrics["quality_cohort"]].copy()
    wide = quality.pivot(index=["rr100_index", "outcome"], columns="predictor_variant", values="cv_r2_vs_training_mean")
    rng = np.random.default_rng(SEED)
    outcome_labels = [("mean_rate_delta_hz", "Rate"), ("ssi_delta_bits_per_spike", "SSI")]
    positions = []
    data = []
    colors_plot = []
    ticks = []
    pos = 0
    variant_colors = {"total_power": "#d65238", "parametric_sftf": "#356fa3", "measured_2d_sftf": "#8e6bb3"}
    for outcome, short in outcome_labels:
        frame = wide.xs(outcome, level="outcome")
        for variant, variant_short in (("total_power", "total"), ("parametric_sftf", "fit"), ("measured_2d_sftf", "measured")):
            values = frame[variant].to_numpy(float)
            positions.append(pos); data.append(values); colors_plot.append(variant_colors[variant]); ticks.append(f"{short}\n{variant_short}")
            ax.scatter(np.full(len(values), pos) + rng.uniform(-0.10, 0.10, len(values)), np.clip(values, -1, 1), s=9, color=variant_colors[variant], alpha=0.38)
            ax.plot([pos - 0.20, pos + 0.20], [np.median(values)] * 2, color="#172029", lw=2)
            pos += 1
        pos += 0.5
    ax.axhline(0, color="#777", ls="--", lw=1)
    ax.set_xticks(positions, ticks)
    ax.set_ylim(-1.02, 1.02)
    ax.set_ylabel("Crossed held-out R² (display clipped at −1)")
    ax.set_title("F  Corrected-cache spectral prediction", loc="left", weight="bold")

    for ax in axes.flat:
        ax.grid(color="#e9ecef", lw=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Checkpoint 16B: native-time corrected-cache population analysis", fontsize=16, weight="bold")
    fig.savefig(OUT / "checkpoint_16b_corrected_cache_population_analysis.png", dpi=190, facecolor="white")
    fig.savefig(OUT / "checkpoint_16b_corrected_cache_population_analysis.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit, observations, arrays, row_grid = cache_audit()
    cubes, rank, reliability, selected = rank_analysis(arrays, row_grid)
    power_table, power, sf, tf = build_power_and_predictors(observations)
    predictors, metrics, predictions = predictor_analysis(cubes, power_table, power, sf, tf)

    rank.to_csv(OUT / "corrected_cache_rank_summary.csv", index=False)
    reliability.to_csv(OUT / "corrected_cache_split_half_reliability.csv", index=False)
    selected.to_csv(OUT / "corrected_cache_selected_units.csv", index=False)
    power_table.to_csv(OUT / "corrected_cache_supported_dynamic_power.csv", index=False)
    predictors.to_csv(OUT / "corrected_cache_predictor_and_response_values.csv", index=False)
    metrics.to_csv(OUT / "corrected_cache_crossed_cv_metrics.csv", index=False)
    predictions.to_csv(OUT / "corrected_cache_crossed_cv_predictions.csv", index=False)
    np.savez_compressed(
        OUT / "corrected_cache_effect_and_power_arrays.npz",
        mean_rate_delta_hz=cubes["mean_rate_signed"].astype(np.float32),
        ssi_delta_bits_per_spike=cubes["ssi_signed"].astype(np.float32),
        supported_power=power.astype(np.float64),
        sf_centers_cpd=sf,
        tf_centers_hz=tf,
    )

    make_example_figure(cubes, selected, audit)
    make_population_figure(rank, reliability, cubes, metrics)

    quality = metrics.loc[metrics["quality_cohort"]]
    medians = quality.groupby(["outcome", "predictor_variant"])["cv_r2_vs_training_mean"].median().unstack()
    wins = {}
    for outcome, frame in quality.groupby("outcome"):
        wide = frame.pivot(index="rr100_index", columns="predictor_variant", values="cv_r2_vs_training_mean")
        wins[outcome] = {
            "parametric_beats_total_fraction": float(np.mean(wide["parametric_sftf"] > wide["total_power"])),
            "measured_2d_beats_total_fraction": float(np.mean(wide["measured_2d_sftf"] > wide["total_power"])),
        }
    manifest = {
        "analysis": "corrected_figure4_native_time_cache_checkpoint_16",
        "audit": audit,
        "rank_summary": rank.to_dict("records"),
        "split_half_medians": reliability.groupby(["metric", "split_type"])["pearson_r"].median().unstack().to_dict("index"),
        "n_quality_fit_units": int(quality["rr100_index"].nunique()),
        "crossed_cv_median_r2": medians.to_dict("index"),
        "crossed_cv_win_fractions": wins,
        "power_grid": {"sf_cpd": sf, "tf_hz": tf},
        "interpretation": (
            "The corrected cache supports a shared mean-rate effect across crossed image-trace pairs, but the common rate axis is primarily trace-driven. "
            "Spatial-SSI changes are less rank-1 and primarily image-driven. Detailed grating SFxTF weighting is compared against total supported dynamic power under image- and trace-disjoint crossed CV."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Corrected Figure-4 native-time cache checkpoint 16\n\n"
        "This analysis uses the scored Figure-4 SSI/temporal-remapping cache directly: 16 images crossed with 32 traces and 32 timing conditions, all rendered as 32 frames at 120 Hz. "
        "The primary contrast is original natural timing minus stabilized static for all 512 image-trace combinations. Mean-rate and spatial-SSI effects are kept separate. "
        "Retinal SFxTF power is reconstructed from the exact saved 32-frame traces, and predictor performance uses crossed image- and trace-disjoint folds.\n",
        encoding="utf-8",
    )

    print(rank.to_string(index=False))
    print("\nCorrected-cache crossed-CV median R2 (quality units):")
    print(medians.to_string())


if __name__ == "__main__":
    main()
