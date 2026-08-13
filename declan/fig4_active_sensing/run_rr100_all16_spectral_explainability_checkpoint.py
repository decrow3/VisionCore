#!/usr/bin/env python3
"""Checkpoint 11: held-out-image test of FEM spectral-drive explainability.

The analysis uses every eligible image/eye-trace pair in the existing 16-image
Figure-4 pool.  No image or trajectory rotations enter the primary analysis.
For each exact retinal movie it computes orientation-collapsed SF x TF power,
weights that power by each RR100 unit's fixed-retina parametric F0 sensitivity,
and asks whether the resulting amplitude-like drive predicts the frozen unit's
FEM-minus-zero response modulation on a completely held-out image.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr, spearmanr

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import (
    FRAME_RATE_HZ,
    SF_FIT_MAX_CPD,
    SF_FIT_MIN_CPD,
    TF_FIT_MAX_HZ,
    TF_FIT_MIN_HZ,
    radialize_power,
    render_retinal_movie,
    spectral_decomposition,
    support_summary,
)
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import (
    one_trace_from_source,
    source_row_by_id,
)
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import (
    load_source_rows,
)
from declan.fig4_active_sensing.make_rr100_kuang_unit_overlap_checkpoint import surface
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import run_condition
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _load_twin_common,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "outputs/active_sensing_movie_information/temporal_remapping/backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
SOURCE_CSV = ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
MODELS_CSV = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_models.csv"
MAPPING_CSV = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
CACHE08 = ROOT / "outputs/fig4_active_sensing/rr100_multiimage_trajectory_generalization_checkpoint_08_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
ALL_UNITS = np.arange(100, dtype=np.int64)
EPS = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--models-csv", type=Path, default=MODELS_CSV)
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING_CSV)
    parser.add_argument("--cache08-dir", type=Path, default=CACHE08)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def quality_mask(models: pd.DataFrame) -> pd.Series:
    return (
        models["model_valid"].astype(bool)
        & models["sf_fit_r2"].ge(0.70)
        & models["tf_fit_r2"].ge(0.70)
        & models["joint_parametric_surface_r2"].ge(0.50)
    )


def eligible_images(feature_table: pd.DataFrame) -> pd.DataFrame:
    selected = feature_table.loc[
        feature_table["image_feature_ok"].astype(bool)
        & feature_table["image_patch_fraction_inside_image"].ge(0.99)
    ].copy().sort_values("image_index").reset_index(drop=True)
    if len(selected) != 16 or not np.array_equal(selected["image_index"].to_numpy(int), np.arange(16)):
        raise ValueError(f"Expected the exact eligible image_index 0..15 pool; got {selected['image_index'].tolist()}")
    selected.insert(0, "analysis_order", np.arange(1, len(selected) + 1))
    selected.insert(1, "selection_role", "complete_existing_eligible_pool_no_neural_selection")
    return selected


def build_movies(images: pd.DataFrame, source_rows: pd.DataFrame, ppd: float) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    movies: dict[str, np.ndarray] = {}
    audit = []
    for _, row in images.iterrows():
        image_index = int(row["image_index"])
        source_row_id = int(row["source_row"])
        source_row = source_row_by_id(source_rows, source_row_id)
        patch, _ = _extract_patch(source_row, canvas_cache={}, patch_size_px=540)
        trace = one_trace_from_source(
            source_rows, source_row_id, n_timepoints=128, bin_seconds=1.0 / FRAME_RATE_HZ
        ).astype(np.float32)
        trace -= trace.mean(axis=0, keepdims=True)
        zero = render_retinal_movie(np.asarray(patch, dtype=np.float32), np.zeros_like(trace), ppd=ppd)
        fem = render_retinal_movie(np.asarray(patch, dtype=np.float32), trace, ppd=ppd)
        movies[f"image_{image_index:02d}_zero"] = zero
        movies[f"image_{image_index:02d}_fem"] = fem
        decomp = spectral_decomposition(fem, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
        audit.append({
            "image_index": image_index,
            "source_row": source_row_id,
            "session": str(row["session"]),
            "trajectory_definition": "original_measured_centered_eye_trace",
            "trace_rms_radius_deg": float(np.sqrt(np.mean(np.sum(trace**2, axis=1)))),
            "trace_step_rms_deg": float(np.sqrt(np.mean(np.sum(np.diff(trace, axis=0) ** 2, axis=1)))),
            "zero_gaze_max_frame_difference": float(np.max(np.abs(zero - zero[:1]))),
            **support_summary(decomp),
        })
    return movies, pd.DataFrame(audit)


def validate_cache08(movies: dict[str, np.ndarray], cache08_dir: Path) -> pd.DataFrame:
    prior = np.load(cache08_dir / "multiimage_trajectory_retinal_movies.npz")
    rows = []
    for image_index in (9, 3, 8, 14, 2, 10):
        for condition, old_key in (
            ("zero", f"image_{image_index:02d}_zero"),
            ("fem", f"image_{image_index:02d}_trace_000_fem"),
        ):
            new_key = f"image_{image_index:02d}_{condition}"
            error = np.abs(movies[new_key].astype(float) - prior[old_key].astype(float))
            rows.append({
                "image_index": image_index,
                "condition": condition,
                "prior_key": old_key,
                "new_key": new_key,
                "max_abs_pixel_error": float(error.max()),
                "mean_abs_pixel_error": float(error.mean()),
                "exact_match": bool(np.array_equal(movies[new_key], prior[old_key])),
            })
    result = pd.DataFrame(rows)
    if not result["exact_match"].all():
        raise ValueError(f"All-16 reconstruction failed checkpoint-08 exact cache check:\n{result}")
    return result


def compute_drive_tables(
    movies: dict[str, np.ndarray], images: pd.DataFrame, models: pd.DataFrame, ppd: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    rows = []
    power_archive: dict[str, np.ndarray] = {}
    power_summary = []
    for image_index in images["image_index"].to_numpy(int):
        fem = movies[f"image_{image_index:02d}_fem"]
        decomp = spectral_decomposition(fem, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
        radial = radialize_power(decomp, ppd=ppd, frame_size=fem.shape[-1])
        sf_all = radial["sf_centers_cpd"].astype(float)
        tf_all = decomp["temporal_frequency_hz"].astype(float)
        annular = radial["dynamic_radial_power"].astype(float) * radial["spatial_mode_count"][:, None]
        sf_mask = (sf_all >= SF_FIT_MIN_CPD) & (sf_all <= SF_FIT_MAX_CPD)
        tf_mask = (tf_all >= TF_FIT_MIN_HZ) & (tf_all <= TF_FIT_MAX_HZ)
        sf = sf_all[sf_mask]
        tf = tf_all[tf_mask]
        power = annular[np.ix_(sf_mask, tf_mask)]
        if not np.all(np.isfinite(power)):
            raise ValueError(f"Nonfinite supported power for image {image_index}")
        power_archive[f"image_{image_index:02d}_supported_power_sf_tf"] = power.astype(np.float64)
        power_summary.append({
            "image_index": image_index,
            "total_supported_dynamic_power": float(power.sum()),
            "power_weighted_sf_centroid_cpd": float(2 ** np.average(np.log2(sf), weights=power.sum(axis=1))),
            "power_weighted_tf_centroid_hz": float(2 ** np.average(np.log2(tf), weights=power.sum(axis=0))),
        })
        for _, model in models.iterrows():
            unit = int(model["rr100_index"])
            if bool(model["model_valid"]):
                gain = surface(model, sf, tf)
                matched = power * gain**2
                matched_power = float(matched.sum())
                drive = float(np.sqrt(max(matched_power, 0.0)))
                gain_scaled = float(model["joint_rank1_gain_f0_hz"]) * drive
                matched_sf = float(2 ** np.average(np.log2(sf), weights=matched.sum(axis=1))) if matched_power > 0 else np.nan
                matched_tf = float(2 ** np.average(np.log2(tf), weights=matched.sum(axis=0))) if matched_power > 0 else np.nan
            else:
                matched_power = drive = gain_scaled = matched_sf = matched_tf = np.nan
            rows.append({
                "image_index": image_index,
                "rr100_index": unit,
                "spectral_matched_power_arbitrary": matched_power,
                "spectral_drive_amplitude_arbitrary": drive,
                "gain_scaled_spectral_drive_arbitrary": gain_scaled,
                "matched_power_fraction_of_supported_power": matched_power / max(float(power.sum()), EPS),
                "matched_power_sf_centroid_cpd": matched_sf,
                "matched_power_tf_centroid_hz": matched_tf,
                "orientation_handling": "radial_sum_across_all_spatial_orientations_no_alignment_weight",
            })
    power_archive["sf_centers_cpd"] = sf.astype(float)
    power_archive["tf_centers_hz"] = tf.astype(float)
    return pd.DataFrame(rows), power_archive, pd.DataFrame(power_summary)


def consolidate_responses(
    movies: dict[str, np.ndarray], args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    cache_path = args.out_dir / "all16_original_pair_all_rr100_response_cache.npz"
    if cache_path.exists():
        archive = np.load(cache_path)
        responses = {key: archive[key] for key in archive.files if key.startswith("image_")}
        return responses, archive["source_movie_frame_indices"], pd.DataFrame([{
            "response_source": "checkpoint_11_consolidated_cache", "n_movies": len(responses)
        }])

    prior = np.load(args.cache08_dir / "multiimage_all_rr100_response_cache.npz")
    responses: dict[str, np.ndarray] = {}
    reuse_rows = []
    for image_index in range(16):
        for condition in ("zero", "fem"):
            key = f"image_{image_index:02d}_{condition}"
            old_key = (
                f"image_{image_index:02d}_zero" if condition == "zero"
                else f"image_{image_index:02d}_trace_000_fem"
            )
            if old_key in prior.files:
                responses[key] = prior[old_key].astype(np.float32)
                reuse_rows.append({"movie_key": key, "response_source": "checkpoint_08_exact_cache"})

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(args.mapping_csv).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping["canonical_channel"].to_numpy(int)):
        raise ValueError("RR100 movie-medoid view does not match grating-fit mapping")
    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    n_lags = int(scorer.common.N_LAGS)
    for key in sorted(movies):
        if key not in responses:
            print(f"running missing all-RR100 movie: {key}", flush=True)
            responses[key] = run_condition(scorer, view, movies[key], ALL_UNITS, n_lags)
            reuse_rows.append({"movie_key": key, "response_source": "checkpoint_11_new_inference"})
    shapes = {key: value.shape for key, value in responses.items()}
    if len(set(shapes.values())) != 1 or len(responses) != 32:
        raise ValueError(f"Unexpected response cache: n={len(responses)}, shapes={shapes}")
    source_frames = np.arange(n_lags - 1, n_lags - 1 + next(iter(responses.values())).shape[0])
    np.savez_compressed(
        cache_path, rr100_indices=ALL_UNITS, source_movie_frame_indices=source_frames, **responses
    )
    return responses, source_frames, pd.DataFrame(reuse_rows)


def response_table(responses: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for image_index in range(16):
        zero = responses[f"image_{image_index:02d}_zero"][:, :, 0, 0].astype(float)
        fem = responses[f"image_{image_index:02d}_fem"][:, :, 0, 0].astype(float)
        delta = fem - zero
        for unit in ALL_UNITS:
            values = delta[:, int(unit)]
            rows.append({
                "image_index": image_index,
                "rr100_index": int(unit),
                "zero_mean_rate_hz": float(zero[:, int(unit)].mean()),
                "fem_mean_rate_hz": float(fem[:, int(unit)].mean()),
                "fem_minus_zero_mean_hz": float(values.mean()),
                "fem_delta_temporal_sd_hz": float(values.std()),
                "fem_delta_rms_hz": float(np.sqrt(np.mean(values**2))),
                "primary_outcome": "temporal SD across 97 valid frames of paired FEM-minus-zero response",
            })
    return pd.DataFrame(rows)


def fit_nonnegative_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_mean, y_mean = float(x.mean()), float(y.mean())
    denominator = float(np.sum((x - x_mean) ** 2))
    slope = max(float(np.sum((x - x_mean) * (y - y_mean)) / max(denominator, EPS)), 0.0)
    return y_mean - slope * x_mean, slope


def cross_validate(
    joined: pd.DataFrame, models: pd.DataFrame,
    predictor_col: str = "gain_scaled_spectral_drive_arbitrary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    unit_rows = []
    model_index = models.set_index("rr100_index")
    for unit, frame in joined.groupby("rr100_index", sort=True):
        frame = frame.sort_values("image_index")
        x = frame[predictor_col].to_numpy(float)
        y = frame["fem_delta_temporal_sd_hz"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        model = model_index.loc[int(unit)]
        if valid.sum() != 16:
            unit_rows.append({
                "rr100_index": int(unit), "n_images": int(valid.sum()), "cv_status": "not_evaluable",
                "model_valid": bool(model["model_valid"]), "quality_cohort": bool(quality_mask(models).loc[models["rr100_index"].eq(int(unit))].iloc[0]),
            })
            continue
        yhat = np.full(16, np.nan)
        baseline = np.full(16, np.nan)
        slopes = np.full(16, np.nan)
        for held in range(16):
            train = np.arange(16) != held
            intercept, slope = fit_nonnegative_line(x[train], y[train])
            yhat[held] = intercept + slope * x[held]
            baseline[held] = float(y[train].mean())
            slopes[held] = slope
            fold_rows.append({
                "rr100_index": int(unit), "held_out_image_index": int(frame.iloc[held]["image_index"]),
                "observed_modulation_sd_hz": float(y[held]), "held_out_predicted_modulation_sd_hz": float(yhat[held]),
                "held_out_intercept_only_prediction_hz": float(baseline[held]),
                "training_slope_nonnegative": float(slope), "training_intercept_hz": float(intercept),
                "predictor_value": float(x[held]), "predictor_column": predictor_col,
            })
        sse = float(np.sum((y - yhat) ** 2))
        baseline_sse = float(np.sum((y - baseline) ** 2))
        global_sst = float(np.sum((y - y.mean()) ** 2))
        full_intercept, full_slope = fit_nonnegative_line(x, y)
        p = pearsonr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan
        rho = spearmanr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan
        oof_p = pearsonr(yhat, y).statistic if np.std(yhat) > 0 and np.std(y) > 0 else np.nan
        unit_rows.append({
            "rr100_index": int(unit), "n_images": 16, "cv_status": "evaluated",
            "model_valid": bool(model["model_valid"]),
            "quality_cohort": bool(
                bool(model["model_valid"]) and model["sf_fit_r2"] >= 0.70 and model["tf_fit_r2"] >= 0.70
                and model["joint_parametric_surface_r2"] >= 0.50
            ),
            "preferred_sf_cpd": float(model["preferred_sf_cpd"]), "preferred_tf_hz": float(model["preferred_tf_hz"]),
            "sf_fit_r2": float(model["sf_fit_r2"]), "tf_fit_r2": float(model["tf_fit_r2"]),
            "joint_parametric_surface_r2": float(model["joint_parametric_surface_r2"]),
            "cv_r2_vs_train_mean_baseline": 1.0 - sse / max(baseline_sse, EPS),
            "oof_r2_vs_global_mean": 1.0 - sse / max(global_sst, EPS),
            "oof_rmse_hz": float(np.sqrt(sse / 16)), "response_modulation_sd_across_images_hz": float(np.std(y)),
            "oof_nrmse_by_response_sd": float(np.sqrt(sse / 16) / max(float(np.std(y)), EPS)),
            "oof_pearson_r": float(oof_p), "in_sample_pearson_r": float(p), "in_sample_spearman_rho": float(rho),
            "full_fit_intercept_hz": float(full_intercept), "full_fit_nonnegative_slope": float(full_slope),
            "n_positive_loo_slopes": int(np.sum(slopes > 0)),
        })
    return pd.DataFrame(unit_rows), pd.DataFrame(fold_rows)


def predictor_variant_audit(joined: pd.DataFrame, models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = joined.copy()
    audit["matched_power_amplitude"] = np.sqrt(
        np.maximum(audit["spectral_matched_power_arbitrary"].to_numpy(float), 0.0)
    )
    audit["normalized_matched_power_amplitude"] = np.sqrt(
        np.maximum(audit["matched_power_fraction_of_supported_power"].to_numpy(float), 0.0)
    )
    audit["total_supported_power_amplitude_control"] = np.sqrt(
        np.maximum(audit["total_supported_dynamic_power"].to_numpy(float), 0.0)
    )
    variants = (
        ("primary_unit_specific_amplitude", "gain_scaled_spectral_drive_arbitrary",
         "unit SFxTF sensitivity; power converted to amplitude"),
        ("unit_specific_power", "spectral_matched_power_arbitrary",
         "unit SFxTF sensitivity; power kept on power scale"),
        ("unit_specific_amplitude_without_f0_gain", "matched_power_amplitude",
         "same as primary without unit-constant fitted F0 gain"),
        ("unit_specific_normalized_amplitude", "normalized_matched_power_amplitude",
         "unit-matched power divided by total supported power"),
        ("total_power_amplitude_no_unit_tuning", "total_supported_power_amplitude_control",
         "negative control: total supported power; no unit SFxTF sensitivity"),
    )
    metric_frames = []
    fold_frames = []
    for variant, column, definition in variants:
        metrics, folds = cross_validate(audit, models, predictor_col=column)
        metrics.insert(0, "predictor_variant", variant)
        metrics.insert(1, "predictor_definition", definition)
        folds.insert(0, "predictor_variant", variant)
        folds.insert(1, "predictor_definition", definition)
        metric_frames.append(metrics)
        fold_frames.append(folds)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True)


def cross_validate_total_plus_composition(joined: pd.DataFrame, models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out nested test: generic power amplitude plus unit-specific spectral composition."""
    audit = joined.copy()
    audit["total_supported_power_amplitude_control"] = np.sqrt(
        np.maximum(audit["total_supported_dynamic_power"].to_numpy(float), 0.0)
    )
    audit["normalized_matched_power_amplitude"] = np.sqrt(
        np.maximum(audit["matched_power_fraction_of_supported_power"].to_numpy(float), 0.0)
    )
    model_index = models.set_index("rr100_index")
    metric_rows, fold_rows = [], []
    for unit, frame in audit.groupby("rr100_index", sort=True):
        frame = frame.sort_values("image_index")
        x = frame[["total_supported_power_amplitude_control", "normalized_matched_power_amplitude"]].to_numpy(float)
        y = frame["fem_delta_temporal_sd_hz"].to_numpy(float)
        model = model_index.loc[int(unit)]
        valid = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        if valid.sum() != 16:
            metric_rows.append({"rr100_index": int(unit), "n_images": int(valid.sum()), "cv_status": "not_evaluable"})
            continue
        yhat = np.full(16, np.nan); baseline = np.full(16, np.nan)
        for held in range(16):
            train = np.arange(16) != held
            train_mean = x[train].mean(axis=0)
            train_scale = x[train].std(axis=0)
            train_scale = np.where(train_scale > EPS, train_scale, 1.0)
            train_x = (x[train] - train_mean) / train_scale
            held_x = (x[held] - train_mean) / train_scale
            design = np.column_stack([np.ones(train.sum()), train_x])
            fit = lsq_linear(design, y[train], bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]))
            yhat[held] = float(np.r_[1.0, held_x] @ fit.x)
            baseline[held] = float(y[train].mean())
            fold_rows.append({
                "rr100_index": int(unit), "held_out_image_index": int(frame.iloc[held]["image_index"]),
                "observed_modulation_sd_hz": float(y[held]), "held_out_predicted_modulation_sd_hz": float(yhat[held]),
                "held_out_intercept_only_prediction_hz": float(baseline[held]),
                "training_intercept_hz": float(fit.x[0]), "training_total_power_slope_nonnegative": float(fit.x[1]),
                "training_spectral_composition_slope_nonnegative": float(fit.x[2]),
            })
        sse = float(np.sum((y - yhat) ** 2)); baseline_sse = float(np.sum((y - baseline) ** 2))
        metric_rows.append({
            "rr100_index": int(unit), "n_images": 16, "cv_status": "evaluated",
            "model_valid": bool(model["model_valid"]),
            "quality_cohort": bool(
                bool(model["model_valid"]) and model["sf_fit_r2"] >= 0.70 and model["tf_fit_r2"] >= 0.70
                and model["joint_parametric_surface_r2"] >= 0.50
            ),
            "preferred_sf_cpd": float(model["preferred_sf_cpd"]), "preferred_tf_hz": float(model["preferred_tf_hz"]),
            "cv_r2_vs_train_mean_baseline": 1.0 - sse / max(baseline_sse, EPS),
            "oof_rmse_hz": float(np.sqrt(sse / 16)),
            "oof_pearson_r": float(pearsonr(yhat, y).statistic) if np.std(yhat) > 0 and np.std(y) > 0 else np.nan,
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(fold_rows)


def bootstrap_population_summary(
    variant_metrics: pd.DataFrame, nested_metrics: pd.DataFrame, n_boot: int = 10000,
) -> pd.DataFrame:
    frames = [variant_metrics, nested_metrics.assign(
        predictor_variant="total_power_plus_spectral_composition",
        predictor_definition="two-predictor nested model: total power amplitude plus normalized SFxTF composition",
    )]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    rng = np.random.default_rng(20260812)
    rows = []
    for variant, frame in combined.loc[combined["quality_cohort"].eq(True)].groupby("predictor_variant"):
        values = frame["cv_r2_vs_train_mean_baseline"].to_numpy(float)
        boot_index = rng.integers(0, len(values), size=(int(n_boot), len(values)))
        samples = values[boot_index]
        boot_median = np.median(samples, axis=1)
        boot_positive = np.mean(samples > 0, axis=1)
        rows.append({
            "predictor_variant": variant,
            "predictor_definition": str(frame["predictor_definition"].iloc[0]),
            "n_quality_units": len(values),
            "median_cv_r2": float(np.median(values)),
            "median_cv_r2_unit_bootstrap_ci95_low": float(np.quantile(boot_median, 0.025)),
            "median_cv_r2_unit_bootstrap_ci95_high": float(np.quantile(boot_median, 0.975)),
            "fraction_units_positive_cv_r2": float(np.mean(values > 0)),
            "fraction_positive_unit_bootstrap_ci95_low": float(np.quantile(boot_positive, 0.025)),
            "fraction_positive_unit_bootstrap_ci95_high": float(np.quantile(boot_positive, 0.975)),
            "bootstrap_unit": "RR100 unit",
            "n_bootstrap": int(n_boot),
        })
    return pd.DataFrame(rows)


def select_examples(metrics: pd.DataFrame) -> pd.DataFrame:
    cohort = metrics.loc[metrics["quality_cohort"].eq(True)].copy()
    response_floor = float(cohort["response_modulation_sd_across_images_hz"].median())
    strong = cohort.loc[cohort["response_modulation_sd_across_images_hz"].ge(response_floor)].copy()
    high = strong.loc[strong["cv_r2_vs_train_mean_baseline"].idxmax()]
    ordered = strong.sort_values("cv_r2_vs_train_mean_baseline")
    positive = ordered.loc[ordered["cv_r2_vs_train_mean_baseline"].gt(0)]
    target = float(positive["cv_r2_vs_train_mean_baseline"].median()) if len(positive) else 0.0
    mid = strong.loc[(strong["cv_r2_vs_train_mean_baseline"] - target).abs().idxmin()]
    low = strong.loc[strong["cv_r2_vs_train_mean_baseline"].idxmin()]
    specs = [
        ("high_explainability", high, "highest held-out R2 among quality-fit units with above-median response variation"),
        ("intermediate_explainability", mid, "closest to median positive held-out R2 in the same auditable cohort"),
        ("low_explainability_strong_modulation", low, "lowest held-out R2 despite above-median response variation"),
    ]
    rows = []
    for order, (role, row, criterion) in enumerate(specs, start=1):
        rows.append({"display_order": order, "example_role": role, "selection_criterion": criterion, **row.to_dict()})
    return pd.DataFrame(rows)


def plot_mechanism(
    out: Path, movies: dict[str, np.ndarray], images: pd.DataFrame, power_archive: dict[str, np.ndarray],
    models: pd.DataFrame, examples: pd.DataFrame, drive: pd.DataFrame, dpi: int,
) -> int:
    power_summary = pd.DataFrame({
        "image_index": range(16),
        "power": [power_archive[f"image_{i:02d}_supported_power_sf_tf"].sum() for i in range(16)],
    })
    target = float(power_summary["power"].median())
    image_index = int(power_summary.loc[(power_summary["power"] - target).abs().idxmin(), "image_index"])
    sf, tf = power_archive["sf_centers_cpd"], power_archive["tf_centers_hz"]
    power = power_archive[f"image_{image_index:02d}_supported_power_sf_tf"]
    zero = movies[f"image_{image_index:02d}_zero"]
    fem = movies[f"image_{image_index:02d}_fem"]
    fig = plt.figure(figsize=(16.2, 13.2), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 1.0])
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    frame_indices = [0, 32, 64, 96]
    axes[0].imshow(np.concatenate([fem[i] for i in frame_indices], axis=1), cmap="gray")
    axes[0].set_title(f"A  Original retinal movie · image {image_index}\nmeasured eye trace moves a static image")
    axes[0].axis("off")
    residual = fem - zero
    vmax = float(np.percentile(np.abs(residual), 99))
    axes[1].imshow(np.concatenate([residual[i] for i in frame_indices], axis=1), cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("B  FEM-created contrast change\nframewise FEM minus true zero gaze")
    axes[1].axis("off")
    im = axes[2].pcolormesh(sf, tf, power.T, shading="auto", cmap="turbo", norm=LogNorm(max(power[power > 0].min(), power.max()*1e-7), power.max()))
    axes[2].set(xscale="log", yscale="log", xlabel="spatial frequency (cycles/deg)", ylabel="temporal frequency (Hz)")
    axes[2].set_title("C  Dynamic power created by FEM\norientation collapsed; exact movie")
    fig.colorbar(im, ax=axes[2], label="power (a.u.)")
    axes[3].axis("off")
    axes[3].text(0.02, 0.95, "Prediction for each unit", va="top", fontsize=13, weight="bold")
    axes[3].text(0.02, 0.78, "movie power × fixed-retina\nSF–TF sensitivity²", va="top", fontsize=12)
    axes[3].text(0.02, 0.49, "sum over SF and TF\nthen square root → drive amplitude", va="top", fontsize=12)
    axes[3].text(0.02, 0.18, "No orientation-alignment term\nNo trajectory rotation", va="top", fontsize=11, color="#555555")

    model_index = models.set_index("rr100_index")
    for row_index, (_, example) in enumerate(examples.sort_values("display_order").iterrows(), start=1):
        unit = int(example["rr100_index"])
        model = model_index.loc[unit]
        gain = surface(model, sf, tf)
        overlap = power * gain**2
        row_axes = [fig.add_subplot(grid[row_index, col]) for col in range(4)]
        im0 = row_axes[0].pcolormesh(sf, tf, gain.T, shading="auto", cmap="magma", vmin=0, vmax=1)
        row_axes[0].set_title(f"{chr(67 + row_index)}  RR100 {unit}: {str(example['example_role']).replace('_', ' ')}\nfixed-retina SF–TF sensitivity")
        fig.colorbar(im0, ax=row_axes[0], label="relative F0 sensitivity")
        im1 = row_axes[1].pcolormesh(sf, tf, power.T, shading="auto", cmap="turbo", norm=LogNorm(max(power[power > 0].min(), power.max()*1e-7), power.max()))
        row_axes[1].set_title("Same FEM-created power\nfor every unit in this figure")
        fig.colorbar(im1, ax=row_axes[1], label="power (a.u.)")
        positive = overlap[overlap > 0]
        im2 = row_axes[2].pcolormesh(sf, tf, overlap.T, shading="auto", cmap="turbo", norm=LogNorm(max(positive.min(), overlap.max()*1e-7), overlap.max()))
        row_axes[2].set_title("Power falling inside this unit’s passband\npower × sensitivity²")
        fig.colorbar(im2, ax=row_axes[2], label="matched power (a.u.)")
        row_axes[3].axis("off")
        d = drive.loc[(drive["image_index"].eq(image_index)) & (drive["rr100_index"].eq(unit))].iloc[0]
        row_axes[3].text(0.04, 0.87, f"Preferred SF: {model['preferred_sf_cpd']:.2f} c/deg", fontsize=11)
        row_axes[3].text(0.04, 0.70, f"Preferred TF: {model['preferred_tf_hz']:.1f} Hz", fontsize=11)
        row_axes[3].text(0.04, 0.49, "Matched power", fontsize=10, color="#555555")
        row_axes[3].text(0.04, 0.35, f"{d['spectral_matched_power_arbitrary']:.2e} a.u.", fontsize=13, weight="bold")
        row_axes[3].text(0.04, 0.12, f"Held-out R²: {example['cv_r2_vs_train_mean_baseline']:+.2f}", fontsize=13, weight="bold")
        for axis in row_axes[:3]:
            axis.set(xscale="log", yscale="log", xlabel="SF (cycles/deg)", ylabel="TF (Hz)")
    fig.suptitle(
        "Checkpoint 11 mechanism: the same FEM-created movie power is filtered by different unit SF–TF passbands",
        fontsize=15, weight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return image_index


def plot_validation(out: Path, metrics: pd.DataFrame, folds: pd.DataFrame, examples: pd.DataFrame, dpi: int) -> None:
    fig = plt.figure(figsize=(16.0, 9.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    colors = ["#0072B2", "#E69F00", "#CC79A7"]
    for col, (_, example) in enumerate(examples.sort_values("display_order").iterrows()):
        unit = int(example["rr100_index"])
        frame = folds.loc[folds["rr100_index"].eq(unit)].sort_values("held_out_image_index")
        axis = fig.add_subplot(grid[0, col])
        axis.scatter(frame["predictor_value"], frame["observed_modulation_sd_hz"], s=46, color=colors[col], edgecolor="white", lw=0.6)
        axis.scatter(frame["predictor_value"], frame["held_out_predicted_modulation_sd_hz"], s=42, facecolor="white", edgecolor="black", lw=1.0, marker="o", label="held-out prediction")
        for _, point in frame.iterrows():
            axis.plot(
                [point["predictor_value"], point["predictor_value"]],
                [point["observed_modulation_sd_hz"], point["held_out_predicted_modulation_sd_hz"]],
                color="0.65", lw=0.7, zorder=0,
            )
        for _, point in frame.iterrows():
            axis.annotate(str(int(point["held_out_image_index"])), (point["predictor_value"], point["observed_modulation_sd_hz"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set(xlabel="spectral drive from exact movie (a.u.)", ylabel="measured FEM modulation SD (Hz)")
        axis.set_title(
            f"{chr(71+col)}  {str(example['example_role']).replace('_', ' ')} · RR100 {unit}\n"
            f"each number is one held-out image · CV R²={example['cv_r2_vs_train_mean_baseline']:+.2f}", fontsize=11
        )
        axis.grid(color="0.92")
        if col == 0:
            axis.legend(frameon=False, fontsize=8)
    cohort = metrics.loc[metrics["quality_cohort"].eq(True)].copy()
    cohort_units = set(cohort["rr100_index"].astype(int))
    pop = folds.loc[folds["rr100_index"].isin(cohort_units)].copy()
    stats = metrics.set_index("rr100_index")
    pop["unit_response_mean"] = pop["rr100_index"].map(folds.groupby("rr100_index")["observed_modulation_sd_hz"].mean())
    pop["unit_response_sd"] = pop["rr100_index"].map(folds.groupby("rr100_index")["observed_modulation_sd_hz"].std(ddof=0)).clip(lower=EPS)
    pop["observed_z_display"] = (pop["observed_modulation_sd_hz"] - pop["unit_response_mean"]) / pop["unit_response_sd"]
    pop["predicted_z_display"] = (pop["held_out_predicted_modulation_sd_hz"] - pop["unit_response_mean"]) / pop["unit_response_sd"]
    axis = fig.add_subplot(grid[1, :2])
    hb = axis.hexbin(pop["predicted_z_display"], pop["observed_z_display"], gridsize=34, cmap="turbo", mincnt=1)
    limits = (-3.5, 3.5)
    axis.plot(limits, limits, color="black", ls="--", lw=1)
    axis.set(xlim=limits, ylim=limits, xlabel="held-out predicted modulation (within-unit SD units)", ylabel="measured modulation (within-unit SD units)")
    axis.set_title(f"J  Population held-out predictions · {len(cohort)} quality-fit RR100 units × 16 images\nDisplay normalization only; each prediction was fit without that image")
    fig.colorbar(hb, ax=axis, label="unit–image count")
    axis = fig.add_subplot(grid[1, 2])
    values = cohort["cv_r2_vs_train_mean_baseline"].to_numpy(float)
    bins = np.linspace(min(-1.0, np.nanpercentile(values, 2)), max(1.0, np.nanpercentile(values, 98)), 25)
    axis.hist(values, bins=bins, color="#56B4E9", edgecolor="white")
    axis.axvline(0, color="black", lw=1, ls="--")
    axis.axvline(float(np.median(values)), color="#D55E00", lw=2, label=f"median={np.median(values):+.2f}")
    for color, (_, example) in zip(colors, examples.sort_values("display_order").iterrows(), strict=True):
        axis.axvline(float(example["cv_r2_vs_train_mean_baseline"]), color=color, lw=2, alpha=0.9)
    axis.set(xlabel="leave-one-image-out R² vs training-mean baseline", ylabel="RR100 unit count")
    axis.set_title(f"K  How much image-to-image modulation is predicted?\n{np.mean(values > 0)*100:.0f}% of quality units beat intercept-only prediction")
    axis.legend(frameon=False)
    fig.suptitle(
        "Checkpoint 11 validation: can SF–TF power redistribution predict frozen-unit FEM modulation on a new image?",
        fontsize=15, weight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_predictor_controls(out: Path, variant_metrics: pd.DataFrame, nested_metrics: pd.DataFrame, dpi: int) -> None:
    quality = variant_metrics.loc[variant_metrics["quality_cohort"].eq(True)].copy()
    order = [
        "primary_unit_specific_amplitude", "unit_specific_power",
        "unit_specific_normalized_amplitude", "total_power_amplitude_no_unit_tuning",
    ]
    labels = ["unit SF–TF\namplitude", "unit SF–TF\npower", "unit SF–TF\nnormalized", "total power\nno tuning", "total power +\nSF–TF composition"]
    colors = ["#0072B2", "#56B4E9", "#009E73", "#999999", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    nested_quality = nested_metrics.loc[nested_metrics["quality_cohort"].eq(True)]
    display_floor = -3.0
    data = [np.maximum(quality.loc[quality["predictor_variant"].eq(value), "cv_r2_vs_train_mean_baseline"].to_numpy(float), display_floor) for value in order]
    data.append(np.maximum(nested_quality["cv_r2_vs_train_mean_baseline"].to_numpy(float), display_floor))
    parts = axes[0].violinplot(data, positions=np.arange(len(data)), showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color); body.set_edgecolor("black"); body.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set(ylim=(display_floor - 0.15, 1.0), ylabel="leave-one-image-out R² (display clipped at −3)", title="A  Prediction depends on response and predictor scale")
    primary = quality.loc[quality["predictor_variant"].eq(order[0])].set_index("rr100_index")
    control = quality.loc[quality["predictor_variant"].eq(order[-1])].set_index("rr100_index").reindex(primary.index)
    nested = nested_quality.set_index("rr100_index").reindex(primary.index)
    axes[1].scatter(control["cv_r2_vs_train_mean_baseline"], primary["cv_r2_vs_train_mean_baseline"], s=28, color="#0072B2", alpha=0.65, label="unit-specific overlap alone")
    axes[1].scatter(control["cv_r2_vs_train_mean_baseline"], nested["cv_r2_vs_train_mean_baseline"], s=28, color="#D55E00", alpha=0.7, label="total power + composition")
    lo = float(min(control["cv_r2_vs_train_mean_baseline"].min(), primary["cv_r2_vs_train_mean_baseline"].min(), nested["cv_r2_vs_train_mean_baseline"].min(), -1))
    axes[1].plot([lo, 1], [lo, 1], color="black", ls="--", lw=1)
    better = float((nested["cv_r2_vs_train_mean_baseline"] > control["cv_r2_vs_train_mean_baseline"]).mean())
    axes[1].set(xlim=(lo, 1), ylim=(lo, 1), xlabel="R² from total dynamic power only", ylabel="R² from unit-specific SF–TF drive",
                title=f"B  Does SF–TF composition add beyond total power?\nnested model is better for {better*100:.0f}% of quality units")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes: axis.grid(color="0.92")
    fig.suptitle("Checkpoint 11 predictor controls: separating generic image power from frequency-specific filtering", fontsize=14, weight="bold")
    fig.savefig(out.with_suffix(".png"), dpi=dpi); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = eligible_images(pd.read_csv(args.run_dir / "image_feature_table.csv"))
    images.to_csv(args.out_dir / "all16_original_pair_image_contract.csv", index=False)
    models = pd.read_csv(args.models_csv).sort_values("rr100_index").reset_index(drop=True)
    source_rows = load_source_rows(args.source_csv)
    ppd = float(_load_twin_common().PPD)
    movie_path = args.out_dir / "all16_original_pair_retinal_movies.npz"
    if movie_path.exists():
        archive = np.load(movie_path)
        movies = {key: archive[key] for key in archive.files}
        audit = pd.read_csv(args.out_dir / "all16_original_pair_stimulus_audit.csv")
    else:
        movies, audit = build_movies(images, source_rows, ppd)
        np.savez_compressed(movie_path, **movies)
        audit.to_csv(args.out_dir / "all16_original_pair_stimulus_audit.csv", index=False)
    cache_check = validate_cache08(movies, args.cache08_dir)
    cache_check.to_csv(args.out_dir / "checkpoint08_exact_reconstruction_audit.csv", index=False)
    drive, power_archive, power_summary = compute_drive_tables(movies, images, models, ppd)
    drive.to_csv(args.out_dir / "all16_original_pair_spectral_drive_all_rr100.csv", index=False)
    power_summary.to_csv(args.out_dir / "all16_original_pair_power_summary.csv", index=False)
    np.savez_compressed(args.out_dir / "all16_original_pair_supported_sf_tf_power.npz", **power_archive)
    if args.prepare_only:
        print(f"Prepared exact movies and spectral predictions in {args.out_dir}")
        return
    responses, source_frames, provenance = consolidate_responses(movies, args)
    provenance.to_csv(args.out_dir / "all16_response_cache_provenance.csv", index=False)
    if len(source_frames) != 97:
        raise ValueError(f"Expected 97 valid response frames, got {len(source_frames)}")
    responses_long = response_table(responses)
    responses_long.to_csv(args.out_dir / "all16_original_pair_response_metrics_all_rr100.csv", index=False)
    joined = drive.merge(power_summary, on="image_index", validate="many_to_one").merge(
        responses_long, on=["image_index", "rr100_index"], validate="one_to_one"
    )
    joined.to_csv(args.out_dir / "all16_spectral_drive_and_response_all_rr100.csv", index=False)
    metrics, folds = cross_validate(joined, models)
    metrics.to_csv(args.out_dir / "per_unit_leave_one_image_out_explainability.csv", index=False)
    folds.to_csv(args.out_dir / "leave_one_image_out_predictions.csv", index=False)
    variant_metrics, variant_folds = predictor_variant_audit(joined, models)
    variant_metrics.to_csv(args.out_dir / "predictor_variant_per_unit_explainability.csv", index=False)
    variant_folds.to_csv(args.out_dir / "predictor_variant_leave_one_image_out_predictions.csv", index=False)
    nested_metrics, nested_folds = cross_validate_total_plus_composition(joined, models)
    nested_metrics.to_csv(args.out_dir / "total_power_plus_spectral_composition_per_unit_explainability.csv", index=False)
    nested_folds.to_csv(args.out_dir / "total_power_plus_spectral_composition_leave_one_image_out_predictions.csv", index=False)
    bootstrap = bootstrap_population_summary(variant_metrics, nested_metrics)
    bootstrap.to_csv(args.out_dir / "population_explainability_bootstrap_summary.csv", index=False)
    examples = select_examples(metrics)
    examples.to_csv(args.out_dir / "auditable_explainability_example_selection.csv", index=False)
    example_image = plot_mechanism(
        args.out_dir / "checkpoint_11_mechanism_and_three_explainability_cases",
        movies, images, power_archive, models, examples, drive, args.dpi,
    )
    plot_validation(
        args.out_dir / "checkpoint_11_held_out_image_prediction_and_population_explainability",
        metrics, folds, examples, args.dpi,
    )
    plot_predictor_controls(
        args.out_dir / "checkpoint_11_predictor_scale_and_total_power_controls", variant_metrics, nested_metrics, args.dpi
    )
    cohort = metrics.loc[metrics["quality_cohort"].eq(True)]
    total_control = variant_metrics.loc[
        variant_metrics["quality_cohort"].eq(True)
        & variant_metrics["predictor_variant"].eq("total_power_amplitude_no_unit_tuning")
    ]
    nested_quality = nested_metrics.loc[nested_metrics["quality_cohort"].eq(True)]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "all-16 original image/eye-pair orientation-collapsed SFxTF spectral-drive explainability",
        "status": "checkpoint_11_complete",
        "primary_predictor": "joint F0 gain multiplied by square root of sum(P_image_FEM(SF,TF) * normalized_unit_sensitivity(SF,TF)^2)",
        "primary_outcome": "temporal SD across valid frames of frozen RR100 FEM-minus-zero response",
        "cross_validation": "leave one complete original image/eye-trajectory pair out; fit intercept plus nonnegative slope on other 15",
        "primary_r2_baseline": "fold-specific mean of the 15 training images",
        "orientation_contract": "radial power summed across orientation; no orientation-alignment weight; no trajectory rotation",
        "n_images": 16, "n_rr100": 100, "n_valid_parametric_models": int(models["model_valid"].sum()),
        "n_quality_models": int(len(cohort)), "representative_mechanism_image_index": int(example_image),
        "quality_cohort_median_cv_r2": float(cohort["cv_r2_vs_train_mean_baseline"].median()),
        "quality_cohort_fraction_positive_cv_r2": float(cohort["cv_r2_vs_train_mean_baseline"].gt(0).mean()),
        "quality_cohort_fraction_cv_r2_ge_0p25": float(cohort["cv_r2_vs_train_mean_baseline"].ge(0.25).mean()),
        "quality_cohort_total_power_control_median_cv_r2": float(total_control["cv_r2_vs_train_mean_baseline"].median()),
        "quality_cohort_total_plus_composition_median_cv_r2": float(nested_quality["cv_r2_vs_train_mean_baseline"].median()),
        "quality_cohort_fraction_unit_specific_better_than_total_power": float(
            (
                variant_metrics.loc[
                    variant_metrics["quality_cohort"].eq(True)
                    & variant_metrics["predictor_variant"].eq("primary_unit_specific_amplitude")
                ].set_index("rr100_index")["cv_r2_vs_train_mean_baseline"]
                > variant_metrics.loc[
                    variant_metrics["quality_cohort"].eq(True)
                    & variant_metrics["predictor_variant"].eq("total_power_amplitude_no_unit_tuning")
                ].set_index("rr100_index")["cv_r2_vs_train_mean_baseline"]
            ).mean()
        ),
        "quality_cohort_fraction_total_plus_composition_better_than_total_power": float(
            (
                nested_metrics.loc[nested_metrics["quality_cohort"].eq(True)].sort_values("rr100_index")["cv_r2_vs_train_mean_baseline"].to_numpy(float)
                > variant_metrics.loc[
                    variant_metrics["quality_cohort"].eq(True)
                    & variant_metrics["predictor_variant"].eq("total_power_amplitude_no_unit_tuning")
                ].sort_values("rr100_index")["cv_r2_vs_train_mean_baseline"].to_numpy(float)
            ).mean()
        ),
        "checks": {
            "maximum_checkpoint08_reconstruction_pixel_error": float(cache_check["max_abs_pixel_error"].max()),
            "maximum_zero_gaze_frame_difference": float(audit["zero_gaze_max_frame_difference"].max()),
            "response_frames_per_movie": int(len(source_frames)),
        },
        "inputs": {
            "image_features": file_identity(args.run_dir / "image_feature_table.csv"),
            "source_windows": file_identity(args.source_csv),
            "parametric_models": file_identity(args.models_csv),
            "mapping": file_identity(args.mapping_csv),
            "checkpoint08_response_cache": file_identity(args.cache08_dir / "multiimage_all_rr100_response_cache.npz"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 11: all-16 held-out-image spectral explainability\n\n"
        "This checkpoint uses all 16 original natural-image/measured-eye-trajectory pairs. The primary predictor is "
        "orientation-collapsed SF×TF power filtered by each unit's fixed-retina parametric F0 sensitivity. The primary "
        "outcome is temporal SD of the frozen RR100 FEM-minus-zero response. Every reported CV R² predicts a complete "
        "held-out image/trajectory pair from a nonnegative linear calibration fit to the other 15. Negative R² means the "
        "spectral-drive model predicts that unit worse than the training-image mean.\n\n"
        f"Among the 66 quality-fit units, unit-specific SF×TF drive has median held-out R²={summary['quality_cohort_median_cv_r2']:.2f} "
        f"and {summary['quality_cohort_fraction_positive_cv_r2']*100:.0f}% of units beat the training-mean baseline. However, "
        f"total supported FEM power without unit tuning has a slightly higher median R²={summary['quality_cohort_total_power_control_median_cv_r2']:.2f}. "
        f"Unit-specific overlap beats that control for only {summary['quality_cohort_fraction_unit_specific_better_than_total_power']*100:.0f}% of units, "
        f"and adding normalized SF×TF composition to total power beats total power alone for only "
        f"{summary['quality_cohort_fraction_total_plus_composition_better_than_total_power']*100:.0f}%. Thus this checkpoint supports "
        "a broad image-dependent FEM-power/gain effect, but it does not establish a population-level increment from the present scalar "
        "unit-specific SF×TF overlap model. Orientation alignment is not used anywhere in this primary analysis.\n"
    )
    print(json.dumps({key: summary[key] for key in (
        "n_quality_models", "quality_cohort_median_cv_r2", "quality_cohort_fraction_positive_cv_r2",
        "quality_cohort_fraction_cv_r2_ge_0p25")}, indent=2))
    print(examples[["example_role", "rr100_index", "cv_r2_vs_train_mean_baseline", "response_modulation_sd_across_images_hz"]].to_string(index=False))


if __name__ == "__main__":
    main()
