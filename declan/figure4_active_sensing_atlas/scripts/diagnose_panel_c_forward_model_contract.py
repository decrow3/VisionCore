"""Diagnose whether the Panel C compact forward model uses physical pose correctly.

This is a contract diagnostic for ``build_panel_c_linear_synthetic_prior_feature_observer``.
It reuses the same source-disjoint fitted ``F(z, tau)`` and asks:

* Does ``F(z_true, tau_true)`` predict observed/known compact responses?
* Does the known-response identity control reproduce observed-response decoding?
* Do simple tau convention variants beat recorded tau?
* Are estimated/hidden tau modes winning by improving fixed-tau design conditioning?
* How sensitive are fixed-tau solves to the latent prior precision?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_linear_synthetic_prior_feature_observer import (
    OUT_DIR as OBSERVER_OUT_DIR,
    REPO_ROOT,
    SampleBank,
    TestSet,
    _assign_source_folds,
    _bootstrap_mean,
    _build_feature_conditioned_fold_state,
    _build_sample_banks,
    _canonical_feature_modes,
    _continuous_args_for_scale,
    _feature_space_config,
    _fit_feature_transform,
    _fit_forward_posterior,
    _fit_transform_for_fold,
    _forward_model_z_design_for_tau,
    _json_ready,
    _load_basis,
    _load_feature_table,
    _load_feature_weights,
    _load_npz,
    _load_table_with_sidecar,
    _metrics,
    _parse_csv_values,
    _parse_scales,
    _parse_str_list,
    _predict_compact_from_z_tau,
    _predict_z,
    _read_manifest,
    _solve_z_given_tau,
    _transform_feature_sources,
    build_parser as observer_build_parser,
)


OUT_DIR = REPO_ROOT / "outputs" / "panel_c_forward_model_contract_diagnostic"


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _copy_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def _mean_squared(lhs: np.ndarray, rhs: np.ndarray) -> float:
    diff = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return float(np.mean(diff * diff)) if np.isfinite(diff).all() else float("nan")


def _condition_metrics(z_design: np.ndarray) -> dict[str, float]:
    svals = np.linalg.svd(np.asarray(z_design, dtype=np.float64), compute_uv=False)
    finite = svals[np.isfinite(svals)]
    if finite.size == 0:
        return {
            "z_design_singular_max": float("nan"),
            "z_design_singular_min": float("nan"),
            "z_design_condition": float("nan"),
            "z_design_rank": 0,
            "z_design_fro_norm": float("nan"),
        }
    smax = float(np.max(finite))
    positive = finite[finite > 1e-12]
    smin = float(np.min(positive)) if positive.size else float("nan")
    return {
        "z_design_singular_max": smax,
        "z_design_singular_min": smin,
        "z_design_condition": float(smax / smin) if np.isfinite(smin) and smin > 0.0 else float("inf"),
        "z_design_rank": int(np.sum(finite > 1e-10)),
        "z_design_fro_norm": float(np.linalg.norm(z_design)),
    }


def _response_shape(tests: TestSet) -> tuple[int, int]:
    if tests.observed_compact.ndim != 3:
        raise ValueError(f"observed_compact must be (row, time, basis), got {tests.observed_compact.shape}")
    return int(tests.observed_compact.shape[1]), int(tests.observed_compact.shape[2])


def _known_compact_by_row(banks: dict[str, SampleBank], tests: TestSet) -> np.ndarray:
    n_time, basis_dim = _response_shape(tests)
    known_x = np.asarray(banks["known_eye_response_only"].x, dtype=np.float64)
    if known_x.shape[0] != tests.rows.shape[0] or known_x.shape[1] != n_time * basis_dim:
        raise ValueError(
            "known_eye_response_only bank does not align with test rows: "
            f"bank={known_x.shape}, tests={tests.rows.shape[0]}, response={(n_time, basis_dim)}"
        )
    return known_x.reshape(known_x.shape[0], n_time, basis_dim)


def _tau_variants(
    *,
    tau_true: np.ndarray,
    tau_estimated: np.ndarray,
    tests: TestSet,
    row_index: int,
    scales: list[float],
) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {}
    n_time = int(tests.observed_compact.shape[1])
    zero = np.zeros((n_time, 2), dtype=np.float64)
    variants["zero"] = zero
    true = np.asarray(tau_true, dtype=np.float64)
    if true.shape == (n_time, 2) and np.isfinite(true).all():
        variants["true"] = true
        variants["neg_true"] = -true
        variants["mean_centered_true"] = true - np.mean(true, axis=0, keepdims=True)
        variants["start_centered_true"] = true - true[:1]
        for scale in scales:
            if np.isclose(scale, 0.0):
                continue
            variants[f"scaled_true_{scale:g}"] = float(scale) * true
    estimated = np.asarray(tau_estimated, dtype=np.float64)
    if estimated.shape == (n_time, 2) and np.isfinite(estimated).all():
        variants["estimated"] = estimated

    meta = tests.rows.iloc[int(row_index)]
    nearest_index = int(meta.get("nearest_trajectory_index", -1))
    true_candidate_index = int(meta.get("true_candidate_index", -1))
    table = tests.geometry_tables[int(row_index)]
    xy = np.asarray(table.trajectory_xy, dtype=np.float64)
    if (
        xy.ndim == 4
        and 0 <= true_candidate_index < xy.shape[0]
        and 0 <= nearest_index < xy.shape[1]
    ):
        nearest = xy[true_candidate_index, nearest_index]
        if nearest.shape == (n_time, 2) and np.isfinite(nearest).all():
            variants["nearest_catalog"] = nearest
    return variants


def _fit_decoder_rows(
    *,
    decoder_name: str,
    bank: SampleBank,
    x_all: np.ndarray,
    tests: TestSet,
    feature_table: Any,
    transform: Any,
    fold_by_source: dict[int, int],
    fold: int,
    ridge: float,
    noise_floor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_mask = np.asarray(
        [fold_by_source.get(int(source), -1) != int(fold) for source in bank.source_rows],
        dtype=bool,
    )
    train_mask &= np.isfinite(bank.x).all(axis=1)
    test_folds = tests.rows["true_source_row"].map(fold_by_source).to_numpy(dtype=int)
    test_mask = (test_folds == int(fold)) & np.isfinite(x_all).all(axis=1)
    if int(np.sum(train_mask)) <= transform.feature_dim or int(np.sum(test_mask)) == 0:
        return [], {
            "decoder_name": decoder_name,
            "fold": int(fold),
            "skipped": True,
            "skip_reason": "too_few_source_disjoint_rows",
            "n_train": int(np.sum(train_mask)),
            "n_test": int(np.sum(test_mask)),
        }
    bank_z = _transform_feature_sources(transform, feature_table, bank.source_rows)
    model = _fit_forward_posterior(
        z_train=bank_z[train_mask],
        x_train=np.asarray(bank.x, dtype=np.float64)[train_mask],
        ridge=float(ridge),
        noise_floor=float(noise_floor),
    )
    z_true = _transform_feature_sources(
        transform,
        feature_table,
        tests.rows.loc[test_mask, "true_source_row"].to_numpy(dtype=int),
    )
    z_hat = _predict_z(model, np.asarray(x_all, dtype=np.float64)[test_mask])
    rows: list[dict[str, Any]] = []
    test_indices = np.flatnonzero(test_mask)
    for local_index, global_index in enumerate(test_indices.tolist()):
        row = dict(tests.rows.iloc[int(global_index)].to_dict())
        row.update(
            {
                "diagnostic": decoder_name,
                "fold": int(fold),
                "latent": transform.latent,
                "feature_space_mode": transform.feature_space_mode,
                "feature_fit_scope": transform.fit_scope,
                "n_train_samples": int(model.n_train),
            }
        )
        row.update(_metrics(z_hat[local_index], z_true[local_index]))
        rows.append(row)
    model_row = {
        "decoder_name": decoder_name,
        "fold": int(fold),
        "latent": transform.latent,
        "feature_space_mode": transform.feature_space_mode,
        "n_train_samples": int(model.n_train),
        "n_test_rows": int(np.sum(test_mask)),
        "response_dim": int(bank.x.shape[1]),
        "feature_dim": int(transform.feature_dim),
        "ridge": float(model.ridge),
        "noise_variance": float(model.noise_variance),
    }
    return rows, model_row


def _diagnose_fold(
    *,
    banks: dict[str, SampleBank],
    tests: TestSet,
    known_compact: np.ndarray,
    feature_table: Any,
    transform: Any,
    fold_by_source: dict[int, int],
    fold: int,
    args: argparse.Namespace,
    tau_variant_scales: list[float],
    z_prior_precisions: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    state = _build_feature_conditioned_fold_state(
        banks=banks,
        tests=tests,
        feature_table=feature_table,
        transform=transform,
        fold_by_source=fold_by_source,
        fold=int(fold),
        ridge=float(args.linear_ridge),
        noise_floor=float(args.noise_floor),
        continuous_args=args,
        compute_z0=True,
        precompute_feature_conditioned_tau=True,
    )
    test_folds = tests.rows["true_source_row"].map(fold_by_source).to_numpy(dtype=int)
    base_test_mask = test_folds == int(fold)
    z_true_all = _transform_feature_sources(
        transform,
        feature_table,
        tests.rows["true_source_row"].to_numpy(dtype=int),
    )
    variant_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    fit_rows = list(state["fit_rows"])

    for decoder_name, bank, x_all in [
        ("observed_response_direct_decoder", banks["observed_response_only"], tests.x_by_mode["observed_response_only"]),
        ("known_response_identity_control_decoder", banks["known_eye_response_only"], banks["known_eye_response_only"].x),
    ]:
        rows, model_row = _fit_decoder_rows(
            decoder_name=decoder_name,
            bank=bank,
            x_all=np.asarray(x_all, dtype=np.float64),
            tests=tests,
            feature_table=feature_table,
            transform=transform,
            fold_by_source=fold_by_source,
            fold=int(fold),
            ridge=float(args.linear_ridge),
            noise_floor=float(args.noise_floor),
        )
        decoder_rows.extend(rows)
        model_rows.append(model_row)

    for global_index in np.flatnonzero(base_test_mask).tolist():
        compact_observed = np.asarray(tests.observed_compact[int(global_index)], dtype=np.float64)
        compact_known = np.asarray(known_compact[int(global_index)], dtype=np.float64)
        if not np.isfinite(compact_observed).all() or not np.isfinite(compact_known).all():
            continue
        row_meta = tests.rows.iloc[int(global_index)]
        table_args = _continuous_args_for_scale(args, float(row_meta["observation_scale"]))
        common = {
            "geometry_table": tests.geometry_tables[int(global_index)],
            "baseline_coef": np.asarray(state["baseline_coef"], dtype=np.float64),
            "baseline_residual_var": float(state["feature_conditioned_baseline_residual_variance"]),
            "observation_coef": np.asarray(state["observation_coef"], dtype=np.float64),
            "observation_residual_var": float(state["feature_conditioned_observation_residual_variance"]),
            "include_intercept": bool(state["include_intercept"]),
        }
        variants = _tau_variants(
            tau_true=np.asarray(tests.tau_true[int(global_index)], dtype=np.float64),
            tau_estimated=np.asarray(state["tau_hat"][int(global_index)], dtype=np.float64),
            tests=tests,
            row_index=int(global_index),
            scales=tau_variant_scales,
        )
        z_true = np.asarray(z_true_all[int(global_index)], dtype=np.float64)
        for variant_name, tau in sorted(variants.items()):
            try:
                pred_true_z = _predict_compact_from_z_tau(
                    z=z_true,
                    tau=tau,
                    observation_scale=float(common["geometry_table"].observation_scale),
                    baseline_coef=common["baseline_coef"],
                    observation_coef=common["observation_coef"],
                    include_intercept=bool(common["include_intercept"]),
                )
                _offset, z_design, _shape = _forward_model_z_design_for_tau(
                    tau=tau,
                    observation_scale=float(common["geometry_table"].observation_scale),
                    baseline_coef=common["baseline_coef"],
                    observation_coef=common["observation_coef"],
                    include_intercept=bool(common["include_intercept"]),
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            condition = _condition_metrics(z_design)
            true_z_mse_to_observed = _mean_squared(pred_true_z, compact_observed)
            true_z_mse_to_known = _mean_squared(pred_true_z, compact_known)
            for target_name, target_compact in [
                ("observed", compact_observed),
                ("known", compact_known),
            ]:
                for precision in z_prior_precisions:
                    solve_args = _copy_args(table_args, forward_model_z_prior_precision=float(precision))
                    try:
                        z_hat, _pred_hat, solve_meta = _solve_z_given_tau(
                            observed_compact=target_compact,
                            tau=tau,
                            continuous_args=solve_args,
                            **common,
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    row = dict(row_meta.to_dict())
                    row.update(
                        {
                            "fold": int(fold),
                            "latent": transform.latent,
                            "feature_space_mode": transform.feature_space_mode,
                            "feature_fit_scope": transform.fit_scope,
                            "tau_variant": variant_name,
                            "target_response": target_name,
                            "z_prior_precision": float(precision),
                            "true_z_forward_mse_to_observed": true_z_mse_to_observed,
                            "true_z_forward_mse_to_known": true_z_mse_to_known,
                            "known_vs_observed_compact_mse": _mean_squared(compact_known, compact_observed),
                            "tau_norm": float(np.linalg.norm(tau)),
                            "tau_step_norm": float(np.linalg.norm(np.diff(tau, axis=0))),
                            "forward_response_residual_mse": float(
                                solve_meta.get("forward_response_residual_mse", np.nan)
                            ),
                            "forward_prediction_norm": float(solve_meta.get("forward_prediction_norm", np.nan)),
                            "forward_profile_energy": float(solve_meta.get("forward_profile_energy", np.nan)),
                        }
                    )
                    row.update(condition)
                    row.update(_metrics(z_hat, z_true))
                    variant_rows.append(row)
    return variant_rows, decoder_rows, model_rows, fit_rows


def _summarize_variant_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    group_cols = [
        "latent",
        "feature_space_mode",
        "tau_variant",
        "target_response",
        "z_prior_precision",
        "observation_scale",
    ]
    summary = (
        rows.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_pred_norm=("feature_pred_norm", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            mean_forward_response_residual_mse=("forward_response_residual_mse", "mean"),
            mean_true_z_forward_mse_to_observed=("true_z_forward_mse_to_observed", "mean"),
            mean_true_z_forward_mse_to_known=("true_z_forward_mse_to_known", "mean"),
            median_z_design_condition=("z_design_condition", "median"),
            median_z_design_singular_min=("z_design_singular_min", "median"),
            median_z_design_rank=("z_design_rank", "median"),
        )
        .sort_values(["target_response", "z_prior_precision", "observation_scale", "tau_variant"])
    )
    overall = (
        rows.groupby(
            ["latent", "feature_space_mode", "tau_variant", "target_response", "z_prior_precision"],
            as_index=False,
        )
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_pred_norm=("feature_pred_norm", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
            mean_forward_response_residual_mse=("forward_response_residual_mse", "mean"),
            mean_true_z_forward_mse_to_observed=("true_z_forward_mse_to_observed", "mean"),
            mean_true_z_forward_mse_to_known=("true_z_forward_mse_to_known", "mean"),
            median_z_design_condition=("z_design_condition", "median"),
            median_z_design_singular_min=("z_design_singular_min", "median"),
            median_z_design_rank=("z_design_rank", "median"),
        )
        .sort_values(["target_response", "z_prior_precision", "tau_variant"])
    )
    overall["observation_scale"] = "all"
    return pd.concat([summary, overall[summary.columns]], ignore_index=True)


def _summarize_decoder_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    group_cols = ["latent", "feature_space_mode", "diagnostic", "observation_scale"]
    summary = (
        rows.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_pred_norm=("feature_pred_norm", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
        )
        .sort_values(["observation_scale", "diagnostic"])
    )
    overall = (
        rows.groupby(["latent", "feature_space_mode", "diagnostic"], as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_pred_norm=("feature_pred_norm", "mean"),
            median_feature_pred_norm=("feature_pred_norm", "median"),
        )
        .sort_values("diagnostic")
    )
    overall["observation_scale"] = "all"
    return pd.concat([summary, overall[summary.columns]], ignore_index=True)


def _variant_contrasts(rows: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    if rows.empty:
        return rows
    key_cols = [
        "latent",
        "feature_space_mode",
        "table_index",
        "trial_id",
        "target_response",
        "z_prior_precision",
        "observation_scale",
        "prior_family",
        "true_source_row",
    ]
    pivot = rows.pivot_table(index=key_cols, columns="tau_variant", values="feature_cosine", aggfunc="first")
    pairs = [
        ("true", "zero", "true_minus_zero"),
        ("true", "estimated", "true_minus_estimated"),
        ("estimated", "zero", "estimated_minus_zero"),
        ("neg_true", "true", "neg_true_minus_true"),
        ("mean_centered_true", "true", "mean_centered_true_minus_true"),
        ("start_centered_true", "true", "start_centered_true_minus_true"),
        ("nearest_catalog", "true", "nearest_catalog_minus_true"),
    ]
    rng = np.random.default_rng(int(seed))
    out_rows: list[dict[str, Any]] = []
    for lhs, rhs, name in pairs:
        if lhs not in pivot.columns or rhs not in pivot.columns:
            continue
        vals = (pivot[lhs] - pivot[rhs]).rename("delta").reset_index()
        vals = vals[np.isfinite(vals["delta"].to_numpy(dtype=float))]
        for keys, group in vals.groupby(
            ["latent", "feature_space_mode", "target_response", "z_prior_precision", "observation_scale"],
            sort=True,
        ):
            values = group["delta"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(values, rng, int(n_boot))
            latent, feature_space_mode, target_response, precision, scale = keys
            out_rows.append(
                {
                    "contrast": name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "latent": str(latent),
                    "feature_space_mode": str(feature_space_mode),
                    "target_response": str(target_response),
                    "z_prior_precision": float(precision),
                    "observation_scale": scale,
                    "n": int(values.size),
                    "mean_feature_cosine_delta": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(out_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = observer_build_parser()
    parser.description = __doc__
    parser.set_defaults(
        out_dir=OUT_DIR,
        observer_modes="response_only,pose_known_forward_model,hidden_joint_forward_model,estimated_tau_forward_model,zero_tau_forward_model",
    )
    parser.add_argument("--tau-variant-scales", default="-2,-1,-0.5,0.5,1,2")
    parser.add_argument("--z-prior-precision-grid", default="0.01,0.1,1,10,100")
    return parser


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    manifest_path = Path(args.response_manifest) if args.response_manifest else run_dir / "response_cache_manifest.csv"
    manifest = _read_manifest(
        manifest_path,
        scales=_parse_scales(str(args.scales)),
        prior_family_filter=_parse_csv_values(str(args.prior_family_filter)),
        skip_tables=int(args.skip_tables),
        max_tables=int(args.max_tables),
    )
    sidecar_dir = (
        Path(args.trajectory_sidecar_dir)
        if args.trajectory_sidecar_dir is not None
        else run_dir / "continuous_joint_trajectory_sidecars"
    )
    if not sidecar_dir.exists():
        sidecar_dir = None
    trajectory_npz = _load_npz(Path(args.trajectory_npz)) if args.trajectory_npz is not None else None
    first_table = _load_table_with_sidecar(
        run_dir=run_dir,
        response_cache_path=str(manifest.iloc[0]["response_cache_path"]),
        trajectory_sidecar_dir=sidecar_dir,
    )
    n_units = int(np.asarray(first_table["y_obs_counts"]).shape[1])
    basis, basis_meta = _load_basis(
        Path(args.compact_basis_path),
        n_units=n_units,
        basis_key=str(args.basis_key),
        max_dim=int(args.basis_max_dim),
    )
    feature_table, feature_meta = _load_feature_table(Path(args.feature_npz), latent=str(args.latent))
    feature_weights, feature_weight_meta = _load_feature_weights(
        Path(args.feature_weights_npz) if args.feature_weights_npz is not None else None,
        latent=str(args.latent),
        raw_feature_dim=int(feature_table.features.shape[1]),
    )
    banks, tests, pooled_fit_rows = _build_sample_banks(
        run_dir=run_dir,
        manifest=manifest,
        basis=basis,
        feature_sources=set(int(value) for value in feature_table.source_rows.tolist()),
        trajectory_npz=trajectory_npz,
        trajectory_key=str(args.trajectory_key),
        observed_trajectory_key=str(args.observed_trajectory_key),
        trajectory_sidecar_dir=sidecar_dir,
        continuous_args=args,
        compute_pooled_tau_hat=False,
        progress_every=int(args.progress_every),
    )
    known_compact = _known_compact_by_row(banks, tests)
    fold_by_source = _assign_source_folds(
        tests.rows["true_source_row"].to_numpy(dtype=int),
        n_folds=int(args.n_folds),
        seed=int(args.fold_seed),
    )
    feature_space_modes = _parse_str_list(args.feature_space_modes)
    if not feature_space_modes:
        feature_space_modes = ["fold_zscore_whitened_pca"]
    canonical_modes = _canonical_feature_modes(feature_space_modes)
    global_transforms: dict[str, Any] = {}
    for mode in canonical_modes:
        if _feature_space_config(mode)["fit_scope"] == "global":
            global_transforms[mode] = _fit_feature_transform(
                feature_table,
                fit_sources=feature_table.source_rows,
                feature_dim=int(args.feature_dim),
                feature_space_mode=mode,
                feature_weights=feature_weights,
            )

    tau_variant_scales = _parse_float_list(str(args.tau_variant_scales))
    z_prior_precisions = _parse_float_list(str(args.z_prior_precision_grid))
    if not z_prior_precisions:
        z_prior_precisions = [float(args.forward_model_z_prior_precision)]

    variant_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    test_folds = tests.rows["true_source_row"].map(fold_by_source).to_numpy(dtype=int)
    for mode in canonical_modes:
        for fold in sorted(set(test_folds.tolist())):
            transform = _fit_transform_for_fold(
                mode=mode,
                fold=int(fold),
                fold_by_source=fold_by_source,
                feature_table=feature_table,
                feature_dim=int(args.feature_dim),
                feature_weights=feature_weights,
                global_transforms=global_transforms,
            )
            fold_variant, fold_decoder, fold_model, fold_fit = _diagnose_fold(
                banks=banks,
                tests=tests,
                known_compact=known_compact,
                feature_table=feature_table,
                transform=transform,
                fold_by_source=fold_by_source,
                fold=int(fold),
                args=args,
                tau_variant_scales=tau_variant_scales,
                z_prior_precisions=z_prior_precisions,
            )
            variant_rows.extend(fold_variant)
            decoder_rows.extend(fold_decoder)
            model_rows.extend(fold_model)
            fit_rows.extend(fold_fit)

    variants = pd.DataFrame(variant_rows)
    decoders = pd.DataFrame(decoder_rows)
    models = pd.DataFrame(model_rows)
    fit = pd.concat([pooled_fit_rows, pd.DataFrame(fit_rows)], ignore_index=True, sort=False)
    variant_summary = _summarize_variant_rows(variants)
    decoder_summary = _summarize_decoder_rows(decoders)
    contrasts = _variant_contrasts(variants, n_boot=int(args.n_bootstrap), seed=int(args.fold_seed) + 97)

    paths = {
        "tau_variants": out_dir / "forward_model_contract_tau_variants.csv",
        "tau_variant_summary": out_dir / "forward_model_contract_tau_variant_summary.csv",
        "tau_variant_contrasts": out_dir / "forward_model_contract_tau_variant_contrasts.csv",
        "decoder_rows": out_dir / "forward_model_contract_decoder_rows.csv",
        "decoder_summary": out_dir / "forward_model_contract_decoder_summary.csv",
        "models": out_dir / "forward_model_contract_models.csv",
        "fit_rows": out_dir / "forward_model_contract_fit_rows.csv",
        "manifest": out_dir / "forward_model_contract_manifest.json",
    }
    variants.to_csv(paths["tau_variants"], index=False)
    variant_summary.to_csv(paths["tau_variant_summary"], index=False)
    contrasts.to_csv(paths["tau_variant_contrasts"], index=False)
    decoders.to_csv(paths["decoder_rows"], index=False)
    decoder_summary.to_csv(paths["decoder_summary"], index=False)
    models.to_csv(paths["models"], index=False)
    fit.to_csv(paths["fit_rows"], index=False)

    manifest_payload = {
        "analysis": "panel_c_forward_model_contract_diagnostic",
        "cache_contract": {
            "y_obs_definition": (
                "The response-table writer stores y_obs_counts as known_lambda_counts[true_candidate_index]. "
                "The known-response direct decoder is therefore an identity/control check, not an independent "
                "known-pose response target."
            ),
            "known_response_direct_decoder_status": "quarantined_identity_control",
        },
        "source_observer_out_dir": OBSERVER_OUT_DIR,
        "run_dir": run_dir,
        "response_manifest": manifest_path,
        "trajectory_sidecar_dir": sidecar_dir,
        "n_response_tables": int(manifest.shape[0]),
        "feature": {
            **feature_meta,
            "feature_dim_requested": int(args.feature_dim),
            "feature_space_modes_requested": feature_space_modes,
            "feature_weights": feature_weight_meta,
        },
        "basis": basis_meta,
        "crossfit": {"n_folds": int(args.n_folds), "fold_seed": int(args.fold_seed)},
        "tau_variant_scales": tau_variant_scales,
        "z_prior_precision_grid": z_prior_precisions,
        "compute_pooled_tau_hat": False,
        "diagnostic_questions": [
            "Does F(z_true, tau_true) predict observed/known compact responses?",
            "Does the known-response identity control reproduce the observed-response direct decoder?",
            "Do transformed tau conventions beat tau_true?",
            "Do estimated tau paths improve fixed-tau z-design conditioning?",
            "Are fixed-tau solves sensitive to z prior precision?",
        ],
        "outputs": paths,
    }
    paths["manifest"].write_text(
        json.dumps(_json_ready(manifest_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
