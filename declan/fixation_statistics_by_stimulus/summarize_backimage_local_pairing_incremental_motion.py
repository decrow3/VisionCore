"""Incremental local-pairing motion decoding for saved BackImage caches.

This cache-only posthoc is the local-pairing counterpart to
``summarize_backimage_aggregate_incremental_motion``. It keeps two conventions
that matter for the local analysis:

* motion summaries are decoded as static mean response plus a motion-rendered
  response summary, so ``delta_mean`` is an increment over the stabilized image
  baseline rather than a zero-baseline response alone;
* sampled families such as ``matched_unpaired_empirical`` are summarized by the
  mean over sample decoders, not by decoding the averaged response vector.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from .summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _decode,
        _decode_groups_from_images,
        _filter_latents,
        _information_fold_rows,
        _load_npz,
        _parse_contrast_pairs,
        _parse_float_list,
        _parse_int_list,
        _parse_list,
        _response_key,
        _session_bootstrap_delta,
        _summarize_information_contrast,
        _summarize_information_rows,
        _validate_information_intervals,
        _write_csv,
        _write_json,
    )
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _decode,
        _decode_groups_from_images,
        _filter_latents,
        _information_fold_rows,
        _load_npz,
        _parse_contrast_pairs,
        _parse_float_list,
        _parse_int_list,
        _parse_list,
        _response_key,
        _session_bootstrap_delta,
        _summarize_information_contrast,
        _summarize_information_rows,
        _validate_information_intervals,
        _write_csv,
        _write_json,
    )


DEFAULT_LOCAL_RUN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_0p5_1_seed7_v1"
)


def _sample_sort_key(key: str) -> tuple[str, int | str]:
    family = key.split("__", 2)[1]
    suffix = family.rsplit("_sample", 1)[-1]
    return (family.rsplit("_sample", 1)[0], int(suffix) if suffix.isdigit() else suffix)


def _sample_response_keys(
    responses: dict[str, np.ndarray],
    *,
    summary: str,
    family: str,
    scale_id: str,
) -> list[str]:
    prefix = f"{summary}__{family}_sample"
    suffix = f"__{scale_id}"
    return sorted(
        [key for key in responses if key.startswith(prefix) and key.endswith(suffix)],
        key=_sample_sort_key,
    )


def _mean_sample_information_rows(
    rows_by_sample: list[list[dict[str, Any]]],
    *,
    family: str,
) -> list[dict[str, Any]]:
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for rows in rows_by_sample:
        for row in rows:
            by_fold.setdefault(int(row["fold"]), []).append(row)
    out: list[dict[str, Any]] = []
    for fold in sorted(by_fold):
        rows = by_fold[fold]
        first = dict(rows[0])
        n_samples = len(rows)
        for column in (
            "incremental_gain_info_diag_bits",
            "incremental_gain_info_diag_bits_per_dim",
            "incremental_gain_info_full_bits",
            "incremental_gain_info_full_bits_per_dim",
        ):
            first[column] = float(np.nanmean([float(row[column]) for row in rows]))
        first["family"] = family
        first["ridge_alpha_matched"] = bool(all(bool(row["ridge_alpha_matched"]) for row in rows))
        first["baseline_alpha"] = float(np.nanmedian([float(row["baseline_alpha"]) for row in rows]))
        first["condition_alpha"] = float(np.nanmedian([float(row["condition_alpha"]) for row in rows]))
        first["information_estimator"] = "mean_over_sample_information_gains"
        first["n_unpaired_samples"] = int(n_samples)
        out.append(first)
    return out


def _decode_augmented_information(
    *,
    X_static: np.ndarray,
    X_motion: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    static_result: dict[str, Any],
    static_scores: np.ndarray,
    k: int,
    alphas: list[float],
    alpha_mode: str,
    fixed_alpha: float | None,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    variance_floor: float,
    allow_unmatched_alpha: bool,
    motion_summary: str,
    family: str,
    scale_id: str,
    latent: str,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    X_aug = np.concatenate([X_static, X_motion], axis=1)
    result = _decode(
        X_aug,
        Z,
        groups,
        k=int(k),
        alphas=alphas,
        alpha_mode=alpha_mode,
        fixed_alpha=fixed_alpha,
        outer_folds=int(outer_folds),
        inner_folds=int(inner_folds),
        seed=int(seed),
    )
    gain = np.asarray(result["per_window_score"], dtype=np.float64) - static_scores
    rows = _information_fold_rows(
        condition_result=result,
        baseline_result=static_result,
        motion_summary=motion_summary,
        family=family,
        scale_id=scale_id,
        latent=latent,
        k=int(k),
        variance_floor=float(variance_floor),
    )
    if rows and not bool(allow_unmatched_alpha):
        alpha_matched = all(bool(row["ridge_alpha_matched"]) for row in rows)
        if not alpha_matched:
            raise ValueError(
                "Information gain requires matched ridge alpha between static and "
                "static-plus-motion folds. Use fixed ridge alpha, shared-alpha decoding, "
                "or pass --allow-unmatched-alpha-information for audit-only output."
            )
    return result, gain, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_LOCAL_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--latent-npz",
        type=Path,
        default=None,
        help=(
            "Optional latent feature NPZ. Defaults to latent_feature_arrays.npz "
            "inside --run-dir; useful for cache-only feature-space screens."
        ),
    )
    parser.add_argument("--summaries", default="delta_mean")
    parser.add_argument(
        "--families",
        default=(
            "actual_paired_empirical,matched_unpaired_empirical,rotated_actual_90,"
            "ou_matched_actual,brownian_matched_actual"
        ),
    )
    parser.add_argument(
        "--contrast-pairs",
        default=(
            "actual_paired_empirical:matched_unpaired_empirical,"
            "actual_paired_empirical:rotated_actual_90,"
            "actual_paired_empirical:ou_matched_actual,"
            "actual_paired_empirical:brownian_matched_actual"
        ),
    )
    parser.add_argument("--scale-ids", default="all")
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--pca-k-list", default="4,8")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--ridge-alpha-mode", choices=("fixed", "nested_per_candidate"), default="fixed")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--decode-group-mode",
        choices=("image", "source_trial", "session"),
        default="source_trial",
        help=(
            "CV grouping for decoding. source_trial groups all windows from the same "
            "session/trial; image keeps each selected crop/source row in one fold; "
            "session is stricter by recording session."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--max-sample-keys-per-family",
        type=int,
        default=0,
        help=(
            "For sampled families, use at most this many deterministic sample arrays. "
            "The default 0 uses every cached sample."
        ),
    )
    parser.add_argument("--information-variance-floor", type=float, default=1e-12)
    parser.add_argument(
        "--allow-unmatched-alpha-information",
        action="store_true",
        help="Allow information columns when static and static-plus-motion folds chose different ridge alphas.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "local_incremental_static_mean_plus_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    images = pd.read_csv(run_dir / "analysis_images.csv")
    sessions = images["session"].to_numpy()
    decode_groups = _decode_groups_from_images(images, str(args.decode_group_mode))
    responses = _load_npz(run_dir / "response_summary_arrays.npz")
    latent_npz = Path(args.latent_npz) if args.latent_npz is not None else run_dir / "latent_feature_arrays.npz"
    latents = _filter_latents(_load_npz(latent_npz), _parse_list(args.latent_names))
    summaries = _parse_list(args.summaries)
    families = _parse_list(args.families)
    contrast_pairs = _parse_contrast_pairs(args.contrast_pairs)
    scale_ids = _parse_list(args.scale_ids)
    if not scale_ids or "all" in scale_ids:
        scale_ids = _available_scale_ids(responses, families, summaries)
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    alpha_mode = str(args.ridge_alpha_mode)
    pca_k_list = _parse_int_list(args.pca_k_list)
    max_sample_keys = max(0, int(args.max_sample_keys_per_family))

    decode_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    per_window_gain_rows: list[dict[str, Any]] = []
    per_window_contrast_rows: list[dict[str, Any]] = []
    information_fold_rows: list[dict[str, Any]] = []
    per_window_by_key: dict[tuple[str, str, str, str, int], np.ndarray] = {}
    info_rows_by_key: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    static_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    image_records = images.to_dict(orient="records")

    for summary in summaries:
        static_summary = STATIC_SUMMARY_FOR_MOTION.get(summary)
        if static_summary is None:
            raise ValueError(f"No static summary mapping is defined for {summary!r}")
        static_key = _response_key(static_summary, "static", "static")
        if static_key not in responses:
            raise ValueError(f"Missing static response array {static_key!r}")
        X_static = responses[static_key]
        for latent_name, Z in latents.items():
            for k in pca_k_list:
                cache_key = (static_summary, latent_name, int(k))
                if cache_key not in static_cache:
                    static_cache[cache_key] = _decode(
                        X_static,
                        Z,
                        decode_groups,
                        k=int(k),
                        alphas=alphas,
                        alpha_mode=alpha_mode,
                        fixed_alpha=fixed_alpha,
                        outer_folds=int(args.outer_folds),
                        inner_folds=int(args.inner_folds),
                        seed=int(args.seed),
                    )
                static_result = static_cache[cache_key]
                static_scores = np.asarray(static_result["per_window_score"], dtype=np.float64)
                decode_rows.append(
                    {
                        "motion_summary": summary,
                        "static_summary": static_summary,
                        "model": "static_only",
                        "family": "static",
                        "scale_id": "static",
                        "latent": latent_name,
                        "k": int(k),
                        "mean_neg_mse": float(static_result["mean_neg_mse"]),
                        "r2": float(static_result["r2"]),
                        "chosen_alpha_median": float(static_result["chosen_alpha_median"]),
                        "ridge_alpha_mode": str(static_result["ridge_alpha_mode"]),
                        "fixed_ridge_alpha": float(fixed_alpha) if alpha_mode == "fixed" else float("nan"),
                        "target_dim": int(static_result["target_dim"]),
                        "n_images": int(X_static.shape[0]),
                        "decode_group_mode": str(args.decode_group_mode),
                        "n_decode_groups": int(np.unique(decode_groups).size),
                        "feature_dim": int(X_static.shape[1]),
                    }
                )
                for scale_id in scale_ids:
                    for family in families:
                        sample_keys = _sample_response_keys(
                            responses,
                            summary=summary,
                            family=family,
                            scale_id=scale_id,
                        )
                        if max_sample_keys > 0 and len(sample_keys) > max_sample_keys:
                            sample_keys = sample_keys[:max_sample_keys]
                        sample_gains: list[np.ndarray] = []
                        sample_information_rows: list[list[dict[str, Any]]] = []
                        if sample_keys:
                            for sample_key in sample_keys:
                                sample_family = sample_key.split("__", 2)[1]
                                result, gain, rows = _decode_augmented_information(
                                    X_static=X_static,
                                    X_motion=responses[sample_key],
                                    Z=Z,
                                    groups=decode_groups,
                                    static_result=static_result,
                                    static_scores=static_scores,
                                    k=int(k),
                                    alphas=alphas,
                                    alpha_mode=alpha_mode,
                                    fixed_alpha=fixed_alpha,
                                    outer_folds=int(args.outer_folds),
                                    inner_folds=int(args.inner_folds),
                                    seed=int(args.seed),
                                    variance_floor=float(args.information_variance_floor),
                                    allow_unmatched_alpha=bool(args.allow_unmatched_alpha_information),
                                    motion_summary=summary,
                                    family=sample_family,
                                    scale_id=scale_id,
                                    latent=latent_name,
                                )
                                sample_gains.append(gain)
                                sample_information_rows.append(rows)
                                decode_rows.append(
                                    {
                                        "motion_summary": summary,
                                        "static_summary": static_summary,
                                        "model": "static_plus_motion_sample",
                                        "family": sample_family,
                                        "parent_family": family,
                                        "scale_id": scale_id,
                                        "latent": latent_name,
                                        "k": int(k),
                                        "mean_neg_mse": float(result["mean_neg_mse"]),
                                        "r2": float(result["r2"]),
                                        "chosen_alpha_median": float(result["chosen_alpha_median"]),
                                        "ridge_alpha_mode": str(result["ridge_alpha_mode"]),
                                        "fixed_ridge_alpha": float(fixed_alpha) if alpha_mode == "fixed" else float("nan"),
                                        "target_dim": int(result["target_dim"]),
                                        "n_images": int(responses[sample_key].shape[0]),
                                        "decode_group_mode": str(args.decode_group_mode),
                                        "n_decode_groups": int(np.unique(decode_groups).size),
                                        "feature_dim": int(X_static.shape[1] + responses[sample_key].shape[1]),
                                    }
                                )
                            mean_gain = np.nanmean(np.vstack(sample_gains), axis=0)
                            rows = _mean_sample_information_rows(sample_information_rows, family=family)
                            information_fold_rows.extend(rows)
                            key = (summary, family, scale_id, latent_name, int(k))
                            per_window_by_key[key] = mean_gain
                            info_rows_by_key[key] = rows
                            boot = _session_bootstrap_delta(
                                mean_gain,
                                np.zeros_like(mean_gain),
                                sessions,
                                rng=rng,
                                n_bootstrap=int(args.n_bootstrap),
                            )
                            row = {
                                "motion_summary": summary,
                                "static_summary": static_summary,
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "incremental_gain_neg_mse": boot["mean"],
                                "ci95_low": boot["ci_low"],
                                "ci95_high": boot["ci_high"],
                                "n_images": boot["n"],
                                "n_sessions": boot["n_sessions"],
                                "gain_estimator": "mean_over_sample_decoder_gains",
                                "information_estimator": "mean_over_sample_information_gains",
                                "n_unpaired_samples": int(len(sample_keys)),
                                "information_variance_floor": float(args.information_variance_floor),
                            }
                            row.update(_summarize_information_rows(rows, rng=rng, n_bootstrap=int(args.n_bootstrap)))
                            gain_rows.append(row)
                            for image_row, gain_value in zip(image_records, mean_gain, strict=True):
                                per_window_gain_rows.append(
                                    {
                                        "motion_summary": summary,
                                        "static_summary": static_summary,
                                        "family": family,
                                        "scale_id": scale_id,
                                        "latent": latent_name,
                                        "k": int(k),
                                        "incremental_gain_neg_mse": float(gain_value),
                                        "gain_estimator": "mean_over_sample_decoder_gains",
                                        "n_unpaired_samples": int(len(sample_keys)),
                                        **image_row,
                                    }
                                )
                            continue

                        motion_key = _response_key(summary, family, scale_id)
                        if motion_key not in responses:
                            continue
                        result, gain, rows = _decode_augmented_information(
                            X_static=X_static,
                            X_motion=responses[motion_key],
                            Z=Z,
                            groups=decode_groups,
                            static_result=static_result,
                            static_scores=static_scores,
                            k=int(k),
                            alphas=alphas,
                            alpha_mode=alpha_mode,
                            fixed_alpha=fixed_alpha,
                            outer_folds=int(args.outer_folds),
                            inner_folds=int(args.inner_folds),
                            seed=int(args.seed),
                            variance_floor=float(args.information_variance_floor),
                            allow_unmatched_alpha=bool(args.allow_unmatched_alpha_information),
                            motion_summary=summary,
                            family=family,
                            scale_id=scale_id,
                            latent=latent_name,
                        )
                        information_fold_rows.extend(rows)
                        key = (summary, family, scale_id, latent_name, int(k))
                        per_window_by_key[key] = gain
                        info_rows_by_key[key] = rows
                        decode_rows.append(
                            {
                                "motion_summary": summary,
                                "static_summary": static_summary,
                                "model": "static_plus_motion",
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "mean_neg_mse": float(result["mean_neg_mse"]),
                                "r2": float(result["r2"]),
                                "chosen_alpha_median": float(result["chosen_alpha_median"]),
                                "ridge_alpha_mode": str(result["ridge_alpha_mode"]),
                                "fixed_ridge_alpha": float(fixed_alpha) if alpha_mode == "fixed" else float("nan"),
                                "target_dim": int(result["target_dim"]),
                                "n_images": int(responses[motion_key].shape[0]),
                                "decode_group_mode": str(args.decode_group_mode),
                                "n_decode_groups": int(np.unique(decode_groups).size),
                                "feature_dim": int(X_static.shape[1] + responses[motion_key].shape[1]),
                            }
                        )
                        boot = _session_bootstrap_delta(
                            gain,
                            np.zeros_like(gain),
                            sessions,
                            rng=rng,
                            n_bootstrap=int(args.n_bootstrap),
                        )
                        row = {
                            "motion_summary": summary,
                            "static_summary": static_summary,
                            "family": family,
                            "scale_id": scale_id,
                            "latent": latent_name,
                            "k": int(k),
                            "incremental_gain_neg_mse": boot["mean"],
                            "ci95_low": boot["ci_low"],
                            "ci95_high": boot["ci_high"],
                            "n_images": boot["n"],
                            "n_sessions": boot["n_sessions"],
                            "gain_estimator": "direct_family_gain",
                            "information_estimator": "direct_family_information_gain",
                            "n_unpaired_samples": 0,
                            "information_variance_floor": float(args.information_variance_floor),
                        }
                        row.update(_summarize_information_rows(rows, rng=rng, n_bootstrap=int(args.n_bootstrap)))
                        gain_rows.append(row)
                        for image_row, gain_value in zip(image_records, gain, strict=True):
                            per_window_gain_rows.append(
                                {
                                    "motion_summary": summary,
                                    "static_summary": static_summary,
                                    "family": family,
                                    "scale_id": scale_id,
                                    "latent": latent_name,
                                    "k": int(k),
                                    "incremental_gain_neg_mse": float(gain_value),
                                    "gain_estimator": "direct_family_gain",
                                    "n_unpaired_samples": 0,
                                    **image_row,
                                }
                            )

                    for lhs, rhs in contrast_pairs:
                        lhs_key = (summary, lhs, scale_id, latent_name, int(k))
                        rhs_key = (summary, rhs, scale_id, latent_name, int(k))
                        if lhs_key not in per_window_by_key or rhs_key not in per_window_by_key:
                            continue
                        boot = _session_bootstrap_delta(
                            per_window_by_key[lhs_key],
                            per_window_by_key[rhs_key],
                            sessions,
                            rng=rng,
                            n_bootstrap=int(args.n_bootstrap),
                        )
                        row = {
                            "motion_summary": summary,
                            "static_summary": static_summary,
                            "lhs_family": lhs,
                            "rhs_family": rhs,
                            "scale_id": scale_id,
                            "latent": latent_name,
                            "k": int(k),
                            "incremental_gain_delta_neg_mse": boot["mean"],
                            "ci95_low": boot["ci_low"],
                            "ci95_high": boot["ci_high"],
                            "n_images": boot["n"],
                            "n_sessions": boot["n_sessions"],
                            "lhs_gain_estimator": next(
                                (
                                    gain_row["gain_estimator"]
                                    for gain_row in gain_rows
                                    if gain_row["motion_summary"] == summary
                                    and gain_row["family"] == lhs
                                    and gain_row["scale_id"] == scale_id
                                    and gain_row["latent"] == latent_name
                                    and int(gain_row["k"]) == int(k)
                                ),
                                "",
                            ),
                            "rhs_gain_estimator": next(
                                (
                                    gain_row["gain_estimator"]
                                    for gain_row in gain_rows
                                    if gain_row["motion_summary"] == summary
                                    and gain_row["family"] == rhs
                                    and gain_row["scale_id"] == scale_id
                                    and gain_row["latent"] == latent_name
                                    and int(gain_row["k"]) == int(k)
                                ),
                                "",
                            ),
                            "information_variance_floor": float(args.information_variance_floor),
                        }
                        row.update(
                            _summarize_information_contrast(
                                info_rows_by_key[lhs_key],
                                info_rows_by_key[rhs_key],
                                rng=rng,
                                n_bootstrap=int(args.n_bootstrap),
                            )
                        )
                        contrast_rows.append(row)
                        contrast = per_window_by_key[lhs_key] - per_window_by_key[rhs_key]
                        for image_row, contrast_value in zip(image_records, contrast, strict=True):
                            per_window_contrast_rows.append(
                                {
                                    "motion_summary": summary,
                                    "static_summary": static_summary,
                                    "lhs_family": lhs,
                                    "rhs_family": rhs,
                                    "scale_id": scale_id,
                                    "latent": latent_name,
                                    "k": int(k),
                                    "incremental_gain_delta_neg_mse": float(contrast_value),
                                    **image_row,
                                }
                            )

    _validate_information_intervals(gain_rows, value_col="incremental_gain_info_diag_bits")
    _validate_information_intervals(contrast_rows, value_col="incremental_gain_delta_info_diag_bits")

    _write_csv(out_dir / "incremental_decode_summary.csv", decode_rows)
    _write_csv(out_dir / "incremental_gain_vs_static.csv", gain_rows)
    _write_csv(out_dir / "incremental_gain_contrasts.csv", contrast_rows)
    _write_csv(out_dir / "incremental_gain_by_window.csv", per_window_gain_rows)
    _write_csv(out_dir / "incremental_gain_contrasts_by_window.csv", per_window_contrast_rows)
    _write_csv(out_dir / "incremental_information_by_fold.csv", information_fold_rows)
    _write_json(
        out_dir / "run_metadata.json",
        {
            "source_run_dir": run_dir,
            "analysis_images": run_dir / "analysis_images.csv",
            "response_summary_arrays": run_dir / "response_summary_arrays.npz",
            "latent_feature_arrays": latent_npz,
            "summaries": summaries,
            "families": families,
            "contrast_pairs": contrast_pairs,
            "scale_ids": scale_ids,
            "latent_names": list(latents),
            "pca_k_list": pca_k_list,
            "ridge_alphas": alphas,
            "ridge_alpha_mode": alpha_mode,
            "fixed_ridge_alpha": fixed_alpha,
            "outer_folds": int(args.outer_folds),
            "inner_folds": int(args.inner_folds),
            "decode_group_mode": str(args.decode_group_mode),
            "n_decode_groups": int(np.unique(decode_groups).size),
            "n_bootstrap": int(args.n_bootstrap),
            "max_sample_keys_per_family": max_sample_keys,
            "seed": int(args.seed),
            "static_baseline_contract": {
                "motion_summary_to_static_summary": {
                    summary: STATIC_SUMMARY_FOR_MOTION.get(summary) for summary in summaries
                },
                "interpretation": "motion summaries are incremental response features decoded with the static mean response baseline",
            },
            "sample_family_contract": {
                "sample_key_pattern": "<summary>__<family>_sampleK__<scale_id>",
                "gain_estimator": "mean_over_sample_decoder_gains",
                "information_estimator": "mean_over_sample_information_gains",
            },
            "information_axis": {
                "headline": "diagonal_gaussian_variational_bound_increment_bits",
                "diag_formula": "0.5 * sum_j log(var_static_j / var_condition_j) / log(2)",
                "full_covariance_formula": "0.5 * (logdet(cov_static) - logdet(cov_condition)) / log(2)",
                "ci_method": "outer_fold_weighted_bootstrap",
                "variance_floor": float(args.information_variance_floor),
                "allow_unmatched_alpha_information": bool(args.allow_unmatched_alpha_information),
            },
        },
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Local Pairing Incremental Motion Posthoc",
                "",
                "This cache-only posthoc re-scores local pairing response arrays as static mean plus motion summary.",
                "For sampled families, the promoted rows average decoder gains across samples rather than decoding an averaged response vector.",
                "",
                "Primary files:",
                "- `incremental_gain_vs_static.csv`",
                "- `incremental_gain_contrasts.csv`",
                "- `incremental_gain_by_window.csv`",
                "- `incremental_gain_contrasts_by_window.csv`",
                "- `incremental_information_by_fold.csv`",
                "- `run_metadata.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote local incremental summaries to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
