"""Local BackImage I_z pairing revisit.

This runner reopens the local BackImage active-sensing question after the
aggregate FEM result.  Unlike the aggregate runner, it keeps each image paired
with its own measured drift trace and compares that actual pairing against
matched unpaired empirical traces and synthetic controls.

The output intentionally mirrors ``run_backimage_aggregate_fem_information`` so
the incremental static-plus-motion summarizer can be reused directly.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .run_backimage_aggregate_fem_information import (
        DEFAULT_INPUT,
        HAVE_STEERABLE_PYRAMID,
        CanonicalTwinScorer,
        _add_temporal_basis_summaries,
        _align_response_to_trace,
        _build_trace_bank,
        _bootstrap_condition_delta,
        _clip_patch,
        _contrast_rows,
        _covariance_rows,
        _decode_rows,
        _eligible_trace_bank_indices,
        _extract_requested_latents,
        _family_raw_trace,
        _fit_temporal_basis,
        _fixed_dct_basis,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _prepare_windows,
        _scale_family_raw_trace,
        _scale_token,
        _session_dataset_cache,
        _stack_condition_features,
        _static_trace,
        _summarize_response,
        _trace_filter_kwargs,
        _trace_rms,
        _cross_validated_decode,
        _parse_contrast_pairs,
        _write_csv,
        _write_json,
        _path_length,
        _progress as _aggregate_progress,
    )
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        DEFAULT_INPUT,
        HAVE_STEERABLE_PYRAMID,
        CanonicalTwinScorer,
        _add_temporal_basis_summaries,
        _align_response_to_trace,
        _build_trace_bank,
        _bootstrap_condition_delta,
        _clip_patch,
        _contrast_rows,
        _covariance_rows,
        _decode_rows,
        _eligible_trace_bank_indices,
        _extract_requested_latents,
        _family_raw_trace,
        _fit_temporal_basis,
        _fixed_dct_basis,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _prepare_windows,
        _scale_family_raw_trace,
        _scale_token,
        _session_dataset_cache,
        _stack_condition_features,
        _static_trace,
        _summarize_response,
        _trace_filter_kwargs,
        _trace_rms,
        _cross_validated_decode,
        _parse_contrast_pairs,
        _write_csv,
        _write_json,
        _path_length,
        _progress as _aggregate_progress,
    )


DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_local_pairing_Iz_revisit"
)


@dataclass(frozen=True)
class LocalPairingConfig:
    input: str
    out_dir: str
    window_manifest: str | None
    max_images: int
    unpaired_samples_per_image: int
    candidate_families: list[str]
    observed_rms_scales: list[float]
    patch_size_px: int
    latent_crop_px: int
    center_crop_px: int
    local_field_grid: int
    n_timepoints: int
    temporal_pc_components: int
    pca_k_list: list[int]
    latent_names: list[str]
    ridge_alphas: list[float]
    fixed_ridge_alpha: float | None
    outer_folds: int
    inner_folds: int
    decode_group_mode: str
    max_rms_deg: float
    max_trace_source_rms_deg: float | None
    max_trace_source_radius_deg: float | None
    max_rendered_trace_path_length_deg: float | None
    max_source_trace_path_length_deg: float | None
    max_trace_source_speed_p95_deg_s: float | None
    max_trace_source_microsaccade_events: int | None
    unpaired_exclude_same_trial: bool
    reuse_trace_sources_across_scales: bool
    twin_batch_size: int
    twin_trace_batch_size: int
    device: str
    progress_every: int
    seed: int
    dry_run: bool


def _progress(message: str) -> None:
    print(f"[backimage-local-pairing-Iz] {message}", flush=True)


def _axis_trace(axis_deg: float, rms_radius_deg: float, n_timepoints: int) -> np.ndarray:
    theta = np.radians(float(axis_deg))
    amp = float(rms_radius_deg) * np.sqrt(2.0)
    t = np.linspace(0.0, 2.0 * np.pi, int(n_timepoints), endpoint=False)
    trace = amp * np.sin(t)[:, None] * np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)[None, :]
    trace -= np.mean(trace, axis=0, keepdims=True)
    return trace.astype(np.float32)


def _axis_constrained_temporal_trace(source_trace: np.ndarray, axis_deg: float) -> np.ndarray:
    """Put the actual trace's dominant 1D temporal waveform onto a requested axis."""
    source = np.asarray(source_trace, dtype=np.float64)
    source -= np.mean(source, axis=0, keepdims=True)
    theta = np.radians(float(axis_deg))
    axis = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    if source.shape[0] < 2 or _trace_rms(source) <= 1e-12:
        return np.zeros_like(source, dtype=np.float32)
    cov = np.cov(source, rowvar=False)
    if not np.all(np.isfinite(cov)):
        scalar = source[:, 0]
    else:
        vals, vecs = np.linalg.eigh(cov)
        pc = vecs[:, int(np.argmax(vals))]
        scalar = source @ pc
    scalar = scalar - np.mean(scalar)
    out = scalar[:, None] * axis[None, :]
    out -= np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32)


def _rotated_trace_fixed(trace: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.radians(float(degrees))
    rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    out = np.asarray(trace, dtype=np.float64) @ rot.T
    out -= np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32)


def _actual_trace_passes(item: dict[str, Any], args: argparse.Namespace) -> bool:
    source_row = int(item["source_row"])
    return bool(
        _eligible_trace_bank_indices(
            [item],
            current_source_row=source_row + 1_000_000,
            **_trace_filter_kwargs(args),
        )
    )


def _trace_match_features(item: dict[str, Any]) -> np.ndarray:
    vals = np.asarray(
        [
            np.log(float(item["observed_rms_deg"]) + 1e-6),
            np.log(float(item["path_length_deg"]) + 1e-6),
            np.log(float(item["source_max_radius_deg"]) + 1e-6),
            float(item.get("source_anisotropy", np.nan)),
            float(item.get("trace_cov_anisotropy", np.nan)),
            float(item.get("source_trace_cov_anisotropy", np.nan)),
            float(item["lag1_autocorr"]),
        ],
        dtype=np.float64,
    )
    vals[~np.isfinite(vals)] = 0.0
    return vals


def _matched_unpaired_indices(
    trace_bank: list[dict[str, Any]],
    actual_item: dict[str, Any],
    eligible: list[int],
    *,
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    if not eligible:
        return []
    features = np.vstack([_trace_match_features(trace_bank[j]) for j in eligible])
    center = _trace_match_features(actual_item)
    scale = np.nanstd(features, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    dist = np.sqrt(np.sum(((features - center) / scale) ** 2, axis=1))
    same_session = np.asarray([str(trace_bank[j]["session"]) == str(actual_item["session"]) for j in eligible])
    rank_score = dist - 0.25 * same_session.astype(np.float64) + rng.uniform(0.0, 1e-6, size=len(eligible))
    order = np.argsort(rank_score)
    return [int(eligible[i]) for i in order[: min(int(n), len(order))]]


def _motion_summary_rows(motion_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not motion_rows:
        return []
    df = pd.DataFrame(motion_rows)
    if "effective_rms_deg" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for (family, scale_id), block in df.loc[df["family"] != "static"].groupby(["family", "scale_id"], dropna=False):
        rows.append(
            {
                "family": family,
                "scale_id": scale_id,
                "n": int(block.shape[0]),
                "median_effective_rms_deg": float(np.nanmedian(block["effective_rms_deg"])),
                "iqr_effective_rms_deg": float(np.nanpercentile(block["effective_rms_deg"], 75) - np.nanpercentile(block["effective_rms_deg"], 25)),
                "median_effective_to_requested_rms": float(np.nanmedian(block["effective_to_requested_rms"])),
                "median_path_length_deg": float(np.nanmedian(block["path_length_deg"])),
                "clipped_fraction": float(np.nanmean(block["rms_clipped_high"].astype(float))),
            }
        )
    return rows


def _stack_condition_features_with_samples(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_name: str,
    *,
    sample_families: set[str],
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], dict[int, list[np.ndarray]]] = {}
    for rec in records:
        family = str(rec["family"])
        scale_id = "static" if family == "static" else str(rec["scale_id"])
        image_index = int(rec["image_index"])
        values = summaries[int(rec["response_id"])][summary_name]
        grouped.setdefault((family, scale_id), {}).setdefault(image_index, []).append(values)
        if family in sample_families:
            sample_family = f"{family}_sample{int(rec['sample_index'])}"
            grouped.setdefault((sample_family, scale_id), {}).setdefault(image_index, []).append(values)
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, by_image in grouped.items():
        out[key] = np.vstack(
            [np.mean(np.vstack(by_image[image_index]), axis=0) for image_index in sorted(by_image)]
        ).astype(np.float32)
    return out


def _decode_fixed(
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    *,
    k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    return _cross_validated_decode(
        X,
        Z,
        groups,
        k=int(k),
        alphas=alphas,
        alpha_mode="fixed",
        fixed_alpha=fixed_alpha,
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        seed=int(args.seed),
    )


def _run_local_incremental_decoding(
    feature_by_condition_by_summary: dict[str, dict[tuple[str, str], np.ndarray]],
    latent_arrays: dict[str, np.ndarray],
    sessions: np.ndarray,
    groups: np.ndarray,
    summary_names: list[str],
    families: list[str],
    scales: list[float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(args.seed) + 909)
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    contrast_pairs = _parse_contrast_pairs(str(args.contrast_pairs))
    pca_k_list = _parse_int_list(args.pca_k_list)
    scale_ids = [f"rel_{_scale_token(scale)}x" for scale in scales]
    decode_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for summary in summary_names:
        feature_by_condition = feature_by_condition_by_summary[summary]
        X_static = feature_by_condition[("static", "static")]
        for latent_name, Z in latent_arrays.items():
            for k in pca_k_list:
                static_result = _decode_fixed(X_static, Z, groups, k=int(k), args=args)
                static_scores = np.asarray(static_result["per_window_score"], dtype=np.float64)
                decode_rows.append(
                    {
                        "motion_summary": summary,
                        "model": "static_only",
                        "family": "static",
                        "scale_id": "static",
                        "latent": latent_name,
                        "k": int(k),
                        "mean_neg_mse": float(static_result["mean_neg_mse"]),
                        "r2": float(static_result["r2"]),
                        "chosen_alpha_median": float(static_result["chosen_alpha_median"]),
                        "target_dim": int(static_result["target_dim"]),
                        "n_images": int(X_static.shape[0]),
                        "decode_group_mode": str(args.decode_group_mode),
                        "n_decode_groups": int(np.unique(groups).size),
                        "feature_dim": int(X_static.shape[1]),
                        "ridge_alpha_mode": "fixed",
                        "fixed_ridge_alpha": fixed_alpha,
                    }
                )
                for scale_id in scale_ids:
                    gains_by_family: dict[str, np.ndarray] = {}
                    for family in families:
                        sample_keys = sorted(
                            key for key in feature_by_condition
                            if key[1] == scale_id and key[0].startswith(f"{family}_sample")
                        )
                        if sample_keys:
                            sample_gains = []
                            for sample_key in sample_keys:
                                X_aug = np.concatenate([X_static, feature_by_condition[sample_key]], axis=1)
                                result = _decode_fixed(X_aug, Z, groups, k=int(k), args=args)
                                sample_gain = np.asarray(result["per_window_score"], dtype=np.float64) - static_scores
                                sample_gains.append(sample_gain)
                                decode_rows.append(
                                    {
                                        "motion_summary": summary,
                                        "model": "static_plus_motion_sample",
                                        "family": sample_key[0],
                                        "parent_family": family,
                                        "scale_id": scale_id,
                                        "latent": latent_name,
                                        "k": int(k),
                                        "mean_neg_mse": float(result["mean_neg_mse"]),
                                        "r2": float(result["r2"]),
                                        "chosen_alpha_median": float(result["chosen_alpha_median"]),
                                        "target_dim": int(result["target_dim"]),
                                        "n_images": int(X_aug.shape[0]),
                                        "decode_group_mode": str(args.decode_group_mode),
                                        "n_decode_groups": int(np.unique(groups).size),
                                        "feature_dim": int(X_aug.shape[1]),
                                        "ridge_alpha_mode": "fixed",
                                        "fixed_ridge_alpha": fixed_alpha,
                                    }
                                )
                            gain = np.nanmean(np.vstack(sample_gains), axis=0)
                            gains_by_family[family] = gain
                            boot = _bootstrap_condition_delta(
                                gain,
                                np.zeros_like(gain),
                                sessions,
                                n_bootstrap=int(args.n_bootstrap),
                                rng=rng,
                            )
                            gain_rows.append(
                                {
                                    "motion_summary": summary,
                                    "family": family,
                                    "scale_id": scale_id,
                                    "latent": latent_name,
                                    "k": int(k),
                                    "incremental_gain_neg_mse": boot[0],
                                    "ci95_low": boot[1],
                                    "ci95_high": boot[2],
                                    "n_images": int(gain.size),
                                    "n_sessions": int(np.unique(sessions).size),
                                    "gain_estimator": "mean_over_unpaired_sample_gains",
                                    "n_unpaired_samples": int(len(sample_keys)),
                                }
                            )
                        condition_key = (family, scale_id)
                        if condition_key not in feature_by_condition:
                            continue
                        X_aug = np.concatenate([X_static, feature_by_condition[condition_key]], axis=1)
                        result = _decode_fixed(X_aug, Z, groups, k=int(k), args=args)
                        direct_gain = np.asarray(result["per_window_score"], dtype=np.float64) - static_scores
                        decode_rows.append(
                            {
                                "motion_summary": summary,
                                "model": "static_plus_motion",
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "mean_neg_mse": float(result["mean_neg_mse"]),
                                "r2": float(result["r2"]),
                                "chosen_alpha_median": float(result["chosen_alpha_median"]),
                                "target_dim": int(result["target_dim"]),
                                "n_images": int(X_aug.shape[0]),
                                "decode_group_mode": str(args.decode_group_mode),
                                "n_decode_groups": int(np.unique(groups).size),
                                "feature_dim": int(X_aug.shape[1]),
                                "ridge_alpha_mode": "fixed",
                                "fixed_ridge_alpha": fixed_alpha,
                            }
                        )
                        if family not in gains_by_family:
                            gains_by_family[family] = direct_gain
                            boot = _bootstrap_condition_delta(
                                direct_gain,
                                np.zeros_like(direct_gain),
                                sessions,
                                n_bootstrap=int(args.n_bootstrap),
                                rng=rng,
                            )
                            gain_rows.append(
                                {
                                    "motion_summary": summary,
                                    "family": family,
                                    "scale_id": scale_id,
                                    "latent": latent_name,
                                    "k": int(k),
                                    "incremental_gain_neg_mse": boot[0],
                                    "ci95_low": boot[1],
                                    "ci95_high": boot[2],
                                    "n_images": int(direct_gain.size),
                                    "n_sessions": int(np.unique(sessions).size),
                                    "gain_estimator": "direct_family_gain",
                                    "n_unpaired_samples": 0,
                                }
                            )
                    for lhs, rhs in contrast_pairs:
                        if lhs == "static" or rhs == "static":
                            continue
                        if lhs not in gains_by_family or rhs not in gains_by_family:
                            continue
                        boot = _bootstrap_condition_delta(
                            gains_by_family[lhs],
                            gains_by_family[rhs],
                            sessions,
                            n_bootstrap=int(args.n_bootstrap),
                            rng=rng,
                        )
                        contrast_rows.append(
                            {
                                "motion_summary": summary,
                                "lhs_family": lhs,
                                "rhs_family": rhs,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "incremental_gain_delta_neg_mse": boot[0],
                                "ci95_low": boot[1],
                                "ci95_high": boot[2],
                                "n_images": int(gains_by_family[lhs].size),
                                "n_sessions": int(np.unique(sessions).size),
                            }
                        )
    return decode_rows, gain_rows, contrast_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-manifest", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--unpaired-samples-per-image", type=int, default=4)
    parser.add_argument(
        "--candidate-families",
        default=(
            "actual_paired_empirical,matched_unpaired_empirical,rotated_actual_90,"
            "ou_matched_actual,brownian_matched_actual,edge_axis,edge_orthogonal"
        ),
    )
    parser.add_argument("--observed-rms-scales", default="0.25,0.5,1.0")
    parser.add_argument("--patch-size-px", type=int, default=160)
    parser.add_argument("--latent-crop-px", type=int, default=96)
    parser.add_argument("--center-crop-px", type=int, default=64)
    parser.add_argument("--local-field-grid", type=int, default=4)
    parser.add_argument(
        "--n-timepoints",
        type=int,
        default=48,
        help="Trace/movie length expected by the canonical BackImage twin stimulus path.",
    )
    parser.add_argument("--temporal-pc-components", type=int, default=4)
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--pca-k-list", default="4,8")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--contrast-pairs",
        default=(
            "actual_paired_empirical:matched_unpaired_empirical,"
            "actual_paired_empirical:rotated_actual_90,"
            "actual_paired_empirical:ou_matched_actual,"
            "actual_paired_empirical:brownian_matched_actual,"
            "edge_axis:edge_orthogonal,"
            "actual_paired_empirical:static"
        ),
    )
    parser.add_argument("--decode-group-mode", choices=("image", "session"), default="image")
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--max-rms-deg", type=float, default=0.12)
    parser.add_argument("--max-trace-source-rms-deg", type=float, default=0.06)
    parser.add_argument("--max-trace-source-radius-deg", type=float, default=0.2)
    parser.add_argument("--max-trace-source-path-length-deg", type=float, default=None)
    parser.add_argument("--max-rendered-trace-path-length-deg", type=float, default=1.5)
    parser.add_argument("--max-source-trace-path-length-deg", type=float, default=None)
    parser.add_argument("--max-trace-source-speed-p95-deg-s", type=float, default=20.0)
    parser.add_argument("--max-trace-source-microsaccade-events", type=int, default=0)
    parser.add_argument(
        "--unpaired-exclude-same-trial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude matched-unpaired traces from the same session/trial as the actual trace.",
    )
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=None)
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument("--reuse-trace-sources-across-scales", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--twin-batch-size", type=int, default=48)
    parser.add_argument("--twin-trace-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    target_accepted_images = int(args.max_images)
    prepare_args = argparse.Namespace(**vars(args))
    if args.window_manifest is None and target_accepted_images > 0:
        prepare_args.max_images = 0
    work = _prepare_windows(prepare_args)
    if work.empty:
        raise ValueError("No BackImage windows survived filters.")
    if not bool(args.reuse_trace_sources_across_scales):
        raise ValueError("The local pairing primary analysis requires trace sources to be reused across scales.")

    families = _parse_str_list(args.candidate_families)
    valid = {
        "actual_paired_empirical",
        "matched_unpaired_empirical",
        "rotated_actual_90",
        "rotated_actual_random",
        "ou_matched_actual",
        "brownian_matched_actual",
        "edge_axis",
        "edge_orthogonal",
    }
    invalid = sorted(set(families).difference(valid))
    if invalid:
        raise ValueError(f"Unknown candidate families: {invalid}")
    if {"edge_axis", "edge_orthogonal"}.intersection(families):
        if "image_edge_axis_deg" not in work.columns:
            raise ValueError("edge_axis/edge_orthogonal candidates require image_edge_axis_deg in the input table.")
        before = work.shape[0]
        work = work[np.isfinite(work["image_edge_axis_deg"].astype(float))].reset_index(drop=True)
        work["image_index"] = np.arange(work.shape[0], dtype=int)
        if work.empty:
            raise ValueError("No windows remain with finite image_edge_axis_deg.")
        if work.shape[0] < before:
            _progress(f"finite edge-axis filter keeps {work.shape[0]}/{before} windows")
    scales = _parse_float_list(args.observed_rms_scales)
    latent_filter = set(_parse_str_list(args.latent_names))
    if "all" in latent_filter:
        latent_filter = set()

    cfg = LocalPairingConfig(
        input=str(args.input),
        out_dir=str(out_dir),
        window_manifest=str(args.window_manifest) if args.window_manifest is not None else None,
        max_images=int(args.max_images),
        unpaired_samples_per_image=int(args.unpaired_samples_per_image),
        candidate_families=families,
        observed_rms_scales=scales,
        patch_size_px=int(args.patch_size_px),
        latent_crop_px=int(args.latent_crop_px),
        center_crop_px=int(args.center_crop_px),
        local_field_grid=int(args.local_field_grid),
        n_timepoints=int(args.n_timepoints),
        temporal_pc_components=int(args.temporal_pc_components),
        pca_k_list=_parse_int_list(args.pca_k_list),
        latent_names=sorted(latent_filter),
        ridge_alphas=_parse_float_list(args.ridge_alphas),
        fixed_ridge_alpha=float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else None,
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        decode_group_mode=str(args.decode_group_mode),
        max_rms_deg=float(args.max_rms_deg),
        max_trace_source_rms_deg=float(args.max_trace_source_rms_deg) if args.max_trace_source_rms_deg is not None else None,
        max_trace_source_radius_deg=float(args.max_trace_source_radius_deg) if args.max_trace_source_radius_deg is not None else None,
        max_rendered_trace_path_length_deg=(
            float(args.max_rendered_trace_path_length_deg) if args.max_rendered_trace_path_length_deg is not None else None
        ),
        max_source_trace_path_length_deg=(
            float(args.max_source_trace_path_length_deg) if args.max_source_trace_path_length_deg is not None else None
        ),
        max_trace_source_speed_p95_deg_s=(
            float(args.max_trace_source_speed_p95_deg_s) if args.max_trace_source_speed_p95_deg_s is not None else None
        ),
        max_trace_source_microsaccade_events=(
            int(args.max_trace_source_microsaccade_events) if args.max_trace_source_microsaccade_events is not None else None
        ),
        unpaired_exclude_same_trial=bool(args.unpaired_exclude_same_trial),
        reuse_trace_sources_across_scales=bool(args.reuse_trace_sources_across_scales),
        twin_batch_size=int(args.twin_batch_size),
        twin_trace_batch_size=int(args.twin_trace_batch_size),
        device=str(args.device),
        progress_every=int(args.progress_every),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
    )
    _write_json(out_dir / "run_metadata.json", {"config": asdict(cfg), "steerable_pyramid": HAVE_STEERABLE_PYRAMID})
    _progress(f"prepared {work.shape[0]} windows; families={families}; scales={scales}; dry_run={args.dry_run}")

    eyepos_by_session = _session_dataset_cache(work["session"].astype(str).to_list())
    trace_bank = _build_trace_bank(
        work,
        eyepos_by_session,
        int(args.n_timepoints),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps) if args.microsaccade_speed_threshold_dps is not None else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
    )
    by_source = {int(item["source_row"]): j for j, item in enumerate(trace_bank)}
    trace_rows = [
        {
            "bank_index": j,
            "source_row": int(item["source_row"]),
            "session": str(item["session"]),
            "trial_idx": int(item["trial_idx"]),
            "global_start": int(item["global_start"]),
            "global_stop": int(item["global_stop"]),
            "observed_rms_deg": float(item["observed_rms_deg"]),
            "source_trace_observed_rms_deg": float(item["source_trace_observed_rms_deg"]),
            "source_rms_radius_deg": float(item["source_rms_radius_deg"]),
            "source_max_radius_deg": float(item["source_max_radius_deg"]),
            "path_length_deg": float(item["path_length_deg"]),
            "source_path_length_deg": float(item["source_path_length_deg"]),
            "source_speed_p95_deg_s": float(item["source_speed_p95_deg_s"]),
            "duration_s": float(item["duration_s"]),
            "lag1_autocorr": float(item["lag1_autocorr"]),
            "source_anisotropy": float(item["source_anisotropy"]),
            "trace_cov_anisotropy": float(item["trace_cov_anisotropy"]),
            "source_trace_cov_anisotropy": float(item["source_trace_cov_anisotropy"]),
            "n_microsaccade_events": int(item["n_microsaccade_events"]),
            "fraction_microsaccade_samples": float(item["fraction_microsaccade_samples"]),
            "peak_microsaccade_speed_dps": float(item["peak_microsaccade_speed_dps"]),
            "rendered_n_microsaccade_events": int(item["rendered_n_microsaccade_events"]),
            "rendered_fraction_microsaccade_samples": float(item["rendered_fraction_microsaccade_samples"]),
            "rendered_peak_microsaccade_speed_dps": float(item["rendered_peak_microsaccade_speed_dps"]),
        }
        for j, item in enumerate(trace_bank)
    ]
    _write_csv(out_dir / "trace_bank_metadata.csv", trace_rows)

    strict_rows: list[dict[str, Any]] = []
    keep_indices: list[int] = []
    for idx, row in work.iterrows():
        bank_index = by_source.get(int(row["source_row"]), -1)
        passes = bank_index >= 0 and _actual_trace_passes(trace_bank[bank_index], args)
        strict_rows.append({"image_index": int(idx), "source_row": int(row["source_row"]), "actual_trace_bank_index": int(bank_index), "actual_trace_passes_filters": bool(passes)})
        if passes:
            keep_indices.append(int(idx))
    _write_csv(out_dir / "actual_trace_filter_qc.csv", strict_rows)
    if len(keep_indices) < work.shape[0]:
        _progress(f"strict actual-trace filter keeps {len(keep_indices)}/{work.shape[0]} windows")
        work = work.iloc[keep_indices].reset_index(drop=True)
        work["image_index"] = np.arange(work.shape[0], dtype=int)
    if work.empty:
        raise ValueError("No windows remain after strict actual-trace filtering.")
    if args.window_manifest is None and target_accepted_images > 0 and work.shape[0] > target_accepted_images:
        work = (
            work.sample(n=target_accepted_images, replace=False, random_state=int(args.seed))
            .sort_values(["session", "trial_idx", "source_row"])
            .reset_index(drop=True)
        )
        work["image_index"] = np.arange(work.shape[0], dtype=int)
        _progress(f"sampled {work.shape[0]} accepted paired windows for analysis")

    scorer = None if args.dry_run else CanonicalTwinScorer(device=str(args.device), batch_size=int(args.twin_batch_size))
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    image_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    latent_values: dict[str, list[np.ndarray]] = {}
    raw_responses: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for image_index, row in tqdm(work.iterrows(), total=work.shape[0], desc="local pairing responses"):
        canvas_key = (str(row["session"]), int(row["trial_idx"]))
        if canvas_key not in canvas_cache:
            canvas_cache[canvas_key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
        canvas, ppd, screen_shape = canvas_cache[canvas_key]
        center_px = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
            ppd=ppd,
            screen_shape=screen_shape,
        )
        patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(args.patch_size_px))
        latents = _extract_requested_latents(
            patch,
            latent_crop_px=int(args.latent_crop_px),
            center_crop_px=int(args.center_crop_px),
            local_field_grid=int(args.local_field_grid),
            requested=latent_filter,
        )
        if not latents:
            raise ValueError(f"No requested latent features were available for image {image_index}.")
        if latent_filter:
            missing_latents = sorted(latent_filter.difference(latents))
            if missing_latents:
                raise ValueError(
                    f"Requested latent features are missing for image {image_index}: {missing_latents}. "
                    f"Available: {sorted(latents)}"
                )
        for name, value in latents.items():
            latent_values.setdefault(name, []).append(value)

        actual_bank_index = by_source[int(row["source_row"])]
        actual_item = trace_bank[actual_bank_index]
        image_rows.append(
            {
                "image_index": int(image_index),
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row.get("phase", "")),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "drift_anisotropy": float(row["anisotropy"]),
                "image_edge_axis_deg": float(row.get("image_edge_axis_deg", np.nan)),
                "drift_orientation_deg": float(row.get("drift_orientation_deg", np.nan)),
                "actual_observed_rms_deg": float(actual_item["observed_rms_deg"]),
                "actual_path_length_deg": float(actual_item["path_length_deg"]),
                "actual_lag1_autocorr": float(actual_item["lag1_autocorr"]),
            }
        )

        traces = [_static_trace(int(args.n_timepoints))]
        trace_specs: list[dict[str, Any]] = [
            {
                "family": "static",
                "scale_id": "static",
                "scale": 0.0,
                "sample_index": 0,
                "trace_bank_index": -1,
                "trace_source_row": int(row["source_row"]),
                "pairing_mode": "static",
                "requested_rms_deg": 0.0,
                "effective_rms_deg": 0.0,
                "effective_to_requested_rms": np.nan,
                "rms_clipped_high": False,
                "path_length_deg": 0.0,
            }
        ]

        eligible = _eligible_trace_bank_indices(trace_bank, current_source_row=int(row["source_row"]), **_trace_filter_kwargs(args))
        if bool(args.unpaired_exclude_same_trial):
            eligible = [
                j for j in eligible
                if not (
                    str(trace_bank[j]["session"]) == str(actual_item["session"])
                    and int(trace_bank[j]["trial_idx"]) == int(actual_item["trial_idx"])
                )
            ]
        matched_indices = _matched_unpaired_indices(
            trace_bank,
            actual_item,
            eligible,
            n=int(args.unpaired_samples_per_image),
            rng=rng,
        )
        if "matched_unpaired_empirical" in families and len(matched_indices) != int(args.unpaired_samples_per_image):
            raise ValueError(
                "Matched unpaired trace pool is too small after filtering: "
                f"needed {int(args.unpaired_samples_per_image)}, got {len(matched_indices)} "
                f"for source_row={int(row['source_row'])}."
            )

        raw_cache: dict[str, list[tuple[int, np.ndarray, dict[str, Any]]]] = {}
        raw_cache["actual_paired_empirical"] = [(actual_bank_index, np.asarray(actual_item["trace"], dtype=np.float32), actual_item)]
        raw_cache["rotated_actual_90"] = [(actual_bank_index, _rotated_trace_fixed(actual_item["trace"], 90.0), actual_item)]
        raw_cache["edge_axis"] = [
            (actual_bank_index, _axis_constrained_temporal_trace(actual_item["trace"], float(row["image_edge_axis_deg"])), actual_item)
        ]
        raw_cache["edge_orthogonal"] = [
            (actual_bank_index, _axis_constrained_temporal_trace(actual_item["trace"], float(row["image_edge_axis_deg"]) + 90.0), actual_item)
        ]
        if "rotated_actual_random" in families:
            raw_cache["rotated_actual_random"] = [
                (
                    actual_bank_index,
                    _family_raw_trace(
                        "rotated",
                        actual_item["trace"],
                        float(actual_item["lag1_autocorr"]),
                        rng=rng,
                        max_rms_deg=float(args.max_rms_deg),
                        source_shape=actual_item.get("covariance_shape"),
                        selection_rms=float(actual_item["observed_rms_deg"]),
                        target_path_length=float(actual_item["path_length_deg"]),
                    ),
                    actual_item,
                )
            ]
        if "ou_matched_actual" in families:
            raw_cache["ou_matched_actual"] = [
                (
                    actual_bank_index,
                    _family_raw_trace(
                        "ou",
                        actual_item["trace"],
                        float(actual_item["lag1_autocorr"]),
                        rng=rng,
                        max_rms_deg=float(args.max_rms_deg),
                        source_shape=actual_item.get("covariance_shape"),
                        selection_rms=float(actual_item["observed_rms_deg"]),
                        target_path_length=float(actual_item["path_length_deg"]),
                    ),
                    actual_item,
                )
            ]
        if "brownian_matched_actual" in families:
            raw_cache["brownian_matched_actual"] = [
                (
                    actual_bank_index,
                    _family_raw_trace(
                        "brownian",
                        actual_item["trace"],
                        float(actual_item["lag1_autocorr"]),
                        rng=rng,
                        max_rms_deg=float(args.max_rms_deg),
                    ),
                    actual_item,
                )
            ]
        if "matched_unpaired_empirical" in families:
            raw_cache["matched_unpaired_empirical"] = [
                (j, np.asarray(trace_bank[j]["trace"], dtype=np.float32), trace_bank[j]) for j in matched_indices
            ]

        for scale in scales:
            scale_id = f"rel_{_scale_token(scale)}x"
            target_rms = float(scale) * float(actual_item["observed_rms_deg"])
            for family in families:
                for sample_index, (bank_index, raw, source_item) in enumerate(raw_cache.get(family, [])):
                    trace, meta = _scale_family_raw_trace(raw, target_rms, max_rms_deg=float(args.max_rms_deg))
                    traces.append(trace)
                    requested = float(meta["requested_rms_deg"])
                    trace_specs.append(
                        {
                            "family": family,
                            "scale_id": scale_id,
                            "scale": float(scale),
                            "sample_index": int(sample_index),
                            "pairing_mode": "paired" if family != "matched_unpaired_empirical" else "matched_unpaired",
                            "trace_bank_index": int(bank_index),
                            "trace_source_row": int(source_item["source_row"]),
                            "trace_source_session": str(source_item["session"]),
                            "trace_source_trial_idx": int(source_item["trial_idx"]),
                            "matched_to_actual_source_row": int(row["source_row"]),
                            "same_session_as_actual": bool(str(source_item["session"]) == str(actual_item["session"])),
                            "same_trial_as_actual": bool(
                                str(source_item["session"]) == str(actual_item["session"])
                                and int(source_item["trial_idx"]) == int(actual_item["trial_idx"])
                            ),
                            "source_trace_rms_deg": float(source_item["observed_rms_deg"]),
                            "source_trace_rendered_rms_deg": float(source_item["observed_rms_deg"]),
                            "source_trace_original_rms_deg": float(source_item["source_trace_observed_rms_deg"]),
                            "source_trace_path_length_deg": float(source_item["path_length_deg"]),
                            "source_table_path_length_deg": float(source_item["source_path_length_deg"]),
                            "source_trace_duration_s": float(source_item["duration_s"]),
                            "rendered_trace_n_timepoints": int(args.n_timepoints),
                            "source_trace_lag1": float(source_item["lag1_autocorr"]),
                            "source_anisotropy": float(source_item["source_anisotropy"]),
                            "source_trace_cov_anisotropy": float(source_item["trace_cov_anisotropy"]),
                            "source_original_trace_cov_anisotropy": float(source_item["source_trace_cov_anisotropy"]),
                            "source_n_microsaccade_events": int(source_item["n_microsaccade_events"]),
                            "source_fraction_microsaccade_samples": float(source_item["fraction_microsaccade_samples"]),
                            "source_peak_microsaccade_speed_dps": float(source_item["peak_microsaccade_speed_dps"]),
                            "rendered_n_microsaccade_events": int(source_item["rendered_n_microsaccade_events"]),
                            "rendered_fraction_microsaccade_samples": float(source_item["rendered_fraction_microsaccade_samples"]),
                            "rendered_peak_microsaccade_speed_dps": float(source_item["rendered_peak_microsaccade_speed_dps"]),
                            "requested_rms_deg": requested,
                            "effective_rms_deg": float(meta["effective_rms_deg"]),
                            "effective_to_requested_rms": float(meta["effective_rms_deg"]) / requested if requested > 0 else np.nan,
                            "rms_clipped_high": bool(meta["rms_clipped_high"]),
                            "generated_lag1_autocorr": float(meta["generated_lag1_autocorr"]),
                            "path_length_deg": float(meta["path_length_deg"]),
                            "speed_mean_deg_s": float(meta["speed_mean_deg_s"]),
                            "speed_median_deg_s": float(meta["speed_median_deg_s"]),
                            "speed_p95_deg_s": float(meta["speed_p95_deg_s"]),
                        }
                    )

        if args.dry_run:
            for spec in trace_specs:
                motion_rows.append(
                    {
                        "response_id": -1,
                        "image_index": int(image_index),
                        "source_row": int(row["source_row"]),
                        **spec,
                        "response_frames": 0,
                        "response_units": 0,
                    }
                )
            done = int(image_index) + 1
            if done == 1 or done == work.shape[0] or (int(args.progress_every) > 0 and done % int(args.progress_every) == 0):
                _progress(f"images {done}/{work.shape[0]}; dry trace specs={len(motion_rows)}")
            continue

        responses = scorer.responses(patch, traces, trace_batch_size=int(args.twin_trace_batch_size))
        aligned = [_align_response_to_trace(resp, int(args.n_timepoints)) for resp in responses]
        for spec, resp in zip(trace_specs, aligned, strict=True):
            response_id = len(raw_responses)
            raw_responses.append(resp.astype(np.float32, copy=False))
            records.append(
                {
                    "response_id": response_id,
                    "image_index": int(image_index),
                    "family": str(spec["family"]),
                    "scale_id": str(spec["scale_id"]),
                    "sample_index": int(spec["sample_index"]),
                }
            )
            motion_rows.append(
                {
                    "response_id": response_id,
                    "image_index": int(image_index),
                    "source_row": int(row["source_row"]),
                    **spec,
                    "response_frames": int(resp.shape[0]),
                    "response_units": int(resp.shape[1]),
                }
            )
        done = int(image_index) + 1
        if done == 1 or done == work.shape[0] or (int(args.progress_every) > 0 and done % int(args.progress_every) == 0):
            _progress(f"images {done}/{work.shape[0]}; responses={len(raw_responses)}")

    image_df = pd.DataFrame(image_rows)
    image_df.to_csv(out_dir / "analysis_images.csv", index=False)
    _write_csv(out_dir / "local_pairing_motion_metadata.csv", motion_rows)
    _write_csv(out_dir / "local_pairing_motion_summary.csv", _motion_summary_rows(motion_rows))
    latent_arrays = {name: np.vstack(values).astype(np.float32) for name, values in latent_values.items()}
    np.savez_compressed(out_dir / "latent_feature_arrays.npz", **latent_arrays)

    if args.dry_run:
        _progress("dry run complete; skipped twin responses and summaries")
        return out_dir

    _progress("fitting temporal basis and writing aggregate-compatible response summaries")
    basis = _fit_temporal_basis(raw_responses, int(args.temporal_pc_components))
    dct_basis = _fixed_dct_basis(int(args.n_timepoints), int(args.temporal_pc_components))
    static_by_image = {
        int(rec["image_index"]): raw_responses[int(rec["response_id"])]
        for rec in records
        if rec["family"] == "static"
    }
    response_summaries: dict[int, dict[str, np.ndarray]] = {}
    for rec in records:
        response_id = int(rec["response_id"])
        image_index = int(rec["image_index"])
        summaries = _summarize_response(raw_responses[response_id], static_by_image[image_index], basis)
        _add_temporal_basis_summaries(
            summaries,
            raw_responses[response_id],
            static_by_image[image_index],
            dct_basis,
            prefix="temporal_dct",
        )
        response_summaries[response_id] = summaries
    summary_names = ["temporal_pca", "temporal_delta_pca", "temporal_dct", "temporal_dct_delta", "mean", "delta_mean"]
    summary_arrays: dict[str, np.ndarray] = {}
    feature_by_condition_by_summary: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    for summary in summary_names:
        by_condition = _stack_condition_features_with_samples(
            records,
            response_summaries,
            summary,
            sample_families={"matched_unpaired_empirical"},
        )
        feature_by_condition_by_summary[summary] = by_condition
        for (family, scale_id), arr in by_condition.items():
            summary_arrays[f"{summary}__{family}__{scale_id}"] = arr
    np.savez_compressed(out_dir / "response_summary_arrays.npz", temporal_basis=basis, temporal_dct_basis=dct_basis, **summary_arrays)

    sessions = image_df["session"].to_numpy()
    decode_groups = image_df["image_index"].to_numpy(dtype=int) if str(args.decode_group_mode) == "image" else sessions
    if str(args.decode_group_mode) == "session" and np.unique(sessions).size < 2:
        raise ValueError(
            "--decode-group-mode session requires at least two sessions after filtering; "
            "otherwise the shared decoder falls back to row-wise folds."
        )
    all_decode_rows: list[dict[str, Any]] = []
    all_per_image: dict[tuple[str, str, str, str, int], np.ndarray] = {}
    for summary in summary_names:
        by_condition = {
            key: value
            for key, value in feature_by_condition_by_summary[summary].items()
            if "_sample" not in key[0]
        }
        rows, per_image = _decode_rows(by_condition, latent_arrays, decode_groups, args)
        for row in rows:
            row["response_summary"] = summary
            all_decode_rows.append(row)
        for key, values in per_image.items():
            all_per_image[(summary, *key)] = values
        _progress(f"decoded summary={summary}; jobs={len(rows)}")
    _write_csv(out_dir / "decode_summary.csv", all_decode_rows)

    contrast_rows: list[dict[str, Any]] = []
    for summary in summary_names:
        rows = [row for row in all_decode_rows if row["response_summary"] == summary]
        per_image = {key[1:]: value for key, value in all_per_image.items() if key[0] == summary}
        for crow in _contrast_rows(rows, per_image, sessions, args):
            crow["response_summary"] = summary
            contrast_rows.append(crow)
    _write_csv(out_dir / "decode_contrasts.csv", contrast_rows)

    inc_decode_rows, inc_gain_rows, inc_contrast_rows = _run_local_incremental_decoding(
        feature_by_condition_by_summary,
        latent_arrays,
        sessions,
        decode_groups,
        summary_names,
        families,
        scales,
        args,
    )
    _write_csv(out_dir / "incremental_decode_summary.csv", inc_decode_rows)
    _write_csv(out_dir / "incremental_gain_vs_static.csv", inc_gain_rows)
    _write_csv(out_dir / "incremental_gain_contrasts.csv", inc_contrast_rows)

    condition_keys = sorted({("static", "static")} | {(str(rec["family"]), str(rec["scale_id"])) for rec in records if rec["family"] != "static"})
    cov_rows = _covariance_rows(records, response_summaries, summary_names, condition_keys, overlap_dim=5)
    _write_csv(out_dir / "covariance_summary.csv", cov_rows)

    report = [
        "# BackImage Local I_z Pairing Revisit",
        "",
        f"- Images: {image_df.shape[0]}",
        f"- Unpaired samples per image: {args.unpaired_samples_per_image}",
        f"- Families: {', '.join(families)}",
        f"- Scales: {', '.join(str(v) for v in scales)}",
        f"- Temporal basis components: {basis.shape[1]}",
        f"- Latents: {', '.join(latent_arrays)}",
        "",
        "Primary files:",
        "- `local_pairing_motion_metadata.csv`",
        "- `local_pairing_motion_summary.csv`",
        "- `response_summary_arrays.npz`",
        "- `decode_summary.csv`",
        "- `decode_contrasts.csv`",
        "- `incremental_gain_vs_static.csv`",
        "- `incremental_gain_contrasts.csv`",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _progress(f"complete; wrote summaries to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
