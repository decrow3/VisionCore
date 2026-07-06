"""Pose-unaware static-plus-motion decoding for aggregate BackImage FEM pilots.

This posthoc consumes an aggregate FEM run that saved
``response_sample_summary_arrays.npz``. Unlike the standard Figure 4B posthoc,
it does not average trace samples before decoding. Each motion response sample is
decoded without a trajectory label, while cross-validation remains grouped by
image. That makes trajectory-induced response variation act as nuisance
variability in the -MSE score.
"""
from __future__ import annotations

import argparse
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

try:
    from .run_backimage_latent_information_screen import _split_outer
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _split_outer

try:
    from .summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _bootstrap_indices_by_group,
        _ci_from_bootstrap,
        _decode,
        _decode_groups_from_images,
        _filter_latents,
        _information_fold_rows,
        _information_point_estimates,
        _load_npz,
        _parse_float_list,
        _parse_int_list,
        _parse_list,
        _response_key,
        _session_bootstrap_delta,
        _summarize_information_rows,
        _validate_information_intervals,
        _write_csv,
        _write_json,
    )
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _bootstrap_indices_by_group,
        _ci_from_bootstrap,
        _decode,
        _decode_groups_from_images,
        _filter_latents,
        _information_fold_rows,
        _information_point_estimates,
        _load_npz,
        _parse_float_list,
        _parse_int_list,
        _parse_list,
        _response_key,
        _session_bootstrap_delta,
        _summarize_information_rows,
        _validate_information_intervals,
        _write_csv,
        _write_json,
    )


DEFAULT_RUN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_pose_unaware_pilot"
)


def _flatten_samples(samples: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected sample array with shape images x samples x features, got {arr.shape}")
    n_images, n_samples, n_features = arr.shape
    return arr.reshape(n_images * n_samples, n_features), int(n_samples)


def _repeat_rows(arr: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    return np.repeat(values[:, None, :], int(n_samples), axis=1).reshape(values.shape[0] * int(n_samples), values.shape[1])


def _repeat_vector(values: np.ndarray, n_samples: int) -> np.ndarray:
    return np.repeat(np.asarray(values), int(n_samples), axis=0)


def _per_image_mean(values: np.ndarray, n_images: int, n_samples: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(int(n_images), int(n_samples)).mean(axis=1)


def _standardize_params(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0, keepdims=True)
    sd = np.nanstd(train, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return mean, sd


def _standardize_with(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, sd = _standardize_params(train)
    return (train - mean) / sd, (values - mean) / sd, mean


def _scale_order(scale_id: str) -> tuple[int, str]:
    preferred = {
        "rel_0p125x": 0,
        "rel_0p25x": 1,
        "rel_0p5x": 2,
        "rel_0p75x": 3,
        "rel_1x": 4,
        "rel_1p5x": 5,
        "rel_2x": 6,
        "rel_3x": 7,
    }
    return (preferred.get(str(scale_id), 100), str(scale_id))


def _fold_residual_row(fold: int, test_idx: np.ndarray, residual: np.ndarray, *, alpha: float) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64)
    return {
        "fold": int(fold),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "residual": residual,
        "target_dim": int(residual.shape[1]),
        "n_test": int(residual.shape[0]),
        "alpha": float(alpha),
    }


def _subset_response_dict(responses: dict[str, np.ndarray], idx: np.ndarray, *, n_rows: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in responses.items():
        arr = np.asarray(value)
        out[key] = arr[idx] if arr.ndim > 0 and arr.shape[0] == int(n_rows) else arr
    return out


def _summarize_info_with_decode_bootstrap(
    info_rows: list[dict[str, Any]],
    boot_points: list[dict[str, Any]],
) -> dict[str, Any]:
    point = _information_point_estimates(info_rows)

    def ci(column: str) -> tuple[float, float, int]:
        values = np.asarray([float(row.get(column, np.nan)) for row in boot_points], dtype=np.float64)
        ok = np.isfinite(values)
        lo, hi = _ci_from_bootstrap(values[ok], float(point[column]))
        return lo, hi, int(np.sum(ok))

    diag_lo, diag_hi, n_diag = ci("incremental_gain_info_diag_bits")
    diag_per_dim_lo, diag_per_dim_hi, _ = ci("incremental_gain_info_diag_bits_per_dim")
    full_lo, full_hi, _ = ci("incremental_gain_info_full_bits")
    full_per_dim_lo, full_per_dim_hi, _ = ci("incremental_gain_info_full_bits_per_dim")
    return {
        "incremental_gain_info_diag_bits": point["incremental_gain_info_diag_bits"],
        "info_diag_ci95_low": diag_lo,
        "info_diag_ci95_high": diag_hi,
        "incremental_gain_info_diag_bits_per_dim": point["incremental_gain_info_diag_bits_per_dim"],
        "info_diag_per_dim_ci95_low": diag_per_dim_lo,
        "info_diag_per_dim_ci95_high": diag_per_dim_hi,
        "incremental_gain_info_full_bits": point["incremental_gain_info_full_bits"],
        "info_full_ci95_low": full_lo,
        "info_full_ci95_high": full_hi,
        "incremental_gain_info_full_bits_per_dim": point["incremental_gain_info_full_bits_per_dim"],
        "info_full_per_dim_ci95_low": full_per_dim_lo,
        "info_full_per_dim_ci95_high": full_per_dim_hi,
        "information_ci_method": "decode_pipeline_group_bootstrap_point_centered",
        "ridge_alpha_matched_all_folds": bool(point["ridge_alpha_matched_all_folds"]),
        "n_information_folds": int(point["n_information_folds"]),
        "n_information_bootstrap_success": n_diag,
    }


def _attach_information_summaries(
    rows: list[dict[str, Any]],
    info_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    information_ci_mode: str,
    boot_points_by_key: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> None:
    for row in rows:
        key = (str(row.get("observer", "")), str(row.get("scale_id", "")))
        info_rows = info_rows_by_key.get(key, [])
        if not info_rows:
            continue
        if str(information_ci_mode) == "decode_bootstrap":
            summary = _summarize_info_with_decode_bootstrap(info_rows, (boot_points_by_key or {}).get(key, []))
        else:
            summary = _summarize_information_rows(info_rows, rng=rng, n_bootstrap=int(n_bootstrap))
        row.update(summary)


def _compute_train_mean_test_samples_proxy_once(
    *,
    X_static: np.ndarray,
    Z: np.ndarray,
    sessions: np.ndarray,
    groups: np.ndarray,
    averaged_responses: dict[str, np.ndarray],
    sample_responses: dict[str, np.ndarray],
    summary: str,
    static_summary: str,
    family: str,
    scale_ids: list[str],
    latent_name: str,
    k: int,
    alpha: float,
    outer_folds: int,
    seed: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    information_variance_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Train on the known-eye mean response, test on hidden trajectory samples."""

    X_static = np.asarray(X_static, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    sessions = np.asarray(sessions)
    groups = np.asarray(groups)
    n_images = int(X_static.shape[0])
    k_eff = int(min(int(k), Z.shape[1], max(1, n_images - 2)))
    static_score = np.full(n_images, np.nan, dtype=np.float64)
    known_scores = {scale_id: np.full(n_images, np.nan, dtype=np.float64) for scale_id in scale_ids}
    hidden_scores: dict[str, np.ndarray] = {}
    n_samples_by_scale: dict[str, int] = {}
    static_folds: list[dict[str, Any]] = []
    known_folds: dict[str, list[dict[str, Any]]] = {scale_id: [] for scale_id in scale_ids}
    hidden_folds: dict[str, list[dict[str, Any]]] = {scale_id: [] for scale_id in scale_ids}
    static_sample_folds: dict[str, list[dict[str, Any]]] = {scale_id: [] for scale_id in scale_ids}
    known_sample_folds: dict[str, list[dict[str, Any]]] = {scale_id: [] for scale_id in scale_ids}
    for scale_id in scale_ids:
        samples = np.asarray(sample_responses[_response_key(summary, family, scale_id)], dtype=np.float64)
        if samples.ndim != 3:
            raise ValueError(f"Expected sampled motion array for {summary}/{family}/{scale_id}, got {samples.shape}")
        hidden_scores[scale_id] = np.full(samples.shape[:2], np.nan, dtype=np.float64)
        n_samples_by_scale[scale_id] = int(samples.shape[1])

    for fold, (train_idx, test_idx) in enumerate(_split_outer(groups, int(outer_folds), int(seed))):
        X_static_train, X_static_test, _ = _standardize_with(X_static[train_idx], X_static[test_idx])
        Z_train_raw, Z_test_raw, _ = _standardize_with(Z[train_idx], Z[test_idx])
        pca = PCA(n_components=k_eff, svd_solver="full")
        y_train = pca.fit_transform(Z_train_raw)
        y_test = pca.transform(Z_test_raw)

        static_model = Ridge(alpha=float(alpha), fit_intercept=True)
        static_model.fit(X_static_train, y_train)
        static_pred = static_model.predict(X_static_test)
        static_residual = np.asarray(y_test - static_pred, dtype=np.float64)
        static_score[test_idx] = -np.mean(static_residual**2, axis=1)
        static_folds.append(_fold_residual_row(fold, test_idx, static_residual, alpha=float(alpha)))

        for scale_id in scale_ids:
            X_mean = np.asarray(averaged_responses[_response_key(summary, family, scale_id)], dtype=np.float64)
            X_samples = np.asarray(sample_responses[_response_key(summary, family, scale_id)], dtype=np.float64)
            X_aug_train = np.concatenate([X_static[train_idx], X_mean[train_idx]], axis=1)
            X_aug_test_mean = np.concatenate([X_static[test_idx], X_mean[test_idx]], axis=1)
            aug_mean, aug_sd = _standardize_params(X_aug_train)
            X_aug_train_z = (X_aug_train - aug_mean) / aug_sd
            X_aug_test_mean_z = (X_aug_test_mean - aug_mean) / aug_sd

            motion_model = Ridge(alpha=float(alpha), fit_intercept=True)
            motion_model.fit(X_aug_train_z, y_train)
            known_pred = motion_model.predict(X_aug_test_mean_z)
            known_residual = np.asarray(y_test - known_pred, dtype=np.float64)
            known_scores[scale_id][test_idx] = -np.mean(known_residual**2, axis=1)
            known_folds[scale_id].append(_fold_residual_row(fold, test_idx, known_residual, alpha=float(alpha)))

            n_test = int(test_idx.size)
            n_samples = int(X_samples.shape[1])
            X_static_rep = np.repeat(X_static[test_idx, None, :], n_samples, axis=1)
            X_aug_samples = np.concatenate([X_static_rep, X_samples[test_idx]], axis=2).reshape(n_test * n_samples, -1)
            X_aug_samples_z = (X_aug_samples - aug_mean) / aug_sd
            sample_pred = motion_model.predict(X_aug_samples_z).reshape(n_test, n_samples, -1)
            hidden_residual_3d = np.asarray(y_test[:, None, :] - sample_pred, dtype=np.float64)
            hidden_scores[scale_id][test_idx, :] = -np.mean(hidden_residual_3d**2, axis=2)
            flat_test_idx = (
                np.repeat(np.asarray(test_idx, dtype=np.int64), n_samples) * int(n_samples)
                + np.tile(np.arange(n_samples, dtype=np.int64), n_test)
            )
            hidden_residual = hidden_residual_3d.reshape(n_test * n_samples, -1)
            static_sample_residual = np.repeat(static_residual[:, None, :], n_samples, axis=1).reshape(
                n_test * n_samples,
                -1,
            )
            known_sample_residual = np.repeat(known_residual[:, None, :], n_samples, axis=1).reshape(
                n_test * n_samples,
                -1,
            )
            hidden_folds[scale_id].append(_fold_residual_row(fold, flat_test_idx, hidden_residual, alpha=float(alpha)))
            static_sample_folds[scale_id].append(
                _fold_residual_row(fold, flat_test_idx, static_sample_residual, alpha=float(alpha))
            )
            known_sample_folds[scale_id].append(
                _fold_residual_row(fold, flat_test_idx, known_sample_residual, alpha=float(alpha))
            )

    rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = [
        {
            "observer": "static_baseline_train_mean_test_mean_proxy",
            "motion_summary": summary,
            "static_summary": static_summary,
            "family": "static",
            "scale_id": "static",
            "latent": latent_name,
            "k": int(k),
            "mean_neg_mse": float(np.nanmean(static_score)),
            "n_images": n_images,
        }
    ]
    for scale_id in scale_ids:
        hidden_per_image = np.nanmean(hidden_scores[scale_id], axis=1)
        known_boot = _session_bootstrap_delta(
            known_scores[scale_id],
            static_score,
            sessions,
            rng=rng,
            n_bootstrap=int(n_bootstrap),
        )
        hidden_boot = _session_bootstrap_delta(
            hidden_per_image,
            static_score,
            sessions,
            rng=rng,
            n_bootstrap=int(n_bootstrap),
        )
        penalty_boot = _session_bootstrap_delta(
            hidden_per_image,
            known_scores[scale_id],
            sessions,
            rng=rng,
            n_bootstrap=int(n_bootstrap),
        )
        common = {
            "motion_summary": summary,
            "static_summary": static_summary,
            "family": family,
            "scale_id": scale_id,
            "latent": latent_name,
            "k": int(k),
            "n_images": known_boot["n"],
            "n_sessions": known_boot["n_sessions"],
            "trace_samples_per_image": int(n_samples_by_scale[scale_id]),
            "sample_rows": int(n_images * n_samples_by_scale[scale_id]),
        }
        rows.extend(
            [
                {
                    "observer": "known_eye_train_mean_test_mean_proxy",
                    **common,
                    "incremental_gain_neg_mse": known_boot["mean"],
                    "ci95_low": known_boot["ci_low"],
                    "ci95_high": known_boot["ci_high"],
                },
                {
                    "observer": "pose_unaware_train_mean_test_hidden_samples",
                    **common,
                    "incremental_gain_neg_mse": hidden_boot["mean"],
                    "ci95_low": hidden_boot["ci_low"],
                    "ci95_high": hidden_boot["ci_high"],
                },
                {
                    "observer": "hidden_sample_minus_known_eye_penalty",
                    **common,
                    "incremental_gain_neg_mse": penalty_boot["mean"],
                    "ci95_low": penalty_boot["ci_low"],
                    "ci95_high": penalty_boot["ci_high"],
                },
            ]
        )
        decode_rows.extend(
            [
                {
                    "observer": "known_eye_train_mean_test_mean_proxy",
                    "motion_summary": summary,
                    "static_summary": static_summary,
                    "family": family,
                    "scale_id": scale_id,
                    "latent": latent_name,
                    "k": int(k),
                    "mean_neg_mse": float(np.nanmean(known_scores[scale_id])),
                    "n_images": n_images,
                },
                {
                    "observer": "pose_unaware_train_mean_test_hidden_samples",
                    "motion_summary": summary,
                    "static_summary": static_summary,
                    "family": family,
                    "scale_id": scale_id,
                    "latent": latent_name,
                    "k": int(k),
                    "mean_neg_mse": float(np.nanmean(hidden_per_image)),
                    "n_images": n_images,
                },
            ]
        )
    info_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scale_id in scale_ids:
        common = {
            "motion_summary": summary,
            "family": family,
            "scale_id": scale_id,
            "latent": latent_name,
            "k": int(k),
            "variance_floor": float(information_variance_floor),
        }
        info_rows_by_key[("known_eye_train_mean_test_mean_proxy", scale_id)] = _information_fold_rows(
            condition_result={"fold_residuals": known_folds[scale_id]},
            baseline_result={"fold_residuals": static_folds},
            **common,
        )
        info_rows_by_key[("pose_unaware_train_mean_test_hidden_samples", scale_id)] = _information_fold_rows(
            condition_result={"fold_residuals": hidden_folds[scale_id]},
            baseline_result={"fold_residuals": static_sample_folds[scale_id]},
            **common,
        )
        info_rows_by_key[("hidden_sample_minus_known_eye_penalty", scale_id)] = _information_fold_rows(
            condition_result={"fold_residuals": hidden_folds[scale_id]},
            baseline_result={"fold_residuals": known_sample_folds[scale_id]},
            **common,
        )
    return rows, decode_rows, info_rows_by_key


def _compute_train_mean_test_samples_proxy(
    *,
    X_static: np.ndarray,
    Z: np.ndarray,
    sessions: np.ndarray,
    groups: np.ndarray,
    averaged_responses: dict[str, np.ndarray],
    sample_responses: dict[str, np.ndarray],
    summary: str,
    static_summary: str,
    family: str,
    scale_ids: list[str],
    latent_name: str,
    k: int,
    alpha: float,
    outer_folds: int,
    seed: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    information_variance_floor: float,
    information_ci_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, decode_rows, info_rows_by_key = _compute_train_mean_test_samples_proxy_once(
        X_static=X_static,
        Z=Z,
        sessions=sessions,
        groups=groups,
        averaged_responses=averaged_responses,
        sample_responses=sample_responses,
        summary=summary,
        static_summary=static_summary,
        family=family,
        scale_ids=scale_ids,
        latent_name=latent_name,
        k=int(k),
        alpha=float(alpha),
        outer_folds=int(outer_folds),
        seed=int(seed),
        n_bootstrap=int(n_bootstrap),
        rng=rng,
        information_variance_floor=float(information_variance_floor),
    )
    boot_points_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in info_rows_by_key}
    if str(information_ci_mode) == "decode_bootstrap":
        groups = np.asarray(groups)
        for boot_idx in range(int(n_bootstrap)):
            idx = _bootstrap_indices_by_group(groups, rng=rng)
            if np.unique(groups[idx]).size < 2:
                continue
            _, _, boot_info_rows_by_key = _compute_train_mean_test_samples_proxy_once(
                X_static=np.asarray(X_static)[idx],
                Z=np.asarray(Z)[idx],
                sessions=np.asarray(sessions)[idx],
                groups=groups[idx],
                averaged_responses=_subset_response_dict(averaged_responses, idx, n_rows=groups.size),
                sample_responses=_subset_response_dict(sample_responses, idx, n_rows=groups.size),
                summary=summary,
                static_summary=static_summary,
                family=family,
                scale_ids=scale_ids,
                latent_name=latent_name,
                k=int(k),
                alpha=float(alpha),
                outer_folds=int(outer_folds),
                seed=int(seed) + 30000 + int(boot_idx),
                n_bootstrap=0,
                rng=rng,
                information_variance_floor=float(information_variance_floor),
            )
            for key, boot_info_rows in boot_info_rows_by_key.items():
                boot_points_by_key.setdefault(key, []).append(_information_point_estimates(boot_info_rows))
    _attach_information_summaries(
        rows,
        info_rows_by_key,
        rng=rng,
        n_bootstrap=int(n_bootstrap),
        information_ci_mode=str(information_ci_mode),
        boot_points_by_key=boot_points_by_key,
    )
    return rows, decode_rows


def _write_proxy_plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    df = pd.DataFrame(rows)
    if df.empty:
        return
    order = sorted(df["scale_id"].drop_duplicates().astype(str), key=_scale_order)
    labels = [s.removeprefix("rel_").replace("p", ".").replace("x", "x") for s in order]
    styles = {
        "known_eye_train_mean_test_mean_proxy": ("known-eye mean trace", "#1f77b4", "-"),
        "pose_unaware_train_mean_test_hidden_samples": ("pose-unaware test samples", "#d62728", "--"),
        "hidden_sample_minus_known_eye_penalty": ("hidden-sample penalty", "#666666", "-"),
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=160)
    for observer, (label, color, linestyle) in styles.items():
        sub = df[df["observer"] == observer].set_index("scale_id")
        if sub.empty:
            continue
        sub = sub.loc[[scale_id for scale_id in order if scale_id in sub.index]]
        x = np.arange(sub.shape[0])
        y = sub["incremental_gain_neg_mse"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=2.0, color=color, linestyle=linestyle, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.65)
    ax.set_xticks(np.arange(len(order)), labels)
    ax.set_xlabel("empirical drift RMS scale")
    ax.set_ylabel("incremental gain over static mean (-MSE)")
    ax.set_title("Pose-unaware hidden-trajectory proxy")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "pose_unaware_train_mean_test_samples_proxy.png")
    fig.savefig(out_dir / "pose_unaware_train_mean_test_samples_proxy.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summaries", default="delta_mean")
    parser.add_argument("--families", default="empirical")
    parser.add_argument("--scale-ids", default="all")
    parser.add_argument("--latent-names", default="pyramid_local_field")
    parser.add_argument("--pca-k-list", default="16")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--ridge-alpha-mode", choices=("fixed", "nested_per_candidate"), default="fixed")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--decode-group-mode", choices=("image", "source_trial", "session"), default="image")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument(
        "--information-variance-floor",
        type=float,
        default=1e-12,
        help="Variance floor for residual covariance information increments.",
    )
    parser.add_argument(
        "--information-ci-mode",
        choices=("fold", "decode_bootstrap"),
        default="fold",
        help=(
            "CI mode for pose-unaware information columns. `fold` bootstraps outer-fold scalar estimates; "
            "`decode_bootstrap` resamples decode groups, refits the proxy decoder, and recenters CIs on the point."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "pose_unaware_staticmean_plus_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    images = pd.read_csv(run_dir / "analysis_images.csv")
    sessions = images["session"].to_numpy()
    image_groups = _decode_groups_from_images(images, str(args.decode_group_mode))
    latents = _filter_latents(_load_npz(run_dir / "latent_feature_arrays.npz"), _parse_list(args.latent_names))
    averaged_responses = _load_npz(run_dir / "response_summary_arrays.npz")
    sample_responses = _load_npz(run_dir / "response_sample_summary_arrays.npz")
    summaries = _parse_list(args.summaries)
    families = _parse_list(args.families)
    scale_ids = _parse_list(args.scale_ids)
    if not scale_ids or "all" in scale_ids:
        scale_ids = _available_scale_ids(sample_responses, families, summaries)
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    alpha_mode = str(args.ridge_alpha_mode)
    pca_k_list = _parse_int_list(args.pca_k_list)

    gain_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    static_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    static_per_image_cache: dict[tuple[str, str, int, int], np.ndarray] = {}

    for summary in summaries:
        static_summary = STATIC_SUMMARY_FOR_MOTION.get(summary)
        if static_summary is None:
            raise ValueError(f"No static summary mapping is defined for {summary!r}")
        static_key = _response_key(static_summary, "static", "static")
        if static_key not in averaged_responses:
            raise ValueError(f"Missing static response array {static_key!r}")
        X_static = averaged_responses[static_key]
        n_images = int(X_static.shape[0])
        for latent_name, Z in latents.items():
            for k in pca_k_list:
                for scale_id in scale_ids:
                    for family in families:
                        motion_key = _response_key(summary, family, scale_id)
                        if motion_key not in sample_responses:
                            continue
                        X_motion, n_samples = _flatten_samples(sample_responses[motion_key])
                        X_static_rep = _repeat_rows(X_static, n_samples)
                        Z_rep = _repeat_rows(Z, n_samples)
                        groups_rep = _repeat_vector(image_groups, n_samples)

                        static_cache_key = (static_summary, latent_name, int(k), int(n_samples))
                        if static_cache_key not in static_cache:
                            static_result = _decode(
                                X_static_rep,
                                Z_rep,
                                groups_rep,
                                k=int(k),
                                alphas=alphas,
                                alpha_mode=alpha_mode,
                                fixed_alpha=fixed_alpha,
                                outer_folds=int(args.outer_folds),
                                inner_folds=int(args.inner_folds),
                                seed=int(args.seed),
                            )
                            static_cache[static_cache_key] = static_result
                            static_per_image_cache[static_cache_key] = _per_image_mean(
                                static_result["per_window_score"],
                                n_images,
                                n_samples,
                            )
                        else:
                            static_result = static_cache[static_cache_key]

                        X_aug = np.concatenate([X_static_rep, X_motion], axis=1)
                        aug_result = _decode(
                            X_aug,
                            Z_rep,
                            groups_rep,
                            k=int(k),
                            alphas=alphas,
                            alpha_mode=alpha_mode,
                            fixed_alpha=fixed_alpha,
                            outer_folds=int(args.outer_folds),
                            inner_folds=int(args.inner_folds),
                            seed=int(args.seed),
                        )
                        aug_per_image = _per_image_mean(aug_result["per_window_score"], n_images, n_samples)
                        static_per_image = static_per_image_cache[static_cache_key]
                        boot = _session_bootstrap_delta(
                            aug_per_image,
                            static_per_image,
                            sessions,
                            rng=rng,
                            n_bootstrap=int(args.n_bootstrap),
                        )
                        gain_rows.append(
                            {
                                "observer": "pose_unaware_samplewise_hidden_trajectory",
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
                                "trace_samples_per_image": int(n_samples),
                                "sample_rows": int(n_images * n_samples),
                                "decode_group_mode": str(args.decode_group_mode),
                            }
                        )
                        decode_rows.append(
                            {
                                "observer": "pose_unaware_samplewise_hidden_trajectory",
                                "motion_summary": summary,
                                "static_summary": static_summary,
                                "model": "static_only_repeated",
                                "family": "static",
                                "scale_id": "static",
                                "latent": latent_name,
                                "k": int(k),
                                "mean_neg_mse": float(static_result["mean_neg_mse"]),
                                "r2": float(static_result["r2"]),
                                "chosen_alpha_median": float(static_result["chosen_alpha_median"]),
                                "target_dim": int(static_result["target_dim"]),
                                "n_images": int(n_images),
                                "sample_rows": int(n_images * n_samples),
                                "feature_dim": int(X_static_rep.shape[1]),
                            }
                        )
                        decode_rows.append(
                            {
                                "observer": "pose_unaware_samplewise_hidden_trajectory",
                                "motion_summary": summary,
                                "static_summary": static_summary,
                                "model": "static_plus_motion_sample",
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "mean_neg_mse": float(aug_result["mean_neg_mse"]),
                                "r2": float(aug_result["r2"]),
                                "chosen_alpha_median": float(aug_result["chosen_alpha_median"]),
                                "target_dim": int(aug_result["target_dim"]),
                                "n_images": int(n_images),
                                "sample_rows": int(n_images * n_samples),
                                "feature_dim": int(X_aug.shape[1]),
                            }
                        )

    _write_csv(out_dir / "pose_unaware_incremental_gain_vs_static.csv", gain_rows)
    _write_csv(out_dir / "pose_unaware_incremental_decode_summary.csv", decode_rows)

    proxy_rows: list[dict[str, Any]] = []
    proxy_decode_rows: list[dict[str, Any]] = []
    for summary in summaries:
        static_summary = STATIC_SUMMARY_FOR_MOTION.get(summary)
        if static_summary is None:
            continue
        static_key = _response_key(static_summary, "static", "static")
        if static_key not in averaged_responses:
            continue
        for latent_name, Z in latents.items():
            for k in pca_k_list:
                for family in families:
                    available_scales = [
                        scale_id
                        for scale_id in scale_ids
                        if _response_key(summary, family, scale_id) in averaged_responses
                        and _response_key(summary, family, scale_id) in sample_responses
                    ]
                    if not available_scales:
                        continue
                    rows, decodes = _compute_train_mean_test_samples_proxy(
                        X_static=averaged_responses[static_key],
                        Z=Z,
                        sessions=sessions,
                        groups=image_groups,
                        averaged_responses=averaged_responses,
                        sample_responses=sample_responses,
                        summary=summary,
                        static_summary=static_summary,
                        family=family,
                        scale_ids=sorted(available_scales, key=_scale_order),
                        latent_name=latent_name,
                        k=int(k),
                        alpha=fixed_alpha,
                        outer_folds=int(args.outer_folds),
                        seed=int(args.seed),
                        n_bootstrap=int(args.n_bootstrap),
                        rng=rng,
                        information_variance_floor=float(args.information_variance_floor),
                        information_ci_mode=str(args.information_ci_mode),
                    )
                    proxy_rows.extend(rows)
                    proxy_decode_rows.extend(decodes)
    _validate_information_intervals(proxy_rows, value_col="incremental_gain_info_diag_bits")
    _write_csv(out_dir / "pose_unaware_train_mean_test_samples_proxy.csv", proxy_rows)
    _write_csv(out_dir / "pose_unaware_train_mean_test_samples_decode_scores.csv", proxy_decode_rows)
    _write_proxy_plot(proxy_rows, out_dir)

    _write_json(
        out_dir / "run_metadata.json",
        {
            "source_run_dir": run_dir,
            "observer": "pose_unaware_samplewise_hidden_trajectory",
            "definition": "Decode static_plus_one_motion_summary_sample without trajectory labels; CV grouped by image/session.",
            "proxy_definition": "Train static-plus-motion decoder on known-eye image-mean motion summaries, then evaluate held-out images on hidden trajectory samples.",
            "summaries": summaries,
            "families": families,
            "scale_ids": scale_ids,
            "latent_names": list(latents),
            "pca_k_list": pca_k_list,
            "ridge_alpha_mode": alpha_mode,
            "fixed_ridge_alpha": fixed_alpha if alpha_mode == "fixed" else None,
            "decode_group_mode": str(args.decode_group_mode),
            "outer_folds": int(args.outer_folds),
            "n_bootstrap": int(args.n_bootstrap),
            "information_axis": {
                "headline": "diagonal_gaussian_variational_bound_increment_bits",
                "diag_formula": "0.5 * sum_j log(var_static_or_known_j / var_condition_j) / log(2)",
                "ci_mode": str(args.information_ci_mode),
                "ci_method": (
                    "decode_pipeline_group_bootstrap_point_centered"
                    if str(args.information_ci_mode) == "decode_bootstrap"
                    else "outer_fold_weighted_bootstrap"
                ),
                "variance_floor": float(args.information_variance_floor),
            },
            "seed": int(args.seed),
        },
    )
    report = [
        "# Pose-Unaware Static Plus Motion",
        "",
        f"Source run: `{run_dir}`",
        "",
        "Question:",
        "",
        "`z ~ R_static` versus `z ~ R_static + R_motion_sample`, with the trajectory label hidden.",
        "",
        "Primary files:",
        "- `pose_unaware_incremental_gain_vs_static.csv`",
        "- `pose_unaware_incremental_decode_summary.csv`",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote pose-unaware summaries to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
