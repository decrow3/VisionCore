"""Incremental static-plus-motion decoding for BackImage aggregate FEM runs.

This cache-only posthoc asks whether motion summaries add image-feature
decoding signal beyond the static mean response summary:

    z ~ R_static_mean
    z ~ R_static_mean + R_motion

It also compares the incremental gain from empirical motion against matched
OU/Brownian/rotated controls.
"""
from __future__ import annotations

import argparse
import csv
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
    from sklearn.covariance import LedoitWolf
except Exception:  # pragma: no cover - optional robustness metric
    LedoitWolf = None

try:
    from .run_backimage_latent_information_screen import _cross_validated_decode
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _cross_validated_decode


DEFAULT_RUN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_n128_k4_rel025-1_gabor_pyramid"
)


STATIC_SUMMARY_FOR_MOTION = {
    # Static temporal summaries are near zero for a static trace; use the
    # static mean response as the real static image baseline for every motion
    # summary.
    "temporal_pca": "mean",
    "temporal_delta_pca": "mean",
    "temporal_dct": "mean",
    "temporal_dct_delta": "mean",
    "mean": "mean",
    "delta_mean": "mean",
}

LN2 = float(np.log(2.0))


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in _parse_list(text)]


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in _parse_list(text)]


def _parse_contrast_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in _parse_list(text):
        if ":" in part:
            lhs, rhs = part.split(":", 1)
        elif ">" in part:
            lhs, rhs = part.split(">", 1)
        elif "-" in part:
            lhs, rhs = part.split("-", 1)
        else:
            raise ValueError(
                "Contrast pairs must use lhs:rhs, lhs>rhs, or lhs-rhs syntax; "
                f"got {part!r}"
            )
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or not rhs:
            raise ValueError(f"Invalid empty contrast pair entry: {part!r}")
        pairs.append((lhs, rhs))
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: np.asarray(loaded[key]) for key in loaded.files}


def _filter_latents(latents: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    if not names or "all" in names:
        return {key: value for key, value in latents.items()}
    missing = sorted(set(names).difference(latents))
    if missing:
        raise ValueError(f"Requested latent arrays are missing: {missing}")
    return {name: latents[name] for name in names}


def _response_key(summary: str, family: str, scale_id: str) -> str:
    return f"{summary}__{family}__{scale_id}"


def _available_scale_ids(responses: dict[str, np.ndarray], families: list[str], summaries: list[str]) -> list[str]:
    scales: set[str] = set()
    for key in responses:
        parts = key.split("__")
        if len(parts) != 3:
            continue
        summary, family, scale_id = parts
        if summary in summaries and family in families and scale_id != "static":
            scales.add(scale_id)
    return sorted(scales, key=lambda s: (len(s), s))


def _session_bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    sessions: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    sessions = np.asarray(sessions)
    ok = np.isfinite(delta)
    delta = delta[ok]
    sessions = sessions[ok]
    if delta.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_sessions": 0}
    session_values = pd.DataFrame({"session": sessions, "delta": delta}).groupby("session")["delta"].mean().to_numpy(dtype=np.float64)
    mean = float(np.nanmean(session_values))
    if int(n_bootstrap) <= 0 or session_values.size <= 1:
        lo = hi = mean
    else:
        boot = np.empty(int(n_bootstrap), dtype=np.float64)
        for i in range(int(n_bootstrap)):
            boot[i] = float(np.nanmean(rng.choice(session_values, size=session_values.size, replace=True)))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "mean": mean,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(delta.size),
        "n_sessions": int(session_values.size),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.average(values[ok], weights=weights[ok]))


def _bootstrap_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[ok]
    weights = weights[ok]
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = _weighted_mean(values, weights)
    if int(n_bootstrap) <= 0 or values.size <= 1:
        lo = hi = mean
    else:
        boot = np.empty(int(n_bootstrap), dtype=np.float64)
        for i in range(int(n_bootstrap)):
            idx = rng.integers(0, values.size, size=values.size)
            boot[i] = _weighted_mean(values[idx], weights[idx])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"mean": mean, "ci_low": float(lo), "ci_high": float(hi), "n": int(values.size)}


def _residual_variance(residual: np.ndarray, *, floor: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2 or residual.shape[0] == 0:
        return np.full((0,), float("nan"), dtype=np.float64)
    ddof = 1 if residual.shape[0] > 1 else 0
    var = np.nanvar(residual, axis=0, ddof=ddof)
    return np.clip(var, float(floor), np.inf)


def _ledoit_logdet(residual: np.ndarray, *, floor: float) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2 or residual.shape[0] < 2:
        return float("nan")
    if LedoitWolf is None:
        return float("nan")
    ok = np.all(np.isfinite(residual), axis=1)
    residual = residual[ok]
    if residual.shape[0] < 2:
        return float("nan")
    cov = LedoitWolf().fit(residual).covariance_
    cov = np.asarray(cov, dtype=np.float64)
    cov.flat[:: cov.shape[0] + 1] += float(floor)
    sign, logdet = np.linalg.slogdet(cov)
    return float(logdet) if sign > 0 else float("nan")


def _information_fold_rows(
    *,
    condition_result: dict[str, Any],
    baseline_result: dict[str, Any],
    motion_summary: str,
    family: str,
    scale_id: str,
    latent: str,
    k: int,
    variance_floor: float,
) -> list[dict[str, Any]]:
    condition_folds = {int(row["fold"]): row for row in condition_result.get("fold_residuals", [])}
    baseline_folds = {int(row["fold"]): row for row in baseline_result.get("fold_residuals", [])}
    rows: list[dict[str, Any]] = []
    for fold in sorted(set(condition_folds).intersection(baseline_folds)):
        condition = condition_folds[fold]
        baseline = baseline_folds[fold]
        resid_c = np.asarray(condition["residual"], dtype=np.float64)
        resid_0 = np.asarray(baseline["residual"], dtype=np.float64)
        test_c = np.asarray(condition.get("test_idx", []), dtype=np.int64)
        test_0 = np.asarray(baseline.get("test_idx", []), dtype=np.int64)
        if test_c.size and test_0.size and not np.array_equal(test_c, test_0):
            raise ValueError(
                "Cannot compare residual covariances across unmatched held-out samples: "
                f"fold {fold} condition test_idx != baseline test_idx"
            )
        if resid_c.shape != resid_0.shape:
            raise ValueError(
                "Cannot compare residual covariances across unmatched fold targets: "
                f"condition {resid_c.shape}, baseline {resid_0.shape}"
            )
        var_c = _residual_variance(resid_c, floor=float(variance_floor))
        var_0 = _residual_variance(resid_0, floor=float(variance_floor))
        diag_bits = float(0.5 * np.nansum(np.log(var_0) - np.log(var_c)) / LN2)
        full_c = _ledoit_logdet(resid_c, floor=float(variance_floor))
        full_0 = _ledoit_logdet(resid_0, floor=float(variance_floor))
        full_bits = float(0.5 * (full_0 - full_c) / LN2) if np.isfinite(full_0) and np.isfinite(full_c) else float("nan")
        alpha_c = float(condition.get("alpha", float("nan")))
        alpha_0 = float(baseline.get("alpha", float("nan")))
        rows.append(
            {
                "motion_summary": motion_summary,
                "family": family,
                "scale_id": scale_id,
                "latent": latent,
                "k": int(k),
                "fold": int(fold),
                "n_test": int(resid_c.shape[0]),
                "target_dim": int(resid_c.shape[1]),
                "incremental_gain_info_diag_bits": diag_bits,
                "incremental_gain_info_diag_bits_per_dim": diag_bits / float(resid_c.shape[1]),
                "incremental_gain_info_full_bits": full_bits,
                "incremental_gain_info_full_bits_per_dim": (
                    full_bits / float(resid_c.shape[1]) if np.isfinite(full_bits) else float("nan")
                ),
                "baseline_alpha": alpha_0,
                "condition_alpha": alpha_c,
                "ridge_alpha_matched": bool(np.isclose(alpha_0, alpha_c, rtol=1e-10, atol=1e-12)),
                "variance_floor": float(variance_floor),
                "full_covariance_method": "ledoit_wolf" if LedoitWolf is not None else "unavailable",
            }
        )
    return rows


def _information_point_estimates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.asarray([row["n_test"] for row in rows], dtype=np.float64)
    diag = _weighted_mean(
        np.asarray([row["incremental_gain_info_diag_bits"] for row in rows], dtype=np.float64),
        weights,
    )
    diag_per_dim = _weighted_mean(
        np.asarray([row["incremental_gain_info_diag_bits_per_dim"] for row in rows], dtype=np.float64),
        weights,
    )
    full = _weighted_mean(
        np.asarray([row["incremental_gain_info_full_bits"] for row in rows], dtype=np.float64),
        weights,
    )
    full_per_dim = _weighted_mean(
        np.asarray([row["incremental_gain_info_full_bits_per_dim"] for row in rows], dtype=np.float64),
        weights,
    )
    return {
        "incremental_gain_info_diag_bits": diag,
        "incremental_gain_info_diag_bits_per_dim": diag_per_dim,
        "incremental_gain_info_full_bits": full,
        "incremental_gain_info_full_bits_per_dim": full_per_dim,
        "ridge_alpha_matched_all_folds": bool(rows and all(bool(row["ridge_alpha_matched"]) for row in rows)),
        "n_information_folds": int(np.sum(np.isfinite(weights) & (weights > 0))),
    }


def _ci_from_bootstrap(values: np.ndarray, point: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return float(point), float(point)
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _validate_information_intervals(rows: list[dict[str, Any]], *, value_col: str) -> None:
    if not rows:
        return
    lo_col = "info_diag_ci95_low"
    hi_col = "info_diag_ci95_high"
    if value_col not in rows[0] or lo_col not in rows[0] or hi_col not in rows[0]:
        return
    bad_rows = []
    for row in rows:
        value = float(row.get(value_col, np.nan))
        lo = float(row.get(lo_col, np.nan))
        hi = float(row.get(hi_col, np.nan))
        if np.isfinite(value) and np.isfinite(lo) and np.isfinite(hi) and (value < lo - 1e-9 or value > hi + 1e-9):
            bad_rows.append(row)
    if bad_rows:
        preview_cols = ("motion_summary", "family", "lhs_family", "rhs_family", "scale_id", "latent", "k")
        preview = pd.DataFrame(
            [{col: row.get(col, "") for col in preview_cols if col in row} for row in bad_rows[:6]]
        )
        raise ValueError(
            f"{len(bad_rows)} information rows have point estimates outside their diagonal CI; "
            f"refusing to write misleading tables. First bad rows:\n{preview.to_string(index=False)}"
        )


def _summarize_information_rows(
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    weights = np.asarray([row["n_test"] for row in rows], dtype=np.float64)

    def boot(column: str) -> dict[str, float]:
        return _bootstrap_weighted_mean(
            np.asarray([row[column] for row in rows], dtype=np.float64),
            weights,
            rng=rng,
            n_bootstrap=int(n_bootstrap),
        )

    diag = boot("incremental_gain_info_diag_bits")
    diag_per_dim = boot("incremental_gain_info_diag_bits_per_dim")
    full = boot("incremental_gain_info_full_bits")
    full_per_dim = boot("incremental_gain_info_full_bits_per_dim")
    return {
        "incremental_gain_info_diag_bits": diag["mean"],
        "info_diag_ci95_low": diag["ci_low"],
        "info_diag_ci95_high": diag["ci_high"],
        "incremental_gain_info_diag_bits_per_dim": diag_per_dim["mean"],
        "info_diag_per_dim_ci95_low": diag_per_dim["ci_low"],
        "info_diag_per_dim_ci95_high": diag_per_dim["ci_high"],
        "incremental_gain_info_full_bits": full["mean"],
        "info_full_ci95_low": full["ci_low"],
        "info_full_ci95_high": full["ci_high"],
        "incremental_gain_info_full_bits_per_dim": full_per_dim["mean"],
        "info_full_per_dim_ci95_low": full_per_dim["ci_low"],
        "info_full_per_dim_ci95_high": full_per_dim["ci_high"],
        "information_ci_method": "outer_fold_weighted_bootstrap",
        "ridge_alpha_matched_all_folds": bool(rows and all(bool(row["ridge_alpha_matched"]) for row in rows)),
        "n_information_folds": int(diag["n"]),
    }


def _summarize_information_contrast(
    lhs_rows: list[dict[str, Any]],
    rhs_rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    lhs_by_fold = {int(row["fold"]): row for row in lhs_rows}
    rhs_by_fold = {int(row["fold"]): row for row in rhs_rows}
    common_folds = sorted(set(lhs_by_fold).intersection(rhs_by_fold))
    weights = np.asarray([lhs_by_fold[fold]["n_test"] for fold in common_folds], dtype=np.float64)
    diag_delta = np.asarray(
        [
            lhs_by_fold[fold]["incremental_gain_info_diag_bits"]
            - rhs_by_fold[fold]["incremental_gain_info_diag_bits"]
            for fold in common_folds
        ],
        dtype=np.float64,
    )
    full_delta = np.asarray(
        [
            lhs_by_fold[fold]["incremental_gain_info_full_bits"]
            - rhs_by_fold[fold]["incremental_gain_info_full_bits"]
            for fold in common_folds
        ],
        dtype=np.float64,
    )
    diag = _bootstrap_weighted_mean(diag_delta, weights, rng=rng, n_bootstrap=int(n_bootstrap))
    full = _bootstrap_weighted_mean(full_delta, weights, rng=rng, n_bootstrap=int(n_bootstrap))
    return {
        "incremental_gain_delta_info_diag_bits": diag["mean"],
        "info_diag_ci95_low": diag["ci_low"],
        "info_diag_ci95_high": diag["ci_high"],
        "incremental_gain_delta_info_full_bits": full["mean"],
        "info_full_ci95_low": full["ci_low"],
        "info_full_ci95_high": full["ci_high"],
        "information_ci_method": "outer_fold_weighted_bootstrap",
        "ridge_alpha_matched_all_folds": bool(
            common_folds
            and all(
                bool(lhs_by_fold[fold]["ridge_alpha_matched"]) and bool(rhs_by_fold[fold]["ridge_alpha_matched"])
                for fold in common_folds
            )
        ),
        "n_information_folds": int(diag["n"]),
    }


def _bootstrap_indices_by_group(
    groups: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    chosen = rng.choice(unique, size=unique.size, replace=True)
    parts = [np.flatnonzero(groups == group) for group in chosen]
    return np.concatenate(parts).astype(np.int64) if parts else np.arange(groups.size, dtype=np.int64)


def _decode_information_rows_for_matrices(
    *,
    X_static: np.ndarray,
    X_condition: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
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
) -> list[dict[str, Any]]:
    baseline_result = _decode(
        X_static,
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
    condition_result = _decode(
        X_condition,
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
    rows = _information_fold_rows(
        condition_result=condition_result,
        baseline_result=baseline_result,
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
    return rows


def _decode_pipeline_information_bootstrap(
    *,
    X_static: np.ndarray,
    X_condition: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    point_rows: list[dict[str, Any]],
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
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    point = _information_point_estimates(point_rows)
    diag_boot: list[float] = []
    diag_per_dim_boot: list[float] = []
    full_boot: list[float] = []
    full_per_dim_boot: list[float] = []
    for boot_idx in range(int(n_bootstrap)):
        idx = _bootstrap_indices_by_group(groups, rng=rng)
        if np.unique(np.asarray(groups)[idx]).size < 2:
            continue
        rows = _decode_information_rows_for_matrices(
            X_static=np.asarray(X_static)[idx],
            X_condition=np.asarray(X_condition)[idx],
            Z=np.asarray(Z)[idx],
            groups=np.asarray(groups)[idx],
            k=int(k),
            alphas=alphas,
            alpha_mode=alpha_mode,
            fixed_alpha=fixed_alpha,
            outer_folds=int(outer_folds),
            inner_folds=int(inner_folds),
            seed=int(seed) + 10000 + boot_idx,
            variance_floor=float(variance_floor),
            allow_unmatched_alpha=bool(allow_unmatched_alpha),
            motion_summary=motion_summary,
            family=family,
            scale_id=scale_id,
            latent=latent,
        )
        boot_point = _information_point_estimates(rows)
        diag_boot.append(float(boot_point["incremental_gain_info_diag_bits"]))
        diag_per_dim_boot.append(float(boot_point["incremental_gain_info_diag_bits_per_dim"]))
        full_boot.append(float(boot_point["incremental_gain_info_full_bits"]))
        full_per_dim_boot.append(float(boot_point["incremental_gain_info_full_bits_per_dim"]))

    diag_lo, diag_hi = _ci_from_bootstrap(np.asarray(diag_boot), point["incremental_gain_info_diag_bits"])
    diag_per_dim_lo, diag_per_dim_hi = _ci_from_bootstrap(
        np.asarray(diag_per_dim_boot),
        point["incremental_gain_info_diag_bits_per_dim"],
    )
    full_lo, full_hi = _ci_from_bootstrap(np.asarray(full_boot), point["incremental_gain_info_full_bits"])
    full_per_dim_lo, full_per_dim_hi = _ci_from_bootstrap(
        np.asarray(full_per_dim_boot),
        point["incremental_gain_info_full_bits_per_dim"],
    )
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
        "information_ci_method": "decode_pipeline_group_bootstrap",
        "ridge_alpha_matched_all_folds": bool(point["ridge_alpha_matched_all_folds"]),
        "n_information_folds": int(point["n_information_folds"]),
        "n_information_bootstrap_success": int(np.sum(np.isfinite(diag_boot))),
    }


def _decode_pipeline_information_contrast_bootstrap(
    *,
    X_static: np.ndarray,
    X_lhs: np.ndarray,
    X_rhs: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    lhs_rows: list[dict[str, Any]],
    rhs_rows: list[dict[str, Any]],
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
    lhs_family: str,
    rhs_family: str,
    scale_id: str,
    latent: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    point_lhs = _information_point_estimates(lhs_rows)
    point_rhs = _information_point_estimates(rhs_rows)
    point_diag = float(point_lhs["incremental_gain_info_diag_bits"] - point_rhs["incremental_gain_info_diag_bits"])
    point_full = float(point_lhs["incremental_gain_info_full_bits"] - point_rhs["incremental_gain_info_full_bits"])
    diag_boot: list[float] = []
    full_boot: list[float] = []
    for boot_idx in range(int(n_bootstrap)):
        idx = _bootstrap_indices_by_group(groups, rng=rng)
        if np.unique(np.asarray(groups)[idx]).size < 2:
            continue
        boot_common = {
            "X_static": np.asarray(X_static)[idx],
            "Z": np.asarray(Z)[idx],
            "groups": np.asarray(groups)[idx],
            "k": int(k),
            "alphas": alphas,
            "alpha_mode": alpha_mode,
            "fixed_alpha": fixed_alpha,
            "outer_folds": int(outer_folds),
            "inner_folds": int(inner_folds),
            "variance_floor": float(variance_floor),
            "allow_unmatched_alpha": bool(allow_unmatched_alpha),
            "motion_summary": motion_summary,
            "scale_id": scale_id,
            "latent": latent,
        }
        lhs_boot_rows = _decode_information_rows_for_matrices(
            **boot_common,
            X_condition=np.asarray(X_lhs)[idx],
            seed=int(seed) + 20000 + boot_idx,
            family=lhs_family,
        )
        rhs_boot_rows = _decode_information_rows_for_matrices(
            **boot_common,
            X_condition=np.asarray(X_rhs)[idx],
            seed=int(seed) + 20000 + boot_idx,
            family=rhs_family,
        )
        lhs_boot = _information_point_estimates(lhs_boot_rows)
        rhs_boot = _information_point_estimates(rhs_boot_rows)
        diag_boot.append(float(lhs_boot["incremental_gain_info_diag_bits"] - rhs_boot["incremental_gain_info_diag_bits"]))
        full_boot.append(float(lhs_boot["incremental_gain_info_full_bits"] - rhs_boot["incremental_gain_info_full_bits"]))

    diag_lo, diag_hi = _ci_from_bootstrap(np.asarray(diag_boot), point_diag)
    full_lo, full_hi = _ci_from_bootstrap(np.asarray(full_boot), point_full)
    return {
        "incremental_gain_delta_info_diag_bits": point_diag,
        "info_diag_ci95_low": diag_lo,
        "info_diag_ci95_high": diag_hi,
        "incremental_gain_delta_info_full_bits": point_full,
        "info_full_ci95_low": full_lo,
        "info_full_ci95_high": full_hi,
        "information_ci_method": "decode_pipeline_group_bootstrap",
        "ridge_alpha_matched_all_folds": bool(
            point_lhs["ridge_alpha_matched_all_folds"] and point_rhs["ridge_alpha_matched_all_folds"]
        ),
        "n_information_folds": int(min(point_lhs["n_information_folds"], point_rhs["n_information_folds"])),
        "n_information_bootstrap_success": int(np.sum(np.isfinite(diag_boot))),
    }


def _decode(
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    *,
    k: int,
    alphas: list[float],
    alpha_mode: str,
    fixed_alpha: float | None,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    return _cross_validated_decode(
        X,
        Z,
        groups,
        k=int(k),
        alphas=alphas,
        alpha_mode=str(alpha_mode),
        fixed_alpha=float(fixed_alpha) if fixed_alpha is not None else None,
        outer_folds=int(outer_folds),
        inner_folds=int(inner_folds),
        seed=int(seed),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summaries", default="temporal_pca,temporal_dct,temporal_delta_pca,temporal_dct_delta,mean,delta_mean")
    parser.add_argument("--families", default="empirical,ou,brownian,rotated")
    parser.add_argument(
        "--contrast-pairs",
        default="empirical:ou,empirical:brownian,empirical:rotated",
        help=(
            "Comma-separated lhs:rhs family contrasts for incremental gains. "
            "Use names from --families; e.g. actual_paired_empirical:matched_unpaired_empirical."
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
        choices=("image", "session"),
        default="image",
        help="CV grouping for decoding. Use image for the pathfinder; session is stricter by recording session.",
    )
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
            "CI mode for information columns. `fold` bootstraps outer-fold scalar estimates; "
            "`decode_bootstrap` resamples decode groups, refits decoders, and recomputes residual covariances."
        ),
    )
    parser.add_argument(
        "--allow-unmatched-alpha-information",
        action="store_true",
        help="Allow information columns when static and static-plus-motion folds chose different ridge alphas.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "incremental_static_plus_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    images = pd.read_csv(run_dir / "analysis_images.csv")
    sessions = images["session"].to_numpy()
    decode_groups = images["image_index"].to_numpy(dtype=int) if str(args.decode_group_mode) == "image" else sessions
    latents = _filter_latents(_load_npz(run_dir / "latent_feature_arrays.npz"), _parse_list(args.latent_names))
    responses = _load_npz(run_dir / "response_summary_arrays.npz")
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

    decode_rows: list[dict[str, Any]] = []
    per_image: dict[tuple[str, str, str, str, str, int], np.ndarray] = {}
    information_by_key: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = {}
    information_fold_rows: list[dict[str, Any]] = []
    condition_matrix_by_key: dict[tuple[str, str, str, str, str, int], np.ndarray] = {}
    static_matrix_by_summary: dict[str, np.ndarray] = {}
    latent_matrix_by_name: dict[str, np.ndarray] = {}
    static_decode_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    for summary in summaries:
        static_summary = STATIC_SUMMARY_FOR_MOTION.get(summary)
        if static_summary is None:
            raise ValueError(f"No static summary mapping is defined for {summary!r}")
        static_key = _response_key(static_summary, "static", "static")
        if static_key not in responses:
            raise ValueError(f"Missing static response array {static_key!r}")
        X_static = responses[static_key]
        static_matrix_by_summary[summary] = X_static
        for latent_name, Z in latents.items():
            latent_matrix_by_name[latent_name] = Z
            for k in pca_k_list:
                static_cache_key = (static_summary, latent_name, int(k))
                if static_cache_key not in static_decode_cache:
                    static_decode_cache[static_cache_key] = _decode(
                        X_static,
                        Z,
                        decode_groups,
                        k=k,
                        alphas=alphas,
                        alpha_mode=alpha_mode,
                        fixed_alpha=fixed_alpha,
                        outer_folds=int(args.outer_folds),
                        inner_folds=int(args.inner_folds),
                        seed=int(args.seed),
                    )
                static_result = static_decode_cache[static_cache_key]
                static_per_key = (summary, "static_only", "static", "static", latent_name, int(k))
                per_image[static_per_key] = np.asarray(static_result["per_window_score"], dtype=np.float64)
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
                        motion_key = _response_key(summary, family, scale_id)
                        if motion_key not in responses:
                            continue
                        X_motion = responses[motion_key]
                        X_aug = np.concatenate([X_static, X_motion], axis=1)
                        aug_result = _decode(
                            X_aug,
                            Z,
                            decode_groups,
                            k=k,
                            alphas=alphas,
                            alpha_mode=alpha_mode,
                            fixed_alpha=fixed_alpha,
                            outer_folds=int(args.outer_folds),
                            inner_folds=int(args.inner_folds),
                            seed=int(args.seed),
                        )
                        key = (summary, "static_plus_motion", family, scale_id, latent_name, int(k))
                        per_image[key] = np.asarray(aug_result["per_window_score"], dtype=np.float64)
                        condition_matrix_by_key[key] = X_aug
                        info_rows = _information_fold_rows(
                            condition_result=aug_result,
                            baseline_result=static_result,
                            motion_summary=summary,
                            family=family,
                            scale_id=scale_id,
                            latent=latent_name,
                            k=int(k),
                            variance_floor=float(args.information_variance_floor),
                        )
                        if info_rows and not bool(args.allow_unmatched_alpha_information):
                            alpha_matched = all(bool(row["ridge_alpha_matched"]) for row in info_rows)
                            if not alpha_matched:
                                raise ValueError(
                                    "Information gain requires matched ridge alpha between static and "
                                    "static-plus-motion folds. Use fixed ridge alpha, shared-alpha decoding, "
                                    "or pass --allow-unmatched-alpha-information for audit-only output."
                                )
                        information_by_key[key] = info_rows
                        information_fold_rows.extend(info_rows)
                        decode_rows.append(
                            {
                                "motion_summary": summary,
                                "static_summary": static_summary,
                                "model": "static_plus_motion",
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "mean_neg_mse": float(aug_result["mean_neg_mse"]),
                                "r2": float(aug_result["r2"]),
                                "chosen_alpha_median": float(aug_result["chosen_alpha_median"]),
                                "ridge_alpha_mode": str(aug_result["ridge_alpha_mode"]),
                                "fixed_ridge_alpha": float(fixed_alpha) if alpha_mode == "fixed" else float("nan"),
                                "target_dim": int(aug_result["target_dim"]),
                                "n_images": int(X_aug.shape[0]),
                                "decode_group_mode": str(args.decode_group_mode),
                                "n_decode_groups": int(np.unique(decode_groups).size),
                                "feature_dim": int(X_aug.shape[1]),
                            }
                        )

    gain_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for summary in summaries:
        for latent_name in latents:
            for k in pca_k_list:
                static_key = (summary, "static_only", "static", "static", latent_name, int(k))
                if static_key not in per_image:
                    continue
                for scale_id in scale_ids:
                    gain_by_family: dict[str, np.ndarray] = {}
                    for family in families:
                        aug_key = (summary, "static_plus_motion", family, scale_id, latent_name, int(k))
                        if aug_key not in per_image:
                            continue
                        gain = per_image[aug_key] - per_image[static_key]
                        gain_by_family[family] = gain
                        boot = _session_bootstrap_delta(per_image[aug_key], per_image[static_key], sessions, rng=rng, n_bootstrap=int(args.n_bootstrap))
                        row = {
                            "motion_summary": summary,
                            "family": family,
                            "scale_id": scale_id,
                            "latent": latent_name,
                            "k": int(k),
                            "incremental_gain_neg_mse": boot["mean"],
                            "ci95_low": boot["ci_low"],
                            "ci95_high": boot["ci_high"],
                            "n_images": boot["n"],
                            "n_sessions": boot["n_sessions"],
                            "information_variance_floor": float(args.information_variance_floor),
                        }
                        row.update(
                            (
                                _decode_pipeline_information_bootstrap(
                                    X_static=static_matrix_by_summary[summary],
                                    X_condition=condition_matrix_by_key[aug_key],
                                    Z=latent_matrix_by_name[latent_name],
                                    groups=decode_groups,
                                    point_rows=information_by_key.get(aug_key, []),
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
                                    rng=rng,
                                    n_bootstrap=int(args.n_bootstrap),
                                )
                                if str(args.information_ci_mode) == "decode_bootstrap"
                                else _summarize_information_rows(
                                    information_by_key.get(aug_key, []),
                                    rng=rng,
                                    n_bootstrap=int(args.n_bootstrap),
                                )
                            )
                        )
                        gain_rows.append(row)
                    for lhs, rhs in contrast_pairs:
                        if lhs not in gain_by_family or rhs not in gain_by_family:
                            continue
                        boot = _session_bootstrap_delta(
                            gain_by_family[lhs],
                            gain_by_family[rhs],
                            sessions,
                            rng=rng,
                            n_bootstrap=int(args.n_bootstrap),
                        )
                        lhs_key = (summary, "static_plus_motion", lhs, scale_id, latent_name, int(k))
                        rhs_key = (summary, "static_plus_motion", rhs, scale_id, latent_name, int(k))
                        row = {
                            "motion_summary": summary,
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
                            "information_variance_floor": float(args.information_variance_floor),
                        }
                        row.update(
                            (
                                _decode_pipeline_information_contrast_bootstrap(
                                    X_static=static_matrix_by_summary[summary],
                                    X_lhs=condition_matrix_by_key[lhs_key],
                                    X_rhs=condition_matrix_by_key[rhs_key],
                                    Z=latent_matrix_by_name[latent_name],
                                    groups=decode_groups,
                                    lhs_rows=information_by_key.get(lhs_key, []),
                                    rhs_rows=information_by_key.get(rhs_key, []),
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
                                    lhs_family=lhs,
                                    rhs_family=rhs,
                                    scale_id=scale_id,
                                    latent=latent_name,
                                    rng=rng,
                                    n_bootstrap=int(args.n_bootstrap),
                                )
                                if str(args.information_ci_mode) == "decode_bootstrap"
                                else _summarize_information_contrast(
                                    information_by_key.get(lhs_key, []),
                                    information_by_key.get(rhs_key, []),
                                    rng=rng,
                                    n_bootstrap=int(args.n_bootstrap),
                                )
                            )
                        )
                        contrast_rows.append(row)

    _validate_information_intervals(gain_rows, value_col="incremental_gain_info_diag_bits")
    _validate_information_intervals(contrast_rows, value_col="incremental_gain_delta_info_diag_bits")

    _write_csv(out_dir / "incremental_decode_summary.csv", decode_rows)
    _write_csv(out_dir / "incremental_gain_vs_static.csv", gain_rows)
    _write_csv(out_dir / "incremental_gain_contrasts.csv", contrast_rows)
    _write_csv(out_dir / "incremental_information_by_fold.csv", information_fold_rows)
    _write_json(
        out_dir / "run_metadata.json",
        {
            "source_run_dir": run_dir,
            "summaries": summaries,
            "static_summary_for_motion": STATIC_SUMMARY_FOR_MOTION,
            "families": families,
            "contrast_pairs": contrast_pairs,
            "scale_ids": scale_ids,
            "latent_names": list(latents),
            "pca_k_list": pca_k_list,
            "ridge_alpha_mode": alpha_mode,
            "fixed_ridge_alpha": fixed_alpha if alpha_mode == "fixed" else None,
            "decode_group_mode": str(args.decode_group_mode),
            "n_decode_groups": int(np.unique(decode_groups).size),
            "outer_folds": int(args.outer_folds),
            "n_bootstrap": int(args.n_bootstrap),
            "information_axis": {
                "headline": "diagonal_gaussian_variational_bound_increment_bits",
                "diag_formula": "0.5 * sum_j log(var_static_j / var_condition_j) / log(2)",
                "full_covariance_formula": "0.5 * (logdet(cov_static) - logdet(cov_condition)) / log(2)",
                "full_covariance_method": "ledoit_wolf" if LedoitWolf is not None else "unavailable",
                "ci_method": (
                    "decode_pipeline_group_bootstrap"
                    if str(args.information_ci_mode) == "decode_bootstrap"
                    else "outer_fold_weighted_bootstrap"
                ),
                "ci_mode": str(args.information_ci_mode),
                "variance_floor": float(args.information_variance_floor),
                "ridge_alpha_requirement": "interpret cleanly when static and static-plus-motion folds use matched alpha",
                "allow_unmatched_alpha_information": bool(args.allow_unmatched_alpha_information),
            },
            "seed": int(args.seed),
        },
    )
    report = [
        "# Incremental Static Plus Motion",
        "",
        f"Source run: `{run_dir}`",
        "",
        "Question:",
        "",
        "`z ~ R_static` versus `z ~ R_static + R_motion_summary`.",
        "",
        "Primary files:",
        "- `incremental_gain_vs_static.csv`",
        "- `incremental_gain_contrasts.csv`",
        "- `incremental_information_by_fold.csv`",
        "- `incremental_decode_summary.csv`",
        "",
        "Information axis:",
        "",
        "The promoted information columns report the diagonal Gaussian decoder lower-bound",
        "increment in bits over the static baseline. Full-covariance Ledoit-Wolf log-det",
        "columns are included as a robustness supplement. Existing `-MSE` columns are kept",
        "for legacy provenance only.",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote incremental summaries to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
