"""Cache-only compact chart-swap analysis for BackImage response tables.

This analysis tests whether the compact motion residual is best explained by
the candidate image's own local translation chart, rather than by a wrong
candidate chart, a global pooled chart, or low-dimensional controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .analyze_compact_mechanism import (
        _load_basis,
        _parse_int_list,
        _parse_list,
        _random_basis,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from .analyze_feature_posterior import _add_uncertainty_fields, _filter_manifest, _filter_response_cache_manifest
    from .likelihood import effective_count, entropy, logmeanexp, posterior_from_log_scores, rank_desc, true_margin
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.backimage_trajectory_observer.analyze_compact_mechanism import (
        _load_basis,
        _parse_int_list,
        _parse_list,
        _random_basis,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from declan.backimage_trajectory_observer.analyze_feature_posterior import (
        _add_uncertainty_fields,
        _filter_manifest,
        _filter_response_cache_manifest,
    )
    from declan.backimage_trajectory_observer.likelihood import (
        effective_count,
        entropy,
        logmeanexp,
        posterior_from_log_scores,
        rank_desc,
        true_margin,
    )


CHART_FAMILIES = (
    "correct_chart",
    "wrong_chart_roll",
    "wrong_chart_pool",
    "global_chart",
    "static_no_motion_chart",
    "zero_chart",
)
CHART_ALIASES = {
    "zero_chart": "static_no_motion_chart",
}
SLOT_ALIGNED_CHART_FAMILIES = frozenset({"wrong_chart_roll", "global_chart"})
BASIS_TYPES = (
    "compact",
    "unit_shuffle_compact",
    "random_k",
    "gain_axis",
    "static_pc_k",
)
SUMMARY_METRICS = (
    "chart_correct",
    "chart_true_score",
    "chart_true_margin",
    "candidate_posterior_true_mass",
)


def _progress(message: str) -> None:
    print(f"[compact-chart-swap] {message}", flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_npz_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(int(n_candidates))]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _scalar_float_from_table_or_row(table: dict[str, np.ndarray], row: pd.Series, key: str, default: float = np.nan) -> float:
    if key in table:
        arr = np.asarray(table[key], dtype=np.float64).reshape(-1)
        if arr.size and np.isfinite(arr[0]):
            return float(arr[0])
    if key in row.index:
        value = float(row.get(key, default))
        if np.isfinite(value):
            return value
    return float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "", "nan"}:
        return False
    return bool(value)


def _canonical_chart_family(value: str) -> str:
    text = str(value).strip()
    return CHART_ALIASES.get(text, text)


def _stable_trajectory_key(value: Any) -> str:
    text = str(value)
    parts = text.split(":")
    if len(parts) >= 2:
        return ":".join(parts[:-1])
    return text


def _trajectory_slot_alignment(
    table: dict[str, np.ndarray],
    *,
    n_candidates: int,
    n_trajectories: int,
) -> dict[str, Any]:
    if "prior_trajectory_ids" not in table:
        return {
            "trajectory_slot_alignment_status": "missing_prior_trajectory_ids",
            "trajectory_slots_aligned": False,
            "trajectory_slot_aligned_fraction": float("nan"),
            "trajectory_slot_mismatch_count": -1,
        }
    ids = np.asarray(table["prior_trajectory_ids"])
    if ids.shape == (int(n_trajectories),):
        return {
            "trajectory_slot_alignment_status": "shared_vector_prior_trajectory_ids",
            "trajectory_slots_aligned": True,
            "trajectory_slot_aligned_fraction": 1.0,
            "trajectory_slot_mismatch_count": 0,
        }
    expected = (int(n_candidates), int(n_trajectories))
    if ids.shape != expected:
        raise ValueError(f"prior_trajectory_ids shape {ids.shape} does not match expected {(int(n_trajectories),)} or {expected}")
    stable = np.empty(expected, dtype=object)
    for index, value in np.ndenumerate(ids):
        stable[index] = _stable_trajectory_key(value)
    aligned = stable == stable[0:1, :]
    mismatch_count = int(np.count_nonzero(~aligned))
    return {
        "trajectory_slot_alignment_status": "candidate_table_prior_trajectory_ids",
        "trajectory_slots_aligned": bool(mismatch_count == 0),
        "trajectory_slot_aligned_fraction": float(np.mean(aligned)),
        "trajectory_slot_mismatch_count": mismatch_count,
    }


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return out


def _static_candidate_set_key(table: dict[str, np.ndarray], *, candidate_set_mode: str, n_candidates: int) -> str:
    candidate_ids = _candidate_ids(table, int(n_candidates))
    return f"{candidate_set_mode}|{'/'.join(candidate_ids)}"


def _load_static_zero_records(
    manifest: pd.DataFrame,
    base: Path,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    records_by_key: dict[str, dict[str, Any]] = {}
    table_key_by_index: dict[int, str] = {}
    for table_index, row in manifest.iterrows():
        table = _load_npz_table(base / str(row["response_cache_path"]))
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float32)
        key = _static_candidate_set_key(
            table,
            candidate_set_mode=str(row.get("candidate_set_mode", "")),
            n_candidates=zero.shape[0],
        )
        table_key_by_index[int(table_index)] = key
        if key not in records_by_key:
            records_by_key[key] = {
                "static_candidate_set_key": key,
                "first_table_index": int(table_index),
                "first_trial_id": int(row.get("trial_id", table_index)),
                "zero": zero,
            }
    return list(records_by_key.values()), table_key_by_index


def _assign_static_folds(keys: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
    if not keys:
        return {}
    n = len(keys)
    folds = int(max(1, min(int(n_folds), n)))
    ordered = np.asarray(sorted(str(key) for key in keys), dtype=object)
    rng = np.random.default_rng(int(seed))
    shuffled = ordered[rng.permutation(n)]
    return {str(key): int(i % folds) for i, key in enumerate(shuffled.tolist())}


def _build_static_pc_context(
    *,
    manifest: pd.DataFrame,
    base: Path,
    n_units: int,
    k_max: int,
    scope: str,
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    records, table_key_by_index = _load_static_zero_records(manifest, base)
    zero_tables = [np.asarray(record["zero"], dtype=np.float32) for record in records]
    if not records:
        basis = _static_pc_basis([], n_units=n_units, k_max=k_max)
        return {
            "scope": "empty_static_pc_fallback",
            "basis": basis,
            "basis_by_fold": {},
            "fold_by_key": {},
            "table_key_by_index": table_key_by_index,
            "n_unique_candidate_sets": 0,
            "n_folds": 0,
        }
    if str(scope) == "selected_tables":
        return {
            "scope": "in_sample_unique_candidate_sets",
            "basis": _static_pc_basis(zero_tables, n_units=n_units, k_max=k_max),
            "basis_by_fold": {},
            "fold_by_key": {},
            "table_key_by_index": table_key_by_index,
            "n_unique_candidate_sets": int(len(records)),
            "n_folds": 1,
        }
    if str(scope) != "candidate_set_fold_disjoint":
        raise ValueError("static_pc_scope must be 'selected_tables' or 'candidate_set_fold_disjoint'")
    fold_by_key = _assign_static_folds(
        [str(record["static_candidate_set_key"]) for record in records],
        n_folds=int(n_folds),
        seed=int(seed),
    )
    folds = sorted(set(fold_by_key.values()))
    basis_by_fold: dict[int, np.ndarray] = {}
    for fold in folds:
        train = [
            np.asarray(record["zero"], dtype=np.float32)
            for record in records
            if int(fold_by_key[str(record["static_candidate_set_key"])]) != int(fold)
        ]
        if not train:
            train = zero_tables
        basis_by_fold[int(fold)] = _static_pc_basis(train, n_units=n_units, k_max=k_max)
    return {
        "scope": f"candidate_set_fold_disjoint_{len(folds)}fold",
        "basis": None,
        "basis_by_fold": basis_by_fold,
        "fold_by_key": fold_by_key,
        "table_key_by_index": table_key_by_index,
        "n_unique_candidate_sets": int(len(records)),
        "n_folds": int(len(folds)),
    }


def _static_basis_for_table(static_pc: dict[str, Any] | None, table_index: int) -> tuple[np.ndarray | None, str, int, str]:
    if static_pc is None:
        return None, "none", -1, ""
    if static_pc.get("basis") is not None:
        return np.asarray(static_pc["basis"], dtype=np.float64), str(static_pc["scope"]), -1, ""
    key = str(static_pc["table_key_by_index"].get(int(table_index), ""))
    fold = int(static_pc["fold_by_key"].get(key, -1))
    basis_by_fold = static_pc.get("basis_by_fold", {})
    if fold not in basis_by_fold:
        raise ValueError(f"No static-PC fold basis for table_index={table_index} key={key!r} fold={fold}")
    return np.asarray(basis_by_fold[fold], dtype=np.float64), str(static_pc["scope"]), fold, key


def _project_coords(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project response deltas into basis coordinates without reconstructing."""
    arr = np.asarray(delta, dtype=np.float64)
    u = np.asarray(basis, dtype=np.float64)
    if arr.shape[-1] != u.shape[0]:
        raise ValueError(f"delta unit dimension {arr.shape[-1]} does not match basis units {u.shape[0]}")
    return np.tensordot(arr, u, axes=([-1], [0]))


def _coord_scale(coords: np.ndarray, *, min_scale: float) -> np.ndarray:
    flat = np.asarray(coords, dtype=np.float64).reshape(-1, coords.shape[-1])
    sd = np.std(flat, axis=0)
    rms = np.sqrt(np.mean(flat * flat, axis=0))
    scale = np.where(np.isfinite(sd) & (sd > float(min_scale)), sd, rms)
    scale = np.where(np.isfinite(scale) & (scale > float(min_scale)), scale, 1.0)
    return scale.astype(np.float64, copy=False)


def _template_log_scores(obs_coords: np.ndarray, templates: np.ndarray, scale: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs_coords, dtype=np.float64)
    pred = np.asarray(templates, dtype=np.float64)
    if pred.ndim != 3:
        raise ValueError(f"templates must be (template, time, coord), got {pred.shape}")
    if obs.shape != pred.shape[1:]:
        raise ValueError(f"obs coords shape {obs.shape} does not match templates trailing shape {pred.shape[1:]}")
    sc = np.asarray(scale, dtype=np.float64)
    if sc.shape != (obs.shape[-1],):
        raise ValueError(f"scale shape {sc.shape} does not match coord dimension {obs.shape[-1]}")
    diff = (pred - obs[None, :, :]) / sc[None, None, :]
    return -0.5 * np.sum(diff * diff, axis=(1, 2))


def _templates_for_candidate(prior_coords: np.ndarray, chart_family: str, candidate_index: int) -> np.ndarray:
    chart_family = _canonical_chart_family(chart_family)
    n_candidates, n_traj, n_time, k_dim = prior_coords.shape
    c = int(candidate_index)
    if chart_family == "correct_chart":
        return np.asarray(prior_coords[c], dtype=np.float64)
    if chart_family == "wrong_chart_roll":
        if n_candidates <= 1:
            return np.asarray(prior_coords[c], dtype=np.float64)
        return np.asarray(prior_coords[(c + 1) % n_candidates], dtype=np.float64)
    if chart_family == "wrong_chart_pool":
        if n_candidates <= 1:
            return np.asarray(prior_coords[c], dtype=np.float64)
        sources = [idx for idx in range(n_candidates) if idx != c]
        return np.asarray(prior_coords[sources], dtype=np.float64).reshape(-1, n_time, k_dim)
    if chart_family == "global_chart":
        return np.mean(np.asarray(prior_coords, dtype=np.float64), axis=0)
    if chart_family == "static_no_motion_chart":
        return np.zeros((1, n_time, k_dim), dtype=np.float64)
    raise ValueError(f"Unsupported chart family {chart_family!r}")


def _prediction(scores: np.ndarray) -> int:
    vals = np.asarray(scores, dtype=np.float64)
    if vals.size == 0 or not np.isfinite(vals).any():
        return -1
    return int(np.nanargmax(vals))


def _score_chart_family(
    *,
    y_obs_counts: np.ndarray,
    prior_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    basis: np.ndarray,
    chart_family: str,
    true_candidate_index: int,
    candidate_ids: list[str],
    nearest_trajectory_index: int = -1,
    true_trajectory_index: int = -1,
    scale_floor: float = 1e-8,
) -> dict[str, Any]:
    """Score candidate images by chart-family residual matching in basis coordinates."""
    prior = np.asarray(prior_lambda_counts, dtype=np.float64)
    zero = np.asarray(zero_lambda_counts, dtype=np.float64)
    y_obs = np.asarray(y_obs_counts, dtype=np.float64)
    if prior.ndim != 4:
        raise ValueError(f"prior_lambda_counts must be (candidate, trajectory, time, unit), got {prior.shape}")
    n_candidates, n_traj, n_time, n_units = prior.shape
    if zero.shape != (n_candidates, n_time, n_units):
        raise ValueError(f"zero_lambda_counts shape {zero.shape} does not match expected {(n_candidates, n_time, n_units)}")
    if y_obs.shape != (n_time, n_units):
        raise ValueError(f"y_obs_counts shape {y_obs.shape} does not match expected {(n_time, n_units)}")
    if len(candidate_ids) != n_candidates:
        raise ValueError(f"candidate_ids length {len(candidate_ids)} does not match n_candidates={n_candidates}")
    true_idx = int(true_candidate_index)
    if true_idx < 0 or true_idx >= n_candidates:
        raise ValueError(f"true_candidate_index {true_idx} outside candidate table size {n_candidates}")

    prior_delta = prior - zero[:, None, :, :]
    obs_delta_by_candidate = y_obs[None, :, :] - zero
    prior_coords = _project_coords(prior_delta, basis)
    obs_coords = _project_coords(obs_delta_by_candidate, basis)
    scale = _coord_scale(prior_coords, min_scale=float(scale_floor))
    return _score_chart_family_from_coords(
        prior_coords=prior_coords,
        obs_coords=obs_coords,
        scale=scale,
        chart_family=chart_family,
        true_candidate_index=true_idx,
        candidate_ids=candidate_ids,
        nearest_trajectory_index=nearest_trajectory_index,
        true_trajectory_index=true_trajectory_index,
        n_units=n_units,
    )


def _score_chart_family_from_coords(
    *,
    prior_coords: np.ndarray,
    obs_coords: np.ndarray,
    scale: np.ndarray,
    chart_family: str,
    true_candidate_index: int,
    candidate_ids: list[str],
    nearest_trajectory_index: int = -1,
    true_trajectory_index: int = -1,
    n_units: int = -1,
) -> dict[str, Any]:
    """Score candidate images from precomputed projected residual coordinates."""
    chart_family = _canonical_chart_family(chart_family)
    prior_coords = np.asarray(prior_coords, dtype=np.float64)
    obs_coords = np.asarray(obs_coords, dtype=np.float64)
    if prior_coords.ndim != 4:
        raise ValueError(f"prior_coords must be (candidate, trajectory, time, coord), got {prior_coords.shape}")
    n_candidates, n_traj, n_time, k_dim = prior_coords.shape
    if obs_coords.shape != (n_candidates, n_time, k_dim):
        raise ValueError(f"obs_coords shape {obs_coords.shape} does not match expected {(n_candidates, n_time, k_dim)}")
    if len(candidate_ids) != n_candidates:
        raise ValueError(f"candidate_ids length {len(candidate_ids)} does not match n_candidates={n_candidates}")
    true_idx = int(true_candidate_index)
    if true_idx < 0 or true_idx >= n_candidates:
        raise ValueError(f"true_candidate_index {true_idx} outside candidate table size {n_candidates}")

    candidate_scores = np.empty(n_candidates, dtype=np.float64)
    best_template_scores = np.empty(n_candidates, dtype=np.float64)
    template_counts = np.empty(n_candidates, dtype=np.int64)
    true_template_scores: np.ndarray | None = None
    for cand_i in range(n_candidates):
        templates = _templates_for_candidate(prior_coords, chart_family, cand_i)
        template_scores = _template_log_scores(obs_coords[cand_i], templates, scale)
        candidate_scores[cand_i] = float(logmeanexp(template_scores))
        best_template_scores[cand_i] = float(np.max(template_scores))
        template_counts[cand_i] = int(template_scores.size)
        if cand_i == true_idx:
            true_template_scores = template_scores

    if true_template_scores is None:
        raise RuntimeError("Internal error: true candidate template scores were not computed")

    pred = _prediction(candidate_scores)
    posterior = posterior_from_log_scores(candidate_scores)
    template_posterior = posterior_from_log_scores(true_template_scores)
    top_template = _prediction(true_template_scores)
    nearest_tau_rank = float("nan")
    true_tau_rank = float("nan")
    if int(template_counts[true_idx]) == n_traj:
        if 0 <= int(nearest_trajectory_index) < n_traj:
            nearest_tau_rank = rank_desc(true_template_scores, int(nearest_trajectory_index))
        if 0 <= int(true_trajectory_index) < n_traj:
            true_tau_rank = rank_desc(true_template_scores, int(true_trajectory_index))

    return {
        "chart_family": str(chart_family),
        "chart_score_type": "basis_coordinate_gaussian_sse",
        "chart_score_normalization": "per_table_basis_coordinate_sd",
        "n_candidates": int(n_candidates),
        "n_trajectories": int(n_traj),
        "n_timebins": int(n_time),
        "n_units": int(n_units),
        "basis_coordinate_dim": int(prior_coords.shape[-1]),
        "template_count_true_candidate": int(template_counts[true_idx]),
        "scale_min": float(np.min(scale)),
        "scale_median": float(np.median(scale)),
        "scale_max": float(np.max(scale)),
        "true_candidate_index": int(true_idx),
        "true_image_id": str(candidate_ids[true_idx]),
        "chart_pred_candidate_index": int(pred),
        "chart_pred_image_id": candidate_ids[pred] if 0 <= pred < len(candidate_ids) else "",
        "chart_correct": bool(pred == true_idx) if pred >= 0 else False,
        "chart_true_rank": rank_desc(candidate_scores, true_idx),
        "chart_true_margin": true_margin(candidate_scores, true_idx),
        "chart_true_score": float(candidate_scores[true_idx]),
        "chart_best_single_template_true_score": float(best_template_scores[true_idx]),
        "chart_joint_minus_best_template_true_gap": float(candidate_scores[true_idx] - best_template_scores[true_idx]),
        "candidate_posterior_true_mass": float(posterior[true_idx]) if np.isfinite(posterior).all() else float("nan"),
        "candidate_posterior_entropy": entropy(posterior),
        "candidate_posterior_N_eff": effective_count(posterior),
        "candidate_posterior_N_eff_fraction": (
            float(effective_count(posterior) / n_candidates)
            if np.isfinite(effective_count(posterior))
            else float("nan")
        ),
        "true_candidate_template_posterior_entropy": entropy(template_posterior),
        "true_candidate_template_posterior_N_eff": effective_count(template_posterior),
        "true_candidate_template_posterior_N_eff_fraction": (
            float(effective_count(template_posterior) / true_template_scores.size)
            if np.isfinite(effective_count(template_posterior))
            else float("nan")
        ),
        "top_template_index_true_candidate": int(top_template),
        "nearest_tau_rank_true_candidate_templates": nearest_tau_rank,
        "true_tau_rank_true_candidate_templates": true_tau_rank,
    }


def _basis_entries(
    *,
    basis_types: list[str],
    compact_basis: np.ndarray,
    static_basis: np.ndarray | None,
    static_basis_fit_scope: str,
    static_pc_fold: int,
    n_units: int,
    requested_k: int,
    k_index: int,
    n_random: int,
    seed: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    wanted = set(basis_types)
    k = int(requested_k)
    if "compact" in wanted:
        entries.append(
            {
                "basis_type": "compact",
                "basis_control": "observed_compact",
                "basis_fit_scope": "provided_compact_basis",
                "static_pc_fold": -1,
                "random_seed_or_null_id": -1,
                "requested_k_dim": k,
                "effective_k_dim": k,
                "basis": compact_basis[:, :k],
            }
        )
    if "unit_shuffle_compact" in wanted:
        shuffled, _perm = _unit_shuffle_basis(compact_basis[:, :k], np.random.default_rng(int(seed) + 10_000 + k))
        entries.append(
            {
                "basis_type": "unit_shuffle_compact",
                "basis_control": "unit_shuffle_compact",
                "basis_fit_scope": "derived_from_provided_compact_basis",
                "static_pc_fold": -1,
                "random_seed_or_null_id": -1,
                "requested_k_dim": k,
                "effective_k_dim": k,
                "basis": shuffled,
            }
        )
    if "random_k" in wanted:
        for null_id in range(max(0, int(n_random))):
            rng = np.random.default_rng(int(seed) + 100_000 * k + null_id)
            entries.append(
                {
                    "basis_type": "random_k",
                    "basis_control": "random_orthonormal",
                    "basis_fit_scope": "synthetic_random",
                    "static_pc_fold": -1,
                    "random_seed_or_null_id": int(null_id),
                    "requested_k_dim": k,
                    "effective_k_dim": k,
                    "basis": _random_basis(n_units, k, rng),
                }
            )
    if "gain_axis" in wanted and int(k_index) == 0:
        gain = np.ones((int(n_units), 1), dtype=np.float64)
        gain /= np.linalg.norm(gain)
        entries.append(
            {
                "basis_type": "gain_axis",
                "basis_control": "global_rate_axis",
                "basis_fit_scope": "analytic_global_rate",
                "static_pc_fold": -1,
                "random_seed_or_null_id": -1,
                "requested_k_dim": k,
                "effective_k_dim": 1,
                "basis": gain,
            }
        )
    if "static_pc_k" in wanted:
        if static_basis is None:
            raise ValueError("static_pc_k requested but static basis was not built")
        entries.append(
            {
                "basis_type": "static_pc_k",
                "basis_control": "static_response_pc",
                "basis_fit_scope": str(static_basis_fit_scope),
                "static_pc_fold": int(static_pc_fold),
                "random_seed_or_null_id": -1,
                "requested_k_dim": k,
                "effective_k_dim": k,
                "basis": static_basis[:, :k],
            }
        )
    return entries


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = [
        "candidate_set_mode",
        "observation_family",
        "prior_family",
        "motion_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_slot_alignment_status",
        "trajectory_slots_aligned",
        "basis_type",
        "basis_control",
        "basis_fit_scope",
        "basis_mode",
        "requested_k_dim",
        "effective_k_dim",
        "chart_family",
    ]
    out: list[dict[str, Any]] = []
    for key, grp in df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n_rows": int(len(grp)),
                "n_trials": int(grp["trial_id"].nunique()),
                "chart_accuracy": float(grp["chart_correct"].mean()),
                "mean_true_score": float(grp["chart_true_score"].mean()),
                "median_true_score": float(grp["chart_true_score"].median()),
                "mean_true_margin": float(grp["chart_true_margin"].mean()),
                "median_true_margin": float(grp["chart_true_margin"].median()),
                "mean_true_rank": float(grp["chart_true_rank"].mean()),
                "median_true_rank": float(grp["chart_true_rank"].median()),
                "mean_candidate_posterior_true_mass": float(grp["candidate_posterior_true_mass"].mean()),
                "median_candidate_posterior_true_mass": float(grp["candidate_posterior_true_mass"].median()),
                "median_candidate_posterior_N_eff_fraction": float(grp["candidate_posterior_N_eff_fraction"].median()),
                "median_template_N_eff_fraction": float(grp["true_candidate_template_posterior_N_eff_fraction"].median()),
                "median_nearest_tau_rank": float(grp["nearest_tau_rank_true_candidate_templates"].median()),
                "min_trajectory_slot_aligned_fraction": float(grp["trajectory_slot_aligned_fraction"].min()),
                "max_trajectory_slot_mismatch_count": int(grp["trajectory_slot_mismatch_count"].max()),
            }
        )
        out.append(row)
    return out


def _contrast_rows(
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    confidence: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    index_cols = [
        "table_index",
        "trial_id",
        "candidate_set_mode",
        "observation_family",
        "prior_family",
        "motion_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_slot_alignment_status",
        "trajectory_slots_aligned",
        "basis_type",
        "basis_control",
        "basis_fit_scope",
        "basis_mode",
        "requested_k_dim",
        "effective_k_dim",
        "random_seed_or_null_id",
    ]
    value_cols = list(SUMMARY_METRICS)
    duplicate_key = index_cols + ["chart_family"]
    if bool(df.duplicated(duplicate_key, keep=False).any()):
        sample = df.loc[df.duplicated(duplicate_key, keep=False), duplicate_key].head(5).to_dict("records")
        raise ValueError(f"Duplicate chart-swap rows for contrast key: {sample}")
    wide = df.pivot(index=index_cols, columns="chart_family", values=value_cols).reset_index()
    wide.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in wide.columns
    ]
    pairs = [
        ("correct_chart", "wrong_chart_roll"),
        ("correct_chart", "wrong_chart_pool"),
        ("correct_chart", "global_chart"),
        ("correct_chart", "static_no_motion_chart"),
    ]
    per_table: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        base = {col: row[col] for col in index_cols if col in row.index}
        for lhs, rhs in pairs:
            if f"chart_true_score_{lhs}" not in row.index or f"chart_true_score_{rhs}" not in row.index:
                continue
            out = dict(base)
            out["lhs_chart_family"] = lhs
            out["rhs_chart_family"] = rhs
            for metric in value_cols:
                left = float(row.get(f"{metric}_{lhs}", np.nan))
                right = float(row.get(f"{metric}_{rhs}", np.nan))
                out[f"{metric}_lhs_minus_rhs"] = left - right
            per_table.append(out)
    if not per_table:
        return []
    frame = pd.DataFrame(per_table)
    metric_cols = {f"{metric}_lhs_minus_rhs" for metric in value_cols}
    group_cols = [
        col
        for col in frame.columns
        if col not in metric_cols and col not in {"table_index", "trial_id", "random_seed_or_null_id"}
    ]
    out_rows: list[dict[str, Any]] = []
    for key, grp in frame.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row["n_rows"] = int(len(grp))
        row["n_trials"] = int(grp["trial_id"].nunique())
        for metric_col in sorted(metric_cols):
            vals = pd.to_numeric(grp[metric_col], errors="coerce")
            finite = vals[np.isfinite(vals)]
            row[f"mean_{metric_col}"] = float(finite.mean()) if len(finite) else float("nan")
            row[f"median_{metric_col}"] = float(finite.median()) if len(finite) else float("nan")
            row[f"fraction_positive_{metric_col}"] = float((finite > 0.0).mean()) if len(finite) else float("nan")
            _add_uncertainty_fields(
                row,
                prefix=f"mean_{metric_col}",
                values=finite,
                rng=rng,
                n_bootstrap=int(n_bootstrap),
                n_permutations=int(n_permutations),
                confidence=float(confidence),
            )
        out_rows.append(row)
    return out_rows


def _write_report(out_dir: Path, summary: list[dict[str, Any]], contrasts: list[dict[str, Any]], basis_meta: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Compact Chart-Swap Analysis",
        "",
        "This is a cache-only projected-residual chart test. It scores candidate images by matching",
        "`B^T (y_obs - zero_i)` against chart families derived from `B^T (prior_i,k - zero_i)`.",
        "`static_no_motion_chart` is a candidate-specific static-baseline residual, not a no-information baseline.",
        "`static_pc_k` defaults to a candidate-set fold-disjoint basis; check `basis_fit_scope` for the exact fit scope.",
        "",
        "## Basis",
        "",
        f"- path: `{basis_meta['basis_path']}`",
        f"- key: `{basis_meta['basis_key']}`",
        f"- shape: `{basis_meta['basis_shape']}`",
        f"- basis_mode: `{args.basis_mode}`",
        f"- orthonormalized: `{basis_meta['orthonormalized']}`",
        "",
        "## Summary Preview",
        "",
    ]
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        cols = [
            "candidate_set_mode",
            "prior_family",
            "motion_scale",
            "basis_type",
            "effective_k_dim",
            "basis_fit_scope",
            "chart_family",
            "n_rows",
            "chart_accuracy",
            "median_true_margin",
            "median_candidate_posterior_true_mass",
        ]
        lines.append("```text")
        lines.append(summary_df[cols].head(80).to_csv(index=False).strip())
        lines.append("```")
        lines.append("")
    lines.append("## Correct-Chart Contrasts")
    lines.append("")
    contrast_df = pd.DataFrame(contrasts)
    if not contrast_df.empty:
        score_col = "mean_chart_true_score_lhs_minus_rhs"
        cols = [
            "candidate_set_mode",
            "prior_family",
            "motion_scale",
            "basis_type",
            "effective_k_dim",
            "basis_fit_scope",
            "lhs_chart_family",
            "rhs_chart_family",
            "n_rows",
            score_col,
        ]
        cols = [col for col in cols if col in contrast_df.columns]
        lines.append("```text")
        lines.append(contrast_df[cols].head(80).to_csv(index=False).strip())
        lines.append("```")
    (out_dir / "compact_chart_swap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> Path:
    base = Path(args.base_run_dir)
    out_dir = Path(args.output_dir) if args.output_dir is not None else base / "compact_chart_swap_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_raw = pd.read_csv(base / "response_cache_manifest.csv")
    manifest, skipped = _filter_response_cache_manifest(manifest_raw, base)
    manifest = _filter_manifest(manifest, args)
    if manifest.empty:
        raise ValueError("No response tables selected after filters")
    _progress(
        f"selected response tables={len(manifest)}; skipped_missing_cache_rows={skipped}; "
        f"k_dims={args.k_dims}; basis_types={args.basis_types}; chart_families={args.chart_families}"
    )

    first = _load_npz_table(base / str(manifest.iloc[0]["response_cache_path"]))
    n_units = int(np.asarray(first["prior_lambda_counts"]).shape[-1])
    k_dims = _parse_int_list(str(args.k_dims))
    if not k_dims:
        raise ValueError("--k-dims must contain at least one value")
    if min(k_dims) <= 0:
        raise ValueError("--k-dims entries must be positive")
    max_k = max(k_dims)
    compact_basis, basis_meta = _load_basis(Path(args.compact_basis_path), n_units=n_units, basis_key=str(args.basis_key))
    _validate_basis_mode(args, basis_meta)
    if compact_basis.shape[1] < max_k:
        raise ValueError(f"Basis has only {compact_basis.shape[1]} columns, but max requested k={max_k}")

    basis_types = _parse_list(str(args.basis_types))
    unknown_basis = sorted(set(basis_types).difference(BASIS_TYPES))
    if unknown_basis:
        raise ValueError(f"Unsupported basis types: {unknown_basis}")
    chart_families = []
    for family in _parse_list(str(args.chart_families)):
        canonical = _canonical_chart_family(family)
        if canonical not in chart_families:
            chart_families.append(canonical)
    unknown_charts = sorted(set(chart_families).difference(CHART_FAMILIES))
    if unknown_charts:
        raise ValueError(f"Unsupported chart families: {unknown_charts}")
    slot_alignment_required = bool(set(chart_families).intersection(SLOT_ALIGNED_CHART_FAMILIES))

    static_pc_context = None
    if "static_pc_k" in set(basis_types):
        _progress(f"building static response PC basis ({args.static_pc_scope})")
        static_pc_context = _build_static_pc_context(
            manifest=manifest,
            base=base,
            n_units=n_units,
            k_max=max_k,
            scope=str(args.static_pc_scope),
            n_folds=int(args.static_pc_folds),
            seed=int(args.seed),
        )
        _progress(
            "static PC context ready: "
            f"scope={static_pc_context['scope']} "
            f"unique_candidate_sets={static_pc_context['n_unique_candidate_sets']} "
            f"folds={static_pc_context['n_folds']}"
        )

    rows: list[dict[str, Any]] = []
    for progress_i, (table_index, man_row) in enumerate(manifest.iterrows(), start=1):
        if progress_i == 1 or progress_i % int(args.progress_every) == 0 or progress_i == len(manifest):
            _progress(f"scoring table {progress_i}/{len(manifest)}")
        table_path = base / str(man_row["response_cache_path"])
        table = _load_npz_table(table_path)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        y_obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        prior_delta = prior - zero[:, None, :, :]
        obs_delta_by_candidate = y_obs[None, :, :] - zero
        table_n_units = int(prior.shape[-1])
        true_idx = _scalar_int(table, "true_candidate_index", 0)
        true_tau = _scalar_int(table, "true_trajectory_index", -1)
        nearest_tau = _scalar_int(table, "nearest_trajectory_index", _safe_int(man_row.get("nearest_trajectory_index", -1), -1))
        nearest_distance = _scalar_float_from_table_or_row(table, man_row, "nearest_trajectory_distance")
        candidate_ids = _candidate_ids(table, prior.shape[0])
        static_basis, static_basis_fit_scope, static_pc_fold, static_candidate_set_key = _static_basis_for_table(
            static_pc_context,
            int(table_index),
        )
        trajectory_alignment = _trajectory_slot_alignment(
            table,
            n_candidates=prior.shape[0],
            n_trajectories=prior.shape[1],
        )
        if slot_alignment_required and not bool(trajectory_alignment["trajectory_slots_aligned"]):
            if not bool(args.allow_unaligned_trajectory_slots):
                raise ValueError(
                    "Slot-aligned chart families require candidate-aligned prior_trajectory_ids; "
                    f"{man_row['response_cache_path']} has "
                    f"status={trajectory_alignment['trajectory_slot_alignment_status']} "
                    f"mismatches={trajectory_alignment['trajectory_slot_mismatch_count']}. "
                    "Use --allow-unaligned-trajectory-slots only if this confound is acceptable."
                )

        for k_index, k in enumerate(k_dims):
            entries = _basis_entries(
                basis_types=basis_types,
                compact_basis=compact_basis,
                static_basis=static_basis,
                static_basis_fit_scope=static_basis_fit_scope,
                static_pc_fold=static_pc_fold,
                n_units=n_units,
                requested_k=k,
                k_index=k_index,
                n_random=int(args.n_random),
                seed=int(args.seed),
            )
            for entry in entries:
                basis = np.asarray(entry["basis"], dtype=np.float64)
                prior_coords = _project_coords(prior_delta, basis)
                obs_coords = _project_coords(obs_delta_by_candidate, basis)
                scale = _coord_scale(prior_coords, min_scale=float(args.scale_floor))
                for chart_family in chart_families:
                    score = _score_chart_family_from_coords(
                        prior_coords=prior_coords,
                        obs_coords=obs_coords,
                        scale=scale,
                        chart_family=chart_family,
                        true_candidate_index=true_idx,
                        candidate_ids=candidate_ids,
                        nearest_trajectory_index=nearest_tau,
                        true_trajectory_index=true_tau,
                        n_units=table_n_units,
                    )
                    rows.append(
                        {
                            "table_index": int(table_index),
                            "trial_id": int(man_row["trial_id"]),
                            "candidate_set_mode": str(man_row["candidate_set_mode"]),
                            "observation_family": str(man_row.get("observation_family", "")),
                            "prior_family": str(man_row.get("prior_family", "")),
                            "motion_scale": float(man_row.get("scale", np.nan)),
                            "axis_catalog_mode": str(man_row.get("axis_catalog_mode", "")),
                            "axis_shared_source_catalog": _as_bool(man_row.get("axis_shared_source_catalog", False)),
                            "response_cache_path": str(man_row["response_cache_path"]),
                            "nearest_tau_distance": nearest_distance,
                            "basis_path": str(args.compact_basis_path),
                            "basis_key": str(basis_meta["basis_key"]),
                            "basis_mode": str(args.basis_mode),
                            "basis_type": str(entry["basis_type"]),
                            "basis_control": str(entry["basis_control"]),
                            "basis_fit_scope": str(entry["basis_fit_scope"]),
                            "static_pc_fold": int(entry["static_pc_fold"]),
                            "static_candidate_set_key": str(static_candidate_set_key),
                            "requested_k_dim": int(entry["requested_k_dim"]),
                            "effective_k_dim": int(entry["effective_k_dim"]),
                            "random_seed_or_null_id": int(entry["random_seed_or_null_id"]),
                            **trajectory_alignment,
                            **score,
                        }
                    )

    summary = _summary_rows(rows)
    contrasts = _contrast_rows(
        rows,
        rng=np.random.default_rng(int(args.seed) + 910_003),
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        confidence=float(args.uncertainty_confidence),
    )
    _write_csv(out_dir / "compact_chart_swap_trials.csv", rows)
    _write_csv(out_dir / "compact_chart_swap_summary.csv", summary)
    _write_csv(out_dir / "compact_chart_swap_contrasts.csv", contrasts)
    _write_json(
        out_dir / "compact_chart_swap_run_metadata.json",
        {
            "base_run_dir": str(base),
            "n_selected_tables": int(len(manifest)),
            "basis": basis_meta,
            "basis_mode": str(args.basis_mode),
            "image_disjoint_basis_verified": bool(
                str(args.basis_mode) == "image_disjoint" and basis_meta.get("declares_image_disjoint", False)
            ),
            "allow_unverified_image_disjoint_basis": bool(args.allow_unverified_image_disjoint_basis),
            "config": vars(args),
            "slot_alignment_required": slot_alignment_required,
            "static_pc_context": {
                key: value
                for key, value in (static_pc_context or {}).items()
                if key not in {"basis", "basis_by_fold", "fold_by_key", "table_key_by_index"}
            },
            "analysis_note": (
                "Candidate scores are log-mean Gaussian negative SSEs in basis-coordinate residual space. "
                "This tests content-conditioned chart likelihoods, not a continuous eye-motion prior."
            ),
        },
    )
    _write_report(out_dir, summary, contrasts, basis_meta, args)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--compact-basis-path", type=Path, required=True)
    parser.add_argument("--basis-key", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--basis-mode", default="global")
    parser.add_argument("--allow-unverified-image-disjoint-basis", action="store_true")
    parser.add_argument("--k-dims", default="10")
    parser.add_argument("--basis-types", default="compact,unit_shuffle_compact,gain_axis,random_k")
    parser.add_argument(
        "--chart-families",
        default="correct_chart,wrong_chart_roll,wrong_chart_pool,global_chart,static_no_motion_chart",
    )
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--static-pc-scope", default="candidate_set_fold_disjoint", choices=["selected_tables", "candidate_set_fold_disjoint"])
    parser.add_argument("--static-pc-folds", type=int, default=5)
    parser.add_argument("--candidate-set-modes", default="")
    parser.add_argument("--motion-scales", default="")
    parser.add_argument("--priors", default="")
    parser.add_argument("--scale-floor", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--allow-unaligned-trajectory-slots", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--uncertainty-confidence", type=float, default=0.95)
    return parser


def main() -> None:
    out = analyze(build_parser().parse_args())
    print(f"Wrote compact chart-swap analysis to {out}")


if __name__ == "__main__":
    main()
