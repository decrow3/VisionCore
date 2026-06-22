"""Cache-only compact-aware trajectory prior analysis for BackImage tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .analyze_compact_mechanism import (
        _load_basis,
        _project_delta,
        _random_basis,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from .analyze_feature_posterior import (
        _auto_likelihood_scales,
        _candidate_set_lookup,
        _candidate_window_indices,
        _filter_manifest,
        _filter_response_cache_manifest,
        _fit_feature_spaces,
        _load_npz,
        _load_observer_trial_metadata,
        _load_or_compute_latents,
        _mode_row,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _safe_bool,
    )
    from .likelihood import effective_count, entropy, normalized_log_weights
    from .observer import score_image_identity_score_vectors
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.backimage_trajectory_observer.analyze_compact_mechanism import (
        _load_basis,
        _project_delta,
        _random_basis,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from declan.backimage_trajectory_observer.analyze_feature_posterior import (
        _auto_likelihood_scales,
        _candidate_set_lookup,
        _candidate_window_indices,
        _filter_manifest,
        _filter_response_cache_manifest,
        _fit_feature_spaces,
        _load_npz,
        _load_observer_trial_metadata,
        _load_or_compute_latents,
        _mode_row,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _safe_bool,
    )
    from declan.backimage_trajectory_observer.likelihood import effective_count, entropy, normalized_log_weights
    from declan.backimage_trajectory_observer.observer import score_image_identity_score_vectors


OBSERVER_MODES = ("known", "zero", "joint", "best_single_tau", "motion_delta")
PRIOR_FAMILIES = (
    "uniform_base",
    "image_independent_compact_prior",
    "candidate_conditioned_compact_weight",
    "random_subspace_aware",
    "candidate_conditioned_random_subspace_weight",
    "unit_shuffle_compact_aware",
    "candidate_conditioned_unit_shuffle_compact_weight",
    "gain_axis_aware",
    "candidate_conditioned_gain_axis_weight",
    "static_pc_aware",
    "candidate_conditioned_static_pc_weight",
    "inverse_compact_control",
    "candidate_conditioned_inverse_compact_control",
)


def _progress(message: str) -> None:
    print(f"[compact-aware-prior] {message}", flush=True)


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


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(int(n_candidates))]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _zscore(values: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.mean(arr, axis=axis, keepdims=True)
    sd = np.std(arr, axis=axis, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    out = (arr - mean) / sd
    return np.where(np.isfinite(out), out, 0.0)


def _compact_leakage(prior_lambda_counts: np.ndarray, zero_lambda_counts: np.ndarray, basis: np.ndarray, *, eps: float) -> np.ndarray:
    """Return non-basis leakage fraction for each candidate and trajectory."""
    prior = np.asarray(prior_lambda_counts, dtype=np.float64)
    zero = np.asarray(zero_lambda_counts, dtype=np.float64)
    if prior.ndim != 4:
        raise ValueError(f"prior_lambda_counts must be 4D, got {prior.shape}")
    if zero.shape != (prior.shape[0], prior.shape[2], prior.shape[3]):
        raise ValueError(f"zero_lambda_counts shape {zero.shape} does not match prior table {prior.shape}")
    delta = prior - zero[:, None, :, :]
    projected = _project_delta(delta, np.asarray(basis, dtype=np.float64))
    residual = delta - projected
    residual_energy = np.sum(residual * residual, axis=(-2, -1))
    total_energy = np.sum(delta * delta, axis=(-2, -1))
    return residual_energy / (total_energy + float(eps))


def _stable_trajectory_key(value: Any) -> str:
    text = str(value)
    parts = text.split(":")
    if len(parts) >= 2:
        return ":".join(parts[:-1])
    return text


def _trajectory_key_matrix(table: dict[str, np.ndarray], *, n_candidates: int, n_trajectories: int) -> np.ndarray:
    if "prior_trajectory_ids" not in table:
        slots = np.asarray([f"trajectory_slot:{idx}" for idx in range(int(n_trajectories))], dtype=object)
        return np.broadcast_to(slots[None, :], (int(n_candidates), int(n_trajectories))).copy()
    ids = np.asarray(table["prior_trajectory_ids"])
    if ids.shape == (int(n_trajectories),):
        ids = np.broadcast_to(ids[None, :], (int(n_candidates), int(n_trajectories))).copy()
    if ids.shape != (int(n_candidates), int(n_trajectories)):
        raise ValueError(
            "prior_trajectory_ids must have shape "
            f"{(int(n_trajectories),)} or {(int(n_candidates), int(n_trajectories))}, got {ids.shape}"
        )
    out = np.empty(ids.shape, dtype=object)
    for index, value in np.ndenumerate(ids):
        out[index] = _stable_trajectory_key(value)
    return out


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, int, int, bool]:
    return (
        str(entry["family"]),
        str(entry["basis_type"]),
        int(entry["effective_k_dim"]),
        int(entry["null_id"]),
        bool(entry["inverse"]),
    )


def _new_pool() -> dict[str, Any]:
    return {
        "sum": {},
        "count": {},
        "table_sum": {},
        "table_count": {},
        "n_tables": 0,
    }


def _add_pool_value(pool: dict[str, Any], *, table_key: str, trajectory_key: str, value: float) -> None:
    if not np.isfinite(value):
        return
    key = str(trajectory_key)
    val = float(value)
    pool["sum"][key] = float(pool["sum"].get(key, 0.0)) + val
    pool["count"][key] = int(pool["count"].get(key, 0)) + 1
    table_sum = pool["table_sum"].setdefault(str(table_key), {})
    table_count = pool["table_count"].setdefault(str(table_key), {})
    table_sum[key] = float(table_sum.get(key, 0.0)) + val
    table_count[key] = int(table_count.get(key, 0)) + 1


def _pool_raw_for_table(
    pool: dict[str, Any],
    *,
    table_key: str,
    trajectory_keys: np.ndarray,
    inverse: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    keys = np.asarray(trajectory_keys, dtype=object)
    if keys.ndim != 2:
        raise ValueError(f"trajectory_keys must be (candidate, trajectory), got {keys.shape}")
    n_traj = int(keys.shape[1])
    global_sum: dict[str, float] = pool["sum"]
    global_count: dict[str, int] = pool["count"]
    table_sum: dict[str, float] = pool["table_sum"].get(str(table_key), {})
    table_count: dict[str, int] = pool["table_count"].get(str(table_key), {})
    outside_sum = float(sum(float(v) for v in global_sum.values())) - float(sum(float(v) for v in table_sum.values()))
    outside_count = int(sum(int(v) for v in global_count.values())) - int(sum(int(v) for v in table_count.values()))
    if outside_count > 0:
        fallback = outside_sum / float(outside_count)
    else:
        global_means = [
            float(global_sum[key]) / float(global_count[key])
            for key in global_sum
            if int(global_count.get(key, 0)) > 0
        ]
        fallback = float(np.mean(global_means)) if global_means else 0.0
    raw_mean = np.empty(n_traj, dtype=np.float64)
    matched_slots = 0
    nonmatching_fallback = 0
    missing_fallback = 0
    for traj_i in range(n_traj):
        values = []
        for key in sorted(set(str(v) for v in keys[:, traj_i].tolist())):
            total_count = int(global_count.get(key, 0))
            total_sum = float(global_sum.get(key, 0.0))
            local_count = int(table_count.get(key, 0))
            local_sum = float(table_sum.get(key, 0.0))
            outside_count = total_count - local_count
            if outside_count > 0:
                matched_slots += 1
                values.append((total_sum - local_sum) / float(outside_count))
            elif total_count > 0:
                nonmatching_fallback += 1
                values.append(fallback)
            else:
                missing_fallback += 1
                values.append(fallback)
        raw_mean[traj_i] = float(np.mean(values)) if values else fallback
    sign = 1.0 if bool(inverse) else -1.0
    total_slots = int(matched_slots + nonmatching_fallback + missing_fallback)
    fallback_slots = int(nonmatching_fallback + missing_fallback)
    return sign * _zscore(raw_mean, axis=None), {
        "shared_prior_key_scope": "hash_stripped_stable_trajectory_key",
        "shared_prior_matched_stable_keys": int(matched_slots),
        "shared_prior_total_stable_keys": int(total_slots),
        "shared_prior_stable_key_fallback_fraction": float(fallback_slots / total_slots) if total_slots > 0 else 0.0,
        "shared_prior_nonmatching_fallback_stable_keys": int(nonmatching_fallback),
        "shared_prior_missing_fallback_stable_keys": int(missing_fallback),
        # Backward-compatible names retained for existing readers. These are
        # stable-key counts after hash stripping, not exact trajectory counts.
        "shared_prior_matched_slots": int(matched_slots),
        "shared_prior_total_slots": int(total_slots),
        "shared_prior_fallback_fraction": float(fallback_slots / total_slots) if total_slots > 0 else 0.0,
        "shared_prior_nonmatching_fallback_slots": int(nonmatching_fallback),
        "shared_prior_missing_fallback_slots": int(missing_fallback),
    }


def _raw_weight_from_leakage(leakage: np.ndarray, *, shape_kind: str, inverse: bool) -> np.ndarray:
    sign = 1.0 if bool(inverse) else -1.0
    rho = np.asarray(leakage, dtype=np.float64)
    if shape_kind == "image_independent":
        return sign * _zscore(np.mean(rho, axis=0), axis=None)
    if shape_kind == "candidate_conditioned":
        return sign * _zscore(rho, axis=1)
    raise ValueError(f"Unsupported prior shape kind {shape_kind!r}")


def _prior_probability_diagnostics(log_prior: np.ndarray | None, *, n_candidates: int, n_trajectories: int) -> dict[str, Any]:
    norm = normalized_log_weights(log_prior, int(n_trajectories), n_rows=int(n_candidates))
    probs = np.exp(norm)
    if probs.ndim == 1:
        entropies = np.asarray([entropy(probs)], dtype=np.float64)
        neff = np.asarray([effective_count(probs)], dtype=np.float64)
        shape = f"({int(n_trajectories)},)"
        min_log = float(np.min(norm))
        max_log = float(np.max(norm))
    else:
        entropies = np.asarray([entropy(row) for row in probs], dtype=np.float64)
        neff = np.asarray([effective_count(row) for row in probs], dtype=np.float64)
        shape = f"({int(n_candidates)},{int(n_trajectories)})"
        min_log = float(np.min(norm))
        max_log = float(np.max(norm))
    return {
        "log_prior_shape": shape,
        "prior_entropy_mean": float(np.mean(entropies)),
        "prior_entropy_median": float(np.median(entropies)),
        "prior_N_eff_mean": float(np.mean(neff)),
        "prior_N_eff_median": float(np.median(neff)),
        "prior_N_eff_fraction_mean": float(np.mean(neff) / float(n_trajectories)),
        "prior_N_eff_fraction_median": float(np.median(neff) / float(n_trajectories)),
        "prior_log_weight_min": min_log,
        "prior_log_weight_max": max_log,
    }


def _mean_neff_fraction_for_beta(raw: np.ndarray, beta: float, *, n_candidates: int, n_trajectories: int) -> float:
    diag = _prior_probability_diagnostics(float(beta) * np.asarray(raw, dtype=np.float64), n_candidates=n_candidates, n_trajectories=n_trajectories)
    return float(diag["prior_N_eff_fraction_mean"])


def _beta_to_match_neff(raw: np.ndarray, target_neff_fraction: float, *, n_candidates: int, n_trajectories: int, beta_max: float) -> float:
    target = float(target_neff_fraction)
    if not np.isfinite(target):
        return 0.0
    if target >= 1.0 - 1e-9:
        return 0.0
    if target <= 1.0 / max(1, int(n_trajectories)):
        return float(beta_max)
    raw_arr = np.asarray(raw, dtype=np.float64)
    if raw_arr.size == 0 or float(np.std(raw_arr)) <= 1e-12:
        return 0.0
    hi = float(beta_max)
    hi_neff = _mean_neff_fraction_for_beta(raw_arr, hi, n_candidates=n_candidates, n_trajectories=n_trajectories)
    if hi_neff > target:
        return hi
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        mid_neff = _mean_neff_fraction_for_beta(raw_arr, mid, n_candidates=n_candidates, n_trajectories=n_trajectories)
        if mid_neff > target:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _family_config(family: str) -> dict[str, Any]:
    name = str(family)
    if name == "uniform_base":
        return {
            "basis_type": "none",
            "shape_kind": "shared_uniform",
            "inverse": False,
            "candidate_conditioned": False,
        }
    if name in {"image_independent_compact_prior", "candidate_conditioned_compact_weight", "inverse_compact_control", "candidate_conditioned_inverse_compact_control"}:
        return {
            "basis_type": "compact",
            "shape_kind": "candidate_conditioned" if name.startswith("candidate_conditioned") else "image_independent",
            "inverse": "inverse" in name,
            "candidate_conditioned": name.startswith("candidate_conditioned"),
        }
    mapping = {
        "random_subspace_aware": ("random", "image_independent"),
        "candidate_conditioned_random_subspace_weight": ("random", "candidate_conditioned"),
        "unit_shuffle_compact_aware": ("unit_shuffle_compact", "image_independent"),
        "candidate_conditioned_unit_shuffle_compact_weight": ("unit_shuffle_compact", "candidate_conditioned"),
        "gain_axis_aware": ("gain_ones", "image_independent"),
        "candidate_conditioned_gain_axis_weight": ("gain_ones", "candidate_conditioned"),
        "static_pc_aware": ("static_pc", "image_independent"),
        "candidate_conditioned_static_pc_weight": ("static_pc", "candidate_conditioned"),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported prior family {name!r}; supported={', '.join(PRIOR_FAMILIES)}")
    basis_type, shape_kind = mapping[name]
    return {
        "basis_type": basis_type,
        "shape_kind": shape_kind,
        "inverse": False,
        "candidate_conditioned": shape_kind == "candidate_conditioned",
    }


def _gain_basis(n_units: int) -> np.ndarray:
    vec = np.ones((int(n_units), 1), dtype=np.float64)
    return vec / float(np.linalg.norm(vec))


def _basis_entries_for_family(
    *,
    family: str,
    compact_u: np.ndarray,
    static_basis: np.ndarray | None,
    random_bases: dict[tuple[int, int], np.ndarray],
    n_units: int,
    k_dim: int,
    n_random: int,
    seed: int,
) -> list[dict[str, Any]]:
    cfg = _family_config(family)
    basis_type = str(cfg["basis_type"])
    if basis_type == "none":
        return [{**cfg, "family": family, "basis": None, "basis_type": basis_type, "effective_k_dim": 0, "null_id": -1}]
    if basis_type == "compact":
        return [{**cfg, "family": family, "basis": compact_u, "basis_type": basis_type, "effective_k_dim": int(k_dim), "null_id": -1}]
    if basis_type == "unit_shuffle_compact":
        basis, _perm = _unit_shuffle_basis(compact_u, np.random.default_rng(int(seed) + 10_000 + int(k_dim)))
        return [{**cfg, "family": family, "basis": basis, "basis_type": basis_type, "effective_k_dim": int(k_dim), "null_id": -1}]
    if basis_type == "gain_ones":
        return [{**cfg, "family": family, "basis": _gain_basis(n_units), "basis_type": basis_type, "effective_k_dim": 1, "null_id": -1}]
    if basis_type == "static_pc":
        if static_basis is None:
            raise ValueError(f"{family} requested but static PC basis was not built")
        return [{**cfg, "family": family, "basis": static_basis[:, : int(k_dim)], "basis_type": basis_type, "effective_k_dim": int(k_dim), "null_id": -1}]
    if basis_type == "random":
        out = []
        for null_id in range(max(1, int(n_random))):
            out.append(
                {
                    **cfg,
                    "family": family,
                    "basis": random_bases[(int(k_dim), int(null_id))],
                    "basis_type": basis_type,
                    "effective_k_dim": int(k_dim),
                    "null_id": int(null_id),
                }
            )
        return out
    raise ValueError(f"Unsupported basis_type={basis_type!r}")


def _build_prior_entries(
    *,
    prior_families: list[str],
    prior_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    compact_u: np.ndarray,
    static_basis: np.ndarray | None,
    random_bases: dict[tuple[int, int], np.ndarray],
    n_units: int,
    k_dim: int,
    n_random: int,
    seed: int,
    prior_beta: float,
    prior_beta_max: float,
    entropy_match_target: str,
    eps: float,
    trajectory_keys: np.ndarray | None = None,
    table_key: str = "",
    shared_prior_pools: dict[tuple[str, str, int, int, bool], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    n_candidates, n_trajectories = int(prior_lambda_counts.shape[0]), int(prior_lambda_counts.shape[1])
    entries: list[dict[str, Any]] = []
    for family in prior_families:
        for entry in _basis_entries_for_family(
            family=family,
            compact_u=compact_u,
            static_basis=static_basis,
            random_bases=random_bases,
            n_units=n_units,
            k_dim=k_dim,
            n_random=n_random,
            seed=seed,
        ):
            raw_source = "uniform"
            shared_diag = {
                "shared_prior_key_scope": "",
                "shared_prior_matched_stable_keys": 0,
                "shared_prior_total_stable_keys": 0,
                "shared_prior_stable_key_fallback_fraction": 0.0,
                "shared_prior_nonmatching_fallback_stable_keys": 0,
                "shared_prior_missing_fallback_stable_keys": 0,
                "shared_prior_matched_slots": 0,
                "shared_prior_total_slots": 0,
                "shared_prior_fallback_fraction": 0.0,
                "shared_prior_nonmatching_fallback_slots": 0,
                "shared_prior_missing_fallback_slots": 0,
            }
            if entry["basis"] is None:
                raw = None
            elif str(entry["shape_kind"]) == "image_independent" and shared_prior_pools is not None:
                if trajectory_keys is None:
                    raise ValueError("trajectory_keys are required for shared image-independent priors")
                pool_key = _entry_key(entry)
                if pool_key not in shared_prior_pools:
                    raise ValueError(f"No shared prior pool was built for {pool_key}")
                raw, shared_diag = _pool_raw_for_table(
                    shared_prior_pools[pool_key],
                    table_key=str(table_key),
                    trajectory_keys=trajectory_keys,
                    inverse=bool(entry["inverse"]),
                )
                raw_source = "selected_manifest_stable_trajectory_leave_one_table_out"
            else:
                leakage = _compact_leakage(
                    prior_lambda_counts,
                    zero_lambda_counts,
                    np.asarray(entry["basis"], dtype=np.float64),
                    eps=float(eps),
                )
                raw = _raw_weight_from_leakage(
                    leakage,
                    shape_kind=str(entry["shape_kind"]),
                    inverse=bool(entry["inverse"]),
                )
                raw_source = "current_candidate_table"
            entries.append({**entry, "raw": raw, "raw_source": raw_source, **shared_diag})

    target_name = str(entropy_match_target).strip()
    target_neff = float("nan")
    if target_name:
        if target_name == "uniform_base":
            raise ValueError(
                "entropy_match_target='uniform_base' would force every non-uniform prior to beta=0. "
                "Use an empty target for fixed beta, or match controls to a non-uniform family such as "
                "'image_independent_compact_prior'."
            )
        else:
            target_matches = [entry for entry in entries if str(entry["family"]) == target_name and entry["raw"] is not None]
            if not target_matches:
                raise ValueError(
                    f"entropy_match_target={target_name!r} was requested, but no non-uniform prior entry "
                    "with that family was built for this table"
                )
            target_neff = _mean_neff_fraction_for_beta(
                np.asarray(target_matches[0]["raw"], dtype=np.float64),
                float(prior_beta),
                n_candidates=n_candidates,
                n_trajectories=n_trajectories,
            )

    out: list[dict[str, Any]] = []
    for entry in entries:
        raw = entry["raw"]
        if raw is None:
            log_prior = None
            beta = 0.0
            matching = "uniform"
        elif target_name and str(entry["family"]) != target_name:
            beta = _beta_to_match_neff(
                np.asarray(raw, dtype=np.float64),
                target_neff,
                n_candidates=n_candidates,
                n_trajectories=n_trajectories,
                beta_max=float(prior_beta_max),
            )
            log_prior = beta * np.asarray(raw, dtype=np.float64)
            matching = f"matched_to:{target_name}"
        else:
            beta = float(prior_beta)
            log_prior = beta * np.asarray(raw, dtype=np.float64)
            matching = "fixed_beta" if not target_name else "entropy_target"
        diag = _prior_probability_diagnostics(log_prior, n_candidates=n_candidates, n_trajectories=n_trajectories)
        out.append({**entry, "raw": raw, "log_prior": log_prior, "beta": float(beta), "entropy_matching": matching, **diag})
    return out


def _build_shared_prior_pools(
    *,
    run_dir: Path,
    manifest: pd.DataFrame,
    prior_families: list[str],
    basis_full: np.ndarray,
    static_basis: np.ndarray | None,
    random_bases: dict[tuple[int, int], np.ndarray],
    n_units: int,
    k_dims: list[int],
    n_random: int,
    seed: int,
    eps: float,
) -> dict[tuple[str, str, int, int, bool], dict[str, Any]]:
    needs_pool = False
    for family in prior_families:
        cfg = _family_config(family)
        if str(cfg["shape_kind"]) == "image_independent" and str(cfg["basis_type"]) != "none":
            needs_pool = True
            break
    if not needs_pool:
        return {}

    pools: dict[tuple[str, str, int, int, bool], dict[str, Any]] = {}
    for _, man_row in tqdm(list(manifest.iterrows()), desc="shared prior pool"):
        table_key = str(man_row["response_cache_path"])
        table = _load_npz(run_dir / table_key)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        trajectory_keys = _trajectory_key_matrix(
            table,
            n_candidates=int(prior.shape[0]),
            n_trajectories=int(prior.shape[1]),
        )
        for k_dim in k_dims:
            compact_u = basis_full[:, : int(k_dim)]
            for family in prior_families:
                for entry in _basis_entries_for_family(
                    family=family,
                    compact_u=compact_u,
                    static_basis=static_basis,
                    random_bases=random_bases,
                    n_units=n_units,
                    k_dim=int(k_dim),
                    n_random=int(n_random),
                    seed=int(seed),
                ):
                    if entry["basis"] is None or str(entry["shape_kind"]) != "image_independent":
                        continue
                    leakage = _compact_leakage(
                        prior,
                        zero,
                        np.asarray(entry["basis"], dtype=np.float64),
                        eps=float(eps),
                    )
                    pool_key = _entry_key(entry)
                    pool = pools.setdefault(pool_key, _new_pool())
                    pool["n_tables"] = int(pool.get("n_tables", 0)) + 1
                    for index, value in np.ndenumerate(leakage):
                        _add_pool_value(
                            pool,
                            table_key=table_key,
                            trajectory_key=str(trajectory_keys[index]),
                            value=float(value),
                        )
    return pools


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
        "observer_mode",
        "trajectory_weight_family",
        "trajectory_weight_shape",
        "trajectory_weight_basis_type",
        "raw_weight_source",
        "basis_mode",
        "k_dim",
        "effective_k_dim",
        "random_seed_or_null_id",
    ]
    out: list[dict[str, Any]] = []
    for key, grp in df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row["n_trial_rows"] = int(len(grp))
        row["n_trials"] = int(grp["trial_id"].nunique())
        for col in [
            "prior_beta",
            "prior_N_eff_fraction_mean",
            "prior_N_eff_fraction_median",
            "prior_entropy_mean",
            "prior_entropy_median",
            "shared_prior_matched_stable_keys",
            "shared_prior_total_stable_keys",
            "shared_prior_stable_key_fallback_fraction",
            "shared_prior_nonmatching_fallback_stable_keys",
            "shared_prior_missing_fallback_stable_keys",
            "shared_prior_matched_slots",
            "shared_prior_total_slots",
            "shared_prior_fallback_fraction",
            "shared_prior_nonmatching_fallback_slots",
            "shared_prior_missing_fallback_slots",
            "candidate_posterior_N_eff_fraction",
            "candidate_posterior_true_mass",
            "feature_cosine",
            "feature_neg_mse",
            "feature_rmse",
            "score_true_rank",
            "score_true_margin",
        ]:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[f"mean_{col}"] = float(vals.mean())
            row[f"median_{col}"] = float(vals.median())
        out.append(row)
    return out


def _contrast_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not summary:
        return []
    df = pd.DataFrame(summary)
    metrics = [
        "mean_feature_neg_mse",
        "mean_feature_cosine",
        "mean_candidate_posterior_true_mass",
        "mean_candidate_posterior_N_eff_fraction",
        "mean_score_true_margin",
    ]
    context_cols = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
        "observer_mode",
        "basis_mode",
        "k_dim",
    ]
    out: list[dict[str, Any]] = []
    for key, grp in df.groupby(context_cols, dropna=False):
        uniform = grp[grp["trajectory_weight_family"].eq("uniform_base")]
        if uniform.empty:
            continue
        base = uniform.iloc[0]
        for _, row in grp[~grp["trajectory_weight_family"].eq("uniform_base")].iterrows():
            payload = {col: value for col, value in zip(context_cols, key, strict=True)}
            payload.update(
                {
                    "lhs_trajectory_weight_family": str(row["trajectory_weight_family"]),
                    "rhs_trajectory_weight_family": "uniform_base",
                    "lhs_trajectory_weight_shape": str(row["trajectory_weight_shape"]),
                    "lhs_trajectory_weight_basis_type": str(row["trajectory_weight_basis_type"]),
                    "lhs_random_seed_or_null_id": int(row["random_seed_or_null_id"]),
                    "n_trials": int(row["n_trials"]),
                }
            )
            for metric in metrics:
                payload[f"{metric}_lhs_minus_uniform"] = float(row[metric]) - float(base[metric])
            out.append(payload)
    return out


def _write_report(out_dir: Path, *, run_dir: Path, summary: list[dict[str, Any]], contrasts: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary_df = pd.DataFrame(summary)
    contrast_df = pd.DataFrame(contrasts)
    lines = [
        "# Compact-Aware Prior Analysis",
        "",
        "This cache-only analysis reweights the finite trajectory catalog using",
        "response-space leakage scores for compact and matched control subspaces,",
        "then reuses the existing joint observer and feature-posterior endpoint.",
        "",
        "## Inputs",
        "",
        f"- source run: `{run_dir}`",
        f"- compact basis: `{args.compact_basis_path}`",
        f"- basis mode: `{args.basis_mode}`",
        f"- k dimensions: `{args.k_dims}`",
        f"- trajectory weight families: `{args.prior_families}`",
        f"- beta: `{args.prior_beta}`",
        f"- entropy match target: `{args.entropy_match_target}`",
        "",
        "## Claim Boundary",
        "",
        "`image_independent_compact_prior` is a leave-one-table-out catalog-statistic",
        "trajectory reweighting comparison keyed by stable trajectory IDs. It is clean",
        "with respect to `y_obs_counts`, but it is not a universal biological",
        "eye-motion prior.",
        "`candidate_conditioned_compact_weight` is a geometry-aware proposal or",
        "marginalization weight; a win by that family alone is diagnostic, not",
        "evidence for a content-independent eye-motion prior.",
        "",
        "## Primary Files",
        "",
        "- `compact_aware_prior_trials.csv`",
        "- `compact_aware_prior_summary.csv`",
        "- `compact_aware_prior_contrasts.csv`",
        "- `compact_aware_prior_qc.csv`",
        "- `compact_aware_prior_metadata.json`",
        "",
    ]
    if not summary_df.empty:
        lines.extend(["## Summary Preview", "", "```text"])
        cols = [
            "observer_mode",
            "trajectory_weight_family",
            "trajectory_weight_shape",
            "observation_scale",
            "prior_family",
            "latent",
            "requested_k",
            "k_dim",
            "mean_feature_neg_mse",
            "mean_feature_cosine",
            "mean_prior_N_eff_fraction_mean",
            "n_trials",
        ]
        existing = [col for col in cols if col in summary_df.columns]
        lines.append(summary_df[existing].head(80).to_csv(index=False).strip())
        lines.append("```")
    if not contrast_df.empty:
        lines.extend(["", "## Contrast Preview", "", "```text"])
        preview = contrast_df[contrast_df["observer_mode"].eq("joint")]
        cols = [
            "lhs_trajectory_weight_family",
            "observation_scale",
            "prior_family",
            "latent",
            "requested_k",
            "mean_feature_neg_mse_lhs_minus_uniform",
            "mean_feature_cosine_lhs_minus_uniform",
            "mean_candidate_posterior_true_mass_lhs_minus_uniform",
        ]
        existing = [col for col in cols if col in preview.columns]
        lines.append(preview[existing].head(80).to_csv(index=False).strip())
        lines.append("```")
    (out_dir / "compact_aware_prior_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-npz", type=Path, default=None)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--latent-names", default="pyramid_local_field")
    parser.add_argument("--pca-k-list", default="8")
    parser.add_argument("--likelihood-scales", default="auto")
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--candidate-set-modes", default="")
    parser.add_argument("--priors", default="")
    parser.add_argument("--motion-scales", default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument("--local-field-grid", type=int, default=8)
    parser.add_argument("--trust-feature-row-order", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--compact-basis-path", type=Path, required=True)
    parser.add_argument("--basis-key", default="auto")
    parser.add_argument("--basis-mode", default="global")
    parser.add_argument("--allow-unverified-image-disjoint-basis", action="store_true")
    parser.add_argument("--k-dims", default="10")
    parser.add_argument(
        "--prior-families",
        default="uniform_base,image_independent_compact_prior,candidate_conditioned_compact_weight,random_subspace_aware,unit_shuffle_compact_aware,gain_axis_aware,static_pc_aware,inverse_compact_control",
    )
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--prior-beta-max", type=float, default=8.0)
    parser.add_argument("--entropy-match-target", default="")
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def analyze(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(run_dir / "selected_windows.csv")
    manifest = pd.read_csv(run_dir / "response_cache_manifest.csv")
    manifest = _filter_manifest(manifest, args)
    if manifest.empty:
        raise ValueError("No response cache rows remain after filtering")
    manifest, skipped_cache_rows = _filter_response_cache_manifest(manifest, run_dir)
    candidate_sets_path = run_dir / "candidate_sets.csv"
    candidate_sets = pd.read_csv(candidate_sets_path) if candidate_sets_path.exists() and candidate_sets_path.stat().st_size > 0 else pd.DataFrame()
    candidate_lookup = _candidate_set_lookup(candidate_sets)
    source_row_to_pos = {
        int(row["source_row"]): int(pos)
        for pos, row in windows.iterrows()
    } if "source_row" in windows.columns else {}

    first = _load_npz(run_dir / str(manifest.iloc[0]["response_cache_path"]))
    n_units = int(np.asarray(first["prior_lambda_counts"]).shape[-1])
    k_dims = _parse_int_list(args.k_dims)
    if not k_dims:
        raise ValueError("--k-dims must request at least one dimension")
    prior_families = _parse_str_list(args.prior_families)
    unknown = sorted(set(prior_families).difference(PRIOR_FAMILIES))
    if unknown:
        raise ValueError(f"Unsupported prior families: {unknown}")
    max_k = max(k_dims)
    basis_full, basis_meta = _load_basis(Path(args.compact_basis_path), n_units=n_units, basis_key=str(args.basis_key))
    _validate_basis_mode(args, basis_meta)
    if basis_full.shape[1] < max_k:
        raise ValueError(f"Basis has only {basis_full.shape[1]} columns, but max k={max_k}")

    static_basis = None
    if any(_family_config(fam)["basis_type"] == "static_pc" for fam in prior_families):
        _progress("building static response PC basis")
        zero_tables = []
        for _, row in manifest.iterrows():
            tab = _load_npz(run_dir / str(row["response_cache_path"]))
            zero_tables.append(np.asarray(tab["zero_lambda_counts"], dtype=np.float32))
        static_basis = _static_pc_basis(zero_tables, n_units=n_units, k_max=max_k)

    random_bases: dict[tuple[int, int], np.ndarray] = {}
    if any(_family_config(fam)["basis_type"] == "random" for fam in prior_families):
        for k_dim in k_dims:
            for null_id in range(max(1, int(args.n_random))):
                rng = np.random.default_rng(int(args.seed) + 100_000 * int(k_dim) + int(null_id))
                random_bases[(int(k_dim), int(null_id))] = _random_basis(n_units, int(k_dim), rng)

    shared_prior_pools = _build_shared_prior_pools(
        run_dir=run_dir,
        manifest=manifest,
        prior_families=prior_families,
        basis_full=basis_full,
        static_basis=static_basis,
        random_bases=random_bases,
        n_units=n_units,
        k_dims=k_dims,
        n_random=int(args.n_random),
        seed=int(args.seed),
        eps=float(args.eps),
    )

    feature_k_list = _parse_int_list(args.pca_k_list)
    likelihood_scales = _auto_likelihood_scales(run_dir, str(args.likelihood_scales))
    trial_metadata = _load_observer_trial_metadata(run_dir)
    latent_arrays, latent_qc, feature_source = _load_or_compute_latents(args, windows, out_dir)
    feature_spaces, feature_qc = _fit_feature_spaces(latent_arrays, feature_k_list)
    _progress(
        f"selected tables={manifest.shape[0]}; prior_families={','.join(prior_families)}; "
        f"k={','.join(str(k) for k in k_dims)}; feature_k={','.join(str(k) for k in feature_k_list)}"
    )

    trial_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = list(latent_qc) + list(feature_qc)
    progress_every = max(1, int(args.progress_every))
    manifest_items = list(manifest.iterrows())
    for progress_i, (table_index, man_row) in enumerate(tqdm(manifest_items, desc="compact-aware prior tables"), start=1):
        table_path = run_dir / str(man_row["response_cache_path"])
        table = _load_npz(table_path)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        y_obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        true_idx = _scalar_int(table, "true_candidate_index", 0)
        candidate_ids = _candidate_ids(table, prior.shape[0])
        trajectory_keys = _trajectory_key_matrix(
            table,
            n_candidates=int(prior.shape[0]),
            n_trajectories=int(prior.shape[1]),
        )
        candidate_indices, candidate_index_source = _candidate_window_indices(
            manifest_row=man_row,
            candidate_ids=candidate_ids,
            candidate_lookup=candidate_lookup,
            source_row_to_pos=source_row_to_pos,
            n_windows=int(windows.shape[0]),
        )
        qc_rows.append(
            {
                "qc_type": "candidate_alignment",
                "table_index": int(table_index),
                "trial_id": int(man_row["trial_id"]),
                "response_cache_path": str(man_row["response_cache_path"]),
                "candidate_set_mode": str(man_row["candidate_set_mode"]),
                "n_candidates": int(len(candidate_ids)),
                "candidate_index_source": candidate_index_source,
                "candidate_indices": ";".join(str(v) for v in candidate_indices),
                "candidate_ids": ";".join(candidate_ids),
            }
        )
        meta = trial_metadata.get(str(man_row["response_cache_path"]), {})
        for k_dim in k_dims:
            compact_u = basis_full[:, : int(k_dim)]
            prior_entries = _build_prior_entries(
                prior_families=prior_families,
                prior_lambda_counts=prior,
                zero_lambda_counts=zero,
                compact_u=compact_u,
                static_basis=static_basis,
                random_bases=random_bases,
                n_units=n_units,
                k_dim=int(k_dim),
                n_random=int(args.n_random),
                seed=int(args.seed),
                prior_beta=float(args.prior_beta),
                prior_beta_max=float(args.prior_beta_max),
                entropy_match_target=str(args.entropy_match_target),
                eps=float(args.eps),
                trajectory_keys=trajectory_keys,
                table_key=str(man_row["response_cache_path"]),
                shared_prior_pools=shared_prior_pools,
            )
            for prior_entry in prior_entries:
                qc_rows.append(
                    {
                        "qc_type": "trajectory_weight",
                        "table_index": int(table_index),
                        "trial_id": int(man_row["trial_id"]),
                        "response_cache_path": str(man_row["response_cache_path"]),
                        "k_dim": int(k_dim),
                        "trajectory_weight_family": str(prior_entry["family"]),
                        "trajectory_weight_shape": str(prior_entry["shape_kind"]),
                        "trajectory_weight_basis_type": str(prior_entry["basis_type"]),
                        "effective_k_dim": int(prior_entry["effective_k_dim"]),
                        "random_seed_or_null_id": int(prior_entry["null_id"]),
                        "prior_beta": float(prior_entry["beta"]),
                        "entropy_matching": str(prior_entry["entropy_matching"]),
                        "prior_N_eff_fraction_mean": float(prior_entry["prior_N_eff_fraction_mean"]),
                        "prior_N_eff_fraction_median": float(prior_entry["prior_N_eff_fraction_median"]),
                        "log_prior_shape": str(prior_entry["log_prior_shape"]),
                        "raw_weight_source": str(prior_entry["raw_source"]),
                        "shared_prior_key_scope": str(prior_entry["shared_prior_key_scope"]),
                        "shared_prior_matched_stable_keys": int(prior_entry["shared_prior_matched_stable_keys"]),
                        "shared_prior_total_stable_keys": int(prior_entry["shared_prior_total_stable_keys"]),
                        "shared_prior_stable_key_fallback_fraction": float(
                            prior_entry["shared_prior_stable_key_fallback_fraction"]
                        ),
                        "shared_prior_nonmatching_fallback_stable_keys": int(
                            prior_entry["shared_prior_nonmatching_fallback_stable_keys"]
                        ),
                        "shared_prior_missing_fallback_stable_keys": int(
                            prior_entry["shared_prior_missing_fallback_stable_keys"]
                        ),
                        "shared_prior_matched_slots": int(prior_entry["shared_prior_matched_slots"]),
                        "shared_prior_total_slots": int(prior_entry["shared_prior_total_slots"]),
                        "shared_prior_fallback_fraction": float(prior_entry["shared_prior_fallback_fraction"]),
                        "shared_prior_nonmatching_fallback_slots": int(prior_entry["shared_prior_nonmatching_fallback_slots"]),
                        "shared_prior_missing_fallback_slots": int(prior_entry["shared_prior_missing_fallback_slots"]),
                        "uses_y_obs_counts": False,
                    }
                )
                for likelihood_scale in likelihood_scales:
                    vectors = score_image_identity_score_vectors(
                        y_obs_counts=y_obs,
                        prior_lambda_counts=prior,
                        known_lambda_counts=known,
                        zero_lambda_counts=zero,
                        true_candidate_index=true_idx,
                        candidate_ids=candidate_ids,
                        log_trajectory_prior=prior_entry["log_prior"],
                        eps=float(args.eps),
                        likelihood_scale=float(likelihood_scale),
                    )
                    score_by_mode = {
                        "known": np.asarray(vectors["known_scores"], dtype=np.float64),
                        "zero": np.asarray(vectors["zero_scores"], dtype=np.float64),
                        "joint": np.asarray(vectors["joint_scores"], dtype=np.float64),
                        "best_single_tau": np.asarray(vectors["best_single_tau_scores"], dtype=np.float64),
                    }
                    score_by_mode["motion_delta"] = score_by_mode["joint"] - score_by_mode["zero"]
                    for (latent, k_requested), space in feature_spaces.items():
                        features_all = np.asarray(space["scores"], dtype=np.float64)
                        candidate_features = features_all[np.asarray(candidate_indices, dtype=int)]
                        z_true = candidate_features[int(true_idx)]
                        base_cols = {
                            "table_index": int(table_index),
                            "trial_id": int(man_row["trial_id"]),
                            "response_cache_path": str(man_row["response_cache_path"]),
                            "candidate_set_mode": str(man_row["candidate_set_mode"]),
                            "observation_condition": str(meta.get("observation_condition", man_row.get("observation_family", ""))),
                            "observation_family": str(man_row.get("observation_family", "")),
                            "observation_scale": float(meta.get("observation_scale", man_row.get("scale", np.nan))),
                            "prior_condition": str(meta.get("prior_condition", man_row.get("prior_family", ""))),
                            "prior_family": str(man_row.get("prior_family", "")),
                            "prior_scale": float(meta.get("prior_scale", man_row.get("scale", np.nan))),
                            "axis_catalog_mode": str(man_row.get("axis_catalog_mode", "shared")),
                            "axis_shared_source_catalog": _safe_bool(man_row.get("axis_shared_source_catalog", False)),
                            "trajectory_prior_mode": str(
                                meta.get("trajectory_prior_mode", man_row.get("trajectory_prior_mode", "unknown"))
                            ),
                            "zero_reference_mode": str(meta.get("zero_reference_mode", man_row.get("zero_reference_mode", ""))),
                            "bin_seconds": float(meta.get("bin_seconds", man_row.get("bin_seconds", np.nan))),
                            "likelihood_scale": float(likelihood_scale),
                            "likelihood_family": "poisson_expected_count",
                            "eps": float(args.eps),
                            "n_candidates": int(prior.shape[0]),
                            "n_trajectories": int(prior.shape[1]),
                            "n_timebins": int(prior.shape[2]),
                            "n_units": int(prior.shape[3]),
                            "true_candidate_index": int(true_idx),
                            "true_image_id": str(candidate_ids[int(true_idx)]),
                            "latent": str(latent),
                            "requested_k": int(k_requested),
                            "k_eff": int(space["k_eff"]),
                            "raw_feature_dim": int(space["raw_feature_dim"]),
                            "feature_variance_fraction": float(space["variance_fraction"]),
                            "feature_space": "selected_windows_zscore_pca",
                            "feature_source": feature_source,
                            "trajectory_weight_family": str(prior_entry["family"]),
                            "trajectory_weight_shape": str(prior_entry["shape_kind"]),
                            "trajectory_weight_basis_type": str(prior_entry["basis_type"]),
                            "prior_beta": float(prior_entry["beta"]),
                            "entropy_matching": str(prior_entry["entropy_matching"]),
                            "prior_N_eff_mean": float(prior_entry["prior_N_eff_mean"]),
                            "prior_N_eff_median": float(prior_entry["prior_N_eff_median"]),
                            "prior_N_eff_fraction_mean": float(prior_entry["prior_N_eff_fraction_mean"]),
                            "prior_N_eff_fraction_median": float(prior_entry["prior_N_eff_fraction_median"]),
                            "prior_entropy_mean": float(prior_entry["prior_entropy_mean"]),
                            "prior_entropy_median": float(prior_entry["prior_entropy_median"]),
                            "prior_log_weight_min": float(prior_entry["prior_log_weight_min"]),
                            "prior_log_weight_max": float(prior_entry["prior_log_weight_max"]),
                            "log_prior_shape": str(prior_entry["log_prior_shape"]),
                            "raw_weight_source": str(prior_entry["raw_source"]),
                            "shared_prior_key_scope": str(prior_entry["shared_prior_key_scope"]),
                            "shared_prior_matched_stable_keys": int(prior_entry["shared_prior_matched_stable_keys"]),
                            "shared_prior_total_stable_keys": int(prior_entry["shared_prior_total_stable_keys"]),
                            "shared_prior_stable_key_fallback_fraction": float(
                                prior_entry["shared_prior_stable_key_fallback_fraction"]
                            ),
                            "shared_prior_nonmatching_fallback_stable_keys": int(
                                prior_entry["shared_prior_nonmatching_fallback_stable_keys"]
                            ),
                            "shared_prior_missing_fallback_stable_keys": int(
                                prior_entry["shared_prior_missing_fallback_stable_keys"]
                            ),
                            "shared_prior_matched_slots": int(prior_entry["shared_prior_matched_slots"]),
                            "shared_prior_total_slots": int(prior_entry["shared_prior_total_slots"]),
                            "shared_prior_fallback_fraction": float(prior_entry["shared_prior_fallback_fraction"]),
                            "shared_prior_nonmatching_fallback_slots": int(prior_entry["shared_prior_nonmatching_fallback_slots"]),
                            "shared_prior_missing_fallback_slots": int(prior_entry["shared_prior_missing_fallback_slots"]),
                            "basis_mode": str(args.basis_mode),
                            "k_dim": int(k_dim),
                            "effective_k_dim": int(prior_entry["effective_k_dim"]),
                            "random_seed_or_null_id": int(prior_entry["null_id"]),
                            "basis_path": str(args.compact_basis_path),
                            "basis_key": str(basis_meta.get("basis_key", "")),
                            "basis_shape": json.dumps(basis_meta.get("basis_shape", [])),
                            "image_disjoint_basis_verified": bool(
                                str(args.basis_mode) == "image_disjoint" and basis_meta.get("declares_image_disjoint", False)
                            ),
                        }
                        for mode, scores in score_by_mode.items():
                            trial_rows.append(
                                _mode_row(
                                    base_cols=base_cols,
                                    observer_mode=mode,
                                    scores=scores,
                                    candidate_features=candidate_features,
                                    z_true=z_true,
                                    true_idx=int(true_idx),
                                    temperature=float(args.posterior_temperature),
                                )
                            )
        if progress_i == 1 or progress_i == len(manifest_items) or progress_i % progress_every == 0:
            _progress(f"scored {progress_i}/{len(manifest_items)} response tables")

    qc_rows.append(
        {
            "qc_type": "response_cache_manifest",
            "n_manifest_rows_after_cli_filters": int(manifest.shape[0] + skipped_cache_rows),
            "n_response_cache_rows_scored": int(manifest.shape[0]),
            "n_manifest_rows_without_response_cache": int(skipped_cache_rows),
        }
    )
    summary = _summary_rows(trial_rows)
    contrasts = _contrast_rows(summary)
    _write_csv(out_dir / "compact_aware_prior_trials.csv", trial_rows)
    _write_csv(out_dir / "compact_aware_prior_summary.csv", summary)
    _write_csv(out_dir / "compact_aware_prior_contrasts.csv", contrasts)
    _write_csv(out_dir / "compact_aware_prior_qc.csv", qc_rows)
    _write_json(
        out_dir / "compact_aware_prior_metadata.json",
        {
            "run_dir": run_dir,
            "n_selected_tables": int(manifest.shape[0]),
            "n_manifest_rows_without_response_cache": int(skipped_cache_rows),
            "feature_source": feature_source,
            "likelihood_scales": likelihood_scales,
            "k_dims": k_dims,
            "prior_families": prior_families,
            "prior_beta": float(args.prior_beta),
            "prior_beta_max": float(args.prior_beta_max),
            "entropy_match_target": str(args.entropy_match_target),
            "basis": basis_meta,
            "shared_prior_pool_count": int(len(shared_prior_pools)),
            "shared_prior_scope": "hash_stripped_stable_trajectory_key_leave_one_table_out",
            "shared_prior_interpretation": "leave_one_table_out_catalog_statistic_trajectory_reweighting_not_universal_eye_motion_prior",
            "shared_prior_exact_trajectory_match": False,
            "basis_mode": str(args.basis_mode),
            "image_disjoint_basis_verified": bool(
                str(args.basis_mode) == "image_disjoint" and basis_meta.get("declares_image_disjoint", False)
            ),
            "outputs": [
                "compact_aware_prior_trials.csv",
                "compact_aware_prior_summary.csv",
                "compact_aware_prior_contrasts.csv",
                "compact_aware_prior_qc.csv",
                "compact_aware_prior_report.md",
            ],
            "config": vars(args),
        },
    )
    _write_report(out_dir, run_dir=run_dir, summary=summary, contrasts=contrasts, args=args)
    _progress(f"wrote compact-aware prior outputs to {out_dir}")
    return out_dir


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
