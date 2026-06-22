"""Post-hoc compact-subspace mechanism analysis for BackImage response tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .likelihood import effective_count, entropy, logsumexp, normalized_log_weights, posterior_from_log_scores, rank_desc, true_margin


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(float(part)) for part in _parse_list(text)]


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in _parse_list(text)]


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


def _nanmedian(values: pd.Series) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _load_npz_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(n_candidates)]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _npz_scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        return ""
    arr = np.asarray(data[key])
    if arr.size != 1:
        return ""
    return str(arr.reshape(-1)[0])


def _npz_scalar_bool(data: np.lib.npyio.NpzFile, key: str) -> bool:
    if key not in data.files:
        return False
    arr = np.asarray(data[key])
    if arr.size != 1:
        return False
    value = arr.reshape(-1)[0]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "image_disjoint"}


def _load_basis(path: Path, *, n_units: int, basis_key: str = "auto") -> tuple[np.ndarray, dict[str, Any]]:
    if path.suffix.lower() != ".npz":
        raise ValueError("Only .npz compact basis files are currently supported")
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        if basis_key == "auto":
            preferred = ["U", "basis", "basis_delta_0p25", "basis_uncentered", "basis_centered_across_tangents_per_unit"]
            chosen = next((key for key in preferred if key in data.files), None)
            if chosen is None:
                chosen = next((key for key in data.files if str(key).startswith("basis")), None)
            if chosen is None:
                raise ValueError(f"No basis-like key found in {path}; available keys={keys}")
        else:
            chosen = str(basis_key)
            if chosen not in data.files:
                raise ValueError(f"basis_key={chosen!r} not found in {path}; available keys={keys}")
        basis = np.asarray(data[chosen], dtype=np.float64)
        provenance_text = " ".join(
            _npz_scalar_string(data, key).lower()
            for key in ["basis_mode", "basis_provenance", "provenance", "split_mode", "basis_split_mode"]
        )
        declares_image_disjoint = bool(
            _npz_scalar_bool(data, "image_disjoint")
            or _npz_scalar_bool(data, "is_image_disjoint")
            or "image_disjoint" in provenance_text
            or "image-disjoint" in provenance_text
        )
    if basis.ndim != 2:
        raise ValueError(f"Basis must be 2D, got {basis.shape}")
    if basis.shape[0] != int(n_units):
        raise ValueError(f"Basis unit count {basis.shape[0]} does not match response table units {n_units}")
    gram = basis.T @ basis
    err_before = float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro"))
    orthonormalized = False
    if err_before > 1e-5:
        basis, _r = np.linalg.qr(basis)
        orthonormalized = True
    gram_after = basis.T @ basis
    err_after = float(np.linalg.norm(gram_after - np.eye(gram_after.shape[0]), ord="fro"))
    return basis, {
        "basis_path": str(path),
        "basis_key": chosen,
        "basis_shape": list(basis.shape),
        "orthonormalized": bool(orthonormalized),
        "orthonormal_error_before": err_before,
        "orthonormal_error_after": err_after,
        "available_keys": keys,
        "declares_image_disjoint": declares_image_disjoint,
    }


def _validate_basis_mode(args: argparse.Namespace, basis_meta: dict[str, Any]) -> None:
    mode = str(args.basis_mode)
    if mode not in {"global", "image_disjoint"}:
        raise ValueError("basis_mode must be either 'global' or 'image_disjoint'")
    if mode == "image_disjoint" and not bool(basis_meta.get("declares_image_disjoint", False)):
        if not bool(args.allow_unverified_image_disjoint_basis):
            raise ValueError(
                "basis_mode='image_disjoint' requires basis-file provenance declaring image_disjoint. "
                "Use --allow-unverified-image-disjoint-basis only when provenance was verified out of band."
            )


def _random_basis(n_units: int, k_dim: int, rng: np.random.Generator) -> np.ndarray:
    q, _r = np.linalg.qr(rng.standard_normal((int(n_units), int(k_dim))))
    return q[:, : int(k_dim)]


def _unit_shuffle_basis(u: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(u.shape[0])
    q, _r = np.linalg.qr(u[perm])
    return q[:, : u.shape[1]], perm


def _orth_residual_basis(primary: np.ndarray, nuisance: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    if primary.ndim != 2 or nuisance.ndim != 2:
        raise ValueError("primary and nuisance bases must be 2D")
    if primary.shape[0] != nuisance.shape[0]:
        raise ValueError("primary and nuisance bases must have matching unit counts")
    if primary.shape[1] == 0:
        return primary[:, :0]
    if nuisance.shape[1] > 0:
        qn, _ = np.linalg.qr(nuisance)
        residual = primary - qn @ (qn.T @ primary)
    else:
        residual = primary.copy()
    q, r = np.linalg.qr(residual)
    diag = np.abs(np.diag(r)) if r.ndim == 2 else np.asarray([], dtype=np.float64)
    scale = float(np.max(diag)) if diag.size else 0.0
    keep = diag > max(float(tol), scale * float(tol))
    return q[:, keep]


def _static_pc_basis(zero_tables: list[np.ndarray], n_units: int, k_max: int) -> np.ndarray:
    if not zero_tables:
        return np.eye(int(n_units), min(int(k_max), int(n_units)), dtype=np.float64)
    x = np.concatenate([np.asarray(z, dtype=np.float64).reshape(-1, int(n_units)) for z in zero_tables], axis=0)
    x = x - np.mean(x, axis=0, keepdims=True)
    if x.shape[0] < 2 or not np.isfinite(x).all():
        return np.eye(int(n_units), min(int(k_max), int(n_units)), dtype=np.float64)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return vt[: min(int(k_max), vt.shape[0])].T


def _project_delta(delta: np.ndarray, u: np.ndarray) -> np.ndarray:
    tmp = np.tensordot(np.asarray(delta, dtype=np.float64), np.asarray(u, dtype=np.float64), axes=([-1], [0]))
    return np.tensordot(tmp, np.asarray(u, dtype=np.float64).T, axes=([-1], [0]))


def _rate_audit(arr: np.ndarray, eps: float) -> dict[str, float]:
    vals = np.asarray(arr, dtype=np.float64)
    neg = vals < 0.0
    clipped = vals < float(eps)
    return {
        "negative_rate_fraction_before_clamp": float(np.mean(neg)) if vals.size else float("nan"),
        "negative_rate_min": float(np.min(vals)) if vals.size else float("nan"),
        "negative_rate_mass": float(np.sum(-vals[neg])) if np.any(neg) else 0.0,
        "clipped_rate_fraction": float(np.mean(clipped)) if vals.size else float("nan"),
    }


def _safe_for_likelihood(arr: np.ndarray, eps: float) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("Variant response table contains non-finite values")
    return np.maximum(vals, float(eps))


def _poisson_score_allow_projected(y_obs_counts: np.ndarray, lambda_counts: np.ndarray, *, eps: float, likelihood_scale: float) -> np.ndarray:
    obs = np.asarray(y_obs_counts, dtype=np.float64)
    pred = _safe_for_likelihood(lambda_counts, eps)
    if obs.ndim != 2:
        raise ValueError(f"Observation must be (time, units), got {obs.shape}")
    if pred.ndim < 2:
        raise ValueError(f"Prediction must have at least 2 dimensions, got {pred.shape}")
    if pred.shape[-2:] != obs.shape:
        raise ValueError(f"Prediction trailing shape {pred.shape[-2:]} does not match observation {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError("Observation contains non-finite values")
    if np.any(obs < 0.0):
        raise ValueError("Observation contains negative counts")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    if float(likelihood_scale) <= 0.0:
        raise ValueError("likelihood_scale must be positive")
    score = np.sum(obs * np.log(pred) - pred, axis=(-2, -1))
    return float(likelihood_scale) * score


def _score_variant(
    *,
    y_obs_counts: np.ndarray,
    prior_lambda_counts: np.ndarray,
    known_lambda_counts: np.ndarray,
    zero_lambda_counts: np.ndarray,
    true_candidate_index: int,
    candidate_ids: list[str],
    true_trajectory_index: int,
    nearest_trajectory_index: int,
    nearest_trajectory_distance: float,
    eps: float,
    likelihood_scale: float,
) -> dict[str, Any]:
    n_candidates, n_traj, n_time, n_units = prior_lambda_counts.shape
    if known_lambda_counts.shape != (n_candidates, n_time, n_units):
        raise ValueError(f"known_lambda_counts shape {known_lambda_counts.shape} does not match {(n_candidates, n_time, n_units)}")
    if zero_lambda_counts.shape != (n_candidates, n_time, n_units):
        raise ValueError(f"zero_lambda_counts shape {zero_lambda_counts.shape} does not match {(n_candidates, n_time, n_units)}")
    if int(true_candidate_index) < 0 or int(true_candidate_index) >= n_candidates:
        raise ValueError(f"true_candidate_index {true_candidate_index} outside candidate table size {n_candidates}")
    if len(candidate_ids) != n_candidates:
        raise ValueError(f"candidate_ids length {len(candidate_ids)} does not match n_candidates={n_candidates}")
    log_prior = normalized_log_weights(None, n_traj)
    prior_ll = _poisson_score_allow_projected(y_obs_counts, prior_lambda_counts, eps=eps, likelihood_scale=likelihood_scale)
    known_scores = _poisson_score_allow_projected(y_obs_counts, known_lambda_counts, eps=eps, likelihood_scale=likelihood_scale)
    zero_scores = _poisson_score_allow_projected(y_obs_counts, zero_lambda_counts, eps=eps, likelihood_scale=likelihood_scale)
    joint_scores = logsumexp(prior_ll + log_prior[None, :], axis=1)
    best_single_tau_scores = np.max(prior_ll, axis=1)
    true_idx = int(true_candidate_index)
    true_log_posterior = prior_ll[true_idx] + log_prior
    posterior = posterior_from_log_scores(true_log_posterior)
    neff = effective_count(posterior)

    def pred(prefix: str, scores: np.ndarray) -> dict[str, Any]:
        pred_idx = int(np.nanargmax(scores)) if np.isfinite(scores).any() else -1
        return {
            f"{prefix}_pred_candidate_index": pred_idx,
            f"{prefix}_pred_image_id": candidate_ids[pred_idx] if 0 <= pred_idx < len(candidate_ids) else "",
            f"{prefix}_correct": bool(pred_idx == true_idx) if pred_idx >= 0 else False,
            f"{prefix}_true_rank": rank_desc(scores, true_idx),
            f"{prefix}_true_margin": true_margin(scores, true_idx),
            f"{prefix}_true_score": float(scores[true_idx]),
        }

    out: dict[str, Any] = {
        "n_candidates": int(n_candidates),
        "n_trajectories": int(n_traj),
        "n_timebins": int(n_time),
        "n_units": int(n_units),
        "true_candidate_index": true_idx,
        "true_image_id": candidate_ids[true_idx],
        "N_eff_true_image": float(neff),
        "N_eff_true_image_fraction": float(neff / n_traj) if np.isfinite(neff) else float("nan"),
        "posterior_entropy_true_image": entropy(posterior),
        "nearest_tau_rank": rank_desc(true_log_posterior, int(nearest_trajectory_index)) if int(nearest_trajectory_index) >= 0 else float("nan"),
        "nearest_tau_distance": float(nearest_trajectory_distance),
        "true_tau_rank": rank_desc(true_log_posterior, int(true_trajectory_index)) if int(true_trajectory_index) >= 0 else float("nan"),
        "best_single_tau_score": float(best_single_tau_scores[true_idx]),
        "joint_vs_best_single_tau_gap": float(best_single_tau_scores[true_idx] - joint_scores[true_idx]),
        "joint_minus_zero_true_score": float(joint_scores[true_idx] - zero_scores[true_idx]),
        "known_minus_zero_true_score": float(known_scores[true_idx] - zero_scores[true_idx]),
    }
    out.update(pred("known", known_scores))
    out.update(pred("zero", zero_scores))
    out.update(pred("joint", joint_scores))
    out.update(pred("best_single_tau", best_single_tau_scores))
    return out


def _variant_tables(
    variant: str,
    *,
    prior_full: np.ndarray,
    known_full: np.ndarray,
    zero: np.ndarray,
    u: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant == "full_exact":
        return prior_full, known_full, zero
    if variant == "zero_static":
        return np.broadcast_to(zero[:, None, :, :], prior_full.shape).copy(), zero.copy(), zero.copy()
    if u is None:
        raise ValueError(f"response_variant={variant!r} requires a basis")
    eps = 1e-12
    prior_delta = prior_full.astype(np.float64) - zero[:, None, :, :].astype(np.float64)
    known_delta = known_full.astype(np.float64) - zero.astype(np.float64)
    prior_proj = _project_delta(prior_delta, u)
    known_proj = _project_delta(known_delta, u)
    only_variants = {
        "compact_only",
        "random_k",
        "random_only",
        "unit_shuffle_compact",
        "unit_shuffle_only",
        "gain_only",
        "static_pc_k",
        "static_pc_only",
        "compact_residualized_against_static_pc_only",
        "static_pc_residualized_against_compact_only",
    }
    removed_variants = {
        "compact_removed",
        "random_removed",
        "unit_shuffle_removed",
        "gain_removed",
        "static_pc_removed",
        "compact_residualized_against_static_pc_removed",
        "static_pc_residualized_against_compact_removed",
    }
    if variant in only_variants:
        return zero[:, None, :, :].astype(np.float64) + prior_proj, zero.astype(np.float64) + known_proj, zero.astype(np.float64)
    if variant in removed_variants:
        return (
            zero[:, None, :, :].astype(np.float64) + (prior_delta - prior_proj),
            zero.astype(np.float64) + (known_delta - known_proj),
            zero.astype(np.float64),
        )
    if variant in {"log_compact_only", "log_compact_removed"}:
        zero_prior = np.maximum(zero[:, None, :, :].astype(np.float64), eps)
        zero_known = np.maximum(zero.astype(np.float64), eps)
        prior_log_delta = np.log(np.maximum(prior_full.astype(np.float64), eps)) - np.log(zero_prior)
        known_log_delta = np.log(np.maximum(known_full.astype(np.float64), eps)) - np.log(zero_known)
        prior_log_proj = _project_delta(prior_log_delta, u)
        known_log_proj = _project_delta(known_log_delta, u)
        if variant == "log_compact_only":
            return (
                zero_prior * np.exp(prior_log_proj),
                zero_known * np.exp(known_log_proj),
                zero.astype(np.float64),
            )
        return (
            zero_prior * np.exp(prior_log_delta - prior_log_proj),
            zero_known * np.exp(known_log_delta - known_log_proj),
            zero.astype(np.float64),
        )
    raise ValueError(f"Unsupported response_variant={variant!r}")


def _filter_manifest(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = manifest.copy()
    if args.candidate_set_modes:
        keep = set(_parse_list(args.candidate_set_modes))
        out = out[out["candidate_set_mode"].astype(str).isin(keep)]
    if args.motion_scales:
        keep_scales = _parse_float_list(args.motion_scales)
        out = out[out["scale"].astype(float).round(10).isin([round(v, 10) for v in keep_scales])]
    if args.priors:
        keep_priors = {p.lower() for p in _parse_list(args.priors)}
        out = out[out["prior_family"].astype(str).str.lower().isin(keep_priors)]
    if int(args.max_tables) > 0:
        out = out.head(int(args.max_tables))
    return out.reset_index(drop=True)


def _load_nearest_distance_lookup(base: Path) -> dict[str, float]:
    path = base / "observer_trials.csv"
    if not path.exists():
        return {}
    try:
        trials = pd.read_csv(path, usecols=["response_cache_path", "nearest_tau_distance"])
    except (ValueError, FileNotFoundError):
        return {}
    lookup: dict[str, float] = {}
    for cache_path, grp in trials.groupby("response_cache_path", dropna=False):
        values = pd.to_numeric(grp["nearest_tau_distance"], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            lookup[str(cache_path)] = float(finite[0])
    return lookup


def _nearest_distance(table: dict[str, np.ndarray], man_row: pd.Series, observer_lookup: dict[str, float]) -> float:
    if "nearest_trajectory_distance" in table:
        arr = np.asarray(table["nearest_trajectory_distance"], dtype=np.float64).reshape(-1)
        if arr.size and np.isfinite(arr[0]):
            return float(arr[0])
    if "nearest_trajectory_distance" in man_row.index:
        value = float(man_row.get("nearest_trajectory_distance", np.nan))
        if np.isfinite(value):
            return value
    cache_path = str(man_row.get("response_cache_path", ""))
    return float(observer_lookup.get(cache_path, float("nan")))


def analyze(args: argparse.Namespace) -> Path:
    base = Path(args.base_run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base / "compact_mechanism_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(base / "response_cache_manifest.csv")
    manifest = _filter_manifest(manifest, args)
    if manifest.empty:
        raise ValueError("No response tables selected after filters")
    print(
        f"[compact-mechanism] selected response tables={len(manifest)}; "
        f"k_dims={args.k_dims}; variants={args.variants}; likelihood_scales={args.likelihood_scales}",
        flush=True,
    )
    first = _load_npz_table(base / str(manifest.iloc[0]["response_cache_path"]))
    n_units = int(first["prior_lambda_counts"].shape[-1])
    max_k = max(_parse_int_list(args.k_dims))
    basis_full, basis_meta = _load_basis(Path(args.compact_basis_path), n_units=n_units, basis_key=str(args.basis_key))
    _validate_basis_mode(args, basis_meta)
    if basis_full.shape[1] < max_k:
        raise ValueError(f"Basis has only {basis_full.shape[1]} columns, but max k={max_k}")

    zero_tables_for_static = []
    static_variants = {
        "static_pc_k",
        "static_pc_only",
        "static_pc_removed",
        "compact_residualized_against_static_pc_only",
        "compact_residualized_against_static_pc_removed",
        "static_pc_residualized_against_compact_only",
        "static_pc_residualized_against_compact_removed",
    }
    if set(_parse_list(args.variants)).intersection(static_variants):
        print("[compact-mechanism] building static response PC basis", flush=True)
        for static_i, (_, row) in enumerate(manifest.iterrows(), start=1):
            tab = _load_npz_table(base / str(row["response_cache_path"]))
            zero_tables_for_static.append(np.asarray(tab["zero_lambda_counts"], dtype=np.float32))
            if static_i == 1 or static_i % 64 == 0 or static_i == len(manifest):
                print(f"[compact-mechanism] static PC cache load {static_i}/{len(manifest)}", flush=True)
        static_basis = _static_pc_basis(zero_tables_for_static, n_units=n_units, k_max=max_k)
        print("[compact-mechanism] static response PC basis ready", flush=True)
    else:
        static_basis = None

    variants = _parse_list(args.variants)
    k_dims = _parse_int_list(args.k_dims)
    likelihood_scales = _parse_float_list(args.likelihood_scales)
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    random_bases: dict[tuple[int, int], np.ndarray] = {}
    n_random = max(0, int(args.n_random))
    for k in k_dims:
        for null_id in range(n_random):
            random_bases[(k, null_id)] = _random_basis(n_units, k, np.random.default_rng(int(args.seed) + 100_000 * k + null_id))
    observer_distance_lookup = _load_nearest_distance_lookup(base)

    for progress_i, (table_index, man_row) in enumerate(manifest.iterrows(), start=1):
        if progress_i == 1 or progress_i % 16 == 0 or progress_i == len(manifest):
            print(f"[compact-mechanism] scoring table {progress_i}/{len(manifest)}", flush=True)
        table_path = base / str(man_row["response_cache_path"])
        table = _load_npz_table(table_path)
        prior_full = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known_full = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        y_obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        true_idx = _scalar_int(table, "true_candidate_index", 0)
        true_tau = _scalar_int(table, "true_trajectory_index", -1)
        nearest_tau = _scalar_int(table, "nearest_trajectory_index", -1)
        candidate_ids = _candidate_ids(table, prior_full.shape[0])
        nearest_dist = _nearest_distance(table, man_row, observer_distance_lookup)

        prior_delta = prior_full - zero[:, None, :, :]
        known_delta = known_full - zero
        for k_i, k in enumerate(k_dims):
            compact_u = basis_full[:, :k]
            basis_by_variant: list[tuple[str, str, int, int, np.ndarray | None]] = []
            for variant in variants:
                if variant in {"full_exact", "zero_static"}:
                    if k_i == 0:
                        basis_by_variant.append((variant, "none", -1, 0, None))
                elif variant in {"compact_only", "compact_removed", "log_compact_only", "log_compact_removed"}:
                    basis_by_variant.append((variant, "compact", -1, 0, compact_u))
                elif variant == "unit_shuffle_compact":
                    u_shuffle, _perm = _unit_shuffle_basis(compact_u, np.random.default_rng(int(args.seed) + 10_000 + k))
                    basis_by_variant.append((variant, "unit_shuffle_compact", -1, 0, u_shuffle))
                elif variant in {"unit_shuffle_only", "unit_shuffle_removed"}:
                    u_shuffle, _perm = _unit_shuffle_basis(compact_u, np.random.default_rng(int(args.seed) + 10_000 + k))
                    basis_by_variant.append((variant, "unit_shuffle_compact", -1, int(u_shuffle.shape[1]), u_shuffle))
                elif variant in {"gain_only", "gain_removed"}:
                    if k_i == 0:
                        gain = np.ones((n_units, 1), dtype=np.float64)
                        gain /= np.linalg.norm(gain)
                        basis_by_variant.append((variant, "gain_ones", -1, 1, gain))
                elif variant in {"static_pc_k", "static_pc_only", "static_pc_removed"}:
                    if static_basis is None:
                        raise ValueError(f"{variant} requested but static basis was not built")
                    basis_by_variant.append((variant, "static_pc", -1, k, static_basis[:, :k]))
                elif variant in {"compact_residualized_against_static_pc_only", "compact_residualized_against_static_pc_removed"}:
                    if static_basis is None:
                        raise ValueError(f"{variant} requested but static basis was not built")
                    u_resid = _orth_residual_basis(compact_u, static_basis[:, :k])
                    basis_by_variant.append((variant, "compact_residualized_against_static_pc", -1, int(u_resid.shape[1]), u_resid))
                elif variant in {"static_pc_residualized_against_compact_only", "static_pc_residualized_against_compact_removed"}:
                    if static_basis is None:
                        raise ValueError(f"{variant} requested but static basis was not built")
                    u_resid = _orth_residual_basis(static_basis[:, :k], compact_u)
                    basis_by_variant.append((variant, "static_pc_residualized_against_compact", -1, int(u_resid.shape[1]), u_resid))
                elif variant in {"random_k", "random_only", "random_removed"}:
                    for null_id in range(n_random):
                        basis_by_variant.append((variant, "random", null_id, k, random_bases[(k, null_id)]))
                else:
                    raise ValueError(f"Unsupported variant {variant!r}")

            # One reconstruction sanity check per table/k for compact basis.
            compact_proj_prior = _project_delta(prior_delta, compact_u)
            compact_proj_known = _project_delta(known_delta, compact_u)
            resid_prior = prior_delta - compact_proj_prior
            resid_known = known_delta - compact_proj_known
            reconstruction_rows.append(
                {
                    "table_index": int(table_index),
                    "response_cache_path": str(man_row["response_cache_path"]),
                    "k_dim": int(k),
                    "prior_delta_reconstruction_max_abs_error": float(np.max(np.abs((compact_proj_prior + resid_prior) - prior_delta))),
                    "known_delta_reconstruction_max_abs_error": float(np.max(np.abs((compact_proj_known + resid_known) - known_delta))),
                }
            )

            for variant, basis_type, null_id, effective_k, u in basis_by_variant:
                prior_var, known_var, zero_var = _variant_tables(
                    variant,
                    prior_full=prior_full,
                    known_full=known_full,
                    zero=zero,
                    u=u,
                )
                audit = _rate_audit(np.concatenate([prior_var.reshape(-1), known_var.reshape(-1), zero_var.reshape(-1)]), float(args.eps))
                for likelihood_scale in likelihood_scales:
                    score = _score_variant(
                        y_obs_counts=y_obs,
                        prior_lambda_counts=prior_var,
                        known_lambda_counts=known_var,
                        zero_lambda_counts=zero_var,
                        true_candidate_index=true_idx,
                        candidate_ids=candidate_ids,
                        true_trajectory_index=true_tau,
                        nearest_trajectory_index=nearest_tau,
                        nearest_trajectory_distance=nearest_dist,
                        eps=float(args.eps),
                        likelihood_scale=float(likelihood_scale),
                    )
                    base_cols = {
                        "table_index": int(table_index),
                        "trial_id": int(man_row["trial_id"]),
                        "candidate_set_mode": str(man_row["candidate_set_mode"]),
                        "prior_condition": str(man_row["prior_family"]),
                        "prior_family": str(man_row["prior_family"]),
                        "motion_scale": float(man_row["scale"]),
                        "likelihood_scale": float(likelihood_scale),
                        "response_variant": str(variant),
                        "basis_type": str(basis_type),
                        "basis_mode": str(args.basis_mode),
                        "k_dim": int(effective_k if effective_k else (k if u is not None else 0)),
                        "requested_k_dim": int(k),
                        "random_seed_or_null_id": int(null_id),
                        "response_cache_path": str(man_row["response_cache_path"]),
                        "basis_path": str(args.compact_basis_path),
                        **audit,
                    }
                    rows.append({**base_cols, **score})
                audit_rows.append(
                    {
                        "table_index": int(table_index),
                        "response_cache_path": str(man_row["response_cache_path"]),
                        "response_variant": str(variant),
                        "basis_type": str(basis_type),
                        "basis_mode": str(args.basis_mode),
                        "k_dim": int(effective_k if effective_k else (k if u is not None else 0)),
                        "random_seed_or_null_id": int(null_id),
                        **audit,
                    }
                )

    rows = _add_rescue_metrics(rows)
    _write_csv(out_dir / "compact_mechanism_trials.csv", rows)
    _write_csv(out_dir / "compact_mechanism_rate_clipping_audit.csv", audit_rows)
    _write_csv(out_dir / "compact_mechanism_reconstruction_checks.csv", reconstruction_rows)
    summary = summarize(rows)
    _write_csv(out_dir / "compact_mechanism_summary.csv", summary)
    _write_csv(out_dir / "compact_mechanism_by_variant.csv", summarize_by_variant(rows))
    _write_csv(out_dir / "compact_mechanism_random_null_summary.csv", summarize_random_nulls(rows))
    _write_csv(out_dir / "compact_mechanism_posterior_summary.csv", summarize_posterior(rows))
    _write_report(out_dir, summary, reconstruction_rows, basis_meta, args)
    _write_json(
        out_dir / "compact_mechanism_run_metadata.json",
        {
            "base_run_dir": str(base),
            "n_selected_tables": int(len(manifest)),
            "basis": basis_meta,
            "basis_mode": str(args.basis_mode),
            "image_disjoint_basis_verified": bool(str(args.basis_mode) == "image_disjoint" and basis_meta.get("declares_image_disjoint", False)),
            "allow_unverified_image_disjoint_basis": bool(args.allow_unverified_image_disjoint_basis),
            "nearest_distance_source": "response_cache_npz_or_manifest_or_observer_trials",
            "config": vars(args),
        },
    )
    return out_dir


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = [
        "candidate_set_mode",
        "prior_condition",
        "motion_scale",
        "likelihood_scale",
        "response_variant",
        "basis_type",
        "basis_mode",
        "k_dim",
    ]
    out = []
    for key, grp in df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n_rows": int(len(grp)),
                "known_eye_accuracy": float(grp["known_correct"].mean()),
                "zero_eye_accuracy": float(grp["zero_correct"].mean()),
                "joint_eye_accuracy": float(grp["joint_correct"].mean()),
                "joint_minus_zero_accuracy": float(grp["joint_correct"].mean() - grp["zero_correct"].mean()),
                "median_joint_true_score": float(grp["joint_true_score"].median()),
                "median_zero_true_score": float(grp["zero_true_score"].median()),
                "median_known_true_score": float(grp["known_true_score"].median()),
                "median_joint_minus_zero_true_score": float(grp["joint_minus_zero_true_score"].median()),
                "median_known_minus_zero_true_score": float(grp["known_minus_zero_true_score"].median()),
                "median_N_eff_fraction": float(grp["N_eff_true_image_fraction"].median()),
                "median_nearest_tau_rank": float(grp["nearest_tau_rank"].median()),
                "median_negative_rate_fraction_before_clamp": float(grp["negative_rate_fraction_before_clamp"].median()),
                "median_negative_rate_min": float(grp["negative_rate_min"].median()),
                "median_negative_rate_mass": float(grp["negative_rate_mass"].median()),
                "median_clipped_rate_fraction": float(grp["clipped_rate_fraction"].median()),
                "median_joint_rescue_fraction_accuracy": _nanmedian(grp["joint_rescue_fraction_accuracy"]),
                "median_known_gap_recovery_accuracy": _nanmedian(grp["known_gap_recovery_accuracy"]),
                "median_variant_delta_from_full_accuracy": _nanmedian(grp["variant_delta_from_full_accuracy"]),
                "median_compact_sufficiency_accuracy": _nanmedian(grp["compact_sufficiency_accuracy"]),
                "median_compact_necessity_accuracy": _nanmedian(grp["compact_necessity_accuracy"]),
                "median_joint_rescue_fraction_true_score": _nanmedian(grp["joint_rescue_fraction_true_score"]),
                "median_known_gap_recovery_true_score": _nanmedian(grp["known_gap_recovery_true_score"]),
                "median_variant_delta_from_full_true_score": _nanmedian(grp["variant_delta_from_full_true_score"]),
                "median_compact_sufficiency_true_score": _nanmedian(grp["compact_sufficiency_true_score"]),
                "median_compact_necessity_true_score": _nanmedian(grp["compact_necessity_true_score"]),
            }
        )
        out.append(row)
    return out


def summarize_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for key, grp in df.groupby(["response_variant", "basis_type", "basis_mode", "k_dim"], dropna=False):
        response_variant, basis_type, basis_mode, k_dim = key
        out.append(
            {
                "response_variant": response_variant,
                "basis_type": basis_type,
                "basis_mode": basis_mode,
                "k_dim": int(k_dim),
                "n_rows": int(len(grp)),
                "joint_eye_accuracy": float(grp["joint_correct"].mean()),
                "zero_eye_accuracy": float(grp["zero_correct"].mean()),
                "known_eye_accuracy": float(grp["known_correct"].mean()),
                "median_joint_minus_zero_true_score": float(grp["joint_minus_zero_true_score"].median()),
                "median_N_eff_fraction": float(grp["N_eff_true_image_fraction"].median()),
                "median_negative_rate_fraction_before_clamp": float(grp["negative_rate_fraction_before_clamp"].median()),
                "median_clipped_rate_fraction": float(grp["clipped_rate_fraction"].median()),
                "median_joint_rescue_fraction_true_score": _nanmedian(grp["joint_rescue_fraction_true_score"]),
                "median_compact_sufficiency_true_score": _nanmedian(grp["compact_sufficiency_true_score"]),
                "median_compact_necessity_true_score": _nanmedian(grp["compact_necessity_true_score"]),
            }
        )
    return out


def summarize_random_nulls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df = df[df["basis_type"].astype(str) == "random"]
    if df.empty:
        return []
    out = []
    for key, grp in df.groupby(["candidate_set_mode", "prior_condition", "motion_scale", "likelihood_scale", "k_dim"], dropna=False):
        candidate_set_mode, prior_condition, motion_scale, likelihood_scale, k_dim = key
        out.append(
            {
                "candidate_set_mode": candidate_set_mode,
                "prior_condition": prior_condition,
                "motion_scale": float(motion_scale),
                "likelihood_scale": float(likelihood_scale),
                "k_dim": int(k_dim),
                "n_rows": int(len(grp)),
                "n_nulls": int(grp["random_seed_or_null_id"].nunique()),
                "joint_eye_accuracy_mean": float(grp["joint_correct"].mean()),
                "joint_eye_accuracy_median_by_null": float(grp.groupby("random_seed_or_null_id")["joint_correct"].mean().median()),
                "joint_eye_accuracy_p95_by_null": float(grp.groupby("random_seed_or_null_id")["joint_correct"].mean().quantile(0.95)),
                "median_joint_rescue_fraction_true_score": _nanmedian(grp["joint_rescue_fraction_true_score"]),
            }
        )
    return out


def summarize_posterior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for key, grp in df.groupby(["response_variant", "basis_type", "basis_mode", "k_dim"], dropna=False):
        response_variant, basis_type, basis_mode, k_dim = key
        out.append(
            {
                "response_variant": response_variant,
                "basis_type": basis_type,
                "basis_mode": basis_mode,
                "k_dim": int(k_dim),
                "n_rows": int(len(grp)),
                "median_N_eff_true_image": float(grp["N_eff_true_image"].median()),
                "median_N_eff_true_image_fraction": float(grp["N_eff_true_image_fraction"].median()),
                "median_posterior_entropy_true_image": float(grp["posterior_entropy_true_image"].median()),
                "median_nearest_tau_rank": float(grp["nearest_tau_rank"].median()),
                "median_true_tau_rank": float(grp["true_tau_rank"].median()),
                "median_joint_vs_best_single_tau_gap": float(grp["joint_vs_best_single_tau_gap"].median()),
            }
        )
    return out


def _safe_ratio(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(float(denom)) <= 1e-12:
        return float("nan")
    return float(numer) / float(denom)


def _add_rescue_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    metric_cols = [
        "joint_rescue_fraction_accuracy",
        "known_gap_recovery_accuracy",
        "compact_sufficiency_accuracy",
        "compact_necessity_accuracy",
        "compact_removed_gain_accuracy",
        "variant_delta_from_full_accuracy",
        "joint_rescue_fraction_true_score",
        "known_gap_recovery_true_score",
        "compact_sufficiency_true_score",
        "compact_necessity_true_score",
        "compact_removed_gain_true_score",
        "variant_delta_from_full_true_score",
    ]
    df = pd.DataFrame(rows)
    key_cols = ["table_index", "likelihood_scale"]
    full = df[df["response_variant"] == "full_exact"].copy()
    if full.empty:
        for row in rows:
            for col in metric_cols:
                row[col] = float("nan")
        return rows
    full = full.drop_duplicates(key_cols)
    full_lookup = {
        tuple(row[col] for col in key_cols): row
        for _, row in full.iterrows()
    }
    out = []
    for row in rows:
        for col in metric_cols:
            row[col] = float("nan")
        key = tuple(row[col] for col in key_cols)
        base = full_lookup.get(key)
        if base is None:
            out.append(row)
            continue
        zero_correct = float(bool(base["zero_correct"]))
        full_joint_correct = float(bool(base["joint_correct"]))
        known_correct = float(bool(base["known_correct"]))
        row_joint_correct = float(bool(row["joint_correct"]))
        zero_score = float(base["zero_true_score"])
        full_joint_score = float(base["joint_true_score"])
        known_score = float(base["known_true_score"])
        row_joint_score = float(row["joint_true_score"])

        acc_rescue = _safe_ratio(row_joint_correct - zero_correct, full_joint_correct - zero_correct)
        score_rescue = _safe_ratio(row_joint_score - zero_score, full_joint_score - zero_score)
        row["joint_rescue_fraction_accuracy"] = acc_rescue
        row["known_gap_recovery_accuracy"] = _safe_ratio(row_joint_correct - zero_correct, known_correct - zero_correct)
        row["variant_delta_from_full_accuracy"] = row_joint_correct - full_joint_correct
        row["compact_sufficiency_accuracy"] = acc_rescue if row["response_variant"] in {"compact_only", "log_compact_only"} else float("nan")
        row["compact_removed_gain_accuracy"] = (
            row_joint_correct - zero_correct if row["response_variant"] in {"compact_removed", "log_compact_removed"} else float("nan")
        )
        row["compact_necessity_accuracy"] = (
            _safe_ratio(full_joint_correct - row_joint_correct, full_joint_correct - zero_correct)
            if row["response_variant"] in {"compact_removed", "log_compact_removed"}
            else float("nan")
        )
        row["joint_rescue_fraction_true_score"] = score_rescue
        row["known_gap_recovery_true_score"] = _safe_ratio(row_joint_score - zero_score, known_score - zero_score)
        row["variant_delta_from_full_true_score"] = row_joint_score - full_joint_score
        row["compact_sufficiency_true_score"] = score_rescue if row["response_variant"] in {"compact_only", "log_compact_only"} else float("nan")
        row["compact_removed_gain_true_score"] = (
            row_joint_score - zero_score if row["response_variant"] in {"compact_removed", "log_compact_removed"} else float("nan")
        )
        row["compact_necessity_true_score"] = (
            _safe_ratio(full_joint_score - row_joint_score, full_joint_score - zero_score)
            if row["response_variant"] in {"compact_removed", "log_compact_removed"}
            else float("nan")
        )
        out.append(row)
    return out


def _write_report(out_dir: Path, summary: list[dict[str, Any]], reconstruction_rows: list[dict[str, Any]], basis_meta: dict[str, Any], args: argparse.Namespace) -> None:
    df = pd.DataFrame(summary)
    lines = [
        "# Compact Mechanism Analysis",
        "",
        "This is a cache-only projection analysis. It does not rerun the V1 twin.",
        "",
        "## Basis",
        "",
        f"- path: `{basis_meta['basis_path']}`",
        f"- key: `{basis_meta['basis_key']}`",
        f"- shape: `{basis_meta['basis_shape']}`",
        f"- basis_mode: `{args.basis_mode}`",
        f"- orthonormalized: `{basis_meta['orthonormalized']}`",
        "",
        "## Sanity Checks",
        "",
        f"- max prior reconstruction error: `{max((r['prior_delta_reconstruction_max_abs_error'] for r in reconstruction_rows), default=float('nan')):.6g}`",
        f"- max known reconstruction error: `{max((r['known_delta_reconstruction_max_abs_error'] for r in reconstruction_rows), default=float('nan')):.6g}`",
        "",
        "## Summary Preview",
        "",
    ]
    if not df.empty:
        preview_cols = [
            "candidate_set_mode",
            "motion_scale",
            "prior_condition",
            "likelihood_scale",
            "response_variant",
            "basis_type",
            "k_dim",
            "joint_eye_accuracy",
            "joint_minus_zero_accuracy",
            "median_N_eff_fraction",
            "median_negative_rate_fraction_before_clamp",
        ]
        lines.append("```text")
        lines.append(df[preview_cols].head(80).to_csv(index=False).strip())
        lines.append("```")
    (out_dir / "compact_mechanism_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--compact-basis-path", type=Path, required=True)
    parser.add_argument("--basis-key", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--basis-mode", default="global")
    parser.add_argument("--allow-unverified-image-disjoint-basis", action="store_true")
    parser.add_argument("--k-dims", default="2,5,10,20")
    parser.add_argument(
        "--variants",
        default="full_exact,zero_static,compact_only,compact_removed,random_k,unit_shuffle_compact,gain_only,static_pc_k",
    )
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--likelihood-scales", default="0.5,1.0")
    parser.add_argument("--candidate-set-modes", default="")
    parser.add_argument("--motion-scales", default="")
    parser.add_argument("--priors", default="")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tables", type=int, default=0)
    return parser


def main() -> None:
    out = analyze(build_parser().parse_args())
    print(f"Wrote compact mechanism analysis to {out}")


if __name__ == "__main__":
    main()
