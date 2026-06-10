"""Run Figure 5 checks 5-9 on cached e-optotype model rate files.

This is a runnable model-side scaffold for the later recorded-V1 checks. It
uses cached deterministic rate arrays under ``scripts/temporal_decoding/data``
and labels the source as ``model_cached_rates`` throughout. The goal is to make
the population-coding checks concrete without pretending these are recorded
covariance estimates.

Checks implemented
------------------
5. Reafference-signal subspace alignment: alpha, pairwise L_ij, principal angles.
6. Constrained population coding metric: full-covariance versus diagonal dprime.
7. Tangent/reafference-aware remove-out: CV nearest-centroid decoding before and
   after projecting out a training-fold-fitted top reafferent subspace.
8. Compact add-back/remove-out: evaluate stabilized + compact/orthogonal real
   motion deltas, when real/stabilized trials can be index-matched and an
   external compact basis is supplied.
9. Amplitude/condition sweep: run 5-7 metrics for every available requested
   condition, including cached scaled conditions.

Example
-------
.venv/bin/python declan/active_sensing_movie_information/run_figure5_cached_rate_checks_5_to_9.py \
  --logmar -0.20 \
  --conditions real,stabilized,matched_null,fixed_center,scaled_0.05_current_n4_first,scaled_0.1_current_n4_first \
  --max-trials 128
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RATES_DIR = ROOT / "scripts" / "temporal_decoding" / "data" / "rates"
DEFAULT_OUT_DIR = ROOT / "outputs" / "active_sensing_movie_information" / "figure5_cached_rate_checks_5_to_9"
ORIENTATIONS = (0, 90, 180, 270)
ORI_KEYS = tuple(f"ori{ori}" for ori in ORIENTATIONS)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_group_value(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return "__nan__"
    return value


def condition_scale(condition: str) -> float:
    if condition == "stabilized":
        return 0.0
    if condition == "real":
        return 1.0
    if condition.startswith("scaled_"):
        try:
            return float(condition.split("_")[1])
        except (IndexError, ValueError):
            return float("nan")
    return float("nan")


def rate_file(logmar: float, orientation: int, condition: str, rates_dir: Path) -> Path:
    return rates_dir / f"rates_hires_lm{logmar:.2f}_ori{orientation}_{condition}.npz"


def load_time_averaged_rates(
    *,
    logmar: float,
    condition: str,
    rates_dir: Path,
    max_trials: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for orientation in ORIENTATIONS:
        path = rate_file(logmar, orientation, condition, rates_dir)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as npz:
            rates = np.asarray(npz["rates"], dtype=np.float32)
            lengths = np.asarray(npz["lengths"], dtype=np.int64)
            n_trials = min(int(rates.shape[0]), int(max_trials) if max_trials > 0 else int(rates.shape[0]))
            ravg = np.empty((n_trials, int(rates.shape[2])), dtype=np.float64)
            for i in range(n_trials):
                length = max(1, min(int(lengths[i]), int(rates.shape[1])))
                ravg[i] = np.mean(rates[i, :length], axis=0, dtype=np.float64)
        out[f"ori{orientation}"] = ravg
    return equalize_trials(out)


def equalize_trials(ravg_by_ori: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = min(arr.shape[0] for arr in ravg_by_ori.values())
    return {key: arr[:n].astype(np.float64, copy=False) for key, arr in ravg_by_ori.items()}


def top_subspace_from_samples(samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    if x.shape[0] < 2:
        return np.zeros((x.shape[1], 0)), np.zeros((0,))
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    k_eff = max(1, min(int(k), vt.shape[0]))
    eigvals = (s[:k_eff] ** 2) / max(x.shape[0] - 1, 1)
    return vt[:k_eff].T, eigvals


def pooled_residuals(ravg_by_ori: dict[str, np.ndarray]) -> np.ndarray:
    chunks = []
    for key in ORI_KEYS:
        x = ravg_by_ori[key]
        chunks.append(x - np.mean(x, axis=0, keepdims=True))
    return np.concatenate(chunks, axis=0)


def pooled_residuals_for_indices(ravg_by_ori: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    chunks = []
    for key in ORI_KEYS:
        x = ravg_by_ori[key][indices]
        chunks.append(x - np.mean(x, axis=0, keepdims=True))
    return np.concatenate(chunks, axis=0)


def signal_means(ravg_by_ori: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.mean(ravg_by_ori[key], axis=0) for key in ORI_KEYS], axis=0)


def signal_covariance(ravg_by_ori: dict[str, np.ndarray]) -> np.ndarray:
    means = signal_means(ravg_by_ori)
    means = means - np.mean(means, axis=0, keepdims=True)
    c = means.T @ means / max(means.shape[0] - 1, 1)
    return (c + c.T) / 2.0


def covariance_from_residuals(residuals: np.ndarray) -> np.ndarray:
    x = np.asarray(residuals, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    c = x.T @ x / max(x.shape[0] - 1, 1)
    return (c + c.T) / 2.0


def principal_angles_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if u.size == 0 or v.size == 0:
        return np.asarray([], dtype=np.float64)
    qu, _ = np.linalg.qr(u)
    qv, _ = np.linalg.qr(v)
    s = np.linalg.svd(qu.T @ qv, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def ridge_inverse(cov: np.ndarray, ridge_fraction: float) -> np.ndarray:
    c = np.asarray(cov, dtype=np.float64)
    scale = float(np.trace(c) / max(c.shape[0], 1))
    ridge = max(float(ridge_fraction) * scale, 1e-8)
    return np.linalg.pinv(c + np.eye(c.shape[0]) * ridge, hermitian=True)


def pairwise_signal_vectors(means: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out = []
    for i, oi in enumerate(ORIENTATIONS):
        for j, oj in enumerate(ORIENTATIONS):
            if j <= i:
                continue
            out.append((f"{oi}_vs_{oj}", means[i] - means[j]))
    return out


def alignment_metrics(
    *,
    condition: str,
    ravg_by_ori: dict[str, np.ndarray],
    k_list: list[int],
    n_nulls: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    residuals = pooled_residuals(ravg_by_ori)
    c_signal = signal_covariance(ravg_by_ori)
    means = signal_means(ravg_by_ori)
    u_signal_full, signal_eigs = top_subspace_from_samples(means, max(k_list))
    trace_signal = float(np.trace(c_signal))
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    c_reaff = covariance_from_residuals(residuals)
    n_units = residuals.shape[1]

    for k in k_list:
        u_reaff, reaff_eigs = top_subspace_from_samples(residuals, k)
        u_signal = u_signal_full[:, : min(k, u_signal_full.shape[1])]
        alpha = float(np.trace(u_reaff.T @ c_signal @ u_reaff) / (trace_signal + 1e-12))
        null_alphas = []
        for _ in range(int(n_nulls)):
            q, _ = np.linalg.qr(rng.normal(size=(n_units, u_reaff.shape[1])))
            null_alphas.append(float(np.trace(q.T @ c_signal @ q) / (trace_signal + 1e-12)))
        null_arr = np.asarray(null_alphas, dtype=np.float64)
        angles = principal_angles_deg(u_reaff, u_signal)
        rows.append(
            {
                "check": "5_reafference_signal_alignment",
                "source": "model_cached_rates",
                "condition": condition,
                "scale": condition_scale(condition),
                "k": int(k),
                "n_trials_per_orientation": int(next(iter(ravg_by_ori.values())).shape[0]),
                "n_units": int(n_units),
                "alpha": alpha,
                "alpha_null_mean": float(np.mean(null_arr)) if null_arr.size else float("nan"),
                "alpha_null_std": float(np.std(null_arr, ddof=1)) if null_arr.size > 1 else float("nan"),
                "alpha_x_null": alpha / (float(np.mean(null_arr)) + 1e-12) if null_arr.size else float("nan"),
                "principal_angle_mean_deg": float(np.mean(angles)) if angles.size else float("nan"),
                "principal_angle_min_deg": float(np.min(angles)) if angles.size else float("nan"),
                "top_reaff_eig_sum": float(np.sum(reaff_eigs)),
                "top_signal_eig_sum": float(np.sum(signal_eigs[: min(k, signal_eigs.size)])),
            }
        )
        for pair, dmu in pairwise_signal_vectors(means):
            denom = float(dmu @ dmu) + 1e-12
            pair_rows.append(
                {
                    "check": "5_pairwise_information_limiting_projection",
                    "source": "model_cached_rates",
                    "condition": condition,
                    "scale": condition_scale(condition),
                    "k": int(k),
                    "pair": pair,
                    "L_ij": float(dmu @ c_reaff @ dmu / denom),
                    "projected_signal_norm_frac": float(np.sum((dmu @ u_reaff) ** 2) / denom),
                }
            )
    return rows, pair_rows


def dprime_metrics(
    *,
    condition: str,
    ravg_by_ori: dict[str, np.ndarray],
    ridge_fraction: float,
) -> list[dict[str, Any]]:
    residuals = pooled_residuals(ravg_by_ori)
    c = covariance_from_residuals(residuals)
    inv_full = ridge_inverse(c, ridge_fraction)
    diag = np.diag(np.diag(c))
    inv_diag = ridge_inverse(diag, ridge_fraction)
    means = signal_means(ravg_by_ori)
    rows = []
    for pair, dmu in pairwise_signal_vectors(means):
        d_full = float(dmu @ inv_full @ dmu)
        d_diag = float(dmu @ inv_diag @ dmu)
        rows.append(
            {
                "check": "6_constrained_population_coding",
                "source": "model_cached_rates",
                "condition": condition,
                "scale": condition_scale(condition),
                "pair": pair,
                "dprime2_pop": d_full,
                "dprime2_indep": d_diag,
                "eta_pop_over_indep": d_full / (d_diag + 1e-12),
                "ridge_fraction": float(ridge_fraction),
            }
        )
    return rows


def nearest_centroid_cv(
    ravg_by_ori: dict[str, np.ndarray],
    *,
    n_splits: int,
    remove_k: int | None = None,
) -> tuple[float, float]:
    n = min(arr.shape[0] for arr in ravg_by_ori.values())
    if n < 2:
        return float("nan"), float("nan")
    n_splits_eff = min(max(2, int(n_splits)), n)
    folds = np.array_split(np.arange(n), n_splits_eff)
    accs = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        u_remove = None
        if remove_k is not None and int(remove_k) > 0:
            train_residuals = pooled_residuals_for_indices(ravg_by_ori, train_idx)
            u_remove, _ = top_subspace_from_samples(train_residuals, int(remove_k))
        centroids = []
        for key in ORI_KEYS:
            x = ravg_by_ori[key]
            if u_remove is not None and u_remove.size:
                x = x - (x @ u_remove) @ u_remove.T
            centroids.append(np.mean(x[train_idx], axis=0))
        centroids_arr = np.stack(centroids, axis=0)
        correct = 0
        total = 0
        for label, key in enumerate(ORI_KEYS):
            x = ravg_by_ori[key][test_idx]
            if u_remove is not None and u_remove.size:
                x = x - (x @ u_remove) @ u_remove.T
            dist = np.sum((x[:, None, :] - centroids_arr[None, :, :]) ** 2, axis=2)
            pred = np.argmin(dist, axis=1)
            correct += int(np.sum(pred == label))
            total += int(pred.size)
        accs.append(correct / max(total, 1))
    return float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0


def recoverability_removeout_metrics(
    *,
    condition: str,
    ravg_by_ori: dict[str, np.ndarray],
    k_list: list[int],
    n_splits: int,
) -> list[dict[str, Any]]:
    base_acc, base_std = nearest_centroid_cv(ravg_by_ori, n_splits=n_splits, remove_k=None)
    rows = []
    for k in k_list:
        clean_acc, clean_std = nearest_centroid_cv(ravg_by_ori, n_splits=n_splits, remove_k=k)
        rows.append(
            {
                "check": "7_reafference_aware_removeout",
                "source": "model_cached_rates",
                "condition": condition,
                "scale": condition_scale(condition),
                "k": int(k),
                "cv_decoder": "nearest_centroid",
                "removeout_basis_fit": "training_fold_residual_pca",
                "acc_original": base_acc,
                "acc_original_std": base_std,
                "acc_reaff_removed": clean_acc,
                "acc_reaff_removed_std": clean_std,
                "delta_removed_minus_original": clean_acc - base_acc,
                "n_splits": int(n_splits),
            }
        )
    return rows


def load_compact_basis(path: Path | None, key: str, n_units: int) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as npz:
        keys = [key] if key else ["basis", "U", "U10", "compact_basis", "eigvecs", "components", "vecs"]
        for candidate in keys:
            if candidate not in npz:
                continue
            basis = np.asarray(npz[candidate], dtype=np.float64)
            if basis.ndim != 2:
                raise ValueError(f"Compact basis key {candidate!r} in {path} is not 2D")
            if basis.shape[0] == n_units:
                raw = basis
            elif basis.shape[1] == n_units:
                raw = basis.T
            else:
                raise ValueError(
                    f"Compact basis key {candidate!r} in {path} has shape {basis.shape}, "
                    f"which does not match n_units={n_units}"
                )
            q, r = np.linalg.qr(raw)
            keep = np.abs(np.diag(r)) > 1e-12
            return q[:, keep]
        available = ", ".join(npz.files)
    wanted = key if key else "one of basis,U,U10,compact_basis,eigvecs,components,vecs"
    raise KeyError(f"Could not find compact basis key {wanted!r} in {path}; available keys: {available}")


def addback_conditions(
    real: dict[str, np.ndarray],
    stabilized: dict[str, np.ndarray],
    compact_basis: np.ndarray,
    k: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    delta_by_ori = {}
    for key in ORI_KEYS:
        n = min(real[key].shape[0], stabilized[key].shape[0])
        delta = real[key][:n] - stabilized[key][:n]
        delta_by_ori[key] = delta
    u_delta = compact_basis[:, : min(int(k), compact_basis.shape[1])]
    compact = {}
    orth = {}
    for key in ORI_KEYS:
        n = min(real[key].shape[0], stabilized[key].shape[0])
        delta = delta_by_ori[key]
        delta_compact = (delta @ u_delta) @ u_delta.T
        compact[key] = stabilized[key][:n] + delta_compact
        orth[key] = stabilized[key][:n] + (delta - delta_compact)
    return compact, orth


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(stable_group_value(row.get(k)) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        base = {k: group_rows[0].get(k) for k in group_keys}
        base["n"] = len(group_rows)
        for value_key in value_keys:
            vals = np.asarray([float(row.get(value_key, np.nan)) for row in group_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            base[f"{value_key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            base[f"{value_key}_sem"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out.append(base)
    return out


def condition_sweep_summary(
    alignment_summary: list[dict[str, Any]],
    dprime_summary: list[dict[str, Any]],
    removeout_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def ck(row: dict[str, Any]) -> tuple[Any, Any]:
        return row.get("condition"), stable_group_value(row.get("scale"))

    def ckk(row: dict[str, Any]) -> tuple[Any, Any, Any]:
        return row.get("condition"), stable_group_value(row.get("scale")), row.get("k")

    dprime_by_condition = {ck(row): row for row in dprime_summary}
    removeout_by_condition_k = {
        ckk(row): row for row in removeout_summary
    }
    keys = sorted(
        set(dprime_by_condition)
        | {ck(row) for row in alignment_summary}
        | {ck(row) for row in removeout_summary},
        key=str,
    )
    k_values = sorted({row.get("k") for row in alignment_summary + removeout_summary}, key=str)
    rows = []
    scale_by_key = {ck(row): row.get("scale") for row in alignment_summary + dprime_summary + removeout_summary}
    for condition, stable_scale in keys:
        scale = scale_by_key.get((condition, stable_scale))
        for k in k_values or [None]:
            row: dict[str, Any] = {"condition": condition, "scale": scale}
            if k is not None:
                row["k"] = k
            for source in (
                next(
                    (
                        item
                        for item in alignment_summary
                        if ck(item) == (condition, stable_scale) and item.get("k") == k
                    ),
                    {},
                ),
                dprime_by_condition.get((condition, stable_scale), {}),
                removeout_by_condition_k.get((condition, stable_scale, k), {}),
            ):
                for key, value in source.items():
                    if key not in row:
                        row[key] = value
            rows.append(row)
    return rows


def run_checks(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    conditions = parse_csv(args.conditions)
    k_list = [int(k) for k in parse_csv(args.k_list)]

    loaded: dict[str, dict[str, np.ndarray]] = {}
    inventory_rows = []
    for condition in conditions:
        missing = [str(rate_file(args.logmar, ori, condition, args.rates_dir)) for ori in ORIENTATIONS if not rate_file(args.logmar, ori, condition, args.rates_dir).exists()]
        inventory_rows.append(
            {
                "condition": condition,
                "scale": condition_scale(condition),
                "available": not missing,
                "missing_files": "|".join(missing),
            }
        )
        if missing:
            continue
        loaded[condition] = load_time_averaged_rates(
            logmar=float(args.logmar),
            condition=condition,
            rates_dir=Path(args.rates_dir),
            max_trials=int(args.max_trials),
        )
        inventory_rows[-1]["n_trials_per_orientation"] = int(next(iter(loaded[condition].values())).shape[0])
        inventory_rows[-1]["n_units"] = int(next(iter(loaded[condition].values())).shape[1])

    alignment_rows: list[dict[str, Any]] = []
    pair_projection_rows: list[dict[str, Any]] = []
    dprime_rows: list[dict[str, Any]] = []
    removeout_rows: list[dict[str, Any]] = []
    for condition, ravg_by_ori in loaded.items():
        a_rows, p_rows = alignment_metrics(
            condition=condition,
            ravg_by_ori=ravg_by_ori,
            k_list=k_list,
            n_nulls=int(args.n_nulls),
            rng=rng,
        )
        alignment_rows.extend(a_rows)
        pair_projection_rows.extend(p_rows)
        dprime_rows.extend(dprime_metrics(condition=condition, ravg_by_ori=ravg_by_ori, ridge_fraction=float(args.ridge_fraction)))
        removeout_rows.extend(recoverability_removeout_metrics(condition=condition, ravg_by_ori=ravg_by_ori, k_list=k_list, n_splits=int(args.n_splits)))

    addback_rows: list[dict[str, Any]] = []
    compact_basis: np.ndarray | None = None
    compact_basis_status = "not_requested"
    if loaded and args.compact_basis_npz is not None:
        n_units = int(next(iter(next(iter(loaded.values())).values())).shape[1])
        compact_basis = load_compact_basis(Path(args.compact_basis_npz), str(args.compact_basis_key), n_units)
        compact_basis_status = "loaded"
    if "real" in loaded and "stabilized" in loaded:
        if compact_basis is None:
            compact_basis_status = "skipped_missing_compact_basis"
            for k in k_list:
                addback_rows.append(
                    {
                        "check": "8_compact_addback_removeout",
                        "source": "model_cached_rates",
                        "k": int(k),
                        "row_status": compact_basis_status,
                        "message": "Pass --compact-basis-npz to run Check 8 with an external compact basis.",
                    }
                )
        else:
            for k in k_list:
                compact, orth = addback_conditions(loaded["real"], loaded["stabilized"], compact_basis, k)
                for label, ravg_by_ori in (("compact_addback", compact), ("orthogonal_addback", orth)):
                    a_rows = alignment_metrics(condition=label, ravg_by_ori=ravg_by_ori, k_list=[k], n_nulls=int(args.n_nulls), rng=rng)[0]
                    for row in a_rows:
                        row["k_addback"] = int(k)
                        row["compact_basis_source"] = str(args.compact_basis_npz)
                        row["compact_basis_key"] = str(args.compact_basis_key)
                        row["compact_basis_k_used"] = int(min(int(k), compact_basis.shape[1]))
                        addback_rows.append(row)
                    for row in dprime_metrics(condition=label, ravg_by_ori=ravg_by_ori, ridge_fraction=float(args.ridge_fraction)):
                        row["k_addback"] = int(k)
                        row["compact_basis_source"] = str(args.compact_basis_npz)
                        row["compact_basis_key"] = str(args.compact_basis_key)
                        row["compact_basis_k_used"] = int(min(int(k), compact_basis.shape[1]))
                        addback_rows.append(row)
                    for row in recoverability_removeout_metrics(condition=label, ravg_by_ori=ravg_by_ori, k_list=[k], n_splits=int(args.n_splits)):
                        row["k_addback"] = int(k)
                        row["compact_basis_source"] = str(args.compact_basis_npz)
                        row["compact_basis_key"] = str(args.compact_basis_key)
                        row["compact_basis_k_used"] = int(min(int(k), compact_basis.shape[1]))
                        addback_rows.append(row)

    dprime_summary = aggregate_rows(
        dprime_rows,
        group_keys=["condition", "scale"],
        value_keys=["dprime2_pop", "dprime2_indep", "eta_pop_over_indep"],
    )
    removeout_summary = aggregate_rows(
        removeout_rows,
        group_keys=["condition", "scale", "k"],
        value_keys=["acc_original", "acc_reaff_removed", "delta_removed_minus_original"],
    )
    alignment_summary = aggregate_rows(
        alignment_rows,
        group_keys=["condition", "scale", "k"],
        value_keys=["alpha", "alpha_x_null", "principal_angle_mean_deg"],
    )
    sweep_summary = condition_sweep_summary(alignment_summary, dprime_summary, removeout_summary)

    write_csv_rows(out_dir / "cached_rate_inventory.csv", inventory_rows)
    write_csv_rows(out_dir / "check5_reafference_signal_alignment.csv", alignment_rows)
    write_csv_rows(out_dir / "check5_pairwise_Lij.csv", pair_projection_rows)
    write_csv_rows(out_dir / "check6_constrained_dprime.csv", dprime_rows)
    write_csv_rows(out_dir / "check6_constrained_dprime_summary.csv", dprime_summary)
    write_csv_rows(out_dir / "check7_reafference_removeout.csv", removeout_rows)
    write_csv_rows(out_dir / "check7_reafference_removeout_summary.csv", removeout_summary)
    write_csv_rows(out_dir / "check8_compact_addback_removeout.csv", addback_rows)
    write_csv_rows(out_dir / "check9_condition_amplitude_sweep_alignment_summary.csv", alignment_summary)
    write_csv_rows(out_dir / "check9_condition_amplitude_sweep_summary.csv", sweep_summary)

    manifest = {
        "source": "model_cached_rates",
        "logmar": float(args.logmar),
        "conditions_requested": conditions,
        "conditions_available": sorted(loaded),
        "k_list": k_list,
        "max_trials": int(args.max_trials),
        "n_nulls": int(args.n_nulls),
        "n_splits": int(args.n_splits),
        "ridge_fraction": float(args.ridge_fraction),
        "compact_basis_npz": str(args.compact_basis_npz) if args.compact_basis_npz is not None else "",
        "compact_basis_key": str(args.compact_basis_key),
        "compact_basis_status": compact_basis_status,
        "outputs": [
            "cached_rate_inventory.csv",
            "check5_reafference_signal_alignment.csv",
            "check5_pairwise_Lij.csv",
            "check6_constrained_dprime.csv",
            "check6_constrained_dprime_summary.csv",
            "check7_reafference_removeout.csv",
            "check7_reafference_removeout_summary.csv",
            "check8_compact_addback_removeout.csv",
            "check9_condition_amplitude_sweep_alignment_summary.csv",
            "check9_condition_amplitude_sweep_summary.csv",
        ],
        "caveat": "Model-side cached-rate scaffold. Recorded-V1 Check 5 requires packaged recorded C_reaff/C_signal inputs.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Figure 5 Cached-Rate Checks 5-9",
                "",
                "This run implements Checks 5-9 on cached deterministic model rate files.",
                "It is explicitly labelled `model_cached_rates`; it is not a substitute for",
                "the recorded-V1 reafference-signal alignment fork once recorded covariance",
                "matrices are packaged.",
                "",
                f"LogMAR: `{float(args.logmar):+.2f}`",
                f"Available conditions: `{', '.join(sorted(loaded))}`",
                "",
                "Main outputs:",
                "",
                "- `check5_reafference_signal_alignment.csv`",
                "- `check6_constrained_dprime_summary.csv`",
                "- `check7_reafference_removeout_summary.csv`",
                "- `check8_compact_addback_removeout.csv`",
                "- `check9_condition_amplitude_sweep_summary.csv`",
                "- `check9_condition_amplitude_sweep_alignment_summary.csv`",
                "",
                f"Check 8 compact basis status: `{compact_basis_status}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("Figure 5 cached-rate checks 5-9 complete")
    print(f"  out_dir: {out_dir}")
    print(f"  available conditions: {', '.join(sorted(loaded))}")
    print(f"  alignment rows: {len(alignment_rows)}")
    print(f"  dprime rows: {len(dprime_rows)}")
    print(f"  removeout rows: {len(removeout_rows)}")
    print(f"  addback rows: {len(addback_rows)}")
    print(f"  compact basis status: {compact_basis_status}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logmar", type=float, default=-0.20)
    p.add_argument(
        "--conditions",
        type=str,
        default="real,stabilized,matched_null,fixed_center,scaled_0.05_current_n4_first,scaled_0.1_current_n4_first",
    )
    p.add_argument("--k-list", type=str, default="2,10")
    p.add_argument("--max-trials", type=int, default=128)
    p.add_argument("--n-nulls", type=int, default=100)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--ridge-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rates-dir", type=Path, default=RATES_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--compact-basis-npz", type=Path, default=None)
    p.add_argument("--compact-basis-key", type=str, default="")
    return p


def main() -> None:
    run_checks(build_parser().parse_args())


if __name__ == "__main__":
    main()
